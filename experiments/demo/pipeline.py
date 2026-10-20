"""Pipeline logic for the single-image demo notebook.

Every prior experiment argues in aggregate: a mean ROC-AUC over 50 folds, a
consolidated table of arms. This demo instead follows a handful of actual
photographs through the whole image stream, one condition at a time, and
shows that the model's predicted probability barely moves even as the pixels
it is fed degrade to nothing recognisable. That argument is only honest if
every probability shown came from a model that never saw that image during
training -- so the one thing this module adds beyond what every other
experiment already cached is out-of-fold *predictions* (not just fold
*metrics*) for a handful of chosen images, plus an explicit, re-derived
assertion that each one really was held out.

Nothing here re-trains a backbone, re-extracts a feature, or re-detects a
face or landmark: every array below is either read directly from another
experiment's cache, or is a small logistic-regression probe refit on that
cached array (seconds, not minutes) so a per-row prediction -- not just a
fold-level AUC -- is available for the images this notebook chooses to show.
The one genuinely new computation is the photographic-counterfactual arm
(Part 3): a handful of images pushed through a frozen backbone a second
time, after an in-memory transform that changes no content.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.background_images.pipeline import step_2_boxes, step_3_exclude_boxless
from experiments.demo.config import Config
from experiments.why_not_faces.cnn import _load_run_result, load_corrected_protocol_manifest
from experiments.why_not_faces.config import Config as WhyNotFacesConfig
from experiments.why_not_faces.config import FEATURE_GROUPS
from src.data.images import build_image_manifest
from src.data.landmarks import align_mediapipe
from src.data.transforms import (
    Box,
    blur_stage,
    crop_stage,
    geometry_stage,
    masked_stage,
    resample_stage,
    to_tensor_stage,
)
from src.models.embeddings import load_embeddings
from src.models.landmark_detect import load_landmarks

CONDITION_ORDER: Tuple[str, ...] = (
    "Intact photograph",
    "Blurred sigma=32",
    "Blurred sigma=64",
    "Face masked out",
    "Face crop only",
    "Named features only",
    "Facial shape only",
    "Fine-tuned CNN",
)

ARM_TO_CONDITION: Dict[str, str] = {
    "intact": "Intact photograph",
    "blur_sigma32": "Blurred sigma=32",
    "blur_sigma64": "Blurred sigma=64",
    "face_masked": "Face masked out",
    "face_crop": "Face crop only",
    "named_features": "Named features only",
    "facial_shape": "Facial shape only",
}
_ARM_TO_CONDITION = ARM_TO_CONDITION  # internal alias used elsewhere in this module


# ---------------------------------------------------------------------------
# Part 1: manifest, arm coverage, image selection.
# ---------------------------------------------------------------------------


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the full canonical image manifest -- identical to every other experiment."""
    return build_image_manifest(cfg.data_root)


def step_2_eligible_images(
    cfg: Config, manifest_full: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rows present in every arm this demo shows, plus each arm's own row count.

    The frozen-probe arms cover between 2,938 and 2,940 of the 2,940 images
    (see each arm's own experiment for why); the corrected-protocol CNN
    covers only 2,500 (holdout-excluded, hash-deduplicated). The intersection
    of all of them -- the only images every condition in the demo sequence
    can honestly show -- is smaller than any single arm's own row count.

    Args:
        cfg: Demo configuration.
        manifest_full: Output of `step_1_manifest`.

    Returns:
        `(eligible, features, boxes, corrected_manifest_r0, coverage)`:
        `eligible` is the manifest restricted to the intersection;
        `features` is `why_not_faces`'s cached named-feature table (also
        the source of the `named_features` and `facial_shape` demo
        conditions); `boxes` is the cached sigma=0 MTCNN box for every
        image in `features` (for rendering the masked/cropped conditions);
        `corrected_manifest_r0` is the corrected-protocol CV manifest
        restricted to `cfg.repeat_index`; `coverage` reports each arm's
        own row count before intersection, for the notebook to print.
    """
    _, intact_index, _ = load_embeddings(cfg.intact_cache_dir, name="embeddings")
    _, sigma32_index, _ = load_embeddings(cfg.blurred_cache_dir, name="embeddings_sigma32")
    _, sigma64_index, _ = load_embeddings(cfg.blurred_cache_dir, name="embeddings_sigma64")

    boxes_full = step_2_boxes(manifest_full)
    _, boxes, _ = step_3_exclude_boxless(manifest_full, boxes_full)

    _, masked_index, _ = load_embeddings(cfg.background_cache_dir, name="embeddings_face_masked_1.5")
    _, crop_index, _ = load_embeddings(cfg.face_crop_blur_cache_dir, name="embeddings_crop_1.0_sigma0")

    features = pd.read_csv(cfg.why_not_faces_cache_dir / "features.csv")

    wnf_cfg = WhyNotFacesConfig()
    corrected_manifest = load_corrected_protocol_manifest(wnf_cfg, manifest_full)
    corrected_manifest_r0 = corrected_manifest.loc[corrected_manifest["repeat"] == cfg.repeat_index].reset_index(
        drop=True
    )

    path_sets: Dict[str, set] = {
        "intact": set(intact_index["path"]),
        "blur_sigma32": set(sigma32_index["path"]),
        "blur_sigma64": set(sigma64_index["path"]),
        "face_masked": set(masked_index["path"]),
        "face_crop": set(crop_index["path"]),
        "named_features": set(features["path"]),
        "facial_shape": set(features["path"]),
        "cnn_corrected (repeat 0)": set(corrected_manifest_r0["path"]),
    }
    eligible_paths = set.intersection(*path_sets.values())
    eligible = manifest_full.loc[manifest_full["path"].isin(eligible_paths)].reset_index(drop=True)

    coverage = pd.DataFrame(
        {
            "arm": list(path_sets),
            "n_rows": [len(v) for v in path_sets.values()],
        }
    )
    coverage.loc[len(coverage)] = {"arm": "intersection (eligible)", "n_rows": len(eligible)}
    return eligible, features, boxes, corrected_manifest_r0, coverage


# ---------------------------------------------------------------------------
# Out-of-fold predictions: one fold split per frozen-probe arm, re-derived
# with the exact seed/repeat every other experiment's own validation uses.
# ---------------------------------------------------------------------------


@dataclass
class ArmOOFResult:
    """Out-of-fold predictions for one frozen-probe arm, one fold split.

    Attributes:
        path_order: The arm's own native row order (`(n,)` array of paths).
        y: Binary labels, aligned to `path_order`.
        proba: Out-of-fold predicted P(autistic), aligned to `path_order`.
        fold_id: Which fold held each row out, aligned to `path_order`.
        repeat_seed: The `StratifiedKFold` `random_state` used -- the same
            value `repeated_stratified_cv` would derive for
            `cfg.repeat_index` of `cfg.n_repeats`, from `cfg.seed`.
        fold_models: `{fold: fitted sklearn Pipeline}` -- kept so Part 3's
            photographic counterfactual can score a new (degraded) image
            with the exact model that never saw the original.
    """

    path_order: np.ndarray
    y: np.ndarray
    proba: np.ndarray
    fold_id: np.ndarray
    repeat_seed: int
    fold_models: Dict[int, Pipeline]


def _repeat_seed(cfg: Config) -> int:
    """Reproduce `repeated_stratified_cv`'s own per-repeat seed derivation, for `cfg.repeat_index`.

    Identical to `experiments.why_not_faces.pipeline._repeat_seeds` (and
    every arm's own validation) up to indexing: same base `cfg.seed`, same
    `np.random.default_rng` call, same number of repeats drawn -- so the
    fold split this produces for `repeat_index=0` is not merely *a*
    plausible split, it is the exact split every arm's own repeat-0 fold
    already used, e.g. `why_not_faces.pipeline._repeat_seeds(cfg)[0]`.
    """
    rng = np.random.default_rng(cfg.seed)
    repeat_seeds = rng.integers(0, 2**31 - 1, size=cfg.n_repeats)
    return int(repeat_seeds[cfg.repeat_index])


def _build_probe(cfg: Config) -> Pipeline:
    """The one probe shape every arm in this project uses: L2 logistic regression behind a StandardScaler."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
        ]
    )


def out_of_fold_predict(cfg: Config, X: np.ndarray, y: np.ndarray, path_order: Sequence[str]) -> ArmOOFResult:
    """Out-of-fold predicted P(autistic) for every row of `X`, one fold split.

    Fits a fresh probe per fold on that fold's training rows only and reads
    `predict_proba` on that fold's own held-out rows -- so the value stored
    for row `i` always came from a model that never saw row `i` during
    training. See `assert_frozen_arm_out_of_fold` for the independent check.

    Args:
        cfg: Demo configuration.
        X: Feature matrix (an arm's cached embeddings or named/facial-shape
            feature columns).
        y: Binary labels, aligned to `X`.
        path_order: Image paths, aligned to `X` -- this arm's own native row
            order, not necessarily any other arm's.

    Returns:
        See `ArmOOFResult`.
    """
    repeat_seed = _repeat_seed(cfg)
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=repeat_seed)

    proba = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_models: Dict[int, Pipeline] = {}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        assert not (set(train_idx) & set(test_idx)), "train/test indices overlap -- StratifiedKFold is broken"
        model = _build_probe(cfg)
        model.fit(X[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        fold_id[test_idx] = fold
        fold_models[fold] = model

    assert not np.isnan(proba).any(), "every row must be a test row exactly once"
    assert (fold_id >= 0).all()
    return ArmOOFResult(
        path_order=np.asarray(path_order),
        y=np.asarray(y),
        proba=proba,
        fold_id=fold_id,
        repeat_seed=repeat_seed,
        fold_models=fold_models,
    )


def _label_lookup(manifest_full: pd.DataFrame) -> Dict[str, int]:
    return dict(zip(manifest_full["path"], (manifest_full["class_label"] == "autistic").astype(int)))


def oof_intact(cfg: Config, manifest_full: pd.DataFrame) -> ArmOOFResult:
    """Out-of-fold predictions for the intact-image arm (`intact_images`'s cached embeddings)."""
    array, index, _ = load_embeddings(cfg.intact_cache_dir, name="embeddings")
    y = index["path"].map(_label_lookup(manifest_full)).to_numpy()
    return out_of_fold_predict(cfg, array, y, index["path"].to_numpy())


def oof_blur(cfg: Config, manifest_full: pd.DataFrame, sigma: int) -> ArmOOFResult:
    """Out-of-fold predictions for one blur strength (`blurred_images`'s cached embeddings)."""
    array, index, _ = load_embeddings(cfg.blurred_cache_dir, name=f"embeddings_sigma{sigma}")
    y = index["path"].map(_label_lookup(manifest_full)).to_numpy()
    return out_of_fold_predict(cfg, array, y, index["path"].to_numpy())


def oof_face_masked(cfg: Config, manifest_full: pd.DataFrame) -> ArmOOFResult:
    """Out-of-fold predictions for the face-masked-out arm (`background_images`, wide box)."""
    array, index, _ = load_embeddings(cfg.background_cache_dir, name="embeddings_face_masked_1.5")
    y = index["path"].map(_label_lookup(manifest_full)).to_numpy()
    return out_of_fold_predict(cfg, array, y, index["path"].to_numpy())


def oof_face_crop(cfg: Config, manifest_full: pd.DataFrame) -> ArmOOFResult:
    """Out-of-fold predictions for the face-crop-only arm (`face_crop_blur`'s `crop_1.0`, sigma=0)."""
    array, index, _ = load_embeddings(cfg.face_crop_blur_cache_dir, name="embeddings_crop_1.0_sigma0")
    y = index["path"].map(_label_lookup(manifest_full)).to_numpy()
    return out_of_fold_predict(cfg, array, y, index["path"].to_numpy())


def _feature_columns(groups: Iterable[str]) -> List[str]:
    return [name for group in groups for name in FEATURE_GROUPS[group]]


def oof_named_features(cfg: Config, features: pd.DataFrame) -> ArmOOFResult:
    """Out-of-fold predictions for the named-features-only arm: acquisition + colour + framing, no face geometry."""
    columns = _feature_columns(cfg.named_feature_groups)
    X = features[columns].to_numpy()
    y = (features["class_label"] == "autistic").astype(int).to_numpy()
    return out_of_fold_predict(cfg, X, y, features["path"].to_numpy())


def oof_facial_shape(cfg: Config, features: pd.DataFrame) -> ArmOOFResult:
    """Out-of-fold predictions for the facial-shape-only arm: the 11 morphology measurements, no pixels."""
    columns = _feature_columns([cfg.facial_shape_group])
    X = features[columns].to_numpy()
    y = (features["class_label"] == "autistic").astype(int).to_numpy()
    return out_of_fold_predict(cfg, X, y, features["path"].to_numpy())


def compute_all_frozen_arms(
    cfg: Config, manifest_full: pd.DataFrame, features: pd.DataFrame
) -> Dict[str, ArmOOFResult]:
    """Out-of-fold predictions for all seven frozen-probe conditions the demo sequence shows."""
    return {
        "intact": oof_intact(cfg, manifest_full),
        "blur_sigma32": oof_blur(cfg, manifest_full, 32),
        "blur_sigma64": oof_blur(cfg, manifest_full, 64),
        "face_masked": oof_face_masked(cfg, manifest_full),
        "face_crop": oof_face_crop(cfg, manifest_full),
        "named_features": oof_named_features(cfg, features),
        "facial_shape": oof_facial_shape(cfg, features),
    }


def cnn_oof(cfg: Config, manifest_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold CNN predictions, reassembled from the five cached corrected-protocol fold results.

    No inference is run here: `cnn_corrected_fold{k}_predictions.npz` is
    already the score of fold `k`'s fine-tuned model on fold `k`'s own held-
    out test images (`experiments.why_not_faces.cnn.run_corrected_protocol`
    never evaluates a fold's model on that fold's own training rows). This
    only concatenates the five folds' cached `(test_paths, probs, true)`
    into one path-indexed table.

    Args:
        cfg: Demo configuration.
        manifest_full: Output of `step_1_manifest`.

    Returns:
        `(cnn_table, corrected_manifest_r0)`: `cnn_table` has one row per
        image: `path`, `fold`, `proba`, `y`; `corrected_manifest_r0` is
        `cv_folds.csv`'s own repeat-0 manifest, for the independent
        assertion in `assert_cnn_out_of_fold`.

    Raises:
        RuntimeError: If any corrected-protocol fold has no cached result
            (run `why_not_faces.ipynb` first -- this demo never trains).
    """
    wnf_cfg = WhyNotFacesConfig()
    corrected_manifest = load_corrected_protocol_manifest(wnf_cfg, manifest_full)
    corrected_manifest_r0 = corrected_manifest.loc[corrected_manifest["repeat"] == cfg.repeat_index].reset_index(
        drop=True
    )

    rows = []
    for fold in range(5):
        name = f"cnn_corrected_fold{fold}"
        cached = _load_run_result(wnf_cfg, name)
        if cached is None:
            raise RuntimeError(f"{name} has no cached result -- run notebooks/why_not_faces.ipynb first.")
        test_paths, probs, true = cached["test_paths"], cached["probs"], cached["true"]
        assert len(test_paths) == len(probs) == len(true)
        for path, prob, label in zip(test_paths, probs, true):
            rows.append({"path": path, "fold": fold, "proba": float(prob), "y": int(label)})

    cnn_table = pd.DataFrame(rows)
    assert cnn_table["path"].is_unique, "a path appeared in more than one corrected-protocol fold's test set"
    assert set(cnn_table["path"]) == set(corrected_manifest_r0["path"]), (
        "cached CNN predictions do not cover exactly repeat 0's non-holdout images -- "
        "cv_folds.csv or the cached predictions are stale relative to each other"
    )
    return cnn_table, corrected_manifest_r0


# ---------------------------------------------------------------------------
# Part 1: demo-image selection.
# ---------------------------------------------------------------------------


def select_demo_images(
    cfg: Config, eligible: pd.DataFrame, intact_result: ArmOOFResult
) -> Tuple[pd.DataFrame, int]:
    """Rejection-sample 2 autistic + 2 non_autistic demo images from `eligible`, fixed seed.

    A plain, unconstrained draw from `cfg.selection_seed` is accepted as-is
    unless every one of the four images it drew is correctly classified by
    the intact-arm out-of-fold probe at `cfg.decision_threshold` -- in which
    case that draw is discarded and the *next* draw from the same seeded
    stream is tried. This is the only departure from an unconstrained random
    draw, and it exists for a stated reason: a demo where every prediction
    is confident and correct looks staged. Every accepted draw is otherwise
    an ordinary random sample of the eligible pool (see Part 1's typicality
    plots for the honesty check on that claim).

    Args:
        cfg: Demo configuration.
        eligible: Output of `step_2_eligible_images`'s first element.
        intact_result: Output of `oof_intact` -- supplies both the
            ground-truth labels and the predicted probabilities the
            rejection criterion checks.

    Returns:
        `(selected, n_attempts)`: `selected` is 4 rows of `eligible`, in
        draw order (2 autistic, then 2 non_autistic); `n_attempts` is how
        many draws the rejection sampling needed before this one was
        accepted.

    Raises:
        RuntimeError: If no qualifying draw is found within
            `cfg.max_selection_attempts`.
    """
    proba_by_path = dict(zip(intact_result.path_order, intact_result.proba))
    y_by_path = dict(zip(intact_result.path_order, intact_result.y))

    rng = np.random.default_rng(cfg.selection_seed)
    autistic_pool = eligible.loc[eligible["class_label"] == "autistic", "path"].to_numpy()
    non_autistic_pool = eligible.loc[eligible["class_label"] == "non_autistic", "path"].to_numpy()

    for attempt in range(1, cfg.max_selection_attempts + 1):
        chosen_autistic = rng.choice(autistic_pool, size=cfg.n_demo_per_class, replace=False)
        chosen_non_autistic = rng.choice(non_autistic_pool, size=cfg.n_demo_per_class, replace=False)
        chosen = [*chosen_autistic, *chosen_non_autistic]

        predicted = np.array([proba_by_path[p] >= cfg.decision_threshold for p in chosen])
        truth = np.array([y_by_path[p] for p in chosen])
        if (predicted != truth).any():
            order = {path: i for i, path in enumerate(chosen)}
            selected = eligible.loc[eligible["path"].isin(chosen)].copy()
            selected["_order"] = selected["path"].map(order)
            selected = selected.sort_values("_order").drop(columns="_order").reset_index(drop=True)
            return selected, attempt

    raise RuntimeError(
        f"select_demo_images: no draw with at least one misclassified image found in "
        f"{cfg.max_selection_attempts} attempts."
    )


# ---------------------------------------------------------------------------
# Part 1 / 3: independent out-of-fold assertions.
# ---------------------------------------------------------------------------


def assert_frozen_arm_out_of_fold(cfg: Config, arm_name: str, result: ArmOOFResult, path: str) -> int:
    """Independently re-derive `result`'s fold split and assert `path` was a held-out test row.

    Does not trust the bookkeeping `out_of_fold_predict` produced: re-runs
    `StratifiedKFold` with the recorded `repeat_seed` from scratch and
    checks the row really does fall in the test side of the fold it claims,
    and never in that fold's training side.

    Args:
        cfg: Demo configuration.
        arm_name: Label for the assertion message only.
        result: Output of `out_of_fold_predict` for this arm.
        path: The image path to check.

    Returns:
        The fold index, for the notebook to print.

    Raises:
        AssertionError: If `path` is not found exactly once, or the
            recorded fold does not match the re-derived one.
    """
    matches = np.where(result.path_order == path)[0]
    assert len(matches) == 1, f"{arm_name}: {path!r} not found exactly once in this arm's row set"
    row = int(matches[0])
    recorded_fold = int(result.fold_id[row])

    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=result.repeat_seed)
    dummy_X = np.zeros((len(result.y), 1))
    for fold, (train_idx, test_idx) in enumerate(cv.split(dummy_X, result.y)):
        if row in test_idx:
            assert fold == recorded_fold, f"{arm_name}: recorded fold {recorded_fold} != re-derived fold {fold}"
            assert row not in train_idx, f"{arm_name}: row also appears in its own fold's training indices"
            return fold
    raise AssertionError(f"{arm_name}: row {row} ({path!r}) was not a test row in any fold")


def assert_cnn_out_of_fold(cnn_table: pd.DataFrame, corrected_manifest_r0: pd.DataFrame, path: str) -> int:
    """Assert `path`'s cached CNN prediction's fold matches `cv_folds.csv`'s own repeat-0 fold assignment.

    Args:
        cnn_table: Output of `cnn_oof`'s first element.
        corrected_manifest_r0: Output of `cnn_oof`'s second element (or
            `step_2_eligible_images`'s fourth element -- identical).
        path: The image path to check.

    Returns:
        The fold index, for the notebook to print.

    Raises:
        AssertionError: If `path` is missing/duplicated in either table, or
            the two tables disagree on which fold held it out.
    """
    expected = corrected_manifest_r0.loc[corrected_manifest_r0["path"] == path, "fold"]
    assert len(expected) == 1, f"{path!r} missing from (or duplicated in) the repeat-0 CV manifest"
    expected_fold = int(expected.iloc[0])

    cached = cnn_table.loc[cnn_table["path"] == path, "fold"]
    assert len(cached) == 1, f"{path!r} missing from (or duplicated in) the cached CNN out-of-fold predictions"
    cached_fold = int(cached.iloc[0])

    assert cached_fold == expected_fold, (
        f"CNN: cached prediction's fold ({cached_fold}) != cv_folds.csv's own repeat-0 fold ({expected_fold})"
    )
    return cached_fold


# ---------------------------------------------------------------------------
# Part 2: rendering each condition's image.
# ---------------------------------------------------------------------------


def geometry_image(path: str, cfg: Config) -> Image.Image:
    """The post-geometry image every condition starts from (EXIF, RGB, letterbox, resize)."""
    return geometry_stage(Image.open(path), cfg.image_size)


def blurred_image(path: str, sigma: float, cfg: Config) -> Image.Image:
    """This image, Gaussian-blurred at `sigma` -- what the blur arm's embedding actually saw."""
    return blur_stage(geometry_image(path, cfg), sigma)


def masked_image(path: str, box: Box, cfg: Config) -> Image.Image:
    """This image with the (expanded) face box filled in -- background only."""
    return masked_stage(geometry_image(path, cfg), box, cfg.mask_expand, keep="outside")


def cropped_image(path: str, box: Box, cfg: Config) -> Image.Image:
    """This image cropped to the (expanded) face box, resized back up -- face only, position/scale removed."""
    return crop_stage(geometry_image(path, cfg), box, cfg.crop_expand, cfg.image_size)


def named_feature_swatch(features_row: pd.Series) -> np.ndarray:
    """A `(1, 1, 3)` solid-colour array at this image's own mean RGB -- "no pixels, just numbers," rendered literally."""
    rgb = np.array([features_row["mean_r"], features_row["mean_g"], features_row["mean_b"]]) / 255.0
    return np.clip(rgb, 0.0, 1.0).reshape(1, 1, 3)


def facial_shape_points(path: str, cfg: Config) -> np.ndarray:
    """This image's aligned MediaPipe mesh coordinates -- the input the facial-shape probe actually sees.

    Returns:
        `(478, 2)` float64 array in the canonical aligned frame (eye
        midpoint at the origin, inter-ocular axis horizontal, inter-ocular
        distance 1) -- no pixels, only geometry.
    """
    points, index, _ = load_landmarks(cfg.face_landmarks_cache_dir, "landmarks_mediapipe")
    matches = np.where(index["path"].to_numpy() == path)[0]
    assert len(matches) == 1, f"{path!r} not found exactly once in the cached MediaPipe landmarks"
    return align_mediapipe(points[matches[0]]).aligned


def box_lookup(boxes: pd.DataFrame) -> Dict[str, Box]:
    """`path -> best_box` for `masked_image` / `cropped_image` callers."""
    return dict(zip(boxes["path"], boxes["best_box"]))


def assemble_demo_table(
    selected: pd.DataFrame, frozen_arms: Dict[str, ArmOOFResult], cnn_table: pd.DataFrame
) -> pd.DataFrame:
    """One row per (demo image, condition): the long-format table the sequence grid and slope chart both read.

    Args:
        selected: Output of `select_demo_images`'s first element.
        frozen_arms: Output of `compute_all_frozen_arms`.
        cnn_table: Output of `cnn_oof`'s first element.

    Returns:
        One row per `(path, condition)`: `path`, `class_label`,
        `condition` (categorical, in `CONDITION_ORDER`), `probability`,
        `fold`, `arm`.
    """
    cnn_proba = dict(zip(cnn_table["path"], cnn_table["proba"]))
    cnn_fold = dict(zip(cnn_table["path"], cnn_table["fold"]))

    rows = []
    for _, demo_row in selected.iterrows():
        path = demo_row["path"]
        for arm_name, condition in _ARM_TO_CONDITION.items():
            result = frozen_arms[arm_name]
            idx = int(np.where(result.path_order == path)[0][0])
            rows.append(
                {
                    "path": path,
                    "class_label": demo_row["class_label"],
                    "condition": condition,
                    "probability": float(result.proba[idx]),
                    "fold": int(result.fold_id[idx]),
                    "arm": arm_name,
                }
            )
        rows.append(
            {
                "path": path,
                "class_label": demo_row["class_label"],
                "condition": "Fine-tuned CNN",
                "probability": float(cnn_proba[path]),
                "fold": int(cnn_fold[path]),
                "arm": "cnn_corrected",
            }
        )

    table = pd.DataFrame(rows)
    table["condition"] = pd.Categorical(table["condition"], categories=list(CONDITION_ORDER), ordered=True)
    return table.sort_values(["path", "condition"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 3: the photographic counterfactual.
# ---------------------------------------------------------------------------


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip `img` through an in-memory JPEG encode/decode at `quality` -- a technical property only."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade_photo(img: Image.Image, cfg: Config) -> Image.Image:
    """Downscale-then-upscale, re-encode at lower JPEG quality, soften slightly -- content unchanged throughout.

    Args:
        img: Post-geometry RGB image (`cfg.image_size` square) -- see
            `geometry_image`.
        cfg: Demo configuration (`counterfactual_downsample_size`,
            `counterfactual_jpeg_quality`, `counterfactual_sharpness_factor`).

    Returns:
        A new RGB image, same size, same content, different photographic
        character.
    """
    degraded = resample_stage(img, cfg.counterfactual_downsample_size, cfg.image_size)
    degraded = jpeg_recompress(degraded, cfg.counterfactual_jpeg_quality)
    degraded = ImageEnhance.Sharpness(degraded).enhance(cfg.counterfactual_sharpness_factor)
    return degraded


def embed_images(images: Sequence[Image.Image], cfg: Config) -> np.ndarray:
    """Frozen-backbone embeddings for images that exist only in memory (never written under `data/raw/images/`).

    Mirrors `src.models.embeddings.extract_embeddings`, but takes already-
    opened PIL images directly instead of a path list, since the
    photographically degraded images this demo scores have no path to open.

    Args:
        images: Post-geometry RGB images, `cfg.image_size` square.
        cfg: Demo configuration (`backbone`, `device`).

    Returns:
        `(len(images), n_features)` float32 array.
    """
    from src.models.embeddings import _build_frozen_backbone

    torch_device = torch.device(cfg.device)
    model = _build_frozen_backbone(cfg.backbone, torch_device)
    tensors = torch.stack([to_tensor_stage(image) for image in images]).to(torch_device)
    with torch.no_grad():
        out = model(tensors)
    return out.cpu().numpy().astype(np.float32)


def select_studio_portrait(features: pd.DataFrame) -> str:
    """The non_autistic image with the highest cached `sharpness` -- an objective, reproducible "studio-style" pick.

    Not a manual/visual curation: `sharpness` (from the Phase-0 corpus
    audit, part of the `acquisition` named-feature group) is the cached
    quantity closest to what "studio-style portrait" means photographically
    -- in sharp focus -- so the single displayed counterfactual image is
    chosen by that one deterministic rule, not by eyeballing the corpus.
    """
    non_autistic = features.loc[features["class_label"] == "non_autistic"]
    return str(non_autistic.sort_values("sharpness", ascending=False).iloc[0]["path"])


def select_counterfactual_batch(cfg: Config, features: pd.DataFrame, must_include: str) -> List[str]:
    """A random sample of non_autistic images for the flip-rate test, always including `must_include`.

    Args:
        cfg: Demo configuration (`selection_seed`, `counterfactual_n_images`).
        features: `why_not_faces`'s cached feature table (any row set with
            `class_label` is enough -- only `path` is used here).
        must_include: The displayed counterfactual image (see
            `select_studio_portrait`) -- included so the flip-rate statistic
            and the one image shown are drawn from the same batch.

    Returns:
        Sorted list of image paths (sorted only for a deterministic
        iteration order downstream, not a further sampling step).
    """
    pool = features.loc[features["class_label"] == "non_autistic", "path"].to_numpy()
    rng = np.random.default_rng(cfg.selection_seed)
    n = min(cfg.counterfactual_n_images, len(pool))
    sampled = set(rng.choice(pool, size=n, replace=False))
    sampled.add(must_include)
    return sorted(sampled)


def counterfactual_flip_test(
    cfg: Config, paths: Sequence[str], intact_result: ArmOOFResult
) -> pd.DataFrame:
    """Before/after out-of-fold probability for each of `paths`, scored by that image's own intact-arm fold model.

    "Before" reuses the already-computed intact-arm out-of-fold prediction
    (the identical computation -- no reason to redo it); "after" re-embeds
    the photographically degraded image with the same frozen backbone and
    scores it with the *same* fold's already-fitted probe. That probe never
    saw this image, original or degraded, during training -- degrading an
    image after the fact cannot leak it into a fold it was never a training
    member of.

    Args:
        cfg: Demo configuration.
        paths: Output of `select_counterfactual_batch`.
        intact_result: Output of `oof_intact`.

    Returns:
        One row per path: `path`, `fold`, `p_before`, `p_after`,
        `crossed_threshold` (whether the prediction flipped sides of
        `cfg.decision_threshold`).
    """
    proba_by_path = dict(zip(intact_result.path_order, intact_result.proba))
    fold_by_path = dict(zip(intact_result.path_order, intact_result.fold_id))

    degraded_images = [degrade_photo(geometry_image(path, cfg), cfg) for path in paths]
    after_embeddings = embed_images(degraded_images, cfg)

    rows = []
    for path, after_embedding in zip(paths, after_embeddings):
        fold = int(fold_by_path[path])
        model = intact_result.fold_models[fold]
        p_before = float(proba_by_path[path])
        p_after = float(model.predict_proba(after_embedding.reshape(1, -1))[0, 1])
        rows.append(
            {
                "path": path,
                "fold": fold,
                "p_before": p_before,
                "p_after": p_after,
                "crossed_threshold": (p_before < cfg.decision_threshold) != (p_after < cfg.decision_threshold),
            }
        )
    return pd.DataFrame(rows)
