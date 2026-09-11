"""Pipeline steps for the face_landmarks experiment.

The only experiment in this project that tests the actual hypothesis
rather than bounding a confound: every prior arm fed pixels to a frozen
backbone and asked what it could read. This one feeds no pixels at all --
just a handful of numbers describing where the eyes, nose and mouth sit
relative to one another, after a canonical alignment that removes
position, scale and in-plane rotation. Texture, lighting, sharpness,
compression, retouching, background and framing are absent by
construction.

Two independent landmark sources are run and kept comparable throughout:
MediaPipe Face Mesh (478 points, the primary source -- dense enough for
the full morphology set) and MTCNN's 5 points (eyes, nose, mouth
corners -- a cross-check able to support only a 4-feature subset).
Alignment and feature math live in `src.data.landmarks`; detection and
caching live in `src.models.landmark_detect`. Boxes are never
(re-)detected here: they are the same cached sigma=0 MTCNN boxes every
prior experiment traces back to, read via `background_images.pipeline`'s
own `step_2_boxes` / `step_3_exclude_boxless` so the box-handling logic
is identical by construction, not by copy-paste.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.background_images.pipeline import (
    step_2_boxes,
    step_3_exclude_boxless,
    step_4_box_geometry,
)
from experiments.face_landmarks.config import Config
from src.data.images import build_image_manifest
from src.data.landmarks import (
    MORPHOLOGY_FEATURE_NAMES,
    MTCNN_MORPHOLOGY_FEATURE_NAMES,
    POSE_FEATURE_NAMES,
    align_mediapipe,
    align_mtcnn,
    apply_synthetic_transform,
    morphology_features_mediapipe,
    morphology_features_mtcnn,
    pose_features_mediapipe,
    pose_features_mtcnn,
)
from src.data.transforms import geometry_stage
from src.evaluation.cross_val import repeated_stratified_cv
from src.evaluation.effect_size import benjamini_hochberg, cliffs_delta
from src.models.embeddings import load_embeddings
from src.models.landmark_detect import (
    detect_landmarks_mediapipe,
    detect_landmarks_mtcnn,
    download_face_landmarker_model,
    load_landmarks,
    save_landmarks,
)

logger = logging.getLogger(__name__)

__all__ = [
    "step_1_manifest",
    "step_2_boxes",
    "step_3_exclude_boxless",
    "step_4_check_mtcnn_landmark_cache",
    "step_5_mediapipe_landmarks",
    "step_6_mtcnn_landmarks",
    "step_7_combine_exclusions",
    "step_8_align_and_features",
    "step_9_box_geometry",
    "step_10_invariance_test",
    "step_11_correlation_vs_box_area",
    "step_12_effect_sizes",
    "step_13_probes",
    "step_14_probe_summary",
    "step_15_size_restriction",
    "step_16_noise_perturbation",
    "step_17_detector_agreement",
    "step_18_reference_lines",
]


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the full image manifest -- identical to every prior experiment, all 2,940 rows."""
    return build_image_manifest(cfg.data_root)


def step_4_check_mtcnn_landmark_cache(cfg: Config) -> bool:
    """Check whether `blurred_images`'s cached sigma=0 MTCNN detections already carry landmarks.

    Args:
        cfg: Experiment configuration.

    Returns:
        `True` if a `landmarks` (or similarly-named) column is present in
        `blurred_images`'s cached `detections_mtcnn.csv`; `False`
        otherwise. `False` here means `step_6_mtcnn_landmarks` below must
        re-run MTCNN at sigma=0 with `landmarks=True`.
    """
    cache_path = cfg.blurred_cache_dir / "detections_mtcnn.csv"
    columns = set(pd.read_csv(cache_path, nrows=0).columns)
    has_landmarks = bool(columns & {"landmarks", "landmark", "keypoints", "points"})
    logger.info(
        "step_4_check_mtcnn_landmark_cache: %s columns=%s -> landmarks cached=%s",
        cache_path,
        sorted(columns),
        has_landmarks,
    )
    return has_landmarks


def step_5_mediapipe_landmarks(
    cfg: Config, manifest: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract (or load cached) MediaPipe Face Mesh landmarks for every image in `manifest`.

    Downloads the Face Landmarker model bundle once if not already
    present (see `download_face_landmarker_model`), then runs it on the
    post-geometry 224x224 image for every row -- the same input every
    other experiment's transform pipeline produced.

    Args:
        cfg: Experiment configuration.
        manifest: Row order every returned array matches.

    Returns:
        `(points, success, index)`: `points` is `(len(manifest), 478, 2)`
        float32 pixel coordinates (`NaN`-filled where `success` is
        `False`); `success` is `(len(manifest),)` boolean; `index` is the
        cached `path`/`success` table (kept only so callers do not need
        to reconstruct it).
    """
    name = "landmarks_mediapipe"
    array_path = cfg.output_dir / f"{name}.npy"
    paths = [Path(p) for p in manifest["path"]]

    if array_path.exists():
        points, index, meta = load_landmarks(cfg.output_dir, name)
        if list(index["path"]) == [str(p) for p in paths] and meta.get("n_images") == len(paths):
            logger.info("step_5_mediapipe_landmarks: using cached landmarks, shape=%s", points.shape)
            return points, index["success"].to_numpy(), index
        logger.info("step_5_mediapipe_landmarks: cache present but stale, re-extracting")

    model_path = download_face_landmarker_model(cfg.mediapipe_model_path)
    images = [geometry_stage(Image.open(p), cfg.image_size) for p in manifest["path"]]
    points, success = detect_landmarks_mediapipe(images, model_path)

    meta = {
        "detector": "mediapipe_face_landmarker",
        "model_path": str(model_path),
        "image_size": cfg.image_size,
        "n_images": len(paths),
        "n_success": int(success.sum()),
    }
    save_landmarks(points, success, paths, meta, cfg.output_dir, name)
    logger.info("step_5_mediapipe_landmarks: extracted and cached, shape=%s", points.shape)
    _, index, _ = load_landmarks(cfg.output_dir, name)
    return points, success, index


def step_6_mtcnn_landmarks(cfg: Config, manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract (or load cached) MTCNN 5-point landmarks for every image in `manifest`.

    Always re-runs MTCNN at sigma=0 (never sigma>0) with `landmarks=True`
    -- `step_4_check_mtcnn_landmark_cache` establishes that the existing
    `blurred_images` detection cache has no landmarks to reuse, so this
    is a fresh detection pass, cached independently under this
    experiment's own output directory.

    Args:
        cfg: Experiment configuration.
        manifest: Row order every returned array matches.

    Returns:
        Same shape as `step_5_mediapipe_landmarks`, but `points` is
        `(len(manifest), 5, 2)`.
    """
    name = "landmarks_mtcnn"
    array_path = cfg.output_dir / f"{name}.npy"
    paths = [Path(p) for p in manifest["path"]]

    if array_path.exists():
        points, index, meta = load_landmarks(cfg.output_dir, name)
        if list(index["path"]) == [str(p) for p in paths] and meta.get("n_images") == len(paths):
            logger.info("step_6_mtcnn_landmarks: using cached landmarks, shape=%s", points.shape)
            return points, index["success"].to_numpy(), index
        logger.info("step_6_mtcnn_landmarks: cache present but stale, re-extracting")

    images = [geometry_stage(Image.open(p), cfg.image_size) for p in manifest["path"]]
    points, success = detect_landmarks_mtcnn(images, device=cfg.device)

    meta = {
        "detector": "mtcnn_facenet_pytorch",
        "sigma": 0,
        "image_size": cfg.image_size,
        "n_images": len(paths),
        "n_success": int(success.sum()),
    }
    save_landmarks(points, success, paths, meta, cfg.output_dir, name)
    logger.info("step_6_mtcnn_landmarks: extracted and cached, shape=%s", points.shape)
    _, index, _ = load_landmarks(cfg.output_dir, name)
    return points, success, index


def step_7_combine_exclusions(
    manifest: pd.DataFrame,
    boxes: pd.DataFrame,
    mp_points: np.ndarray,
    mp_success: np.ndarray,
    mt_points: np.ndarray,
    mt_success: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Drop any image where either detector failed, from every array at once.

    An image excluded here is excluded from every probe below -- the
    morphology, pose and box-geometry probes all run on identical rows,
    and the MediaPipe/MTCNN cross-check is only meaningful if both saw
    the same image set to begin with.

    Args:
        manifest: Row-aligned to `boxes`, `mp_points`, `mt_points`.
        boxes: Row-aligned to `manifest`.
        mp_points: `(len(manifest), 478, 2)`, from `step_5_mediapipe_landmarks`.
        mp_success: `(len(manifest),)` boolean, from the same.
        mt_points: `(len(manifest), 5, 2)`, from `step_6_mtcnn_landmarks`.
        mt_success: `(len(manifest),)` boolean, from the same.

    Returns:
        `(manifest, boxes, mp_points, mt_points, excluded, counts_by_class)`:
        the first four are row-aligned to each other with failing images
        dropped; `excluded` has one row per dropped image (`path`,
        `class_label`, `mediapipe_failed`, `mtcnn_failed`);
        `counts_by_class` has one row per class: `n_excluded`.
    """
    both_ok = mp_success & mt_success
    excluded_mask = ~both_ok

    excluded = pd.DataFrame(
        {
            "path": manifest.loc[excluded_mask, "path"].to_numpy(),
            "class_label": manifest.loc[excluded_mask, "class_label"].to_numpy(),
            "mediapipe_failed": ~mp_success[excluded_mask],
            "mtcnn_failed": ~mt_success[excluded_mask],
        }
    )
    counts_by_class = (
        excluded.groupby("class_label").size().rename("n_excluded").reset_index()
        if len(excluded)
        else pd.DataFrame(columns=["class_label", "n_excluded"])
    )

    manifest_final = manifest.loc[both_ok].reset_index(drop=True)
    boxes_final = boxes.loc[both_ok].reset_index(drop=True)
    mp_points_final = mp_points[both_ok]
    mt_points_final = mt_points[both_ok]

    logger.info(
        "step_7_combine_exclusions: mediapipe failed=%d, mtcnn failed=%d, either failed=%d -- %d rows remain",
        int((~mp_success).sum()),
        int((~mt_success).sum()),
        int(excluded_mask.sum()),
        len(manifest_final),
    )
    return manifest_final, boxes_final, mp_points_final, mt_points_final, excluded, counts_by_class


def _features_dataframe(paths, rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "path", paths)
    return df


def step_8_align_and_features(
    cfg: Config, manifest: pd.DataFrame, mp_points: np.ndarray, mt_points: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """Align every face and compute the morphology and pose feature sets, per detector.

    Cached as four CSVs (`features_{mediapipe,mtcnn}_{morphology,pose}.csv`)
    -- cheap pure-geometry computation, but caching keeps repeated
    notebook runs from redoing it.

    Args:
        cfg: Experiment configuration.
        manifest: Row order every returned table matches.
        mp_points: `(len(manifest), 478, 2)`, all rows successful (post
            `step_7_combine_exclusions`).
        mt_points: `(len(manifest), 5, 2)`, all rows successful.

    Returns:
        Dict with four DataFrames, each `path` plus its feature columns:
        `"mediapipe_morphology"` (`MORPHOLOGY_FEATURE_NAMES`),
        `"mediapipe_pose"` (`POSE_FEATURE_NAMES`), `"mtcnn_morphology"`
        (`MTCNN_MORPHOLOGY_FEATURE_NAMES`), `"mtcnn_pose"` (`POSE_FEATURE_NAMES`).
    """
    cache_names = {
        "mediapipe_morphology": "features_mediapipe_morphology",
        "mediapipe_pose": "features_mediapipe_pose",
        "mtcnn_morphology": "features_mtcnn_morphology",
        "mtcnn_pose": "features_mtcnn_pose",
    }
    cache_paths = {k: cfg.output_dir / f"{v}.csv" for k, v in cache_names.items()}
    if all(p.exists() for p in cache_paths.values()):
        cached = {k: pd.read_csv(p) for k, p in cache_paths.items()}
        if all(len(df) == len(manifest) and list(df["path"]) == list(manifest["path"]) for df in cached.values()):
            logger.info("step_8_align_and_features: using cached feature tables")
            return cached
        logger.info("step_8_align_and_features: cache present but stale, recomputing")

    mp_morph_rows, mp_pose_rows, mt_morph_rows, mt_pose_rows = [], [], [], []
    for i in range(len(manifest)):
        mp_align = align_mediapipe(mp_points[i])
        mp_morph_rows.append(morphology_features_mediapipe(mp_align.aligned))
        mp_pose_rows.append(pose_features_mediapipe(mp_points[i], mp_align))

        mt_align = align_mtcnn(mt_points[i])
        mt_morph_rows.append(morphology_features_mtcnn(mt_align.aligned))
        mt_pose_rows.append(pose_features_mtcnn(mt_points[i], mt_align))

    paths = manifest["path"].to_numpy()
    result = {
        "mediapipe_morphology": _features_dataframe(paths, mp_morph_rows),
        "mediapipe_pose": _features_dataframe(paths, mp_pose_rows),
        "mtcnn_morphology": _features_dataframe(paths, mt_morph_rows),
        "mtcnn_pose": _features_dataframe(paths, mt_pose_rows),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for key, df in result.items():
        df.to_csv(cache_paths[key], index=False)
    logger.info("step_8_align_and_features: computed and cached %d rows per table", len(manifest))
    return result


def step_9_box_geometry(cfg: Config, boxes: pd.DataFrame) -> pd.DataFrame:
    """Re-run `background_images`'s box-geometry features on this experiment's own rows.

    Thin wrapper around `background_images.pipeline.step_4_box_geometry`
    -- same computation, same feature names (`GEOMETRY_FEATURE_NAMES`),
    a different (smaller) row set.

    Args:
        cfg: Experiment configuration.
        boxes: A boxes table with `path`, `class_label`, `best_box`.

    Returns:
        One row per image: `path`, `class_label`, plus `GEOMETRY_FEATURE_NAMES`.
    """
    return step_4_box_geometry(cfg, boxes)


def step_10_invariance_test(
    cfg: Config, manifest: pd.DataFrame, mp_points: np.ndarray, mt_points: np.ndarray
) -> pd.DataFrame:
    """Verify morphology features are invariant to a synthetic scale/rotate/translate.

    For `cfg.n_invariance_samples` images (the first N in `manifest`'s
    sorted order, for determinism), applies
    `apply_synthetic_transform(scale=cfg.invariance_scale,
    rotate_deg=cfg.invariance_rotate_deg,
    translate=cfg.invariance_translate_px)` directly to the raw landmark
    coordinates, recomputes every morphology feature, and compares
    against the untransformed value. This tests the alignment math
    itself, not detector robustness to a re-photographed image -- the
    input points are transformed directly, not re-detected on a
    transformed image.

    Args:
        cfg: Experiment configuration.
        manifest: Row order matching `mp_points` / `mt_points`.
        mp_points: `(len(manifest), 478, 2)`, all rows successful.
        mt_points: `(len(manifest), 5, 2)`, all rows successful.

    Returns:
        One row per `(source, path, feature)`: `source`, `path`,
        `feature`, `original`, `transformed`, `abs_diff`.

    Raises:
        RuntimeError: If any `abs_diff` exceeds `cfg.invariance_tolerance`.
    """
    n = min(cfg.n_invariance_samples, len(manifest))
    idx = np.arange(n)

    rows = []
    for i in idx:
        path = manifest["path"].iloc[i]

        mp_orig_feats = morphology_features_mediapipe(align_mediapipe(mp_points[i]).aligned)
        mp_transformed = apply_synthetic_transform(
            mp_points[i], cfg.invariance_scale, cfg.invariance_rotate_deg, cfg.invariance_translate_px
        )
        mp_new_feats = morphology_features_mediapipe(align_mediapipe(mp_transformed).aligned)
        for feature in MORPHOLOGY_FEATURE_NAMES:
            rows.append(
                {
                    "source": "mediapipe",
                    "path": path,
                    "feature": feature,
                    "original": mp_orig_feats[feature],
                    "transformed": mp_new_feats[feature],
                    "abs_diff": abs(mp_orig_feats[feature] - mp_new_feats[feature]),
                }
            )

        mt_orig_feats = morphology_features_mtcnn(align_mtcnn(mt_points[i]).aligned)
        mt_transformed = apply_synthetic_transform(
            mt_points[i], cfg.invariance_scale, cfg.invariance_rotate_deg, cfg.invariance_translate_px
        )
        mt_new_feats = morphology_features_mtcnn(align_mtcnn(mt_transformed).aligned)
        for feature in MTCNN_MORPHOLOGY_FEATURE_NAMES:
            rows.append(
                {
                    "source": "mtcnn",
                    "path": path,
                    "feature": feature,
                    "original": mt_orig_feats[feature],
                    "transformed": mt_new_feats[feature],
                    "abs_diff": abs(mt_orig_feats[feature] - mt_new_feats[feature]),
                }
            )

    result = pd.DataFrame(rows)
    max_diff = result["abs_diff"].max()
    logger.info(
        "step_10_invariance_test: %d images x scale=%s rotate=%s translate=%s -- max abs diff=%.2e (tolerance=%.0e)",
        n,
        cfg.invariance_scale,
        cfg.invariance_rotate_deg,
        cfg.invariance_translate_px,
        max_diff,
        cfg.invariance_tolerance,
    )
    if max_diff > cfg.invariance_tolerance:
        worst = result.loc[result["abs_diff"].idxmax()]
        raise RuntimeError(
            f"step_10_invariance_test: alignment is NOT invariant -- max abs diff {max_diff:.2e} "
            f"exceeds tolerance {cfg.invariance_tolerance:.0e} (worst: source={worst['source']} "
            f"feature={worst['feature']} path={worst['path']})"
        )
    return result


def step_11_correlation_vs_box_area(
    morphology_df: pd.DataFrame, geometry_df: pd.DataFrame, feature_names, threshold: float
) -> pd.DataFrame:
    """Pearson correlation of each morphology feature against `box_area_fraction`.

    A strong correlation here means face scale is leaking through the
    supposedly scale-invariant alignment -- the feature is not measuring
    what it claims to.

    Args:
        morphology_df: A morphology feature table with `path` + feature
            columns (either detector).
        geometry_df: Output of `step_9_box_geometry`, same rows.
        feature_names: Which columns of `morphology_df` to test.
        threshold: `|r|` at or above which a feature is flagged.

    Returns:
        One row per feature, sorted by `|pearson_r|` descending:
        `feature`, `pearson_r`, `p_value`, `flagged`.
    """
    aligned = morphology_df.merge(
        geometry_df[["path", "box_area_fraction"]], on="path", how="inner", validate="one_to_one"
    )
    rows = []
    for feature in feature_names:
        r, p = stats.pearsonr(aligned[feature], aligned["box_area_fraction"])
        rows.append({"feature": feature, "pearson_r": r, "p_value": p, "flagged": abs(r) >= threshold})
    result = pd.DataFrame(rows)
    return result.reindex(result["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def step_12_effect_sizes(
    feature_tables: Dict[str, pd.DataFrame], manifest: pd.DataFrame, bh_alpha: float
) -> pd.DataFrame:
    """Cliff's delta (autistic vs. non_autistic) for every morphology and pose feature, per detector.

    Publishable morphometric content independent of what the probes
    conclude: which measurements do and do not differ between the two
    sets of photographs, ranked by effect size, with a Mann-Whitney U
    p-value and its Benjamini-Hochberg-adjusted q-value. BH correction is
    applied once per detector (`"mediapipe"` / `"mtcnn"`), across that
    detector's full morphology + pose feature family together.

    Args:
        feature_tables: The dict `step_8_align_and_features` returns.
        manifest: Row-aligned to every table in `feature_tables`, for
            `class_label`.
        bh_alpha: Reported alongside `q_value` so the table is
            self-describing; not used to filter rows (every feature is
            reported, significant or not).

    Returns:
        One row per `(source, feature_set, feature)`, sorted by `source`
        then `|cliffs_delta|` descending: `source`, `feature_set`,
        `feature`, `n_autistic`, `n_non_autistic`, `mean_autistic`,
        `mean_non_autistic`, `cliffs_delta`, `p_value`, `q_value`,
        `bh_alpha`, `significant`.
    """
    class_label = manifest["class_label"].to_numpy()
    autistic_mask = class_label == "autistic"
    non_autistic_mask = class_label == "non_autistic"

    specs = (
        ("mediapipe", "morphology", feature_tables["mediapipe_morphology"], MORPHOLOGY_FEATURE_NAMES),
        ("mediapipe", "pose", feature_tables["mediapipe_pose"], POSE_FEATURE_NAMES),
        ("mtcnn", "morphology", feature_tables["mtcnn_morphology"], MTCNN_MORPHOLOGY_FEATURE_NAMES),
        ("mtcnn", "pose", feature_tables["mtcnn_pose"], POSE_FEATURE_NAMES),
    )

    frames = []
    for source in ("mediapipe", "mtcnn"):
        rows = []
        for src_name, feature_set, df, names in specs:
            if src_name != source:
                continue
            for feature in names:
                x = df.loc[autistic_mask, feature].to_numpy()
                y = df.loc[non_autistic_mask, feature].to_numpy()
                _, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
                rows.append(
                    {
                        "source": source,
                        "feature_set": feature_set,
                        "feature": feature,
                        "n_autistic": len(x),
                        "n_non_autistic": len(y),
                        "mean_autistic": float(np.mean(x)),
                        "mean_non_autistic": float(np.mean(y)),
                        "cliffs_delta": cliffs_delta(x, y),
                        "p_value": float(p_value),
                    }
                )
        block = pd.DataFrame(rows)
        block["q_value"] = benjamini_hochberg(block["p_value"].to_numpy())
        block["bh_alpha"] = bh_alpha
        block["significant"] = block["q_value"] < bh_alpha
        frames.append(block)

    result = pd.concat(frames, ignore_index=True)
    result["_abs_delta"] = result["cliffs_delta"].abs()
    result = result.sort_values(["source", "_abs_delta"], ascending=[True, False]).drop(columns="_abs_delta")
    return result.reset_index(drop=True)


def _build_model(cfg: Config) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
        ]
    )


def _aligned_matrix(df: pd.DataFrame, feature_names, manifest: pd.DataFrame) -> np.ndarray:
    aligned = df.set_index("path").loc[manifest["path"]].reset_index()
    return aligned[list(feature_names)].to_numpy()


def _probe_feature_matrix(
    probe: str,
    feature_tables: Dict[str, pd.DataFrame],
    geometry_df: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    if probe == "morphology":
        return _aligned_matrix(feature_tables["mediapipe_morphology"], MORPHOLOGY_FEATURE_NAMES, manifest)
    if probe == "pose":
        return _aligned_matrix(feature_tables["mediapipe_pose"], POSE_FEATURE_NAMES, manifest)
    if probe == "box_geometry":
        return _aligned_matrix(geometry_df, cfg.box_geometry_feature_names, manifest)
    if probe == "morphology_plus_pose":
        morph = _aligned_matrix(feature_tables["mediapipe_morphology"], MORPHOLOGY_FEATURE_NAMES, manifest)
        pose = _aligned_matrix(feature_tables["mediapipe_pose"], POSE_FEATURE_NAMES, manifest)
        return np.concatenate([morph, pose], axis=1)
    raise ValueError(f"Unknown probe: {probe!r}. Known probes: {['morphology', 'pose', 'box_geometry', 'morphology_plus_pose']}")


def step_13_probes(
    cfg: Config, feature_tables: Dict[str, pd.DataFrame], geometry_df: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Cross-validate all four probes (`cfg.probe_names`) on identical rows.

    Same L2-logistic-regression-behind-a-`StandardScaler`, same 5x10
    repeated stratified CV, same seed as every prior experiment -- only
    the feature set differs.

    Args:
        cfg: Experiment configuration.
        feature_tables: Output of `step_8_align_and_features`.
        geometry_df: Output of `step_9_box_geometry`.
        manifest: Row order every probe is evaluated on.

    Returns:
        One row per `(probe, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for probe in cfg.probe_names:
        X = _probe_feature_matrix(probe, feature_tables, geometry_df, manifest, cfg)
        fold_results = repeated_stratified_cv(
            X, y, _build_model(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        fold_results.insert(0, "probe", probe)
        rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_14_probe_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold results with an explicitly paired, uncertainty-carrying gap-to-chance.

    Unlike a bare `mean(roc_auc) - 0.5`, this reports the mean *and sd*
    of the per-fold series `roc_auc - 0.5` itself, plus a paired
    one-sample t-test of that series against 0 -- so "how far above
    chance" always comes with a sense of fold-to-fold spread attached,
    not just a difference of two point estimates.

    Args:
        fold_results: Output of `step_13_probes` (or any table sharing
            its `probe`/`repeat`/`fold`/metric shape).

    Returns:
        One row per probe: mean and sd of each metric
        (`{metric}_mean`, `{metric}_sd`), plus `gap_to_chance_mean`,
        `gap_to_chance_sd`, `gap_to_chance_tstat`, `gap_to_chance_pvalue`
        (paired one-sample t-test of `roc_auc - 0.5` against 0).
    """
    metric_cols = [c for c in fold_results.columns if c not in ("probe", "repeat", "fold")]

    rows = []
    for probe, group in fold_results.groupby("probe"):
        row = {"probe": probe}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std()

        gap = group["roc_auc"].to_numpy() - 0.5
        tstat, pvalue = stats.ttest_1samp(gap, 0.0)
        row["gap_to_chance_mean"] = float(gap.mean())
        row["gap_to_chance_sd"] = float(gap.std())
        row["gap_to_chance_tstat"] = float(tstat)
        row["gap_to_chance_pvalue"] = float(pvalue)
        rows.append(row)

    return pd.DataFrame(rows)


def step_15_size_restriction(
    cfg: Config, feature_tables: Dict[str, pd.DataFrame], boxes: pd.DataFrame, manifest: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-run the `morphology` probe restricted to images with a generously large face box.

    Args:
        cfg: Experiment configuration (`min_face_side_px`).
        feature_tables: Output of `step_8_align_and_features`.
        boxes: Row-aligned to `manifest`, unexpanded `best_box`.
        manifest: Full (post-exclusion) row set.

    Returns:
        `(manifest_restricted, survivor_counts, fold_results)`:
        `manifest_restricted` is the size-filtered row set;
        `survivor_counts` has one row per class (`class_label`,
        `n_total`, `n_survive`, `min_face_side_px`); `fold_results` is
        the `morphology` probe re-run on `manifest_restricted`, same
        shape as one probe's slice of `step_13_probes`'s output.
    """
    min_side = np.array(
        [min(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes["best_box"]]
    )
    keep = min_side >= cfg.min_face_side_px

    manifest_restricted = manifest.loc[keep].reset_index(drop=True)

    survivor_rows = []
    for class_label, total_group in manifest.groupby("class_label"):
        n_survive = int(keep[(manifest["class_label"] == class_label).to_numpy()].sum())
        survivor_rows.append(
            {
                "class_label": class_label,
                "n_total": len(total_group),
                "n_survive": n_survive,
                "min_face_side_px": cfg.min_face_side_px,
            }
        )
    survivor_counts = pd.DataFrame(survivor_rows)

    y = (manifest_restricted["class_label"] == "autistic").astype(int).to_numpy()
    X = _aligned_matrix(feature_tables["mediapipe_morphology"], MORPHOLOGY_FEATURE_NAMES, manifest_restricted)
    fold_results = repeated_stratified_cv(
        X, y, _build_model(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
    )
    fold_results.insert(0, "probe", "morphology_size_restricted")

    logger.info(
        "step_15_size_restriction: threshold=%.0fpx -- %d/%d rows survive",
        cfg.min_face_side_px,
        len(manifest_restricted),
        len(manifest),
    )
    return manifest_restricted, survivor_counts, fold_results


def step_16_noise_perturbation(
    cfg: Config, mp_points: np.ndarray, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Re-run the `morphology` probe after perturbing every MediaPipe landmark with Gaussian noise.

    Noise is added in raw pixel space (224-space), independently per
    coordinate, *before* alignment -- simulating detector imprecision
    rather than a post-hoc feature-level nudge. Each magnitude uses its
    own deterministic seed (`cfg.seed` offset by the magnitude), so
    results are reproducible without every magnitude sharing identical
    noise draws.

    Args:
        cfg: Experiment configuration (`noise_sigmas_px`).
        mp_points: `(len(manifest), 478, 2)`, all rows successful.
        manifest: Row order matching `mp_points`.

    Returns:
        One row per noise magnitude: `noise_sigma_px`, `roc_auc_mean`,
        `roc_auc_sd`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()
    rows = []

    for sigma_px in cfg.noise_sigmas_px:
        rng = np.random.default_rng(cfg.seed + int(round(sigma_px * 1000)))
        noisy_points = mp_points + rng.normal(0.0, sigma_px, size=mp_points.shape)

        feature_rows = [
            morphology_features_mediapipe(align_mediapipe(noisy_points[i]).aligned)
            for i in range(len(manifest))
        ]
        X = pd.DataFrame(feature_rows)[list(MORPHOLOGY_FEATURE_NAMES)].to_numpy()

        fold_results = repeated_stratified_cv(
            X, y, _build_model(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        rows.append(
            {
                "noise_sigma_px": sigma_px,
                "roc_auc_mean": fold_results["roc_auc"].mean(),
                "roc_auc_sd": fold_results["roc_auc"].std(),
            }
        )
        logger.info(
            "step_16_noise_perturbation: sigma=%spx -> ROC-AUC=%.4f +/- %.4f",
            sigma_px,
            rows[-1]["roc_auc_mean"],
            rows[-1]["roc_auc_sd"],
        )

    return pd.DataFrame(rows)


def step_17_detector_agreement(
    cfg: Config, feature_tables: Dict[str, pd.DataFrame], manifest: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare MediaPipe and MTCNN directly, on the 4-feature subset both can compute.

    Args:
        cfg: Experiment configuration.
        feature_tables: Output of `step_8_align_and_features`.
        manifest: Row order to evaluate on.

    Returns:
        `(probe_comparison, value_correlation)`:
        `probe_comparison` has one row per detector (`"mediapipe"`,
        `"mtcnn"`), each the `morphology` probe re-run using only
        `MTCNN_MORPHOLOGY_FEATURE_NAMES` from that detector's own
        features (`roc_auc_mean`, `roc_auc_sd`); `value_correlation` has
        one row per shared feature: `feature`, `pearson_r`, `p_value`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    probe_rows = []
    for source in ("mediapipe", "mtcnn"):
        X = _aligned_matrix(feature_tables[f"{source}_morphology"], MTCNN_MORPHOLOGY_FEATURE_NAMES, manifest)
        fold_results = repeated_stratified_cv(
            X, y, _build_model(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        probe_rows.append(
            {
                "source": source,
                "roc_auc_mean": fold_results["roc_auc"].mean(),
                "roc_auc_sd": fold_results["roc_auc"].std(),
            }
        )
    probe_comparison = pd.DataFrame(probe_rows)

    mp_aligned = feature_tables["mediapipe_morphology"].set_index("path").loc[manifest["path"]]
    mt_aligned = feature_tables["mtcnn_morphology"].set_index("path").loc[manifest["path"]]
    corr_rows = []
    for feature in MTCNN_MORPHOLOGY_FEATURE_NAMES:
        r, p = stats.pearsonr(mp_aligned[feature], mt_aligned[feature])
        corr_rows.append({"feature": feature, "pearson_r": r, "p_value": p})
    value_correlation = pd.DataFrame(corr_rows)

    return probe_comparison, value_correlation


def step_18_reference_lines(cfg: Config) -> pd.DataFrame:
    """Reference AUCs from earlier experiments, for the main figure.

    Each is re-run from that experiment's own cached features/embeddings
    on that experiment's own native row set (not row-matched to this
    experiment) -- clearly informational reference lines, not a
    like-for-like comparison.

    Args:
        cfg: Experiment configuration.

    Returns:
        One row per reference: `label`, `roc_auc_mean`, `roc_auc_sd`,
        `n_rows`, `source`.
    """
    full_manifest = build_image_manifest(cfg.data_root)
    label_by_path = dict(zip(full_manifest["path"], full_manifest["class_label"]))

    def _run(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        fold_results = repeated_stratified_cv(
            X, y, _build_model(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        return float(fold_results["roc_auc"].mean()), float(fold_results["roc_auc"].std())

    from src.data.colour_features import COLOUR_FEATURE_NAMES

    rows = []

    whole_image_colour = pd.read_csv(cfg.blurred_cache_dir / "colour_features_sigma0.csv")
    whole_image_colour["class_label"] = whole_image_colour["path"].map(label_by_path)
    y = (whole_image_colour["class_label"] == "autistic").astype(int).to_numpy()
    X = whole_image_colour[list(COLOUR_FEATURE_NAMES)].to_numpy()
    mean, sd = _run(X, y)
    rows.append(
        {
            "label": "whole_image colour floor (blurred_images, sigma=0)",
            "roc_auc_mean": mean,
            "roc_auc_sd": sd,
            "n_rows": len(whole_image_colour),
            "source": "blurred_images/colour_features_sigma0.csv",
        }
    )

    crop_colour = pd.read_csv(cfg.face_crop_blur_cache_dir / "colour_features_crop_1.0_sigma0.csv")
    crop_colour["class_label"] = crop_colour["path"].map(label_by_path)
    y = (crop_colour["class_label"] == "autistic").astype(int).to_numpy()
    X = crop_colour[list(COLOUR_FEATURE_NAMES)].to_numpy()
    mean, sd = _run(X, y)
    rows.append(
        {
            "label": "crop_1.0 colour floor (face_crop_blur, sigma=0)",
            "roc_auc_mean": mean,
            "roc_auc_sd": sd,
            "n_rows": len(crop_colour),
            "source": "face_crop_blur/colour_features_crop_1.0_sigma0.csv",
        }
    )

    embeddings, index, _ = load_embeddings(cfg.face_crop_blur_cache_dir, name="embeddings_crop_1.0_sigma32")
    class_labels = index["path"].map(label_by_path)
    y = (class_labels == "autistic").astype(int).to_numpy()
    mean, sd = _run(embeddings, y)
    rows.append(
        {
            "label": "crop_1.0 blur-surviving residual (face_crop_blur, sigma=32 embedding)",
            "roc_auc_mean": mean,
            "roc_auc_sd": sd,
            "n_rows": len(index),
            "source": "face_crop_blur/embeddings_crop_1.0_sigma32.npy",
        }
    )

    return pd.DataFrame(rows)
