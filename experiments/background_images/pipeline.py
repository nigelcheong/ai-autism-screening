"""Pipeline steps for the background_images experiment.

Splits the post-geometry image into face and not-face and asks which half
carries the signal -- the test `blurred_images` couldn't run, because blur
destroys facial detail and framing at the same time. Same manifest, same
model, same 5x10 repeated stratified CV, same seed as `intact_images` and
`blurred_images`; only the transform (and, here, which rows are eligible at
all) changes.

Face boxes are never (re-)detected here: they are read from
`blurred_images`'s cached sigma=0 MTCNN detections. Exactly one of the
2,940 images has no box at sigma=0 (see `step_3_exclude_boxless`); it is
dropped from every arm, including the `"intact"` reference, so no number in
this experiment is compared across a 2,940-row and a 2,939-row result.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.background_images.config import Config
from experiments.blurred_images.config import Config as BlurredConfig
from experiments.blurred_images.pipeline import raw_mtcnn_detections
from src.data.colour_features import COLOUR_FEATURE_NAMES, colour_features
from src.data.images import build_image_manifest
from src.data.transforms import (
    Box,
    crop_stage,
    expand_box,
    geometry_stage,
    intact_transform,
    masked_stage,
    to_tensor_stage,
)
from src.evaluation.cross_val import repeated_stratified_cv
from src.evaluation.effect_size import cliffs_delta
from src.models.embeddings import (
    extract_embeddings,
    get_backbone_weights_name,
    load_embeddings,
    save_embeddings,
)

logger = logging.getLogger(__name__)

GEOMETRY_FEATURE_NAMES = ("box_area_fraction", "box_centre_x", "box_centre_y", "box_aspect")

# Maps each non-intact arm name to (mask kind, which expand factor to use).
# The expand factor itself is read off `Config` at call time (`expand_tight`
# / `expand_wide`) rather than hardcoded, so the arm names stay descriptive
# labels, not the source of truth for the numbers they contain.
_ARM_KIND: Dict[str, Tuple[str, str]] = {
    "face_masked_1.0": ("face_masked", "tight"),
    "face_masked_1.5": ("face_masked", "wide"),
    "background_masked_1.5": ("background_masked", "wide"),
    "face_crop_1.5": ("face_crop", "wide"),
}


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the full image manifest -- identical to `intact_images`, all 2,940 rows.

    Args:
        cfg: Experiment configuration.

    Returns:
        One row per image: `path`, `filename`, `split`, `class_label`.
    """
    return build_image_manifest(cfg.data_root)


def step_2_boxes(manifest_full: pd.DataFrame) -> pd.DataFrame:
    """Sigma=0 MTCNN boxes for every image, reused from the `blurred_images` cache.

    This experiment never runs a face detector: `blurred_images` already
    detected every image at sigma=0 (unblurred) with MTCNN, and that
    detection is cached. Calling `raw_mtcnn_detections` with a fresh,
    default `blurred_images.Config` reproduces the exact config that cache
    was written under, so this call is a pure cache read.

    Args:
        manifest_full: Output of `step_1_manifest` -- must be the full,
            2,940-row manifest (same rows/order `blurred_images` used),
            not a subset, or the cache lookup will consider itself stale
            and attempt to re-detect.

    Returns:
        One row per image: `path`, `class_label`, `n_faces`, `best_box`
        (real tuple or `None`), `best_score`.
    """
    blurred_cfg = BlurredConfig()
    all_detections = raw_mtcnn_detections(blurred_cfg, manifest_full)
    sigma0 = all_detections.loc[
        all_detections["sigma"] == 0, ["path", "class_label", "n_faces", "best_box", "best_score"]
    ]
    return sigma0.reset_index(drop=True)


def step_3_exclude_boxless(
    manifest_full: pd.DataFrame, boxes_full: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop the one image with no sigma=0 box, from both the manifest and the boxes.

    Detection at sigma=0 is 99.93%/100% by class (2940 images), so exactly
    one image is expected to have no box. Raises rather than silently
    coping with a different count, since every downstream arm (including
    `"intact"`) depends on this producing exactly 2,939 aligned rows.

    Args:
        manifest_full: Output of `step_1_manifest` (2,940 rows).
        boxes_full: Output of `step_2_boxes` (2,940 rows).

    Returns:
        `(manifest, boxes, excluded)`: `manifest` and `boxes` are both
        2,939 rows, row-aligned to each other by `path`; `excluded` is
        `{"path": ..., "class_label": ...}` for the one dropped image.

    Raises:
        RuntimeError: If the number of box-less images is not exactly one.
    """
    missing = boxes_full["best_box"].isna()
    n_missing = int(missing.sum())
    if n_missing != 1:
        raise RuntimeError(
            f"Expected exactly one sigma=0 image with no MTCNN box, found {n_missing}. "
            "background_images assumes a single excluded row shared by every arm; "
            "check whether the blurred_images detection cache changed."
        )

    excluded_row = boxes_full.loc[missing].iloc[0]
    excluded = {"path": excluded_row["path"], "class_label": excluded_row["class_label"]}

    boxes = boxes_full.loc[~missing].reset_index(drop=True)
    manifest = manifest_full.loc[manifest_full["path"] != excluded["path"]].reset_index(drop=True)
    boxes = boxes.set_index("path").loc[manifest["path"]].reset_index()

    logger.info(
        "step_3_exclude_boxless: excluded path=%s class=%s -- %d rows remain",
        excluded["path"],
        excluded["class_label"],
        len(manifest),
    )
    return manifest, boxes, excluded


def step_4_box_geometry(cfg: Config, boxes: pd.DataFrame) -> pd.DataFrame:
    """Four scalar geometry features per image, from the (unexpanded, clipped) box.

    Checks whether the mask *shape* alone -- not its contents -- encodes
    class: a systematically larger or differently-placed box in one class
    would let a fully-masked image still leak class information through
    the shape of the hole.

    Args:
        cfg: Experiment configuration.
        boxes: Output of `step_3_exclude_boxless` (the 2,939-row boxes).

    Returns:
        One row per image: `path`, `class_label`, `box_area_fraction`,
        `box_centre_x`, `box_centre_y`, `box_aspect` (the last three
        normalised to `[0, 1]` fractions of `cfg.image_size`, except
        `box_aspect` which is width/height).
    """
    rows = []
    for path, box, class_label in zip(boxes["path"], boxes["best_box"], boxes["class_label"]):
        x1, y1, x2, y2 = expand_box(box, 1.0, cfg.image_size)
        width, height = x2 - x1, y2 - y1
        rows.append(
            {
                "path": path,
                "class_label": class_label,
                "box_area_fraction": (width * height) / (cfg.image_size**2),
                "box_centre_x": (x1 + x2) / 2.0 / cfg.image_size,
                "box_centre_y": (y1 + y2) / 2.0 / cfg.image_size,
                "box_aspect": width / height if height > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def step_5_geometry_effect_sizes(geometry_df: pd.DataFrame) -> pd.DataFrame:
    """Cliff's delta for each geometry feature, autistic vs. non_autistic.

    Args:
        geometry_df: Output of `step_4_box_geometry`.

    Returns:
        One row per feature in `GEOMETRY_FEATURE_NAMES`: `feature`,
        `n_autistic`, `n_non_autistic`, `mean_autistic`,
        `mean_non_autistic`, `cliffs_delta`.
    """
    autistic = geometry_df.loc[geometry_df["class_label"] == "autistic"]
    non_autistic = geometry_df.loc[geometry_df["class_label"] == "non_autistic"]

    rows = []
    for feature in GEOMETRY_FEATURE_NAMES:
        x, y = autistic[feature].to_numpy(), non_autistic[feature].to_numpy()
        rows.append(
            {
                "feature": feature,
                "n_autistic": len(x),
                "n_non_autistic": len(y),
                "mean_autistic": float(np.mean(x)),
                "mean_non_autistic": float(np.mean(y)),
                "cliffs_delta": cliffs_delta(x, y),
            }
        )
    return pd.DataFrame(rows)


def step_6_geometry_probe(cfg: Config, geometry_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate a probe on the four box-geometry features alone.

    Same model, same 5x10 repeated stratified CV, same seed as every other
    probe here. Establishes what mask *shape* alone could achieve -- every
    masked arm below must be read against this, not against 0.5, exactly
    as the colour-only floor in `blurred_images` was the correct null for
    "no facial signal remains" there.

    Args:
        cfg: Experiment configuration.
        geometry_df: Output of `step_4_box_geometry`.
        manifest: Output of `step_3_exclude_boxless` (2,939 rows).

    Returns:
        One row per `(repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()
    aligned = geometry_df.set_index("path").loc[manifest["path"]].reset_index()
    X = aligned[list(GEOMETRY_FEATURE_NAMES)].to_numpy()

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
        ]
    )
    return repeated_stratified_cv(X, y, model, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed)


def arm_masked_image(arm: str, geo_image: Image.Image, box: Box, cfg: Config) -> Image.Image:
    """The post-geometry, pre-normalisation image a given arm actually sees.

    The single source of truth for what each arm does to an image --
    embedding extraction (via `to_tensor_stage`) and the per-arm
    colour-only floor (via `colour_features`) both call this, so both see
    exactly the same pixels for a given arm.

    Args:
        arm: One of `cfg.arm_names` (`"intact"` or a key of `_ARM_KIND`).
        geo_image: RGB PIL image, already post-`geometry_stage` at
            `cfg.image_size`.
        box: `(x1, y1, x2, y2)` face box in `geo_image`'s coordinates.
        cfg: Experiment configuration (supplies `expand_tight`/`expand_wide`
            and `image_size`).

    Returns:
        A new RGB PIL image, same size as `geo_image` (`"intact"` returns
        `geo_image` itself, unmodified).

    Raises:
        ValueError: If `arm` is not `"intact"` and not a key of `_ARM_KIND`.
    """
    if arm == "intact":
        return geo_image
    if arm not in _ARM_KIND:
        raise ValueError(f"Unknown arm: {arm!r}. Known arms: {['intact', *_ARM_KIND]}")

    kind, which = _ARM_KIND[arm]
    expand = cfg.expand_tight if which == "tight" else cfg.expand_wide

    if kind == "face_masked":
        return masked_stage(geo_image, box, expand, keep="outside")
    if kind == "background_masked":
        return masked_stage(geo_image, box, expand, keep="inside")
    if kind == "face_crop":
        return crop_stage(geo_image, box, expand, geo_image.size[0])
    raise AssertionError(kind)  # pragma: no cover -- _ARM_KIND is exhaustive by construction


def _embedding_transform_for_arm(arm: str, box_lookup: Dict[str, Box], cfg: Config):
    """A transform usable by `extract_embeddings`: one image in, one tensor out.

    Per-image boxes can't be threaded through `extract_embeddings`'
    shared, unmodified `Dataset`/`transform` interface directly, since its
    `transform` callable only ever receives the opened `PIL.Image` -- not
    its path. `PIL.Image.open(path)` sets `.filename` on the object it
    returns, though, so the box for a given image is recovered from
    `image.filename` rather than needing a wider change to `embeddings.py`.
    """
    if arm == "intact":
        return intact_transform(size=cfg.image_size)

    def _transform(image: Image.Image) -> torch.Tensor:
        geo = geometry_stage(image, cfg.image_size)
        box = box_lookup[image.filename]
        masked = arm_masked_image(arm, geo, box, cfg)
        return to_tensor_stage(masked)

    return _transform


def _embeddings_cache_matches(
    meta: dict, cfg: Config, index: pd.DataFrame, paths, transform_name: str
) -> bool:
    same_order = list(index["path"]) == [str(p) for p in paths]
    return (
        same_order
        and meta.get("backbone") == cfg.backbone
        and meta.get("transform_name") == transform_name
        and meta.get("image_size") == cfg.image_size
        and meta.get("n_images") == len(paths)
    )


def step_7_embeddings(
    cfg: Config, manifest: pd.DataFrame, manifest_full: pd.DataFrame, boxes: pd.DataFrame
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Extract (or load cached) embeddings for every arm in `cfg.arm_names`.

    `"intact"` is never re-extracted: it is a row-subset of the
    `intact_images` cache (which has all 2,940 rows) down to the 2,939 in
    `manifest`, selected by path. The four masked/cropped arms are cached
    independently under `outputs/background_images/embeddings_{arm}.npy`.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_3_exclude_boxless` (2,939 rows) -- the
            row order every returned array matches.
        manifest_full: Output of `step_1_manifest` (2,940 rows) -- used
            only to validate the `intact_images` cache before subsetting
            it.
        boxes: Output of `step_3_exclude_boxless`, row-aligned to
            `manifest`.

    Returns:
        `(embeddings_by_arm, status)`: `embeddings_by_arm` maps each arm
        in `cfg.arm_names` to its `(2939, n_features)` array; `status` has
        one row per arm -- `arm`, `n_rows`, `cache_hit_before_call`,
        `wall_clock_seconds`.

    Raises:
        RuntimeError: If the cached `intact_images` embeddings don't match
            `manifest_full` (wrong transform, stale row order/count) --
            subsetting a mismatched cache would silently mislabel rows.
    """
    paths = [Path(p) for p in manifest["path"]]
    box_lookup: Dict[str, Box] = {str(Path(p)): box for p, box in zip(boxes["path"], boxes["best_box"])}

    embeddings_by_arm: Dict[str, np.ndarray] = {}
    status_rows = []

    for arm in cfg.arm_names:
        if arm == "intact":
            t0 = time.perf_counter()
            full_array, full_index, meta = load_embeddings(cfg.intact_cache_dir)
            full_paths = [Path(p) for p in manifest_full["path"]]
            if not _embeddings_cache_matches(meta, cfg, full_index, full_paths, "intact_transform"):
                raise RuntimeError(
                    "step_7_embeddings: the intact_images embedding cache does not match "
                    "the full 2,940-row manifest -- refusing to subset a mismatched cache. "
                    "Re-run intact_images.step_2_embeddings first."
                )
            row_by_path = {p: i for i, p in enumerate(full_index["path"])}
            array = full_array[[row_by_path[str(p)] for p in paths]]
            elapsed = time.perf_counter() - t0

            embeddings_by_arm[arm] = array
            status_rows.append(
                {"arm": arm, "n_rows": array.shape[0], "cache_hit_before_call": True, "wall_clock_seconds": elapsed}
            )
            logger.info("step_7_embeddings: arm=intact row-subset of intact_images cache, shape=%s", array.shape)
            continue

        name = f"embeddings_{arm}"
        transform_name = f"{arm}_transform"
        array_path = cfg.output_dir / f"{name}.npy"
        cache_hit_before = array_path.exists()

        if cache_hit_before:
            cached_array, cached_index, meta = load_embeddings(cfg.output_dir, name=name)
            if _embeddings_cache_matches(meta, cfg, cached_index, paths, transform_name):
                embeddings_by_arm[arm] = cached_array
                status_rows.append(
                    {"arm": arm, "n_rows": cached_array.shape[0], "cache_hit_before_call": True, "wall_clock_seconds": 0.0}
                )
                logger.info("step_7_embeddings: arm=%s using cached embeddings, shape=%s", arm, cached_array.shape)
                continue
            logger.info("step_7_embeddings: arm=%s cache present but stale, re-extracting", arm)

        transform = _embedding_transform_for_arm(arm, box_lookup, cfg)
        start = time.perf_counter()
        embeddings = extract_embeddings(
            paths, transform, backbone=cfg.backbone, batch_size=cfg.batch_size, device=cfg.device
        )
        elapsed = time.perf_counter() - start

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else cfg.device
        meta = {
            "backbone": cfg.backbone,
            "weights": get_backbone_weights_name(cfg.backbone),
            "transform_name": transform_name,
            "arm": arm,
            "image_size": cfg.image_size,
            "n_images": len(paths),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": device_name,
            "wall_clock_seconds": elapsed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_embeddings(embeddings, paths, meta, cfg.output_dir, name=name)
        logger.info(
            "step_7_embeddings: arm=%s extracted shape=%s in %.1fs (cache written)", arm, embeddings.shape, elapsed
        )
        embeddings_by_arm[arm] = embeddings
        status_rows.append(
            {"arm": arm, "n_rows": embeddings.shape[0], "cache_hit_before_call": False, "wall_clock_seconds": elapsed}
        )

    return embeddings_by_arm, pd.DataFrame(status_rows)


def step_8_verify_embeddings(embeddings_by_arm: Dict[str, np.ndarray], expected_rows: int) -> pd.DataFrame:
    """Sanity-check every arm's embedding array before trusting it in a probe.

    Args:
        embeddings_by_arm: Output of `step_7_embeddings`.
        expected_rows: The row count every array must have (2,939 here).

    Returns:
        One row per arm: `arm`, `n_rows`, `rows_match_expected`, `n_nan`,
        `n_all_zero_rows`.
    """
    rows = []
    for arm, array in embeddings_by_arm.items():
        rows.append(
            {
                "arm": arm,
                "n_rows": array.shape[0],
                "rows_match_expected": array.shape[0] == expected_rows,
                "n_nan": int(np.isnan(array).sum()),
                "n_all_zero_rows": int((np.abs(array).sum(axis=1) == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def step_9_probe(cfg: Config, embeddings_by_arm: Dict[str, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate the shared probe independently on each arm's embeddings.

    Same L2-logistic-regression-behind-a-`StandardScaler`, same 5x10
    repeated stratified CV, same seed as `intact_images` and
    `blurred_images` -- only the embeddings (and therefore the transform
    behind them) differ by arm.

    Args:
        cfg: Experiment configuration.
        embeddings_by_arm: Output of `step_7_embeddings`.
        manifest: Output of `step_3_exclude_boxless` (2,939 rows).

    Returns:
        One row per `(arm, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for arm in cfg.arm_names:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
            ]
        )
        fold_results = repeated_stratified_cv(
            embeddings_by_arm[arm], y, model, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        fold_results.insert(0, "arm", arm)
        rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_10_summary(cfg: Config, fold_results: pd.DataFrame, baseline_arm: str = "intact") -> pd.DataFrame:
    """Summarize per-fold results into mean/sd per arm, plus the paired AUC delta.

    Because the seed, fold count and manifest order are unchanged across
    arms, fold *k* of repeat *r* holds out the same images for every arm --
    so the AUC delta against `baseline_arm` is a paired per-fold
    difference, not a difference of independent means.

    Args:
        cfg: Experiment configuration (for `cfg.arm_names`' order).
        fold_results: Output of `step_9_probe` (or `step_11_colour_probe`,
            which shares this shape).
        baseline_arm: The arm every other arm's AUC delta is computed
            against.

    Returns:
        One row per arm, in `cfg.arm_names` order: mean and sd of each
        metric (`{metric}_mean`, `{metric}_sd`), plus
        `auc_delta_vs_{baseline_arm}_mean` and
        `auc_delta_vs_{baseline_arm}_sd` (both exactly 0 for
        `baseline_arm` itself).
    """
    metric_cols = [c for c in fold_results.columns if c not in ("arm", "repeat", "fold")]
    baseline = fold_results.loc[fold_results["arm"] == baseline_arm, ["repeat", "fold", "roc_auc"]].rename(
        columns={"roc_auc": "roc_auc_baseline"}
    )

    summary_rows = []
    for arm in cfg.arm_names:
        group = fold_results.loc[fold_results["arm"] == arm]
        row = {"arm": arm}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std()

        paired = group.merge(baseline, on=["repeat", "fold"])
        delta = paired["roc_auc"] - paired["roc_auc_baseline"]
        row[f"auc_delta_vs_{baseline_arm}_mean"] = delta.mean()
        row[f"auc_delta_vs_{baseline_arm}_sd"] = delta.std()
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _colour_features_cache_matches(cached: pd.DataFrame, manifest: pd.DataFrame) -> bool:
    return len(cached) == len(manifest) and list(cached["path"]) == list(manifest["path"])


def step_11_colour_features(cfg: Config, manifest: pd.DataFrame, boxes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extract (or load cached) six colour statistics per arm.

    Masking changes an image's colour statistics -- removing the face
    leaves background colour, removing the background leaves skin colour --
    so each arm has its own colour-only floor, computed on exactly the
    image that arm's embedding saw (`arm_masked_image`). These floors are
    not interchangeable with each other or with `blurred_images`'.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_3_exclude_boxless` (2,939 rows).
        boxes: Output of `step_3_exclude_boxless`, row-aligned to
            `manifest`.

    Returns:
        Dict mapping each arm in `cfg.arm_names` to a DataFrame with
        `path` plus one column per name in
        `src.data.colour_features.COLOUR_FEATURE_NAMES`.
    """
    box_by_path = dict(zip(boxes["path"], boxes["best_box"]))
    features_by_arm: Dict[str, pd.DataFrame] = {}

    for arm in cfg.arm_names:
        name = f"colour_features_{arm}"
        cache_path = cfg.output_dir / f"{name}.csv"
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            if _colour_features_cache_matches(cached, manifest):
                logger.info("step_11_colour_features: arm=%s using cached features", arm)
                features_by_arm[arm] = cached
                continue
            logger.info("step_11_colour_features: arm=%s cache present but stale, recomputing", arm)

        rows = []
        for path in manifest["path"]:
            geo = geometry_stage(Image.open(path), cfg.image_size)
            masked = arm_masked_image(arm, geo, box_by_path[path], cfg)
            rows.append(colour_features(masked))
        df = pd.DataFrame(rows, columns=list(COLOUR_FEATURE_NAMES))
        df.insert(0, "path", manifest["path"].to_numpy())
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("step_11_colour_features: arm=%s computed and cached %d rows", arm, len(df))
        features_by_arm[arm] = df

    return features_by_arm


def step_12_colour_probe(
    cfg: Config, colour_features_by_arm: Dict[str, pd.DataFrame], manifest: pd.DataFrame
) -> pd.DataFrame:
    """Cross-validate the colour-only reference probe, independently per arm.

    Same model shape, same 5x10 repeated stratified CV, same seed as
    `step_9_probe` -- only the six per-arm colour features replace the
    2048-d embedding as input. Same output shape as `step_9_probe`, so
    `step_10_summary` summarises either directly.

    Args:
        cfg: Experiment configuration.
        colour_features_by_arm: Output of `step_11_colour_features`.
        manifest: Output of `step_3_exclude_boxless` (2,939 rows).

    Returns:
        One row per `(arm, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for arm in cfg.arm_names:
        aligned = colour_features_by_arm[arm].set_index("path").loc[manifest["path"]].reset_index()
        X = aligned[list(COLOUR_FEATURE_NAMES)].to_numpy()
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
            ]
        )
        fold_results = repeated_stratified_cv(
            X, y, model, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        fold_results.insert(0, "arm", arm)
        rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)
