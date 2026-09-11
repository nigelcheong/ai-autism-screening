"""Pipeline steps for the face_crop_blur experiment.

Part 0 first: `background_images` reported mean box area fractions of
0.586 (autistic) / 0.704 (non_autistic) at the *unexpanded* box. Taking
square roots, the unexpanded box already spans ~77%/84% of each frame
side, so expanding by 1.5x (the factor `face_crop_1.5` and
`face_masked_1.5` both used) very often clips to the full 224x224 frame.
`step_4_coverage_diagnostics` below quantifies exactly how often, from the
same cached sigma=0 boxes those experiments used, before anything else in
this module runs.

Everything downstream repeats `intact_images` / `blurred_images` /
`background_images` exactly -- same manifest source, same model, same
5x10 repeated stratified CV, same seed -- and changes only the transform:
crop to a *tight* (never 1.5x-expanded) face box, optionally
resolution-equalised, then blur. Boxes are never (re-)detected: they are
the same cached sigma=0 MTCNN boxes `background_images` used, read via
that module's own `step_2_boxes` / `step_3_exclude_boxless` so the two
experiments' box-handling logic is identical by construction, not by
copy-paste.
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

from experiments.background_images.pipeline import step_2_boxes, step_3_exclude_boxless
from experiments.face_crop_blur.config import Config
from src.data.colour_features import COLOUR_FEATURE_NAMES, colour_features
from src.data.images import build_image_manifest
from src.data.transforms import (
    Box,
    blur_stage,
    crop_stage,
    expand_box,
    geometry_stage,
    resample_stage,
    to_tensor_stage,
)
from src.evaluation.cross_val import repeated_stratified_cv
from src.models.embeddings import (
    extract_embeddings,
    get_backbone_weights_name,
    load_embeddings,
    save_embeddings,
)

logger = logging.getLogger(__name__)

# Re-exported so the notebook can build the full (2940-row) manifest and
# reuse background_images's exact box-handling logic without importing
# from two different modules for one concept.
__all__ = [
    "step_1_manifest",
    "step_2_boxes",
    "step_3_exclude_boxless",
    "step_4_coverage_diagnostics",
    "step_5_image_sizes",
    "step_6_apply_min_size_exclusion",
    "step_7_upsampling_factors",
    "step_8_embeddings",
    "step_9_verify_embeddings",
    "step_10_colour_features",
    "step_11_probe",
    "step_12_colour_probe",
    "step_13_summary",
    "arm_crop_image",
]


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the full image manifest -- identical to every prior experiment, all 2,940 rows.

    Args:
        cfg: Experiment configuration.

    Returns:
        One row per image: `path`, `filename`, `split`, `class_label`.
    """
    return build_image_manifest(cfg.data_root)


def step_4_coverage_diagnostics(
    cfg: Config, boxes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Part 0: per-class, per-factor frame coverage of the expanded, clipped box.

    Computed directly from the cached sigma=0 boxes `background_images`
    used -- the same boxes, the same `expand_box`, no re-detection. This
    is the diagnostic that decides whether `background_images`'s
    `face_crop_1.5` / `face_masked_1.5` arms were doing what their names
    claim.

    Args:
        cfg: Experiment configuration (supplies `coverage_factors` and
            `min_coverage_fraction`).
        boxes: A boxes table with `path`, `class_label`, `best_box`
            columns (e.g. `step_3_exclude_boxless`'s second return value).

    Returns:
        `(coverage, retained)`:
        `coverage` has one row per `(factor, class_label)`: `n`,
        `mean_area_fraction` (mean, over images, of the expanded-and-
        clipped box's area as a fraction of the 224x224 frame), and
        `prop_ge_threshold` (proportion of images at or above
        `cfg.min_coverage_fraction`).
        `retained` has one row per `(factor, class_label)`:
        `mean_retained_fraction_face_masked` -- the mean fraction of
        pixels a `face_masked`-style transform (`keep="outside"`) at that
        factor would retain, i.e. `1 - area_fraction` per image, then
        averaged.
    """
    coverage_rows = []
    retained_rows = []

    for factor in cfg.coverage_factors:
        for class_label, group in boxes.groupby("class_label"):
            area_fractions = []
            for box in group["best_box"]:
                x1, y1, x2, y2 = expand_box(box, factor, cfg.image_size)
                area_fractions.append(((x2 - x1) * (y2 - y1)) / (cfg.image_size**2))
            area_fractions = np.asarray(area_fractions)
            retained_fractions = 1.0 - area_fractions

            coverage_rows.append(
                {
                    "factor": factor,
                    "class_label": class_label,
                    "n": len(area_fractions),
                    "mean_area_fraction": float(area_fractions.mean()),
                    "prop_ge_threshold": float((area_fractions >= cfg.min_coverage_fraction).mean()),
                }
            )
            retained_rows.append(
                {
                    "factor": factor,
                    "class_label": class_label,
                    "n": len(retained_fractions),
                    "mean_retained_fraction_face_masked": float(retained_fractions.mean()),
                }
            )

    coverage = pd.DataFrame(coverage_rows).sort_values(["factor", "class_label"]).reset_index(drop=True)
    retained = pd.DataFrame(retained_rows).sort_values(["factor", "class_label"]).reset_index(drop=True)
    return coverage, retained


def step_5_image_sizes(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Original (pre-letterbox) pixel dimensions for every image in `manifest`.

    Reads only each file's header via `PIL.Image.open(...).size` -- no
    pixel decode, and no write to `data/raw/images/`. Needed to convert a
    box's post-geometry (224x224-space) size back into original-image
    pixel coordinates, for the minimum-box-size exclusion and the
    upsampling-factor report.

    Args:
        cfg: Experiment configuration.
        manifest: Any manifest with a `path` column.

    Returns:
        DataFrame with `path`, `width`, `height` (original pixels).
    """
    cache_path = cfg.output_dir / "image_sizes.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(manifest) and set(cached["path"]) == set(manifest["path"]):
            logger.info("step_5_image_sizes: using cached sizes (%d rows)", len(cached))
            return cached
        logger.info("step_5_image_sizes: cache present but stale, recomputing")

    rows = []
    for path in manifest["path"]:
        with Image.open(path) as img:
            width, height = img.size
        rows.append({"path": path, "width": width, "height": height})

    df = pd.DataFrame(rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("step_5_image_sizes: computed and cached %d rows", len(df))
    return df


def step_6_apply_min_size_exclusion(
    cfg: Config, manifest: pd.DataFrame, boxes: pd.DataFrame, sizes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop images whose `crop_0.8` box is smaller than `cfg.min_box_size_px`, in original pixels.

    `geometry_stage` letterboxes each image to a square canvas of side
    `max(width, height)` before resizing to `cfg.image_size`, so the scale
    factor back from 224-space to original-pixel-space is
    `max(width, height) / cfg.image_size`, uniform in x and y. Applied
    once, at the tighter (`crop_0.8`) factor, and the resulting exclusion
    set is shared by every arm -- including `crop_1.0` and `whole_image`
    -- so no comparison in this experiment ever mixes rows across
    different exclusion sets.

    Args:
        cfg: Experiment configuration.
        manifest: Row-aligned to `boxes` (e.g. from `step_3_exclude_boxless`).
        boxes: Row-aligned to `manifest`.
        sizes: Output of `step_5_image_sizes`, covering every path in
            `manifest`.

    Returns:
        `(manifest_final, boxes_final, excluded, counts_by_class)`:
        `manifest_final` and `boxes_final` are row-aligned to each other,
        with the small-box images dropped; `excluded` has one row per
        dropped image (`path`, `class_label`, `box_w_orig`, `box_h_orig`);
        `counts_by_class` has one row per class: `n_excluded`.
    """
    sizes_by_path = sizes.set_index("path")
    tight_factor = cfg.crop_factors["crop_0.8"]

    rows = []
    for path, box, class_label in zip(boxes["path"], boxes["best_box"], boxes["class_label"]):
        x1, y1, x2, y2 = expand_box(box, tight_factor, cfg.image_size)
        width_224, height_224 = x2 - x1, y2 - y1

        width_px = float(sizes_by_path.loc[path, "width"])
        height_px = float(sizes_by_path.loc[path, "height"])
        scale = max(width_px, height_px) / cfg.image_size

        box_w_orig, box_h_orig = width_224 * scale, height_224 * scale
        rows.append(
            {
                "path": path,
                "class_label": class_label,
                "box_w_orig": box_w_orig,
                "box_h_orig": box_h_orig,
                "excluded": min(box_w_orig, box_h_orig) < cfg.min_box_size_px,
            }
        )

    diagnostics = pd.DataFrame(rows)
    excluded = diagnostics.loc[diagnostics["excluded"]].drop(columns=["excluded"]).reset_index(drop=True)
    counts_by_class = (
        excluded.groupby("class_label").size().rename("n_excluded").reset_index()
        if len(excluded)
        else pd.DataFrame(columns=["class_label", "n_excluded"])
    )

    keep_paths = set(diagnostics.loc[~diagnostics["excluded"], "path"])
    manifest_final = manifest.loc[manifest["path"].isin(keep_paths)].reset_index(drop=True)
    boxes_final = boxes.set_index("path").loc[manifest_final["path"]].reset_index()

    logger.info(
        "step_6_apply_min_size_exclusion: excluded %d/%d images (box_0.8 < %.0fpx in original coords) -- %d rows remain",
        len(excluded),
        len(manifest),
        cfg.min_box_size_px,
        len(manifest_final),
    )
    return manifest_final, boxes_final, excluded, counts_by_class


def step_7_upsampling_factors(cfg: Config, boxes: pd.DataFrame) -> pd.DataFrame:
    """Per-image upsampling factor for `crop_1.0` and `crop_0.8`.

    A crop is taken from the box's extent in the post-geometry 224x224
    image, then resized back up to 224x224. The upsampling factor
    reported here is the linear scale of that resize, `image_size /
    sqrt(box_area)`, using the geometric mean of the (possibly unequal)
    width and height scale factors as a single summary number per image.

    Args:
        cfg: Experiment configuration.
        boxes: A boxes table with `path`, `class_label`, `best_box`
            columns.

    Returns:
        DataFrame with one row per `(arm, path)`: `arm` (`"crop_1.0"` or
        `"crop_0.8"`), `path`, `class_label`, `upsample_factor`.
    """
    rows = []
    for arm in ("crop_1.0", "crop_0.8"):
        factor = cfg.crop_factors[arm]
        for path, box, class_label in zip(boxes["path"], boxes["best_box"], boxes["class_label"]):
            x1, y1, x2, y2 = expand_box(box, factor, cfg.image_size)
            area = max((x2 - x1) * (y2 - y1), 1e-9)
            upsample_factor = cfg.image_size / np.sqrt(area)
            rows.append(
                {"arm": arm, "path": path, "class_label": class_label, "upsample_factor": upsample_factor}
            )
    return pd.DataFrame(rows)


def arm_crop_image(arm: str, geo_image: Image.Image, box: Box, cfg: Config) -> Image.Image:
    """The post-crop, pre-blur image a given crop arm actually sees.

    The single source of truth for what each crop arm does to an image
    before blur -- embedding extraction and the per-arm colour-only floor
    both call this (indirectly, via the sigma=0 case), so both see
    exactly the same pixels.

    Args:
        arm: One of `"crop_1.0"`, `"crop_0.8"`, `"crop_1.0_equalised"`
            (`"whole_image"` is not a crop and is not handled here).
        geo_image: RGB PIL image, already post-`geometry_stage` at
            `cfg.image_size`.
        box: `(x1, y1, x2, y2)` face box in `geo_image`'s coordinates.
        cfg: Experiment configuration.

    Returns:
        A new RGB PIL image, size `(cfg.image_size, cfg.image_size)`.

    Raises:
        ValueError: If `arm` is not a known crop arm.
    """
    if arm not in cfg.crop_factors:
        raise ValueError(f"Unknown crop arm: {arm!r}. Known crop arms: {list(cfg.crop_factors)}")

    cropped = crop_stage(geo_image, box, cfg.crop_factors[arm], cfg.image_size)
    if arm in cfg.equalised_arms:
        cropped = resample_stage(cropped, cfg.small_size, cfg.image_size)
    return cropped


def _crop_blur_transform(arm: str, sigma: int, box_lookup: Dict[str, Box], cfg: Config):
    """A transform usable by `extract_embeddings`: one image in, one tensor out.

    Per-image boxes are recovered from `image.filename` (set by
    `PIL.Image.open`), the same mechanism `background_images` uses --
    `extract_embeddings`'s shared `Dataset`/`transform` interface only
    ever passes the opened image, not its path.
    """

    def _transform(image: Image.Image) -> torch.Tensor:
        geo = geometry_stage(image, cfg.image_size)
        box = box_lookup[image.filename]
        cropped = arm_crop_image(arm, geo, box, cfg)
        blurred = blur_stage(cropped, sigma)
        return to_tensor_stage(blurred)

    return _transform


def _cache_matches(meta: dict, cfg: Config, index: pd.DataFrame, paths, transform_name: str) -> bool:
    same_order = list(index["path"]) == [str(p) for p in paths]
    return (
        same_order
        and meta.get("backbone") == cfg.backbone
        and meta.get("transform_name") == transform_name
        and meta.get("image_size") == cfg.image_size
        and meta.get("n_images") == len(paths)
    )


def _whole_image_embeddings_for_sigma(cfg: Config, sigma: int, manifest: pd.DataFrame) -> np.ndarray:
    """Row-subset of the cached whole-image embedding for `sigma`, aligned to `manifest`.

    sigma=0 comes from `intact_images` (the only place it was cached
    standalone); sigma>0 comes from `blurred_images`. Neither is
    re-extracted here.
    """
    if sigma == 0:
        full_array, full_index, _ = load_embeddings(cfg.intact_cache_dir)
    else:
        full_array, full_index, _ = load_embeddings(cfg.blurred_cache_dir, name=f"embeddings_sigma{sigma}")

    row_by_path = {p: i for i, p in enumerate(full_index["path"])}
    idx = [row_by_path[str(Path(p))] for p in manifest["path"]]
    return full_array[idx]


def _whole_image_colour_features_for_sigma(cfg: Config, sigma: int, manifest: pd.DataFrame) -> pd.DataFrame:
    """Row-subset of `blurred_images`'s cached colour features for `sigma`, aligned to `manifest`."""
    cached = pd.read_csv(cfg.blurred_cache_dir / f"colour_features_sigma{sigma}.csv")
    return cached.set_index("path").loc[manifest["path"]].reset_index()


def step_8_embeddings(
    cfg: Config, manifest: pd.DataFrame, boxes: pd.DataFrame
) -> Tuple[Dict[Tuple[str, int], np.ndarray], pd.DataFrame]:
    """Extract (or load cached) embeddings for every `(arm, sigma)` pair.

    `"whole_image"` is never (re-)extracted here: it is a row-subset of
    the `intact_images` (sigma=0) / `blurred_images` (sigma>0) caches,
    selected down to `manifest`'s rows. The three crop arms are each
    cached independently under
    `outputs/face_crop_blur/embeddings_{arm}_sigma{sigma}.npy`.

    Args:
        cfg: Experiment configuration.
        manifest: The row order every returned array matches (already
            reduced by every exclusion step).
        boxes: Row-aligned to `manifest`.

    Returns:
        `(embeddings, status)`: `embeddings` maps each `(arm, sigma)` in
        `cfg.arm_names x cfg.sigmas` to its `(len(manifest), n_features)`
        array; `status` has one row per `(arm, sigma)` -- `arm`, `sigma`,
        `n_rows`, `cache_hit_before_call`, `wall_clock_seconds`.
    """
    paths = [Path(p) for p in manifest["path"]]
    box_lookup: Dict[str, Box] = {str(Path(p)): box for p, box in zip(boxes["path"], boxes["best_box"])}

    embeddings: Dict[Tuple[str, int], np.ndarray] = {}
    status_rows = []

    for arm in cfg.arm_names:
        for sigma in cfg.sigmas:
            if arm == "whole_image":
                t0 = time.perf_counter()
                array = _whole_image_embeddings_for_sigma(cfg, sigma, manifest)
                elapsed = time.perf_counter() - t0
                embeddings[(arm, sigma)] = array
                status_rows.append(
                    {
                        "arm": arm,
                        "sigma": sigma,
                        "n_rows": array.shape[0],
                        "cache_hit_before_call": True,
                        "wall_clock_seconds": elapsed,
                    }
                )
                logger.info(
                    "step_8_embeddings: arm=whole_image sigma=%s row-subset of cache, shape=%s",
                    sigma,
                    array.shape,
                )
                continue

            name = f"embeddings_{arm}_sigma{sigma}"
            transform_name = f"{arm}_sigma{sigma}_transform"
            array_path = cfg.output_dir / f"{name}.npy"
            cache_hit_before = array_path.exists()

            if cache_hit_before:
                cached_array, cached_index, meta = load_embeddings(cfg.output_dir, name=name)
                if _cache_matches(meta, cfg, cached_index, paths, transform_name):
                    embeddings[(arm, sigma)] = cached_array
                    status_rows.append(
                        {
                            "arm": arm,
                            "sigma": sigma,
                            "n_rows": cached_array.shape[0],
                            "cache_hit_before_call": True,
                            "wall_clock_seconds": 0.0,
                        }
                    )
                    logger.info(
                        "step_8_embeddings: arm=%s sigma=%s using cached embeddings, shape=%s",
                        arm,
                        sigma,
                        cached_array.shape,
                    )
                    continue
                logger.info(
                    "step_8_embeddings: arm=%s sigma=%s cache present but stale, re-extracting", arm, sigma
                )

            transform = _crop_blur_transform(arm, sigma, box_lookup, cfg)
            start = time.perf_counter()
            array = extract_embeddings(
                paths, transform, backbone=cfg.backbone, batch_size=cfg.batch_size, device=cfg.device
            )
            elapsed = time.perf_counter() - start

            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else cfg.device
            meta = {
                "backbone": cfg.backbone,
                "weights": get_backbone_weights_name(cfg.backbone),
                "transform_name": transform_name,
                "arm": arm,
                "sigma": sigma,
                "image_size": cfg.image_size,
                "n_images": len(paths),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "device": device_name,
                "wall_clock_seconds": elapsed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_embeddings(array, paths, meta, cfg.output_dir, name=name)
            logger.info(
                "step_8_embeddings: arm=%s sigma=%s extracted shape=%s in %.1fs (cache written)",
                arm,
                sigma,
                array.shape,
                elapsed,
            )
            embeddings[(arm, sigma)] = array
            status_rows.append(
                {
                    "arm": arm,
                    "sigma": sigma,
                    "n_rows": array.shape[0],
                    "cache_hit_before_call": False,
                    "wall_clock_seconds": elapsed,
                }
            )

    return embeddings, pd.DataFrame(status_rows)


def step_9_verify_embeddings(
    embeddings: Dict[Tuple[str, int], np.ndarray], expected_rows: int
) -> pd.DataFrame:
    """Sanity-check every `(arm, sigma)` embedding array before trusting it in a probe.

    Args:
        embeddings: Output of `step_8_embeddings`.
        expected_rows: The row count every array must have.

    Returns:
        One row per `(arm, sigma)`: `arm`, `sigma`, `n_rows`,
        `rows_match_expected`, `n_nan`, `n_all_zero_rows`.
    """
    rows = []
    for (arm, sigma), array in embeddings.items():
        rows.append(
            {
                "arm": arm,
                "sigma": sigma,
                "n_rows": array.shape[0],
                "rows_match_expected": array.shape[0] == expected_rows,
                "n_nan": int(np.isnan(array).sum()),
                "n_all_zero_rows": int((np.abs(array).sum(axis=1) == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "sigma"]).reset_index(drop=True)


def _colour_features_cache_matches(cached: pd.DataFrame, manifest: pd.DataFrame) -> bool:
    return len(cached) == len(manifest) and list(cached["path"]) == list(manifest["path"])


def step_10_colour_features(
    cfg: Config, manifest: pd.DataFrame, boxes: pd.DataFrame
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """Extract (or load cached) six colour statistics for every `(arm, sigma)` pair.

    A crop's colour statistics are skin, not background -- this is not
    interchangeable with `background_images`'s per-arm floors, nor with
    `blurred_images`'s whole-image floor, except for `"whole_image"`
    itself, which reuses `blurred_images`'s cached colour features
    directly (row-subset to `manifest`).

    Args:
        cfg: Experiment configuration.
        manifest: Row order every returned table matches.
        boxes: Row-aligned to `manifest`.

    Returns:
        Dict mapping each `(arm, sigma)` in `cfg.arm_names x cfg.sigmas`
        to a DataFrame with `path` plus one column per name in
        `src.data.colour_features.COLOUR_FEATURE_NAMES`.
    """
    box_lookup = dict(zip(boxes["path"], boxes["best_box"]))
    features: Dict[Tuple[str, int], pd.DataFrame] = {}

    for arm in cfg.arm_names:
        for sigma in cfg.sigmas:
            if arm == "whole_image":
                features[(arm, sigma)] = _whole_image_colour_features_for_sigma(cfg, sigma, manifest)
                logger.info("step_10_colour_features: arm=whole_image sigma=%s row-subset of cache", sigma)
                continue

            name = f"colour_features_{arm}_sigma{sigma}"
            cache_path = cfg.output_dir / f"{name}.csv"
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                if _colour_features_cache_matches(cached, manifest):
                    logger.info("step_10_colour_features: arm=%s sigma=%s using cached features", arm, sigma)
                    features[(arm, sigma)] = cached
                    continue
                logger.info(
                    "step_10_colour_features: arm=%s sigma=%s cache present but stale, recomputing", arm, sigma
                )

            rows = []
            for path in manifest["path"]:
                geo = geometry_stage(Image.open(path), cfg.image_size)
                cropped = arm_crop_image(arm, geo, box_lookup[path], cfg)
                blurred = blur_stage(cropped, sigma)
                rows.append(colour_features(blurred))

            df = pd.DataFrame(rows, columns=list(COLOUR_FEATURE_NAMES))
            df.insert(0, "path", manifest["path"].to_numpy())
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(
                "step_10_colour_features: arm=%s sigma=%s computed and cached %d rows", arm, sigma, len(df)
            )
            features[(arm, sigma)] = df

    return features


def _build_model(cfg: Config) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
        ]
    )


def step_11_probe(
    cfg: Config, embeddings: Dict[Tuple[str, int], np.ndarray], manifest: pd.DataFrame
) -> pd.DataFrame:
    """Cross-validate the shared probe independently on every `(arm, sigma)` embedding.

    Same L2-logistic-regression-behind-a-`StandardScaler`, same 5x10
    repeated stratified CV, same seed as every prior experiment -- only
    the embeddings differ.

    Args:
        cfg: Experiment configuration.
        embeddings: Output of `step_8_embeddings`.
        manifest: Row-aligned to every array in `embeddings`.

    Returns:
        One row per `(arm, sigma, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for arm in cfg.arm_names:
        for sigma in cfg.sigmas:
            fold_results = repeated_stratified_cv(
                embeddings[(arm, sigma)],
                y,
                _build_model(cfg),
                n_splits=cfg.n_splits,
                n_repeats=cfg.n_repeats,
                seed=cfg.seed,
            )
            fold_results.insert(0, "sigma", sigma)
            fold_results.insert(0, "arm", arm)
            rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_12_colour_probe(
    cfg: Config, colour_features_by_arm_sigma: Dict[Tuple[str, int], pd.DataFrame], manifest: pd.DataFrame
) -> pd.DataFrame:
    """Cross-validate the colour-only reference probe, independently per `(arm, sigma)`.

    Same shape as `step_11_probe`'s output, so `step_13_summary`
    summarises either directly. This is the correct null for "no facial
    signal remains" *within a crop*: colour is skin tone here, not
    background, so it must be recomputed per arm rather than reused from
    `blurred_images` or `background_images`.

    Args:
        cfg: Experiment configuration.
        colour_features_by_arm_sigma: Output of `step_10_colour_features`.
        manifest: Row-aligned to every table in `colour_features_by_arm_sigma`.

    Returns:
        One row per `(arm, sigma, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for arm in cfg.arm_names:
        for sigma in cfg.sigmas:
            aligned = (
                colour_features_by_arm_sigma[(arm, sigma)].set_index("path").loc[manifest["path"]].reset_index()
            )
            X = aligned[list(COLOUR_FEATURE_NAMES)].to_numpy()
            fold_results = repeated_stratified_cv(
                X, y, _build_model(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
            )
            fold_results.insert(0, "sigma", sigma)
            fold_results.insert(0, "arm", arm)
            rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_13_summary(cfg: Config, fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold results into mean/sd per `(arm, sigma)`, plus the paired AUC delta vs. that arm's own sigma=0.

    Because the seed, fold count and manifest order are unchanged across
    sigma within an arm, fold *k* of repeat *r* holds out the same images
    at every sigma -- so the AUC delta against `sigma=min(cfg.sigmas)` is
    a paired per-fold difference, computed separately per arm (an
    embedding-probe curve is never compared against a colour-probe
    baseline, or one arm against another arm's sigma=0).

    Args:
        cfg: Experiment configuration (for `cfg.arm_names` order).
        fold_results: Output of `step_11_probe` or `step_12_colour_probe`
            (same shape).

    Returns:
        One row per `(arm, sigma)`: mean and sd of each metric
        (`{metric}_mean`, `{metric}_sd`), plus `auc_delta_vs_sigma0_mean`
        and `auc_delta_vs_sigma0_sd` (both exactly 0 at that arm's own
        sigma=0).
    """
    metric_cols = [c for c in fold_results.columns if c not in ("arm", "sigma", "repeat", "fold")]
    min_sigma = min(cfg.sigmas)

    summary_rows = []
    for arm in cfg.arm_names:
        arm_df = fold_results.loc[fold_results["arm"] == arm]
        baseline = arm_df.loc[arm_df["sigma"] == min_sigma, ["repeat", "fold", "roc_auc"]].rename(
            columns={"roc_auc": "roc_auc_baseline"}
        )
        for sigma, group in arm_df.groupby("sigma"):
            row = {"arm": arm, "sigma": sigma}
            for col in metric_cols:
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_sd"] = group[col].std()

            paired = group.merge(baseline, on=["repeat", "fold"])
            delta = paired["roc_auc"] - paired["roc_auc_baseline"]
            row["auc_delta_vs_sigma0_mean"] = delta.mean()
            row["auc_delta_vs_sigma0_sd"] = delta.std()
            summary_rows.append(row)

    order = {arm: i for i, arm in enumerate(cfg.arm_names)}
    summary = pd.DataFrame(summary_rows)
    summary["_arm_order"] = summary["arm"].map(order)
    return summary.sort_values(["_arm_order", "sigma"]).drop(columns=["_arm_order"]).reset_index(drop=True)
