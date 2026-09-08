"""Pipeline steps for the blurred_images experiment.

Repeats `intact_images` exactly -- same manifest, same model, same 5x10
repeated stratified CV, same seed -- and changes only the transform, sweeping
Gaussian blur strength (sigma) applied after resizing. sigma=0 reuses the
`intact_images` cache rather than recomputing it, so it is the same numbers
under a different name.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.blurred_images.config import Config
from src.data.colour_features import COLOUR_FEATURE_NAMES, colour_features
from src.data.images import build_image_manifest
from src.data.transforms import blur_stage, blurred_transform, geometry_stage
from src.evaluation.cross_val import repeated_stratified_cv
from src.models.embeddings import (
    extract_embeddings,
    get_backbone_weights_name,
    load_embeddings,
    save_embeddings,
)
from src.models.face_detect import (
    corroborated_detection_rate,
    detect_faces_haar,
    detect_faces_mtcnn,
    detection_rate,
    parse_box,
)

logger = logging.getLogger(__name__)


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the image manifest for the intact archive.

    Identical to `intact_images.step_1_manifest` -- same manifest, same
    row order.

    Args:
        cfg: Experiment configuration.

    Returns:
        One row per image: `path`, `filename`, `split`, `class_label`.
    """
    return build_image_manifest(cfg.data_root)


def _detections_cache_matches(cached: pd.DataFrame, sigmas: Sequence[int], manifest: pd.DataFrame) -> bool:
    expected_rows = len(manifest) * len(sigmas)
    return (
        len(cached) == expected_rows
        and set(cached["sigma"].unique()) == set(sigmas)
        and set(cached["path"].unique()) == set(manifest["path"])
    )


def _run_detector_sweep(
    cfg: Config,
    manifest: pd.DataFrame,
    sigmas: Sequence[int],
    detect_fn,
    cache_name: str,
) -> pd.DataFrame:
    """Run `detect_fn` at every sigma in `sigmas`, cached to `{cache_name}.csv`.

    Shared by the MTCNN and Haar sweeps -- the only difference between them
    is which detector function and sigma set is passed in.
    """
    cache_path = cfg.output_dir / f"{cache_name}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if _detections_cache_matches(cached, sigmas, manifest):
            logger.info("%s: using cached detections (%d rows)", cache_name, len(cached))
            cached["best_box"] = cached["best_box"].apply(parse_box)
            return cached
        logger.info("%s: cache present but stale, recomputing", cache_name)

    frames = []
    for sigma in sigmas:
        images = [
            blur_stage(geometry_stage(Image.open(p), cfg.image_size), sigma)
            for p in manifest["path"]
        ]
        detections = detect_fn(images)
        detections["path"] = manifest["path"].to_numpy()
        detections["class_label"] = manifest["class_label"].to_numpy()
        detections["sigma"] = sigma
        frames.append(detections)
        logger.info(
            "%s: sigma=%s detection rate=%.3f",
            cache_name,
            sigma,
            (detections["n_faces"] > 0).mean(),
        )

    result = pd.concat(frames, ignore_index=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    logger.info("%s: computed and cached %d rows", cache_name, len(result))
    return result


def raw_mtcnn_detections(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """MTCNN detections for every image at every sigma in `cfg.sigmas`.

    Cached to `outputs/blurred_images/detections_mtcnn.csv`. This is the
    primary detector for the experiment -- exposed at module level (not
    just folded into `step_2_detection`) so the notebook can also draw the
    boxes it found, e.g. on the highest-sigma sample grid.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_1_manifest`.

    Returns:
        One row per `(path, sigma)`: `n_faces`, `best_box` (real tuple or
        `None`), `best_score`, `path`, `class_label`, `sigma`.
    """
    return _run_detector_sweep(
        cfg,
        manifest,
        cfg.sigmas,
        lambda images: detect_faces_mtcnn(images, device=cfg.device),
        "detections_mtcnn",
    )


def raw_haar_detections(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Haar-cascade detections, kept only as a comparison against MTCNN.

    Fixed to `cfg.legacy_haar_sigmas` (the first run's sweep) rather than
    the (now longer) `cfg.sigmas`: the first run's cache
    (`outputs/blurred_images/detections_haar.csv`) is reused as-is, not
    recomputed or extended, since Haar is documentation here, not the
    "unrecognisable" anchor.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_1_manifest`.

    Returns:
        One row per `(path, sigma)`, same schema as `raw_mtcnn_detections`.
    """
    return _run_detector_sweep(cfg, manifest, cfg.legacy_haar_sigmas, detect_faces_haar, "detections_haar")


def step_2_detection(cfg: Config, manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Face-detection rates at every sigma: MTCNN raw, MTCNN corroborated, Haar raw.

    MTCNN is the primary detector, run (or loaded from cache) across the
    full `cfg.sigmas` sweep. Haar is run only across the first run's
    `cfg.legacy_haar_sigmas` and kept purely for comparison -- the two
    detectors' raw rates are not on the same footing (Haar is prone to
    false positives on blurred patches), which is why the *corroborated*
    MTCNN rate, not either raw rate, is this experiment's "unrecognisable"
    anchor.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_1_manifest`.

    Returns:
        Dict with three DataFrames, all one row per `(sigma, class_label)`:
        `"mtcnn_raw"` (`detection_rate`), `"mtcnn_corroborated"`
        (`n_images`, `n_no_sigma0_detection`, `n_eligible`,
        `corroborated_rate`), `"haar_raw"` (`detection_rate`).
    """
    mtcnn_raw_detections = raw_mtcnn_detections(cfg, manifest)
    haar_raw_detections = raw_haar_detections(cfg, manifest)

    return {
        "mtcnn_raw": detection_rate(mtcnn_raw_detections, by=["sigma", "class_label"]),
        "mtcnn_corroborated": corroborated_detection_rate(
            mtcnn_raw_detections, iou_threshold=cfg.iou_threshold
        ),
        "haar_raw": detection_rate(haar_raw_detections, by=["sigma", "class_label"]),
    }


def _cache_matches(meta: dict, cfg: Config, index: pd.DataFrame, paths: List[Path], transform_name: str) -> bool:
    same_order = list(index["path"]) == [str(p) for p in paths]
    return (
        same_order
        and meta.get("backbone") == cfg.backbone
        and meta.get("transform_name") == transform_name
        and meta.get("image_size") == cfg.image_size
        and meta.get("n_images") == len(paths)
    )


def step_3_embeddings(cfg: Config, manifest: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Extract (or load cached) embeddings at every sigma in the sweep.

    sigma=0 reuses the `intact_images` cache (`cfg.intact_cache_dir`)
    instead of re-extracting, since `blurred_transform(0)` reproduces
    `intact_transform` exactly. The remaining sigmas are cached under
    `cfg.output_dir` as `embeddings_sigma{sigma}.npy`.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_1_manifest`, in the row order embeddings
            must match.

    Returns:
        Dict mapping each sigma in `cfg.sigmas` to its
        `(len(manifest), n_features)` embedding array.
    """
    paths = [Path(p) for p in manifest["path"]]
    embeddings_by_sigma: Dict[int, np.ndarray] = {}

    for sigma in cfg.sigmas:
        if sigma == 0 and (cfg.intact_cache_dir / "embeddings.npy").exists():
            cached_array, cached_index, meta = load_embeddings(cfg.intact_cache_dir)
            if _cache_matches(meta, cfg, cached_index, paths, "intact_transform"):
                logger.info(
                    "step_3_embeddings: sigma=0 reusing intact_images cache, shape=%s",
                    cached_array.shape,
                )
                embeddings_by_sigma[sigma] = cached_array
                continue
            logger.warning(
                "step_3_embeddings: sigma=0 intact cache present but mismatched -- re-extracting"
            )

        name = f"embeddings_sigma{sigma}"
        transform_name = f"blurred_transform_sigma{sigma}"
        array_path = cfg.output_dir / f"{name}.npy"
        if array_path.exists():
            cached_array, cached_index, meta = load_embeddings(cfg.output_dir, name=name)
            if _cache_matches(meta, cfg, cached_index, paths, transform_name):
                logger.info(
                    "step_3_embeddings: sigma=%s using cached embeddings, shape=%s",
                    sigma,
                    cached_array.shape,
                )
                embeddings_by_sigma[sigma] = cached_array
                continue
            logger.info("step_3_embeddings: sigma=%s cache present but stale, re-extracting", sigma)

        transform = blurred_transform(sigma=sigma, size=cfg.image_size)
        start = time.perf_counter()
        embeddings = extract_embeddings(
            paths,
            transform,
            backbone=cfg.backbone,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )
        elapsed = time.perf_counter() - start

        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else cfg.device
        )
        meta = {
            "backbone": cfg.backbone,
            "weights": get_backbone_weights_name(cfg.backbone),
            "transform_name": transform_name,
            "sigma": sigma,
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
            "step_3_embeddings: sigma=%s extracted shape=%s in %.1fs (cache written)",
            sigma,
            embeddings.shape,
            elapsed,
        )
        embeddings_by_sigma[sigma] = embeddings

    return embeddings_by_sigma


def step_4_probe(cfg: Config, embeddings: Dict[int, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate the reference-arm probe at every sigma.

    Same model (L2 logistic regression behind a `StandardScaler`), same
    5x10 repeated stratified CV, same seed as `intact_images` -- only the
    embeddings (and therefore the transform behind them) differ by sigma.

    Args:
        cfg: Experiment configuration.
        embeddings: Output of `step_3_embeddings`.
        manifest: Output of `step_1_manifest`.

    Returns:
        One row per `(sigma, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for sigma in cfg.sigmas:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
            ]
        )
        fold_results = repeated_stratified_cv(
            embeddings[sigma],
            y,
            model,
            n_splits=cfg.n_splits,
            n_repeats=cfg.n_repeats,
            seed=cfg.seed,
        )
        fold_results.insert(0, "sigma", sigma)
        rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_5_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold results into mean/sd per metric and the paired AUC delta.

    Because the seed, fold count and manifest order are unchanged across
    sigmas, fold k of repeat r contains the same held-out images at every
    sigma -- so the AUC delta against sigma=0 is computed as a paired
    per-fold difference, not just a difference of independent means.

    Args:
        fold_results: Output of `step_4_probe`.

    Returns:
        One row per sigma: mean and sd of each metric
        (`{metric}_mean`, `{metric}_sd`), plus `auc_delta_vs_sigma0_mean`
        and `auc_delta_vs_sigma0_sd` (both exactly 0 at sigma=0).
    """
    metric_cols = [c for c in fold_results.columns if c not in ("sigma", "repeat", "fold")]
    baseline = fold_results.loc[
        fold_results["sigma"] == fold_results["sigma"].min(), ["repeat", "fold", "roc_auc"]
    ].rename(columns={"roc_auc": "roc_auc_baseline"})

    summary_rows = []
    for sigma, group in fold_results.groupby("sigma"):
        row = {"sigma": sigma}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std()

        paired = group.merge(baseline, on=["repeat", "fold"])
        delta = paired["roc_auc"] - paired["roc_auc_baseline"]
        row["auc_delta_vs_sigma0_mean"] = delta.mean()
        row["auc_delta_vs_sigma0_sd"] = delta.std()
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("sigma").reset_index(drop=True)


def _colour_features_cache_matches(cached: pd.DataFrame, manifest: pd.DataFrame) -> bool:
    return len(cached) == len(manifest) and list(cached["path"]) == list(manifest["path"])


def step_6_colour_features(cfg: Config, manifest: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Extract (or load cached) six colour statistics at every sigma.

    Computed on the same post-geometry, post-blur image the embeddings and
    face detector see -- see `src.data.colour_features.colour_features`.
    Cached per sigma to `outputs/blurred_images/colour_features_sigma{sigma}.csv`.

    Args:
        cfg: Experiment configuration.
        manifest: Output of `step_1_manifest`, in the row order the
            returned features must match.

    Returns:
        Dict mapping each sigma in `cfg.sigmas` to a DataFrame with `path`
        plus one column per name in
        `src.data.colour_features.COLOUR_FEATURE_NAMES`.
    """
    features_by_sigma: Dict[int, pd.DataFrame] = {}

    for sigma in cfg.sigmas:
        name = f"colour_features_sigma{sigma}"
        cache_path = cfg.output_dir / f"{name}.csv"
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            if _colour_features_cache_matches(cached, manifest):
                logger.info("step_6_colour_features: sigma=%s using cached features", sigma)
                features_by_sigma[sigma] = cached
                continue
            logger.info("step_6_colour_features: sigma=%s cache present but stale, recomputing", sigma)

        rows = [
            colour_features(blur_stage(geometry_stage(Image.open(p), cfg.image_size), sigma))
            for p in manifest["path"]
        ]
        df = pd.DataFrame(rows, columns=list(COLOUR_FEATURE_NAMES))
        df.insert(0, "path", manifest["path"].to_numpy())
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("step_6_colour_features: sigma=%s computed and cached %d rows", sigma, len(df))
        features_by_sigma[sigma] = df

    return features_by_sigma


def step_7_colour_probe(
    cfg: Config, features_by_sigma: Dict[int, pd.DataFrame], manifest: pd.DataFrame
) -> pd.DataFrame:
    """Cross-validate the colour-only reference probe at every sigma.

    Same model shape (L2 logistic regression behind a `StandardScaler`),
    same 5x10 repeated stratified CV, same seed as the embedding probe in
    `step_4_probe` -- only the six colour features replace the 2048-d
    embedding as input. This is the null for "no facial signal remains":
    blur cannot remove colour, so any embedding-based curve should not fall
    below what colour alone achieves.

    Args:
        cfg: Experiment configuration.
        features_by_sigma: Output of `step_6_colour_features`.
        manifest: Output of `step_1_manifest`.

    Returns:
        One row per `(sigma, repeat, fold)`, with every metric from
        `src.evaluation.metrics.fold_metrics` -- same shape as
        `step_4_probe`'s output, so it can be summarised with
        `step_5_summary` directly.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()

    rows = []
    for sigma in cfg.sigmas:
        X = features_by_sigma[sigma][list(COLOUR_FEATURE_NAMES)].to_numpy()
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
            ]
        )
        fold_results = repeated_stratified_cv(
            X, y, model, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
        )
        fold_results.insert(0, "sigma", sigma)
        rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_8_stopping_criterion(
    cfg: Config, summary: pd.DataFrame, mtcnn_corroborated: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate the pre-registered stopping criterion at every sigma.

    The sweep is declared to have plateaued at the first sigma where
    *both* hold: corroborated detection rate < `cfg.detection_threshold`,
    and the per-step AUC-mean change has absolute value <
    `cfg.auc_step_threshold`. Both conditions are necessary -- the first
    run's own per-step AUC deltas (-0.007, -0.040, -0.009, -0.046) show
    that landing inside a plausible-looking band is not the same as the
    curve having flattened.

    Args:
        cfg: Experiment configuration.
        summary: Output of `step_5_summary` (the embedding probe).
        mtcnn_corroborated: The `"mtcnn_corroborated"` table from
            `step_2_detection`.

    Returns:
        One row per sigma, sorted ascending: `sigma`, `roc_auc_mean`,
        `auc_step_delta` (NaN at the first row), `corroborated_rate_overall`
        (mean across classes, matching the convention used for the raw
        detection-rate plot), `detection_below_threshold`,
        `auc_step_flat`, `plateau_reached`.
    """
    overall_corrob = (
        mtcnn_corroborated.groupby("sigma")["corroborated_rate"]
        .mean()
        .rename("corroborated_rate_overall")
        .reset_index()
    )

    table = (
        summary[["sigma", "roc_auc_mean"]]
        .merge(overall_corrob, on="sigma", how="left")
        .sort_values("sigma")
        .reset_index(drop=True)
    )
    table["auc_step_delta"] = table["roc_auc_mean"].diff()
    table["detection_below_threshold"] = table["corroborated_rate_overall"] < cfg.detection_threshold
    table["auc_step_flat"] = table["auc_step_delta"].abs() < cfg.auc_step_threshold
    table["plateau_reached"] = table["detection_below_threshold"] & table["auc_step_flat"]
    return table
