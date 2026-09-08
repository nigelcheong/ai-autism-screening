"""Pipeline steps for the intact_images reference-arm experiment.

Each function does one step and returns something the notebook can display
directly. This is the reference arm: frozen backbone features plus a linear
probe, no fine-tuning. The `blurred_images` and `background_images`
experiments reuse every function here unchanged except for the transform
passed into embedding extraction.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.intact_images.config import Config
from src.data.images import build_image_manifest
from src.data.transforms import intact_transform
from src.evaluation.cross_val import repeated_stratified_cv
from src.models.embeddings import (
    extract_embeddings,
    get_backbone_weights_name,
    load_embeddings,
    save_embeddings,
)

logger = logging.getLogger(__name__)

_TRANSFORM_NAME = "intact_transform"


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the image manifest for the intact archive.

    Args:
        cfg: Experiment configuration.

    Returns:
        One row per image: ``path``, ``filename``, ``split``, ``class_label``.
    """
    return build_image_manifest(cfg.data_root)


def _cache_matches(meta: dict, cfg: Config, index: pd.DataFrame, paths: list) -> bool:
    """Whether a cached embedding run's metadata matches the current config."""
    same_order = list(index["path"]) == [str(p) for p in paths]
    return (
        same_order
        and meta.get("backbone") == cfg.backbone
        and meta.get("transform_name") == _TRANSFORM_NAME
        and meta.get("image_size") == cfg.image_size
        and meta.get("n_images") == len(paths)
    )


def step_2_embeddings(cfg: Config, manifest: pd.DataFrame) -> np.ndarray:
    """Extract (or load cached) frozen-backbone embeddings for the manifest.

    Skips re-extraction when a cache exists under ``cfg.output_dir`` and its
    metadata (backbone, transform, image size, image count, row order)
    matches the current config. Logs shape, cache status and wall-clock at
    INFO level.

    Args:
        cfg: Experiment configuration.
        manifest: Output of :func:`step_1_manifest`, in the row order
            embeddings must match.

    Returns:
        Float32 array of shape ``(len(manifest), n_features)``.
    """
    paths = [Path(p) for p in manifest["path"]]

    if (cfg.output_dir / "embeddings.npy").exists():
        cached_array, cached_index, meta = load_embeddings(cfg.output_dir)
        if _cache_matches(meta, cfg, cached_index, paths):
            logger.info(
                "step_2_embeddings: using cached embeddings, shape=%s", cached_array.shape
            )
            return cached_array
        logger.info("step_2_embeddings: cache present but stale, re-extracting")

    transform = intact_transform(size=cfg.image_size)
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
        "transform_name": _TRANSFORM_NAME,
        "image_size": cfg.image_size,
        "n_images": len(paths),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": device_name,
        "wall_clock_seconds": elapsed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_embeddings(embeddings, paths, meta, cfg.output_dir)
    logger.info(
        "step_2_embeddings: extracted shape=%s in %.1fs (cache written)",
        embeddings.shape,
        elapsed,
    )
    return embeddings


def step_3_probe(cfg: Config, embeddings: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate an L2-logistic-regression probe on the embeddings.

    Pools all three published splits and cross-validates over the pool --
    frozen features plus a linear model, no fine-tuning, no threshold
    tuning.

    Args:
        cfg: Experiment configuration.
        embeddings: Output of :func:`step_2_embeddings`, row-aligned with
            ``manifest``.
        manifest: Output of :func:`step_1_manifest`.

    Returns:
        One row per fold: ``repeat``, ``fold``, and every metric from
        :func:`src.evaluation.metrics.fold_metrics`.
    """
    y = (manifest["class_label"] == "autistic").astype(int).to_numpy()
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
        ]
    )
    return repeated_stratified_cv(
        embeddings,
        y,
        model,
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        seed=cfg.seed,
    )


def step_4_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold results into mean and standard deviation per metric.

    Args:
        fold_results: Output of :func:`step_3_probe`.

    Returns:
        DataFrame with one row per metric and columns ``metric``, ``mean``,
        ``sd``.
    """
    metric_cols = [c for c in fold_results.columns if c not in ("repeat", "fold")]
    summary = fold_results[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "sd"]
    summary.index.name = "metric"
    return summary.reset_index()
