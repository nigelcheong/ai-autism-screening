"""Part 3 / Part 4: the fine-tuned CNN and quantified Grad-CAM.

The arm that answers "you only used a linear probe on frozen features."
DenseNet201, ImageNet-initialised, fine-tuned end to end -- every
parameter trainable, not just a new head. Run under two protocols so the
gap between them is itself a measured result (see `run_published_protocol`
/ `run_corrected_protocol`).

Four specific errors are named in the accompanying dataset's published
code; each is avoided at a specific, cited point below:

1. Preprocessing: `src.data.transforms.to_tensor_stage` normalises with
   the exact ImageNet mean/std `DenseNet201_Weights.IMAGENET1K_V1`
   declares (verified to match, not assumed) -- not a bare `1/255`
   rescale and not raw 0-255 input.
2. Augmentation: applied to the PIL image, before `to_tensor_stage`, in
   `_ImageLabelDataset.__getitem__` -- never to a feature map the
   backbone has already produced.
3. Epoch selection: `train_model` always selects the best epoch by a
   validation split, never by the set a result is reported on. The
   *published* protocol reproduces the literature's flaw on purpose (its
   validation split and its reported split are the same `valid/` folder,
   by construction of the protocol itself, not by this code cutting a
   corner) -- the *corrected* protocol's validation split is carved out
   of the training fold and never touches the held-out test fold.
4. Thresholding: `evaluate` calls `src.evaluation.metrics.fold_metrics`
   on `sigmoid(logits)` directly -- a single-output probability
   thresholded at 0.5, never `argmax` over a size-1 last axis (which
   would return 0 for every row).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from experiments.why_not_faces.config import Config
from src.data.transforms import geometry_stage, to_tensor_stage
from src.evaluation.metrics import fold_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "TrainResult",
    "build_model",
    "train_model",
    "evaluate",
    "run_published_protocol",
    "load_corrected_protocol_manifest",
    "run_corrected_protocol",
    "GradCAM",
    "gradcam_off_face_fraction",
    "run_gradcam_analysis",
]


def _verify_preprocessing_matches_backbone() -> None:
    """Assert `to_tensor_stage`'s normalisation matches DenseNet201's own declared preprocessing.

    A runtime check, not a comment -- if torchvision ever changes the
    weights' declared statistics, this fails loudly instead of silently
    reproducing error #1.
    """
    from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    weights_transform = models.DenseNet201_Weights.IMAGENET1K_V1.transforms()
    assert tuple(weights_transform.mean) == IMAGENET_MEAN, "mean mismatch: to_tensor_stage != DenseNet201's own preprocessing"
    assert tuple(weights_transform.std) == IMAGENET_STD, "std mismatch: to_tensor_stage != DenseNet201's own preprocessing"


class _ImageLabelDataset(Dataset):
    """Opens each path, applies geometry (+ optional augmentation), normalises.

    Augmentation, when enabled, is applied to the post-geometry PIL image
    -- input-image space -- strictly before `to_tensor_stage`. It is
    never applied to anything the backbone has already computed.
    """

    def __init__(self, paths: Sequence[Path], labels: Sequence[int], augment: bool, image_size: int) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.image_size = image_size
        self._augment = (
            T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = geometry_stage(Image.open(self.paths[idx]), self.image_size)
        if self._augment is not None:
            image = self._augment(image)
        tensor = to_tensor_stage(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


def build_model(cfg: Config) -> nn.Module:
    """DenseNet201, ImageNet weights, classifier replaced with a single sigmoid-ready logit.

    Args:
        cfg: Experiment configuration (`cnn_backbone` must be `"densenet201"`).

    Returns:
        An `nn.Module` with every parameter `requires_grad=True` --
        "fine-tuned", not a frozen-backbone linear probe like
        `intact_images`.

    Raises:
        ValueError: If `cfg.cnn_backbone` is not `"densenet201"`.
    """
    if cfg.cnn_backbone != "densenet201":
        raise ValueError(f"Only 'densenet201' is implemented, got {cfg.cnn_backbone!r}")

    _verify_preprocessing_matches_backbone()
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


@dataclass
class TrainResult:
    """Output of `train_model`.

    Attributes:
        model: The model, with the best-validation-AUC epoch's weights
            loaded (not necessarily the last epoch trained).
        history: One row per epoch trained: `epoch`, `train_loss`,
            `val_loss`, `val_auc`.
        best_epoch: Index (0-based) of the epoch `model`'s weights come from.
        best_val_auc: That epoch's validation ROC-AUC.
    """

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_auc: float


def train_model(
    cfg: Config,
    train_paths: Sequence[Path],
    train_labels: Sequence[int],
    val_paths: Sequence[Path],
    val_labels: Sequence[int],
    run_seed: int,
) -> TrainResult:
    """Fine-tune DenseNet201, selecting the epoch by `val_paths`'s ROC-AUC.

    Args:
        cfg: Experiment configuration.
        train_paths: Training image paths.
        train_labels: `1`=autistic, `0`=non_autistic, aligned to `train_paths`.
        val_paths: Validation image paths used for epoch selection --
            *only* for epoch selection; the caller decides separately
            what to report on (see module docstring, error #3).
        val_labels: Aligned to `val_paths`.
        run_seed: Seed for this specific run's weight init / data order.
            Exact bit-for-bit reproducibility is not guaranteed on GPU
            even with a fixed seed (non-deterministic cuDNN kernels can
            still be selected for some ops) -- fixed here so any
            variation across runs is only ever that residual, not
            uncontrolled randomness.

    Returns:
        See `TrainResult`.
    """
    torch.manual_seed(run_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.cnn_device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cnn_lr, weight_decay=cfg.cnn_weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        _ImageLabelDataset(train_paths, train_labels, augment=True, image_size=cfg.cnn_image_size),
        batch_size=cfg.cnn_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(run_seed),
    )
    val_loader = DataLoader(
        _ImageLabelDataset(val_paths, val_labels, augment=False, image_size=cfg.cnn_image_size),
        batch_size=cfg.cnn_batch_size,
        shuffle=False,
        num_workers=0,
    )

    best_state, best_val_auc, best_epoch, epochs_no_improve = None, -np.inf, -1, 0
    history_rows = []

    for epoch in range(cfg.cnn_max_epochs):
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(y)
            n_train += len(y)
        train_loss = train_loss_sum / n_train

        model.eval()
        val_loss_sum, val_probs, val_true = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X).squeeze(1)
                val_loss_sum += criterion(logits, y).item() * len(y)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_true.append(y.cpu().numpy())
        val_loss = val_loss_sum / len(val_paths)
        val_probs, val_true = np.concatenate(val_probs), np.concatenate(val_true)
        val_auc = fold_metrics(val_true, val_probs)["roc_auc"]

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
        logger.info(
            "train_model: epoch=%d train_loss=%.4f val_loss=%.4f val_auc=%.4f", epoch, train_loss, val_loss, val_auc
        )

        if val_auc > best_val_auc:
            best_val_auc, best_epoch, epochs_no_improve = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.cnn_patience:
                logger.info(
                    "train_model: early stopping at epoch=%d (best epoch=%d, best val_auc=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_auc,
                )
                break

    model.load_state_dict(best_state)
    return TrainResult(model=model, history=pd.DataFrame(history_rows), best_epoch=best_epoch, best_val_auc=best_val_auc)


def evaluate(cfg: Config, model: nn.Module, paths: Sequence[Path], labels: Sequence[int]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Report every `fold_metrics` metric for `model` on `(paths, labels)`.

    Thresholds the single-output sigmoid probability directly (error #4)
    -- never `argmax`.

    Args:
        cfg: Experiment configuration.
        model: A trained model, in eval mode by the time this returns.
        paths: Image paths to evaluate on.
        labels: Aligned to `paths`.

    Returns:
        `(metrics, probs, true)`: `metrics` is `fold_metrics`'s dict;
        `probs` and `true` are the raw per-image sigmoid probabilities
        and labels, for downstream use (e.g. plotting).
    """
    device = torch.device(cfg.cnn_device)
    loader = DataLoader(
        _ImageLabelDataset(paths, labels, augment=False, image_size=cfg.cnn_image_size),
        batch_size=cfg.cnn_batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    probs, true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            true.append(y.numpy())
    probs, true = np.concatenate(probs), np.concatenate(true)
    return fold_metrics(true, probs), probs, true


def _save_run_result(cfg: Config, name: str, result: Dict) -> None:
    """Cache one `run_published_protocol` / `run_corrected_protocol`-fold result to disk.

    Model weights (`.pt`), training history (`.csv`), predictions
    (`.npz`), and everything else JSON-serialisable (`.json`, including
    `test_paths` where present) -- so `_load_run_result` can rebuild an
    identical `dict`, and re-running this experiment's notebook does not
    re-train.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), cfg.output_dir / f"{name}_model.pt")
    result["history"].to_csv(cfg.output_dir / f"{name}_history.csv", index=False)
    np.savez(cfg.output_dir / f"{name}_predictions.npz", probs=result["probs"], true=result["true"])

    meta = {k: v for k, v in result.items() if k not in ("model", "history", "probs", "true")}
    (cfg.output_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_run_result(cfg: Config, name: str) -> Optional[Dict]:
    """Load a result `_save_run_result` cached, or `None` if no cache exists."""
    model_path = cfg.output_dir / f"{name}_model.pt"
    if not model_path.exists():
        return None

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(torch.device(cfg.cnn_device))
    model.eval()

    history = pd.read_csv(cfg.output_dir / f"{name}_history.csv")
    meta = json.loads((cfg.output_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
    predictions = np.load(cfg.output_dir / f"{name}_predictions.npz")

    logger.info("_load_run_result: using cached run %r", name)
    return {**meta, "model": model, "history": history, "probs": predictions["probs"], "true": predictions["true"]}


def run_published_protocol(cfg: Config, manifest_full: pd.DataFrame, use_cache: bool = True) -> Dict:
    """Train on `train/`, select the epoch on `valid/`, report on that same `valid/`.

    Deliberately reproduces the published code's flaw #3 (see module
    docstring) -- this is the point of the "published protocol" arm: it
    reports the number the literature's methodology would produce, for
    direct comparison against the corrected protocol.

    Args:
        cfg: Experiment configuration.
        manifest_full: Output of `build_image_manifest` -- the full,
            as-shipped `train`/`valid`/`test` split, no deduplication.

    Returns:
        Dict with `history` (`TrainResult.history`), `best_epoch`,
        `metrics` (on `valid/`), `probs`, `true`, `n_train`, `n_report`.
    """
    if use_cache:
        cached = _load_run_result(cfg, "cnn_published")
        if cached is not None:
            return cached

    train_rows = manifest_full.loc[manifest_full["split"] == "train"]
    valid_rows = manifest_full.loc[manifest_full["split"] == "valid"]

    train_paths = train_rows["path"].tolist()
    train_labels = (train_rows["class_label"] == "autistic").astype(int).tolist()
    valid_paths = valid_rows["path"].tolist()
    valid_labels = (valid_rows["class_label"] == "autistic").astype(int).tolist()

    logger.info("run_published_protocol: train=%d, valid=%d (early-stop AND report set)", len(train_paths), len(valid_paths))
    result = train_model(cfg, train_paths, train_labels, valid_paths, valid_labels, run_seed=cfg.seed)
    metrics, probs, true = evaluate(cfg, result.model, valid_paths, valid_labels)

    output = {
        "history": result.history,
        "best_epoch": result.best_epoch,
        "metrics": metrics,
        "probs": probs,
        "true": true,
        "n_train": len(train_paths),
        "n_report": len(valid_paths),
        "model": result.model,
    }
    _save_run_result(cfg, "cnn_published", output)
    return output


def load_corrected_protocol_manifest(cfg: Config, manifest_full: pd.DataFrame) -> pd.DataFrame:
    """Map the pre-built, hash-grouped, deduplicated CV split onto canonical paths.

    `data/splits/index.csv` / `cv_folds.csv` were built against an old
    base directory (`data/AutismDataset/...`) that no longer exists, so
    rows are matched back to `manifest_full`'s canonical `path` by
    `(split, class_label, filename)` -- the same technique
    `experiments.why_not_faces.pipeline._load_acquisition` uses for the
    corpus-audit manifest. The 440 `is_holdout` rows `index.csv` marks
    are excluded here (they are not part of `cv_folds.csv` to begin
    with): the corrected protocol never trains or reports on them.

    Args:
        cfg: Experiment configuration.
        manifest_full: Output of `build_image_manifest`.

    Returns:
        One row per non-holdout image, per repeat -- 2,500 unique paths
        x 10 repeats: `path` (canonical), `class_label`, `group_id`
        (perceptual-hash duplicate-pair group), `repeat`, `fold`.
    """
    index_df = pd.read_csv(cfg.splits_dir / "index.csv").rename(columns={"path": "path_old", "published_split": "split"})
    canonical = manifest_full.merge(
        index_df[["path_old", "split", "class_label", "filename", "group_id", "is_holdout"]],
        on=["split", "class_label", "filename"],
        how="inner",
        validate="one_to_one",
    )
    old_to_canonical = dict(zip(canonical["path_old"], canonical["path"]))

    folds_df = pd.read_csv(cfg.splits_dir / "cv_folds.csv").rename(columns={"path": "path_old"})
    folds_df["path"] = folds_df["path_old"].map(old_to_canonical)
    if folds_df["path"].isna().any():
        raise RuntimeError("load_corrected_protocol_manifest: some cv_folds.csv rows did not map to a canonical path")

    folds_df = folds_df.merge(canonical[["path", "class_label", "group_id"]], on="path", how="left", validate="many_to_one")
    return folds_df[["path", "class_label", "group_id", "repeat", "fold"]]


def _inner_split(fold_manifest: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware, stratified inner train/validation split of one fold's training rows."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1.0 / val_fraction))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = (fold_manifest["class_label"] == "autistic").astype(int).to_numpy()
    groups = fold_manifest["group_id"].to_numpy()
    inner_train_idx, inner_val_idx = next(cv.split(fold_manifest, y, groups))
    return fold_manifest.iloc[inner_train_idx], fold_manifest.iloc[inner_val_idx]


def run_corrected_protocol(
    cfg: Config, corrected_manifest: pd.DataFrame, repeat: int = 0, use_cache: bool = True
) -> List[Dict]:
    """Fine-tune and evaluate one model per fold, on `cv_folds.csv`'s grouped, deduplicated split.

    For fold *k*: the inner train/validation split (for epoch selection)
    is carved out of the *other* folds' rows only, group-aware and
    stratified; fold *k*'s own rows are never touched until the single
    final evaluation call. `cfg.cnn_n_folds` folds are run (a single
    repeat), not the full 50 every non-CNN probe uses -- fine-tuning a
    CNN is orders of magnitude more expensive than fitting a probe on
    cached features, and the constraints explicitly scope "same folds"
    to non-CNN arms only.

    Args:
        cfg: Experiment configuration.
        corrected_manifest: Output of `load_corrected_protocol_manifest`.
        repeat: Which of `cv_folds.csv`'s 10 repeats to use.

    Returns:
        One dict per fold (see `run_published_protocol`'s return shape,
        plus `fold`, `n_inner_train`, `n_inner_val`, `n_test`).
    """
    repeat_df = corrected_manifest.loc[corrected_manifest["repeat"] == repeat]
    results = []

    for fold in range(cfg.cnn_n_folds):
        name = f"cnn_corrected_fold{fold}"
        if use_cache:
            cached = _load_run_result(cfg, name)
            if cached is not None:
                results.append(cached)
                continue

        test_rows = repeat_df.loc[repeat_df["fold"] == fold]
        train_rows = repeat_df.loc[repeat_df["fold"] != fold]

        inner_train, inner_val = _inner_split(train_rows, cfg.cnn_inner_val_fraction, seed=cfg.seed + fold)

        train_paths = inner_train["path"].tolist()
        train_labels = (inner_train["class_label"] == "autistic").astype(int).tolist()
        val_paths = inner_val["path"].tolist()
        val_labels = (inner_val["class_label"] == "autistic").astype(int).tolist()
        test_paths = test_rows["path"].tolist()
        test_labels = (test_rows["class_label"] == "autistic").astype(int).tolist()

        logger.info(
            "run_corrected_protocol: fold=%d inner_train=%d inner_val=%d test=%d (held out)",
            fold,
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )
        result = train_model(cfg, train_paths, train_labels, val_paths, val_labels, run_seed=cfg.seed + fold)
        metrics, probs, true = evaluate(cfg, result.model, test_paths, test_labels)

        fold_output = {
            "fold": fold,
            "history": result.history,
            "best_epoch": result.best_epoch,
            "metrics": metrics,
            "probs": probs,
            "true": true,
            "test_paths": test_paths,
            "n_inner_train": len(train_paths),
            "n_inner_val": len(val_paths),
            "n_test": len(test_paths),
            "model": result.model,
        }
        _save_run_result(cfg, name, fold_output)
        results.append(fold_output)

    return results


class GradCAM:
    """Classic Grad-CAM (Selvaraju et al. 2017), for a single sigmoid output.

    Hooks `target_layer`'s forward activations and backward gradients;
    `__call__` runs one forward + backward pass on a single image and
    returns the (ReLU'd, feature-map-resolution) class activation map.

    `__call__`'s `target_sign` matters for a single-logit sigmoid model
    in a way it would not for a multi-class softmax model: backpropagating
    the raw logit always asks "what pushed this *toward* autistic?", which
    is not a coherent question to ask of an image the model confidently
    calls non_autistic -- there, the weighted feature combination is
    predominantly negative everywhere, and Grad-CAM's final ReLU zeroes
    the entire map (an early version of this analysis found this for 44%
    of non_autistic images and 2% of autistic ones -- not noise, the
    textbook single-logit Grad-CAM failure mode). Backpropagating
    `-logit` on a confidently-negative prediction instead asks "what
    pushed this *toward* non_autistic?", the coherent question for that
    image. `run_gradcam_analysis` sets `target_sign` from the model's own
    prediction for exactly this reason.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output) -> None:
        # A tensor-level `register_hook` here (not a module-level
        # `register_full_backward_hook`) is deliberate: DenseNet's own
        # `forward` applies an in-place ReLU to this exact tensor
        # immediately afterward, and a full-module backward hook's view
        # tracking is incompatible with that in-place op (PyTorch raises
        # "a view ... is being modified inplace"). A plain tensor hook
        # has no such conflict.
        self.activations = output
        if output.requires_grad:
            # `run_gradcam_analysis` also runs a plain `torch.no_grad()`
            # forward pass to read the model's own prediction (to pick
            # `target_sign`) -- this same hook fires then too, and that
            # pass's output has no grad to hook.
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def __call__(self, x: torch.Tensor, target_sign: float = 1.0) -> np.ndarray:
        """Args:
            x: `(1, 3, H, W)` input tensor, already on the model's device.
            target_sign: `+1.0` explains "evidence for autistic" (the
                logit as-is); `-1.0` explains "evidence for non_autistic"
                (backprops `-logit`). See the class docstring.

        Returns:
            `(h, w)` float32 array (feature-map resolution, e.g. 7x7 for
            DenseNet201 at 224x224 input), normalised to `[0, 1]`.
        """
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze()
        (target_sign * logit).backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.detach().squeeze().cpu().numpy().astype(np.float32)
        peak = cam.max()
        return cam / peak if peak > 0 else cam


def gradcam_off_face_fraction(cam: np.ndarray, box, image_size: int) -> float:
    """Fraction of a Grad-CAM map's activation mass falling outside `box`.

    `cam` is upsampled (nearest-neighbour, via `numpy.kron`-free repeat)
    to `(image_size, image_size)` before masking, so a coarse
    feature-resolution map is compared against a pixel-resolution box on
    equal footing.

    Args:
        cam: `(h, w)` non-negative activation map, e.g. from `GradCAM.__call__`.
        box: `(x1, y1, x2, y2)` face box, in the same `image_size` pixel
            coordinates `src.data.transforms.expand_box` uses.
        image_size: Target square side length (224 elsewhere in this project).

    Returns:
        `outside_mass / total_mass`, or `NaN` if the map is all zero.
    """
    h, w = cam.shape
    scale_y, scale_x = image_size // h if h else 1, image_size // w if w else 1
    upsampled = np.repeat(np.repeat(cam, scale_y, axis=0), scale_x, axis=1)
    upsampled = upsampled[:image_size, :image_size]
    if upsampled.shape != (image_size, image_size):
        padded = np.zeros((image_size, image_size), dtype=upsampled.dtype)
        padded[: upsampled.shape[0], : upsampled.shape[1]] = upsampled
        upsampled = padded

    x1, y1, x2, y2 = box
    ys = np.arange(image_size).reshape(-1, 1) + 0.5
    xs = np.arange(image_size).reshape(1, -1) + 0.5
    inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)

    total = float(upsampled.sum())
    if total <= 0:
        return float("nan")
    return float(upsampled[~inside].sum() / total)


def run_gradcam_analysis(
    cfg: Config, fold_results: List[Dict], box_by_path: Dict[str, tuple]
) -> pd.DataFrame:
    """Out-of-fold Grad-CAM off-face-mass, for every corrected-protocol fold's held-out images.

    Each fold's own trained model explains only that fold's own held-out
    test images -- the same out-of-fold discipline `step_5_shap_out_of_fold`
    applies to SHAP. Boxes are the cached sigma=0 MTCNN detections, never
    re-detected.

    Args:
        cfg: Experiment configuration.
        fold_results: Output of `run_corrected_protocol`.
        box_by_path: `path -> (x1, y1, x2, y2)`, e.g. from
            `background_images.pipeline.step_2_boxes`.

    Returns:
        One row per image with a box: `path`, `class_label`, `fold`,
        `predicted_class`, `off_face_fraction`. `target_sign` (see
        `GradCAM.__call__`) is set from the model's own prediction for
        that image -- `+1` (explain "evidence for autistic") when the
        model predicts autistic, `-1` (explain "evidence for
        non_autistic") otherwise -- so the map always answers the
        question that is coherent for what the model actually predicted,
        not always "why autistic" regardless of the prediction.
    """
    device = torch.device(cfg.cnn_device)
    rows = []

    for fold_result in fold_results:
        model = fold_result["model"].to(device)
        target_layer = dict(model.named_modules())[cfg.gradcam_target_layer]
        cam_fn = GradCAM(model, target_layer)

        for path, label in zip(fold_result["test_paths"], fold_result["true"]):
            box = box_by_path.get(path)
            if box is None:
                continue
            image = geometry_stage(Image.open(path), cfg.cnn_image_size)
            tensor = to_tensor_stage(image).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_prob = torch.sigmoid(model(tensor).squeeze()).item()
            predicted_autistic = predicted_prob >= 0.5
            target_sign = 1.0 if predicted_autistic else -1.0

            cam = cam_fn(tensor, target_sign=target_sign)
            off_face = gradcam_off_face_fraction(cam, box, cfg.cnn_image_size)
            rows.append(
                {
                    "path": path,
                    "class_label": "autistic" if label else "non_autistic",
                    "fold": fold_result["fold"],
                    "predicted_class": "autistic" if predicted_autistic else "non_autistic",
                    "off_face_fraction": off_face,
                }
            )

    result = pd.DataFrame(rows)
    n_nan = int(result["off_face_fraction"].isna().sum())
    logger.info(
        "run_gradcam_analysis: %d images, %d with an all-zero CAM (%.1f%%), mean off_face_fraction=%.3f (over the rest)",
        len(result),
        n_nan,
        100 * n_nan / len(result),
        result["off_face_fraction"].mean(),
    )
    return result
