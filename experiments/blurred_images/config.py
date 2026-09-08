"""Configuration for the blurred_images experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _project_root() -> Path:
    """Locate the repository root by walking up from this file.

    Returns:
        Absolute path to the project root.

    Raises:
        RuntimeError: If no ancestor directory contains ``.git``.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate project root (no ancestor has .git).")


@dataclass(frozen=True)
class Config:
    """Paths, seed, backbone, blur sweep and cross-validation settings.

    The cross-validation protocol (model, folds, seed) is identical to
    `intact_images.Config` on purpose -- only the transform changes.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: Where this experiment writes its caches and results.
        intact_cache_dir: Where `intact_images` cached its embeddings;
            reused here for sigma=0 instead of re-extracting.
        seed: Base random seed for splitting and the classifier.
        backbone: Frozen-backbone key.
        image_size: Square side length used by the shared transform.
        batch_size: Batch size for embedding extraction.
        device: Torch device string for embedding extraction and face
            detection.
        n_splits: Folds per cross-validation repeat.
        n_repeats: Number of cross-validation repeats.
        sigmas: The blur-strength sweep, in output-pixel units (applied
            after resizing to `image_size`). Extended from the first run's
            `(0, 2, 4, 8, 16)` to also probe past the point where that run
            stopped (detection was still ~51%, AUC still falling).
        legacy_haar_sigmas: The sigma set the first run's Haar-cascade
            detection was computed over. Kept fixed (not extended to the
            new sigmas) since the Haar numbers are retained only as a
            side-by-side comparison against MTCNN, not recomputed.
        detection_threshold: Corroborated detection rate below which a
            sigma is considered to have destroyed recognisability. Applies
            to the IoU-corroborated rate, not the raw rate -- see
            `src.models.face_detect.corroborated_detection_rate`.
        auc_step_threshold: Per-step AUC-mean change (in absolute value)
            below which the curve is considered to have flattened.
        iou_threshold: Minimum IoU for a blurred-sigma detection to count
            as corroborated by that image's sigma=0 detection.
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "blurred_images"
    )
    intact_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "intact_images"
    )
    seed: int = 42
    backbone: str = "resnet50"
    image_size: int = 224
    batch_size: int = 64
    device: str = "cuda"
    n_splits: int = 5
    n_repeats: int = 10
    sigmas: Tuple[int, ...] = (0, 2, 4, 8, 16, 32, 48, 64)
    legacy_haar_sigmas: Tuple[int, ...] = (0, 2, 4, 8, 16)
    detection_threshold: float = 0.05
    auc_step_threshold: float = 0.01
    iou_threshold: float = 0.5
