"""Configuration for the background_images experiment."""

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
    """Paths, seed, backbone, mask arms and cross-validation settings.

    The cross-validation protocol (model, folds, seed) is identical to
    `intact_images.Config` and `blurred_images.Config` on purpose -- only
    the transform changes, per arm.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: Where this experiment writes its caches and results.
        intact_cache_dir: Where `intact_images` cached its embeddings;
            the `"intact"` arm is a row-subset of this cache, not a
            re-extraction.
        blurred_cache_dir: Where `blurred_images` cached its sigma=0 MTCNN
            detections (`detections_mtcnn.csv`); reused here as the source
            of face boxes -- this experiment never runs the detector.
        seed: Base random seed for splitting and the classifier.
        backbone: Frozen-backbone key.
        image_size: Square side length used by the shared transform.
        batch_size: Batch size for embedding extraction.
        device: Torch device string for embedding extraction. Face
            detection is not run here (boxes are reused from the cache),
            so this config has no detector device setting.
        n_splits: Folds per cross-validation repeat.
        n_repeats: Number of cross-validation repeats.
        expand_tight: The "as detected" box expansion factor -- shows how
            much the answer depends on the box being a tight face-only
            crop that excludes forehead, hair and ears.
        expand_wide: The box expansion factor used everywhere else --
            the honest version of the test, since a tight MTCNN box is not
            actually "just the face."
        arm_names: The five arms compared, in the order results are
            reported: an unmasked reference (row-subset of the
            `intact_images` cache) plus the four masked/cropped transforms.
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "background_images"
    )
    intact_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "intact_images"
    )
    blurred_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "blurred_images"
    )
    seed: int = 42
    backbone: str = "resnet50"
    image_size: int = 224
    batch_size: int = 64
    device: str = "cuda"
    n_splits: int = 5
    n_repeats: int = 10
    expand_tight: float = 1.0
    expand_wide: float = 1.5
    arm_names: Tuple[str, ...] = (
        "intact",
        "face_masked_1.0",
        "face_masked_1.5",
        "background_masked_1.5",
        "face_crop_1.5",
    )
