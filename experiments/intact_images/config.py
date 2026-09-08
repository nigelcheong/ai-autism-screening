"""Configuration for the intact_images reference-arm experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    """Paths, seed, backbone and cross-validation settings.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: Where this experiment writes its cache and results.
        seed: Base random seed for splitting and the classifier.
        backbone: Frozen-backbone key passed to
            :func:`src.models.embeddings.extract_embeddings`.
        image_size: Square side length used by the shared transform.
        batch_size: Batch size for embedding extraction.
        device: Torch device string for embedding extraction only.
        n_splits: Folds per cross-validation repeat.
        n_repeats: Number of cross-validation repeats.
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "intact_images"
    )
    seed: int = 42
    backbone: str = "resnet50"
    image_size: int = 224
    batch_size: int = 64
    device: str = "cuda"
    n_splits: int = 5
    n_repeats: int = 10
