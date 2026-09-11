"""Configuration for the face_crop_blur experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Tuple


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
    """Paths, seed, backbone, crop/blur sweep and cross-validation settings.

    The cross-validation protocol (model, folds, seed) is identical to
    `intact_images.Config`, `blurred_images.Config` and
    `background_images.Config` on purpose -- only the transform changes,
    per arm and sigma.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: Where this experiment writes its caches and results.
        intact_cache_dir: Where `intact_images` cached its embeddings and
            colour-free reference; used as the sigma=0 source for the
            `"whole_image"` arm (`blurred_images` never wrote its own
            sigma=0 embedding cache, since `blurred_transform(0)`
            reproduces `intact_transform` exactly).
        blurred_cache_dir: Where `blurred_images` cached its sigma>0
            embeddings and its per-sigma colour features; the source of
            the `"whole_image"` arm at sigma>0, and of the sigma=0..32
            colour-only floor reused for the whole-image comparison.
        background_cache_dir: Where `background_images` cached its sigma=0
            MTCNN detections are reused *indirectly*, via
            `experiments.background_images.pipeline.step_2_boxes` /
            `step_3_exclude_boxless` (which read `blurred_cache_dir`
            themselves) -- kept here only for reference/documentation, not
            read directly by this config.
        seed: Base random seed for splitting and the classifier.
        backbone: Frozen-backbone key.
        image_size: Square side length used by the shared transform.
        small_size: The common bottleneck side length for the
            resolution-equalised arm (`crop_1.0_equalised`): every crop is
            downsampled to this size, then back up to `image_size`, before
            blur is applied.
        batch_size: Batch size for embedding extraction.
        device: Torch device string for embedding extraction.
        n_splits: Folds per cross-validation repeat.
        n_repeats: Number of cross-validation repeats.
        sigmas: The blur-strength sweep, in output-pixel units (applied
            after crop-and-resize to `image_size`). A subset of
            `blurred_images.Config.sigmas` so the `"whole_image"` arm is a
            pure cache read, never a re-extraction.
        coverage_factors: The `expand_box` factors probed in the Part 0
            diagnostic (not the crop arms below) -- reproduces
            `background_images`'s `expand_tight=1.0` / `expand_wide=1.5`
            plus the intermediate 0.8 used by this experiment's own tight
            crop.
        min_coverage_fraction: Frame-coverage threshold ("covers most of
            the frame") used to report the proportion of images whose
            expanded box exceeds it, per `coverage_factors` entry.
        crop_factors: The `expand_box` factor behind each crop arm.
            `"crop_1.0"` is the unexpanded MTCNN box; `"crop_0.8"` shrinks
            it to 80% of its linear size, centred, excluding forehead,
            hair and ears; `"crop_1.0_equalised"` uses the same box as
            `"crop_1.0"` and differs only in the extra resample stage (see
            `equalised_arms`).
        equalised_arms: Arms that get the extra downsample-then-upsample
            resolution-equalisation stage between crop-and-resize and
            blur.
        min_box_size_px: Minimum box side length, in *original-image*
            pixel coordinates, required at the tighter crop factor
            (`crop_factors["crop_0.8"]`) for an image to stay eligible.
            Applied once and shared by every arm, so no comparison in this
            experiment ever mixes rows across different exclusion sets.
        arm_names: The four arms compared, in the order results are
            reported: the honest crop, its tighter inner-face variant, the
            resolution-equalised control, and the whole-image reference
            (a row-subset of the `blurred_images`/`intact_images` cache).
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "face_crop_blur"
    )
    intact_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "intact_images"
    )
    blurred_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "blurred_images"
    )
    background_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "background_images"
    )
    seed: int = 42
    backbone: str = "resnet50"
    image_size: int = 224
    small_size: int = 64
    batch_size: int = 64
    device: str = "cuda"
    n_splits: int = 5
    n_repeats: int = 10
    sigmas: Tuple[int, ...] = (0, 2, 4, 8, 16, 32)
    coverage_factors: Tuple[float, ...] = (0.8, 1.0, 1.5)
    min_coverage_fraction: float = 0.95
    crop_factors: Dict[str, float] = field(
        default_factory=lambda: {"crop_1.0": 1.0, "crop_0.8": 0.8, "crop_1.0_equalised": 1.0}
    )
    equalised_arms: FrozenSet[str] = frozenset({"crop_1.0_equalised"})
    min_box_size_px: float = 32.0
    arm_names: Tuple[str, ...] = ("crop_1.0", "crop_0.8", "crop_1.0_equalised", "whole_image")
