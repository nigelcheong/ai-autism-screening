"""Configuration for the single-image demo notebook.

Not a new experiment: every number this demo shows is a re-derivation from
another experiment's own cache (embeddings, named/facial-shape features,
MTCNN boxes, CNN fold checkpoints). This config only points at those caches
and fixes the handful of settings the demo adds on top (image selection
seed, out-of-fold protocol, photographic-degradation strengths).
"""

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
    """Paths into every prior experiment's cache, plus the demo's own settings.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: This demo's own output directory (currently unused for
            caching -- every displayed image is rendered in memory from a
            cached embedding/feature/model, never re-saved to disk).
        intact_cache_dir: `intact_images`'s embedding cache.
        blurred_cache_dir: `blurred_images`'s per-sigma embedding and sigma=0
            MTCNN detection caches.
        background_cache_dir: `background_images`'s masked/cropped embedding
            caches.
        face_crop_blur_cache_dir: `face_crop_blur`'s crop embedding caches.
        face_landmarks_cache_dir: `face_landmarks`'s MediaPipe landmark and
            morphology/pose feature caches.
        why_not_faces_cache_dir: `why_not_faces`'s assembled named-feature
            table and fine-tuned CNN fold checkpoints/predictions.
        splits_dir: The pre-built, hash-deduplicated CV split
            (`index.csv`, `cv_folds.csv`) the corrected-protocol CNN folds
            were trained on.
        seed: Base random seed for every re-derived fold split -- identical
            to `seed` in every other experiment's `Config`, so `repeat_index`
            below reproduces the *same* fold membership those experiments'
            own repeat 0 used, not a fresh split.
        n_splits: Folds per repeat (matches every other experiment).
        n_repeats: Repeats the base seed is drawn for (matches every other
            experiment) -- only `repeat_index` of these is actually used
            here, but the seed must be drawn the same way to land on the
            same repeat.
        repeat_index: Which repeat's fold split to use for every frozen-probe
            arm's out-of-fold predictions. `0` on purpose: it is also the
            only repeat the corrected-protocol CNN was fine-tuned under, so
            "which fold held this image out" means the same thing for every
            arm shown in the demo, frozen probe or CNN.
        backbone: Frozen-backbone key, matches every embedding experiment.
        image_size: Square side length of the shared transform pipeline.
        device: Torch device string.
        selection_seed: Seed for the demo-image draw itself (rejection
            sampling described in the notebook). Kept as its own field,
            separate from `seed`, so re-running the notebook with a
            different fold protocol would not silently also change which
            four images get shown.
        n_demo_per_class: Demo images drawn per class.
        max_selection_attempts: Upper bound on rejection-sampling draws
            before giving up (the constraint -- at least one misclassified
            image -- is expected to be satisfied quickly at this dataset's
            base rate, so this is a safety cap, not a tuning knob).
        decision_threshold: Probability threshold a prediction is "wrong"
            or "flipped" against.
        mask_expand: Box expansion factor for the "face masked out" /
            "face crop" demo conditions. `1.5`, matching
            `background_images.Config.expand_wide` -- that experiment's own
            reasoning ("the honest version of the test," not a tight
            detector box that excludes forehead/hair/ears) applies here
            identically.
        crop_expand: Box expansion factor for the face-crop-only condition.
            `1.0`, matching `face_crop_blur`'s `"crop_1.0"` arm, whose
            embedding cache this demo reuses directly.
        named_feature_groups: Which of `why_not_faces.FEATURE_GROUPS` make
            up the "named features only" demo condition -- acquisition,
            colour and framing (16 columns total), deliberately excluding
            `facial_shape` and `pose`, which get their own, separate demo
            condition ("facial shape only") lower in the sequence.
        facial_shape_group: Which `why_not_faces.FEATURE_GROUPS` key is the
            "facial shape only" condition -- the 11 morphology measurements.
        counterfactual_n_images: How many non_autistic images the
            photographic-counterfactual flip-rate is measured over.
        counterfactual_downsample_size: Bottleneck side length for the
            downscale-then-upscale degradation.
        counterfactual_jpeg_quality: JPEG quality factor for the
            re-encoding degradation.
        counterfactual_sharpness_factor: `PIL.ImageEnhance.Sharpness` factor
            for the sharpness-reduction degradation (`1.0` = unchanged,
            `< 1.0` = softened).
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "demo")
    intact_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "intact_images")
    blurred_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "blurred_images")
    background_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "background_images")
    face_crop_blur_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "face_crop_blur")
    face_landmarks_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "face_landmarks")
    why_not_faces_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "why_not_faces")
    splits_dir: Path = field(default_factory=lambda: _project_root() / "data" / "splits")

    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 10
    repeat_index: int = 0

    backbone: str = "resnet50"
    image_size: int = 224
    device: str = "cuda"

    selection_seed: int = 42
    n_demo_per_class: int = 2
    max_selection_attempts: int = 10_000
    decision_threshold: float = 0.5

    mask_expand: float = 1.5
    crop_expand: float = 1.0

    named_feature_groups: tuple = ("acquisition", "colour", "framing")
    facial_shape_group: str = "facial_shape"

    counterfactual_n_images: int = 50
    counterfactual_downsample_size: int = 96
    counterfactual_jpeg_quality: int = 60
    counterfactual_sharpness_factor: float = 0.7
