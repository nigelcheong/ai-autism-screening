"""Configuration for the face_landmarks experiment."""

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
    """Paths, seed, landmark sources and cross-validation settings.

    The cross-validation protocol (model, folds, seed) is identical to
    every prior experiment on purpose -- this experiment changes what is
    fed to the probe (facial geometry, not pixels), not how the probe is
    evaluated.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: Where this experiment writes its caches and results.
        blurred_cache_dir: Where `blurred_images` cached its sigma=0
            MTCNN box detections (`detections_mtcnn.csv`) and its
            sigma=0 colour features -- boxes are reused via
            `background_images.pipeline.step_2_boxes` /
            `step_3_exclude_boxless` (never re-detected here), and the
            colour features are reused as one of the main figure's
            reference lines.
        face_crop_blur_cache_dir: Where `face_crop_blur` cached its
            `crop_1.0` embeddings/colour-features per sigma -- reused for
            two of the main figure's reference lines (that arm's own
            sigma=0 colour floor, and its sigma=32 embedding AUC, the
            "blur-surviving residual").
        mediapipe_model_path: Local path to the MediaPipe Face Landmarker
            `.task` model bundle. Downloaded once (see
            `src.models.landmark_detect.download_face_landmarker_model`)
            and cached here -- not re-fetched on subsequent runs.
        seed: Base random seed for splitting and the classifier.
        image_size: Square side length of the post-geometry image every
            landmark is detected on (matches every other experiment's
            transform pipeline).
        device: Torch device string for MTCNN landmark extraction.
            MediaPipe's Face Landmarker runs on CPU (no GPU delegate on
            this platform for the Tasks API).
        n_splits: Folds per cross-validation repeat.
        n_repeats: Number of cross-validation repeats.
        min_face_side_px: Minimum of the (unexpanded) face box's width
            and height, in post-geometry pixel coordinates, required for
            the Part 4 size-robustness probe. Deliberately generous --
            meant to drop only images where landmark placement is least
            reliable, not to materially shrink the sample.
        noise_sigmas_px: Gaussian-noise magnitudes (pixel units, in the
            224-space landmarks are detected in) swept for the Part 4
            noise-robustness check, applied independently to every
            landmark coordinate before alignment.
        n_invariance_samples: Number of images used for the Part 2
            scale/rotation/translation invariance test.
        invariance_scale: Synthetic scale factor applied in that test.
        invariance_rotate_deg: Synthetic rotation applied in that test.
        invariance_translate_px: Synthetic translation applied in that
            test.
        invariance_tolerance: Maximum allowed absolute difference in any
            morphology feature between the untransformed and
            synthetically-transformed recomputation.
        correlation_flag_threshold: `|Pearson r|` against
            `box_area_fraction` above which a morphology feature is
            flagged as showing scale leakage.
        bh_alpha: Benjamini-Hochberg false-discovery-rate level for the
            per-feature effect-size table.
        box_geometry_feature_names: The box-geometry probe's input
            features -- identical set and computation to
            `background_images`, re-run here on this experiment's own
            row set.
        probe_names: The four probes compared in Part 3, in report order.
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "face_landmarks"
    )
    blurred_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "blurred_images"
    )
    face_crop_blur_cache_dir: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "face_crop_blur"
    )
    mediapipe_model_path: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "face_landmarks" / "models" / "face_landmarker.task"
    )
    seed: int = 42
    image_size: int = 224
    device: str = "cuda"
    n_splits: int = 5
    n_repeats: int = 10
    min_face_side_px: float = 100.0
    noise_sigmas_px: Tuple[float, ...] = (1.0, 2.0, 4.0)
    n_invariance_samples: int = 20
    invariance_scale: float = 2.0
    invariance_rotate_deg: float = 15.0
    invariance_translate_px: Tuple[float, float] = (20.0, 20.0)
    invariance_tolerance: float = 1e-3
    correlation_flag_threshold: float = 0.3
    bh_alpha: float = 0.05
    box_geometry_feature_names: Tuple[str, ...] = (
        "box_area_fraction",
        "box_centre_x",
        "box_centre_y",
        "box_aspect",
    )
    probe_names: Tuple[str, ...] = ("morphology", "pose", "box_geometry", "morphology_plus_pose")
