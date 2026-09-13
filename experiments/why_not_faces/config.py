"""Configuration for the why_not_faces experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


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


# The single source of truth for which named feature belongs to which
# group. Written to `features_groups.json` alongside the assembled
# feature table, and read back from there by every downstream step
# (SHAP attribution, group-alone probes, plotting) -- never redefined
# inline a second time.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "acquisition": ("width", "height", "n_pixels", "file_size_bytes", "bytes_per_pixel", "sharpness"),
    "colour": ("mean_r", "mean_g", "mean_b", "mean_luminance", "std_luminance", "saturation_mean"),
    "framing": ("box_area_fraction", "box_centre_x", "box_centre_y", "box_aspect"),
    "facial_shape": (
        "fwhr",
        "mouth_width",
        "eye_to_mouth",
        "eye_to_nose",
        "nose_to_mouth",
        "nose_width",
        "philtrum_length",
        "palpebral_fissure_left",
        "palpebral_fissure_right",
        "intercanthal_over_iod",
        "asymmetry",
    ),
    "pose": ("roll_deg", "yaw_proxy", "pitch_proxy"),
}


@dataclass(frozen=True)
class Config:
    """Paths, seed, feature groups and cross-validation settings.

    Attributes:
        data_root: Root of the untouched facial-image archive.
        output_dir: Where this experiment writes its caches and results.
        audit_manifest_path: `facial-images.ipynb`'s corpus-audit
            manifest -- source of the `acquisition` feature group. Its
            `path` column uses an old, no-longer-valid base directory
            (`data/AutismDataset/...`), so rows are matched back to the
            canonical manifest by `(split, class_label, filename)`, not
            by `path` directly.
        blurred_cache_dir: Where `blurred_images` cached its sigma=0
            colour features (`colour_features_sigma0.csv`) -- source of
            the `colour` group. Also where the sigma=0 MTCNN box cache
            lives, reused (never re-detected) for both the `framing`
            group and the Grad-CAM off-face-mass computation.
        face_landmarks_cache_dir: Where `face_landmarks` cached its
            MediaPipe morphology/pose feature tables
            (`features_mediapipe_morphology.csv`,
            `features_mediapipe_pose.csv`) -- source of the
            `facial_shape` and `pose` groups. This is also, by
            construction, the tightest row set of any source (it already
            excludes the one boxless image and the one MediaPipe
            failure), so the intersection every other source is joined
            down to is exactly this one.
        image_size: Square side length of the post-geometry image --
            matches every other experiment; only used here to reuse
            `background_images.pipeline.step_4_box_geometry` verbatim.
        excluded_audit_columns: Audit-manifest columns explicitly left
            out of the `acquisition` group -- both were shown (in the
            corpus audit) to carry no class signal, and including them
            would dilute that group's SHAP attribution with noise.
        seed: Base random seed for splitting and every classifier.
        n_splits: Folds per cross-validation repeat (named-feature
            model and every group-alone probe).
        n_repeats: Number of cross-validation repeats.
        shap_top_n_beeswarm: Number of top-ranked features shown in the
            beeswarm plot.
        shap_top_n_dependence: Number of top-ranked features given their
            own dependence plot.
        hgb_seed: `random_state` for `HistGradientBoostingClassifier`
            (kept distinct from the CV splitting seed for clarity, but
            set to the same value).
        cnn_backbone: torchvision model name.
        cnn_image_size: Square input side length -- matches every other
            experiment's `geometry_stage` output.
        cnn_batch_size: Training/eval batch size.
        cnn_max_epochs: Upper bound on training epochs per run (early
            stopping is expected to end most runs well before this).
        cnn_patience: Epochs without validation-AUC improvement before
            early stopping.
        cnn_lr: Adam learning rate, applied to every fine-tuned parameter.
        cnn_weight_decay: Adam weight decay.
        cnn_inner_val_fraction: Fraction of each corrected-protocol
            training fold carved out (stratified, group-aware) as the
            inner early-stopping validation split -- never the fold's
            own held-out test images.
        cnn_n_folds: Number of `cv_folds.csv` folds fine-tuned for the
            corrected protocol. Deliberately a single repeat (5 folds),
            not the 50 every non-CNN probe in this project uses --
            fine-tuning a CNN is orders of magnitude more expensive than
            fitting a linear/GBM probe on cached features, and the
            constraints explicitly scope "same folds, same seed" to
            "every non-CNN arm" only.
        cnn_device: Torch device string.
        gradcam_target_layer: Name of the DenseNet201 submodule whose
            output feature map Grad-CAM hooks -- the last convolutional
            block, before global pooling.
        splits_dir: Where the pre-built, perceptual-hash-deduplicated,
            group-aware CV split lives (`index.csv`, `cv_folds.csv`) --
            the corrected protocol's row/fold source, read as-is, not
            regenerated.
        intact_cache_dir: Where `intact_images` cached its frozen-backbone
            embeddings -- source of the consolidated figure's "Intact
            image, frozen probe" reference row.
        face_crop_blur_cache_dir: Where `face_crop_blur` cached its
            `crop_1.0` sigma=0 embeddings -- source of the consolidated
            figure's "Face crop" reference row.
    """

    data_root: Path = field(default_factory=lambda: _project_root() / "data" / "raw" / "images")
    output_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "why_not_faces")
    audit_manifest_path: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "audit" / "images" / "manifest.csv"
    )
    blurred_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "blurred_images")
    face_landmarks_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "face_landmarks")
    intact_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "intact_images")
    face_crop_blur_cache_dir: Path = field(default_factory=lambda: _project_root() / "outputs" / "face_crop_blur")
    splits_dir: Path = field(default_factory=lambda: _project_root() / "data" / "splits")

    excluded_audit_columns: Tuple[str, ...] = ("jpeg_quality_estimate", "is_grayscale")

    image_size: int = 224
    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 10
    hgb_seed: int = 42

    shap_top_n_beeswarm: int = 15
    shap_top_n_dependence: int = 3

    cnn_backbone: str = "densenet201"
    cnn_image_size: int = 224
    cnn_batch_size: int = 32
    cnn_max_epochs: int = 20
    cnn_patience: int = 4
    cnn_lr: float = 1e-4
    cnn_weight_decay: float = 1e-5
    cnn_inner_val_fraction: float = 0.15
    cnn_n_folds: int = 5
    cnn_device: str = "cuda"
    gradcam_target_layer: str = "features"
