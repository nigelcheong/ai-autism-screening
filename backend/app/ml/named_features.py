"""Named-feature SHAP explanation for facial image screening.

Unlike the pixel-grid SHAP in inference.py (which pools raw CNN gradient
attributions into 3x3 image regions -- a necessary compromise when the only
available model takes pixels), this computes ~30 hand-engineered,
human-readable features (image acquisition stats, colour, face-box framing,
11 facial morphology measurements, head pose) and runs real SHAP
(TreeExplainer) against a HistGradientBoostingClassifier trained on them.

Feature formulas (alignment, morphology, pose, colour) and the model recipe
are reused from github.com/nigelcheong/ai-autism-screening
(src/data/landmarks.py, src/data/colour_features.py,
experiments/why_not_faces/pipeline.py) -- re-implemented here in one
self-contained module rather than vendoring that project's full src/ tree,
since only single-image inference is needed here (that project's own code
is built around batch manifests and cached experiment artefacts).

This is a SEPARATE, independent explanation from the CNN's own risk score --
it does not change risk_score/risk_category, which still come from the CNN
(cnn_corrected_fold4_model.pkl). Caveat, important: the training dataset
(Mendeley f9dycfvwbt v2 / Kaggle ASD Children Facial Image Dataset) has a
documented provenance problem -- cases and controls were collected from
different sources, so some of this signal may reflect acquisition context
rather than true facial phenotype (see explanation_note in reports.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models_named_features"
MODEL_PATH = MODELS_DIR / "named_features_hgb.pkl"
LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

IMAGE_SIZE = 224

# ---------------------------------------------------------------------------
# Feature groups + display names (see experiments/why_not_faces/pipeline.py
# FEATURE_GROUPS / FEATURE_DISPLAY_NAMES in the source project).
# ---------------------------------------------------------------------------
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

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "width": "Image width (px)",
    "height": "Image height (px)",
    "n_pixels": "Image pixel count",
    "file_size_bytes": "File size (bytes)",
    "bytes_per_pixel": "Bytes per pixel (compression)",
    "sharpness": "Image sharpness",
    "mean_r": "Mean red",
    "mean_g": "Mean green",
    "mean_b": "Mean blue",
    "mean_luminance": "Mean brightness",
    "std_luminance": "Brightness contrast (SD)",
    "saturation_mean": "Mean colour saturation",
    "box_area_fraction": "Face size (fraction of image)",
    "box_centre_x": "Face position, horizontal",
    "box_centre_y": "Face position, vertical",
    "box_aspect": "Face box aspect ratio",
    "fwhr": "Facial width-to-height ratio",
    "mouth_width": "Mouth width (IOD)",
    "eye_to_mouth": "Eyes-to-mouth distance (IOD)",
    "eye_to_nose": "Eyes-to-nose distance (IOD)",
    "nose_to_mouth": "Nose-to-mouth distance (IOD)",
    "nose_width": "Nose width (IOD)",
    "philtrum_length": "Philtrum length (IOD)",
    "palpebral_fissure_left": "Left eye width (IOD)",
    "palpebral_fissure_right": "Right eye width (IOD)",
    "intercanthal_over_iod": "Inner-eye-corner distance (IOD)",
    "asymmetry": "Facial asymmetry",
    "roll_deg": "Head roll (degrees)",
    "yaw_proxy": "Head yaw (left/right turn)",
    "pitch_proxy": "Head pitch (up/down tilt)",
}

ALL_FEATURE_NAMES: Tuple[str, ...] = tuple(name for names in FEATURE_GROUPS.values() for name in names)

# ---------------------------------------------------------------------------
# MediaPipe 478-point mesh indices (see module docstring in the source
# project's src/data/landmarks.py for how these were confirmed).
# ---------------------------------------------------------------------------
MP_RIGHT_EYE_OUTER, MP_RIGHT_EYE_INNER = 33, 133
MP_LEFT_EYE_OUTER, MP_LEFT_EYE_INNER = 263, 362
MP_RIGHT_EYE_UPPER_LID, MP_LEFT_EYE_UPPER_LID = 159, 386
MP_NOSE_TIP, MP_NOSE_BOTTOM = 1, 2
MP_NOSE_ALA_RIGHT, MP_NOSE_ALA_LEFT = 48, 278
MP_MOUTH_CORNER_RIGHT, MP_MOUTH_CORNER_LEFT = 61, 291
MP_UPPER_LIP_TOP = 0
MP_CHEEK_RIGHT, MP_CHEEK_LEFT = 234, 454

MP_MIRROR_PAIRS: Tuple[Tuple[int, int], ...] = (
    (MP_RIGHT_EYE_OUTER, MP_LEFT_EYE_OUTER),
    (MP_RIGHT_EYE_INNER, MP_LEFT_EYE_INNER),
    (MP_MOUTH_CORNER_RIGHT, MP_MOUTH_CORNER_LEFT),
    (MP_NOSE_ALA_RIGHT, MP_NOSE_ALA_LEFT),
    (MP_CHEEK_RIGHT, MP_CHEEK_LEFT),
)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _midpoint(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def geometry_stage(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Decode -> EXIF orientation -> RGB -> letterbox pad to square -> resize."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    width, height = img.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def acquisition_features(path: Path, original: Image.Image) -> Dict[str, float]:
    width, height = original.size
    n_pixels = width * height
    file_size_bytes = path.stat().st_size
    bytes_per_pixel = file_size_bytes / n_pixels if n_pixels > 0 else float("nan")

    gray = np.asarray(original.convert("L"), dtype=np.float64)
    gy, gx = np.gradient(gray)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    sharpness = float((gxx + gyy).var())

    return {
        "width": width,
        "height": height,
        "n_pixels": n_pixels,
        "file_size_bytes": file_size_bytes,
        "bytes_per_pixel": bytes_per_pixel,
        "sharpness": sharpness,
    }


def colour_features(img: Image.Image) -> Dict[str, float]:
    rgb = img.convert("RGB")
    array = np.asarray(rgb, dtype=np.float64)
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = np.asarray(rgb.convert("HSV"), dtype=np.float64)[..., 1].mean() / 255.0
    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "saturation_mean": float(saturation),
    }


def _expand_box(box, expand: float, size: int):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * expand / 2.0, (y2 - y1) * expand / 2.0
    clipped = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    return tuple(max(0.0, min(float(size), v)) for v in clipped)


def framing_features(box) -> Dict[str, float]:
    x1, y1, x2, y2 = _expand_box(box, 1.0, IMAGE_SIZE)
    width, height = x2 - x1, y2 - y1
    return {
        "box_area_fraction": (width * height) / (IMAGE_SIZE**2),
        "box_centre_x": (x1 + x2) / 2.0 / IMAGE_SIZE,
        "box_centre_y": (y1 + y2) / 2.0 / IMAGE_SIZE,
        "box_aspect": width / height if height > 0 else float("nan"),
    }


def _similarity_align(points: np.ndarray, left_eye_xy, right_eye_xy):
    points = np.asarray(points, dtype=np.float64)
    left_eye_xy = np.asarray(left_eye_xy, dtype=np.float64)
    right_eye_xy = np.asarray(right_eye_xy, dtype=np.float64)

    eye_mid = (left_eye_xy + right_eye_xy) / 2.0
    delta = right_eye_xy - left_eye_xy
    inter_ocular = float(np.hypot(delta[0], delta[1]))
    roll_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))

    theta = -np.arctan2(delta[1], delta[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    aligned = ((points - eye_mid) @ rotation.T) / inter_ocular
    return aligned, roll_deg


def _align_mediapipe(points: np.ndarray):
    frame_left_eye = _midpoint(points[MP_RIGHT_EYE_OUTER], points[MP_RIGHT_EYE_INNER])
    frame_right_eye = _midpoint(points[MP_LEFT_EYE_OUTER], points[MP_LEFT_EYE_INNER])
    return _similarity_align(points, frame_left_eye, frame_right_eye)


def morphology_and_pose_features(points: np.ndarray) -> Dict[str, float]:
    aligned, roll_deg = _align_mediapipe(points)
    p = aligned
    right_outer, right_inner = p[MP_RIGHT_EYE_OUTER], p[MP_RIGHT_EYE_INNER]
    left_outer, left_inner = p[MP_LEFT_EYE_OUTER], p[MP_LEFT_EYE_INNER]
    right_lid, left_lid = p[MP_RIGHT_EYE_UPPER_LID], p[MP_LEFT_EYE_UPPER_LID]
    nose_tip, nose_bottom = p[MP_NOSE_TIP], p[MP_NOSE_BOTTOM]
    ala_right, ala_left = p[MP_NOSE_ALA_RIGHT], p[MP_NOSE_ALA_LEFT]
    mouth_right, mouth_left = p[MP_MOUTH_CORNER_RIGHT], p[MP_MOUTH_CORNER_LEFT]
    lip_top = p[MP_UPPER_LIP_TOP]
    cheek_right, cheek_left = p[MP_CHEEK_RIGHT], p[MP_CHEEK_LEFT]

    mouth_centre = _midpoint(mouth_right, mouth_left)
    bizygomatic_width = _dist(cheek_right, cheek_left)
    upper_face_top_y = (right_lid[1] + left_lid[1]) / 2.0
    upper_face_height = abs(lip_top[1] - upper_face_top_y)
    asymmetry = float(np.mean([_dist((-p[r][0], p[r][1]), p[l]) for r, l in MP_MIRROR_PAIRS]))

    yaw_proxy = float(nose_tip[0])
    pitch_proxy = float(nose_tip[1] / mouth_centre[1]) if mouth_centre[1] != 0 else float("nan")

    return {
        "fwhr": bizygomatic_width / upper_face_height if upper_face_height > 0 else float("nan"),
        "mouth_width": _dist(mouth_right, mouth_left),
        "eye_to_mouth": _dist((0.0, 0.0), mouth_centre),
        "eye_to_nose": _dist((0.0, 0.0), nose_tip),
        "nose_to_mouth": _dist(nose_tip, mouth_centre),
        "nose_width": _dist(ala_right, ala_left),
        "philtrum_length": _dist(nose_bottom, lip_top),
        "palpebral_fissure_right": _dist(right_outer, right_inner),
        "palpebral_fissure_left": _dist(left_outer, left_inner),
        "intercanthal_over_iod": _dist(right_inner, left_inner),
        "asymmetry": asymmetry,
        "roll_deg": roll_deg,
        "yaw_proxy": yaw_proxy,
        "pitch_proxy": pitch_proxy,
    }


# ---------------------------------------------------------------------------
# Detector + model loading, lazy/cached (mirrors the pattern already used
# for the image CNN in inference.py -- built once per process).
# ---------------------------------------------------------------------------
_mtcnn_detector = None
_landmarker = None
_model_bundle = None
_explainer = None


def _get_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        from facenet_pytorch import MTCNN

        _mtcnn_detector = MTCNN(keep_all=True, device="cpu")
    return _mtcnn_detector


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not LANDMARKER_MODEL_PATH.exists():
            import urllib.request

            LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            logger.info("downloading MediaPipe Face Landmarker model bundle")
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, LANDMARKER_MODEL_PATH)

        base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL_PATH))
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _get_model_and_explainer():
    global _model_bundle, _explainer
    if _model_bundle is None:
        import joblib
        import shap

        _model_bundle = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model_bundle["model"])
    return _model_bundle, _explainer


_PIXEL_FEATURES = {"width", "height", "n_pixels"}
_RATIO_FEATURES = {"fwhr", "box_aspect", "bytes_per_pixel"}
_PERCENT_FEATURES = {"box_area_fraction", "box_centre_x", "box_centre_y", "saturation_mean"}
_DEGREE_FEATURES = {"roll_deg"}
_IOD_FEATURES = {
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
}


def _format_value(name: str, value: float) -> str:
    """A short, human-readable rendering of one feature's measured value,
    so a clinician sees not just 'this pushed the estimate up' but what was
    actually measured -- e.g. 'Facial width-to-height ratio: 1.87' rather
    than just a bare direction and bar length."""
    if name == "file_size_bytes":
        return f"{value / 1024:.1f} KB"
    if name in _PIXEL_FEATURES:
        return f"{value:,.0f} px"
    if name in _PERCENT_FEATURES:
        return f"{value * 100:.0f}%"
    if name in _DEGREE_FEATURES:
        return f"{value:.1f}°"
    if name in _IOD_FEATURES:
        return f"{value:.2f} x inter-ocular distance"
    if name in _RATIO_FEATURES:
        return f"{value:.2f}"
    return f"{value:.1f}"


# Which two landmark indices define each facial_shape feature's measurement
# line, for drawing it directly on the photo (see _render_facial_shape_diagram).
# fwhr and asymmetry are composite (more than one line); handled specially.
_FEATURE_LANDMARK_LINES = {
    "mouth_width": (MP_MOUTH_CORNER_RIGHT, MP_MOUTH_CORNER_LEFT),
    "nose_width": (MP_NOSE_ALA_RIGHT, MP_NOSE_ALA_LEFT),
    "philtrum_length": (MP_NOSE_BOTTOM, MP_UPPER_LIP_TOP),
    "palpebral_fissure_right": (MP_RIGHT_EYE_OUTER, MP_RIGHT_EYE_INNER),
    "palpebral_fissure_left": (MP_LEFT_EYE_OUTER, MP_LEFT_EYE_INNER),
    "intercanthal_over_iod": (MP_RIGHT_EYE_INNER, MP_LEFT_EYE_INNER),
}


def _eye_midpoint_raw(points: np.ndarray) -> Tuple[float, float]:
    """Eye midpoint in raw (unaligned) pixel coordinates -- the same
    anatomical point align_mediapipe/_align_mediapipe places at the aligned
    frame's origin, needed here to draw eye_to_mouth/eye_to_nose/
    nose_to_mouth (which are measured from that origin) on the real photo.
    """
    frame_left_eye = _midpoint(points[MP_RIGHT_EYE_OUTER], points[MP_RIGHT_EYE_INNER])
    frame_right_eye = _midpoint(points[MP_LEFT_EYE_OUTER], points[MP_LEFT_EYE_INNER])
    return _midpoint(frame_left_eye, frame_right_eye)


FACIAL_SHAPE_DIAGRAM_SIZE = 560
FACIAL_SHAPE_DIAGRAM_MARGIN = 220


def _find_font(size: int):
    from PIL import ImageFont

    for path in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_facial_shape_diagram(
    geo_image, points: np.ndarray, facial_shap: Dict[str, float], max_shown: int = 6
) -> "Image.Image":
    """Draw the top facial-shape measurements that increased the estimate
    (SHAP > 0) as lines directly on an enlarged copy of the photo, with a
    percentage label on a ring around it and a thin connector back to the
    measurement. Only the increased ("pushed toward autistic") features are
    drawn -- showing all 11 in both directions got crowded and hard to
    read; this is the subset a clinician is most likely to want to see.
    Percentages are still relative to all 11 (not just the ones shown), so
    they match the magnitudes in "What influenced this result" above.
    """
    import math

    from PIL import Image, ImageDraw

    size = FACIAL_SHAPE_DIAGRAM_SIZE
    margin = FACIAL_SHAPE_DIAGRAM_MARGIN
    scale = size / geo_image.size[0]
    canvas_size = size + margin * 2
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    canvas.paste(geo_image.resize((size, size), Image.BILINEAR), (margin, margin))

    def to_canvas(pt) -> Tuple[float, float]:
        return (margin + pt[0] * scale, margin + pt[1] * scale)

    max_abs = max((abs(v) for v in facial_shap.values()), default=0) or 1.0

    increased = sorted(
        ((name, value) for name, value in facial_shap.items() if value > 0),
        key=lambda kv: -kv[1],
    )[:max_shown]
    facial_shap = dict(increased)

    def colour_for(value: float, pct: float) -> Tuple[int, int, int, int]:
        base = (220, 38, 38) if value > 0 else (37, 99, 235)
        alpha = int(140 + 115 * (pct / 100))
        return base + (alpha,)

    overlay = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    eye_mid = _eye_midpoint_raw(points)
    mouth_centre = _midpoint(points[MP_MOUTH_CORNER_RIGHT], points[MP_MOUTH_CORNER_LEFT])
    nose_tip = points[MP_NOSE_TIP]

    segments: Dict[str, list] = {}
    for name, (a, b) in _FEATURE_LANDMARK_LINES.items():
        segments[name] = [(points[a], points[b])]
    segments["eye_to_mouth"] = [(eye_mid, mouth_centre)]
    segments["eye_to_nose"] = [(eye_mid, nose_tip)]
    segments["nose_to_mouth"] = [(nose_tip, mouth_centre)]
    segments["fwhr"] = [(points[MP_CHEEK_RIGHT], points[MP_CHEEK_LEFT])]
    segments["asymmetry"] = [(points[r], points[l]) for r, l in MP_MIRROR_PAIRS]

    # Draw each measurement's own line(s) on the face first.
    midpoints: Dict[str, Tuple[float, float]] = {}
    for name, value in facial_shap.items():
        pct = abs(value) / max_abs * 100
        color = colour_for(value, pct)
        width = max(2, round(2 + 4 * (pct / 100)))
        for pt_a, pt_b in segments[name]:
            odraw.line([to_canvas(pt_a), to_canvas(pt_b)], fill=color, width=width)
        mid_a, mid_b = segments[name][0]
        midpoints[name] = to_canvas(_midpoint(mid_a, mid_b))

    # Place each label on a ring around the photo, ordered by its
    # measurement's natural angle from the face centre, then spread to even
    # angular spacing so 11 close-together points don't collide.
    cx, cy = margin + size / 2, margin + size / 2
    names = list(facial_shap.keys())
    names.sort(key=lambda n: math.atan2(midpoints[n][1] - cy, midpoints[n][0] - cx))
    n = len(names)
    ring_radius = size / 2 + margin * 0.72

    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _find_font(15)

    for i, name in enumerate(names):
        value = facial_shap[name]
        pct = abs(value) / max_abs * 100
        color = colour_for(value, pct)
        angle = -math.pi + (i + 0.5) * (2 * math.pi / n)
        ax = cx + ring_radius * math.cos(angle)
        ay = cy + ring_radius * math.sin(angle)
        ax = min(max(ax, 18), canvas_size - 18)
        ay = min(max(ay, 18), canvas_size - 18)
        mx, my = midpoints[name]

        draw.line([(ax, ay), (mx, my)], fill=color, width=2)

        arrow = "▲" if value > 0 else "▼"
        label = f"{FEATURE_DISPLAY_NAMES.get(name, name)}: {arrow} {pct:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = min(max(ax - tw / 2, 2), canvas_size - tw - 2)
        ty = min(max(ay - th / 2, 2), canvas_size - th - 2)
        draw.rectangle([tx - 4, ty - 2, tx + tw + 4, ty + th + 2], fill=(255, 255, 255, 235))
        draw.text((tx, ty), label, fill=color, font=font)

    return canvas.convert("RGB")


GROUP_DISPLAY_NAMES = {
    "acquisition": "image quality",
    "colour": "photo colour",
    "framing": "face framing",
    "pose": "head pose",
}


def compute_named_feature_shap(
    image_path: Path, diagram_dir: Path | None = None
) -> tuple[list[dict], list[dict], str | None]:
    """Compute the ~30 named features for one image, rank the 11
    facial-shape (morphology) features by SHAP contribution, surface the
    top non-facial contributors separately, and optionally render a
    diagram drawing each facial-shape measurement's line on the photo.

    SHAP itself is computed against the full model (all ~30 features, so
    attributions are exact). The main ranked list is restricted to the
    facial_shape group -- the clinically interpretable measurements -- but
    the same model also weighs acquisition/colour/framing/pose, and those
    can score higher than any facial measurement; the top 2 of those are
    returned separately (not mixed into the facial_shape list) so a report
    can show them labelled as what they are, not as a facial trait.

    Returns (ranked, other_ranked, diagram_filename). Both ranked lists are
    {"label": str, "detail": str, "direction": "increased"|"decreased",
    "magnitude": float 0-100}; other_ranked additionally has "group" (e.g.
    "image quality"). other_ranked's magnitude is scaled against all ~30
    features' SHAP values (not just its own 2), so it's honestly
    comparable to how much it actually drove the prediction, including
    relative to the facial_shape group above it. All three are
    ([], [], None) if no face/landmarks could be detected (matches the <1%
    failure rate measured against the training set).
    """
    import mediapipe as mp

    image_path = Path(image_path)
    original = Image.open(image_path)
    original.load()
    geo = geometry_stage(original)

    mtcnn = _get_mtcnn()
    boxes, probs = mtcnn.detect(geo)
    if boxes is None:
        logger.warning("compute_named_feature_shap: no face detected in %s", image_path)
        return [], [], None
    best_box = tuple(float(v) for v in boxes[int(np.argmax(probs))])

    landmarker = _get_landmarker()
    array = np.ascontiguousarray(np.asarray(geo, dtype=np.uint8))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        logger.warning("compute_named_feature_shap: no landmarks detected in %s", image_path)
        return [], [], None
    landmarks = result.face_landmarks[0]
    width, height = geo.size
    points = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)

    features = {}
    features.update(acquisition_features(image_path, original))
    features.update(colour_features(geo))
    features.update(framing_features(best_box))
    features.update(morphology_and_pose_features(points))

    bundle, explainer = _get_model_and_explainer()
    feature_names = bundle["feature_names"]
    X = np.array([[features[name] for name in feature_names]])
    shap_values = explainer.shap_values(X)[0]
    shap_by_name = dict(zip(feature_names, (float(v) for v in shap_values)))

    facial_shape_names = FEATURE_GROUPS["facial_shape"]
    facial_shap = {name: shap_by_name[name] for name in facial_shape_names}
    max_abs = max((abs(v) for v in facial_shap.values()), default=0) or 1.0
    ranked_names = sorted(facial_shape_names, key=lambda name: -abs(facial_shap[name]))

    ranked = [
        {
            "label": FEATURE_DISPLAY_NAMES.get(name, name),
            "detail": _format_value(name, features[name]),
            "direction": "increased" if facial_shap[name] > 0 else "decreased",
            "magnitude": round(abs(facial_shap[name]) / max_abs * 100, 1),
        }
        for name in ranked_names
    ]

    group_of = {name: group for group, names in FEATURE_GROUPS.items() for name in names}
    other_names = [name for name in feature_names if name not in facial_shape_names]
    global_max_abs = max((abs(v) for v in shap_by_name.values()), default=0) or 1.0
    other_ranked_names = sorted(other_names, key=lambda name: -abs(shap_by_name[name]))[:2]
    other_ranked = [
        {
            "label": FEATURE_DISPLAY_NAMES.get(name, name),
            "detail": _format_value(name, features[name]),
            "direction": "increased" if shap_by_name[name] > 0 else "decreased",
            "magnitude": round(abs(shap_by_name[name]) / global_max_abs * 100, 1),
            "group": GROUP_DISPLAY_NAMES.get(group_of[name], group_of[name]),
        }
        for name in other_ranked_names
    ]

    diagram_name = None
    if diagram_dir is not None:
        try:
            import uuid

            diagram = _render_facial_shape_diagram(geo, points, facial_shap)
            diagram_name = f"{uuid.uuid4().hex}.jpg"
            diagram.save(diagram_dir / diagram_name, format="JPEG", quality=88)
        except Exception:
            logger.exception("facial-shape SHAP diagram rendering failed")
            diagram_name = None

    return ranked, other_ranked, diagram_name
