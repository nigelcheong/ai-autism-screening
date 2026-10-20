"""Inference wrapper for the autism screening models.

Contract (see plan.md #3):
  questionnaire input  -> answers keyed by question key -> selected option
                          index, scored against each question's snapshotted
                          scoring_indices (see core/diagnosis.py and
                          core/instruments.py), plus the instrument id
                          ("aq10" or "qchat10")
  image input          -> {"image_urls": [...]}
  output (both types)  -> {"risk_score": float, "risk_category": "low"|"moderate"|"high",
                           "model_version": str}

Questionnaire scoring uses the trained logistic-regression pipelines in
app/ml/models/ (joblib-dumped sklearn Pipelines, one per instrument). Image
scoring uses the fine-tuned DenseNet201 CNN in app/ml/models_image/ (a
plain-pickled torchvision nn.Module; see experiments/why_not_faces/cnn.py in
the model-training repo's facial-stream branch for the training code).
USE_ML_STUB_QUESTIONNAIRE / USE_ML_STUB_IMAGE can force the deterministic
rule-based stub for either path independently.
"""

import uuid
from pathlib import Path

from app.config import settings
from app.core.instruments import category_for_raw_score
from app.core.risk_bands import categorize as _categorize
from app.core.uploads import heatmap_dir, resolve_image_path

# SHAP settings: a tradeoff of fidelity vs. request latency on CPU. At
# nsamples=20 with a 4-image background, one explanation takes ~15-25s --
# tested and chosen deliberately. Pooled into a grid of regions and ranked
# (see _compute_shap_regions) rather than shown as a per-pixel heat map,
# since the per-pixel signal is noisy at this sample count and a ranked
# chart is what was actually asked for.
SHAP_NSAMPLES = 20
SHAP_BACKGROUND_SIZE = 4

MODEL_VERSION_STUB = "stub-0.2"

MODELS_DIR = Path(__file__).resolve().parent / "models"
IMAGE_MODEL_PATH = Path(__file__).resolve().parent / "models_image" / "cnn_corrected_fold4_model.pkl"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_image_model = None

# Default trained model per instrument, chosen for the larger training set
# of the two variants shipped for each (see the sidecar *_meta.json files).
QUESTION_MODEL_FILES = {
    "aq10": "children_4-11_962rows_v2.pkl",  # Thabtah dataset, 960 rows
    "qchat10": "qchat10_toddlers_nz_saudi_v2.pkl",  # NZ + Saudi, 1560 rows
}

_question_models: dict[str, dict] = {}


def _load_question_model(instrument: str) -> dict:
    model = _question_models.get(instrument)
    if model is None:
        import joblib

        filename = QUESTION_MODEL_FILES[instrument]
        model = joblib.load(MODELS_DIR / filename)
        _question_models[instrument] = model
    return model


def _ordered_keys(instrument: str) -> list[str]:
    # Must match the actual Question.key values seeded for each instrument
    # ("aq1".."aq10", "qc1".."qc10") -- not an invented convention.
    prefix = "qc" if instrument == "qchat10" else "aq"
    return [f"{prefix}{i}" for i in range(1, 11)]


def run_questionnaire_inference(
    answers: dict[str, int], question_meta: dict[str, dict], instrument: str
) -> dict:
    raw_score = sum(
        1
        for key, selected in answers.items()
        if key in question_meta and selected in question_meta[key]["scoring_indices"]
    )
    # The clinical referral cutoffs (AQ-10 >=6, Q-CHAT-10 >=3) are tied to
    # this raw score, so category always comes from it regardless of which
    # scoring path produced risk_score.
    risk_category = category_for_raw_score(raw_score)

    if settings.use_ml_stub_questionnaire:
        return {
            "risk_score": round(raw_score / 10, 2),
            "risk_category": risk_category,
            "model_version": MODEL_VERSION_STUB,
        }

    import pandas as pd

    model = _load_question_model(instrument)
    pipeline = model["pipeline"]
    feature_names = model["feature_names"]

    ordered_keys = _ordered_keys(instrument)
    row = {
        feature_names[i]: (
            1
            if ordered_keys[i] in question_meta
            and answers.get(ordered_keys[i]) in question_meta[ordered_keys[i]]["scoring_indices"]
            else 0
        )
        for i in range(len(feature_names))
    }
    frame = pd.DataFrame([row], columns=feature_names)
    proba_yes = float(pipeline.predict_proba(frame)[0, 1])

    return {
        "risk_score": round(proba_yes, 4),
        "risk_category": risk_category,
        "model_version": f"{instrument}-logreg-{model.get('n_rows_trained', 'na')}",
        "feature_contributions": _explain_logreg(pipeline, feature_names, row, ordered_keys),
    }


def _explain_logreg(
    pipeline, feature_names: list[str], row: dict[str, int], ordered_keys: list[str]
) -> dict[str, float]:
    """Per-question push on the model's logit, keyed by question key.

    For a linear model this is an exact decomposition, not an approximation:
    logit = intercept + sum_i(coef_i * standardized_value_i), where
    standardized_value_i = (x_i - scaler.mean_i) / scaler.scale_i is exactly
    what the pipeline's own StandardScaler computes. Summing every
    contribution plus the intercept reproduces the model's logit exactly, so
    this is the real reason behind its own score -- the same idea as SHAP,
    without the sampling approximation SHAP needs for non-linear models.
    """
    scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
    clf = pipeline.steps[-1][1]
    values = [row[name] for name in feature_names]
    contributions = [
        float(clf.coef_[0][i] * (values[i] - scaler.mean_[i]) / scaler.scale_[i])
        for i in range(len(feature_names))
    ]
    return dict(zip(ordered_keys, contributions))


def _load_image_model():
    global _image_model
    if _image_model is None:
        import pickle

        with open(IMAGE_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model.eval()
        _image_model = model
    return _image_model


def _load_resized_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def _preprocess_image(path: Path):
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(_load_resized_rgb(path)).unsqueeze(0)


class _GradCAM:
    """Grad-CAM (Selvaraju et al. 2017), ported from the model-training
    repo's own `experiments/why_not_faces/cnn.py` (facial-stream branch) --
    same hook strategy and the same sign-aware backprop so a confidently
    "non_autistic" prediction still gets a coherent "what pushed it that
    way" map instead of Grad-CAM's single-logit all-zero failure mode.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, target_sign: float = 1.0):
        import torch

        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze()
        (target_sign * logit).backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.detach().squeeze().cpu().numpy().astype("float32")
        peak = cam.max()
        return cam / peak if peak > 0 else cam


def _jet_colormap(heat):
    """Vectorized MATLAB-style "jet" colormap: blue (low) -> cyan -> green ->
    yellow -> red (high). Every value gets a fully saturated color -- unlike
    alpha-blending a color by its own heat value (which makes low-attention
    areas fade to invisible instead of reading as "blue"), this stays
    legible across the whole range."""
    import numpy as np

    v = heat * 4
    r = np.clip(np.minimum(v - 1.5, -v + 4.5), 0, 1)
    g = np.clip(np.minimum(v - 0.5, -v + 3.5), 0, 1)
    b = np.clip(np.minimum(v + 0.5, -v + 2.5), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255.0


def _save_colormap_image(heat, directory: Path) -> str:
    """Render a (H, W) map already normalized to [0, 1] as its own pure
    colormap image (not blended over the photo -- shown side by side with
    the original instead, so neither one dilutes the other). Saved under a
    fresh UUID name (never derived from the source filename)."""
    import numpy as np
    from PIL import Image

    heat_rgb = _jet_colormap(heat)
    heat_img = Image.fromarray(heat_rgb.clip(0, 255).astype("uint8"))
    name = f"{uuid.uuid4().hex}.jpg"
    heat_img.save(directory / name, format="JPEG", quality=85)
    return name


def _save_heatmap(cam) -> str:
    """Upsample a Grad-CAM map (coarse, e.g. 7x7) to the image size first."""
    import numpy as np
    from PIL import Image

    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    heat = np.asarray(cam_img).astype("float32") / 255.0  # (H, W) in [0, 1]
    return _save_colormap_image(heat, heatmap_dir())


_shap_background = None


def _load_shap_background():
    """A handful of real previously-uploaded images, preprocessed the same
    way as inference input. SHAP needs a representative baseline of "typical"
    inputs to measure each pixel's marginal contribution against -- these
    don't need to exclude the image being explained, any same-distribution
    sample works, and real uploads are exactly that (no external dataset
    needed). Built once per process and reused across requests."""
    global _shap_background
    if _shap_background is not None:
        return _shap_background

    import torch

    from app.core.uploads import upload_root

    candidates = sorted(p for p in upload_root().glob("*") if p.is_file())[:SHAP_BACKGROUND_SIZE]
    if not candidates:
        raise RuntimeError("no uploaded images available yet to build a SHAP background set")
    _shap_background = torch.cat([_preprocess_image(p) for p in candidates], dim=0)
    return _shap_background


SHAP_GRID = 3  # 3x3 regions

_REGION_ROW_NAMES = ["Top", "Middle", "Bottom"]
_REGION_COL_NAMES = ["left", "center", "right"]


def _region_label(row: int, col: int) -> str:
    if row == 1 and col == 1:
        return "Center region"
    return f"{_REGION_ROW_NAMES[row]}-{_REGION_COL_NAMES[col]} region"


def _compute_shap_grid(model, tensor) -> list[list[float]]:
    """SHAP (shap.GradientExplainer), pooled into a SHAP_GRID x SHAP_GRID
    grid of regions, row-major, signed (not abs'd) so a region can genuinely
    read as "pushed toward" or "pushed away from" the prediction. The single
    source of truth both the ranked-list chart and the on-image diagram are
    built from, so SHAP only ever runs once per image."""
    import numpy as np
    import shap

    background = _load_shap_background()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(tensor, nsamples=SHAP_NSAMPLES)

    sv = shap_values[0, :, :, :, 0]  # (3, H, W) -- this model has one output
    signed = sv.sum(axis=0)  # pool channels, keep sign -> (H, W)

    h, w = signed.shape
    row_edges = np.linspace(0, h, SHAP_GRID + 1).astype(int)
    col_edges = np.linspace(0, w, SHAP_GRID + 1).astype(int)

    return [
        [
            float(
                signed[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]].sum()
            )
            for c in range(SHAP_GRID)
        ]
        for r in range(SHAP_GRID)
    ]


def _grid_to_ranked_list(grid: list[list[float]]) -> list[dict]:
    """Chart data: bars + direction + magnitude, the same shape as the
    questionnaire's per-answer explanation."""
    max_abs = max((abs(v) for row in grid for v in row), default=0) or 1.0
    items = [
        (_region_label(r, c), grid[r][c])
        for r in range(SHAP_GRID)
        for c in range(SHAP_GRID)
    ]
    ranked = sorted(items, key=lambda item: -abs(item[1]))
    return [
        {
            "label": label,
            "direction": "increased" if value > 0 else "decreased",
            "magnitude": round(abs(value) / max_abs * 100, 1),
        }
        for label, value in ranked
    ]


SHAP_DIAGRAM_SIZE = 450
SHAP_DIAGRAM_MARGIN = 160


def _find_font(size: int):
    from PIL import ImageFont

    for path in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_shap_diagram(base_image, grid: list[list[float]]):
    """Draw the SHAP region grid directly on an enlarged copy of the photo:
    each cell tinted red (pushed toward the prediction) or blue (pushed
    away) by magnitude, a 3x3 grid overlay, and an arrow callout per region
    labeling its percentage -- the same numbers as the ranked-list chart,
    shown spatially on the image instead of (as well as) in a chart.
    """
    import math

    from PIL import Image, ImageDraw

    size = SHAP_DIAGRAM_SIZE
    margin = SHAP_DIAGRAM_MARGIN
    canvas_size = size + margin * 2

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    canvas.paste(base_image.resize((size, size), Image.BILINEAR), (margin, margin))

    # PIL's ImageDraw overwrites pixels rather than alpha-blending when
    # drawn directly onto an opaque image, so the translucent region tints
    # are drawn on a separate transparent layer and composited in.
    overlay = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    cell = size / SHAP_GRID
    max_abs = max((abs(v) for row in grid for v in row), default=0) or 1.0

    for r in range(SHAP_GRID):
        for c in range(SHAP_GRID):
            value = grid[r][c]
            pct = abs(value) / max_abs * 100
            color = (220, 38, 38) if value > 0 else (37, 99, 235)
            x0, y0 = margin + c * cell, margin + r * cell
            alpha = int(35 + 90 * (pct / 100))
            odraw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color + (alpha,))

    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _find_font(24)
    font_small = _find_font(18)

    for i in range(1, SHAP_GRID):
        draw.line([(margin + i * cell, margin), (margin + i * cell, margin + size)], fill=(255, 255, 255, 230), width=3)
        draw.line([(margin, margin + i * cell), (margin + size, margin + i * cell)], fill=(255, 255, 255, 230), width=3)
    draw.rectangle([margin, margin, margin + size, margin + size], outline=(50, 50, 50, 255), width=3)

    mid = SHAP_GRID // 2
    for r in range(SHAP_GRID):
        for c in range(SHAP_GRID):
            value = grid[r][c]
            pct = abs(value) / max_abs * 100
            color = (220, 38, 38) if value > 0 else (37, 99, 235)
            arrow = "\u25B2" if value > 0 else "\u25BC"
            label = f"{arrow} {pct:.0f}%"

            x0, y0 = margin + c * cell, margin + r * cell
            cx, cy = x0 + cell / 2, y0 + cell / 2

            if r == mid and c == mid:
                # Center cell: no room for an external arrow, label it directly.
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                box = [cx - tw / 2 - 10, cy - th / 2 - 8, cx + tw / 2 + 10, cy + th / 2 + 8]
                draw.rectangle(box, fill=(255, 255, 255, 235))
                draw.text((cx - tw / 2, cy - th / 2), label, fill=color, font=font)
                continue

            dx, dy = c - mid, r - mid
            ax = min(max(margin + size / 2 + dx * (size / 2 + margin * 0.6), 14), canvas_size - 14)
            ay = min(max(margin + size / 2 + dy * (size / 2 + margin * 0.6), 14), canvas_size - 14)

            draw.line([(ax, ay), (cx, cy)], fill=color + (255,), width=3)
            angle = math.atan2(cy - ay, cx - ax)
            ah = 14
            draw.polygon(
                [
                    (cx, cy),
                    (cx - ah * math.cos(angle - math.pi / 7), cy - ah * math.sin(angle - math.pi / 7)),
                    (cx - ah * math.cos(angle + math.pi / 7), cy - ah * math.sin(angle + math.pi / 7)),
                ],
                fill=color + (255,),
            )

            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = min(max(ax - tw / 2, 4), canvas_size - tw - 4)
            ty = min(max(ay - th / 2, 4), canvas_size - th - 4)
            draw.rectangle([tx - 5, ty - 3, tx + tw + 5, ty + th + 3], fill=(255, 255, 255, 235))
            draw.text((tx, ty), label, fill=color, font=font_small)

    return canvas.convert("RGB")


def _safe_shap(model, tensor, base_image) -> tuple[list[dict], str | None]:
    import logging

    try:
        grid = _compute_shap_grid(model, tensor)
    except Exception:
        # SHAP is a secondary, best-effort explanation -- never fail the
        # whole screening because of it, but always log why so this is
        # debuggable instead of silently disappearing.
        logging.getLogger(__name__).exception("SHAP explanation failed")
        return [], None

    ranked = _grid_to_ranked_list(grid)

    diagram_name = None
    try:
        from app.core.uploads import shap_diagram_dir

        diagram = _render_shap_diagram(base_image, grid)
        diagram_name = f"{uuid.uuid4().hex}.jpg"
        diagram.save(shap_diagram_dir() / diagram_name, format="JPEG", quality=88)
    except Exception:
        logging.getLogger(__name__).exception("SHAP diagram rendering failed")
        diagram_name = None

    return ranked, diagram_name


def run_image_inference(image_urls: list[str]) -> dict:
    if settings.use_ml_stub_image or not image_urls:
        score = 0.5
        return {
            "risk_score": score,
            "risk_category": _categorize(score),
            "model_version": MODEL_VERSION_STUB,
            "per_image_scores": [score for _ in image_urls],
            "per_image_heatmaps": [None for _ in image_urls],
            "per_image_shap_regions": [[] for _ in image_urls],
            "per_image_shap_diagrams": [None for _ in image_urls],
        }

    import torch

    model = _load_image_model()
    cam_fn = _GradCAM(model, dict(model.named_modules())["features"])

    probs = []
    heatmaps = []
    shap_regions = []
    shap_diagrams = []
    for url in image_urls:
        path = resolve_image_path(url)
        tensor = _preprocess_image(path)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor).squeeze()).item()
        probs.append(round(prob, 4))

        # Grad-CAM needs a backward pass, so it runs outside no_grad, as a
        # second forward pass dedicated to the attention map. Explaining
        # "why autistic" is only coherent when the model actually leaned
        # that way; otherwise explain "why non_autistic" instead (see
        # _GradCAM's docstring).
        target_sign = 1.0 if prob >= 0.5 else -1.0
        cam = cam_fn(tensor, target_sign=target_sign)
        heatmaps.append(_save_heatmap(cam))

        ranked, diagram_name = _safe_shap(model, tensor, _load_resized_rgb(path))
        shap_regions.append(ranked)
        shap_diagrams.append(diagram_name)

    # Images are independent evidence about the same patient; average their
    # probabilities rather than picking one, so no single photo dominates.
    risk_score = round(sum(probs) / len(probs), 4)

    return {
        "risk_score": risk_score,
        "risk_category": _categorize(risk_score),
        "model_version": "cnn_corrected_fold4_model",
        "per_image_scores": probs,
        "per_image_heatmaps": heatmaps,
        "per_image_shap_regions": shap_regions,
        "per_image_shap_diagrams": shap_diagrams,
    }
