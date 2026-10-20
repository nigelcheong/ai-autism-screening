"""Colour-only reference features for the `blurred_images` sweep.

Blurring removes spatial detail but not colour: as sigma grows, an image
converges on a flat patch of its own mean colour, and a frozen-backbone
embedding of that patch still encodes it. The audit in `facial-images.ipynb`
found `mean_r` had the largest class effect size in the corpus, so a
colour-only classifier is the correct null for "no facial signal remains" --
not chance (0.5). This module computes the six scalar colour statistics that
probe feeds on, from the same post-geometry, post-blur image the embedding
and face detector see, so the comparison is apples to apples.

The feature set and formulas mirror the audit's `extract_image_record`
(`notebooks/facial-images.ipynb`) exactly, minus the thumbnailing step: the
audit downscaled a full-resolution raw image to a 128px thumbnail before
measuring colour, whereas here the image is already at the pipeline's fixed
224px size, so no extra downscale is needed.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def colour_features(img: Image.Image) -> Dict[str, float]:
    """Six scalar colour statistics from a post-geometry, post-blur,
    pre-normalisation image.

    Args:
        img: RGB PIL image, already at the target size (post-geometry,
            post-blur -- i.e. what `src.data.transforms.blurred_transform`
            would pass to `to_tensor_stage`).

    Returns:
        Dict with keys `mean_r`, `mean_g`, `mean_b`, `mean_luminance`,
        `std_luminance`, `saturation_mean`.
    """
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


COLOUR_FEATURE_NAMES = (
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_luminance",
    "std_luminance",
    "saturation_mean",
)
