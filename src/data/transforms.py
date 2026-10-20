"""Shared image transforms for the intact/blurred/background image experiments.

All three experiments must see the exact same pipeline except for one
deliberate manipulation each, so any difference in downstream results is
attributable to that manipulation alone, not to preprocessing drift. Do not
use a torchvision weights enum's bundled ``transforms()`` -- that would
silently diverge from this pipeline.

The pipeline is split into three stages so the blurred/background
experiments can insert their manipulation between geometry and
normalisation, and so a face detector (which needs pixel images, not
normalised tensors) can run on the post-geometry, pre-normalisation image.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def geometry_stage(img: Image.Image, size: int = 224) -> Image.Image:
    """Decode -> EXIF orientation -> RGB -> letterbox pad to square -> resize.

    Letterbox padding, not a centre crop, is deliberate: the background
    experiment depends on peripheral content surviving.

    Args:
        img: An already-opened PIL image.
        size: Target square side length in pixels.

    Returns:
        An RGB PIL image of size ``(size, size)``.
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    width, height = img.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def blur_stage(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur an RGB image. ``sigma=0`` returns the image unchanged.

    Blurring happens after resizing (see :func:`geometry_stage`), so sigma
    is in output-pixel units and comparable across the whole corpus
    regardless of each image's original resolution.

    Args:
        img: RGB PIL image, already resized to the target size.
        sigma: Gaussian blur radius in pixels. ``0`` is a no-op.

    Returns:
        The blurred image (or the input, unchanged, if ``sigma == 0``).
    """
    if sigma == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def to_tensor_stage(img: Image.Image) -> torch.Tensor:
    """ImageNet-normalise an RGB image into a CHW float32 tensor.

    Args:
        img: RGB PIL image, already at the target size.

    Returns:
        Normalised ``(3, H, W)`` float32 tensor.
    """
    array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def intact_transform(size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Build the reference-arm pipeline: geometry -> tensor (no manipulation).

    Equivalent to ``blurred_transform(sigma=0, size=size)``.

    Args:
        size: Target square side length in pixels.

    Returns:
        A callable mapping an already-opened PIL image to a normalised
        ``(3, size, size)`` float32 tensor.
    """

    def _transform(image: Image.Image) -> torch.Tensor:
        return to_tensor_stage(geometry_stage(image, size))

    return _transform


def blurred_transform(sigma: float, size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Build the blurred-arm pipeline: geometry -> blur -> tensor.

    ``sigma=0`` reproduces :func:`intact_transform` exactly, so the sigma=0
    point of the blur sweep is the intact arm and its cached embeddings can
    be reused rather than recomputed.

    Args:
        sigma: Gaussian blur radius in output-pixel units, applied after
            resizing to ``size``. ``0`` is a no-op.
        size: Target square side length in pixels.

    Returns:
        A callable mapping an already-opened PIL image to a normalised
        ``(3, size, size)`` float32 tensor.
    """

    def _transform(image: Image.Image) -> torch.Tensor:
        image = geometry_stage(image, size)
        image = blur_stage(image, sigma)
        return to_tensor_stage(image)

    return _transform


def resample_stage(img: Image.Image, small_size: int, size: int) -> Image.Image:
    """Downsample to `small_size` then upsample back to `size`.

    Used to equalise effective resolution across images that were cropped
    from differently-sized boxes and therefore upsampled to `size` by
    different factors: routing every image through the same
    small-then-back-up bottleneck removes that per-image upsampling factor
    as a confound, regardless of how much detail the crop had to begin
    with.

    Args:
        img: RGB PIL image, already at `size` (e.g. post-crop-and-resize).
        small_size: The common bottleneck side length.
        size: Side length to resize back up to.

    Returns:
        A new RGB PIL image of size `(size, size)`.
    """
    down = img.resize((small_size, small_size), Image.BILINEAR)
    return down.resize((size, size), Image.BILINEAR)


Box = Tuple[float, float, float, float]

# The fill colour for masked regions, in 0-255 pixel space: the ImageNet mean
# rounded to the nearest integer channel value. `to_tensor_stage` divides by
# 255 and subtracts this same mean before dividing by std, so a masked pixel
# lands within ~1/510 of exactly 0 post-normalisation -- the residual from
# uint8 quantisation, unavoidable if the same masked image is also going to
# be usable as a real (8-bit) PIL image for `colour_features`.
_MASK_FILL_RGB255 = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)


def _clip_box(box: Box, size: int) -> Box:
    """Clip a box to the `[0, size] x [0, size]` canvas, no resizing."""
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(size), x1)),
        max(0.0, min(float(size), y1)),
        max(0.0, min(float(size), x2)),
        max(0.0, min(float(size), y2)),
    )


def expand_box(box: Box, expand: float, size: int) -> Box:
    """Scale a box by `expand` around its own centre, then clip to canvas.

    `expand=1.0` is the box unchanged (aside from clipping). A tight
    detector box excludes forehead, hair and ears; `expand > 1.0` grows the
    box around its centre to include them before it is used as "the face
    region" by any of the masking/cropping transforms below.

    Args:
        box: `(x1, y1, x2, y2)` in the same pixel coordinates as the image
            it was detected on (post-geometry, pre-normalisation).
        expand: Scale factor applied to width and height around the box
            centre. `1.0` leaves the box's size unchanged.
        size: Side length of the square canvas the box is clipped to.

    Returns:
        `(x1, y1, x2, y2)`, expanded and clipped to `[0, size]`.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * expand / 2.0, (y2 - y1) * expand / 2.0
    return _clip_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), size)


def _box_mask(box: Box, expand: float, size: int) -> np.ndarray:
    """Boolean `(size, size)` array, `True` inside the expanded, clipped box.

    Uses pixel-centre semantics (`pixel i` is inside iff its centre `i+0.5`
    falls in `[x1, x2)` / `[y1, y2)`), so `_box_mask` and its complement
    always partition the canvas exactly -- no shared or missing pixel
    between a `face_masked` and `background_masked` transform built from
    the same box and `expand`.
    """
    x1, y1, x2, y2 = expand_box(box, expand, size)
    xs = np.arange(size).reshape(1, -1) + 0.5
    ys = np.arange(size).reshape(-1, 1) + 0.5
    return (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)


def masked_stage(img: Image.Image, box: Box, expand: float, keep: str) -> Image.Image:
    """Fill everything outside `keep` with the ImageNet mean colour.

    Args:
        img: RGB PIL image, already at the target size (post-geometry).
        box: `(x1, y1, x2, y2)` face box in the same pixel coordinates as
            `img`.
        expand: Passed to :func:`expand_box`.
        keep: `"outside"` keeps everything but the (expanded) box --
            "can the background alone predict the class?" -- `"inside"`
            keeps only the (expanded) box, at its original position and
            scale -- box position/scale still leak through, unlike
            :func:`crop_stage`.

    Returns:
        A new RGB PIL image, same size as `img`.
    """
    if keep not in ("outside", "inside"):
        raise ValueError(f"keep must be 'outside' or 'inside', got {keep!r}")

    size = img.size[0]
    inside = _box_mask(box, expand, size)
    zero_where = inside if keep == "outside" else ~inside

    array = np.asarray(img.convert("RGB")).copy()
    array[zero_where] = _MASK_FILL_RGB255
    return Image.fromarray(array, mode="RGB")


def crop_stage(img: Image.Image, box: Box, expand: float, size: int) -> Image.Image:
    """Crop to the expanded box, then resize back to `size`.

    Unlike :func:`masked_stage` with `keep="inside"`, this normalises away
    the face's position and scale within the frame -- what's left after
    cropping is only what was inside the box, filling the whole canvas.

    Args:
        img: RGB PIL image, already at the target size (post-geometry).
        box: `(x1, y1, x2, y2)` face box in the same pixel coordinates as
            `img`.
        expand: Passed to :func:`expand_box`.
        size: Side length to resize the crop back up (or down) to.

    Returns:
        A new RGB PIL image of size `(size, size)`.
    """
    x1, y1, x2, y2 = expand_box(box, expand, img.size[0])
    left, top = int(round(x1)), int(round(y1))
    right, bottom = max(left + 1, int(round(x2))), max(top + 1, int(round(y2)))
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.BILINEAR)


def face_masked_transform(box: Box, expand: float = 1.0, size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Build a transform that masks OUT the face box, keeping the surroundings.

    "Can the background alone predict the class?" The (expanded) box is
    filled with the ImageNet mean colour; everything outside it is
    untouched.

    Args:
        box: `(x1, y1, x2, y2)` face box, in the coordinates
            `geometry_stage(image, size)` produces.
        expand: Passed to :func:`expand_box`.
        size: Target square side length in pixels.

    Returns:
        A callable mapping an already-opened PIL image to a normalised
        `(3, size, size)` float32 tensor.
    """

    def _transform(image: Image.Image) -> torch.Tensor:
        geo = geometry_stage(image, size)
        masked = masked_stage(geo, box, expand, keep="outside")
        return to_tensor_stage(masked)

    return _transform


def background_masked_transform(
    box: Box, expand: float = 1.0, size: int = 224
) -> Callable[[Image.Image], torch.Tensor]:
    """Build a transform that masks everything EXCEPT the face box.

    The face stays at its original position and scale within the frame --
    unlike :func:`face_crop_transform`, box position and scale still leak
    through here, which is the point of running both.

    Args:
        box: `(x1, y1, x2, y2)` face box, in the coordinates
            `geometry_stage(image, size)` produces.
        expand: Passed to :func:`expand_box`.
        size: Target square side length in pixels.

    Returns:
        A callable mapping an already-opened PIL image to a normalised
        `(3, size, size)` float32 tensor.
    """

    def _transform(image: Image.Image) -> torch.Tensor:
        geo = geometry_stage(image, size)
        masked = masked_stage(geo, box, expand, keep="inside")
        return to_tensor_stage(masked)

    return _transform


def face_crop_transform(box: Box, expand: float = 1.0, size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Build a transform that crops to the (expanded) box, then resizes to `size`.

    Face only, with position and scale normalised away -- the cleanest
    positive test of the facial-phenotype hypothesis, since (unlike
    `background_masked`) nothing about the box's location or size in the
    original frame survives into the output.

    Args:
        box: `(x1, y1, x2, y2)` face box, in the coordinates
            `geometry_stage(image, size)` produces.
        expand: Passed to :func:`expand_box`.
        size: Target square side length in pixels.

    Returns:
        A callable mapping an already-opened PIL image to a normalised
        `(3, size, size)` float32 tensor.
    """

    def _transform(image: Image.Image) -> torch.Tensor:
        geo = geometry_stage(image, size)
        cropped = crop_stage(geo, box, expand, size)
        return to_tensor_stage(cropped)

    return _transform
