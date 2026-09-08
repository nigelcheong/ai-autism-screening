"""Face detection to anchor what counts as a "recognisable" face.

Used by the `blurred_images` sweep (to define "unrecognisable" as an
objective detection-rate threshold rather than a judgement call) and by the
`background_images` experiment (to decide which images have a face to mask
out).

MTCNN (`facenet-pytorch`) is the primary detector. It is a required
dependency here, not an optional one with a silent fallback: if it is not
importable, `detect_faces_mtcnn` raises immediately rather than quietly
degrading to a weaker detector. OpenCV's Haar cascade is still available as
`detect_faces_haar`, but only as an explicitly-run, explicitly-labelled
point of comparison -- the two detectors' rates are not comparable to each
other, so call sites must say which one produced a given number.

Detector failure is a *proxy* for human recognisability, not the same
thing -- a face a person could still identify may fail a detector at
moderate blur, and conversely a detector can fire on a blob that is not a
recognisable face (see `corroborated_detection_rate`, which exists
precisely because raw detection rate cannot tell the two apart).
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]


def detect_faces_mtcnn(images: Sequence[Image.Image], device: str = "cuda") -> pd.DataFrame:
    """Detect faces with MTCNN (`facenet-pytorch`). The primary detector.

    Fails loudly if `facenet-pytorch` is not installed -- this experiment's
    "unrecognisable" anchor depends on knowing which detector produced a
    given rate, so a silent fallback to a weaker detector would corrupt the
    number without saying so.

    Args:
        images: RGB PIL images, already at the target size (post-geometry,
            post-blur, pre-normalisation).
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        One row per image, in the given order: `n_faces` (int), `best_box`
        (`(x1, y1, x2, y2)` tuple or `None`), `best_score` (float or
        `None`).

    Raises:
        RuntimeError: If `facenet-pytorch` is not installed.
    """
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError(
            "facenet-pytorch is not installed -- MTCNN is the required primary "
            "face detector for this experiment (see requirements.txt: "
            "`pip install --no-deps facenet-pytorch==2.6.0`). Refusing to "
            "silently fall back to the Haar cascade; call detect_faces_haar "
            "explicitly if a Haar-only comparison is what's wanted."
        ) from exc

    try:
        mtcnn_version = version("facenet-pytorch")
    except PackageNotFoundError:
        mtcnn_version = "unknown"
    logger.info("detect_faces_mtcnn: using MTCNN (facenet-pytorch %s) on %s", mtcnn_version, device)

    detector = MTCNN(keep_all=True, device=device)
    rows: List[dict] = []
    for image in images:
        boxes, probs = detector.detect(image)
        if boxes is None:
            rows.append({"n_faces": 0, "best_box": None, "best_score": None})
            continue
        best_idx = int(np.argmax(probs))
        rows.append(
            {
                "n_faces": int(len(boxes)),
                "best_box": tuple(float(v) for v in boxes[best_idx]),
                "best_score": float(probs[best_idx]),
            }
        )
    return pd.DataFrame(rows, columns=["n_faces", "best_box", "best_score"])


def detect_faces_haar(images: Sequence[Image.Image]) -> pd.DataFrame:
    """Detect faces with OpenCV's Haar cascade (CPU only).

    Kept only as an explicit, labelled point of comparison against MTCNN --
    not used as a fallback. Haar is known to be prone to false positives on
    blurred, low-detail patches, which is exactly the failure mode
    `corroborated_detection_rate` is designed to catch.

    Args:
        images: RGB PIL images, already at the target size (post-geometry,
            post-blur, pre-normalisation).

    Returns:
        One row per image, in the given order: `n_faces` (int), `best_box`
        (`(x1, y1, x2, y2)` tuple or `None`), `best_score` (float or
        `None`).
    """
    import cv2

    logger.info("detect_faces_haar: using OpenCV Haar cascade (CPU), opencv-python %s", cv2.__version__)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    rows: List[dict] = []
    for image in images:
        array = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        boxes, _, weights = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, outputRejectLevels=True
        )
        if len(boxes) == 0:
            rows.append({"n_faces": 0, "best_box": None, "best_score": None})
            continue
        best_idx = int(np.argmax(weights))
        x, y, w, h = boxes[best_idx]
        rows.append(
            {
                "n_faces": int(len(boxes)),
                "best_box": (float(x), float(y), float(x + w), float(y + h)),
                "best_score": float(weights[best_idx]),
            }
        )
    return pd.DataFrame(rows, columns=["n_faces", "best_box", "best_score"])


def parse_box(value) -> Optional[Box]:
    """Coerce a `best_box` cell back into a `(x1, y1, x2, y2)` tuple.

    A cached detections table round-trips through CSV, which turns the
    in-memory tuple into its string repr (or `NaN` for no detection). This
    undoes that so downstream code (IoU, plotting) always sees either a
    4-tuple of floats or `None`.

    Args:
        value: A `best_box` cell -- already a tuple/list, a string repr of
            one, or a missing value (`None`/`NaN`).

    Returns:
        `(x1, y1, x2, y2)` as floats, or `None` if there was no detection.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value)
    if isinstance(value, str):
        import ast

        parsed = ast.literal_eval(value)
        return tuple(float(v) for v in parsed)
    raise TypeError(f"Unrecognised best_box value: {value!r}")


def detection_rate(df: pd.DataFrame, by: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Proportion of images with at least one detected face.

    Args:
        df: A table with an `n_faces` column, and any columns named in
            `by`.
        by: Column name(s) to group by, e.g. `["sigma", "class_label"]`.
            `None` computes one overall rate.

    Returns:
        DataFrame with the grouping columns (if any) and a
        `detection_rate` column.
    """
    has_face = (df["n_faces"] > 0).rename("detection_rate")
    if by is None:
        return pd.DataFrame({"detection_rate": [has_face.mean()]})

    grouped = pd.concat([df[list(by)], has_face], axis=1)
    return grouped.groupby(list(by))["detection_rate"].mean().reset_index()


def _iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two `(x1, y1, x2, y2)` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def corroborated_detection_rate(
    detections: pd.DataFrame, iou_threshold: float = 0.5
) -> pd.DataFrame:
    """Proportion of images whose sigma>0 detection agrees spatially with the
    same image's sigma=0 detection. Images with no sigma=0 detection are
    excluded and counted separately.

    A raw detection rate cannot distinguish a genuine (blurred) face from a
    detector firing on an unrelated blob, because it has no notion of
    *where* the box is. This anchors each image's blurred-sigma detections
    to that same image's own sigma=0 box (its best available ground truth,
    since sigma=0 is unblurred): a detection only counts as corroborated if
    its box overlaps the sigma=0 box by IoU >= `iou_threshold`. At sigma=0
    itself, a detection trivially corroborates against itself (IoU=1).

    Args:
        detections: One row per `(path, sigma)`, with columns `path`,
            `sigma`, `class_label`, `n_faces`, `best_box` (as real tuples
            or `None` -- run values through `parse_box` first if this came
            from a CSV cache). Must include a `sigma=0` row for every
            `path` in the table.
        iou_threshold: Minimum IoU between a sigma>0 box and its image's
            sigma=0 box to count as corroborated.

    Returns:
        One row per `(sigma, class_label)`: `n_images`, `n_no_sigma0_detection`
        (excluded: this image had no sigma=0 detection to corroborate
        against), `n_eligible` (`n_images - n_no_sigma0_detection`), and
        `corroborated_rate` -- the corroborated fraction of `n_images`
        (an excluded image counts as not corroborated, so this is directly
        comparable to the raw `detection_rate`).
    """
    baseline = detections.loc[detections["sigma"] == 0, ["path", "best_box"]].rename(
        columns={"best_box": "sigma0_box"}
    )
    merged = detections.merge(baseline, on="path", how="left", validate="many_to_one")

    def _row_corroborated(row) -> bool:
        if row["best_box"] is None:
            return False
        if row["sigma"] == 0:
            return True
        if row["sigma0_box"] is None:
            return False
        return _iou(row["best_box"], row["sigma0_box"]) >= iou_threshold

    merged["corroborated"] = merged.apply(_row_corroborated, axis=1)
    merged["no_sigma0_detection"] = merged["sigma0_box"].isna()

    grouped = merged.groupby(["sigma", "class_label"])
    result = grouped.agg(
        n_images=("path", "size"),
        n_no_sigma0_detection=("no_sigma0_detection", "sum"),
        corroborated_rate=("corroborated", "mean"),
    ).reset_index()
    result["n_eligible"] = result["n_images"] - result["n_no_sigma0_detection"]
    return result[
        ["sigma", "class_label", "n_images", "n_no_sigma0_detection", "n_eligible", "corroborated_rate"]
    ]
