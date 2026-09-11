"""Facial landmark detection for the face_landmarks experiment.

Two independent detectors, run on the same post-geometry 224x224 image
every other experiment's transform pipeline produces:

- **MediaPipe Face Mesh** (`mediapipe.tasks.python.vision.FaceLandmarker`),
  the primary source: 478 points (468 mesh + 10 iris), dense enough to
  support the morphology feature set in `src.data.landmarks`.
- **MTCNN** (`facenet-pytorch`), the cross-check: 5 points (eyes, nose,
  mouth corners), the same detector the rest of this project already
  relies on, extended here with `landmarks=True`.

Both return `(points, success)`: `points` is `(N, K, 2)` float32 in pixel
coordinates of the 224x224 input (`NaN`-filled for any image where no
face was found), `success` is an `(N,)` boolean array. Kept symmetric on
purpose, so calling code does not need to special-case which detector
produced a given array.
"""

from __future__ import annotations

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# The float16 "latest" MediaPipe Face Landmarker bundle -- Google's own
# hosted model asset, documented at
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker.
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

MEDIAPIPE_N_POINTS = 478
MTCNN_N_POINTS = 5

# facenet-pytorch's MTCNN.detect(..., landmarks=True) point order.
MTCNN_LEFT_EYE, MTCNN_RIGHT_EYE, MTCNN_NOSE, MTCNN_MOUTH_LEFT, MTCNN_MOUTH_RIGHT = range(5)


def download_face_landmarker_model(dest_path: Path) -> Path:
    """Download the MediaPipe Face Landmarker model bundle if not already cached.

    Args:
        dest_path: Where to save the `.task` model bundle.

    Returns:
        `dest_path`, unchanged -- for chaining.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        logger.info("download_face_landmarker_model: using cached model at %s", dest_path)
        return dest_path

    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("download_face_landmarker_model: downloading from %s", FACE_LANDMARKER_MODEL_URL)
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, dest_path)
    logger.info("download_face_landmarker_model: saved %d bytes to %s", dest_path.stat().st_size, dest_path)
    return dest_path


def detect_landmarks_mediapipe(
    images: Sequence[Image.Image], model_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect the 478-point MediaPipe Face Mesh landmarks for each image.

    Runs `FaceLandmarker` in `IMAGE` mode (each call independent, no
    temporal state) with `num_faces=1` -- this experiment only ever wants
    the single primary face per image, matching MTCNN's `keep_all=True` +
    best-score selection used elsewhere in this project.

    Args:
        images: RGB PIL images, already at the target size (post-geometry,
            pre-normalisation). Landmarks are returned in this image's
            own pixel coordinates.
        model_path: Path to the `.task` model bundle (see
            `download_face_landmarker_model`).

    Returns:
        `(points, success)`: `points` is `(len(images), 478, 2)` float32,
        pixel coordinates, `NaN`-filled for any image with no detected
        face; `success` is a `(len(images),)` boolean array.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)

    points = np.full((len(images), MEDIAPIPE_N_POINTS, 2), np.nan, dtype=np.float32)
    success = np.zeros(len(images), dtype=bool)

    with vision.FaceLandmarker.create_from_options(options) as detector:
        for i, image in enumerate(images):
            rgb = image.convert("RGB")
            width, height = rgb.size
            array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
            result = detector.detect(mp_image)

            if not result.face_landmarks:
                continue
            landmarks = result.face_landmarks[0]
            if len(landmarks) != MEDIAPIPE_N_POINTS:
                logger.warning(
                    "detect_landmarks_mediapipe: image %d returned %d points, expected %d -- skipping",
                    i,
                    len(landmarks),
                    MEDIAPIPE_N_POINTS,
                )
                continue
            points[i, :, 0] = [lm.x * width for lm in landmarks]
            points[i, :, 1] = [lm.y * height for lm in landmarks]
            success[i] = True

    logger.info(
        "detect_landmarks_mediapipe: %d/%d images with a detected face", int(success.sum()), len(images)
    )
    return points, success


def detect_landmarks_mtcnn(images: Sequence[Image.Image], device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Detect the 5-point MTCNN landmarks (eyes, nose, mouth corners) for each image.

    The highest-probability face is kept per image, matching
    `src.models.face_detect.detect_faces_mtcnn`'s selection rule --
    except this call additionally requests `landmarks=True`, which that
    function does not.

    Args:
        images: RGB PIL images, already at the target size (post-geometry,
            pre-normalisation).
        device: Torch device string, e.g. `"cuda"` or `"cpu"`.

    Returns:
        `(points, success)`: `points` is `(len(images), 5, 2)` float32,
        pixel coordinates, in the order `(left_eye, right_eye, nose,
        mouth_left, mouth_right)`, `NaN`-filled for any image with no
        detected face; `success` is a `(len(images),)` boolean array.
    """
    from facenet_pytorch import MTCNN

    try:
        mtcnn_version = version("facenet-pytorch")
    except PackageNotFoundError:
        mtcnn_version = "unknown"
    logger.info("detect_landmarks_mtcnn: using MTCNN (facenet-pytorch %s) on %s", mtcnn_version, device)

    detector = MTCNN(keep_all=True, device=device)
    points = np.full((len(images), MTCNN_N_POINTS, 2), np.nan, dtype=np.float32)
    success = np.zeros(len(images), dtype=bool)

    for i, image in enumerate(images):
        boxes, probs, landmarks = detector.detect(image, landmarks=True)
        if boxes is None:
            continue
        best_idx = int(np.argmax(probs))
        points[i] = landmarks[best_idx].astype(np.float32)
        success[i] = True

    logger.info("detect_landmarks_mtcnn: %d/%d images with a detected face", int(success.sum()), len(images))
    return points, success


def save_landmarks(
    points: np.ndarray, success: np.ndarray, paths: Sequence[Path], meta: Dict, out_dir: Path, name: str
) -> None:
    """Persist a landmark array, its success mask, row-order path index, and metadata.

    Writes `{name}.npy` (the `(N, K, 2)` array), `{name}_index.csv` (`path`
    + `success`, in array-row order), and `{name}_meta.json` -- the same
    three-file convention `src.models.embeddings.save_embeddings` uses.

    Args:
        points: `(N, K, 2)` float array, e.g. from `detect_landmarks_mediapipe`.
        success: `(N,)` boolean array, row-aligned to `points`.
        paths: Image paths, same row order as `points`.
        meta: Small JSON-serialisable provenance dict.
        out_dir: Directory to write into (created if missing).
        name: Filename stem for the three written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{name}.npy", points)
    pd.DataFrame({"path": [str(p) for p in paths], "success": success}).to_csv(
        out_dir / f"{name}_index.csv", index=False
    )
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_landmarks(out_dir: Path, name: str) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """Load a previously saved landmark array, its index, and metadata.

    Args:
        out_dir: Directory previously written by `save_landmarks`.
        name: Filename stem passed to the original `save_landmarks` call.

    Returns:
        `(points, index, meta)`.
    """
    out_dir = Path(out_dir)
    points = np.load(out_dir / f"{name}.npy")
    index = pd.read_csv(out_dir / f"{name}_index.csv")
    meta = json.loads((out_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
    return points, index, meta
