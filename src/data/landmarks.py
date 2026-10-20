"""Canonical alignment and morphology/pose feature computation for facial landmarks.

Pure geometry: no image I/O, no detector calls. Everything here operates
on already-extracted `(K, 2)` pixel-coordinate landmark arrays (see
`src.models.landmark_detect`) and is exercised, deterministically, by the
invariance test in `experiments.face_landmarks.pipeline`.

**Alignment removes position, scale and in-plane rotation** before any
feature is computed: translate so the eye midpoint is the origin, rotate
so the inter-ocular axis is horizontal, scale so inter-ocular distance
(eye-centre to eye-centre) is 1. Every distance computed downstream is
therefore already "in units of inter-ocular distance" by construction --
no separate normalisation step is needed or applied.

**MediaPipe point indices** below were not guessed from a diagram: they
were confirmed present in `mediapipe.tasks.python.vision
.FaceLandmarksConnections`'s official per-region connection sets
(`FACE_LANDMARKS_LEFT_EYE`, `RIGHT_EYE`, `LIPS`, `NOSE`, `FACE_OVAL`) for
the installed `mediapipe==1.0.1` package, then a specific point within
each region was picked for its commonly-documented anatomical role
(e.g. 33/133 as the right eye's outer/inner canthus, 61/291 as the mouth
corners). `MP_*_REGION` tuples are copied from those same official
connection sets and used for the two eyes' centroids at alignment time.

**Eye "centre" for alignment** is the midpoint of each eye's outer and
inner canthus (corner) points, not an iris/pupil position -- MediaPipe's
478-point output includes iris landmarks, but using them would require
resolving which of the two iris indices is which eye, and corner-based
centres are simpler, symmetric with the MTCNN cross-check (whose 5-point
"eye" landmarks are themselves an estimated pupil-ish position, not a
corner), and already well-defined from indices confirmed above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# ---------------------------------------------------------------------------
# MediaPipe 468/478-point mesh: single-point indices, confirmed members of
# the official named region connection sets (see module docstring).
# ---------------------------------------------------------------------------
MP_RIGHT_EYE_OUTER = 33
MP_RIGHT_EYE_INNER = 133
MP_LEFT_EYE_OUTER = 263
MP_LEFT_EYE_INNER = 362
MP_RIGHT_EYE_UPPER_LID = 159
MP_LEFT_EYE_UPPER_LID = 386
MP_NOSE_TIP = 1
MP_NOSE_BOTTOM = 2
MP_NOSE_ALA_RIGHT = 48
MP_NOSE_ALA_LEFT = 278
MP_MOUTH_CORNER_RIGHT = 61
MP_MOUTH_CORNER_LEFT = 291
MP_UPPER_LIP_TOP = 0
MP_LOWER_LIP_BOTTOM = 17
MP_CHEEK_RIGHT = 234
MP_CHEEK_LEFT = 454

# Full eye contour (upper + lower lid), from FACE_LANDMARKS_RIGHT_EYE /
# FACE_LANDMARKS_LEFT_EYE -- used only to sanity-check that the two named
# corner points above are self-consistent (see verify_index_constants).
MP_RIGHT_EYE_REGION: Tuple[int, ...] = (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246)
MP_LEFT_EYE_REGION: Tuple[int, ...] = (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466)

# (right_index, left_index) homologous pairs used for the asymmetry feature.
MP_MIRROR_PAIRS: Tuple[Tuple[int, int], ...] = (
    (MP_RIGHT_EYE_OUTER, MP_LEFT_EYE_OUTER),
    (MP_RIGHT_EYE_INNER, MP_LEFT_EYE_INNER),
    (MP_MOUTH_CORNER_RIGHT, MP_MOUTH_CORNER_LEFT),
    (MP_NOSE_ALA_RIGHT, MP_NOSE_ALA_LEFT),
    (MP_CHEEK_RIGHT, MP_CHEEK_LEFT),
)

# facenet-pytorch's MTCNN.detect(..., landmarks=True) point order.
MTCNN_LEFT_EYE, MTCNN_RIGHT_EYE, MTCNN_NOSE, MTCNN_MOUTH_LEFT, MTCNN_MOUTH_RIGHT = range(5)

MORPHOLOGY_FEATURE_NAMES: Tuple[str, ...] = (
    "fwhr",
    "mouth_width",
    "eye_to_mouth",
    "eye_to_nose",
    "nose_to_mouth",
    "nose_width",
    "philtrum_length",
    "palpebral_fissure_right",
    "palpebral_fissure_left",
    "intercanthal_over_iod",
    "asymmetry",
)

# The subset of MORPHOLOGY_FEATURE_NAMES the 5-point MTCNN cross-check can
# support -- no eye corners, nose ala, cheek contour or lip-vermillion
# points are available from 5 points, so fWHR, nose_width, philtrum_length,
# palpebral fissures, intercanthal distance and asymmetry cannot be
# computed from MTCNN's landmarks.
MTCNN_MORPHOLOGY_FEATURE_NAMES: Tuple[str, ...] = (
    "mouth_width",
    "eye_to_mouth",
    "eye_to_nose",
    "nose_to_mouth",
)

POSE_FEATURE_NAMES: Tuple[str, ...] = ("roll_deg", "yaw_proxy", "pitch_proxy")


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _midpoint(a: Sequence[float], b: Sequence[float]) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


@dataclass(frozen=True)
class AlignmentResult:
    """Output of `similarity_align`.

    Attributes:
        aligned: `(K, 2)` float64 array, eye midpoint at the origin,
            inter-ocular axis horizontal, inter-ocular distance 1.
        roll_deg: Pre-alignment roll -- the angle of the inter-ocular axis
            from horizontal, in degrees, in the *input* (unaligned) frame.
            Reported as a pose feature precisely because alignment removes
            it from every other measurement.
        inter_ocular_px: Pre-alignment inter-ocular distance, in the
            input's own pixel units. The scale factor applied is
            `1 / inter_ocular_px`.
        eye_midpoint_px: Pre-alignment eye midpoint, in the input's own
            pixel coordinates. The translation applied is `-eye_midpoint_px`.
    """

    aligned: np.ndarray
    roll_deg: float
    inter_ocular_px: float
    eye_midpoint_px: Point


def similarity_align(points: np.ndarray, left_eye_xy: Sequence[float], right_eye_xy: Sequence[float]) -> AlignmentResult:
    """Map `points` into the canonical frame defined by a left/right eye position.

    Translate so the eye midpoint is the origin, rotate so the
    `left_eye_xy -> right_eye_xy` axis is horizontal (right eye ends up at
    positive x), scale so the distance between the two eye points is
    exactly 1. A pure similarity transform (no reflection, no shear), so
    every distance and angle *between* points is preserved up to the
    single scale factor -- which is exactly what makes every feature
    computed downstream invariant to the face's original position, size
    and in-plane rotation.

    Args:
        points: `(K, 2)` array of pixel coordinates to transform.
        left_eye_xy: `(x, y)` of the left eye reference point, in the same
            pixel coordinates as `points`.
        right_eye_xy: `(x, y)` of the right eye reference point.

    Returns:
        See `AlignmentResult`.
    """
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

    centred = points - eye_mid
    rotated = centred @ rotation.T
    aligned = rotated / inter_ocular

    return AlignmentResult(
        aligned=aligned,
        roll_deg=roll_deg,
        inter_ocular_px=inter_ocular,
        eye_midpoint_px=(float(eye_mid[0]), float(eye_mid[1])),
    )


def align_mediapipe(points: np.ndarray) -> AlignmentResult:
    """`similarity_align` using MediaPipe's eye-corner-midpoint eye positions.

    MediaPipe's `MP_RIGHT_EYE_*` / `MP_LEFT_EYE_*` naming is *anatomical*
    (the subject's own right/left eye), which is the mirror image of
    *frame*-relative left/right for an ordinary forward-facing, unmirrored
    photo: the subject's right eye is the one that appears on the left
    side of the frame. `similarity_align`'s `right_eye_xy` argument is
    frame-relative (it is expected to sit at the larger x, so that
    `roll_deg` comes out near 0 for an upright face) -- so the anatomical
    *right* eye (`MP_RIGHT_EYE_*`, frame-left) is passed as `left_eye_xy`
    here, and the anatomical *left* eye (`MP_LEFT_EYE_*`, frame-right) as
    `right_eye_xy`. Getting this backwards does not change any symmetric
    distance/ratio in the morphology set, but it does put every image's
    `roll_deg` near +/-180 instead of near 0, which is confusing to read
    and, unlike the morphology features, wrong to leave that way.

    Args:
        points: `(478, 2)` (or `(468, 2)`) pixel-coordinate mesh landmarks
            for one face.

    Returns:
        See `AlignmentResult`.
    """
    frame_left_eye = _midpoint(points[MP_RIGHT_EYE_OUTER], points[MP_RIGHT_EYE_INNER])
    frame_right_eye = _midpoint(points[MP_LEFT_EYE_OUTER], points[MP_LEFT_EYE_INNER])
    return similarity_align(points, frame_left_eye, frame_right_eye)


def align_mtcnn(points: np.ndarray) -> AlignmentResult:
    """`similarity_align` using MTCNN's native 5-point eye positions.

    Args:
        points: `(5, 2)` pixel-coordinate landmarks for one face, in MTCNN
            order (`left_eye, right_eye, nose, mouth_left, mouth_right`).

    Returns:
        See `AlignmentResult`.
    """
    return similarity_align(points, points[MTCNN_LEFT_EYE], points[MTCNN_RIGHT_EYE])


def morphology_features_mediapipe(aligned: np.ndarray) -> Dict[str, float]:
    """The full morphology set, from aligned MediaPipe mesh coordinates.

    Args:
        aligned: `AlignmentResult.aligned` from `align_mediapipe` -- `(478,
            2)` (or `(468, 2)`), canonical frame.

    Returns:
        Dict with one key per `MORPHOLOGY_FEATURE_NAMES` entry.
    """
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

    asymmetry = float(
        np.mean([_dist((-p[r][0], p[r][1]), p[l]) for r, l in MP_MIRROR_PAIRS])
    )

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
    }


def morphology_features_mtcnn(aligned: np.ndarray) -> Dict[str, float]:
    """The 4-feature cross-check subset, from aligned MTCNN coordinates.

    Args:
        aligned: `AlignmentResult.aligned` from `align_mtcnn` -- `(5, 2)`,
            canonical frame.

    Returns:
        Dict with one key per `MTCNN_MORPHOLOGY_FEATURE_NAMES` entry.
    """
    nose = aligned[MTCNN_NOSE]
    mouth_left, mouth_right = aligned[MTCNN_MOUTH_LEFT], aligned[MTCNN_MOUTH_RIGHT]
    mouth_centre = _midpoint(mouth_left, mouth_right)

    return {
        "mouth_width": _dist(mouth_left, mouth_right),
        "eye_to_mouth": _dist((0.0, 0.0), mouth_centre),
        "eye_to_nose": _dist((0.0, 0.0), nose),
        "nose_to_mouth": _dist(nose, mouth_centre),
    }


def pose_features_mediapipe(raw: np.ndarray, alignment: AlignmentResult) -> Dict[str, float]:
    """The pose set, from raw + aligned MediaPipe mesh coordinates.

    Args:
        raw: `(478, 2)` pixel-coordinate mesh landmarks, pre-alignment
            (unused directly -- `alignment.roll_deg` already carries the
            only pre-alignment quantity this set needs -- kept as a
            parameter so the MediaPipe and MTCNN versions share a
            signature).
        alignment: Output of `align_mediapipe` for the same face.

    Returns:
        Dict with one key per `POSE_FEATURE_NAMES` entry.
    """
    del raw  # roll_deg (the only pre-alignment quantity used) is already on `alignment`.
    p = alignment.aligned
    nose_tip = p[MP_NOSE_TIP]
    mouth_centre = _midpoint(p[MP_MOUTH_CORNER_RIGHT], p[MP_MOUTH_CORNER_LEFT])
    return _pose_from_aligned(alignment.roll_deg, nose_tip, mouth_centre)


def pose_features_mtcnn(raw: np.ndarray, alignment: AlignmentResult) -> Dict[str, float]:
    """The pose set, from raw + aligned MTCNN coordinates.

    Args:
        raw: Unused (see `pose_features_mediapipe`) -- kept for a matching
            signature.
        alignment: Output of `align_mtcnn` for the same face.

    Returns:
        Dict with one key per `POSE_FEATURE_NAMES` entry.
    """
    del raw
    p = alignment.aligned
    nose = p[MTCNN_NOSE]
    mouth_centre = _midpoint(p[MTCNN_MOUTH_LEFT], p[MTCNN_MOUTH_RIGHT])
    return _pose_from_aligned(alignment.roll_deg, nose, mouth_centre)


def _pose_from_aligned(roll_deg: float, nose: Sequence[float], mouth_centre: Sequence[float]) -> Dict[str, float]:
    """Shared pose-feature math, given a roll angle and two aligned-frame points.

    `yaw_proxy` is the nose's horizontal offset from the eye midline (x=0
    in the aligned frame) -- computed post-alignment so it is not
    contaminated by roll. `pitch_proxy` is the nose's vertical position
    along the eye-to-mouth axis, as a fraction of the eye-to-mouth span
    (0 = at the eye line, 1 = at the mouth line) -- dimensionless, so it
    does not scale with face size.
    """
    yaw_proxy = float(nose[0])
    pitch_proxy = float(nose[1] / mouth_centre[1]) if mouth_centre[1] != 0 else float("nan")
    return {"roll_deg": roll_deg, "yaw_proxy": yaw_proxy, "pitch_proxy": pitch_proxy}


def apply_synthetic_transform(points: np.ndarray, scale: float, rotate_deg: float, translate: Sequence[float]) -> np.ndarray:
    """Scale, then rotate, then translate `points` -- for the Part 2 invariance test.

    Args:
        points: `(K, 2)` pixel coordinates.
        scale: Uniform scale factor.
        rotate_deg: Rotation angle, degrees, applied about the origin.
        translate: `(dx, dy)` translation, applied last.

    Returns:
        `(K, 2)` transformed coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    theta = np.radians(rotate_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return (points * scale) @ rotation.T + np.asarray(translate, dtype=np.float64)
