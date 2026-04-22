"""Geometric feature calculation from 468 face mesh landmarks.

All functions take landmarks as a list of (x, y, z) tuples / lists,
length 468, with coordinates normalized to [0, 1] by MediaPipe.

Output: 5-dimensional feature vector
    [EAR, MAR, eyebrow_eye_dist, mouth_curvature, nasolabial_depth]
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from app.utils.landmarks import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_INDICES,
    LEFT_EYEBROW_INDICES,
    RIGHT_EYEBROW_INDICES,
    NOSE_TIP_INDEX,
    LEFT_EYE_CENTER_INDEX,
    RIGHT_EYE_CENTER_INDEX,
)

Landmarks = Sequence[Sequence[float]]  # shape (468, 3)


def _dist(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_landmarks(landmarks: Landmarks) -> np.ndarray:
    """Normalize landmarks: origin = midpoint of two eye centers, scale = inter-eye distance."""
    pts = np.array(landmarks, dtype=np.float64)
    left_center = pts[LEFT_EYE_CENTER_INDEX]
    right_center = pts[RIGHT_EYE_CENTER_INDEX]
    origin = (left_center + right_center) / 2.0
    scale = np.linalg.norm(left_center - right_center)
    if scale < 1e-6:
        scale = 1e-6
    return (pts - origin) / scale


def calc_ear(landmarks: Landmarks) -> float:
    """Eye Aspect Ratio — average of left and right eyes.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    def _ear_single(indices):
        p = [landmarks[i] for i in indices]
        vertical1 = _dist(p[1], p[5])
        vertical2 = _dist(p[2], p[4])
        horizontal = _dist(p[0], p[3])
        if horizontal < 1e-6:
            return 0.0
        return (vertical1 + vertical2) / (2.0 * horizontal)

    left_ear = _ear_single(LEFT_EYE_INDICES)
    right_ear = _ear_single(RIGHT_EYE_INDICES)
    return (left_ear + right_ear) / 2.0


def calc_mar(landmarks: Landmarks) -> float:
    """Mouth Aspect Ratio.

    MAR = (|top-bottom|) / |left-right|
    Using MOUTH_INDICES: [left, right, top, bottom, inner_top, inner_bottom]
    """
    left = landmarks[MOUTH_INDICES[0]]
    right = landmarks[MOUTH_INDICES[1]]
    top = landmarks[MOUTH_INDICES[2]]
    bottom = landmarks[MOUTH_INDICES[3]]
    horizontal = _dist(left, right)
    vertical = _dist(top, bottom)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def calc_eyebrow_eye_dist(landmarks: Landmarks) -> float:
    """Average vertical distance between eyebrow center and eye center (normalized)."""
    left_brow_y = np.mean([landmarks[i][1] for i in LEFT_EYEBROW_INDICES])
    right_brow_y = np.mean([landmarks[i][1] for i in RIGHT_EYEBROW_INDICES])
    left_eye_y = landmarks[LEFT_EYE_CENTER_INDEX][1]
    right_eye_y = landmarks[RIGHT_EYE_CENTER_INDEX][1]
    left_dist = abs(left_brow_y - left_eye_y)
    right_dist = abs(right_brow_y - right_eye_y)
    return (left_dist + right_dist) / 2.0


def calc_mouth_curvature(landmarks: Landmarks) -> float:
    """Mouth corner curvature — how much the corners are raised relative to the center.

    Positive = smile, negative = frown.
    """
    left = landmarks[MOUTH_INDICES[0]]
    right = landmarks[MOUTH_INDICES[1]]
    mid_y = (left[1] + right[1]) / 2.0
    top = landmarks[MOUTH_INDICES[2]]
    bottom = landmarks[MOUTH_INDICES[3]]
    center_y = (top[1] + bottom[1]) / 2.0
    return center_y - mid_y


def calc_nasolabial_depth(landmarks: Landmarks) -> float:
    """Nasolabial fold depth — distance from nose tip to mouth center line."""
    nose = landmarks[NOSE_TIP_INDEX]
    top = landmarks[MOUTH_INDICES[2]]
    bottom = landmarks[MOUTH_INDICES[3]]
    mouth_center_y = (top[1] + bottom[1]) / 2.0
    mouth_center_x = (top[0] + bottom[0]) / 2.0
    return _dist(nose, (mouth_center_x, mouth_center_y))


def extract_geo_features(landmarks: Landmarks) -> list[float]:
    """Extract the 5-dimensional geometric feature vector from 468 landmarks."""
    return [
        calc_ear(landmarks),
        calc_mar(landmarks),
        calc_eyebrow_eye_dist(landmarks),
        calc_mouth_curvature(landmarks),
        calc_nasolabial_depth(landmarks),
    ]
