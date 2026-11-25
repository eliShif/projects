"""Matching of detections against synthetic ground truth."""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def match_scalars(truth: Sequence[float], detected: Sequence[float], tol: float) -> Tuple[int, int]:
    """Return ``(matched_truth, false_detections)`` for 1-D positions within ``tol``."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    matched = sum(1 for t in truth if any(abs(t - d) <= tol for d in detected))
    false = sum(1 for d in detected if not any(abs(t - d) <= tol for t in truth))
    return matched, false


def match_points(
    truth: Sequence[Tuple[float, float]],
    detected: Sequence[Tuple[float, float]],
    tol: float,
) -> Tuple[int, int]:
    """Return ``(matched_truth, false_detections)`` for 2-D points (Euclidean ``tol``)."""
    if tol < 0:
        raise ValueError("tol must be non-negative")

    def near(p, q) -> bool:
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= tol

    matched = sum(1 for t in truth if any(near(t, d) for d in detected))
    false = sum(1 for d in detected if not any(near(t, d) for t in truth))
    return matched, false
