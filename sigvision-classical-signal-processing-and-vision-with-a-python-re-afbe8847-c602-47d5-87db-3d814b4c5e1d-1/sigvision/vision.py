"""Classical 2-D vision on synthetic images: convolution, Sobel edges, Harris corners."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Mathematical *convolution* kernels (applied with kernel flipping). With the
# flip convention used by ``conv2d`` these give a positive response when the
# intensity increases towards +x (gx) or towards +y / downwards (gy).
SOBEL_X = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
SOBEL_Y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


def _as_2d(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    return arr


def conv2d(image, kernel) -> np.ndarray:
    """'Same'-size 2-D convolution with zero padding and an odd-sized kernel.

    ``out[y, x] = sum_{i,j} K[i, j] * I[y + ci - i, x + cj - j]`` with
    ``ci = kh // 2`` and ``cj = kw // 2``; samples outside the image are zero.
    """
    img = _as_2d(image, "image")
    ker = _as_2d(kernel, "kernel")
    kh, kw = ker.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    h, w = img.shape
    ci, cj = kh // 2, kw // 2
    padded = np.pad(img, ((ci, ci), (cj, cj)))
    flipped = ker[::-1, ::-1]
    out = np.zeros((h, w))
    for a in range(kh):
        for b in range(kw):
            out += flipped[a, b] * padded[a : a + h, b : b + w]
    return out


def sobel_gradients(image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(gx, gy, magnitude)`` using the Sobel operator."""
    gx = conv2d(image, SOBEL_X)
    gy = conv2d(image, SOBEL_Y)
    return gx, gy, np.hypot(gx, gy)


def edge_map(image, threshold_ratio: float = 0.25) -> np.ndarray:
    """Boolean edge mask: Sobel magnitude >= ``threshold_ratio`` * max magnitude."""
    if not (0.0 < threshold_ratio <= 1.0):
        raise ValueError("threshold_ratio must lie in (0, 1]")
    mag = sobel_gradients(image)[2]
    peak = float(mag.max())
    if peak <= 0.0:
        return np.zeros(mag.shape, dtype=bool)
    return mag >= threshold_ratio * peak


def harris_response(image, k: float = 0.04, window: int = 5) -> np.ndarray:
    """Harris corner response ``det(M) - k * trace(M)^2`` with a box window."""
    if not k > 0:
        raise ValueError("k must be positive")
    gx, gy, _ = sobel_gradients(image)
    box = np.ones((window, window))
    sxx = conv2d(gx * gx, box)
    syy = conv2d(gy * gy, box)
    sxy = conv2d(gx * gy, box)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return det - k * trace * trace


def harris_corners(
    image,
    k: float = 0.04,
    window: int = 5,
    rel_threshold: float = 0.05,
    min_distance: int = 5,
) -> List[Tuple[int, int]]:
    """Corner ``(row, col)`` positions: thresholded 3x3 maxima with distance NMS.

    Candidates must exceed ``rel_threshold * max(R)``; they are visited in
    descending response order (ties by row, col) and suppressed when within
    ``min_distance`` (Chebyshev) of an already accepted corner.
    """
    if not (0.0 < rel_threshold < 1.0):
        raise ValueError("rel_threshold must lie in (0, 1)")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    resp = harris_response(image, k=k, window=window)
    rmax = float(resp.max())
    if not rmax > 0.0:
        return []
    h, w = resp.shape
    padded = np.pad(resp, 1, constant_values=-np.inf)
    is_max = np.ones(resp.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            is_max &= resp >= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    candidates = [
        (int(r), int(c)) for r, c in np.argwhere(is_max & (resp > rel_threshold * rmax))
    ]
    candidates.sort(key=lambda rc: (-resp[rc[0], rc[1]], rc[0], rc[1]))
    kept: List[Tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - kr), abs(c - kc)) > min_distance for kr, kc in kept):
            kept.append((r, c))
    return kept


Rect = Tuple[int, int, int, int, float]  # (row0, col0, row1_exclusive, col1_exclusive, value)


def make_scene(
    shape: Tuple[int, int],
    rects: Sequence[Rect],
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic image: zero background, filled rectangles, optional Gaussian noise."""
    h, w = shape
    if h < 1 or w < 1:
        raise ValueError("shape must be positive")
    img = np.zeros((h, w))
    for r0, c0, r1, c1, value in rects:
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            raise ValueError(f"rectangle {(r0, c0, r1, c1)} outside image {shape}")
        img[r0:r1, c0:c1] = value
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise_std, img.shape)
    return img


def rect_corners(rects: Sequence[Rect]) -> List[Tuple[int, int]]:
    """Ground-truth corner pixels (the four extreme pixels) of each rectangle."""
    corners: List[Tuple[int, int]] = []
    for r0, c0, r1, c1, _ in rects:
        corners.extend([(r0, c0), (r0, c1 - 1), (r1 - 1, c0), (r1 - 1, c1 - 1)])
    return corners
