"""Cell-averaging CFAR (constant false-alarm rate) peak detection in 1-D.

The detector is generic: it works on any non-negative 1-D power profile and has
no dependence on a particular sensor. The arithmetic order (prefix sums, window
bounds, comparison rules) is mirrored exactly by ``cpp/src/sigcore.cpp`` so the
two implementations can be compared index-for-index.
"""

from __future__ import annotations

import numpy as np

from .spectral import power_spectrum


def cfar_alpha(pfa: float, num_train: int) -> float:
    """CA-CFAR threshold multiplier for exponential noise power.

    Uses ``N = 2 * num_train`` training cells: ``alpha = N * (pfa**(-1/N) - 1)``.
    """
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa must lie in (0, 1)")
    if not isinstance(num_train, (int, np.integer)) or num_train < 1:
        raise ValueError("num_train must be a positive integer")
    n = 2 * int(num_train)
    return n * (pfa ** (-1.0 / n) - 1.0)


def cfar_ca_detect(power, num_train: int, num_guard: int, alpha: float) -> np.ndarray:
    """Indices of CA-CFAR detections that are also strict-left local maxima.

    For cell ``i`` the noise level is the mean of the available training cells
    ``[i-guard-train, i-guard)`` and ``[i+guard+1, i+guard+train+1)`` clipped to
    the array (so edge cells use fewer cells; cells with none are skipped).
    A cell is reported when ``x[i] > alpha * noise``, ``x[i] > x[i-1]`` and
    ``x[i] >= x[i+1]`` (out-of-range neighbours count as ``-inf``).
    """
    x = np.asarray(power, dtype=float)
    if x.ndim != 1:
        raise ValueError("power must be 1-D")
    if not isinstance(num_train, (int, np.integer)) or num_train < 1:
        raise ValueError("num_train must be a positive integer")
    if not isinstance(num_guard, (int, np.integer)) or num_guard < 0:
        raise ValueError("num_guard must be a non-negative integer")
    if not (np.isfinite(alpha) and alpha > 0.0):
        raise ValueError("alpha must be a positive finite number")
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    train = int(num_train)
    guard = int(num_guard)

    cs = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lead_lo = np.clip(idx - guard - train, 0, n)
    lead_hi = np.clip(idx - guard, 0, n)
    lag_lo = np.clip(idx + guard + 1, 0, n)
    lag_hi = np.clip(idx + guard + train + 1, 0, n)
    counts = (lead_hi - lead_lo) + (lag_hi - lag_lo)
    sums = (cs[lead_hi] - cs[lead_lo]) + (cs[lag_hi] - cs[lag_lo])
    noise = sums / np.maximum(counts, 1)
    thr = alpha * noise

    neg_inf = -np.inf
    left = np.concatenate(([neg_inf], x[:-1]))
    right = np.concatenate((x[1:], [neg_inf]))
    mask = (counts > 0) & (x > thr) & (x > left) & (x >= right)
    return np.flatnonzero(mask).astype(np.int64)


def detect_tones(
    x,
    pfa: float = 1e-6,
    num_train: int = 16,
    num_guard: int = 4,
) -> np.ndarray:
    """Spectral bin indices of detected tones in a real signal.

    Hann-windowed power spectrum (length must be a power of two) followed by
    CA-CFAR. Convert bins to Hz with ``bin * fs / len(x)``.
    """
    spec = power_spectrum(x, window="hann")
    alpha = cfar_alpha(pfa, num_train)
    return cfar_ca_detect(spec, num_train, num_guard, alpha)


def bins_to_hz(bins, fs: float, n: int) -> np.ndarray:
    """Convert FFT bin indices to Hz for a length-``n`` transform at rate ``fs``."""
    if not fs > 0 or n < 1:
        raise ValueError("fs must be positive and n >= 1")
    return np.asarray(bins, dtype=float) * fs / n
