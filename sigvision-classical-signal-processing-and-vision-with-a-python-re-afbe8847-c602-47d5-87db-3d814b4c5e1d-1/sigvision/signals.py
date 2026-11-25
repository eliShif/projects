"""Seeded synthetic 1-D signal generation and window functions."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def hann_window(n: int, periodic: bool = True) -> np.ndarray:
    """Hann window of length ``n``.

    ``periodic=True`` (default) is the DFT-even form used for spectral
    analysis; ``periodic=False`` is the symmetric form used for FIR design.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("window length must be a positive integer")
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    denom = n if periodic else n - 1
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / denom)


def make_signal(
    n: int,
    fs: float,
    tones: Iterable[Tuple[float, float]],
    noise_std: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(noisy, clean)`` sums of sinusoids with optional Gaussian noise.

    ``tones`` is an iterable of ``(frequency_hz, amplitude)``. Every frequency
    must lie in ``[0, fs/2)``. The same ``seed`` always yields the same noise.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not fs > 0:
        raise ValueError("fs must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    t = np.arange(n) / fs
    clean = np.zeros(n)
    for freq, amp in tones:
        if not (0.0 <= freq < fs / 2.0):
            raise ValueError(f"tone frequency {freq} outside [0, fs/2)")
        clean += amp * np.sin(2.0 * np.pi * freq * t)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        noisy = clean + rng.normal(0.0, noise_std, n)
    else:
        noisy = clean.copy()
    return noisy, clean
