"""Radix-2 FFT, windowed power spectrum, windowed-sinc FIR filter and SNR."""

from __future__ import annotations

import numpy as np

from .signals import hann_window


def fft_radix2(x) -> np.ndarray:
    """Iterative radix-2 decimation-in-time FFT (length must be a power of 2)."""
    a = np.asarray(x, dtype=complex)
    if a.ndim != 1:
        raise ValueError("fft input must be 1-D")
    n = a.size
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("fft length must be a power of two")
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    a = a[rev]
    size = 2
    while size <= n:
        half = size // 2
        tw = np.exp(-2j * np.pi * np.arange(half) / size)
        blocks = a.reshape(-1, size)
        even = blocks[:, :half]
        odd = blocks[:, half:] * tw
        a = np.concatenate([even + odd, even - odd], axis=1).reshape(-1)
        size *= 2
    return a


def power_spectrum(x, window: str = "hann") -> np.ndarray:
    """One-sided power spectrum ``|X[k]|^2`` for bins ``0 .. n/2`` (inclusive).

    ``window`` is ``"hann"`` or ``"rect"``. Length must be a power of two.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("signal must be 1-D")
    if window == "hann":
        arr = arr * hann_window(arr.size, periodic=True)
    elif window != "rect":
        raise ValueError("window must be 'hann' or 'rect'")
    spec = fft_radix2(arr)
    half = arr.size // 2
    return np.abs(spec[: half + 1]) ** 2


def design_lowpass_fir(num_taps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """Linear-phase Hann-windowed-sinc low-pass FIR with unity DC gain."""
    if not isinstance(num_taps, (int, np.integer)) or num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer >= 3")
    if not fs > 0:
        raise ValueError("fs must be positive")
    if not (0.0 < cutoff_hz < fs / 2.0):
        raise ValueError("cutoff_hz must lie in (0, fs/2)")
    fc = cutoff_hz / fs
    m = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * m)
    h = h * hann_window(num_taps, periodic=False)
    return h / h.sum()


def apply_fir(x, h) -> np.ndarray:
    """Zero-delay filtering (odd-length symmetric ``h``), output same length as ``x``."""
    sig = np.asarray(x, dtype=float)
    taps = np.asarray(h, dtype=float)
    if sig.ndim != 1 or taps.ndim != 1:
        raise ValueError("signal and taps must be 1-D")
    if taps.size == 0 or taps.size % 2 == 0:
        raise ValueError("taps must have odd, non-zero length")
    if taps.size > sig.size:
        raise ValueError("filter is longer than the signal")
    return np.convolve(sig, taps, mode="same")


def snr_db(reference, estimate, trim: int = 0) -> float:
    """SNR of ``estimate`` against a known clean ``reference``, in dB.

    ``trim`` samples are dropped at both ends (to exclude filter edge transients).
    """
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape or ref.ndim != 1:
        raise ValueError("reference and estimate must be 1-D and the same length")
    if trim < 0 or 2 * trim >= ref.size:
        raise ValueError("trim out of range")
    if trim:
        ref = ref[trim:-trim]
        est = est[trim:-trim]
    sig_power = float(np.sum(ref**2))
    if sig_power == 0.0:
        raise ValueError("reference has zero power")
    err_power = float(np.sum((est - ref) ** 2))
    if err_power == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(sig_power / err_power))
