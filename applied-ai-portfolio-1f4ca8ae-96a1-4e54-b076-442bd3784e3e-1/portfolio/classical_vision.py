"""Classical image and signal processing on small synthetic data.

This module is the NumPy reference for the C++ kernels in ``cpp/filters.cpp``
(see ``portfolio/classical_native.py``). It covers 2-D filtering, Sobel gradient
edges, FIR filtering, FFT/STFT spectra and a synthetic time-activity comparison
between a region of interest (ROI) and its surroundings. All data are generated
from a seed; nothing here uses driving, video or clinical data.
"""
from __future__ import annotations

import numpy as np

SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T.copy()


# ----------------------------------------------------------------- validation

def as_image(image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError('image must be a nonempty 2-D array')
    if not np.all(np.isfinite(array)):
        raise ValueError('image values must be finite')
    return array


def as_kernel(kernel) -> np.ndarray:
    array = as_image(kernel)
    if array.shape[0] % 2 == 0 or array.shape[1] % 2 == 0:
        raise ValueError('kernel dimensions must be odd')
    return array


def as_signal(signal) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError('signal must be a nonempty 1-D array')
    if not np.all(np.isfinite(array)):
        raise ValueError('signal values must be finite')
    return array


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def require_fft_length(n: int) -> None:
    if not is_power_of_two(n):
        raise ValueError('FFT length must be a positive power of two')


def require_stft_shape(length: int, frame: int, hop: int) -> None:
    if not is_power_of_two(frame):
        raise ValueError('frame length must be a positive power of two')
    if int(hop) != hop or hop < 1:
        raise ValueError('hop must be a positive integer')
    if frame > length:
        raise ValueError('frame length exceeds signal length')


# --------------------------------------------------------------- 2-D filtering

def correlate2d(image, kernel) -> np.ndarray:
    """Same-size cross-correlation with zero padding; ``kernel`` has odd dimensions."""
    img, ker = as_image(image), as_kernel(kernel)
    hr, hc = ker.shape[0] // 2, ker.shape[1] // 2
    rows, cols = img.shape
    padded = np.pad(img, ((hr, hr), (hc, hc)))
    out = np.zeros_like(img)
    for a in range(ker.shape[0]):
        for b in range(ker.shape[1]):
            out += ker[a, b] * padded[a:a + rows, b:b + cols]
    return out


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    if int(size) != size or size < 1 or size % 2 == 0:
        raise ValueError('size must be a positive odd integer')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    axis = np.arange(size) - size // 2
    g = np.exp(-(axis.astype(float) ** 2) / (2.0 * sigma * sigma))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def sobel_magnitude(image) -> np.ndarray:
    img = as_image(image)
    return np.hypot(correlate2d(img, SOBEL_X), correlate2d(img, SOBEL_Y))


def threshold_edges(magnitude, fraction: float = 0.4, floor: float = 0.0) -> np.ndarray:
    """Boolean edge map: magnitude >= fraction * max(magnitude).

    A peak at or below ``floor`` (default 0) means "no edges", which keeps rounding residue in
    flat regions from being scaled up into a spurious edge map.
    """
    mag = as_image(magnitude)
    if not 0.0 < fraction <= 1.0:
        raise ValueError('fraction must be in (0, 1]')
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError('floor must be finite and nonnegative')
    peak = mag.max()
    if peak <= floor:
        return np.zeros(mag.shape, dtype=bool)
    return mag >= fraction * peak


# ------------------------------------------------------------------- 1-D signals

def fir_filter(signal, taps) -> np.ndarray:
    """Causal FIR filter, y[n] = sum_k taps[k] * signal[n-k], zero initial state."""
    x, h = as_signal(signal), as_signal(taps)
    return np.convolve(x, h)[:x.size]


def lowpass_taps(num_taps: int = 63, cutoff: float = 0.0625) -> np.ndarray:
    """Hamming-windowed sinc low-pass; ``cutoff`` in cycles/sample (0 < cutoff < 0.5); unit DC gain."""
    if int(num_taps) != num_taps or num_taps < 3 or num_taps % 2 == 0:
        raise ValueError('num_taps must be an odd integer >= 3')
    if not 0.0 < cutoff < 0.5:
        raise ValueError('cutoff must be in (0, 0.5) cycles/sample')
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.hamming(num_taps)
    return taps / taps.sum()


def rfft(signal) -> np.ndarray:
    x = as_signal(signal)
    require_fft_length(x.size)
    return np.fft.rfft(x)


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window, identical to the C++ implementation."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft_magnitude(signal, frame: int, hop: int) -> np.ndarray:
    """Magnitude spectrogram, shape (frames, frame // 2 + 1); frames never run past the signal end."""
    x = as_signal(signal)
    require_stft_shape(x.size, frame, hop)
    window = hann_window(frame)
    frames = 1 + (x.size - frame) // hop
    return np.array([np.abs(np.fft.rfft(x[t * hop:t * hop + frame] * window)) for t in range(frames)])


class NumpyBackend:
    """Reference implementations behind the same interface as the native backend."""
    name = 'numpy'

    def correlate2d(self, image, kernel):
        return correlate2d(image, kernel)

    def sobel_magnitude(self, image):
        return sobel_magnitude(image)

    def fir_filter(self, signal, taps):
        return fir_filter(signal, taps)

    def rfft(self, signal):
        return rfft(signal)

    def stft_magnitude(self, signal, frame, hop):
        return stft_magnitude(signal, frame, hop)


NUMPY = NumpyBackend()


def dominant_frequency(signal, sample_rate: float, backend=NUMPY) -> float:
    """Frequency (Hz) of the largest non-DC FFT bin; the length must be a power of two >= 2."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('sample_rate must be positive and finite')
    x = as_signal(signal)
    if x.size < 2:
        raise ValueError('need at least two samples')
    magnitude = np.abs(backend.rfft(x))
    magnitude[0] = 0.0
    return float(np.argmax(magnitude)) * sample_rate / x.size


# ---------------------------------------------------------------- edge pipeline

def detect_edges(image, backend=NUMPY, blur_size: int = 5, blur_sigma: float = 1.0,
                 fraction: float = 0.4) -> np.ndarray:
    """Gaussian blur -> Sobel magnitude -> relative threshold. ``blur_size=1`` disables blurring.

    The image is edge-replicated before filtering and the added margin is cropped afterwards, so the
    zero padding of the filters creates no artificial gradient at the image border. A peak gradient
    below ``1e-9`` times the image scale counts as flat.
    """
    img = as_image(image)
    kernel = gaussian_kernel(blur_size, blur_sigma)
    margin = kernel.shape[0] // 2 + 1  # blur radius plus the 3x3 Sobel radius
    padded = np.pad(img, margin, mode='edge')
    magnitude = backend.sobel_magnitude(backend.correlate2d(padded, kernel))[margin:-margin, margin:-margin]
    return threshold_edges(magnitude, fraction, floor=1e-9 * max(1.0, float(np.abs(img).max())))


def boundary_mask(mask) -> np.ndarray:
    """Pixels of ``mask`` with at least one 4-neighbour outside it (image border counts as outside)."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0:
        raise ValueError('mask must be a nonempty 2-D array')
    p = np.pad(m, 1)
    eroded = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~eroded


def dilate(mask, radius: int) -> np.ndarray:
    """Chebyshev-distance dilation of a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    if int(radius) != radius or radius < 0:
        raise ValueError('radius must be a nonnegative integer')
    rows, cols = m.shape
    p = np.pad(m, radius)
    out = np.zeros_like(m)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= p[dy:dy + rows, dx:dx + cols]
    return out


def edge_scores(predicted, truth, tolerance: int = 2) -> dict:
    """Pixel precision/recall/F1 where a match lies within ``tolerance`` pixels (Chebyshev)."""
    pred, true = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError('predicted and truth must be 2-D arrays of the same shape')
    if not true.any():
        raise ValueError('truth contains no edge pixels')
    near_truth, near_pred = dilate(true, tolerance), dilate(pred, tolerance)
    precision = float((near_truth & pred).sum() / pred.sum()) if pred.any() else 0.0
    recall = float((near_pred & true).sum() / true.sum())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tolerance_px': int(tolerance),
            'predicted_pixels': int(pred.sum()), 'truth_pixels': int(true.sum())}


# ---------------------------------------------------------------- synthetic data

def synthetic_image(seed: int, size: int = 64, noise: float = 0.02):
    """A rectangle and a disc (intensity 0.9) on a 0.1 background plus Gaussian noise.

    Returns ``(image, region_mask)``; the mask is the noise-free ground truth.
    """
    if int(size) != size or size < 16:
        raise ValueError('size must be an integer >= 16')
    if not np.isfinite(noise) or noise < 0:
        raise ValueError('noise must be finite and nonnegative')
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 8:size // 2, size // 8:size // 2] = True
    center, radius = (3 * size) // 4, size // 6
    mask |= (yy - center) ** 2 + (xx - center) ** 2 <= radius * radius
    image = 0.1 + 0.8 * mask + noise * rng.standard_normal((size, size))
    return image, mask


def synthetic_signal(seed: int, n: int = 256, sample_rate: float = 256.0,
                     tones=((12.0, 1.0), (40.0, 0.5)), noise: float = 0.05) -> np.ndarray:
    """Sum of sine tones ``(frequency_hz, amplitude)`` plus Gaussian noise."""
    if int(n) != n or n < 2:
        raise ValueError('n must be an integer >= 2')
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('sample_rate must be positive and finite')
    if not np.isfinite(noise) or noise < 0:
        raise ValueError('noise must be finite and nonnegative')
    t = np.arange(n) / sample_rate
    x = np.zeros(n)
    for frequency, amplitude in tones:
        x += amplitude * np.sin(2.0 * np.pi * frequency * t)
    return x + noise * np.random.default_rng(seed).standard_normal(n)


def synthetic_activity_stack(seed: int, frames: int = 24, size: int = 32, roi_radius: int = 6,
                             peak_frame: float = 6.0, amplitude: float = 1.0,
                             background: float = 0.2, noise: float = 0.02):
    """Dynamic image stack ``(T, H, W)``: a disc ROI follows a rise-and-decay curve over a flat background.

    The ROI value is ``amplitude * (t/peak_frame) * exp(1 - t/peak_frame)`` (peak ``amplitude`` at
    ``peak_frame``). Returns ``(stack, roi_mask, roi_curve)``. A stand-in for time-activity analysis.
    """
    if frames < 2 or size < 8 or peak_frame <= 0 or noise < 0:
        raise ValueError('invalid stack parameters')
    if not 0 < roi_radius < size / 2:
        raise ValueError('roi_radius must be positive and fit inside the image')
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= roi_radius ** 2
    t = np.arange(frames, dtype=float)
    curve = amplitude * (t / peak_frame) * np.exp(1.0 - t / peak_frame)
    rng = np.random.default_rng(seed)
    stack = background + curve[:, None, None] * mask[None] + noise * rng.standard_normal((frames, size, size))
    return stack, mask, curve


def time_activity_curve(stack, roi_mask) -> dict:
    """Mean signal per frame inside and outside the ROI, and their difference."""
    data = np.asarray(stack, dtype=np.float64)
    mask = np.asarray(roi_mask, dtype=bool)
    if data.ndim != 3 or data.shape[0] < 1:
        raise ValueError('stack must have shape (frames, rows, cols)')
    if not np.all(np.isfinite(data)):
        raise ValueError('stack values must be finite')
    if mask.shape != data.shape[1:]:
        raise ValueError('roi_mask shape must match the stack frames')
    if not mask.any() or mask.all():
        raise ValueError('roi_mask must select some but not all pixels')
    inside, outside = data[:, mask].mean(axis=1), data[:, ~mask].mean(axis=1)
    return {'roi': inside, 'outside': outside, 'contrast': inside - outside}
