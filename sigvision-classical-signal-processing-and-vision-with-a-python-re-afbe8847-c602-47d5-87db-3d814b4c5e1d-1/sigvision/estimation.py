"""Statistical estimation on synthetic data, with explicit error analysis.

Three small estimation problems, each checked against a known truth or an
analytic reference so that bias, variance and RMSE can be reported:

1. **Noise power** - mean-square / sample-variance estimators of Gaussian noise
   power, with Monte Carlo bias, variance and RMSE against the analytic values.
   The same idea is applied to the CA-CFAR training-cell average
   (``monte_carlo_cfar_false_alarm``), reusing ``detection.cfar_alpha``.
2. **Tone frequency** - sub-bin frequency estimate by parabolic interpolation of
   the log power spectrum around the peak of a Hann-windowed FFT (reusing
   ``spectral.power_spectrum`` and ``detection.detect_tones``), compared with
   the Cramer-Rao lower bound for a real sinusoid in white Gaussian noise.
3. **Tracking** - a constant-velocity Kalman filter for a 2-D synthetic object
   track, scored with position RMSE against the raw measurements and with the
   normalised estimation error squared (NEES) for filter consistency.

Every number printed by the CLI is computed by that run on seeded synthetic
data. Nothing here describes real sensors, and the estimators are textbook
methods, not new algorithms.

Run: ``python -m sigvision.estimation --quick``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detection import cfar_alpha, detect_tones
from .spectral import power_spectrum

NOISE_MODES = ("zero_mean", "sample_unbiased", "sample_mle")
STATE_DIM = 4  # [x, y, vx, vy]


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _require_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _require_positive(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not (np.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a positive finite number")
    return v


def _require_nonnegative(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative finite number") from None
    if not (np.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be a non-negative finite number")
    return v


# --------------------------------------------------------------------------
# 1. noise power estimation
# --------------------------------------------------------------------------


def estimate_noise_variance(x, mode: str = "zero_mean") -> float:
    """Estimate the variance (power) of white Gaussian noise from samples.

    * ``zero_mean``: the mean is known to be 0, estimate ``mean(x**2)``
      (unbiased, variance ``2 sigma^4 / n``).
    * ``sample_unbiased``: unknown mean, divide by ``n - 1`` (unbiased).
    * ``sample_mle``: unknown mean, divide by ``n`` (maximum likelihood, biased
      by ``-sigma^2 / n``).
    """
    if mode not in NOISE_MODES:
        raise ValueError(f"mode must be one of {NOISE_MODES}")
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("samples must be 1-D")
    if not np.all(np.isfinite(a)):
        raise ValueError("samples must be finite")
    if mode == "zero_mean":
        if a.size < 1:
            raise ValueError("need at least 1 sample")
        return float(np.mean(a * a))
    if a.size < 2:
        raise ValueError("need at least 2 samples when the mean is unknown")
    return float(np.var(a, ddof=1 if mode == "sample_unbiased" else 0))


def noise_estimator_theory(sigma: float, n: int, mode: str = "zero_mean") -> Dict[str, float]:
    """Analytic bias, variance and RMSE of ``estimate_noise_variance`` (Gaussian noise)."""
    if mode not in NOISE_MODES:
        raise ValueError(f"mode must be one of {NOISE_MODES}")
    s = _require_positive("sigma", sigma)
    n = _require_int("n", n, 1 if mode == "zero_mean" else 2)
    s2 = s * s
    if mode == "zero_mean":
        bias, var = 0.0, 2.0 * s2 * s2 / n
    elif mode == "sample_unbiased":
        bias, var = 0.0, 2.0 * s2 * s2 / (n - 1)
    else:
        bias, var = -s2 / n, 2.0 * s2 * s2 * (n - 1) / (n * n)
    return {"bias": bias, "variance": var, "rmse": float(np.sqrt(var + bias * bias))}


def monte_carlo_noise_estimator(
    n: int, sigma: float, trials: int, seed: int = 0, mode: str = "zero_mean"
) -> Dict[str, float]:
    """Monte Carlo bias / variance / RMSE of the noise-variance estimator vs. theory."""
    s = _require_positive("sigma", sigma)
    n = _require_int("n", n, 1 if mode == "zero_mean" else 2)
    trials = _require_int("trials", trials, 2)
    theory = noise_estimator_theory(s, n, mode)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, s, size=(trials, n))
    if mode == "zero_mean":
        est = np.mean(x * x, axis=1)
    else:
        est = np.var(x, axis=1, ddof=1 if mode == "sample_unbiased" else 0)
    true = s * s
    bias = float(np.mean(est) - true)
    var = float(np.var(est, ddof=1))
    return {
        "mode": mode,
        "n": float(n),
        "trials": float(trials),
        "true_variance": true,
        "bias": bias,
        "theory_bias": theory["bias"],
        "empirical_variance": var,
        "theory_variance": theory["variance"],
        "variance_ratio": var / theory["variance"],
        "rmse": float(np.sqrt(np.mean((est - true) ** 2))),
        "theory_rmse": theory["rmse"],
    }


def monte_carlo_cfar_false_alarm(
    num_train: int, pfa: float, trials: int, seed: int = 0, noise_power: float = 1.0
) -> Dict[str, float]:
    """Empirical false-alarm rate of a CA-CFAR threshold built from ``2*num_train`` cells.

    For exponential noise power the threshold ``alpha * mean(training cells)``
    gives a false-alarm probability of exactly ``pfa`` (see ``cfar_alpha``); the
    training average is itself a noisy noise-level estimate with relative
    standard deviation ``1 / sqrt(2 * num_train)``. Both are measured here.
    """
    alpha = cfar_alpha(pfa, num_train)  # validates pfa and num_train
    trials = _require_int("trials", trials, 2)
    mu = _require_positive("noise_power", noise_power)
    cells = 2 * int(num_train)
    rng = np.random.default_rng(seed)
    train = rng.exponential(mu, size=(trials, cells))
    cut = rng.exponential(mu, size=trials)
    noise_est = train.mean(axis=1)
    empirical = float(np.mean(cut > alpha * noise_est))
    return {
        "num_train": float(num_train),
        "trials": float(trials),
        "pfa_nominal": float(pfa),
        "pfa_empirical": empirical,
        "pfa_std_error": float(np.sqrt(pfa * (1.0 - pfa) / trials)),
        "noise_estimate_bias": float(np.mean(noise_est) - mu),
        "noise_estimate_rel_std": float(np.std(noise_est, ddof=1) / mu),
        "noise_estimate_rel_std_theory": float(1.0 / np.sqrt(cells)),
    }


# --------------------------------------------------------------------------
# 2. tone frequency estimation
# --------------------------------------------------------------------------


def _interpolate_bin(spec: np.ndarray, k: int) -> float:
    """Parabolic interpolation of the log power spectrum around bin ``k`` (fractional bin)."""
    tiny = np.finfo(float).tiny
    la, lb, lc = np.log(np.maximum(spec[k - 1 : k + 2], tiny))
    denom = la - 2.0 * lb + lc
    if denom == 0.0:
        return float(k)
    delta = 0.5 * (la - lc) / denom
    return float(k + np.clip(delta, -0.5, 0.5))


def _checked_signal(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("signal must be 1-D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal must be finite")
    return arr


def estimate_tone_frequency(x, fs: float) -> float:
    """Frequency in Hz of the strongest tone in ``x`` (length a power of two).

    Hann-windowed FFT, peak among bins ``1 .. n/2 - 1`` and parabolic
    interpolation of the log power spectrum, so the result is finer than one bin.
    """
    fs = _require_positive("fs", fs)
    arr = _checked_signal(x)
    if arr.size < 8:
        raise ValueError("signal too short for interpolation")
    spec = power_spectrum(arr, window="hann")  # raises if length is not a power of two
    half = arr.size // 2
    k = 1 + int(np.argmax(spec[1:half]))
    if spec[k] <= 0.0:
        raise ValueError("signal has no spectral peak (all-zero spectrum)")
    return _interpolate_bin(spec, k) * fs / arr.size


def estimate_tone_frequencies(
    x, fs: float, pfa: float = 1e-6, num_train: int = 16, num_guard: int = 4
) -> np.ndarray:
    """Sub-bin frequencies (Hz, ascending) of all CA-CFAR-detected tones in ``x``.

    Detection reuses ``detection.detect_tones``; each detected bin is then
    refined by parabolic interpolation. Bins 0 and ``n/2`` have no two-sided
    neighbours and are skipped.
    """
    fs = _require_positive("fs", fs)
    arr = _checked_signal(x)
    bins = detect_tones(arr, pfa=pfa, num_train=num_train, num_guard=num_guard)
    spec = power_spectrum(arr, window="hann")
    half = arr.size // 2
    freqs = [_interpolate_bin(spec, int(k)) * fs / arr.size for k in bins if 1 <= k <= half - 1]
    return np.array(sorted(freqs), dtype=float)


def tone_frequency_crb_std_hz(snr_db: float, n: int, fs: float) -> float:
    """Cramer-Rao lower bound (standard deviation, Hz) for a real tone in white noise.

    Uses ``var(f) >= 12 / ((2 pi)^2 * eta * N * (N^2 - 1))`` in cycles/sample with
    ``eta = A^2 / (2 sigma^2)`` (amplitude, phase and frequency unknown; the
    interaction with the negative-frequency image is ignored, so the bound is
    meaningful away from DC and Nyquist).
    """
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    n = _require_int("n", n, 2)
    fs = _require_positive("fs", fs)
    eta = 10.0 ** (float(snr_db) / 10.0)
    var_cycles = 12.0 / ((2.0 * np.pi) ** 2 * eta * n * (n * n - 1.0))
    return float(fs * np.sqrt(var_cycles))


def monte_carlo_tone_frequency(
    n: int, fs: float, snr_db: float, trials: int, seed: int = 0
) -> Dict[str, float]:
    """Bias / std / RMSE of ``estimate_tone_frequency`` against truth and the CRB.

    Each trial draws a random off-grid tone in the bin range ``[n/8, 3n/8)``, a
    random phase and white Gaussian noise at the requested per-sample SNR
    (``eta = A^2 / (2 sigma^2)`` with ``A = 1``).
    """
    n = _require_int("n", n, 16)
    if n & (n - 1):
        raise ValueError("n must be a power of two")
    fs = _require_positive("fs", fs)
    trials = _require_int("trials", trials, 2)
    crb = tone_frequency_crb_std_hz(snr_db, n, fs)
    sigma = float(np.sqrt(1.0 / (2.0 * 10.0 ** (float(snr_db) / 10.0))))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    errors = np.empty(trials)
    for t in range(trials):
        f_bins = rng.uniform(n / 8.0, 3.0 * n / 8.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        sig = np.cos(2.0 * np.pi * f_bins * idx / n + phase) + rng.normal(0.0, sigma, n)
        errors[t] = estimate_tone_frequency(sig, fs) - f_bins * fs / n
    return {
        "n": float(n),
        "fs": float(fs),
        "snr_db": float(snr_db),
        "trials": float(trials),
        "bias_hz": float(np.mean(errors)),
        "std_hz": float(np.std(errors, ddof=1)),
        "rmse_hz": float(np.sqrt(np.mean(errors**2))),
        "crb_std_hz": crb,
        "rmse_over_crb": float(np.sqrt(np.mean(errors**2)) / crb),
    }


# --------------------------------------------------------------------------
# 3. constant-velocity Kalman filter
# --------------------------------------------------------------------------


def _cv_matrices(dt: float, accel_std: float):
    """Transition F, acceleration input G, process noise Q and measurement H."""
    f = np.eye(STATE_DIM)
    f[0, 2] = dt
    f[1, 3] = dt
    g = np.array([[dt * dt / 2.0, 0.0], [0.0, dt * dt / 2.0], [dt, 0.0], [0.0, dt]])
    q = accel_std**2 * (g @ g.T)
    h = np.zeros((2, STATE_DIM))
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    return f, g, q, h


def simulate_cv_track(
    steps: int,
    dt: float,
    accel_std: float,
    meas_std: float,
    init_vel_std: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic 2-D track from the discrete white-noise-acceleration model.

    Returns ``(truth, meas)``: ``truth`` is ``(steps, 4)`` with columns
    ``x, y, vx, vy`` (start at the origin, initial velocity drawn from
    ``N(0, init_vel_std^2)``), ``meas`` is ``(steps, 2)`` noisy positions.
    """
    steps = _require_int("steps", steps, 2)
    dt = _require_positive("dt", dt)
    accel_std = _require_nonnegative("accel_std", accel_std)
    meas_std = _require_positive("meas_std", meas_std)
    init_vel_std = _require_positive("init_vel_std", init_vel_std)
    rng = np.random.default_rng(seed)
    f, g, _, _ = _cv_matrices(dt, accel_std)
    truth = np.empty((steps, STATE_DIM))
    truth[0] = [0.0, 0.0, *rng.normal(0.0, init_vel_std, 2)]
    accel = rng.normal(0.0, accel_std, size=(steps - 1, 2))
    for k in range(1, steps):
        truth[k] = f @ truth[k - 1] + g @ accel[k - 1]
    meas = truth[:, :2] + rng.normal(0.0, meas_std, size=(steps, 2))
    return truth, meas


def kalman_cv_filter(
    meas, dt: float, accel_std: float, meas_std: float, init_vel_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity Kalman filter over position measurements.

    Initialised from the first measurement (zero velocity, velocity std
    ``init_vel_std``). Returns state estimates ``(steps, 4)`` and covariances
    ``(steps, 4, 4)``; the covariance update uses the Joseph form.
    """
    z = np.asarray(meas, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or z.shape[0] < 2:
        raise ValueError("meas must have shape (steps >= 2, 2)")
    if not np.all(np.isfinite(z)):
        raise ValueError("meas must be finite")
    dt = _require_positive("dt", dt)
    accel_std = _require_nonnegative("accel_std", accel_std)
    meas_std = _require_positive("meas_std", meas_std)
    init_vel_std = _require_positive("init_vel_std", init_vel_std)
    f, _, q, h = _cv_matrices(dt, accel_std)
    r = meas_std**2 * np.eye(2)
    ident = np.eye(STATE_DIM)
    xs = np.empty((z.shape[0], STATE_DIM))
    ps = np.empty((z.shape[0], STATE_DIM, STATE_DIM))
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    p = np.diag([meas_std**2, meas_std**2, init_vel_std**2, init_vel_std**2])
    xs[0], ps[0] = x, p
    for k in range(1, z.shape[0]):
        x = f @ x
        p = f @ p @ f.T + q
        s = h @ p @ h.T + r
        gain = np.linalg.solve(s, h @ p).T
        x = x + gain @ (z[k] - h @ x)
        ikh = ident - gain @ h
        p = ikh @ p @ ikh.T + gain @ r @ gain.T
        p = 0.5 * (p + p.T)
        xs[k], ps[k] = x, p
    return xs, ps


def nees(truth, est, covs) -> np.ndarray:
    """Normalised estimation error squared per step: ``e^T P^-1 e``.

    For a consistent filter the mean NEES equals the state dimension (4 here).
    """
    t = np.asarray(truth, dtype=float)
    e = np.asarray(est, dtype=float)
    c = np.asarray(covs, dtype=float)
    if t.ndim != 2 or t.shape != e.shape or c.shape != (t.shape[0], t.shape[1], t.shape[1]):
        raise ValueError("truth/est must be (steps, d) and covs (steps, d, d)")
    err = t - e
    sol = np.linalg.solve(c, err[..., None])[..., 0]
    return np.einsum("ij,ij->i", err, sol)


def monte_carlo_tracking(
    trials: int,
    steps: int,
    dt: float = 1.0,
    accel_std: float = 0.1,
    meas_std: float = 5.0,
    init_vel_std: float = 1.0,
    seed: int = 0,
    burn_in: int = 20,
    filter_accel_std: Optional[float] = None,
) -> Dict[str, float]:
    """Position RMSE (filter vs. raw measurements) and mean NEES over seeded tracks.

    ``filter_accel_std`` defaults to the true ``accel_std`` (matched filter);
    pass a different value to study a mismatched process model. RMSE skips the
    first ``burn_in`` steps; NEES uses every step.
    """
    trials = _require_int("trials", trials, 1)
    steps = _require_int("steps", steps, 2)
    burn_in = _require_int("burn_in", burn_in, 0)
    if burn_in >= steps:
        raise ValueError("burn_in must be smaller than steps")
    fa = accel_std if filter_accel_std is None else filter_accel_std
    sq_filter: List[np.ndarray] = []
    sq_raw: List[np.ndarray] = []
    nees_all: List[np.ndarray] = []
    for t in range(trials):
        truth, meas = simulate_cv_track(steps, dt, accel_std, meas_std, init_vel_std, seed + t)
        xs, ps = kalman_cv_filter(meas, dt, fa, meas_std, init_vel_std)
        sq_filter.append(np.sum((xs[burn_in:, :2] - truth[burn_in:, :2]) ** 2, axis=1))
        sq_raw.append(np.sum((meas[burn_in:] - truth[burn_in:, :2]) ** 2, axis=1))
        nees_all.append(nees(truth, xs, ps))
    filter_rmse = float(np.sqrt(np.mean(np.concatenate(sq_filter))))
    raw_rmse = float(np.sqrt(np.mean(np.concatenate(sq_raw))))
    return {
        "trials": float(trials),
        "steps": float(steps),
        "burn_in": float(burn_in),
        "filter_pos_rmse": filter_rmse,
        "raw_pos_rmse": raw_rmse,
        "raw_pos_rmse_theory": float(np.sqrt(2.0) * meas_std),
        "filter_over_raw_rmse": filter_rmse / raw_rmse,
        "mean_nees": float(np.mean(np.concatenate(nees_all))),
        "nees_expected": float(STATE_DIM),
    }


# --------------------------------------------------------------------------
# suite / CLI
# --------------------------------------------------------------------------

TONE_SNRS_DB = (0.0, 10.0, 20.0)
TONE_N = 256
TONE_FS = 256.0  # one FFT bin == 1 Hz


def run_estimation_suite(quick: bool, seed: int = 0) -> Dict[str, object]:
    """Run every estimation experiment with seeded synthetic data."""
    noise_trials, cfar_trials, tone_trials, track_trials = (
        (500, 5000, 100, 40) if quick else (4000, 50000, 500, 200)
    )
    return {
        "seed": seed,
        "quick": bool(quick),
        "noise": {
            mode: monte_carlo_noise_estimator(16, 1.0, noise_trials, seed, mode)
            for mode in NOISE_MODES
        },
        "cfar_noise": monte_carlo_cfar_false_alarm(16, 1e-2, cfar_trials, seed),
        "tone_frequency": [
            monte_carlo_tone_frequency(TONE_N, TONE_FS, snr, tone_trials, seed)
            for snr in TONE_SNRS_DB
        ],
        "tracking": {
            "matched": monte_carlo_tracking(track_trials, 100, seed=seed),
            "mismatched_filter_accel_0.01": monte_carlo_tracking(
                track_trials, 100, accel_std=1.0, filter_accel_std=0.01, seed=seed
            ),
        },
    }


def print_estimation_report(report: Dict[str, object]) -> None:
    """Print a suite report (all values come from the run that produced it)."""
    print("== Estimation (synthetic, seeded; textbook estimators, no real data) ==")
    print("-- Noise variance estimator, sigma^2 = 1, Monte Carlo vs. analytic --")
    for mode, r in report["noise"].items():
        print(
            f"{mode}: n={int(r['n'])} trials={int(r['trials'])} bias={r['bias']:+.4f} "
            f"(theory {r['theory_bias']:+.4f}) var={r['empirical_variance']:.4f} "
            f"(theory {r['theory_variance']:.4f}) rmse={r['rmse']:.4f} "
            f"(theory {r['theory_rmse']:.4f})"
        )
    c = report["cfar_noise"]
    print("-- CA-CFAR training-cell noise estimate (exponential power) --")
    print(
        f"cells={2 * int(c['num_train'])} trials={int(c['trials'])} "
        f"pfa_nominal={c['pfa_nominal']:.4f} pfa_empirical={c['pfa_empirical']:.4f} "
        f"(std error {c['pfa_std_error']:.4f}) noise_estimate_rel_std="
        f"{c['noise_estimate_rel_std']:.4f} (theory {c['noise_estimate_rel_std_theory']:.4f})"
    )
    print("-- Tone frequency: Hann FFT + parabolic interpolation vs. Cramer-Rao bound --")
    for r in report["tone_frequency"]:
        print(
            f"snr={r['snr_db']:.0f} dB n={int(r['n'])} trials={int(r['trials'])} "
            f"bias={r['bias_hz']:+.5f} Hz std={r['std_hz']:.5f} Hz rmse={r['rmse_hz']:.5f} Hz "
            f"crb_std={r['crb_std_hz']:.5f} Hz rmse/crb={r['rmse_over_crb']:.2f}"
        )
    print("-- Constant-velocity Kalman filter, 2-D synthetic tracks --")
    for name, r in report["tracking"].items():
        print(
            f"{name}: trials={int(r['trials'])} steps={int(r['steps'])} "
            f"filter_rmse={r['filter_pos_rmse']:.3f} raw_rmse={r['raw_pos_rmse']:.3f} "
            f"(theory {r['raw_pos_rmse_theory']:.3f}) ratio={r['filter_over_raw_rmse']:.3f} "
            f"mean_nees={r['mean_nees']:.2f} (consistent filter: {r['nees_expected']:.0f})"
        )
    print(
        "Note: Monte Carlo figures carry sampling error that shrinks with the trial count; "
        "the tone-frequency estimator is biased and not efficient, so rmse/crb is not "
        "expected to reach 1."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic statistical-estimation experiments")
    parser.add_argument("--quick", action="store_true", help="small trial counts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)
    report = run_estimation_suite(args.quick, args.seed)
    print_estimation_report(report)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
