"""Reproducible synthetic experiment: accuracy, Python/C++ parity and host timing.

Run with ``python -m sigvision.experiment --quick --require-cpp``. Every number
printed is computed by this run on synthetic data; nothing here is a stored
result, and none of it says anything about real sensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import cpp_bridge
from .cpp_bridge import CppCore, CppCoreError
from .detection import cfar_alpha, cfar_ca_detect, detect_tones
from .metrics import match_points, match_scalars
from .perf import peak_alloc_bytes, time_calls_us
from .signals import make_signal
from .spectral import apply_fir, design_lowpass_fir, fft_radix2, power_spectrum, snr_db
from .vision import SOBEL_X, conv2d, harris_corners, make_scene, rect_corners

N_FFT = 1024
FS_SPECTRAL = 1024.0  # one FFT bin == 1 Hz
PARITY_TOL = 1e-9


def _pick_bins(rng: np.random.Generator, count: int, lo: int, hi: int, min_sep: int) -> List[int]:
    """Random sorted integer bins in ``[lo, hi)`` that are at least ``min_sep`` apart."""
    while True:
        bins = sorted(int(b) for b in rng.integers(lo, hi, size=count))
        if all(b - a >= min_sep for a, b in zip(bins, bins[1:])):
            return bins


def run_spectral_trials(trials: int, seed: int) -> Dict[str, float]:
    """Two off-grid tones in unit-variance noise; Hann + FFT + CA-CFAR vs. truth."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    truth_total = matched_total = false_total = 0
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        bins = _pick_bins(rng, 2, 30, 480, 25)
        offsets = rng.uniform(-0.3, 0.3, size=2)
        freqs = [float(b + o) for b, o in zip(bins, offsets)]
        noisy, _ = make_signal(
            N_FFT, FS_SPECTRAL, [(f, 1.0) for f in freqs], noise_std=1.0, seed=seed + 10_000 + t
        )
        detected = detect_tones(noisy).tolist()
        matched, false = match_scalars(freqs, detected, tol=1.0)
        truth_total += len(freqs)
        matched_total += matched
        false_total += false
    return {
        "trials": float(trials),
        "tone_recall": matched_total / truth_total,
        "false_alarms_per_trial": false_total / trials,
    }


def run_fir_trial(seed: int = 0) -> Dict[str, float]:
    """Low-pass FIR on a 50 Hz tone + 300 Hz interferer + white noise (fs = 1 kHz)."""
    fs, n, taps = 1000.0, 4000, 101
    noisy, _ = make_signal(n, fs, [(50.0, 1.0), (300.0, 0.5)], noise_std=1.0, seed=seed)
    _, clean = make_signal(n, fs, [(50.0, 1.0)])
    h = design_lowpass_fir(taps, 100.0, fs)
    filtered = apply_fir(noisy, h)
    trim = taps // 2
    before = snr_db(clean, noisy, trim=trim)
    after = snr_db(clean, filtered, trim=trim)
    return {"snr_before_db": before, "snr_after_db": after, "improvement_db": after - before}


def _random_scene(rng: np.random.Generator, noise_seed: int):
    """96x96 scene with one random rectangle in each 48x48 cell."""
    rects = []
    for gr in range(2):
        for gc in range(2):
            h = int(rng.integers(14, 31))
            w = int(rng.integers(14, 31))
            r0 = int(rng.integers(gr * 48 + 8, gr * 48 + 48 - 8 - h + 1))
            c0 = int(rng.integers(gc * 48 + 8, gc * 48 + 48 - 8 - w + 1))
            rects.append((r0, c0, r0 + h, c0 + w, float(rng.uniform(0.8, 1.0))))
    return make_scene((96, 96), rects, noise_std=0.01, seed=noise_seed), rects


def run_vision_trials(trials: int, seed: int) -> Dict[str, float]:
    """Harris corner detection vs. known rectangle corners (match radius 3 px)."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    truth_total = matched_total = false_total = 0
    for t in range(trials):
        rng = np.random.default_rng(seed + 50_000 + t)
        image, rects = _random_scene(rng, noise_seed=seed + 60_000 + t)
        truth = rect_corners(rects)
        detected = harris_corners(image)
        matched, false = match_points(truth, detected, tol=3.0)
        truth_total += len(truth)
        matched_total += matched
        false_total += false
    return {
        "trials": float(trials),
        "corner_recall": matched_total / truth_total,
        "false_corners_per_trial": false_total / trials,
    }


def run_parity(core: CppCore, seed: int) -> Dict[str, object]:
    """Compare the C++ core with the Python reference on identical random inputs."""
    rng = np.random.default_rng(seed)

    x = rng.normal(size=256) + 1j * rng.normal(size=256)
    fft_err = float(np.max(np.abs(core.fft(x) - fft_radix2(x))))
    fft_vs_numpy = float(np.max(np.abs(fft_radix2(x) - np.fft.fft(x))))

    noisy, _ = make_signal(N_FFT, FS_SPECTRAL, [(100.0, 1.0), (237.4, 1.0)], 1.0, seed=seed)
    spec = power_spectrum(noisy)
    alpha = cfar_alpha(1e-4, 16)
    cfar_py = cfar_ca_detect(spec, 16, 4, alpha)
    cfar_cpp = core.cfar_detect(spec, 16, 4, alpha)
    expo = rng.exponential(1.0, size=2048)
    expo[[100, 700, 1500]] += 60.0
    cfar_py2 = cfar_ca_detect(expo, 12, 2, cfar_alpha(1e-4, 12))
    cfar_cpp2 = core.cfar_detect(expo, 12, 2, cfar_alpha(1e-4, 12))
    cfar_equal = bool(
        np.array_equal(cfar_py, cfar_cpp) and np.array_equal(cfar_py2, cfar_cpp2)
    )

    img = rng.normal(size=(40, 50))
    ker = rng.normal(size=(5, 5))
    conv_err = float(np.max(np.abs(core.conv2d(img, ker) - conv2d(img, ker))))
    sobel_err = float(np.max(np.abs(core.conv2d(img, SOBEL_X) - conv2d(img, SOBEL_X))))

    ok = (
        fft_err < PARITY_TOL
        and fft_vs_numpy < PARITY_TOL
        and cfar_equal
        and conv_err < PARITY_TOL
        and sobel_err < PARITY_TOL
    )
    return {
        "tolerance": PARITY_TOL,
        "fft_max_abs_err": fft_err,
        "fft_python_vs_numpy_max_abs_err": fft_vs_numpy,
        "cfar_indices_equal": cfar_equal,
        "cfar_detection_counts": [int(cfar_py.size), int(cfar_py2.size)],
        "conv2d_max_abs_err": conv_err,
        "sobel_conv2d_max_abs_err": sobel_err,
        "ok": bool(ok),
    }


def run_timing(core: Optional[CppCore], repeats: int, seed: int) -> Dict[str, Dict[str, object]]:
    """Host-CPU latency (p50/p95, microseconds) of Python vs. C++ kernels."""
    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, object]] = {}

    xs = rng.normal(size=4096)
    power = rng.exponential(1.0, size=4096)
    alpha = cfar_alpha(1e-4, 16)
    img = rng.normal(size=(128, 128))
    ker = rng.normal(size=(5, 5))

    # name -> (C++ bench kind, Python callable, C++ stdin payload builder)
    cases = {
        "fft_4096": ("fft", lambda: fft_radix2(xs), lambda: cpp_bridge.fft_payload(xs)),
        "cfar_4096": (
            "cfar",
            lambda: cfar_ca_detect(power, 16, 4, alpha),
            lambda: cpp_bridge.cfar_payload(power, 16, 4, alpha),
        ),
        "conv2d_128x128_5x5": (
            "conv2d",
            lambda: conv2d(img, ker),
            lambda: cpp_bridge.conv2d_payload(img, ker),
        ),
    }

    for name, (kind, py_fn, make_payload) in cases.items():
        entry: Dict[str, object] = {
            "python": time_calls_us(py_fn, repeats),
            "python_peak_alloc_bytes": peak_alloc_bytes(py_fn),
        }
        if name == "fft_4096":
            entry["numpy_fft"] = time_calls_us(lambda: np.fft.fft(xs), repeats)
        if core is not None:
            entry["cpp"] = core.bench(kind, make_payload(), repeats)
        results[name] = entry
    return results


def _print_report(report: Dict[str, object]) -> None:
    spectral = report["spectral"]
    print("== Spectral tone detection (synthetic, Hann + FFT + CA-CFAR) ==")
    print(
        f"trials={int(spectral['trials'])} tone_recall={spectral['tone_recall']:.3f} "
        f"false_alarms_per_trial={spectral['false_alarms_per_trial']:.3f}"
    )
    fir = report["fir"]
    print("== FIR low-pass SNR (synthetic, known clean reference) ==")
    print(
        f"snr_before={fir['snr_before_db']:.2f} dB snr_after={fir['snr_after_db']:.2f} dB "
        f"improvement={fir['improvement_db']:.2f} dB"
    )
    vision = report["vision"]
    print("== Harris corners on synthetic rectangles (match radius 3 px) ==")
    print(
        f"trials={int(vision['trials'])} corner_recall={vision['corner_recall']:.3f} "
        f"false_corners_per_trial={vision['false_corners_per_trial']:.3f}"
    )
    parity = report.get("parity")
    if parity is None:
        print("== Python/C++ parity == SKIPPED (C++ core unavailable)")
    else:
        status = "OK" if parity["ok"] else "MISMATCH"
        print(f"== Python/C++ parity == {status}")
        print(
            f"fft_max_abs_err={parity['fft_max_abs_err']:.3e} "
            f"conv2d_max_abs_err={parity['conv2d_max_abs_err']:.3e} "
            f"sobel_max_abs_err={parity['sobel_conv2d_max_abs_err']:.3e} "
            f"cfar_indices_equal={parity['cfar_indices_equal']} "
            f"(tolerance {parity['tolerance']:.0e})"
        )
    print("== Host latency, microseconds (this machine only; not an embedded target) ==")
    for name, entry in report["timing"].items():
        py = entry["python"]
        line = f"{name}: python p50={py['p50_us']:.1f} p95={py['p95_us']:.1f}"
        if "numpy_fft" in entry:
            line += f" | numpy.fft p50={entry['numpy_fft']['p50_us']:.1f}"
        if "cpp" in entry:
            cpp = entry["cpp"]
            line += f" | cpp p50={cpp['p50_us']:.1f} p95={cpp['p95_us']:.1f}"
        line += f" | python peak alloc={entry['python_peak_alloc_bytes']} B"
        print(line)
    print(
        "Memory note: Python figures are tracemalloc peaks for one call; the C++ bench reports "
        "whole-process peak RSS (kB) and is not directly comparable."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic signal/vision experiment")
    parser.add_argument("--quick", action="store_true", help="small trial counts / repeats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--binary", type=Path, default=None, help="path of a built sigcore binary")
    parser.add_argument(
        "--require-cpp", action="store_true", help="exit with status 2 if the C++ core is unusable"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    parser.add_argument(
        "--estimation",
        action="store_true",
        help="also run the statistical-estimation experiments (sigvision.estimation)",
    )
    args = parser.parse_args(argv)

    spectral_trials, vision_trials, repeats = (10, 5, 5) if args.quick else (100, 40, 30)

    core: Optional[CppCore] = None
    try:
        binary = args.binary if args.binary is not None else cpp_bridge.ensure_built()
        core = CppCore(binary)
    except CppCoreError as exc:
        if args.require_cpp:
            print(f"error: C++ core unavailable: {exc}", file=sys.stderr)
            return 2
        print(f"note: C++ core unavailable ({exc}); parity and C++ timing skipped")

    report: Dict[str, object] = {
        "seed": args.seed,
        "quick": bool(args.quick),
        "spectral": run_spectral_trials(spectral_trials, args.seed),
        "fir": run_fir_trial(args.seed),
        "vision": run_vision_trials(vision_trials, args.seed),
    }
    if core is not None:
        try:
            report["parity"] = run_parity(core, args.seed)
            report["timing"] = run_timing(core, repeats, args.seed)
        except CppCoreError as exc:
            print(f"error: C++ core failed during the run: {exc}", file=sys.stderr)
            return 2
    else:
        report["timing"] = run_timing(None, repeats, args.seed)

    if args.estimation:
        from .estimation import print_estimation_report, run_estimation_suite

        report["estimation"] = run_estimation_suite(bool(args.quick), args.seed)
    _print_report(report)
    if args.estimation:
        print_estimation_report(report["estimation"])
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n")
    parity = report.get("parity")
    if parity is not None and not parity["ok"]:
        print("error: Python and C++ outputs differ beyond tolerance", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
