"""Finite check of the C++ kernels against the NumPy reference, with timing, on seeded synthetic data.

Without ``--out`` nothing is written to disk. With ``--out DIR`` (must be new or empty) the run
is recorded as protocol.json, summary.json, REPORT.md and a manifest. Every number in the output
is measured by this run; no result files are shipped with the code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import classical_native as native
from . import classical_vision as cv
from .common import write_json
from .measurement import RunRecord, distribution, environment

TOLERANCE = {'rtol': 1e-9, 'atol': 1e-9}
MIN_EDGE_F1 = 0.75
TIMING_METHOD = ('numpy: perf_counter around one call. cpp_wall: perf_counter around one subprocess call, '
                 'including process start, text formatting and parsing. cpp_kernel: compute loop only, '
                 'measured inside the C++ program. Single-threaded, host CPU, uncontrolled background load.')
SCOPE = ('Small seeded synthetic image and signal data only. No video, driving, clinical or production '
         'data; timings are not representative of optimized libraries or other hardware.')


def _cases(seed: int):
    image, region = cv.synthetic_image(seed)
    signal = cv.synthetic_signal(seed)
    kernel = cv.gaussian_kernel(5, 1.0)
    taps = cv.lowpass_taps(63, 0.0625)
    cases = {
        'correlate2d_gauss5x5': (lambda: cv.correlate2d(image, kernel),
                                 lambda: native.correlate2d(image, kernel)),
        'sobel_magnitude': (lambda: cv.sobel_magnitude(image), lambda: native.sobel_magnitude(image)),
        'fir_lowpass63': (lambda: cv.fir_filter(signal, taps), lambda: native.fir_filter(signal, taps)),
        'rfft_256': (lambda: cv.rfft(signal), lambda: native.rfft(signal)),
        'stft_256_frame64_hop32': (lambda: cv.stft_magnitude(signal, 64, 32),
                                   lambda: native.stft_magnitude(signal, 64, 32)),
    }
    return image, region, signal, cases


def _timed_ms(call):
    start = time.perf_counter()
    result = call()
    return (time.perf_counter() - start) * 1000.0, result


def run_checks(seed: int = 11, repeats: int = 5, require_native: bool = False) -> dict:
    if int(repeats) != repeats or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    image, region, signal, cases = _cases(seed)
    use_native = native.native_available()
    if require_native and not use_native:
        raise native.NativeUnavailable('native kernels are required but unavailable')

    reference_edges = cv.detect_edges(image, cv.NUMPY)
    scores = cv.edge_scores(reference_edges, cv.boundary_mask(region), tolerance=2)
    checks = {'numpy_edge_f1_at_least_%.2f' % MIN_EDGE_F1: scores['f1'] >= MIN_EDGE_F1}
    summary = {
        'seed': seed, 'repeats': repeats, 'scope': SCOPE, 'tolerance': TOLERANCE,
        'data': {'image': '64x64 rectangle and disc on background, noise sigma 0.02',
                 'signal': '256 samples at 256 Hz: 12 Hz (amplitude 1.0) + 40 Hz (0.5), noise sigma 0.05'},
        'numpy_edge_scores': scores,
        'numpy_dominant_frequency_hz': cv.dominant_frequency(signal, 256.0, cv.NUMPY),
        'native_executed': use_native, 'equivalence': {}, 'timing_ms': {}, 'timing_method': TIMING_METHOD,
    }

    for name, (numpy_call, native_call) in cases.items():
        numpy_samples = []
        for _ in range(repeats):
            elapsed, reference = _timed_ms(numpy_call)
            numpy_samples.append(elapsed)
        timing = {'numpy': distribution(numpy_samples)}
        if use_native:
            wall, kernel = [], []
            for _ in range(repeats):
                elapsed, result = _timed_ms(native_call)
                wall.append(elapsed)
                kernel.append(result.kernel_ns_per_call / 1e6)
            diff = float(np.max(np.abs(result.value - reference)))
            close = bool(np.allclose(result.value, reference, **TOLERANCE))
            summary['equivalence'][name] = {'max_abs_diff': diff, 'within_tolerance': close}
            checks['equivalent_' + name] = close
            timing['cpp_wall'], timing['cpp_kernel'] = distribution(wall), distribution(kernel)
        summary['timing_ms'][name] = timing

    if use_native:
        native_edges = cv.detect_edges(image, native.NATIVE)
        checks['edge_maps_identical'] = bool(np.array_equal(native_edges, reference_edges))
        native_frequency = cv.dominant_frequency(signal, 256.0, native.NATIVE)
        summary['cpp_dominant_frequency_hz'] = native_frequency
        checks['dominant_frequency_equal'] = native_frequency == summary['numpy_dominant_frequency_hz']
    summary['checks'] = checks
    summary['all_passed'] = all(checks.values())
    return summary


def compact(summary: dict) -> dict:
    """Summary without raw timing samples, for console output."""
    out = dict(summary)
    out['timing_ms'] = {name: {kind: {'count': d['count'], 'p50_ms': d['p50_ms'], 'p95_ms': d['p95_ms']}
                               for kind, d in kinds.items()}
                        for name, kinds in summary['timing_ms'].items()}
    return out


def render_report(summary: dict) -> str:
    lines = ['# Classical vision kernel check', '', summary['scope'], '',
             'Seed %s, %s repetitions per timing. Native kernels executed: %s.' % (
                 summary['seed'], summary['repeats'], summary['native_executed']), '',
             '| Check | Passed |', '|---|---|']
    lines += ['| %s | %s |' % (name, ok) for name, ok in summary['checks'].items()]
    e = summary['numpy_edge_scores']
    lines += ['', 'Edge detection (NumPy path, tolerance %d px): precision %.3f, recall %.3f, F1 %.3f.' % (
        e['tolerance_px'], e['precision'], e['recall'], e['f1']), '',
        '| Operation | NumPy p50 ms | C++ wall p50 ms | C++ kernel p50 ms | Max abs difference |',
        '|---|---:|---:|---:|---:|']
    for name, kinds in summary['timing_ms'].items():
        eq = summary['equivalence'].get(name)
        lines.append('| %s | %.4f | %s | %s | %s |' % (
            name, kinds['numpy']['p50_ms'],
            '%.4f' % kinds['cpp_wall']['p50_ms'] if 'cpp_wall' in kinds else 'n/a',
            '%.4f' % kinds['cpp_kernel']['p50_ms'] if 'cpp_kernel' in kinds else 'n/a',
            '%.3g' % eq['max_abs_diff'] if eq else 'n/a'))
    lines += ['', 'Timing method: ' + summary['timing_method'], '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--out', help='record the run in this new or empty directory')
    parser.add_argument('--require-native', action='store_true',
                        help='fail instead of falling back to NumPy-only checks when C++ is unavailable')
    args = parser.parse_args(argv)
    if args.out:
        sources = [native.SOURCE, Path(cv.__file__), Path(native.__file__), Path(__file__)]
        protocol = {'name': 'classical_vision_check', 'seed': args.seed, 'repeats': args.repeats,
                    'tolerance': TOLERANCE, 'environment': environment()}
        with RunRecord(args.out, protocol, sources):
            summary = run_checks(args.seed, args.repeats, args.require_native)
            write_json(Path(args.out) / 'summary.json', summary)
            (Path(args.out) / 'REPORT.md').write_text(render_report(summary))
    else:
        summary = run_checks(args.seed, args.repeats, args.require_native)
    print(json.dumps(compact(summary), indent=2))
    raise SystemExit(0 if summary['all_passed'] else 1)


if __name__ == '__main__':
    main()
