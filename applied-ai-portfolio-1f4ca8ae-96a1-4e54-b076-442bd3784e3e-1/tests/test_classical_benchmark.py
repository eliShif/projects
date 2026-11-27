"""The finite kernel check: pass path, NumPy-only fallback, failure path and run recording."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portfolio import classical_benchmark as bench
from portfolio import classical_native as native
from portfolio.measurement import file_hash


class RunChecksTests(unittest.TestCase):
    def test_all_checks_pass_and_timings_are_raw_measurements(self):
        summary = bench.run_checks(seed=11, repeats=2)
        self.assertTrue(summary['all_passed'], summary['checks'])
        self.assertEqual(summary['native_executed'], native.native_available())
        self.assertEqual(set(summary['timing_ms']), {'correlate2d_gauss5x5', 'sobel_magnitude', 'fir_lowpass63',
                                                     'rfft_256', 'stft_256_frame64_hop32'})
        for kinds in summary['timing_ms'].values():
            self.assertEqual(len(kinds['numpy']['raw_ms']), 2)
            self.assertGreaterEqual(kinds['numpy']['p50_ms'], 0.0)
        if summary['native_executed']:
            self.assertEqual(set(summary['equivalence']), set(summary['timing_ms']))
            for entry in summary['equivalence'].values():
                self.assertTrue(entry['within_tolerance'])
                self.assertLessEqual(entry['max_abs_diff'], 1e-6)
            self.assertEqual(summary['cpp_dominant_frequency_hz'], summary['numpy_dominant_frequency_hz'])

    def test_numpy_only_fallback_is_explicit_and_native_can_be_required(self):
        with patch.object(bench.native, 'native_available', return_value=False):
            summary = bench.run_checks(seed=11, repeats=1)
            self.assertFalse(summary['native_executed'])
            self.assertEqual(summary['equivalence'], {})
            self.assertNotIn('cpp_wall', summary['timing_ms']['rfft_256'])
            self.assertTrue(summary['all_passed'])
            with self.assertRaises(native.NativeUnavailable):
                bench.run_checks(seed=11, repeats=1, require_native=True)

    def test_repeats_must_be_a_positive_integer(self):
        for repeats in (0, -1, 1.5):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                bench.run_checks(repeats=repeats)

    def test_a_mismatching_native_result_fails_the_run(self):
        if not native.native_available():
            self.skipTest('native kernels unavailable')
        real = native.rfft

        def corrupted(signal, repeat=1):
            result = real(signal, repeat)
            return native.NativeResult(result.value + 1e-3, result.kernel_ns_per_call, result.repeats)

        with patch.object(bench.native, 'rfft', corrupted):
            summary = bench.run_checks(seed=11, repeats=1)
        self.assertFalse(summary['all_passed'])
        self.assertFalse(summary['checks']['equivalent_rfft_256'])
        self.assertTrue(summary['checks']['equivalent_sobel_magnitude'])


class CommandLineTests(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as caught:
            bench.main(argv)
        return caught.exception.code, stream.getvalue()

    def test_console_output_is_valid_json_without_raw_samples(self):
        code, text = self.run_main(['--repeats', '1'])
        self.assertEqual(code, 0)
        parsed = json.loads(text)
        self.assertTrue(parsed['all_passed'])
        self.assertNotIn('raw_ms', json.dumps(parsed))

    def test_recorded_run_has_manifest_report_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            code, _ = self.run_main(['--repeats', '1', '--out', str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
            summary = json.loads((out / 'summary.json').read_text())
            self.assertTrue(summary['all_passed'])
            self.assertIn('raw_ms', json.dumps(summary))
            self.assertIn('Timing method', (out / 'REPORT.md').read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['summary.json'], file_hash(out / 'summary.json'))
            protocol = json.loads((out / 'protocol.json').read_text())
            self.assertIn('cpp/filters.cpp', ' '.join(protocol['source_sha256']))
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                bench.main(['--repeats', '1', '--out', str(out)])
            self.assertEqual(manifest['summary.json'], file_hash(out / 'summary.json'))

    def test_failing_check_returns_nonzero(self):
        with patch.object(bench, 'MIN_EDGE_F1', 1.01):
            code, text = self.run_main(['--repeats', '1'])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(text)['all_passed'])


if __name__ == '__main__':
    unittest.main()
