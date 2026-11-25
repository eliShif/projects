"""Python reference vs. C++ core. The binaries are (re)built via ``make build``."""

import subprocess
import unittest

import numpy as np

from sigvision import cpp_bridge
from sigvision.cpp_bridge import CppCore, CppCoreError, CppToolchainMissing
from sigvision.detection import cfar_alpha, cfar_ca_detect
from sigvision.experiment import run_parity
from sigvision.signals import make_signal
from sigvision.spectral import fft_radix2, power_spectrum
from sigvision.vision import SOBEL_X, SOBEL_Y, conv2d


class CppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            binary = cpp_bridge.ensure_built()
        except CppToolchainMissing as exc:
            raise unittest.SkipTest(str(exc))
        cls.core = CppCore(binary)


class CppUnitTestBinary(CppTestCase):
    def test_native_unit_tests_pass(self):
        proc = subprocess.run(
            [str(cpp_bridge.TEST_BINARY)], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("0 failures", proc.stdout)


class FftParityTests(CppTestCase):
    def test_matches_python_radix2_and_numpy(self):
        rng = np.random.default_rng(10)
        for n in (1, 2, 8, 128, 1024):
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            got = self.core.fft(x)
            np.testing.assert_allclose(got, fft_radix2(x), atol=1e-9)
            np.testing.assert_allclose(got, np.fft.fft(x), atol=1e-9)

    def test_non_power_of_two_is_rejected(self):
        with self.assertRaises(CppCoreError) as ctx:
            self.core.fft(np.ones(12))
        self.assertIn("power of two", str(ctx.exception))


class CfarParityTests(CppTestCase):
    def test_indices_identical_on_noise_with_planted_targets(self):
        rng = np.random.default_rng(11)
        for train, guard in ((8, 2), (16, 4), (3, 0)):
            power = rng.exponential(1.0, size=3000)
            power[[5, 400, 1200, 2999]] += 80.0
            alpha = cfar_alpha(1e-4, train)
            expected = cfar_ca_detect(power, train, guard, alpha)
            got = self.core.cfar_detect(power, train, guard, alpha)
            self.assertGreaterEqual(expected.size, 2)  # the comparison is not vacuous
            np.testing.assert_array_equal(got, expected)

    def test_indices_identical_on_tone_spectrum(self):
        x, _ = make_signal(1024, 1024.0, [(100.0, 1.0), (237.0, 1.0)], noise_std=0.5, seed=0)
        spec = power_spectrum(x)
        alpha = cfar_alpha(1e-6, 16)
        got = self.core.cfar_detect(spec, 16, 4, alpha)
        np.testing.assert_array_equal(got, cfar_ca_detect(spec, 16, 4, alpha))
        self.assertEqual(got.tolist(), [100, 237])

    def test_no_detections_returns_empty(self):
        got = self.core.cfar_detect(np.ones(64), 4, 1, 5.0)
        self.assertEqual(got.size, 0)

    def test_invalid_parameters_are_rejected(self):
        for train, guard, alpha in ((0, 1, 5.0), (4, -1, 5.0), (4, 1, 0.0), (4, 1, -1.0)):
            with self.assertRaises(CppCoreError):
                self.core.cfar_detect(np.ones(32), train, guard, alpha)


class Conv2dParityTests(CppTestCase):
    def test_matches_python_for_random_kernels_and_sobel(self):
        rng = np.random.default_rng(12)
        img = rng.normal(size=(23, 31))
        for shape in ((3, 3), (5, 5), (1, 7), (3, 1)):
            ker = rng.normal(size=shape)
            np.testing.assert_allclose(self.core.conv2d(img, ker), conv2d(img, ker), atol=1e-9)
        for ker in (SOBEL_X, SOBEL_Y):
            np.testing.assert_allclose(self.core.conv2d(img, ker), conv2d(img, ker), atol=1e-9)

    def test_even_kernel_is_rejected(self):
        with self.assertRaises(CppCoreError):
            self.core.conv2d(np.ones((6, 6)), np.ones((2, 2)))


class ProtocolFailureTests(CppTestCase):
    def test_truncated_input_is_reported(self):
        with self.assertRaises(CppCoreError) as ctx:
            self.core.run_raw(["fft"], "4\n1 0\n2 0\n")
        self.assertIn("malformed", str(ctx.exception))

    def test_garbage_input_is_reported(self):
        with self.assertRaises(CppCoreError):
            self.core.run_raw(["cfar"], "abc\n")

    def test_unknown_command_and_bad_bench_arguments(self):
        with self.assertRaises(CppCoreError):
            self.core.run_raw(["nope"], "")
        with self.assertRaises(CppCoreError):
            self.core.run_raw(["bench", "fft", "0"], "1\n1 0\n")
        with self.assertRaises(CppCoreError):
            self.core.run_raw(["bench", "fft", "abc"], "1\n1 0\n")
        with self.assertRaises(CppCoreError):
            self.core.run_raw(["bench", "unknown", "3"], "")

    def test_missing_binary_is_reported(self):
        with self.assertRaises(CppCoreError):
            CppCore("/nonexistent/path/sigcore")

    def test_python_side_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            self.core.cfar_detect(np.array([1.0, np.nan, 2.0]), 2, 0, 5.0)


class BenchAndParityTests(CppTestCase):
    def test_bench_reports_ordered_percentiles_and_checksum(self):
        x = np.random.default_rng(13).normal(size=256)
        result = self.core.bench("fft", cpp_bridge.fft_payload(x), repeats=9)
        self.assertEqual(result["repeats"], 9.0)
        self.assertGreater(result["p50_us"], 0.0)
        self.assertGreaterEqual(result["p95_us"], result["p50_us"])
        expected_checksum = float(np.sum(np.abs(np.fft.fft(x))))
        self.assertAlmostEqual(result["checksum"], expected_checksum, delta=1e-6 * expected_checksum)

    def test_run_parity_report_is_ok(self):
        report = run_parity(self.core, seed=0)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["cfar_indices_equal"])
        self.assertLess(report["fft_max_abs_err"], 1e-9)


if __name__ == "__main__":
    unittest.main()
