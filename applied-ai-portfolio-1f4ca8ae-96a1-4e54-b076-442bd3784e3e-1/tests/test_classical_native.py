"""C++ kernels versus the NumPy reference. Skipped only if g++ and a prebuilt binary are both absent."""
import subprocess
import unittest

import numpy as np

from portfolio import classical_native as native
from portfolio import classical_vision as cv

TOL = {'rtol': 1e-9, 'atol': 1e-9}


def setUpModule():
    try:
        native.binary_path()
    except native.NativeUnavailable as exc:
        raise unittest.SkipTest('native kernels unavailable: %s' % exc)


class EquivalenceTests(unittest.TestCase):
    def test_correlate2d_matches_reference_on_varied_shapes(self):
        rng = np.random.default_rng(10)
        for image_shape, kernel_shape in (((1, 1), (1, 1)), ((1, 9), (3, 3)), ((7, 1), (3, 1)), ((13, 10), (5, 3)),
                                          ((2, 2), (5, 5)), ((24, 24), (7, 7))):
            image, kernel = rng.normal(size=image_shape) * 100, rng.normal(size=kernel_shape)
            with self.subTest(image=image_shape, kernel=kernel_shape):
                np.testing.assert_allclose(native.correlate2d(image, kernel).value, cv.correlate2d(image, kernel), **TOL)

    def test_sobel_matches_reference(self):
        image, _ = cv.synthetic_image(2)
        np.testing.assert_allclose(native.sobel_magnitude(image).value, cv.sobel_magnitude(image), **TOL)
        small = np.arange(6.0).reshape(2, 3)
        np.testing.assert_allclose(native.sobel_magnitude(small).value, cv.sobel_magnitude(small), **TOL)

    def test_fir_matches_reference_including_taps_longer_than_signal(self):
        rng = np.random.default_rng(11)
        for n, m in ((1, 1), (5, 9), (256, 63), (64, 1)):
            x, h = rng.normal(size=n), rng.normal(size=m)
            with self.subTest(n=n, taps=m):
                np.testing.assert_allclose(native.fir_filter(x, h).value, cv.fir_filter(x, h), **TOL)

    def test_fft_matches_reference_for_power_of_two_lengths(self):
        rng = np.random.default_rng(12)
        for n in (1, 2, 4, 8, 64, 256, 1024):
            x = rng.normal(size=n)
            with self.subTest(n=n):
                np.testing.assert_allclose(native.rfft(x).value, cv.rfft(x), **TOL)

    def test_stft_matches_reference_including_single_frame_and_uneven_hop(self):
        rng = np.random.default_rng(13)
        for n, frame, hop in ((256, 64, 32), (100, 16, 7), (64, 64, 1), (33, 1, 5)):
            x = rng.normal(size=n)
            with self.subTest(n=n, frame=frame, hop=hop):
                np.testing.assert_allclose(native.stft_magnitude(x, frame, hop).value,
                                           cv.stft_magnitude(x, frame, hop), **TOL)

    def test_text_protocol_round_trips_doubles_exactly(self):
        x = np.random.default_rng(14).normal(size=(6, 7)) * 10.0 ** np.random.default_rng(15).integers(-8, 8, (6, 7))
        np.testing.assert_array_equal(native.correlate2d(x, [[1.0]]).value, x)

    def test_edge_pipeline_gives_identical_maps_on_both_backends(self):
        for seed in (1, 2):
            image, _ = cv.synthetic_image(seed)
            with self.subTest(seed=seed):
                np.testing.assert_array_equal(cv.detect_edges(image, native.NATIVE), cv.detect_edges(image, cv.NUMPY))

    def test_flat_image_gives_no_edges_on_the_native_backend_too(self):
        for level in (0.0, 0.5, 250.0):
            with self.subTest(level=level):
                self.assertFalse(cv.detect_edges(np.full((12, 12), level), native.NATIVE).any())

    def test_dominant_frequency_agrees(self):
        x = cv.synthetic_signal(6)
        self.assertEqual(cv.dominant_frequency(x, 256.0, native.NATIVE), cv.dominant_frequency(x, 256.0, cv.NUMPY))


class TimingTests(unittest.TestCase):
    def test_repeat_reports_count_and_does_not_change_results(self):
        x = cv.synthetic_signal(1)
        once, many = native.rfft(x), native.rfft(x, repeat=5)
        self.assertEqual((once.repeats, many.repeats), (1, 5))
        np.testing.assert_array_equal(once.value, many.value)
        self.assertGreaterEqual(many.kernel_ns_per_call, 0.0)

    def test_invalid_repeat_is_rejected_before_and_inside_the_program(self):
        with self.assertRaises(ValueError):
            native.rfft(np.ones(4), repeat=0)
        completed = subprocess.run([str(native.binary_path()), 'fft', '--repeat', '0'], input='4 1 2 3 4',
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('repeat', completed.stderr)


class RejectionTests(unittest.TestCase):
    def assert_rejected(self, command, text, fragment):
        with self.assertRaises(native.NativeError) as caught:
            native.run_raw(command, text)
        self.assertIn('exit 2', str(caught.exception))
        self.assertIn(fragment, str(caught.exception))

    def test_malformed_inputs_fail_with_a_clear_message(self):
        cases = (
            ('conv2d', '2 2 2 2 1 2 3 4 1 1 1 1', 'odd'),
            ('conv2d', '0 3 1 1', 'positive'),
            ('conv2d', '-2 3 1 1', 'positive'),
            ('conv2d', '100000 100000 1 1', 'too large'),
            ('conv2d', '2 2 1 1 1 2 3', 'truncated'),
            ('conv2d', '1 1 1 1 5 2 99', 'trailing'),
            ('sobel', '1 2 1 nan', 'invalid number'),
            ('sobel', '1 2 1 inf', 'invalid number'),
            ('sobel', '1 2 1 abc', 'invalid number'),
            ('sobel', 'x 2 1 1', 'invalid integer'),
            ('fir', '0 1', 'nonempty'),
            ('fir', '3 0 1 2 3', 'nonempty'),
            ('fft', '6 1 2 3 4 5 6', 'power of two'),
            ('fft', '0', 'power of two'),
            ('stft', '8 16 2 1 2 3 4 5 6 7 8', 'exceeds'),
            ('stft', '8 6 2 1 2 3 4 5 6 7 8', 'power of two'),
            ('stft', '8 4 0 1 2 3 4 5 6 7 8', 'hop'),
            ('rotate', '1', 'unknown command'),
            ('fft', '', 'truncated'),
        )
        for command, text, fragment in cases:
            with self.subTest(command=command, text=text):
                self.assert_rejected(command, text, fragment)

    def test_command_line_misuse_exits_with_code_2(self):
        for argv in ([], ['fft', '--bogus', '1'], ['fft', '--repeat', 'x'], ['fft', 'extra']):
            completed = subprocess.run([str(native.binary_path()), *argv], input='', capture_output=True,
                                       text=True, timeout=30)
            with self.subTest(argv=argv):
                self.assertEqual(completed.returncode, 2)
                self.assertTrue(completed.stderr.startswith('error:'))

    def test_python_wrappers_reject_bad_arrays_before_starting_the_program(self):
        for call in (lambda: native.correlate2d(np.ones((3, 3)), np.ones((2, 2))),
                     lambda: native.correlate2d(np.full((3, 3), np.nan), np.ones((3, 3))),
                     lambda: native.fir_filter([], [1.0]),
                     lambda: native.rfft(np.ones(12)),
                     lambda: native.stft_magnitude(np.ones(8), 16, 2),
                     lambda: native.sobel_magnitude(np.ones(4))):
            with self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
