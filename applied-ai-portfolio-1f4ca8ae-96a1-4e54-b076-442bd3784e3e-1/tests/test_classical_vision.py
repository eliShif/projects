"""NumPy reference behaviour for the classical vision/signal module (needs numpy and scipy only)."""
import unittest

import numpy as np
from scipy import signal as scipy_signal

from portfolio import classical_vision as cv


class CorrelationTests(unittest.TestCase):
    def test_matches_scipy_for_odd_kernels_including_borders(self):
        rng = np.random.default_rng(0)
        for image_shape, kernel_shape in (((9, 11), (3, 3)), ((12, 7), (5, 5)), ((8, 8), (3, 5)), ((2, 2), (5, 5))):
            image, kernel = rng.normal(size=image_shape), rng.normal(size=kernel_shape)
            expected = scipy_signal.correlate2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
            with self.subTest(image=image_shape, kernel=kernel_shape):
                np.testing.assert_allclose(cv.correlate2d(image, kernel), expected, rtol=1e-12, atol=1e-12)

    def test_zero_padding_counts_only_pixels_inside_the_image(self):
        out = cv.correlate2d(np.ones((3, 3)), np.ones((3, 3)))
        np.testing.assert_array_equal(out, [[4, 6, 4], [6, 9, 6], [4, 6, 4]])

    def test_identity_kernel_returns_image(self):
        image = np.random.default_rng(1).normal(size=(5, 6))
        np.testing.assert_array_equal(cv.correlate2d(image, [[1.0]]), image)

    def test_invalid_inputs_are_rejected(self):
        good = np.ones((4, 4))
        for image, kernel in ((good, np.ones((2, 3))), (good, np.ones((3, 2))), (np.empty((0, 3)), np.ones((3, 3))),
                              (np.ones(5), np.ones((3, 3))), (good * np.nan, np.ones((3, 3))),
                              (good, np.full((3, 3), np.inf))):
            with self.subTest(image=image.shape, kernel=np.shape(kernel)), self.assertRaises(ValueError):
                cv.correlate2d(image, kernel)

    def test_gaussian_kernel_is_normalised_symmetric_and_validated(self):
        kernel = cv.gaussian_kernel(5, 1.0)
        self.assertAlmostEqual(kernel.sum(), 1.0, places=12)
        np.testing.assert_allclose(kernel, kernel.T)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
        self.assertEqual(np.unravel_index(kernel.argmax(), kernel.shape), (2, 2))
        for size, sigma in ((4, 1.0), (0, 1.0), (5, 0.0), (5, -1.0), (5, float('nan'))):
            with self.subTest(size=size, sigma=sigma), self.assertRaises(ValueError):
                cv.gaussian_kernel(size, sigma)


class EdgeTests(unittest.TestCase):
    def test_sobel_on_a_vertical_step_has_known_interior_response(self):
        image = np.zeros((4, 4))
        image[:, 2:] = 1.0
        magnitude = cv.sobel_magnitude(image)
        np.testing.assert_allclose(magnitude[1:3, 1], 4.0)
        np.testing.assert_allclose(magnitude[1:3, 2], 4.0)
        np.testing.assert_allclose(magnitude[1:3, 0], 0.0)

    def test_flat_image_has_no_edges_and_bad_fraction_is_rejected(self):
        self.assertFalse(cv.detect_edges(np.full((16, 16), 0.5)).any())
        self.assertFalse(cv.threshold_edges(np.zeros((3, 3))).any())
        for fraction in (0.0, -0.1, 1.5):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                cv.threshold_edges(np.ones((3, 3)), fraction)

    def test_flat_images_of_any_level_have_no_edges_with_or_without_blur(self):
        # Regression: zero padding used to create border gradients, and rounding residue was scaled up.
        for level in (0.0, 0.1, 0.5, 1.0, 250.0):
            for blur_size in (1, 3, 5):
                with self.subTest(level=level, blur_size=blur_size):
                    self.assertFalse(cv.detect_edges(np.full((16, 20), level), blur_size=blur_size).any())

    def test_threshold_floor_suppresses_tiny_peaks_and_is_validated(self):
        tiny = np.full((3, 3), 1e-15)
        tiny[1, 1] = 3e-15
        self.assertTrue(cv.threshold_edges(tiny).any())
        self.assertFalse(cv.threshold_edges(tiny, floor=1e-12).any())
        for floor in (-1.0, float('nan'), float('inf')):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                cv.threshold_edges(tiny, floor=floor)

    def test_image_border_is_not_reported_as_an_edge(self):
        image = np.full((24, 24), 0.7)
        image[:, 12:] = 0.2  # one vertical step; nothing should be reported near the other borders
        edges = cv.detect_edges(image)
        self.assertTrue(edges[:, 9:15].any())
        self.assertFalse(edges[:, :6].any())
        self.assertFalse(edges[:, 18:].any())

    def test_noise_free_and_lightly_noisy_shapes_are_located(self):
        for seed, noise in ((11, 0.0), (11, 0.02), (3, 0.02)):
            image, region = cv.synthetic_image(seed, noise=noise)
            scores = cv.edge_scores(cv.detect_edges(image), cv.boundary_mask(region), tolerance=2)
            with self.subTest(seed=seed, noise=noise):
                self.assertGreaterEqual(scores['f1'], 0.75, scores)

    def test_blurring_is_needed_under_heavy_noise(self):
        image, region = cv.synthetic_image(5, noise=0.3)
        truth = cv.boundary_mask(region)
        blurred = cv.edge_scores(cv.detect_edges(image), truth)['f1']
        unblurred = cv.edge_scores(cv.detect_edges(image, blur_size=1), truth)['f1']
        self.assertGreaterEqual(blurred, 0.6)
        self.assertLessEqual(unblurred, blurred - 0.2)

    def test_boundary_dilation_and_scores(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        boundary = cv.boundary_mask(mask)
        self.assertEqual(int(boundary.sum()), 8)
        self.assertFalse(boundary[3, 3])
        self.assertEqual(int(cv.dilate(np.eye(1, 1, dtype=bool), 0).sum()), 1)
        single = np.zeros((7, 7), dtype=bool)
        single[3, 3] = True
        self.assertEqual(int(cv.dilate(single, 2).sum()), 25)
        perfect = cv.edge_scores(boundary, boundary, tolerance=0)
        self.assertEqual((perfect['precision'], perfect['recall'], perfect['f1']), (1.0, 1.0, 1.0))
        none = cv.edge_scores(np.zeros_like(boundary), boundary)
        self.assertEqual((none['precision'], none['recall'], none['f1']), (0.0, 0.0, 0.0))
        shifted = np.roll(boundary, 1, axis=1)
        self.assertLess(cv.edge_scores(shifted, boundary, tolerance=0)['f1'], 1.0)
        self.assertEqual(cv.edge_scores(shifted, boundary, tolerance=1)['recall'], 1.0)

    def test_score_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            cv.edge_scores(np.ones((3, 3), bool), np.zeros((3, 3), bool))
        with self.assertRaises(ValueError):
            cv.edge_scores(np.ones((3, 3), bool), np.ones((4, 3), bool))
        with self.assertRaises(ValueError):
            cv.dilate(np.ones((3, 3), bool), -1)


class SignalTests(unittest.TestCase):
    def test_fir_impulse_response_and_causality(self):
        taps = np.array([0.5, 0.25, 0.125])
        impulse = np.zeros(6)
        impulse[0] = 1.0
        np.testing.assert_allclose(cv.fir_filter(impulse, taps), [0.5, 0.25, 0.125, 0, 0, 0])
        late = np.zeros(6)
        late[3] = 1.0
        np.testing.assert_allclose(cv.fir_filter(late, taps), [0, 0, 0, 0.5, 0.25, 0.125])
        self.assertEqual(cv.fir_filter([1.0, 2.0], taps).shape, (2,))

    def test_lowpass_taps_have_unit_dc_gain_and_linear_phase(self):
        taps = cv.lowpass_taps(63, 0.0625)
        self.assertAlmostEqual(taps.sum(), 1.0, places=12)
        np.testing.assert_allclose(taps, taps[::-1], atol=1e-15)
        for num_taps, cutoff in ((4, 0.1), (1, 0.1), (31, 0.0), (31, 0.5), (31, 0.7)):
            with self.subTest(num_taps=num_taps, cutoff=cutoff), self.assertRaises(ValueError):
                cv.lowpass_taps(num_taps, cutoff)

    def test_lowpass_removes_the_high_tone_and_keeps_the_low_tone(self):
        tones = ((8.0, 1.0), (40.0, 0.5))
        x = cv.synthetic_signal(0, tones=tones, noise=0.0)
        y = cv.fir_filter(x, cv.lowpass_taps(63, 0.0625))
        before, after = np.abs(cv.rfft(x)), np.abs(cv.rfft(y))
        self.assertLess(after[40] / before[40], 0.2)
        self.assertGreater(after[8] / before[8], 0.7)

    def test_dominant_frequency_of_default_signal(self):
        self.assertEqual(cv.dominant_frequency(cv.synthetic_signal(4), 256.0), 12.0)
        self.assertEqual(cv.dominant_frequency(cv.synthetic_signal(4), 512.0), 24.0)

    def test_fft_and_dominant_frequency_reject_bad_input(self):
        for bad in (np.ones(6), np.ones(0), np.ones((2, 2)), np.array([1.0, np.nan, 0.0, 0.0])):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                cv.rfft(bad)
        with self.assertRaises(ValueError):
            cv.dominant_frequency(np.ones(8), 0.0)
        with self.assertRaises(ValueError):
            cv.dominant_frequency(np.ones(1), 10.0)

    def test_stft_locates_the_tone_in_each_half(self):
        t = np.arange(256) / 256.0
        x = np.where(t < 0.5, np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 40 * t))
        spectrogram = cv.stft_magnitude(x, 64, 32)
        self.assertEqual(spectrogram.shape, (7, 33))
        self.assertEqual(int(spectrogram[0].argmax()), 2)
        self.assertEqual(int(spectrogram[-1].argmax()), 10)

    def test_stft_rejects_bad_shapes(self):
        x = np.ones(64)
        for frame, hop in ((48, 8), (128, 8), (16, 0), (16, -4), (16, 2.5)):
            with self.subTest(frame=frame, hop=hop), self.assertRaises(ValueError):
                cv.stft_magnitude(x, frame, hop)


class SyntheticDataTests(unittest.TestCase):
    def test_generation_is_deterministic_and_seed_dependent(self):
        a, mask_a = cv.synthetic_image(7)
        b, mask_b = cv.synthetic_image(7)
        c, _ = cv.synthetic_image(8)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask_a, mask_b)
        self.assertFalse(np.array_equal(a, c))
        np.testing.assert_array_equal(cv.synthetic_signal(1), cv.synthetic_signal(1))
        self.assertFalse(np.array_equal(cv.synthetic_signal(1), cv.synthetic_signal(2)))

    def test_generator_arguments_are_validated(self):
        for call in (lambda: cv.synthetic_image(0, size=8), lambda: cv.synthetic_image(0, noise=-1.0),
                     lambda: cv.synthetic_signal(0, n=1), lambda: cv.synthetic_signal(0, sample_rate=0),
                     lambda: cv.synthetic_activity_stack(0, frames=1),
                     lambda: cv.synthetic_activity_stack(0, roi_radius=20)):
            with self.assertRaises(ValueError):
                call()

    def test_time_activity_curve_recovers_the_designed_peak(self):
        stack, mask, designed = cv.synthetic_activity_stack(2, frames=24, peak_frame=6.0)
        curves = cv.time_activity_curve(stack, mask)
        self.assertLessEqual(abs(int(curves['roi'].argmax()) - 6), 1)
        self.assertGreater(curves['contrast'].max(), 0.5 * designed.max())
        self.assertLess(abs(curves['outside'].mean() - 0.2), 0.05)
        np.testing.assert_allclose(curves['contrast'], curves['roi'] - curves['outside'])

    def test_time_activity_curve_rejects_bad_masks_and_stacks(self):
        stack, mask, _ = cv.synthetic_activity_stack(0)
        for bad_stack, bad_mask in ((stack, mask[:-1]), (stack, np.zeros_like(mask)), (stack, np.ones_like(mask)),
                                    (stack[0], mask), (stack * np.nan, mask)):
            with self.subTest(stack=bad_stack.shape), self.assertRaises(ValueError):
                cv.time_activity_curve(bad_stack, bad_mask)


if __name__ == '__main__':
    unittest.main()
