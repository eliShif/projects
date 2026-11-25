import unittest

import numpy as np

from sigvision.experiment import run_fir_trial
from sigvision.signals import hann_window, make_signal
from sigvision.spectral import (
    apply_fir,
    design_lowpass_fir,
    fft_radix2,
    power_spectrum,
    snr_db,
)


class SignalGenerationTests(unittest.TestCase):
    def test_same_seed_is_reproducible_and_different_seed_differs(self):
        a, _ = make_signal(256, 256.0, [(10.0, 1.0)], noise_std=0.5, seed=7)
        b, _ = make_signal(256, 256.0, [(10.0, 1.0)], noise_std=0.5, seed=7)
        c, _ = make_signal(256, 256.0, [(10.0, 1.0)], noise_std=0.5, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_noiseless_signal_equals_clean(self):
        noisy, clean = make_signal(64, 64.0, [(5.0, 2.0)])
        np.testing.assert_array_equal(noisy, clean)
        peak = float(np.max(np.abs(clean)))
        self.assertLessEqual(peak, 2.0 + 1e-12)
        self.assertGreater(peak, 1.8)

    def test_rejects_frequency_at_or_above_nyquist(self):
        with self.assertRaises(ValueError):
            make_signal(64, 100.0, [(50.0, 1.0)])
        with self.assertRaises(ValueError):
            make_signal(64, 100.0, [(-1.0, 1.0)])

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            make_signal(0, 100.0, [])
        with self.assertRaises(ValueError):
            make_signal(8, 0.0, [])
        with self.assertRaises(ValueError):
            make_signal(8, 100.0, [], noise_std=-1.0)

    def test_hann_window_shape_and_validation(self):
        w = hann_window(8, periodic=True)
        self.assertAlmostEqual(float(w[0]), 0.0)
        self.assertAlmostEqual(float(w[4]), 1.0)
        sym = hann_window(9, periodic=False)
        np.testing.assert_allclose(sym, sym[::-1], atol=1e-15)
        with self.assertRaises(ValueError):
            hann_window(0)


class FftTests(unittest.TestCase):
    def test_matches_numpy_for_several_sizes(self):
        rng = np.random.default_rng(1)
        for n in (1, 2, 4, 8, 64, 512):
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            np.testing.assert_allclose(fft_radix2(x), np.fft.fft(x), atol=1e-9)

    def test_impulse_gives_flat_spectrum(self):
        x = np.zeros(16)
        x[0] = 1.0
        np.testing.assert_allclose(fft_radix2(x), np.ones(16), atol=1e-12)

    def test_rejects_non_power_of_two_empty_and_non_1d(self):
        for bad in (np.zeros(3), np.zeros(12), np.zeros(0), np.zeros((4, 4))):
            with self.assertRaises(ValueError):
                fft_radix2(bad)


class PowerSpectrumTests(unittest.TestCase):
    def test_on_bin_tone_peak_and_scale(self):
        n = 1024
        x, _ = make_signal(n, float(n), [(100.0, 1.0)])
        spec = power_spectrum(x)
        self.assertEqual(spec.size, n // 2 + 1)
        self.assertEqual(int(np.argmax(spec)), 100)
        # Hann coherent gain 0.5: |X| = A * N / 4 for a sine of amplitude A.
        self.assertAlmostEqual(float(np.sqrt(spec[100])), n / 4.0, delta=0.5)

    def test_hann_suppresses_leakage_versus_rectangular(self):
        n = 1024
        x, _ = make_signal(n, float(n), [(100.5, 1.0)])
        far = 100 + 20
        hann = power_spectrum(x, "hann")
        rect = power_spectrum(x, "rect")
        hann_ratio = hann[far] / hann.max()
        rect_ratio = rect[far] / rect.max()
        self.assertLess(hann_ratio, rect_ratio * 0.01)

    def test_rejects_unknown_window_and_bad_length(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.zeros(16), window="blackman")
        with self.assertRaises(ValueError):
            power_spectrum(np.zeros(10))


class FirTests(unittest.TestCase):
    def test_design_properties_and_frequency_response(self):
        h = design_lowpass_fir(101, 100.0, 1024.0)
        self.assertAlmostEqual(float(h.sum()), 1.0, places=12)
        np.testing.assert_allclose(h, h[::-1], atol=1e-15)
        padded = np.concatenate([h, np.zeros(1024 - h.size)])
        gain = np.abs(fft_radix2(padded))  # bin k == k Hz at fs = 1024
        self.assertAlmostEqual(float(gain[50]), 1.0, delta=0.02)
        self.assertLess(float(gain[300]), 0.02)

    def test_design_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            design_lowpass_fir(100, 100.0, 1000.0)  # even taps
        with self.assertRaises(ValueError):
            design_lowpass_fir(1, 100.0, 1000.0)
        with self.assertRaises(ValueError):
            design_lowpass_fir(51, 600.0, 1000.0)  # above Nyquist
        with self.assertRaises(ValueError):
            design_lowpass_fir(51, 0.0, 1000.0)

    def test_apply_fir_validation_and_length(self):
        h = design_lowpass_fir(11, 100.0, 1000.0)
        self.assertEqual(apply_fir(np.ones(50), h).size, 50)
        with self.assertRaises(ValueError):
            apply_fir(np.ones(5), h)
        with self.assertRaises(ValueError):
            apply_fir(np.ones(50), np.ones(4))

    def test_fir_improves_snr_on_synthetic_noise(self):
        result = run_fir_trial(seed=0)
        self.assertGreater(result["improvement_db"], 5.0)
        self.assertGreater(result["snr_after_db"], result["snr_before_db"])


class SnrTests(unittest.TestCase):
    def test_known_ratio_and_perfect_estimate(self):
        ref = np.ones(100)
        self.assertAlmostEqual(snr_db(ref, ref * 1.1), 20.0, places=6)
        self.assertEqual(snr_db(ref, ref), float("inf"))

    def test_validation(self):
        with self.assertRaises(ValueError):
            snr_db(np.zeros(10), np.ones(10))
        with self.assertRaises(ValueError):
            snr_db(np.ones(10), np.ones(9))
        with self.assertRaises(ValueError):
            snr_db(np.ones(10), np.ones(10), trim=5)


if __name__ == "__main__":
    unittest.main()
