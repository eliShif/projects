import unittest

import numpy as np

from sigvision.detection import bins_to_hz, cfar_alpha, cfar_ca_detect, detect_tones
from sigvision.signals import make_signal


def flat_with_spike(n, pos, value):
    x = np.ones(n)
    x[pos] = value
    return x


class CfarAlphaTests(unittest.TestCase):
    def test_alpha_reproduces_requested_pfa(self):
        for pfa, train in ((1e-3, 16), (1e-6, 8)):
            alpha = cfar_alpha(pfa, train)
            n = 2 * train
            self.assertAlmostEqual((1 + alpha / n) ** (-n) / pfa, 1.0, places=9)

    def test_rejects_invalid_arguments(self):
        for pfa in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(ValueError):
                cfar_alpha(pfa, 8)
        with self.assertRaises(ValueError):
            cfar_alpha(1e-3, 0)


class CfarDetectTests(unittest.TestCase):
    def test_single_spike_detected_at_interior_and_both_edges(self):
        for pos in (32, 0, 63):
            det = cfar_ca_detect(flat_with_spike(64, pos, 100.0), 4, 1, 5.0)
            self.assertEqual(det.tolist(), [pos])

    def test_flat_and_empty_inputs_have_no_detections(self):
        self.assertEqual(cfar_ca_detect(np.ones(64), 4, 1, 5.0).size, 0)
        self.assertEqual(cfar_ca_detect(np.zeros(0), 4, 1, 5.0).size, 0)

    def test_guard_cells_prevent_target_masking(self):
        x = np.ones(64)
        x[32] = x[33] = 100.0
        self.assertEqual(cfar_ca_detect(x, 4, 0, 10.0).size, 0)
        self.assertEqual(cfar_ca_detect(x, 4, 1, 10.0).tolist(), [32])

    def test_threshold_adapts_to_noise_floor(self):
        quiet = flat_with_spike(64, 32, 30.0)
        loud = 10.0 * quiet  # same shape, noise floor 10x higher, spike 10x higher
        self.assertEqual(cfar_ca_detect(quiet, 4, 1, 5.0).tolist(), [32])
        self.assertEqual(cfar_ca_detect(loud, 4, 1, 5.0).tolist(), [32])
        # Raise only the floor: the same absolute spike is now below the threshold.
        raised = np.full(64, 10.0)
        raised[32] = 30.0
        self.assertEqual(cfar_ca_detect(raised, 4, 1, 5.0).size, 0)

    def test_false_alarm_rate_on_pure_noise_is_bounded(self):
        rng = np.random.default_rng(123)
        power = rng.exponential(1.0, size=20000)
        alpha = cfar_alpha(1e-3, 16)
        det = cfar_ca_detect(power, 16, 2, alpha)
        self.assertLess(det.size / power.size, 5e-3)

    def test_rejects_invalid_parameters(self):
        x = np.ones(16)
        with self.assertRaises(ValueError):
            cfar_ca_detect(x, 0, 1, 5.0)
        with self.assertRaises(ValueError):
            cfar_ca_detect(x, 4, -1, 5.0)
        with self.assertRaises(ValueError):
            cfar_ca_detect(x, 4, 1, 0.0)
        with self.assertRaises(ValueError):
            cfar_ca_detect(x, 4, 1, float("nan"))
        with self.assertRaises(ValueError):
            cfar_ca_detect(np.ones((4, 4)), 4, 1, 5.0)


class DetectTonesTests(unittest.TestCase):
    def test_two_tones_recovered_exactly_with_seeded_noise(self):
        x, _ = make_signal(1024, 1024.0, [(100.0, 1.0), (237.0, 1.0)], noise_std=0.5, seed=0)
        self.assertEqual(detect_tones(x).tolist(), [100, 237])

    def test_off_grid_tone_is_within_one_bin(self):
        x, _ = make_signal(1024, 1024.0, [(150.4, 1.0)], noise_std=0.5, seed=1)
        bins = detect_tones(x)
        self.assertEqual(bins.size, 1)
        self.assertLessEqual(abs(float(bins[0]) - 150.4), 1.0)

    def test_noise_only_signal_has_no_detections(self):
        x, _ = make_signal(1024, 1024.0, [], noise_std=1.0, seed=5)
        self.assertEqual(detect_tones(x).size, 0)

    def test_non_power_of_two_length_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_tones(np.zeros(1000))

    def test_bins_to_hz(self):
        np.testing.assert_allclose(bins_to_hz([0, 10], fs=2000.0, n=1000), [0.0, 20.0])
        with self.assertRaises(ValueError):
            bins_to_hz([1], fs=0.0, n=8)


if __name__ == "__main__":
    unittest.main()
