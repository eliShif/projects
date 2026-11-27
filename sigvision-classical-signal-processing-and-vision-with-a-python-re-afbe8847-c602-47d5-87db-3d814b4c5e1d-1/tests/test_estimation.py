import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sigvision.estimation import (
    estimate_noise_variance,
    estimate_tone_frequencies,
    estimate_tone_frequency,
    kalman_cv_filter,
    main,
    monte_carlo_cfar_false_alarm,
    monte_carlo_noise_estimator,
    monte_carlo_tone_frequency,
    monte_carlo_tracking,
    nees,
    noise_estimator_theory,
    run_estimation_suite,
    simulate_cv_track,
    tone_frequency_crb_std_hz,
)
from sigvision.experiment import main as experiment_main
from sigvision.signals import make_signal


class NoiseEstimatorTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(estimate_noise_variance([1.0, -1.0, 1.0, -1.0]), 1.0)
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(estimate_noise_variance(data, "sample_unbiased"), 5.0 / 3.0)
        self.assertAlmostEqual(estimate_noise_variance(data, "sample_mle"), 1.25)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_noise_variance(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            estimate_noise_variance([1.0, float("nan")])
        with self.assertRaises(ValueError):
            estimate_noise_variance([])
        with self.assertRaises(ValueError):
            estimate_noise_variance([1.0], "sample_unbiased")
        with self.assertRaises(ValueError):
            estimate_noise_variance([1.0, 2.0], "median")

    def test_theory_values(self):
        t = noise_estimator_theory(2.0, 8, "zero_mean")
        self.assertEqual((t["bias"], t["variance"]), (0.0, 4.0))
        t = noise_estimator_theory(2.0, 8, "sample_unbiased")
        self.assertAlmostEqual(t["variance"], 2.0 * 16.0 / 7.0)
        t = noise_estimator_theory(2.0, 8, "sample_mle")
        self.assertAlmostEqual(t["bias"], -0.5)
        self.assertAlmostEqual(t["variance"], 3.5)
        self.assertAlmostEqual(t["rmse"], float(np.sqrt(3.5 + 0.25)))
        with self.assertRaises(ValueError):
            noise_estimator_theory(0.0, 8)
        with self.assertRaises(ValueError):
            noise_estimator_theory(1.0, 1, "sample_mle")

    def test_zero_mean_matches_theory(self):
        r = monte_carlo_noise_estimator(64, 1.0, 2000, seed=3)
        self.assertLess(abs(r["bias"]), 0.02)
        self.assertGreater(r["variance_ratio"], 0.8)
        self.assertLess(r["variance_ratio"], 1.2)
        self.assertAlmostEqual(r["theory_rmse"], float(np.sqrt(2.0 / 64.0)))

    def test_mle_is_biased_low_and_unbiased_variant_is_not(self):
        mle = monte_carlo_noise_estimator(16, 1.0, 4000, seed=5, mode="sample_mle")
        unb = monte_carlo_noise_estimator(16, 1.0, 4000, seed=5, mode="sample_unbiased")
        self.assertLess(mle["bias"], 0.0)
        self.assertLess(abs(mle["bias"] - mle["theory_bias"]), 0.02)
        self.assertLess(abs(unb["bias"]), 0.03)

    def test_seed_dependence(self):
        a = monte_carlo_noise_estimator(16, 1.0, 100, seed=1)
        b = monte_carlo_noise_estimator(16, 1.0, 100, seed=1)
        c = monte_carlo_noise_estimator(16, 1.0, 100, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a["bias"], c["bias"])

    def test_monte_carlo_invalid_inputs(self):
        with self.assertRaises(ValueError):
            monte_carlo_noise_estimator(16, 1.0, 1)
        with self.assertRaises(ValueError):
            monte_carlo_noise_estimator(16, -1.0, 100)
        with self.assertRaises(ValueError):
            monte_carlo_noise_estimator(True, 1.0, 100)


class CfarNoiseEstimateTests(unittest.TestCase):
    def test_empirical_pfa_matches_nominal_and_noise_estimate_spread(self):
        r = monte_carlo_cfar_false_alarm(16, 1e-2, 20000, seed=7)
        self.assertLess(abs(r["pfa_empirical"] - 1e-2), 0.003)
        self.assertLess(abs(r["noise_estimate_bias"]), 0.01)
        ratio = r["noise_estimate_rel_std"] / r["noise_estimate_rel_std_theory"]
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            monte_carlo_cfar_false_alarm(16, 0.0, 100)
        with self.assertRaises(ValueError):
            monte_carlo_cfar_false_alarm(0, 1e-2, 100)
        with self.assertRaises(ValueError):
            monte_carlo_cfar_false_alarm(16, 1e-2, 1)


class ToneFrequencyTests(unittest.TestCase):
    N = 256
    FS = 256.0  # one bin == 1 Hz

    def _tone(self, freq):
        sig, _ = make_signal(self.N, self.FS, [(freq, 1.0)])
        return sig

    def test_bin_centred_tone_is_exact(self):
        self.assertAlmostEqual(estimate_tone_frequency(self._tone(40.0), self.FS), 40.0, places=6)

    def test_noiseless_off_grid_error_is_below_a_few_hundredths_of_a_bin(self):
        for offset in np.linspace(-0.45, 0.45, 19):
            est = estimate_tone_frequency(self._tone(40.0 + offset), self.FS)
            self.assertLess(abs(est - (40.0 + offset)), 0.03, msg=f"offset {offset}")

    def test_scales_with_sampling_rate(self):
        sig, _ = make_signal(self.N, 1000.0, [(150.0, 1.0)])
        est = estimate_tone_frequency(sig, 1000.0)
        self.assertLess(abs(est - 150.0), 0.03 * 1000.0 / self.N)

    def test_two_detected_tones_are_refined(self):
        sig, _ = make_signal(1024, 1024.0, [(40.3, 1.0), (300.7, 1.0)], noise_std=0.5, seed=4)
        est = estimate_tone_frequencies(sig, 1024.0)
        self.assertEqual(est.size, 2)
        self.assertLess(abs(est[0] - 40.3), 0.1)
        self.assertLess(abs(est[1] - 300.7), 0.1)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_tone_frequency(np.zeros(self.N), self.FS)  # no spectral peak
        with self.assertRaises(ValueError):
            estimate_tone_frequency(np.ones(100), self.FS)  # not a power of two
        with self.assertRaises(ValueError):
            estimate_tone_frequency(np.ones((4, 4)), self.FS)
        with self.assertRaises(ValueError):
            estimate_tone_frequency(np.ones(4), self.FS)  # too short
        with self.assertRaises(ValueError):
            estimate_tone_frequency(self._tone(40.0), 0.0)
        bad = self._tone(40.0)
        bad[3] = float("nan")
        with self.assertRaises(ValueError):
            estimate_tone_frequency(bad, self.FS)
        with self.assertRaises(ValueError):
            estimate_tone_frequencies(bad, self.FS)

    def test_crb_known_value_and_scaling(self):
        # 12 / ((2 pi)^2 * 1 * 2 * 3) = 0.05066 cycles^2/sample^2 -> std 0.22508 cycles/sample
        self.assertAlmostEqual(tone_frequency_crb_std_hz(0.0, 2, 1.0), 0.22508, places=4)
        low = tone_frequency_crb_std_hz(0.0, 256, 256.0)
        high = tone_frequency_crb_std_hz(20.0, 256, 256.0)
        self.assertAlmostEqual(low / high, 10.0, places=9)
        self.assertAlmostEqual(tone_frequency_crb_std_hz(0.0, 256, 512.0) / low, 2.0, places=9)

    def test_crb_invalid_inputs(self):
        with self.assertRaises(ValueError):
            tone_frequency_crb_std_hz(float("inf"), 256, 256.0)
        with self.assertRaises(ValueError):
            tone_frequency_crb_std_hz(10.0, 1, 256.0)
        with self.assertRaises(ValueError):
            tone_frequency_crb_std_hz(10.0, 256, -1.0)

    def test_monte_carlo_is_consistent_with_the_bound(self):
        low = monte_carlo_tone_frequency(self.N, self.FS, 0.0, 120, seed=1)
        high = monte_carlo_tone_frequency(self.N, self.FS, 20.0, 120, seed=1)
        for r in (low, high):
            # A biased, non-efficient estimator must not beat the bound and should
            # stay within a small multiple of it.
            self.assertGreater(r["rmse_over_crb"], 0.8)
            self.assertLess(r["rmse_over_crb"], 10.0)
            self.assertGreaterEqual(r["rmse_hz"], abs(r["bias_hz"]))
        self.assertLess(high["rmse_hz"], low["rmse_hz"])
        self.assertLess(high["crb_std_hz"], low["crb_std_hz"])

    def test_monte_carlo_reproducible_and_seed_dependent(self):
        a = monte_carlo_tone_frequency(self.N, self.FS, 10.0, 30, seed=2)
        b = monte_carlo_tone_frequency(self.N, self.FS, 10.0, 30, seed=2)
        c = monte_carlo_tone_frequency(self.N, self.FS, 10.0, 30, seed=3)
        self.assertEqual(a, b)
        self.assertNotEqual(a["rmse_hz"], c["rmse_hz"])

    def test_monte_carlo_invalid_inputs(self):
        with self.assertRaises(ValueError):
            monte_carlo_tone_frequency(100, self.FS, 10.0, 10)  # not a power of two
        with self.assertRaises(ValueError):
            monte_carlo_tone_frequency(8, self.FS, 10.0, 10)  # too short
        with self.assertRaises(ValueError):
            monte_carlo_tone_frequency(self.N, self.FS, 10.0, 1)


class KalmanTests(unittest.TestCase):
    def test_simulation_shapes_and_seed(self):
        t1, m1 = simulate_cv_track(30, 1.0, 0.1, 5.0, 1.0, seed=4)
        t2, m2 = simulate_cv_track(30, 1.0, 0.1, 5.0, 1.0, seed=4)
        t3, _ = simulate_cv_track(30, 1.0, 0.1, 5.0, 1.0, seed=5)
        self.assertEqual(t1.shape, (30, 4))
        self.assertEqual(m1.shape, (30, 2))
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(m1, m2)
        self.assertFalse(np.array_equal(t1, t3))
        # positions advance by velocity (+ acceleration noise): with accel_std = 0 exactly.
        t0, _ = simulate_cv_track(10, 2.0, 0.0, 1.0, 1.0, seed=1)
        np.testing.assert_allclose(t0[1:, :2] - t0[:-1, :2], 2.0 * t0[:-1, 2:])

    def test_noise_free_line_is_recovered(self):
        k = np.arange(40.0)
        meas = np.stack([2.0 * k, -1.0 * k], axis=1)
        xs, _ = kalman_cv_filter(meas, 1.0, 0.0, 0.01, 1.0)
        np.testing.assert_allclose(xs[-1], [78.0, -39.0, 2.0, -1.0], atol=0.05)

    def test_covariances_are_symmetric_positive_definite_and_shrink(self):
        _, meas = simulate_cv_track(50, 1.0, 0.1, 5.0, 1.0, seed=2)
        _, ps = kalman_cv_filter(meas, 1.0, 0.1, 5.0, 1.0)
        for p in ps:
            np.testing.assert_allclose(p, p.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(p).min(), 0.0)
        self.assertLess(ps[-1, 0, 0], ps[0, 0, 0])
        self.assertLess(ps[-1, 2, 2], ps[0, 2, 2])

    def test_invalid_inputs(self):
        good = np.zeros((5, 2))
        with self.assertRaises(ValueError):
            kalman_cv_filter(np.zeros((5, 3)), 1.0, 0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kalman_cv_filter(np.zeros((1, 2)), 1.0, 0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kalman_cv_filter(np.full((5, 2), np.nan), 1.0, 0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kalman_cv_filter(good, 0.0, 0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kalman_cv_filter(good, 1.0, -0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kalman_cv_filter(good, 1.0, 0.1, 0.0, 1.0)
        with self.assertRaises(ValueError):
            kalman_cv_filter(good, 1.0, 0.1, 1.0, 0.0)
        with self.assertRaises(ValueError):
            simulate_cv_track(1, 1.0, 0.1, 1.0, 1.0)

    def test_nees_known_values_and_shape_errors(self):
        truth = np.zeros((3, 4))
        est = np.zeros((3, 4))
        est[:, 0] = 2.0
        covs = np.stack([4.0 * np.eye(4)] * 3)
        np.testing.assert_allclose(nees(truth, est, covs), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(nees(truth, truth, covs), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            nees(truth, est[:2], covs)
        with self.assertRaises(ValueError):
            nees(truth, est, covs[:2])

    def test_matched_filter_beats_raw_measurements_and_is_consistent(self):
        r = monte_carlo_tracking(60, 60, seed=11)
        self.assertLess(r["filter_over_raw_rmse"], 0.8)
        self.assertLess(abs(r["raw_pos_rmse"] / r["raw_pos_rmse_theory"] - 1.0), 0.1)
        self.assertGreater(r["mean_nees"], 2.5)
        self.assertLess(r["mean_nees"], 5.5)

    def test_mismatched_process_model_is_flagged_by_nees(self):
        r = monte_carlo_tracking(30, 60, accel_std=1.0, filter_accel_std=0.01, seed=11)
        self.assertGreater(r["mean_nees"], 2.0 * r["nees_expected"])

    def test_tracking_reproducible_and_validated(self):
        a = monte_carlo_tracking(5, 30, seed=1, burn_in=5)
        b = monte_carlo_tracking(5, 30, seed=1, burn_in=5)
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            monte_carlo_tracking(0, 30)
        with self.assertRaises(ValueError):
            monte_carlo_tracking(5, 30, burn_in=30)
        with self.assertRaises(ValueError):
            monte_carlo_tracking(5, 1)


class SuiteAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_estimation_suite(quick=True, seed=0)

    def test_suite_structure_and_finite_values(self):
        self.assertEqual(set(self.report["noise"]), {"zero_mean", "sample_unbiased", "sample_mle"})
        self.assertEqual(len(self.report["tone_frequency"]), 3)
        self.assertEqual(
            set(self.report["tracking"]), {"matched", "mismatched_filter_accel_0.01"}
        )
        text = json.dumps(self.report, default=float)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_suite_is_reproducible(self):
        self.assertEqual(run_estimation_suite(quick=True, seed=0), self.report)

    def test_cli_prints_and_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "estimation.json"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--quick", "--seed", "0", "--json", str(target)])
            self.assertEqual(code, 0)
            self.assertIn("Tone frequency", out.getvalue())
            self.assertIn("mean_nees", out.getvalue())
            written = json.loads(target.read_text())
        self.assertEqual(written["seed"], 0)
        self.assertIn("matched", written["tracking"])

    def test_cli_rejects_bad_arguments(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            main(["--seed", "not-an-int"])
        self.assertEqual(ctx.exception.code, 2)

    def test_experiment_cli_flag_adds_estimation_section(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = experiment_main(
                ["--quick", "--binary", "/nonexistent/sigcore", "--estimation"]
            )
        self.assertEqual(code, 0)
        self.assertIn("== Estimation", out.getvalue())
        self.assertIn("tone_recall", out.getvalue())


if __name__ == "__main__":
    unittest.main()
