import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from sigvision import cpp_bridge
from sigvision.cpp_bridge import CppToolchainMissing
from sigvision.experiment import main, run_spectral_trials, run_vision_trials


class ExperimentFunctionTests(unittest.TestCase):
    def test_spectral_trials_report_valid_rates_and_are_reproducible(self):
        first = run_spectral_trials(3, seed=0)
        second = run_spectral_trials(3, seed=0)
        self.assertEqual(first, second)
        self.assertEqual(first["trials"], 3.0)
        self.assertGreaterEqual(first["tone_recall"], 0.8)
        self.assertLessEqual(first["tone_recall"], 1.0)
        self.assertGreaterEqual(first["false_alarms_per_trial"], 0.0)

    def test_vision_trials_report_valid_rates(self):
        result = run_vision_trials(2, seed=0)
        self.assertEqual(result["trials"], 2.0)
        self.assertGreaterEqual(result["corner_recall"], 0.75)
        self.assertLessEqual(result["corner_recall"], 1.0)

    def test_trial_counts_must_be_positive(self):
        with self.assertRaises(ValueError):
            run_spectral_trials(0, seed=0)
        with self.assertRaises(ValueError):
            run_vision_trials(0, seed=0)


class ExperimentCliTests(unittest.TestCase):
    def test_missing_binary_with_require_cpp_exits_2(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["--quick", "--require-cpp", "--binary", "/nonexistent/sigcore"])
        self.assertEqual(code, 2)
        self.assertIn("C++ core unavailable", err.getvalue())

    def test_quick_run_without_cpp_still_reports_python_results(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--quick", "--binary", "/nonexistent/sigcore"])
        self.assertEqual(code, 0)
        self.assertIn("SKIPPED", out.getvalue())
        self.assertIn("tone_recall", out.getvalue())

    def test_end_to_end_quick_run_with_cpp_writes_json(self):
        try:
            cpp_bridge.ensure_built()
        except CppToolchainMissing as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--quick", "--require-cpp", "--json", str(target)])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("parity == OK", out.getvalue())
            report = json.loads(target.read_text())
        self.assertTrue(report["parity"]["ok"])
        self.assertIn("fft_4096", report["timing"])
        self.assertIn("cpp", report["timing"]["cfar_4096"])


if __name__ == "__main__":
    unittest.main()
