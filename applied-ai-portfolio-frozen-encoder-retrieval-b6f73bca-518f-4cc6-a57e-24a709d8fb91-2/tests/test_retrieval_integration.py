"""Real TF-IDF end-to-end checks. No encoder weights or network required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from portfolio.common import sha256_json, write_json
from portfolio.retrieval import Retriever
from portfolio.retrieval_experiment import DEFAULT_DATA, ROOT, audit, run, smoke


class RetrievalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "run"

    def test_existing_tfidf_runs_and_saved_metrics_recompute(self):
        summary = run(DEFAULT_DATA, self.out, mode="baseline")
        self.assertEqual(summary["selection"]["selected"], "tfidf")
        self.assertFalse(summary["model_evaluated"])
        self.assertEqual(audit(self.out)["status"], "complete")
        self.assertIn("hypothesis remains untested", (self.out / "REPORT.md").read_text())
        self.assertEqual(smoke(baseline=True)["smoke"], "ok")

    def test_rehashed_tampered_metrics_still_fail_recomputation(self):
        run(DEFAULT_DATA, self.out, mode="baseline")
        rankings = json.loads((self.out / "rankings.json").read_text())
        row = rankings["test"]["tfidf"][0]
        row["mrr_at_k"] = 1.0 if row["mrr_at_k"] == 0 else 0.0
        write_json(self.out / "rankings.json", rankings)
        summary = json.loads((self.out / "summary.json").read_text())
        summary["rankings_sha256"] = sha256_json(rankings)
        write_json(self.out / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "metrics"):
            audit(self.out)

    def test_search_failure_keeps_protocol_but_no_success_marker(self):
        with patch.object(Retriever, "search", side_effect=RuntimeError("controlled test failure")):
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                run(DEFAULT_DATA, self.out, mode="baseline")
        self.assertTrue((self.out / "protocol.json").exists())
        self.assertTrue((self.out / "failure.json").exists())
        self.assertFalse((self.out / "summary.json").exists())
        with self.assertRaisesRegex(ValueError, "failed"):
            audit(self.out)

    def test_missing_local_model_fails_without_simulated_results(self):
        with self.assertRaises((FileNotFoundError, ImportError)):
            run(DEFAULT_DATA, self.out, mode="compare", model_path=Path(self.tmp.name) / "absent")
        self.assertFalse((self.out / "summary.json").exists())
        self.assertFalse((self.out / "rankings.json").exists())
        self.assertEqual(json.loads((self.out / "status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
