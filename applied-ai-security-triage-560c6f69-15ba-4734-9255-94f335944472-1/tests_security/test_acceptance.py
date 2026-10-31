import copy
import json
from pathlib import Path
import tempfile
import unittest

from portfolio.common import strict_json_loads
from portfolio.security_acceptance import run_acceptance, audit_acceptance
from portfolio.security_data import build_dataset


class ServiceAcceptanceTests(unittest.TestCase):
    def test_concurrent_http_queue_recovery_and_plan_boundaries(self):
        cases = build_dataset()["train"]
        with tempfile.TemporaryDirectory() as out:
            record = run_acceptance(cases, out)
            saved = json.loads((Path(out)/"acceptance.json").read_text())
            self.assertEqual(saved, record)
            self.assertEqual(audit_acceptance(cases, record), {
                "jobs": 32, "workers": 4, "submissions": 64, "duplicate_requests": 32,
                "recovered_jobs": 1, "claims": 33, "rejected_requests": 7,
                "rejected_fixture_plans": 5, "stale_writes_rejected": 3, "mutation_isolated": True})
            self.assertEqual(list(Path(out).iterdir()), [Path(out)/"acceptance.json"])

    def test_audit_rejects_changed_findings_traces_statuses_and_counts(self):
        cases = build_dataset()["train"][:8]
        with tempfile.TemporaryDirectory() as out:
            record = run_acceptance(cases, out)
        for corruption in ("finding", "trace", "http", "count", "fixture", "stale"):
            bad = copy.deepcopy(record)
            if corruption == "finding":
                bad["jobs"][0]["job"]["result"]["structured_answer"]["finding"] = "Invented"
            elif corruption == "trace":
                bad["jobs"][0]["job"]["trace"].pop()
            elif corruption == "http":
                bad["negative_requests"][0]["status"] = 202
            elif corruption == "count":
                bad["summary"]["rejected_fixture_plans"] += 1
            elif corruption == "fixture":
                bad["plan_contracts"][0]["result"]["structured_answer"]["status"] = "completed"
            else:
                bad["stale_rejections"]["complete"] = False
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "Acceptance mismatch"):
                audit_acceptance(cases, bad)

    def test_acceptance_population_budget_is_enforced(self):
        cases = build_dataset()["train"]
        with tempfile.TemporaryDirectory() as out:
            for invalid in ([], cases + [cases[0]], [cases[0], cases[0]]):
                with self.assertRaisesRegex(ValueError, "1..32"):
                    run_acceptance(invalid, out)
            self.assertEqual(list(Path(out).iterdir()), [])


class JsonBoundaryTests(unittest.TestCase):
    def test_nested_duplicates_nonfinite_numbers_and_depth_are_rejected(self):
        self.assertEqual(strict_json_loads('{"evidence":[{"count":8}]}'), {"evidence": [{"count": 8}]})
        for raw in ('{"alert":{"complete":false,"complete":true}}', '[NaN]', '[Infinity]',
                    '[1e999]', '[' * 40 + '0' + ']' * 40, '[' * 995 + '0' + ']' * 995):
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                strict_json_loads(raw)
