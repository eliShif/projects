"""Real HTTP parser/handler over byte streams; no socket transport is exercised."""
import json
from pathlib import Path
import tempfile
import unittest

from portfolio.reliable_service import JobStore, work_once
from portfolio.security_acceptance import request
from portfolio.security_data import build_dataset
from portfolio.security_triage import SecurityTriageEngine, RulePlanner

class HttpHandlerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.tmp.name)/"http.db")
        self.case = build_dataset()["test"][0]
        self.payload = {"mode": "security_triage", "alert": self.case["alert"]}
    def tearDown(self):
        self.tmp.cleanup()

    def test_auth_duplicate_worker_and_retrieved_result(self):
        self.assertEqual(request(self.store, self.payload, authorized=False)[0], 401)
        self.assertEqual(self.store.metrics()["total_attempts"], 0)
        code, created = request(self.store, self.payload)
        self.assertEqual(code, 202)
        code, duplicate = request(self.store, self.payload)
        self.assertEqual(code, 200)
        self.assertEqual(created["id"], duplicate["id"])
        self.assertTrue(work_once(self.store, SecurityTriageEngine(RulePlanner(), self.case["events"])))
        code, result = request(self.store, path="/jobs/" + created["id"])
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["result"]["structured_answer"]["disposition"], "escalate")
        self.assertEqual(request(self.store, path="/jobs/" + created["id"], authorized=False)[0], 401)

    def test_conflicts_malformed_inputs_and_training_escalation_rejected(self):
        request(self.store, self.payload)
        changed = {"mode": "security_triage", "alert": {**self.case["alert"], "note": "different"}}
        self.assertEqual(request(self.store, changed)[0], 409)
        for body in ({"mode": "pipeline_stage", "run_id": "a"*64, "stage": "train"},
                     {"mode": "security_triage", "alert": {"id": "bad"}}, []):
            self.assertEqual(request(self.store, body, key="bad")[0], 400)
        self.assertEqual(request(self.store, self.payload, content_type="text/plain")[0], 415)
        self.assertEqual(request(self.store, self.payload, declared_length=9000)[0], 413)
        self.assertEqual(request(self.store, self.payload, declared_length="nonnumeric")[0], 400)
        self.assertEqual(request(self.store, self.payload, declared_length=0)[0], 413)

    def test_redacted_metrics_and_unknown_route(self):
        self.assertEqual(request(self.store, path="/metrics", authorized=False)[0], 401)
        self.assertEqual(request(self.store, path="/jobs/missing")[0], 404)
        self.assertEqual(request(self.store, path="/health", authorized=False), (200, {"status": "ok"}))

    def test_ambiguous_lengths_and_nonascii_authorization_fail_closed(self):
        self.assertEqual(request(self.store, self.payload, extra_headers=("Content-Length: 1",))[0], 400)
        self.assertEqual(request(self.store, self.payload, extra_headers=("Transfer-Encoding: chunked",))[0], 400)
        self.assertEqual(request(self.store, self.payload, authorized=False, extra_headers=("Authorization: Bearer \u00e9",))[0], 401)

    def test_duplicate_json_fields_and_short_body_do_not_enqueue(self):
        raw = json.dumps(self.payload).replace('"complete": true', '"complete": false, "complete": true').encode()
        self.assertEqual(request(self.store, raw_body=raw)[0], 400)
        self.assertEqual(request(self.store, self.payload, declared_length=len(json.dumps(self.payload).encode()) + 10)[0], 400)
        self.assertEqual(self.store.metrics()["jobs_by_status"], {})
