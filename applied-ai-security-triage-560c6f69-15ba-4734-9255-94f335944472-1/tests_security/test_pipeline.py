import concurrent.futures
import copy
import json
from pathlib import Path
import tempfile
import unittest

from portfolio.reliable_service import JobStore, Conflict, LeaseLost, work_once
from portfolio.security_data import build_dataset, observations
from portfolio.security_eval import evaluate, replay, report
from portfolio.security_models import ReferencePolicy, train_reference
from portfolio.security_pipeline import Pipeline, audit, read_artifact, verified_path, file_hash, atomic_json
from portfolio.security_triage import SecurityTriageEngine, RulePlanner


class PipelineTests(unittest.TestCase):
    def test_full_lifecycle_resume_and_manifest_replay(self):
        with tempfile.TemporaryDirectory() as out:
            pipeline = Pipeline(out, steps=24)
            manifest = pipeline.run()
            self.assertEqual(set(manifest["stages"]), {"prepare", "train", "evaluate"})
            self.assertEqual(audit(out)["verified_rows"], 72)
            self.assertEqual(audit(out)["acceptance"]["recovered_jobs"], 1)
            self.assertEqual(manifest["queue_metrics"], {"jobs_by_status": {"succeeded": 3}, "total_attempts": 3})
            self.assertEqual(Pipeline(out, steps=24).run(), manifest)
            with self.assertRaisesRegex(ValueError, "changed"):
                Pipeline(out, steps=25)

    def test_changed_test_labels_cannot_change_training_checkpoint(self):
        data = build_dataset()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            def fit(out):
                return train_reference(observations(data["train"]), data["train"],
                    observations(data["validation"]), data["validation"], out, steps=24)
            a = fit(first)
            for case in data["test"]:
                case["expected"] = {"operation": "abstain", "disposition": "abstain"}
            b = fit(second)
            self.assertEqual(a, b)
            self.assertEqual((Path(first)/"adapter.json").read_bytes(), (Path(second)/"adapter.json").read_bytes())

    def test_checksum_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            manifest = Pipeline(out, steps=1).run()
            receipt = manifest["stages"]["train"]
            path = verified_path(out, receipt, "adapter.json")
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit(out)
            receipt["artifacts"]["adapter.json"]["path"] = "../../outside.json"
            with self.assertRaisesRegex(ValueError, "outside"):
                verified_path(out, receipt, "adapter.json")

    def test_out_of_order_stage_has_persisted_failure_and_can_recover(self):
        with tempfile.TemporaryDirectory() as out:
            pipeline = Pipeline(out, steps=1)
            job_id = pipeline.enqueue("train")
            with self.assertRaisesRegex(ValueError, "Predecessor"):
                pipeline.work_once()
            self.assertEqual(pipeline.store.get(job_id)["error"], "stage_error:ValueError")
            failures = list(Path(out).glob("attempts/*/failure.json"))
            self.assertEqual(len(failures), 1)
            # Complete preparation before retrying the failed stage. The runner
            # processes queued work at most three times per stage.
            pipeline.run()
            self.assertEqual(pipeline.store.get(job_id)["status"], "succeeded")

    def test_report_must_match_raw_metrics_even_if_receipt_is_rehashed(self):
        with tempfile.TemporaryDirectory() as out:
            manifest = Pipeline(out, steps=1).run()
            artifact = manifest["stages"]["evaluate"]["artifacts"]["REPORT.md"]
            for path in (Path(out)/"REPORT.md", Path(out)/artifact["path"]):
                path.write_text("# Unsupported improvement claim\n")
            artifact["sha256"] = file_hash(Path(out)/artifact["path"])
            manifest["report_sha256"] = file_hash(Path(out)/"REPORT.md")
            atomic_json(Path(out)/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Report differs"):
                audit(out)

    def test_expired_attempt_cannot_publish_or_renew(self):
        with tempfile.TemporaryDirectory() as out:
            now = [100.]
            pipeline = Pipeline(out, steps=1, clock=lambda: now[0])
            job_id = pipeline.enqueue("prepare")
            stale = pipeline.store.claim(lease_seconds=1)
            now[0] += 2
            self.assertTrue(pipeline.work_once())
            for operation in (lambda: pipeline.store.complete(stale, {"artifacts": {}}),
                              lambda: pipeline.store.renew(stale), lambda: pipeline.store.fail(stale, "late")):
                with self.assertRaises(LeaseLost):
                    operation()
            self.assertEqual(pipeline.store.get(job_id)["status"], "succeeded")

    def test_renew_before_expiry_and_bounded_retry_exhaustion(self):
        with tempfile.TemporaryDirectory() as out:
            now = [100.]
            pipeline = Pipeline(out, steps=1, clock=lambda: now[0])
            job_id = pipeline.enqueue("prepare")
            job = pipeline.store.claim(lease_seconds=2)
            now[0] += 1
            pipeline.store.renew(job, lease_seconds=5)
            now[0] += 2
            self.assertIsNone(pipeline.store.claim())
            now[0] += 4
            for _ in range(2):
                self.assertIsNotNone(pipeline.store.claim(lease_seconds=1))
                now[0] += 2
            self.assertIsNone(pipeline.store.claim())
            self.assertEqual(pipeline.store.get(job_id)["status"], "failed")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_dataset()["test"]

    def test_replay_rejects_fabricated_metrics_outputs_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as out:
            summary = evaluate(self.cases, ReferencePolicy(), ReferencePolicy(), out)
            rows = json.loads((Path(out)/"runs.json").read_text())
            self.assertEqual(replay(self.cases, rows, summary)["verified_rows"], 72)
            bad_summary = copy.deepcopy(summary)
            bad_summary["arms"]["base"]["correct"] += 1
            with self.assertRaisesRegex(ValueError, "Summary"):
                replay(self.cases, rows, bad_summary)
            with self.assertRaisesRegex(ValueError, "Missing"):
                replay(self.cases, rows[:-1], summary)
            bad_rows = copy.deepcopy(rows)
            bad_rows[0]["result"]["finding"] = "Invented conclusion"
            with self.assertRaisesRegex(ValueError, "Finding"):
                replay(self.cases, bad_rows, summary)
            bad_rows = copy.deepcopy(rows)
            bad_rows[0]["result"]["citations"].append("unseen-source")
            with self.assertRaisesRegex(ValueError, "Score"):
                replay(self.cases, bad_rows, summary)

    def test_backend_failure_is_a_failed_row_without_fallback(self):
        class Broken:
            def plan(self, observation):
                raise RuntimeError("fixture failure")
        with tempfile.TemporaryDirectory() as out:
            summary = evaluate(self.cases, Broken(), ReferencePolicy(), out)
            self.assertEqual(summary["arms"]["base"]["failed"], 24)
            self.assertEqual(summary["arms"]["base"]["correct"], 0)
            rows = json.loads((Path(out)/"runs.json").read_text())
            self.assertEqual(replay(self.cases, rows, summary)["verified_rows"], 72)

    def test_replay_checks_raw_rejected_plan_and_execution_trace(self):
        with tempfile.TemporaryDirectory() as out:
            summary = evaluate(self.cases, ReferencePolicy(), ReferencePolicy(), out)
            rows = json.loads((Path(out)/"runs.json").read_text())
        bad = copy.deepcopy(rows)
        row = next(r for r in bad if r["result"]["status"] == "failed")
        case = next(c for c in self.cases if c["id"] == row["case_id"])
        row["result"]["model_response"] = RulePlanner().plan(observations([case])[0])
        with self.assertRaisesRegex(ValueError, "Finding"):
            replay(self.cases, bad, summary)
        bad = copy.deepcopy(rows)
        bad[0]["result"]["trace"][0]["event_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "Finding"):
            replay(self.cases, bad, summary)

    def test_each_arm_receives_identical_label_free_observation(self):
        class Capturing(RulePlanner):
            def __init__(self):
                self.seen = []
            def plan(self, observation):
                self.seen.append(copy.deepcopy(observation))
                return super().plan(observation)
        a, b = Capturing(), Capturing()
        with tempfile.TemporaryDirectory() as out:
            evaluate(self.cases, a, b, out)
        self.assertEqual(a.seen, b.seen)
        self.assertTrue(all("expected" not in obs and "family" not in obs for obs in a.seen))

    def test_mutating_backend_cannot_contaminate_other_arms_or_source_cases(self):
        original = copy.deepcopy(self.cases)

        class Mutating(RulePlanner):
            def plan(self, observation):
                observation["events"].clear()
                observation["policy"]["failure_threshold"] = 32
                return super().plan(observation)

        with tempfile.TemporaryDirectory() as out:
            summary = evaluate(self.cases, Mutating(), RulePlanner(), out)
            rows = json.loads((Path(out)/"runs.json").read_text())
            self.assertEqual(self.cases, original)
            self.assertEqual(summary["arms"]["adapted"]["grounded_correct"], len(self.cases))
            self.assertEqual(replay(self.cases, rows, summary)["verified_rows"], 72)

    def test_report_uses_observed_counts_and_keeps_method_limits(self):
        with tempfile.TemporaryDirectory() as out:
            summary = evaluate(self.cases, ReferencePolicy(), ReferencePolicy(), out)
        rendered = report(summary, {"backend": ReferencePolicy.backend, "selected_step": 0, "steps": 1})
        for arm, values in summary["arms"].items():
            self.assertIn(f"| {arm} | {values['grounded_correct']} / {values['cases']} |", rendered)
        self.assertIn("2302.12173", rendered)
        self.assertIn("not an LLM", rendered)
        self.assertIn("not measured", rendered)


class BoundedQueueTests(unittest.TestCase):
    def test_32_triage_jobs_four_workers_duplicates_and_conflict(self):
        cases = build_dataset()["train"]
        events = [e for c in cases for e in c["events"]]
        with tempfile.TemporaryDirectory() as out:
            store = JobStore(Path(out)/"queue.db")
            def submit(case):
                return store.enqueue(case["id"], {"mode": "security_triage", "alert": case["alert"]})
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                submitted = list(executor.map(submit, cases * 2))
            self.assertEqual(sum(created for _, created in submitted), 32)
            with self.assertRaises(Conflict):
                store.enqueue(cases[0]["id"], {"mode": "security_triage", "alert": cases[1]["alert"]})
            engine = SecurityTriageEngine(RulePlanner(), events)
            def worker(_):
                done = 0
                for _ in range(32):
                    if not work_once(store, engine):
                        break
                    done += 1
                return done
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                self.assertEqual(sum(executor.map(worker, range(4))), 32)
            self.assertEqual(store.metrics(), {"jobs_by_status": {"succeeded": 32}, "total_attempts": 32})
            for case, (job_id, _) in zip(cases, submitted):
                job = store.get(job_id)
                self.assertEqual(job["result"]["structured_answer"]["disposition"], case["expected"]["disposition"])
                self.assertEqual([r["event"] for r in job["trace"]], ["enqueued", "claimed", "completed"])
