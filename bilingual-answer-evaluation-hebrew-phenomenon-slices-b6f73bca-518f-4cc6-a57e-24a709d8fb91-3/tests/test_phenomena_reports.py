"""Known counts, empty slices, retrieval diagnostics and saved-baseline replay."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from cola_lab.answer_eval import (dataset_named, evaluate, phenomenon_slices, run,
                                 verify_answers)


class PhenomenaReportTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset_named("hebrew-phenomena")

    def test_actual_baseline_has_supported_counts_and_concrete_failures_for_each_phenomenon(self):
        report = evaluate(self.data, "fixture")
        lexical = report["methods"]["lexical_retrieval"]
        self.assertEqual(report["protocol"]["threshold"], 2)
        self.assertEqual(lexical["overall"]["accuracy"], 20 / 24)
        self.assertEqual(lexical["overall"]["grounded_answer_rate"], 12 / 16)
        self.assertEqual(lexical["overall"]["abstention_accuracy"], 1)
        self.assertEqual(lexical["by_language"]["en"]["accuracy"], 1)
        self.assertEqual(lexical["by_language"]["he"]["accuracy"], 8 / 12)
        self.assertEqual(lexical["paired_languages"]["english_only_correct"], 4)
        for name, count in (("inflection", 2), ("spelling", 1), ("mixed_language", 1)):
            row = lexical["by_phenomenon"][name]
            self.assertEqual(row["count"], count)  # English paraphrases are excluded.
            self.assertEqual(row["metrics"]["accuracy"], 0)
            self.assertEqual(row["control_metrics"]["accuracy"], 1)
            self.assertEqual(row["transitions"], {"regressed": count})
            self.assertEqual(row["metrics"]["error_counts"], {"false_abstention": count})
            self.assertIsNone(row["metrics"]["abstention_accuracy"])
            self.assertEqual(row["metrics"]["unanswerable_count"], 0)
            self.assertEqual(len(row["failures"]), count)
            for failure in row["failures"]:
                self.assertEqual(failure["error_stage"], "source_not_retrieved")
                self.assertEqual(failure["retrieval_status"], "source_missed")
                self.assertNotIn(failure["selected_source"]["id"], failure["expected_evidence"])
                self.assertEqual([doc["id"] for doc in failure["expected_sources"]], failure["expected_evidence"])
                self.assertTrue(failure["control_correct"])
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_empty_slices_are_null_with_a_reason_including_original_corpus(self):
        for rows in (phenomenon_slices([], []),
                     evaluate(dataset_named("original"), "fixture")["methods"]["lexical_retrieval"]["by_phenomenon"]):
            self.assertEqual(set(rows), {"inflection", "spelling", "mixed_language"})
            for row in rows.values():
                self.assertEqual(row["count"], 0)
                self.assertIsNone(row["metrics"])
                self.assertIsNone(row["control_metrics"])
                self.assertIn("No Hebrew variants", row["empty_reason"])
                self.assertEqual(row["failures"], [])
                self.assertEqual(row["transitions"], {})

    def spelling_pair(self):
        return [case for case in self.data["cases"] if case["entity"] == "Fjord"
                and case["language"] == "he" and case["phenomenon"] == "spelling"]

    def test_missing_source_and_answer_error_with_source_are_distinct(self):
        cases = self.spelling_pair()
        predictions = [{"answer": None, "citations": []} for _ in cases]
        only_reference = [doc for doc in self.data["documents"] if doc["id"] == "test-he-Fjord-description"]
        row = phenomenon_slices(cases, predictions, only_reference, method="lexical_retrieval")["spelling"]
        self.assertEqual(row["failures"][0]["retrieval_status"], "source_retrieved")
        self.assertEqual(row["failures"][0]["error_stage"], "answer_error_with_source")
        self.assertEqual(row["failures"][0]["error_type"], "false_abstention")
        row = phenomenon_slices(cases, predictions, self.data["documents"], method="lexical_retrieval")["spelling"]
        self.assertEqual(row["failures"][0]["error_stage"], "source_not_retrieved")

    def test_literal_success_is_not_reported_as_a_failure_and_model_exposure_is_not_inferred(self):
        cases = self.spelling_pair()
        predictions = [{"answer": "מרובע", "citations": ["test-he-Fjord-description"]} for _ in cases]
        row = phenomenon_slices(cases, predictions, self.data["documents"], method="local_model")["spelling"]
        self.assertEqual(row["metrics"]["accuracy"], 1)
        self.assertEqual(row["transitions"], {"both_correct": 1})
        self.assertEqual(row["failures"], [])
        self.assertIsNone(row["pairs"][0]["error_stage"])
        row = phenomenon_slices(cases, [{"answer": None, "citations": []}] * 2,
                                self.data["documents"], method="local_model")["spelling"]
        self.assertIsNone(row["failures"][0]["error_stage"])
        self.assertIn("no inferred model exposure", row["diagnostic_scope"])

    def test_partial_slice_and_mismatched_prediction_count_are_errors(self):
        cases = self.spelling_pair()
        with self.assertRaisesRegex(ValueError, "every case"):
            phenomenon_slices(cases, [])
        for case in cases:
            with self.assertRaisesRegex(ValueError, "matched Hebrew"):
                phenomenon_slices([case], [{"answer": None, "citations": []}])

    def test_baseline_roundtrip_replay_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = run(directory, dataset=self.data)
            original = path.read_bytes()
            audit = verify_answers(directory)
            self.assertEqual(audit["status"], "verified")
            self.assertEqual(audit["case_count"], 24)
            self.assertFalse(audit["model_quality_evidence"])
            with self.assertRaisesRegex(ValueError, "already exist"):
                run(directory, dataset=self.data)
            self.assertEqual(path.read_bytes(), original)
            report = json.loads(original)
            protocol = report["protocol"]["challenge_corpus"]
            self.assertFalse(protocol["independent_linguistic_review"])
            self.assertEqual(len(protocol["raw_corpus_sha256"]), 64)
            self.assertEqual(len(protocol["variant_group_splits"]), 8)

    def test_baseline_replay_rejects_false_metrics_diagnostics_and_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = run(directory, dataset=self.data)
            original = json.loads(path.read_text())
            for mutate in (
                lambda r: r["methods"]["lexical_retrieval"]["by_phenomenon"]["spelling"]["metrics"].update(accuracy=1),
                lambda r: r["methods"]["lexical_retrieval"]["by_phenomenon"]["spelling"]["failures"][0].update(error_stage="answer_error_with_source"),
                lambda r: r["protocol"]["challenge_corpus"].update(cleaning="strip spelling differences"),
                lambda r: r["protocol"]["challenge_corpus"].update(independent_linguistic_review=True)):
                changed = copy.deepcopy(original)
                mutate(changed)
                path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "scoring replay"):
                    verify_answers(directory)
            # Original absent slices also participate in replay.
        with tempfile.TemporaryDirectory() as directory:
            path = run(directory)
            report = json.loads(path.read_text())
            report["methods"]["lexical_retrieval"]["by_phenomenon"]["spelling"]["metrics"] = {"accuracy": 0}
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scoring replay"):
                verify_answers(directory)


if __name__ == "__main__":
    unittest.main()
