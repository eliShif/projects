"""Hand-calculated metric expectations, including zero denominators and real reports."""

import json
import tempfile
import unittest

from cola_lab.answer_eval import method_report, run, summarize


def literal_cases():
    # No reference values or predictions are computed by the implementation under test.
    return [
        {"id": "test-en-Ember-color-q", "language": "en", "entity": "Ember",
         "question": "What is the color of Ember?", "expected_answer": "blue", "evidence_ids": ["en-color"]},
        {"id": "test-he-Ember-color-q", "language": "he", "entity": "Ember",
         "question": "מה הצבע של Ember?", "expected_answer": "כחול", "evidence_ids": ["he-color"]},
        {"id": "test-en-Ember-weight-q", "language": "en", "entity": "Ember",
         "question": "What is the weight of Ember?", "expected_answer": None, "evidence_ids": []},
        {"id": "test-he-Ember-weight-q", "language": "he", "entity": "Ember",
         "question": "מה המשקל של Ember?", "expected_answer": None, "evidence_ids": []}]


class SummarySubsetTests(unittest.TestCase):
    def setUp(self):
        self.cases = literal_cases()
        self.predictions = [{"answer": "blue", "citations": ["en-color"]},
                            {"answer": "כחול", "citations": []},
                            {"answer": None, "citations": []},
                            {"answer": "כבד", "citations": ["he-color"]}]

    def test_mixed_subset_uses_the_correct_denominators(self):
        summary = summarize(self.cases, self.predictions)
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["answerable_count"], 2)
        self.assertEqual(summary["unanswerable_count"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["grounded_answer_rate"], 0.5)
        self.assertEqual(summary["abstention_accuracy"], 0.5)
        self.assertEqual(summary["undefined_metrics"], {})
        self.assertEqual(summary["error_counts"], {"unsupported_citation": 1, "false_answer": 1})

    def test_unequal_case_types_use_conditional_not_total_or_prediction_counts(self):
        extra = {"id": "test-en-Ember-shape-q", "language": "en", "entity": "Ember",
                 "question": "What is the shape of Ember?", "expected_answer": "round",
                 "evidence_ids": ["en-shape"]}
        cases = [*self.cases[:2], extra, self.cases[3]]
        predictions = [*self.predictions[:2], {"answer": "round", "citations": ["en-shape"]},
                       self.predictions[3]]
        summary = summarize(cases, predictions)
        self.assertEqual(summary["answerable_count"], 3)
        self.assertEqual(summary["unanswerable_count"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["grounded_answer_rate"], 2 / 3)
        self.assertEqual(summary["abstention_accuracy"], 0)
        self.assertEqual(summary["undefined_metrics"], {})

    def test_answerable_only_report_has_null_abstention_metric(self):
        report = method_report(self.cases[:2], self.predictions[:2])
        summary = report["overall"]
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["grounded_answer_rate"], 0.5)
        self.assertEqual(summary["answerable_count"], 2)
        self.assertEqual(summary["unanswerable_count"], 0)
        self.assertIsNone(summary["abstention_accuracy"])
        self.assertEqual(summary["undefined_metrics"],
                         {"abstention_accuracy": "No unanswerable cases in this subset."})
        self.assertEqual(report["by_language"]["en"]["grounded_answer_rate"], 1)
        self.assertEqual(report["by_language"]["he"]["grounded_answer_rate"], 0)
        for language in ("en", "he"):
            row = report["by_language"][language]
            self.assertEqual(row["answerable_count"], 1)
            self.assertIsNone(row["abstention_accuracy"])
            self.assertEqual(row["undefined_metrics"], summary["undefined_metrics"])
        self.assertIsNone(report["by_entity"]["Ember"]["abstention_accuracy"])
        self.assertEqual(set(report["by_answerability"]), {"answerable"})
        self.assertEqual(report["paired_languages"]["english_only_correct"], 1)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_unanswerable_only_report_has_null_grounded_metric(self):
        report = method_report(self.cases[2:], self.predictions[2:])
        summary = report["overall"]
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["abstention_accuracy"], 0.5)
        self.assertEqual(summary["answerable_count"], 0)
        self.assertEqual(summary["unanswerable_count"], 2)
        self.assertIsNone(summary["grounded_answer_rate"])
        self.assertEqual(summary["undefined_metrics"],
                         {"grounded_answer_rate": "No answerable cases in this subset."})
        self.assertEqual(report["by_language"]["en"]["abstention_accuracy"], 1)
        self.assertEqual(report["by_language"]["he"]["abstention_accuracy"], 0)
        for language in ("en", "he"):
            row = report["by_language"][language]
            self.assertEqual(row["unanswerable_count"], 1)
            self.assertIsNone(row["grounded_answer_rate"])
            self.assertEqual(row["undefined_metrics"], summary["undefined_metrics"])
        self.assertIsNone(report["by_entity"]["Ember"]["grounded_answer_rate"])
        self.assertEqual(set(report["by_answerability"]), {"unanswerable"})
        self.assertEqual(report["paired_languages"]["english_only_correct"], 1)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_single_case_defined_zero_is_distinct_from_undefined(self):
        for index, defined, undefined in ((1, "grounded_answer_rate", "abstention_accuracy"),
                                          (3, "abstention_accuracy", "grounded_answer_rate")):
            with self.subTest(index=index):
                summary = summarize([self.cases[index]], [self.predictions[index]])
                self.assertEqual(summary["accuracy"], 0)
                self.assertEqual(summary[defined], 0)
                self.assertIsNone(summary[undefined])
                self.assertEqual(set(summary["undefined_metrics"]), {undefined})

    def test_empty_input_and_prediction_count_mismatches_still_raise(self):
        for cases, predictions in (([], []), ([], self.predictions), (self.cases, []),
                                    (self.cases, self.predictions[:-1]),
                                    (self.cases, self.predictions + [self.predictions[0]])):
            with self.subTest(counts=(len(cases), len(predictions))):
                with self.assertRaisesRegex(ValueError, "every case"):
                    summarize(cases, predictions)
        with self.assertRaises(ValueError):
            method_report([], [])

    def test_saved_baseline_report_includes_both_answerability_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            report = json.loads(run(directory).read_text(encoding="utf-8"))
        lexical = report["methods"]["lexical_retrieval"]
        self.assertEqual(lexical["overall"]["accuracy"], 22 / 24)
        self.assertEqual(lexical["overall"]["answerable_count"], 16)
        self.assertEqual(lexical["overall"]["unanswerable_count"], 8)
        answerable, missing = (lexical["by_answerability"][kind]
                               for kind in ("answerable", "unanswerable"))
        self.assertEqual(answerable["count"], 16)
        self.assertEqual(answerable["accuracy"], 14 / 16)
        self.assertEqual(answerable["error_counts"], {"false_abstention": 2})
        self.assertIsNone(answerable["abstention_accuracy"])
        self.assertEqual(missing["count"], 8)
        self.assertEqual(missing["accuracy"], 1)
        self.assertIsNone(missing["grounded_answer_rate"])
        self.assertEqual(report["methods"]["always_abstain"]["by_answerability"]
                         ["answerable"]["grounded_answer_rate"], 0)
        for language in ("en", "he"):
            row = lexical["by_language"][language]
            self.assertEqual(row["answerable_count"], 8)
            self.assertEqual(row["unanswerable_count"], 4)
            self.assertEqual(row["accuracy"], 11 / 12)
            self.assertEqual(row["grounded_answer_rate"], 7 / 8)
            self.assertEqual(row["abstention_accuracy"], 1)
            self.assertEqual(row["undefined_metrics"], {})
        self.assertEqual(report["paired_audit"]["pair_count"], 12)


if __name__ == "__main__":
    unittest.main()
