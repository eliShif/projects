import copy
import json
import tempfile
import unittest

from cola_lab.answer_eval import (evaluate, make_dataset, predict, run, score_case,
                                  summarize, validate_dataset)


class AnswerEvaluationTests(unittest.TestCase):
    def test_dataset_and_report_are_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = json.loads(run(first).read_text(encoding="utf-8"))
            two = json.loads(run(second).read_text(encoding="utf-8"))
        self.assertEqual(one, two)
        self.assertEqual(one["protocol"]["case_counts"], {"validation": 24, "test": 24})
        self.assertEqual(one["protocol"]["threshold"], 2)
        lexical = one["methods"]["lexical_retrieval"]
        baseline = one["methods"]["always_abstain"]
        self.assertGreater(lexical["overall"]["accuracy"], baseline["overall"]["accuracy"])
        for language in ("en", "he"):
            self.assertEqual(lexical["by_language"][language]["count"], 12)
            self.assertTrue(lexical["by_language"][language]["errors"])
            self.assertEqual(lexical["by_language"][language]["error_counts"],
                             {"false_abstention": 1})

    def test_same_rule_scores_both_languages_and_abstains_on_missing_fact(self):
        dataset = make_dataset()
        for language in ("en", "he"):
            answerable = next(case for case in dataset["cases"] if case["split"] == "test"
                              and case["language"] == language and "shape-q" in case["id"])
            unanswered = next(case for case in dataset["cases"] if case["split"] == "test"
                              and case["language"] == language and "weight-q" in case["id"])
            self.assertTrue(score_case(answerable, predict(answerable, dataset["documents"], 2))["correct"])
            self.assertTrue(score_case(unanswered, predict(unanswered, dataset["documents"], 2))["correct"])

    def test_wrong_or_missing_evidence_fails(self):
        case = next(case for case in make_dataset()["cases"] if case["expected_answer"])
        self.assertEqual(score_case(case, {"answer": case["expected_answer"], "citations": []})
                         ["error_type"], "unsupported_citation")
        self.assertEqual(score_case(case, {"answer": "wrong", "citations": case["evidence_ids"]})
                         ["error_type"], "wrong_answer")
        self.assertEqual(score_case(case, {"answer": None, "citations": []})["error_type"],
                         "false_abstention")

    def test_rejects_invalid_evidence_and_entity_leakage(self):
        dataset = make_dataset()
        broken = copy.deepcopy(dataset)
        broken["cases"][0]["evidence_ids"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "Evidence does not support"):
            validate_dataset(broken)
        broken = copy.deepcopy(dataset)
        broken["documents"][-1]["entity"] = "Alder"
        with self.assertRaisesRegex(ValueError, "Entity leakage"):
            validate_dataset(broken)

    def test_test_labels_cannot_choose_threshold(self):
        dataset = make_dataset()
        original = evaluate(dataset, "sample")
        changed = copy.deepcopy(dataset)
        case = next(case for case in changed["cases"] if case["split"] == "test"
                    and case["expected_answer"] is not None)
        case["expected_answer"] = None
        case["evidence_ids"] = []
        altered = evaluate(changed, "sample")
        self.assertEqual(original["protocol"]["threshold"], altered["protocol"]["threshold"])
        self.assertNotEqual(original["methods"], altered["methods"])


if __name__ == "__main__":
    unittest.main()
