import copy
import json
import tempfile
import unittest

from cola_lab.answer_eval import (evaluate, load_model_predictions, make_dataset, predict, run, score_case,
                                  summarize, validate_dataset, paired_language_analysis)


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
        self.assertEqual(lexical["paired_languages"],
                         {"pair_count": 12, "both_correct": 11, "both_wrong": 1,
                          "english_only_correct": 0, "hebrew_only_correct": 0})
        self.assertEqual(set(lexical["by_entity"]), {"Ember", "Fjord", "Grove", "Harbor"})
        self.assertEqual(one["protocol"]["data_quality"]["test"]["he"]["answerable"], 8)

    def test_pair_analysis_detects_language_specific_error(self):
        cases = [case for case in make_dataset()["cases"] if case["split"] == "test"]
        predictions = [{"answer": case["expected_answer"], "citations": case["evidence_ids"]}
                       for case in cases]
        english = next(i for i, case in enumerate(cases) if case["language"] == "en"
                       and case["expected_answer"] is not None)
        predictions[english] = {"answer": None, "citations": []}
        self.assertEqual(paired_language_analysis(cases, predictions)["hebrew_only_correct"], 1)
        self.assertEqual(paired_language_analysis(cases, predictions)["english_only_correct"], 0)

    def test_rejects_missing_translation_even_when_language_counts_remain_positive(self):
        broken = make_dataset()
        broken["cases"] = [case for case in broken["cases"]
                           if case["id"] != "test-he-Ember-color-q"]
        with self.assertRaisesRegex(ValueError, "must be paired"):
            validate_dataset(broken)

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
        self.assertEqual(score_case(case, {"answer": case["expected_answer"],
                                            "citations": case["evidence_ids"] + ["other"]})
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

    def test_offline_model_outputs_use_same_bilingual_scoring(self):
        dataset = make_dataset()
        cases = [case for case in dataset["cases"] if case["split"] == "test"]
        payload = {"model_id": "local-test-fixture", "model_version": "fixture-1",
                   "predictions": [{"case_id": case["id"], "answer": case["expected_answer"],
                                    "citations": case["evidence_ids"]} for case in cases]}
        # A plausible saved-output failure: an answer with no supporting citation.
        payload["predictions"][0]["citations"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/predictions.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            report_path = run(f"{directory}/out", path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(set(report["methods"]), {"always_abstain", "lexical_retrieval", "local_model"})
        self.assertEqual(report["protocol"]["model_outputs"]["model_id"], "local-test-fixture")
        self.assertEqual(report["methods"]["local_model"]["overall"]["error_counts"],
                         {"unsupported_citation": 1})
        self.assertEqual(report["methods"]["local_model"]["by_language"]["he"]["count"], 12)

    def test_rejects_incomplete_duplicate_and_mismatched_model_outputs(self):
        dataset = make_dataset()
        cases = [case for case in dataset["cases"] if case["split"] == "test"]
        base = {"model_id": "local-test-fixture", "model_version": "fixture-1",
                "predictions": [{"case_id": case["id"], "answer": None, "citations": []}
                                for case in cases]}
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/predictions.json"
            for change, message in ((lambda p: p["predictions"].pop(), "cover every"),
                                    (lambda p: p["predictions"].append(p["predictions"][0]), "duplicate"),
                                    (lambda p: p["predictions"][0].update(
                                        {"answer": "blue", "citations": ["validation-en-Alder-color"]}),
                                     "mismatched")):
                payload = copy.deepcopy(base)
                change(payload)
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                with self.assertRaisesRegex(ValueError, message):
                    load_model_predictions(path, dataset)


if __name__ == "__main__":
    unittest.main()
