"""Hand-calculated gaps, failure cases, and separation from model selection."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from portfolio.research_lab import (
    CANDIDATES, FAMILIES, GAP_METRICS, TextBaseline, cases_from_generator,
    challenge_cases, evaluate, run, train_validation_review,
)
from scripts.smoke_research_lab import check_artifacts


class GeneralizationTests(unittest.TestCase):
    @staticmethod
    def known_scores():
        # Each of eight labels has a distinct word. Validation swaps two words
        # between labels, giving NB exactly six correct predictions out of eight.
        # This is an arithmetic fixture, not evidence of measured overfitting.
        train = [{"id": f"train-{i}", "entity": "Cobalt", "label": label,
                  "question": f"token{i} token{i}"} for i, label in enumerate(FAMILIES)]
        validation = [{**row, "id": f"validation-{i}", "entity": "Fjord",
                       "question": f"token{1 - i if i < 2 else i}"}
                      for i, row in enumerate(train)]
        models = {name: TextBaseline(name).fit(train) for name in CANDIDATES}
        return ({name: evaluate(model, train) for name, model in models.items()},
                {name: evaluate(model, validation) for name, model in models.items()})

    def test_known_confusions_give_positive_and_zero_gaps(self):
        train, validation = self.known_scores()
        original = deepcopy((train, validation))
        review = train_validation_review(train, validation)
        for name in ("unigram", "bigram"):
            for metric in GAP_METRICS:
                self.assertEqual(review["candidates"][name][metric], {
                    "training": 1.0, "validation": 0.75, "gap": 0.25,
                    "direction": "training_higher"})
        # Majority predicts one label: 1/8 accuracy; its single nonzero F1
        # is 2/9, so fixed-eight-family macro F1 is (2/9)/8 = 1/36.
        for metric, score in (("accuracy", 1 / 8), ("macro_f1", 1 / 36)):
            item = review["candidates"]["majority"][metric]
            self.assertAlmostEqual(item["training"], score)
            self.assertAlmostEqual(item["validation"], score)
            self.assertEqual(item["gap"], 0)
            self.assertEqual(item["direction"], "equal")
        self.assertEqual((train, validation), original)
        json.dumps(review, allow_nan=False)

    def test_negative_gaps_and_different_metric_directions_are_preserved(self):
        train, validation = self.known_scores()
        reversed_review = train_validation_review(validation, train)
        for metric in GAP_METRICS:
            self.assertEqual(reversed_review["candidates"]["unigram"][metric]["gap"], -0.25)
            self.assertEqual(reversed_review["candidates"]["unigram"][metric]["direction"], "validation_higher")
        # Direct score inputs test a mixed-sign case and fraction units. They
        # are never written to experiment output or treated as measured scores.
        train["unigram"].update(accuracy=0.75, macro_f1=0.5)
        validation["unigram"].update(accuracy=0.5, macro_f1=0.625)
        mixed = train_validation_review(train, validation)["candidates"]["unigram"]
        self.assertEqual(mixed["accuracy"]["gap"], 0.25)
        self.assertEqual(mixed["macro_f1"]["gap"], -0.125)
        self.assertEqual(mixed["accuracy"]["direction"], "training_higher")
        self.assertEqual(mixed["macro_f1"]["direction"], "validation_higher")

    def test_gap_review_rejects_invalid_scores_and_incomplete_candidates(self):
        train, validation = self.known_scores()
        for split in (0, 1):
            for metric in GAP_METRICS:
                for invalid in (float("nan"), float("inf"), -float("inf"), -0.1, 1.1, True, None, "0.5"):
                    with self.subTest(split=split, metric=metric, invalid=invalid):
                        scores = deepcopy([train, validation])
                        scores[split]["bigram"][metric] = invalid
                        with self.assertRaisesRegex(ValueError, "finite number in"):
                            train_validation_review(*scores)
            for invalid in ({}, {"unigram": train["unigram"]}, [], None):
                with self.subTest(split=split, candidates=invalid):
                    scores = deepcopy([train, validation])
                    scores[split] = invalid
                    with self.assertRaisesRegex(ValueError, "all existing candidates"):
                        train_validation_review(*scores)
        del train["unigram"]["macro_f1"]
        with self.assertRaisesRegex(ValueError, "finite number in"):
            train_validation_review(train, validation)
        train["unigram"] = None
        with self.assertRaisesRegex(ValueError, "needs metric values"):
            train_validation_review(train, validation)

    def test_run_reports_real_training_scores_and_gaps_for_every_candidate(self):
        with TemporaryDirectory() as directory:
            result = run(directory)
            report = (Path(directory) / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("## Train–validation gaps and overfitting", report)
            self.assertIn("zero gap does not rule out overfitting", report)
            self.assertIn("wording/template shift", report)
            for name in CANDIDATES:
                self.assertEqual(sum(map(sum, result["training"][name]["confusion"])), 24)
                expected = {"accuracy": 1 / 8, "macro_f1": 1 / 36} if name == "majority" else {
                    "accuracy": 1.0, "macro_f1": 1.0}
                for metric, score in expected.items():
                    self.assertAlmostEqual(result["training"][name][metric], score)
                    self.assertAlmostEqual(result["validation"][name][metric], score)
                    item = result["train_validation_review"]["candidates"][name][metric]
                    self.assertEqual(item["gap"], 0)
                    self.assertIn(f"| {name} | {metric} | {score:.6f} | {score:.6f} | +0.000000 | equal |", report)

    def test_training_scores_cannot_replace_validation_selection(self):
        # Both controls must leave the validation tie-break unchanged. The first
        # rewards majority on train; the second gives it the smallest signed gap.
        for favor_majority in (True, False):
            def controlled_training(model, rows):
                result = evaluate(model, rows)
                if all(row["split"] == "train" for row in rows):
                    value = float((model.name == "majority") == favor_majority)
                    result.update(accuracy=value, macro_f1=value)
                return result

            with self.subTest(favor_majority=favor_majority), TemporaryDirectory() as directory:
                with patch("portfolio.research_lab.evaluate", controlled_training):
                    result = run(directory)
                self.assertEqual(result["selected"], "bigram")
                decision = json.loads((Path(directory) / "selection.json").read_text())
                self.assertEqual(decision["selected"], "bigram")
                self.assertEqual(decision["candidate_metrics"]["unigram"]["macro_f1"], 1)

    def test_changed_test_wording_cannot_change_choice_or_train_validation_review(self):
        rows = cases_from_generator()
        alternatives = {r["id"].removeprefix("challenge-"): r["question"]
                        for r in challenge_cases([r for r in rows if r["split"] == "test"])}
        for row in rows:
            if row["split"] == "test":
                row["question"] = "Please answer this question: " + alternatives[row["id"]]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "changed.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            original, changed = run(root / "original"), run(root / "changed", source)
            self.assertNotEqual(original["test"]["bigram"]["accuracy"], changed["test"]["bigram"]["accuracy"])
            self.assertEqual(original["training"], changed["training"])
            self.assertEqual(original["validation"], changed["validation"])
            self.assertEqual(original["train_validation_review"], changed["train_validation_review"])
            self.assertEqual((root / "original" / "selection.json").read_bytes(),
                             (root / "changed" / "selection.json").read_bytes())

    def test_smoke_rejects_tampered_gap_direction_and_training_metric(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            result = run(out)
            winner = result["selected"]
            printed = {"selected": winner, "test_accuracy": result["test"][winner]["accuracy"],
                       "challenge_accuracy": result["new_template_challenge"][winner]["accuracy"]}
            path = out / "summary.json"
            check_artifacts(out, printed)
            for kind, message in (("gap", "Train-validation gap mismatch"),
                                  ("direction", "Gap direction mismatch"),
                                  ("training", "Accuracy/count mismatch")):
                with self.subTest(kind=kind):
                    altered = deepcopy(result)
                    if kind == "training":
                        altered["training"]["bigram"]["accuracy"] = 0.5
                    else:
                        altered["train_validation_review"]["candidates"]["bigram"]["accuracy"][kind] = (
                            0.5 if kind == "gap" else "training_higher")
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        check_artifacts(out, printed)


if __name__ == "__main__":
    unittest.main()
