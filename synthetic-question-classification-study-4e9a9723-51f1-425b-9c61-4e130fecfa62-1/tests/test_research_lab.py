import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from portfolio.research_lab import (cases_from_generator, validate_cases, challenge_cases,
                                    TextBaseline, evaluate, grouped_stability, imbalance_stress,
                                    diagnostic_review, paired_wording_review, run)


class ResearchLabTests(unittest.TestCase):
    def test_grouped_data_and_reproducible_run(self):
        rows = cases_from_generator()
        quality = validate_cases(rows)
        self.assertEqual([quality[s]["cases"] for s in ("train", "validation", "test")], [24, 8, 16])
        self.assertEqual(len(set(quality["train"]["entities"]) & set(quality["test"]["entities"])), 0)
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            a, b = run(first), run(second)
            self.assertEqual(a, b)
            self.assertEqual(json.loads((Path(first) / "protocol.json").read_text()),
                             json.loads((Path(second) / "protocol.json").read_text()))
            self.assertIn(a["selected"], ("unigram", "bigram", "majority"))
            self.assertGreater(a["test"][a["selected"]]["accuracy"], a["test"]["majority"]["accuracy"])
            paired = json.loads((Path(first) / "challenge_cases.json").read_text())
            original = [r for r in rows if r["split"] == "test"]
            self.assertEqual(len(paired), len(original))
            self.assertEqual([(r["entity"], r["label"]) for r in paired],
                             [(r["entity"], r["label"]) for r in original])
            self.assertTrue(all(a["question"] != b["question"] for a, b in zip(paired, original)))
            self.assertEqual(a["new_template_challenge"], b["new_template_challenge"])
            self.assertEqual(sum(map(sum, a["new_template_challenge"][a["selected"]]["confusion"])), len(paired))
            self.assertEqual(a["test"][a["selected"]]["accuracy"], 1.0)
            self.assertEqual(a["new_template_challenge"][a["selected"]]["accuracy"], 0.625)
            self.assertEqual(len(a["new_template_challenge"][a["selected"]]["failures"]), 6)
            paired_review = a["paired_wording_review"]
            self.assertEqual(paired_review["cases"], 16)
            self.assertEqual(paired_review["transitions"]["correct_to_wrong"], 6)
            self.assertEqual(sum(paired_review["transitions"].values()), 16)
            self.assertEqual(sum(row["correct_to_wrong"] for row in paired_review["by_label"].values()), 6)
            self.assertEqual(len(paired_review["examples"]), 16)
            self.assertIn("New-template diagnostic", (Path(first) / "REPORT.md").read_text())
            self.assertEqual(len(a["grouped_stability"]["folds"]), 6)
            self.assertEqual(a["training_imbalance_stress"]["stressed_training_counts"]["lookup"], 1)
            self.assertLess(a["training_imbalance_stress"]["stressed"]["macro_f1"],
                            a["training_imbalance_stress"]["original"]["macro_f1"])
            self.assertEqual(a["training_imbalance_stress"]["stressed"]["minority_recall"], 0)
            self.assertEqual(len(a["challenge_error_review"]["failures"]), 6)
            self.assertTrue(all(0 <= r["unseen_feature_fraction"] <= 1
                                for r in a["challenge_error_review"]["failures"]))
            self.assertIn("Error review", (Path(first) / "REPORT.md").read_text())

    def test_missing_label_and_entity_leakage_fail(self):
        rows = cases_from_generator()
        bad = [dict(r) for r in rows]
        bad[0]["label"] = ""
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_cases(bad)
        bad = [dict(r) for r in rows]
        bad[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_cases(bad)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"
            source.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "leakage"):
                run(Path(directory) / "output", source)
            self.assertFalse((Path(directory) / "output").exists())

    def test_challenge_rejects_incomplete_or_mislabeled_holdout_before_writing(self):
        test = [r for r in cases_from_generator() if r["split"] == "test"]
        with self.assertRaisesRegex(ValueError, "one test case"):
            challenge_cases(test[:-1])
        bad = [dict(r) for r in test]
        bad[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "only use test"):
            challenge_cases(bad)
        with TemporaryDirectory() as directory:
            rows = cases_from_generator()
            rows.remove(next(r for r in rows if r["id"] == test[0]["id"]))
            source = Path(directory) / "incomplete.json"
            source.write_text(json.dumps(rows))
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "one test case"):
                run(output, source)
            self.assertFalse(output.exists())

    def test_metrics_identify_failures(self):
        rows = cases_from_generator()
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test"]
        result = evaluate(TextBaseline("majority").fit(train), test)
        self.assertEqual(result["accuracy"], 1 / 8)
        self.assertAlmostEqual(result["macro_f1"], 1 / 36)
        self.assertEqual(result["entity_accuracy_population_variance"], 0)
        self.assertEqual(len(result["failures"]), 14)
        self.assertEqual(sum(map(sum, result["confusion"])), len(test))

    def test_label_conflict_and_question_leakage_rejected(self):
        rows = cases_from_generator()
        conflict = [dict(r) for r in rows]
        conflict[1]["question"] = conflict[0]["question"]
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            validate_cases(conflict)
        leaked = [dict(r) for r in rows]
        source = next(r for r in leaked if r["split"] == "train")
        target = next(r for r in leaked if r["split"] == "test" and r["label"] == source["label"])
        target["question"] = source["question"]
        with self.assertRaisesRegex(ValueError, "question leakage"):
            validate_cases(leaked)

    def test_stability_imbalance_and_review_failure_paths(self):
        rows = cases_from_generator()
        train = [r for r in rows if r["split"] == "train"]
        held = [r for r in rows if r["split"] == "test"]
        folds = grouped_stability(rows, "majority")
        self.assertEqual({f["held_out_entity"] for f in folds["folds"]}, {r["entity"] for r in rows})
        self.assertTrue(all(f["train_cases"] + f["held_out_cases"] == len(rows) for f in folds["folds"]))
        stress = imbalance_stress(train, held, "majority", "lookup")
        self.assertEqual(stress["original_training_counts"]["lookup"], 3)
        self.assertEqual(stress["stressed_training_counts"]["lookup"], 1)
        self.assertEqual(stress["original"]["minority_recall"], 0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            imbalance_stress(train[:8], held, "majority", "lookup")
        with self.assertRaisesRegex(ValueError, "all training labels"):
            grouped_stability(rows[:8], "majority")
        model = TextBaseline("unigram").fit(train)
        result = evaluate(model, held)
        review = diagnostic_review(model, train, held, result)
        self.assertEqual(len(review["failures"]), len(result["failures"]))

    def test_paired_review_rejects_unmatched_and_changed_truth(self):
        rows = cases_from_generator()
        train = [r for r in rows if r["split"] == "train"]
        held = [r for r in rows if r["split"] == "test"]
        alternate = challenge_cases(held)
        model = TextBaseline("bigram").fit(train)
        with self.assertRaisesRegex(ValueError, "one alternate"):
            paired_wording_review(model, held, alternate[:-1])
        changed = [dict(r) for r in alternate]
        changed[0]["label"] = "ratio"
        with self.assertRaisesRegex(ValueError, "must match"):
            paired_wording_review(model, held, changed)
        changed = [dict(r) for r in alternate]
        changed[0]["question"] = held[0]["question"]
        with self.assertRaisesRegex(ValueError, "different wording"):
            paired_wording_review(model, held, changed)


if __name__ == "__main__":
    unittest.main()
