import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from portfolio.research_lab import (cases_from_generator, validate_cases, challenge_cases,
                                    TextBaseline, evaluate, grouped_stability, imbalance_stress,
                                    diagnostic_review, paired_wording_review, audit_generator_labels,
                                    template_overlap, label_distribution_review, resample_stability,
                                    score_margin_review, run)


class ResearchLabTests(unittest.TestCase):
    def test_grouped_data_and_reproducible_run(self):
        rows = cases_from_generator()
        quality = validate_cases(rows)
        self.assertEqual([quality[s]["cases"] for s in ("train", "validation", "test")], [24, 16, 16])
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
            self.assertEqual(len(a["grouped_stability"]["folds"]), 7)
            resampling = a["training_resample_stability"]
            self.assertEqual(len(resampling["runs"]), 12)
            self.assertEqual([r["seed"] for r in resampling["runs"]], list(range(12)))
            self.assertEqual(set(resampling["test_error_counts"]), {r["id"] for r in original})
            self.assertTrue(all(0 <= count <= 12 for count in resampling["challenge_error_counts"].values()))
            self.assertIn("Twelve seeded", (Path(first) / "REPORT.md").read_text())
            self.assertEqual(a["label_audit"]["matched_source_ids"], len(rows))
            self.assertEqual(a["label_audit"]["unmatched_ids"], [])
            self.assertEqual(a["template_overlap"]["test"]["seen"], 16)
            self.assertEqual(a["template_overlap"]["challenge"]["seen"], 0)
            self.assertIn("Template reuse audit", (Path(first) / "REPORT.md").read_text())
            distribution = a["label_distribution"]
            self.assertEqual(distribution["total_variation_from_train"], {"validation": 0.0, "test": 0.0})
            self.assertAlmostEqual(distribution["per_label"]["lookup"]["mean_split_fraction"], 1 / 8)
            self.assertEqual(distribution["per_label"]["lookup"]["population_variance_split_fraction"], 0)
            self.assertIn("Label distribution across splits", (Path(first) / "REPORT.md").read_text())
            margins = a["score_margins"]
            self.assertEqual(margins["bigram"]["test"]["all"]["count"], len(original))
            self.assertEqual(margins["bigram"]["challenge"]["wrong"]["count"],
                             len(a["new_template_challenge"]["bigram"]["failures"]))
            self.assertIn("Class score margin distribution", (Path(first) / "REPORT.md").read_text())
            fold_accuracy = [fold["accuracy"] for fold in a["grouped_stability"]["folds"]]
            mean = sum(fold_accuracy) / len(fold_accuracy)
            self.assertAlmostEqual(a["grouped_stability"]["accuracy_population_variance"],
                                   sum((value - mean) ** 2 for value in fold_accuracy) / len(fold_accuracy))
            self.assertEqual(a["training_imbalance_stress"]["stressed_training_counts"]["lookup"], 1)
            self.assertLess(a["training_imbalance_stress"]["stressed"]["macro_f1"],
                            a["training_imbalance_stress"]["original"]["macro_f1"])
            self.assertEqual(a["training_imbalance_stress"]["stressed"]["minority_recall"], 0)
            self.assertEqual(len(a["challenge_error_review"]["failures"]), 6)
            self.assertTrue(all(0 <= r["unseen_feature_fraction"] <= 1
                                for r in a["challenge_error_review"]["failures"]))
            self.assertIn("Error review", (Path(first) / "REPORT.md").read_text())
            self.assertIn("Per-family held-out evaluation", (Path(first) / "REPORT.md").read_text())
            self.assertEqual(a["test"]["majority"]["per_label"]["conflict"]["support"], 2)

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

    def test_generator_label_audit_rejects_mismatch_before_writing(self):
        rows = cases_from_generator()
        mislabeled = [dict(row) for row in rows]
        mislabeled[0]["label"] = "ratio"
        with TemporaryDirectory() as directory:
            source = Path(directory) / "mislabeled.json"
            source.write_text(json.dumps(mislabeled), encoding="utf-8")
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "Generator label/entity mismatch"):
                run(output, source)
            self.assertFalse(output.exists())
        custom = [dict(row) for row in rows]
        custom[0]["id"] = "custom-case"
        audit = audit_generator_labels(custom)
        self.assertEqual(audit["matched_source_ids"], len(rows) - 1)
        self.assertEqual(audit["unmatched_ids"], ["custom-case"])

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
        # Alphabetical tie breaking selects conflict. Two true conflict cases
        # and fourteen other cases are predicted conflict.
        selected = result["per_label"]["conflict"]
        self.assertEqual(selected, {"support": 2, "predicted": 16,
                                  "precision": 1 / 8, "recall": 1.0, "f1": 2 / 9})
        self.assertEqual(result["per_label"]["ratio"]["f1"], 0)
        self.assertAlmostEqual(result["macro_f1"],
                               sum(item["f1"] for item in result["per_label"].values()) / 8)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            evaluate(TextBaseline("majority").fit(train), [])

    def test_bigram_representation_can_distinguish_reversed_word_order(self):
        training = [{"label": "lookup", "question": "red blue"},
                    {"label": "ratio", "question": "blue red"}]
        unigram = TextBaseline("unigram").fit(training)
        bigram = TextBaseline("bigram").fit(training)
        # Equal unigram counts lose order. Adding bigrams separates these two
        # constructed examples; this does not claim general language ability.
        self.assertEqual(unigram.predict("red blue"), unigram.predict("blue red"))
        self.assertEqual(bigram.predict("red blue"), "lookup")
        self.assertEqual(bigram.predict("blue red"), "ratio")
        with self.assertRaisesRegex(ValueError, "Unknown candidate"):
            TextBaseline("unimplemented")

    def test_score_margin_statistics_and_failure_paths(self):
        training = [{"label": "lookup", "question": "alpha alpha"},
                    {"label": "ratio", "question": "beta beta"}]
        model = TextBaseline("unigram").fit(training)
        cases = [{"id": "a", "label": "lookup", "question": "alpha"},
                 {"id": "b", "label": "lookup", "question": "beta"}]
        review = score_margin_review(model, cases)
        self.assertEqual([item["correct"] for item in review["examples"]], [True, False])
        self.assertAlmostEqual(review["all"]["mean"], 1.0986122886681098)
        self.assertAlmostEqual(review["all"]["population_variance"], 0)
        self.assertEqual(review["correct"]["count"], 1)
        self.assertEqual(review["wrong"]["count"], 1)
        self.assertEqual(score_margin_review(model, cases[:1])["wrong"]["mean"], None)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            score_margin_review(model, [])
        with self.assertRaisesRegex(ValueError, "scored classifier"):
            score_margin_review(TextBaseline("majority").fit(training), cases)
        with self.assertRaisesRegex(ValueError, "two trained labels"):
            score_margin_review(TextBaseline("unigram").fit(training[:1]), cases)

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
        self.assertTrue(all(f["train_cases"] + f["held_out_cases"] + len(f["excluded_case_ids"]) == len(rows)
                            for f in folds["folds"]))
        for fold in folds["folds"]:
            self.assertFalse(set(fold["training_mentions"]) & set(fold["held_out_mentions"]))
            self.assertTrue(fold["excluded_case_ids"])
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

    def test_seeded_resampling_is_reproducible_and_rejects_bad_inputs(self):
        rows = cases_from_generator()
        train = [r for r in rows if r["split"] == "train"]
        held = [r for r in rows if r["split"] == "test"]
        alternate = challenge_cases(held)
        first = resample_stability(train, held, alternate, "bigram", (3, 7, 11))
        self.assertEqual(first, resample_stability(list(reversed(train)), held, alternate, "bigram", (3, 7, 11)))
        self.assertEqual(len(first["runs"]), 3)
        self.assertEqual(sum(first["test_error_counts"].values()),
                         sum((1 - run["test_accuracy"]) * len(held) for run in first["runs"]))
        self.assertEqual(sum(first["challenge_error_counts"].values()),
                         sum((1 - run["challenge_accuracy"]) * len(alternate) for run in first["runs"]))
        self.assertEqual({len(run["sample_ids"]) for run in first["runs"]}, {len(train)})
        with self.assertRaisesRegex(ValueError, "distinct integer seeds"):
            resample_stability(train, held, alternate, "bigram", (1, 1))
        with self.assertRaisesRegex(ValueError, "distinct integer seeds"):
            resample_stability(train, held, alternate, "bigram", ())
        with self.assertRaisesRegex(ValueError, "every training label"):
            resample_stability([r for r in train if r["label"] != "lookup"], held, alternate, "bigram", (1,))
        with self.assertRaisesRegex(ValueError, "frozen train and test"):
            resample_stability(train, train, alternate, "bigram", (1,))

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

    def test_template_audit_masks_whole_entity_names_and_rejects_empty_input(self):
        training = [{"id": "a", "entity": "Oak", "label": "lookup", "question": "What is Oak storage?"}]
        evaluated = [{"id": "b", "entity": "Pine", "label": "lookup", "question": "What is Pine storage?"},
                     {"id": "c", "entity": "Pine", "label": "lookup", "question": "What is Pineapple storage?"}]
        result = template_overlap(training, evaluated)
        self.assertEqual(result["seen"], 1)
        self.assertEqual(result["seen_fraction"], 0.5)
        self.assertEqual(result["by_label"]["lookup"], {"seen": 1, "total": 2})
        suffix_train = [{"id": "a", "entity": "Oak", "label": "conflict", "question": "Review OakMirror now"}]
        suffix_test = [{"id": "b", "entity": "Pine", "label": "conflict", "question": "Review PineMirror now"},
                       {"id": "c", "entity": "Pine", "label": "conflict", "question": "Review PineSandbox now"}]
        self.assertEqual(template_overlap(suffix_train, suffix_test)["seen"], 1)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            template_overlap(training, [])
        primary_only = [{"id": "x", "entity": "Cobalt", "label": "sum", "question": "Add Cobalt and Delta"}]
        held = [{"id": "y", "entity": "Garnet", "label": "sum", "question": "Add Garnet and Helix"}]
        self.assertEqual(template_overlap(primary_only, held)["seen"], 1)

    def test_label_distribution_detects_skew_and_rejects_bad_counts(self):
        quality = validate_cases(cases_from_generator())
        skew = json.loads(json.dumps(quality))
        skew["validation"]["label_counts"]["lookup"] = 4
        skew["validation"]["label_counts"]["conversion"] = 0
        result = label_distribution_review(skew)
        self.assertAlmostEqual(result["total_variation_from_train"]["validation"], 1 / 8)
        self.assertEqual(result["total_variation_from_train"]["test"], 0)
        self.assertAlmostEqual(result["per_label"]["lookup"]["mean_split_fraction"], 1 / 6)
        self.assertAlmostEqual(result["per_label"]["lookup"]["population_variance_split_fraction"], 1 / 288)
        skew["validation"]["label_counts"]["lookup"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            label_distribution_review(skew)
        skew = json.loads(json.dumps(quality))
        skew["test"]["label_counts"]["lookup"] += 1
        with self.assertRaisesRegex(ValueError, "sum"):
            label_distribution_review(skew)


if __name__ == "__main__":
    unittest.main()
