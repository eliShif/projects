"""Learning-curve arithmetic, split isolation and real test-label controls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from portfolio.research_lab import (
    CANDIDATES, FAMILIES, SPLITS, TextBaseline, cases_from_generator, evaluate,
    generalization_gap, learning_curve, learning_curve_subsets, mentioned_entities, run,
)
from scripts.smoke_research_lab import check_artifacts


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.rows = cases_from_generator()
        self.train = [r for r in self.rows if r["split"] == "train"]
        self.validation = [r for r in self.rows if r["split"] == "validation"]

    def test_hand_calculated_signed_gap_from_real_predictions(self):
        # A deliberately skewed two-family fixture, not an experiment result.
        train = [{"id": str(i), "entity": "Cobalt", "question": "same words",
                  "label": "lookup" if i < 3 else "ratio"} for i in range(4)]
        validation = [{**row, "label": "lookup" if i == 0 else "ratio"}
                      for i, row in enumerate(train)]
        model = TextBaseline("majority").fit(train)
        a, b = evaluate(model, train), evaluate(model, validation)
        # Always predicting lookup: train TP=3, FP=1, FN=0; validation TP=1,
        # FP=3, FN=0. Seven other family F1s are zero in the fixed macro mean.
        self.assertEqual(a["accuracy"], 3 / 4)
        self.assertEqual(b["accuracy"], 1 / 4)
        self.assertAlmostEqual(a["macro_f1"], (6 / 7) / 8)
        self.assertAlmostEqual(b["macro_f1"], (2 / 5) / 8)
        gap = generalization_gap(a, b)
        self.assertEqual(gap["accuracy"], 1 / 2)
        self.assertAlmostEqual(gap["macro_f1"], 2 / 35)
        self.assertEqual(generalization_gap(b, a)["accuracy"], -1 / 2)
        self.assertEqual(generalization_gap(a, a), {"accuracy": 0, "macro_f1": 0})

    def test_gap_rejects_missing_nonfinite_and_invalid_metric_values(self):
        good = {"accuracy": 0.5, "macro_f1": 0.25}
        for metric in good:
            for value in (None, float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
                for side in (0, 1):
                    with self.subTest(metric=metric, value=value, side=side):
                        pair = [good, good]
                        pair[side] = {**good, metric: value}
                        with self.assertRaisesRegex(ValueError, "finite"):
                            generalization_gap(*pair)
        with self.assertRaisesRegex(ValueError, "macro_f1"):
            generalization_gap({"accuracy": 0.5}, good)

    def test_nested_samples_cover_classes_and_keep_all_mentions_disjoint(self):
        before = copy.deepcopy(self.rows)
        subsets = learning_curve_subsets(self.train, self.validation)
        self.assertEqual([len(sample) for sample in subsets], [8, 16, 24])
        previous = []
        validation_mentions = set().union(*(mentioned_entities(r["question"]) for r in self.validation))
        for size, sample in zip((8, 16, 24), subsets):
            ids = [r["id"] for r in sample]
            self.assertEqual(len(set(ids)), size)
            self.assertEqual(ids[:len(previous)], previous)
            previous = ids
            self.assertEqual({r["label"] for r in sample}, set(FAMILIES))
            self.assertEqual({r["split"] for r in sample}, {"train"})
            mentions = set().union(*(mentioned_entities(r["question"]) for r in sample))
            self.assertTrue(mentions <= set(SPLITS["train"]))
            self.assertFalse(mentions & validation_mentions)
            self.assertTrue(all(sum(r["label"] == f for r in sample) == size // 8 for f in FAMILIES))
        self.assertEqual(set(previous), {r["id"] for r in self.train})
        self.assertEqual(self.rows, before)
        self.assertEqual(subsets, learning_curve_subsets(list(reversed(self.train)), self.validation))
        self.assertNotEqual(subsets[0], learning_curve_subsets(self.train, self.validation, seed=23)[0])

    def test_unbalanced_training_still_covers_every_family(self):
        training = [r for r in self.train if r["label"] != "ratio" or r["entity"] == "Cobalt"]
        samples = learning_curve_subsets(training, self.validation, (8, 16, 22))
        for sample in samples:
            self.assertEqual({r["label"] for r in sample}, set(FAMILIES))
            self.assertEqual(sum(r["label"] == "ratio" for r in sample), 1)
        self.assertEqual({r["id"] for r in samples[-1]}, {r["id"] for r in training})

    def test_curve_uses_only_training_fits_and_fixed_validation(self):
        fits = []
        original_fit = TextBaseline.fit

        def trace(model, rows):
            fits.append((model.name, [r["id"] for r in rows], {r["split"] for r in rows}))
            return original_fit(model, rows)

        with patch.object(TextBaseline, "fit", trace):
            curve = learning_curve(self.train, self.validation)
        self.assertEqual(len(fits), 9)
        self.assertTrue(all(splits == {"train"} for _, _, splits in fits))
        self.assertEqual(curve["validation_case_ids"], sorted(r["id"] for r in self.validation))
        self.assertEqual(curve, learning_curve(list(reversed(self.train)), list(reversed(self.validation))))
        by_id = {r["id"]: r for r in self.train}
        for point in curve["points"]:
            ids = point["training_case_ids"]
            for name in CANDIDATES:
                self.assertIn((name, ids, {"train"}), fits)
                sample = [by_id[case_id] for case_id in ids]
                model = TextBaseline(name).fit(sample)
                for split, data in (("train", sample), ("validation", self.validation)):
                    actual = evaluate(model, data)
                    for key, value in point["metrics"][name][split].items():
                        self.assertEqual(value, actual[key])
            majority = point["metrics"]["majority"]
            self.assertEqual(majority["train"]["accuracy"], 1 / 8)
            self.assertAlmostEqual(majority["validation"]["macro_f1"], 1 / 36)

    def test_controlled_validation_label_shift_produces_measured_positive_gap(self):
        # A deliberate label-shift control demonstrates that a gap is not a
        # causal overfitting diagnosis. It is never used as the saved experiment.
        changed = [{**r, "label": FAMILIES[(FAMILIES.index(r["label"]) + 1) % len(FAMILIES)]}
                   for r in self.validation]
        curve = learning_curve(self.train, changed)
        for point in curve["points"]:
            for name in ("unigram", "bigram"):
                score = point["metrics"][name]
                self.assertEqual(score["train"]["accuracy"], 1)
                self.assertEqual(score["validation"]["accuracy"], 0)
                self.assertEqual(score["train_minus_validation"], {"accuracy": 1, "macro_f1": 1})

    def test_invalid_schedule_or_seed_fails_before_fitting_or_writing(self):
        bad_sizes = ((), (8,), (7, 16), (8, 25), (16, 8), (8, 8), (8, 16.0), (8, True), "8 16")
        with TemporaryDirectory() as directory:
            for sizes in bad_sizes:
                with self.subTest(sizes=sizes), patch.object(TextBaseline, "fit") as fit:
                    out = Path(directory) / "rejected"
                    with self.assertRaisesRegex(ValueError, "strictly increasing integers"):
                        run(out, train_sizes=sizes)
                    fit.assert_not_called()
                    self.assertFalse(out.exists())
            for seed in (True, 1.5, "19"):
                with self.subTest(seed=seed), patch.object(TextBaseline, "fit") as fit:
                    with self.assertRaisesRegex(ValueError, "seed must be an integer"):
                        run(Path(directory) / "rejected", curve_seed=seed)
                    fit.assert_not_called()

    def test_curve_rejects_test_rows_missing_classes_and_secondary_leakage(self):
        leaked = copy.deepcopy(self.train)
        target = next(r for r in leaked if r["entity"] == "Cobalt" and r["label"] == "sum")
        target["question"] = "What is the combined quota for Cobalt and fJoRdMirror?"
        invalid = [
            (leaked, self.validation, "leakage through question mentions"),
            (self.train, self.validation + [next(r for r in self.rows if r["split"] == "test")], "Unknown split"),
            ([r for r in self.train if r["label"] != "ratio"], self.validation, "missing: ratio"),
            (self.train, [r for r in self.validation if r["label"] != "ratio"], "validation must cover"),
            (self.validation, self.train, "separate train and validation"),
            (self.train + [self.train[0]], self.validation, "Duplicate case id"),
        ]
        for train, validation, error in invalid:
            with self.subTest(error=error), patch.object(TextBaseline, "fit") as fit:
                with self.assertRaisesRegex(ValueError, error):
                    learning_curve(train, validation)
                fit.assert_not_called()

    def test_permuting_real_test_labels_cannot_change_selection_or_curve(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row["split"] == "test":
                # Keep full family support per entity, but change every truth.
                # Custom IDs correctly opt out of the generator provenance ledger.
                row["id"] = "label-control-" + row["id"]
                row["label"] = FAMILIES[(FAMILIES.index(row["label"]) + 1) % len(FAMILIES)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "permuted.json"
            source.write_text(json.dumps(changed), encoding="utf-8")
            original = run(root / "original")
            permuted = run(root / "permuted", source)
            self.assertEqual((root / "original/selection.json").read_bytes(),
                             (root / "permuted/selection.json").read_bytes())
            self.assertEqual(original["learning_curve"], permuted["learning_curve"])
            self.assertEqual(original["selected"], permuted["selected"])
            self.assertEqual(original["test"]["bigram"]["accuracy"], 1)
            self.assertEqual(permuted["test"]["bigram"]["accuracy"], 0)

    def test_cli_records_custom_schedule_and_rejects_bad_sizes(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "valid"
            command = [sys.executable, "-m", "portfolio.research_lab", "--out", str(out)]
            completed = subprocess.run(command + ["--train-sizes", "8", "24", "--curve-seed", "23"],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            printed = json.loads(completed.stdout)
            _, summary = check_artifacts(out, printed)
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertEqual(protocol["learning_curve"]["training_sizes"], [8, 24])
            self.assertEqual(protocol["learning_curve"]["seed"], 23)
            self.assertEqual(summary["learning_curve"]["protocol"], protocol["learning_curve"])
            self.assertIn("Train–validation learning curve", (out / "REPORT.md").read_text())
            saved = {p.name: p.read_bytes() for p in out.iterdir()}
            rejected = subprocess.run(command + ["--train-sizes", "8", "25"],
                                      capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("strictly increasing integers", rejected.stderr)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_smoke_rejects_altered_curve_gap_and_subset(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            summary = run(out)
            winner = summary["selected"]
            printed = {"selected": winner, "test_accuracy": summary["test"][winner]["accuracy"],
                       "challenge_accuracy": summary["new_template_challenge"][winner]["accuracy"]}
            check_artifacts(out, printed)
            bad = copy.deepcopy(summary)
            bad["learning_curve"]["points"][0]["metrics"]["bigram"]["train_minus_validation"]["accuracy"] += 0.25
            (out / "summary.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Learning-curve gap"):
                check_artifacts(out, printed)
            bad = copy.deepcopy(summary)
            bad["learning_curve"]["points"][0]["training_case_ids"][0] = self.validation[0]["id"]
            (out / "summary.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Learning-curve subset"):
                check_artifacts(out, printed)


if __name__ == "__main__":
    unittest.main()
