import csv
import json
import tempfile
import unittest
from pathlib import Path

from cola_lab.data import FIELDS, generate_rows, read_rows, write_rows
from cola_lab.experiment import (BigramModel, choose_threshold, describe, evaluate,
                                 metrics, roc_auc, run, score_histogram, scores,
                                 training_sensitivity)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.csv"

    def test_generation_and_csv_round_trip(self):
        write_rows(self.path, generate_rows())
        rows = read_rows(self.path)
        self.assertEqual({split: sum(row["split"] == split for row in rows)
                          for split in ("train", "validation", "test")},
                         {"train": 48, "validation": 24, "test": 24})
        self.assertEqual(len({row["sentence"] for row in rows}), len(rows))
        self.assertEqual(len({row["group_id"] for row in rows}), 24)

    def test_rejects_group_leakage(self):
        rows = generate_rows()
        rows[-1]["group_id"] = rows[0]["group_id"]
        write_rows(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            read_rows(self.path)

    def test_rejects_bad_label_and_missing_sentence(self):
        rows = generate_rows()
        rows[0]["label"] = 2
        write_rows(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Invalid split or label"):
            read_rows(self.path)
        rows = generate_rows()
        rows[0]["sentence"] = ""
        write_rows(self.path, rows)
        with self.assertRaisesRegex(ValueError, "missing value"):
            read_rows(self.path)

    def test_rejects_wrong_columns(self):
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1])
            writer.writeheader()
        with self.assertRaisesRegex(ValueError, "Expected CSV columns"):
            read_rows(self.path)

    def test_rejects_wrong_length_bucket(self):
        rows = generate_rows()
        rows[0]["length"] = "long"
        write_rows(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Length bucket disagrees"):
            read_rows(self.path)


class ScoringTests(unittest.TestCase):
    def test_histogram_counts_boundaries_and_constant_scores(self):
        rows = [{"length": "short", "label": 0},
                {"length": "short", "label": 1},
                {"length": "long", "label": 0},
                {"length": "long", "label": 1}]
        bins = score_histogram([0, 1, 2, 4], rows, bins=2)
        self.assertEqual(bins[0]["counts"]["short"], {"0": 1, "1": 1})
        self.assertEqual(bins[1]["counts"]["long"], {"0": 1, "1": 1})
        self.assertEqual((bins[0]["lower"], bins[0]["upper"], bins[1]["upper"]),
                         (0, 2, 4))
        flat = score_histogram([3, 3], rows[:2], bins=2)
        self.assertEqual(flat[0]["counts"]["short"], {"0": 1, "1": 1})
        self.assertEqual(flat[1]["counts"]["short"], {"0": 0, "1": 0})

    def test_histogram_rejects_invalid_inputs(self):
        row = {"length": "short", "label": 0}
        for values, rows, bins in (([], [], 2), ([1], [], 2), ([1], [row], 0),
                                   ([float("nan")], [row], 2)):
            with self.assertRaises(ValueError):
                score_histogram(values, rows, bins)
        with self.assertRaisesRegex(ValueError, "known length and label"):
            score_histogram([1], [{"length": "other", "label": 0}])

    def test_population_variance_and_empty_input(self):
        self.assertEqual(describe([1, 3]), {"count": 2, "mean": 2,
                                            "population_variance": 1,
                                            "minimum": 1, "maximum": 3})
        with self.assertRaisesRegex(ValueError, "empty"):
            describe([])

    def test_sum_and_mean_share_token_scores(self):
        model = BigramModel(["the chef writes", "the pilot reads"])
        sentence = "the chef writes"
        result = scores(model, sentence)
        self.assertAlmostEqual(result["sum"], result["mean"] * 4)
        self.assertLess(result["sum"], result["mean"])

    def test_threshold_and_auc_with_ties(self):
        labels = [0, 0, 1, 1]
        values = [0.1, 0.2, 0.8, 0.9]
        threshold = choose_threshold(labels, values)
        self.assertEqual(metrics(labels, [int(value >= threshold) for value in values])
                         ["balanced_accuracy"], 1)
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        with self.assertRaisesRegex(ValueError, "both classes"):
            roc_auc([1, 1], [0.1, 0.2])

    def test_report_is_reproducible_and_contains_failures(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = run(first)
            second_path = run(second)
            one = json.loads(first_path.read_text())
            two = json.loads(second_path.read_text())
            narrative = (first_path.parent / "REPORT.md").read_text()
            self.assertEqual(narrative, (second_path.parent / "REPORT.md").read_text())
            self.assertIn(f"| sum | {one['methods']['sum']['test']['accuracy']:.3f} |",
                          narrative)
            self.assertIn("pvariance", narrative)
            self.assertIn("Equal-width score bins", narrative)
        self.assertEqual(one, two)
        self.assertEqual(one["protocol"]["split_counts"],
                         {"train": 48, "validation": 24, "test": 24})
        for method in ("sum", "mean"):
            entry = one["methods"][method]
            self.assertIn("short", entry["by_length"])
            self.assertIn("long", entry["by_length"])
            self.assertTrue(entry["errors"])
            self.assertEqual(sum(entry["test"]["confusion"].values()), 24)
            self.assertEqual(entry["score_distribution"]["short"]["0"]["count"], 6)
            histogram = entry["score_histogram"]
            self.assertEqual(len(histogram), 5)
            for length in ("short", "long"):
                for label in ("0", "1"):
                    self.assertEqual(sum(bucket["counts"][length][label]
                                         for bucket in histogram), 6)
            self.assertLessEqual(histogram[0]["lower"], histogram[-1]["upper"])
            self.assertEqual(one["training_sensitivity"]["methods"][method]
                             ["accuracy"]["count"], 24)
            self.assertTrue(all(item["error_draws"] <= 24 for item in
                                one["training_sensitivity"]["methods"][method]
                                ["error_frequency"]))
        self.assertEqual(one["data_quality"]["test"]["label_counts"], {"0": 12, "1": 12})
        self.assertEqual(one["data_quality"]["test"]["group_count"], 6)

    def test_sensitivity_resamples_only_training_groups_and_rejects_zero_draws(self):
        rows = generate_rows()
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        test = [row for row in rows if row["split"] == "test"]
        result = training_sensitivity(train, validation, test, draws=3, seed=9)
        self.assertEqual(result, training_sensitivity(train, validation, test, draws=3, seed=9))
        train_groups = {row["group_id"] for row in train}
        for draw in result["methods"]["sum"]["draws"]:
            self.assertEqual(len(draw["sampled_groups"]), len(train_groups))
            self.assertLessEqual(set(draw["sampled_groups"]), train_groups)
        with self.assertRaisesRegex(ValueError, "At least one"):
            training_sensitivity(train, validation, test, draws=0)

    def test_sensitivity_test_labels_do_not_select_thresholds(self):
        rows = generate_rows()
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        test = [row for row in rows if row["split"] == "test"]
        original = training_sensitivity(train, validation, test, draws=3)
        changed = [{**row, "label": 1 - row["label"]} for row in test]
        altered = training_sensitivity(train, validation, changed, draws=3)
        for method in ("sum", "mean"):
            before = original["methods"][method]["draws"]
            after = altered["methods"][method]["draws"]
            self.assertEqual([r["validation_threshold"] for r in before],
                             [r["validation_threshold"] for r in after])
            self.assertNotEqual([r["accuracy"] for r in before],
                                [r["accuracy"] for r in after])

    def test_test_labels_cannot_change_validation_threshold(self):
        rows = generate_rows()
        original = evaluate(rows, "sample")
        for row in rows:
            if row["split"] == "test":
                row["label"] = 1 - row["label"]
        altered = evaluate(rows, "sample")
        for method in ("sum", "mean"):
            self.assertEqual(original["methods"][method]["validation_threshold"],
                             altered["methods"][method]["validation_threshold"])


if __name__ == "__main__":
    unittest.main()
