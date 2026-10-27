import csv
import json
import tempfile
import unittest
from pathlib import Path

from cola_lab.data import FIELDS, generate_rows, read_rows, write_rows
from cola_lab.experiment import BigramModel, choose_threshold, evaluate, metrics, roc_auc, run, scores


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
            one = json.loads(run(first).read_text())
            two = json.loads(run(second).read_text())
        self.assertEqual(one, two)
        self.assertEqual(one["protocol"]["split_counts"],
                         {"train": 48, "validation": 24, "test": 24})
        for method in ("sum", "mean"):
            entry = one["methods"][method]
            self.assertIn("short", entry["by_length"])
            self.assertIn("long", entry["by_length"])
            self.assertTrue(entry["errors"])
            self.assertEqual(sum(entry["test"]["confusion"].values()), 24)

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
