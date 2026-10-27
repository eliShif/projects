import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from portfolio.research_lab import cases_from_generator, validate_cases, challenge_cases, TextBaseline, evaluate, run


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
            self.assertIn("New-template diagnostic", (Path(first) / "REPORT.md").read_text())

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


if __name__ == "__main__":
    unittest.main()
