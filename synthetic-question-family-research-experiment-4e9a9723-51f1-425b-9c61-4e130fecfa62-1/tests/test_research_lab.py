import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from portfolio.research_lab import cases_from_generator, validate_cases, TextBaseline, evaluate, run


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
