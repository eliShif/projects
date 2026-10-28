import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cola_lab.answer_eval import make_dataset
from cola_lab.local_gguf import parse_generation, prompt_for, run


class GGUFPathTests(unittest.TestCase):
    def test_prompt_excludes_labels_and_other_language(self):
        dataset = make_dataset()
        case = next(case for case in dataset["cases"] if case["id"] == "test-he-Ember-shape-q")
        prompt = prompt_for(case, dataset["documents"])
        self.assertIn("test-he-Ember-shape", prompt)
        self.assertNotIn("test-en-Ember-shape", prompt)
        self.assertNotIn("expected_answer", prompt)
        self.assertNotIn("validation-he", prompt)

    def test_valid_and_invalid_generation(self):
        self.assertEqual(parse_generation('{"answer":null,"citation":null}', {"doc"}),
                         {"answer": None, "citations": []})
        self.assertEqual(parse_generation('{"answer":"blue","citation":"doc"}', {"doc"}),
                         {"answer": "blue", "citations": ["doc"]})
        for raw, message in (("not json", "JSON object"),
                             ('{"answer":"blue"}', "answer and citation"),
                             ('{"answer":null,"citation":"doc"}', "both"),
                             ('{"answer":"blue","citation":"other"}', "outside")):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                parse_generation(raw, {"doc"})

    def test_run_reuses_scoring_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"test backend marker, not model weights")
            calls = []

            def backend(path, settings):
                self.assertEqual(path, model)
                self.assertEqual(settings["temperature"], 0)

                def generate(prompt):
                    calls.append(prompt)
                    return '{"answer":null,"citation":null}'

                return generate

            report_path = run(model, root / "out", backend)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads((root / "out/gguf_run.json").read_text(encoding="utf-8"))
            predictions = json.loads((root / "out/gguf_predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 24)
            self.assertEqual(len(manifest["generations"]), 24)
            self.assertEqual(len(predictions["predictions"]), 24)
            self.assertEqual(manifest["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
            self.assertEqual(manifest["predictions_sha256"],
                             hashlib.sha256((root / "out/gguf_predictions.json").read_bytes()).hexdigest())
            self.assertEqual(report["protocol"]["model_outputs"]["output_sha256"],
                             manifest["predictions_sha256"])
            self.assertEqual(report["methods"]["local_model"]["overall"]["accuracy"], 8 / 24)
            self.assertEqual(report["methods"]["local_model"]["overall"]["error_counts"],
                             {"false_abstention": 16})
            self.assertIn("local_model", report["comparisons"])

    def test_missing_model_and_bad_response_fail_without_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "existing local"):
                run(root / "missing.gguf", root / "out")
            model = root / "fixture.gguf"
            model.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "Invalid generation for test-en-Ember-color-q"):
                run(model, root / "out", lambda path, settings: lambda prompt: "bad output")
            self.assertFalse((root / "out/answer_report.json").exists())


if __name__ == "__main__":
    unittest.main()
