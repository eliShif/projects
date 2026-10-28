import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cola_lab.answer_eval import make_dataset, predict_from_input
from cola_lab.local_gguf import (GENERATION_KEYS, SETTINGS, create_backend, generation_requests,
                                 parse_generation, prompt_for, run, validate_records)


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

                def generate(prompt, call_settings):
                    calls.append((prompt, copy.deepcopy(call_settings)))
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
            self.assertIn("local_model", report["comparisons_with_lexical"])
            self.assertEqual(manifest["backend"], "injected-test-backend")
            self.assertFalse(manifest["model_quality_evidence"])
            self.assertFalse(report["protocol"]["model_outputs"]["model_quality_evidence"])
            self.assertEqual(report["protocol"]["information_control"]["status"],
                             "shared_document_verified")
            for (prompt, settings), record in zip(calls, manifest["generations"]):
                self.assertEqual(prompt, record["prompt"])
                self.assertEqual(settings, record["settings"])
                self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                                 record["prompt_sha256"])
                self.assertEqual(record["inputs"]["lexical_retrieval"], record["inputs"]["local_model"])
                self.assertEqual(record["exposures"]["lexical_retrieval"],
                                 record["exposures"]["local_model"])
                self.assertEqual(len(record["inputs"]["local_model"]["documents"]), 1)
            entry = manifest["repeat_artifacts"][0]
            self.assertEqual(entry["report_sha256"], hashlib.sha256(report_path.read_bytes()).hexdigest())

    def test_missing_model_and_bad_response_fail_without_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "existing local"):
                run(root / "missing.gguf", root / "out")
            model = root / "fixture.gguf"
            model.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "Invalid generation for test-en-Ember-color-q"):
                run(model, root / "out", lambda path, settings: lambda prompt, call_settings: "bad output")
            self.assertFalse((root / "out/answer_report.json").exists())
            self.assertFalse((root / "out/gguf_run.json").exists())

    def test_same_language_but_unexposed_citation_rejected_and_legacy_mode_labeled(self):
        def backend(path, settings):
            def generate(prompt, call_settings):
                if "Question: Which hue does Ember have?" in prompt:
                    # The fixed tie break selects Ember's shape, not this color fact.
                    return '{"answer":"blue","citation":"test-en-Ember-color"}'
                return '{"answer":null,"citation":null}'
            return generate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"injected test marker, not weights")
            with self.assertRaisesRegex(ValueError, "outside the case evidence pool"):
                run(model, root / "shared", backend)
            self.assertFalse((root / "shared/answer_report.json").exists())
            path = run(model, root / "legacy", backend, context_mode="all-documents")
            report = json.loads(path.read_text(encoding="utf-8"))
            control = report["protocol"]["information_control"]
            self.assertEqual(control["status"], "different_evidence_system_comparison")
            for row in control["case_exposures"]:
                self.assertEqual(len(row["methods"]["local_model"]["document_ids"]), 8)
                self.assertEqual(len(row["methods"]["lexical_retrieval"]["document_ids"]), 1)
            self.assertEqual(report["methods"]["local_model"]["overall"]["accuracy"], 9 / 24)

    def test_actual_answerer_inputs_match_the_saved_generation_inputs(self):
        baseline_inputs, model_prompts = [], []

        def observed_baseline(value, threshold):
            if value["documents"][0]["id"].startswith("test-"):
                baseline_inputs.append(copy.deepcopy(value))
            return predict_from_input(value, threshold)

        def backend(path, settings):
            def generate(prompt, call_settings):
                model_prompts.append(prompt)
                return '{"answer":null,"citation":null}'
            return generate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"not weights")
            with patch("cola_lab.answer_eval.predict_from_input", side_effect=observed_baseline):
                run(model, root / "out", backend)
            manifest = json.loads((root / "out/gguf_run.json").read_text(encoding="utf-8"))
        self.assertEqual(len(baseline_inputs), 24)
        self.assertEqual(len(model_prompts), 24)
        for record, actual_prompt in zip(manifest["generations"], model_prompts):
            self.assertEqual(actual_prompt, record["prompt"])
            actual_baseline = next(value for value in baseline_inputs
                                   if value["question"] == record["inputs"]["local_model"]["question"])
            self.assertEqual(actual_baseline, record["inputs"]["local_model"])
            for doc in actual_baseline["documents"]:
                self.assertIn(f"{doc['id']}: {doc['text']}", actual_prompt)

    def test_repeats_keep_cases_and_pairs_fixed_and_are_reproducible(self):
        def backend(path, settings):
            return lambda prompt, call_settings: '{"answer":null,"citation":null}'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"not weights")
            manifests = []
            for name in ("one", "two"):
                run(model, root / name, backend, repeats=2, base_seed=31)
                manifest = json.loads((root / name / "gguf_run.json").read_text(encoding="utf-8"))
                manifests.append(manifest)
                self.assertEqual(len(manifest["generations"]), 48)
                self.assertEqual(len(manifest["repeat_artifacts"]), 2)
                for artifact in manifest["repeat_artifacts"]:
                    path = root / name / artifact["report_file"]
                    report = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(report["protocol"]["information_control"]["repeat_index"],
                                     artifact["repeat_index"])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["report_sha256"])
                    self.assertEqual(report["paired_audit"]["pair_count"], 12)
                    self.assertEqual(artifact["dataset_sha256"], manifest["dataset_sha256"])
            self.assertEqual(manifests[0], manifests[1])
            records = {(record["repeat_index"], record["case_id"]): record
                       for record in manifests[0]["generations"]}
            for repeat in (0, 1):
                en = records[repeat, "test-en-Ember-color-q"]
                he = records[repeat, "test-he-Ember-color-q"]
                self.assertEqual(en["settings"], he["settings"])
            one = records[0, "test-en-Ember-color-q"]
            two = records[1, "test-en-Ember-color-q"]
            self.assertNotEqual(one["settings"]["seed"], two["settings"]["seed"])
            self.assertEqual({key: value for key, value in one["settings"].items() if key != "seed"},
                             {key: value for key, value in two["settings"].items() if key != "seed"})
            self.assertEqual(one["prompt"], two["prompt"])

    def test_backend_resets_and_receives_every_fixed_sampling_setting_per_call(self):
        events = []

        class LlamaSpy:
            def __init__(self, **kwargs):
                events.append(("init", kwargs))

            def reset(self):
                events.append(("reset", None))

            def __call__(self, prompt, **kwargs):
                events.append((prompt, kwargs))
                return {"choices": [{"text": '{"answer":null,"citation":null}'}]}

        requests = generation_requests(make_dataset(), repeats=2)
        with patch.dict("sys.modules", {"llama_cpp": SimpleNamespace(Llama=LlamaSpy)}):
            generate = create_backend(Path("unused-test-path.gguf"), SETTINGS)
            for request in (requests[0], requests[-1]):
                generate(request["prompt"], request["settings"])
        self.assertEqual(events[0][1]["n_threads"], 1)
        self.assertEqual(events[0][1]["n_threads_batch"], 1)
        self.assertEqual(events[0][1]["n_batch"], 128)
        self.assertEqual([events[i][0] for i in (1, 3)], ["reset", "reset"])
        for event, request in zip((events[2], events[4]), (requests[0], requests[-1])):
            self.assertEqual(event[0], request["prompt"])
            self.assertEqual(event[1], {"stream": False,
                **{key: request["settings"][key] for key in GENERATION_KEYS}})

    def test_record_validation_rejects_input_setting_prompt_and_output_drift(self):
        dataset = make_dataset()
        records = [{**request, "raw_generation": '{"answer":null,"citation":null}',
                    "prediction": {"answer": None, "citations": []}}
                   for request in generation_requests(dataset, repeats=2)]
        validate_records(dataset, records, "shared-document", 17, 2)
        mutations = (
            (lambda rows: rows.pop(), "cover every"),
            (lambda rows: rows.append(rows[0]), "duplicate"),
            (lambda rows: rows[0]["inputs"]["local_model"]["documents"].clear(), "changed inputs"),
            (lambda rows: rows[0]["settings"].update(seed=1), "changed settings"),
            (lambda rows: rows[0]["settings"].update(temperature=0.7), "changed settings"),
            (lambda rows: rows[0].update(prompt="extra information"), "changed prompt"),
            (lambda rows: rows[0]["prediction"].update(answer="fabricated"), "disagrees"),
        )
        for mutate, message in mutations:
            broken = copy.deepcopy(records)
            mutate(broken)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_records(dataset, broken, "shared-document", 17, 2)

    def test_invalid_configuration_and_nonempty_output_fail_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"not weights")

            def backend(path, settings):
                self.fail("Invalid configuration must not initialize inference")

            for options in ({"repeats": 0}, {"repeats": 101}, {"repeats": True},
                            {"base_seed": -1}, {"base_seed": 2147483647},
                            {"context_mode": "unknown"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run(model, root / "out", backend, **options)
            output = root / "existing"
            output.mkdir()
            old_report = output / "answer_report.json"
            old_report.write_text("previous result", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                run(model, output, backend)
            self.assertEqual(old_report.read_text(encoding="utf-8"), "previous result")


if __name__ == "__main__":
    unittest.main()
