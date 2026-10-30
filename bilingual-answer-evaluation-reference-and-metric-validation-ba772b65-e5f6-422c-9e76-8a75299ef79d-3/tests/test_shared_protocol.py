"""Protocol checks use synthetic inputs; none of these fixtures are model weights."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cola_lab.answer_eval import (evaluate, make_dataset, method_inputs, predict,
                                  predict_from_input, run)
from cola_lab.local_gguf import generation_requests


class SharedProtocolTests(unittest.TestCase):
    def test_shared_input_is_selected_by_fixed_overlap_and_not_gold_evidence(self):
        dataset = make_dataset()
        case = next(case for case in dataset["cases"] if case["id"] == "test-en-Ember-color-q")
        inputs = method_inputs(case, dataset["documents"])
        value = inputs["lexical_retrieval"]
        self.assertEqual(value, inputs["local_model"])
        # Both documents have one overlapping word: the ID tie break chooses shape.
        self.assertEqual(value["documents"], [{"id": "test-en-Ember-shape",
                                              "text": "The shape of Ember is round."}])
        self.assertNotEqual([doc["id"] for doc in value["documents"]], case["evidence_ids"])
        self.assertEqual(predict_from_input(value, 2)["answer"], None)
        self.assertEqual(inputs["always_abstain"]["documents"], [])

    def test_baseline_reads_source_text_and_never_structured_answer_labels(self):
        dataset = make_dataset()
        for language, expected in (("en", "round"), ("he", "עגול")):
            case = next(case for case in dataset["cases"]
                        if case["id"] == f"test-{language}-Ember-shape-q")
            docs = copy.deepcopy(dataset["documents"])
            for doc in docs:
                doc["answer"] = "THIS LABEL MUST NEVER REACH THE ANSWERER"
            case = {**case, "expected_answer": "different scoring label", "evidence_ids": []}
            prediction = predict(case, docs, 2)
            self.assertEqual(prediction["answer"], expected)
            self.assertEqual(prediction["citations"], [f"test-{language}-Ember-shape"])
            value = method_inputs(case, docs)["local_model"]
            self.assertEqual(set(value), {"question", "documents"})
            self.assertEqual(set(value["documents"][0]), {"id", "text"})
            self.assertNotIn("LABEL", json.dumps(value))
        value = {"question": "shape Ember", "documents": [{"id": "doc", "text": "Ember shape unknown"}]}
        self.assertIsNone(predict_from_input(value, 0)["answer"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            predict_from_input({"question": "shape Ember", "documents": []}, 0)

    def test_document_and_case_order_do_not_change_inputs_prompts_or_pair_seeds(self):
        dataset = make_dataset()
        first = generation_requests(dataset, base_seed=41, repeats=2)
        dataset["documents"].reverse()
        dataset["cases"].reverse()
        self.assertEqual(first, generation_requests(dataset, base_seed=41, repeats=2))
        changed = generation_requests(dataset, base_seed=42, repeats=2)
        self.assertEqual(first[0]["inputs"], changed[0]["inputs"])
        self.assertNotEqual(first[0]["settings"]["seed"], changed[0]["settings"]["seed"])
        pairs = {}
        for request in first:
            key = (request["repeat_index"], request["pair_id"])
            pairs.setdefault(key, []).append(request["settings"])
        self.assertEqual(len(pairs), 24)
        self.assertTrue(all(len(settings) == 2 and settings[0] == settings[1]
                            for settings in pairs.values()))

    def test_label_free_inputs_ignore_changed_labels_but_protocol_rejects_them(self):
        dataset = make_dataset()
        changed = copy.deepcopy(dataset)
        for case in changed["cases"]:
            if case["id"] in ("test-en-Ember-shape-q", "test-he-Ember-shape-q"):
                case["expected_answer"] = None
                case["evidence_ids"] = []
        for original, mutated in zip(dataset["cases"], changed["cases"]):
            self.assertEqual(method_inputs(original, dataset["documents"]),
                             method_inputs(mutated, changed["documents"]))
        with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
            generation_requests(changed)
        with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
            evaluate(changed, "changed")

    def test_frozen_outputs_without_generation_provenance_are_not_labeled_shared(self):
        dataset = make_dataset()
        payload = {"model_id": "test-fixture", "model_version": "fixture-only",
                   "predictions": [{"case_id": case["id"], "answer": None, "citations": []}
                                   for case in dataset["cases"] if case["split"] == "test"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            report = json.loads(run(root / "out", source).read_text(encoding="utf-8"))
        control = report["protocol"]["information_control"]
        self.assertEqual(control["status"], "unverified_frozen_outputs")
        self.assertTrue(all(row["methods"]["local_model"] is None for row in control["case_exposures"]))

    def test_invalid_context_and_missing_candidate_documents_are_rejected(self):
        dataset = make_dataset()
        case = dataset["cases"][0]
        with self.assertRaisesRegex(ValueError, "Unknown context"):
            method_inputs(case, dataset["documents"], "unrecognized")
        with self.assertRaisesRegex(ValueError, "No documents"):
            method_inputs(case, [])


class CommandTests(unittest.TestCase):
    def test_documented_finite_smoke(self):
        result = subprocess.run([sys.executable, "-m", "cola_lab", "answers-smoke"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("baseline only, no model run", result.stdout)

    def test_cli_rejects_missing_model_and_ignored_protocol_flags(self):
        for args, message in ((["gguf"], "requires --model"),
                              (["answers", "--repeats", "2"], "only valid with gguf"),
                              (["answers-smoke", "--predictions", "unused.json"], "only valid with answers"),
                              (["gguf", "--model", "missing.gguf", "--repeats", "0"], "Repeats must")):
            result = subprocess.run([sys.executable, "-m", "cola_lab", *args],
                                    capture_output=True, text=True, timeout=30)
            with self.subTest(args=args):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
