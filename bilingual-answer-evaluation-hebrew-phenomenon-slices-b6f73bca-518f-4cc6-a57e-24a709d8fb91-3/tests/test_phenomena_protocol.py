"""Challenge corpus through the retained CLI, frozen inputs and GGUF replay.

Successful backend calls below are explicitly injected fixtures, never weights.
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cola_lab.gguf_audit import verify_bundle
from cola_lab.local_gguf import load_protocol, prepare, run, sha256_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PhenomenaProtocolTests(unittest.TestCase):
    def test_plan_preserves_variants_and_has_equal_label_free_inputs_and_pair_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = prepare(Path(directory) / "plan", dataset_name="hebrew-phenomena", repeats=2)
            protocol = load_protocol(path)
        self.assertEqual(protocol["dataset_name"], "hebrew-phenomena")
        self.assertEqual(len(protocol["requests"]), 48)
        self.assertIn("phenomena.py", protocol["source_sha256"])
        self.assertIn("resources/hebrew_phenomena.json", protocol["source_sha256"])
        pairs = {}
        for request in protocol["requests"]:
            value = request["inputs"]["local_model"]
            self.assertEqual(value, request["inputs"]["lexical_retrieval"])
            self.assertEqual(set(value), {"question", "documents"})
            self.assertTrue(all(set(doc) == {"id", "text"} for doc in value["documents"]))
            key = (request["pair_id"], request["repeat_index"])
            pairs.setdefault(key, []).append(request["settings"]["seed"])
        self.assertEqual(len(pairs), 24)
        self.assertTrue(all(len(seeds) == 2 and seeds[0] == seeds[1] for seeds in pairs.values()))

    def test_dataset_override_and_frozen_annotation_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prepare(root / "plan", dataset_name="hebrew-phenomena")
            original = read(path)
            with self.assertRaisesRegex(ValueError, "overrides"):
                run(root / "missing.gguf", root / "run", protocol_path=path, dataset_name="original")
            for field, value in (("phenomenon", "spelling"), ("variant_group", "other"),
                                 ("raw_question", "changed raw wording")):
                changed = copy.deepcopy(original)
                changed["dataset"]["cases"][0][field] = value
                write(path, changed)
                with self.assertRaisesRegex(ValueError, "Frozen protocol differs"):
                    run(root / "missing.gguf", root / "run", protocol_path=path)
            self.assertFalse((root / "run").exists())

    def test_injected_success_replays_new_slices_and_rejects_rehashed_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"INJECTED TEST MARKER, NOT WEIGHTS")
            output = root / "run"
            run(model, output, lambda path, settings: lambda prompt, options: '{"answer":null,"citation":null}',
                dataset_name="hebrew-phenomena", repeats=2)
            audit = verify_bundle(output)
            self.assertEqual(audit["status"], "completed")
            self.assertFalse(audit["model_quality_evidence"])
            self.assertEqual(len(audit["quality_summary"]), 2)
            path = output / "repeat-1/answer_report.json"
            original = read(path)
            self.assertEqual(original["methods"]["local_model"]["by_phenomenon"]["inflection"]["count"], 2)
            for mutate, message in (
                (lambda r: r["methods"]["local_model"]["by_phenomenon"]["spelling"]["metrics"].update(accuracy=1), "scoring replay"),
                (lambda r: r["protocol"]["challenge_corpus"].update(slice_policy="different policy"), "challenge_corpus")):
                changed = copy.deepcopy(original)
                mutate(changed)
                write(path, changed)
                manifest = read(output / "gguf_run.json")
                manifest["repeat_artifacts"][1]["report_sha256"] = sha256_file(path)
                write(output / "gguf_run.json", manifest)
                state = read(output / "gguf_status.json")
                state["manifest_sha256"] = sha256_file(output / "gguf_run.json")
                write(output / "gguf_status.json", state)
                with self.assertRaisesRegex(ValueError, message):
                    verify_bundle(output)

    def test_missing_weights_remain_failed_without_quality_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = prepare(root / "plan", dataset_name="hebrew-phenomena", repeats=2)
            with self.assertRaisesRegex(ValueError, "existing local"):
                run(root / "missing.gguf", root / "run", protocol_path=plan)
            audit = verify_bundle(root / "run")
            self.assertEqual(audit["status"], "failed")
            self.assertFalse(audit["model_quality_evidence"])
            self.assertIsNone(audit["quality_summary"])
            self.assertEqual(audit["by_language"]["he"]["planned"], 24)
            self.assertEqual(audit["by_language"]["he"]["attempted"], 0)
            self.assertFalse((root / "run/gguf_run.json").exists())
            self.assertFalse((root / "run/answer_report.json").exists())


class PhenomenaCommandTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, "-m", "cola_lab", *args],
                              capture_output=True, text=True, timeout=30)

    def test_documented_run_and_baseline_replay_then_corruption_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command("answers", "--dataset", "hebrew-phenomena", "--output-dir", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.command("answers-verify", "--output-dir", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["model_quality_evidence"])
            path = Path(directory) / "answer_report.json"
            report = read(path)
            report["methods"]["lexical_retrieval"]["by_phenomenon"]["mixed_language"]["count"] = 0
            write(path, report)
            result = self.command("answers-verify", "--output-dir", directory)
            self.assertEqual(result.returncode, 2)
            self.assertIn("scoring replay", result.stderr)

    def test_finite_smoke_and_invalid_dataset_flags(self):
        result = self.command("phenomena-smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 Hebrew slices", result.stdout)
        self.assertIn("no model quality result", result.stdout)
        for args in (("answers", "--dataset", "unknown"),
                     ("gguf-verify", "--dataset", "hebrew-phenomena"),
                     ("phenomena-smoke", "--dataset", "original"),
                     ("gguf", "--model", "missing.gguf", "--protocol", "unused.json", "--dataset", "hebrew-phenomena")):
            result = self.command(*args)
            self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
