"""Executable protocol/failure checks. All generated answers here are fixtures."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cola_lab.gguf_audit import verify_bundle
from cola_lab.local_gguf import load_protocol, prepare, run, sha256_file


ABSTAIN = '{"answer":null,"citation":null}'


def abstaining_backend(path, settings):
    return lambda prompt, call_settings: ABSTAIN


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "fixture.gguf"
        self.model.write_bytes(b"INJECTED TEST MARKER, NOT WEIGHTS")
        self.output = self.root / "run"

    def rehash_report(self, repeat=0):
        """Show replay rejects false metrics even if the artifact hashes are updated."""
        manifest = read(self.output / "gguf_run.json")
        artifact = manifest["repeat_artifacts"][repeat]
        artifact["report_sha256"] = sha256_file(self.output / artifact["report_file"])
        write(self.output / "gguf_run.json", manifest)
        state = read(self.output / "gguf_status.json")
        state["manifest_sha256"] = sha256_file(self.output / "gguf_run.json")
        write(self.output / "gguf_status.json", state)

    def test_saved_protocol_exists_before_backend_and_is_used_without_overrides(self):
        path = prepare(self.root / "plan", repeats=2, base_seed=41)
        frozen = read(path)
        observed = []

        def backend(model, settings):
            self.assertEqual(read(self.output / "gguf_protocol.json"), frozen)
            self.assertEqual(read(self.output / "gguf_status.json")["status"], "running")
            self.assertEqual(settings["seed"], 41)

            def generate(prompt, call_settings):
                observed.append((prompt, call_settings))
                # One correct English answer, all other fixture answers abstain.
                if "Question: What is the shape of Ember?" in prompt:
                    return '{"answer":"round","citation":"test-en-Ember-shape"}'
                return ABSTAIN
            return generate

        run(self.model, self.output, backend, protocol_path=path)
        self.assertEqual(observed, [(r["prompt"], r["settings"]) for r in frozen["requests"]])
        audit = verify_bundle(self.output, model_path=self.model)
        self.assertEqual(audit["status"], "completed")
        self.assertFalse(audit["model_quality_evidence"])
        self.assertTrue(audit["model_file_checked"])
        self.assertEqual(len(audit["quality_summary"]), 2)
        for repeat in audit["quality_summary"]:
            en, he = (repeat["by_language"][lang] for lang in ("en", "he"))
            self.assertEqual(en["count"], 12)
            self.assertEqual(en["local_model_accuracy"], 5 / 12)
            self.assertEqual(he["local_model_accuracy"], 4 / 12)
            self.assertEqual(en["lexical_accuracy"], 11 / 12)
            self.assertAlmostEqual(he["accuracy_delta"], -7 / 12)
            self.assertEqual(en["error_counts"], {"false_abstention": 7})
            self.assertEqual(he["error_counts"], {"false_abstention": 8})
        self.assertEqual(audit["by_language"]["he"]["parsed"], 24)

    def test_protocol_rejects_drift_and_overrides_before_starting_a_run(self):
        path = prepare(self.root / "plan")
        original = read(path)
        mutations = [lambda p: p["requests"][0]["settings"].update(seed=99),
                     lambda p: p["requests"][0]["inputs"]["local_model"].update(question="changed"),
                     lambda p: p["source_sha256"].update({"answer_eval.py": "0" * 64}),
                     lambda p: p["dataset"]["cases"].pop(),
                     lambda p: p.update(repeats=0)]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            write(path, value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                run(self.model, self.output, abstaining_backend, protocol_path=path)
            self.assertFalse(self.output.exists())
        write(path, original)
        with self.assertRaisesRegex(ValueError, "overrides"):
            run(self.model, self.output, abstaining_backend, protocol_path=path, base_seed=17)
        self.assertEqual(load_protocol(path), original)

    def test_legacy_full_context_stays_a_different_evidence_comparison(self):
        run(self.model, self.output, abstaining_backend, context_mode="all-documents")
        audit = verify_bundle(self.output)
        self.assertEqual(audit["context_mode"], "all-documents")
        self.assertEqual(audit["comparison"], "different_evidence_system_comparison")
        self.assertFalse(audit["model_quality_evidence"])

    def test_missing_model_records_real_preflight_failure_without_quality(self):
        with self.assertRaisesRegex(ValueError, "existing local"):
            run(self.root / "missing.gguf", self.output, repeats=2)
        audit = verify_bundle(self.output)
        self.assertEqual(audit["backend"], "llama-cpp-python")
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["failure_phase"], "model_check")
        self.assertFalse(audit["model_quality_evidence"])
        self.assertIsNone(audit["quality_summary"])
        self.assertIsNone(audit["model_sha256"])
        self.assertEqual(audit["by_language"]["en"]["not_attempted"], 24)
        self.assertEqual(audit["by_language"]["he"]["attempted"], 0)

    def test_non_gguf_bytes_rejected_before_native_backend(self):
        with self.assertRaisesRegex(ValueError, "GGUF header"):
            run(self.model, self.output)
        self.assertEqual(verify_bundle(self.output)["failure_phase"], "model_check")

    def test_backend_initialization_failure_is_inspectable(self):
        def broken_backend(path, settings):
            raise RuntimeError("fixture backend unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            run(self.model, self.output, broken_backend)
        audit = verify_bundle(self.output)
        self.assertEqual(audit["failure_phase"], "backend_initialization")
        self.assertEqual(audit["by_language"]["en"]["attempted"], 0)
        self.assertEqual(audit["error"]["message"], "fixture backend unavailable")

    def test_malformed_hebrew_response_retains_english_calls_and_raw_failure(self):
        def backend(path, settings):
            return lambda prompt, call_settings: "תשובה שאינה JSON" if "Question: איזה" in prompt else ABSTAIN

        with self.assertRaisesRegex(ValueError, "test-he-Ember-color-q"):
            run(self.model, self.output, backend, repeats=2)
        audit = verify_bundle(self.output)
        self.assertIsNone(audit["quality_summary"])
        self.assertFalse(audit["model_quality_evidence"])
        self.assertEqual(audit["by_language"]["en"]["parsed"], 12)
        self.assertEqual(audit["by_language"]["he"]["invalid_output"], 1)
        self.assertEqual(audit["by_language"]["he"]["not_attempted"], 23)
        self.assertEqual(audit["failure_case"]["language"], "he")
        attempts = [json.loads(line) for line in (self.output / "gguf_attempts.jsonl").read_text().splitlines()]
        self.assertEqual(len(attempts), 13)
        self.assertEqual(attempts[-1]["raw_generation"], "תשובה שאינה JSON")
        self.assertFalse((self.output / "answer_report.json").exists())
        self.assertFalse((self.output / "gguf_run.json").exists())

    def test_generation_exception_and_interrupt_keep_completed_attempts(self):
        for error in (RuntimeError("context window exceeded"), KeyboardInterrupt()):
            output = self.root / type(error).__name__
            calls = []

            def backend(path, settings):
                def generate(prompt, call_settings):
                    calls.append(prompt)
                    if len(calls) == 2:
                        raise error
                    return ABSTAIN
                return generate

            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                run(self.model, output, backend)
            audit = verify_bundle(output)
            self.assertEqual(audit["by_language"]["en"]["parsed"], 1)
            self.assertEqual(audit["by_language"]["en"]["backend_error"], 1)
            self.assertEqual(audit["error"]["type"], type(error).__name__)

    def test_scoring_failure_removes_any_completed_repeat_results(self):
        from cola_lab.answer_eval import run as scorer

        def fail_second(output, predictions, **kwargs):
            if output.name == "repeat-1":
                raise RuntimeError("fixture scorer failure")
            return scorer(output, predictions, **kwargs)

        with patch("cola_lab.local_gguf.run_answers", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "scorer failure"):
                run(self.model, self.output, abstaining_backend, repeats=2)
        audit = verify_bundle(self.output)
        self.assertEqual(audit["failure_phase"], "scoring")
        self.assertEqual(audit["by_language"]["en"]["parsed"], 24)
        self.assertIsNone(audit["quality_summary"])
        self.assertFalse((self.output / "repeat-1/gguf_predictions.json").exists())

    def test_changed_weights_abort_before_scoring(self):
        def backend(path, settings):
            path.write_bytes(b"different fixture bytes")
            return lambda prompt, call_settings: ABSTAIN

        with self.assertRaisesRegex(ValueError, "Model file changed"):
            run(self.model, self.output, backend)
        audit = verify_bundle(self.output)
        self.assertEqual(audit["failure_phase"], "record_validation")
        self.assertIsNone(audit["quality_summary"])

    def test_nonempty_plan_or_run_is_preserved(self):
        protocol = prepare(self.root / "plan")
        before = protocol.read_bytes()
        with self.assertRaisesRegex(ValueError, "must be empty"):
            prepare(protocol.parent)
        self.assertEqual(protocol.read_bytes(), before)
        with self.assertRaises(ValueError):
            run(self.root / "missing.gguf", self.output)
        state = (self.output / "gguf_status.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be empty"):
            run(self.model, self.output, abstaining_backend)
        self.assertEqual((self.output / "gguf_status.json").read_bytes(), state)

    def test_verifier_rejects_artifact_drift_missing_repeat_and_wrong_weights(self):
        run(self.model, self.output, abstaining_backend, repeats=2)
        for relative in ("gguf_attempts.jsonl", "gguf_protocol.json", "gguf_run.json",
                         "repeat-1/answer_report.json", "repeat-1/gguf_predictions.json"):
            path = self.output / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "Hash mismatch"):
                verify_bundle(self.output)
            path.write_bytes(original)
        path = self.output / "repeat-1/answer_dataset.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing artifact"):
            verify_bundle(self.output)
        path.write_bytes(original)
        self.model.write_bytes(b"a different fixture")
        with self.assertRaisesRegex(ValueError, "Model file hash mismatch"):
            verify_bundle(self.output, model_path=self.model)

    def test_replay_rejects_rehashed_false_metrics_and_exposure(self):
        run(self.model, self.output, abstaining_backend)
        path = self.output / "answer_report.json"
        original = read(path)
        bad = copy.deepcopy(original)
        bad["methods"]["local_model"]["by_language"]["he"]["accuracy"] = 1
        write(path, bad)
        self.rehash_report()
        with self.assertRaisesRegex(ValueError, "scoring replay"):
            verify_bundle(self.output)
        bad = copy.deepcopy(original)
        bad["protocol"]["information_control"]["case_exposures"][0]["generation_settings"]["seed"] = 0
        write(path, bad)
        self.rehash_report()
        with self.assertRaisesRegex(ValueError, "exposure/settings drift"):
            verify_bundle(self.output)

    def test_reordered_journal_rejected_even_if_rehashed(self):
        run(self.model, self.output, abstaining_backend)
        path = self.output / "gguf_attempts.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        state = read(self.output / "gguf_status.json")
        state["attempts_sha256"] = sha256_file(path)
        write(self.output / "gguf_status.json", state)
        with self.assertRaisesRegex(ValueError, "request_index"):
            verify_bundle(self.output)

    def test_incomplete_status_cannot_be_verified_as_finished(self):
        run(self.model, self.output, abstaining_backend)
        state = read(self.output / "gguf_status.json")
        state["status"] = "running"
        write(self.output / "gguf_status.json", state)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_bundle(self.output)

    def test_manifest_cannot_redirect_verifier_outside_run(self):
        run(self.model, self.output, abstaining_backend)
        manifest = read(self.output / "gguf_run.json")
        manifest["repeat_artifacts"][0]["report_file"] = "../outside.json"
        write(self.output / "gguf_run.json", manifest)
        state = read(self.output / "gguf_status.json")
        state["manifest_sha256"] = sha256_file(self.output / "gguf_run.json")
        write(self.output / "gguf_status.json", state)
        with self.assertRaisesRegex(ValueError, "artifact path"):
            verify_bundle(self.output)

    def test_symlinked_artifact_rejected(self):
        run(self.model, self.output, abstaining_backend)
        path = self.output / "gguf_attempts.jsonl"
        outside = self.root / "copied-journal.jsonl"
        path.replace(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlinked"):
            verify_bundle(self.output)


class CommandTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, "-m", "cola_lab", *args],
                              capture_output=True, text=True, timeout=30)

    def test_documented_plan_attempt_and_verify_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.command("gguf-plan", "--output-dir", str(root / "plan"), "--repeats", "2")
            self.assertEqual(plan.returncode, 0, plan.stderr)
            attempted = self.command("gguf", "--protocol", str(root / "plan/gguf_protocol.json"),
                                     "--model", str(root / "missing.gguf"), "--output-dir", str(root / "run"))
            self.assertEqual(attempted.returncode, 2, attempted.stderr)
            checked = self.command("gguf-verify", "--output-dir", str(root / "run"))
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(checked.stdout)["status"], "failed")
            self.assertIsNone(json.loads(checked.stdout)["quality_summary"])

    def test_finite_smoke_and_rejected_flags(self):
        result = self.command("study-smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no model quality result", result.stdout)
        for args in (("answers", "--protocol", "unused.json"),
                     ("gguf-plan", "--model", "unused.gguf"),
                     ("gguf-verify", "--seed", "17"),
                     ("gguf", "--model", "missing.gguf", "--protocol", "unused.json", "--repeats", "1")):
            result = self.command(*args)
            self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
