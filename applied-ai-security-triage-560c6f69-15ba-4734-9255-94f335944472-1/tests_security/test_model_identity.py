"""Offline identity/queue regressions. Loader doubles are explicitly boundaries,
not evidence of tensor training. Real checkpoint tests are in tests_ml_identity.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from portfolio.local_lm import LocalLM
from portfolio.model_identity import (ModelIdentityError, capture_model_identity,
                                      model_files, verify_model_identity)
from portfolio.security_models import PretrainedPolicy
from portfolio.security_pipeline import Pipeline, atomic_json, verified_path


def identity_fixture(root):
    """One valid safetensors scalar for hashing, not a causal-LM checkpoint."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"llama","hidden_size":16}')
    header = json.dumps({"fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    (root / "tokenizer_config.json").write_text('{"model_max_length":4096}')


def change_same_size(root, name):
    path = Path(root) / name
    stat = path.stat()
    original = path.read_bytes()
    if name == "config.json":
        changed = original.replace(b'16', b'17')
    else:
        changed = original[:-1] + bytes([original[-1] ^ 1])
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return original


@contextmanager
def transformer_boundaries():
    """Import-independent spies around the actual LocalLM loading code."""
    tokenizer = Mock()
    model = Mock()
    torch = SimpleNamespace(set_num_threads=Mock(), float32="float32")
    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
                                   AutoModelForCausalLM=SimpleNamespace(from_pretrained=model))
    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
        yield tokenizer, model


class IdentityInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        identity_fixture(self.root)

    def test_unchanged_inputs_and_unrelated_notes(self):
        expected = capture_model_identity(self.root)
        self.assertEqual(len(expected), 3)
        (self.root / "notes.txt").write_text("Unrelated experiment notes")
        (self.root / "notes.json").write_text("Unrelated non-model JSON")
        verify_model_identity(self.root, expected)
        self.assertEqual(capture_model_identity(self.root), expected)

    def test_same_size_same_mtime_changes_reject_weights_and_config(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                expected = capture_model_identity(self.root)
                before = (self.root / name).stat()
                original = change_same_size(self.root, name)
                after = (self.root / name).stat()
                self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))
                with self.assertRaisesRegex(ModelIdentityError, "SHA-256"):
                    verify_model_identity(self.root, expected)
                (self.root / name).write_bytes(original)
                verify_model_identity(self.root, expected)

    def test_missing_inputs_reject(self):
        for name in ("model.safetensors", "config.json", "tokenizer_config.json"):
            with self.subTest(name=name):
                expected = capture_model_identity(self.root)
                original = (self.root / name).read_bytes()
                (self.root / name).unlink()
                with self.assertRaises(ModelIdentityError):
                    verify_model_identity(self.root, expected)
                (self.root / name).write_bytes(original)

    def test_added_relevant_files_reject(self):
        for name in ("extra.safetensors", "generation_config.json", "added_tokens.json",
                     "vocab.json", "chat_template.jinja", "chat_templates/default.jinja"):
            with self.subTest(name=name):
                expected = capture_model_identity(self.root)
                path = self.root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("{}")
                with self.assertRaisesRegex(ModelIdentityError, "file set"):
                    verify_model_identity(self.root, expected)
                path.unlink()

    def test_shards_and_index_bytes_are_bound_by_relative_path(self):
        (self.root / "shards").mkdir()
        shard = self.root / "shards/part-01.safetensors"
        (self.root / "model.safetensors").rename(shard)
        index = self.root / "model.safetensors.index.json"
        index.write_text('{"weight_map":{"fixture":"shards/part-01.safetensors"},"metadata":{"total_size":4}}')
        expected = capture_model_identity(self.root)
        self.assertIn("shards/part-01.safetensors", expected)
        verify_model_identity(self.root, expected)
        for path in (shard, index):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ModelIdentityError, "SHA-256"):
                    verify_model_identity(self.root, expected)
                path.write_bytes(original)
        shard.unlink()
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.root, expected)

    def test_invalid_or_escaping_shard_indexes_reject_before_loading(self):
        index = self.root / "model.safetensors.index.json"
        values = ["{", '{}', '{"weight_map":{}}', '{"weight_map":[],"weight_map":{}}']
        for ref in ("../escape.safetensors", "/outside.safetensors", "dir/../model.safetensors",
                    "missing.safetensors", "weights.bin", "a\\b.safetensors", 1):
            values.append(json.dumps({"weight_map": {"tensor": ref}}))
        for value in values:
            with self.subTest(index=value):
                index.write_text(value)
                with self.assertRaises(ModelIdentityError):
                    model_files(self.root)

    def test_symlink_and_implicit_base_redirect_reject(self):
        expected = capture_model_identity(self.root)
        (self.root / "other.json").write_text("{}")
        (self.root / "config.json").unlink()
        (self.root / "config.json").symlink_to("other.json")
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.root, expected)
        (self.root / "config.json").unlink()
        identity_fixture(self.root)
        (self.root / "adapter_config.json").write_text("{}")
        with self.assertRaises(ModelIdentityError):
            capture_model_identity(self.root)

    def test_empty_or_malformed_expected_identity_never_recaptures(self):
        expected = capture_model_identity(self.root)
        for invalid in ({}, None, {"config.json": "bad"}, {**expected, "config.json": "A" * 64}):
            with self.subTest(expected=invalid):
                with self.assertRaisesRegex(ModelIdentityError, "frozen"):
                    verify_model_identity(self.root, invalid)


class LoaderBoundaryTests(unittest.TestCase):
    def test_valid_inputs_reach_both_real_locallm_from_pretrained_calls(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root, transformer_boundaries() as (tokenizer, model):
            identity_fixture(root)
            expected = capture_model_identity(root)
            LocalLM(root, expected_model_files=expected)
            tokenizer.assert_called_once_with(str(root), local_files_only=True, trust_remote_code=False)
            model.assert_called_once_with(str(root), local_files_only=True, trust_remote_code=False,
                                          use_safetensors=True, torch_dtype="float32")

    def test_changed_inputs_stop_both_from_pretrained_calls(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                identity_fixture(root)
                expected = capture_model_identity(root)
                change_same_size(root, name)
                with transformer_boundaries() as (tokenizer, model):
                    with self.assertRaises(ModelIdentityError):
                        LocalLM(root, expected_model_files=expected)
                    tokenizer.assert_not_called()
                    model.assert_not_called()

    def test_change_during_tokenizer_loading_stops_model_load(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                identity_fixture(root)
                expected = capture_model_identity(root)
                with transformer_boundaries() as (tokenizer, model):
                    tokenizer.side_effect = lambda *a, **kw: change_same_size(root, name)
                    with self.assertRaises(ModelIdentityError):
                        LocalLM(root, expected_model_files=expected)
                    tokenizer.assert_called_once()
                    model.assert_not_called()


class QueuedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        identity_fixture(self.model)

    def pipeline(self, suffix, clock=None):
        return Pipeline(self.root / suffix, "pretrained", 1, self.model, clock=clock)

    def prepare(self, pipeline):
        pipeline.work_once(pipeline.enqueue("prepare"))

    def training_fixture(self, model_path, train_obs, train_cases, valid_obs, valid_cases, out, **options):
        """Prerequisite receipt only; not a tensor result or model-quality run."""
        verify_model_identity(model_path, options["expected_model_files"])
        atomic_json(Path(out) / "training.json", {"backend": "test_fixture", "steps": 1, "selected_step": 0})
        (Path(out) / "adapters.safetensors").write_bytes(b"adapter-receipt-test-fixture")

    def prepared_and_trained(self, suffix):
        pipeline = self.pipeline(suffix)
        self.prepare(pipeline)
        with patch("portfolio.security_pipeline.train_pretrained", side_effect=self.training_fixture):
            pipeline.work_once(pipeline.enqueue("train"))
        return pipeline

    def assert_failed_attempt(self, pipeline, job_id, stage, count=1):
        job = pipeline.store.get(job_id)
        self.assertNotEqual(job["status"], "succeeded")
        self.assertIsNone(job["result"])
        self.assertEqual(job["attempts"], count)
        self.assertEqual(job["error"], "stage_error:ModelIdentityError")
        attempts = list(pipeline.root.glob("attempts/" + job_id + "-*"))
        self.assertEqual(len(attempts), count)
        for attempt in attempts:
            self.assertEqual({p.name for p in attempt.iterdir()}, {"failure.json"})
            self.assertEqual(json.loads((attempt / "failure.json").read_text()),
                             {"error_type": "ModelIdentityError", "stage": stage})
        self.assertFalse((pipeline.root / "manifest.json").exists())

    def test_queued_training_rejects_weight_and_config_mutations_before_loader(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.pipeline("train-" + name)
                self.prepare(pipeline)
                job_id = pipeline.enqueue("train")
                frozen = copy.deepcopy(pipeline.protocol)
                original = change_same_size(self.model, name)
                with patch("portfolio.local_lm.LocalLM") as loader:
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    loader.assert_not_called()
                self.assert_failed_attempt(pipeline, job_id, "train")
                self.assertEqual(pipeline.protocol, frozen)
                self.assertEqual(json.loads((pipeline.root / "protocol.json").read_text()), frozen)
                (self.model / name).write_bytes(original)

    def test_changes_between_training_and_base_evaluation_stop_loader(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.prepared_and_trained("base-" + name)
                job_id = pipeline.enqueue("evaluate")
                original = change_same_size(self.model, name)
                with patch("portfolio.local_lm.LocalLM") as loader:
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    loader.assert_not_called()
                self.assert_failed_attempt(pipeline, job_id, "evaluate")
                (self.model / name).write_bytes(original)

    def test_changes_between_evaluation_arms_stop_second_loader(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.prepared_and_trained("adapted-" + name)
                job_id = pipeline.enqueue("evaluate")
                original = (self.model / name).read_bytes()
                def first_load(*args, **kwargs):
                    change_same_size(self.model, name)
                    return Mock()
                with patch("portfolio.local_lm.LocalLM", side_effect=first_load) as loader:
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    loader.assert_called_once()
                self.assert_failed_attempt(pipeline, job_id, "evaluate")
                (self.model / name).write_bytes(original)

    def test_retry_and_resume_never_replace_original_identity(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.pipeline("retry-" + name)
                self.prepare(pipeline)
                job_id = pipeline.enqueue("train")
                frozen_bytes = (pipeline.root / "protocol.json").read_bytes()
                # Reopen the queue before mutation, then retain both instances.
                resumed = Pipeline(pipeline.root, "pretrained", 1, self.model)
                original = change_same_size(self.model, name)
                with patch("portfolio.local_lm.LocalLM") as loader:
                    for worker, count in ((pipeline, 1), (resumed, 2), (pipeline, 3)):
                        with self.assertRaises(ModelIdentityError):
                            worker.work_once(job_id)
                        self.assert_failed_attempt(worker, job_id, "train", count)
                    loader.assert_not_called()
                self.assertEqual(pipeline.store.get(job_id)["status"], "failed")
                with self.assertRaises(ModelIdentityError):
                    Pipeline(pipeline.root, "pretrained", 1, self.model)
                self.assertEqual((pipeline.root / "protocol.json").read_bytes(), frozen_bytes)
                (self.model / name).write_bytes(original)

    def test_expired_lease_rechecks_weights_and_config(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                now = [100.0]
                pipeline = self.pipeline("lease-" + name, clock=lambda: now[0])
                self.prepare(pipeline)
                job_id = pipeline.enqueue("train")
                pipeline.store.claim(job_id=job_id, lease_seconds=1)
                original = change_same_size(self.model, name)
                now[0] += 2
                with patch("portfolio.local_lm.LocalLM") as loader:
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    loader.assert_not_called()
                self.assertEqual(pipeline.store.get(job_id)["attempts"], 2)
                self.assertIsNone(pipeline.store.get(job_id)["result"])
                (self.model / name).write_bytes(original)

    def test_corrupt_adapter_receipt_still_stops_adapted_load(self):
        pipeline = self.prepared_and_trained("adapter")
        receipt = pipeline.predecessor("train")
        adapter = verified_path(pipeline.root, receipt, "adapters.safetensors")
        job_id = pipeline.enqueue("evaluate")
        def first_load(*args, **kwargs):
            adapter.write_bytes(b"corrupt-adapter-test-fixture")
            return Mock()
        with patch("portfolio.local_lm.LocalLM", side_effect=first_load) as loader:
            with self.assertRaisesRegex(ValueError, "checksum"):
                pipeline.work_once(job_id)
            loader.assert_called_once()
        self.assertIsNone(pipeline.store.get(job_id)["result"])


if __name__ == "__main__":
    unittest.main()
