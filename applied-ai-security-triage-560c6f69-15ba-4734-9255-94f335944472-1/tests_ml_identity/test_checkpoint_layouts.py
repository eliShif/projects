"""Real Transformers loading for symlinks and versioned tokenizer inputs.

Reuses the existing tiny random Llama fixture and queue. These are compatibility
and integrity tests, not pretrained-model quality experiments.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import get_fast_tokenizer_file

from portfolio.local_lm import LocalLM
from portfolio.model_identity import ModelIdentityError, capture_model_identity, verify_model_identity
from portfolio.security_data import build_dataset, observations
from portfolio.security_models import PretrainedPolicy, train_pretrained
from portfolio.security_pipeline import Pipeline, audit
from tests_ml import test_security_torch as fixture
from tests_ml_identity import test_pretrained_identity as existing
from tests_security.test_model_identity import change_same_size


class CheckpointCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = tempfile.TemporaryDirectory(dir=Path.cwd())
        cls.addClassCleanup(cls.template.cleanup)
        fixture.TorchSecurityTests().model_directory(cls.template.name)
        torch.set_num_threads(2)

    # Keep the existing queue/loader assertions, without duplicating its tests.
    setUp = existing.TensorIdentityTests.setUp
    pipeline = existing.TensorIdentityTests.pipeline
    prepare = existing.TensorIdentityTests.prepare
    train = existing.TensorIdentityTests.train
    loads = existing.TensorIdentityTests.loads
    assert_failed = existing.TensorIdentityTests.assert_failed

    def linked_checkpoint(self):
        checkpoint = self.model
        payloads = checkpoint / "payloads"
        payloads.mkdir()
        for path in list(checkpoint.iterdir()):
            if path.is_file():
                path.rename(payloads / path.name)
                path.symlink_to("payloads/" + path.name)
        self.model = self.root / "model-alias"
        self.model.symlink_to(checkpoint.name, target_is_directory=True)
        return checkpoint

    def versioned_tokenizer(self):
        """Default and older versions differ from the file Transformers selects."""
        default = json.loads((self.model / "tokenizer.json").read_text())
        older = "tokenizer.4.0.0.json"
        selected = "tokenizer.4.1.0.json"
        (self.model / older).write_text(json.dumps(default))
        vocab = default["model"]["vocab"]
        vocab["investigate_auth"], vocab["investigate_access"] = vocab["investigate_access"], vocab["investigate_auth"]
        (self.model / selected).write_text(json.dumps(default))
        config = self.model / "tokenizer_config.json"
        value = json.loads(config.read_text())
        # The absent future version is ignored by the real loader.
        value["fast_tokenizer_files"] = [older, selected, "tokenizer.99.0.0.json"]
        config.write_text(json.dumps(value))
        self.assertEqual(get_fast_tokenizer_file(value["fast_tokenizer_files"]), selected)
        return selected, vocab["investigate_auth"]

    def change_tokenizer(self, name):
        path = self.model / name
        original = path.read_bytes()
        stat = path.stat()
        changed = original.replace(b"investigate_auth", b"investigate_xuth")
        self.assertNotEqual(changed, original)
        self.assertEqual(len(changed), len(original))
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
        return original


class LinkedCheckpointTests(CheckpointCase):
    def test_symlink_checkpoint_supports_standalone_apis_without_expected_map(self):
        self.linked_checkpoint()
        data = build_dataset()
        with self.loads() as (tokenizer, model):
            policy = PretrainedPolicy(self.model)
            self.assertEqual(policy.lm.model.config.hidden_size, 16)
            out = self.root / "trained"
            metadata = train_pretrained(self.model, observations(data["train"][:1]), data["train"][:1],
                                        observations(data["validation"][:1]), data["validation"][:1],
                                        out, steps=1)
            adapted = PretrainedPolicy(self.model, out / "adapters.safetensors")
            self.assertTrue(adapted.plan(observations(data["test"][:1])[0])["candidate_answer_nll"])
            self.assertEqual(metadata["steps"], 1)
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(model.call_count, 3)

    def test_symlink_queue_trains_and_loads_both_arms_after_reopen(self):
        self.linked_checkpoint()
        pipeline = self.pipeline()
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        resumed = self.pipeline()
        with self.loads() as (tokenizer, model):
            manifest = resumed.run()
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(model.call_count, 3)
        self.assertEqual(manifest["queue_metrics"]["jobs_by_status"], {"succeeded": 3})
        self.assertEqual(audit(resumed.root)["verified_rows"], 72)
        verify_model_identity(self.model, pipeline.protocol["model_files_sha256"])

    def test_retargeted_weight_and_configuration_fail_before_load_then_restore(self):
        checkpoint = self.linked_checkpoint()
        for name in ("config.json", "model.safetensors"):
            with self.subTest(name=name):
                pipeline = self.pipeline("retarget-" + name)
                self.prepare(pipeline)
                job_id = pipeline.enqueue("train")
                alternate = checkpoint / "alternate"
                alternate.mkdir(exist_ok=True)
                shutil.copy2(checkpoint / "payloads" / name, alternate / name)
                change_same_size(alternate, name)
                link = checkpoint / name
                link.unlink()
                link.symlink_to("alternate/" + name)
                with self.loads() as (tokenizer, model):
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                self.assert_failed(pipeline, job_id, "train")
                link.unlink()
                link.symlink_to("payloads/" + name)
                with self.loads() as (tokenizer, model):
                    pipeline.work_once(job_id)
                    tokenizer.assert_called_once()
                    model.assert_called_once()
                self.assertEqual(pipeline.store.get(job_id)["status"], "succeeded")

    def test_directory_retarget_and_external_file_retarget_reject_before_load(self):
        checkpoint = self.linked_checkpoint()
        for kind in ("directory", "external_file"):
            with self.subTest(kind=kind):
                pipeline = self.pipeline(kind)
                self.prepare(pipeline)
                job_id = pipeline.enqueue("train")
                if kind == "directory":
                    other = self.root / "other"
                    shutil.copytree(self.template.name, other)
                    change_same_size(other, "config.json")
                    self.model.unlink()
                    self.model.symlink_to(other.name, target_is_directory=True)
                else:
                    # The same-byte external layout is valid; a changed referenced input must fail.
                    external = self.root / "external-config.json"
                    external.write_bytes((checkpoint / "config.json").read_bytes() + b" ")
                    (checkpoint / "config.json").unlink()
                    (checkpoint / "config.json").symlink_to(external)
                with self.loads() as (tokenizer, model):
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                self.assert_failed(pipeline, job_id, "train")
                if kind == "directory":
                    self.model.unlink()
                    self.model.symlink_to(checkpoint.name, target_is_directory=True)
                else:
                    (checkpoint / "config.json").unlink()
                    (checkpoint / "config.json").symlink_to("payloads/config.json")

    def test_real_sharded_checkpoint_supports_internal_directory_alias(self):
        loaded = LocalLM(self.model)
        sharded = self.root / "sharded"
        loaded.tokenizer.save_pretrained(sharded)
        loaded.model.save_pretrained(sharded, safe_serialization=True, max_shard_size="1KB")
        payloads = sharded / "payloads"
        payloads.mkdir()
        for path in sharded.glob("*.safetensors"):
            path.rename(payloads / path.name)
        (sharded / "shards").symlink_to("payloads", target_is_directory=True)
        index = sharded / "model.safetensors.index.json"
        value = json.loads(index.read_text())
        value["weight_map"] = {name: "shards/" + shard for name, shard in value["weight_map"].items()}
        index.write_text(json.dumps(value))
        expected = capture_model_identity(sharded)
        with self.loads() as (tokenizer, model):
            PretrainedPolicy(sharded, expected_model_files=expected)
            tokenizer.assert_called_once()
            model.assert_called_once()


class VersionedTokenizerLoadingTests(CheckpointCase):
    def test_actual_selection_and_unchanged_queued_training_and_evaluation(self):
        name, auth_id = self.versioned_tokenizer()
        actual = AutoTokenizer.from_pretrained(str(self.model), local_files_only=True, trust_remote_code=False)
        self.assertEqual(actual.convert_tokens_to_ids("investigate_auth"), auth_id)
        default = json.loads((self.model / "tokenizer.json").read_text())["model"]["vocab"]
        self.assertNotEqual(auth_id, default["investigate_auth"])
        pipeline = self.pipeline()
        self.assertIn(name, pipeline.protocol["model_files_sha256"])
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        with self.loads() as (tokenizer, model):
            self.pipeline().run()
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(model.call_count, 3)
        self.assertEqual(audit(pipeline.root)["verified_rows"], 72)

    def test_selected_tokenizer_mutation_rejected_at_training_and_each_evaluation_arm(self):
        for boundary in ("train", "base", "adapted"):
            with self.subTest(boundary=boundary):
                name, auth_id = self.versioned_tokenizer()
                pipeline = self.pipeline(boundary)
                (self.prepare if boundary == "train" else self.train)(pipeline)
                stage = "train" if boundary == "train" else "evaluate"
                job_id = pipeline.enqueue(stage)
                frozen = (pipeline.root / "protocol.json").read_bytes()
                original = (self.model / name).read_bytes()

                def load_then_change(*args, **kwargs):
                    policy = PretrainedPolicy(*args, **kwargs)
                    self.assertEqual(policy.lm.tokenizer.convert_tokens_to_ids("investigate_auth"), auth_id)
                    self.change_tokenizer(name)
                    return policy

                if boundary != "adapted":
                    self.change_tokenizer(name)
                with self.loads() as (tokenizer, model):
                    with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
                        with self.assertRaises(ModelIdentityError):
                            pipeline.work_once(job_id)
                    self.assertEqual(tokenizer.call_count, int(boundary == "adapted"))
                    self.assertEqual(model.call_count, int(boundary == "adapted"))
                self.assert_failed(pipeline, job_id, stage)
                self.assertEqual((pipeline.root / "protocol.json").read_bytes(), frozen)
                # Demonstrate that the mutated file is valid and actually changes tokenization.
                mutated = AutoTokenizer.from_pretrained(str(self.model), local_files_only=True, trust_remote_code=False)
                self.assertNotEqual(mutated.convert_tokens_to_ids("investigate_auth"), auth_id)
                (self.model / name).write_bytes(original)
                with self.loads() as (tokenizer, model):
                    pipeline.work_once(job_id)
                    self.assertEqual(model.call_count, 1 if stage == "train" else 2)
                self.assertEqual(pipeline.store.get(job_id)["status"], "succeeded")

    def test_selected_tokenizer_retry_resume_and_expired_lease_keep_original_map(self):
        for stage in ("train", "evaluate"):
            with self.subTest(stage=stage):
                name, _ = self.versioned_tokenizer()
                now = [100.0]
                pipeline = self.pipeline("retry-" + stage, clock=lambda: now[0])
                (self.prepare if stage == "train" else self.train)(pipeline)
                job_id = pipeline.enqueue(stage)
                resumed = self.pipeline("retry-" + stage, clock=lambda: now[0])
                pipeline.store.claim(job_id=job_id, lease_seconds=1)
                frozen = (pipeline.root / "protocol.json").read_bytes()
                original = self.change_tokenizer(name)
                now[0] += 2
                with self.loads() as (tokenizer, model):
                    for worker in (pipeline, resumed):
                        with self.assertRaises(ModelIdentityError):
                            worker.work_once(job_id)
                    with self.assertRaises(ModelIdentityError):
                        self.pipeline("retry-" + stage)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                self.assert_failed(pipeline, job_id, stage, attempts=3)
                self.assertEqual(pipeline.store.get(job_id)["status"], "failed")
                self.assertEqual((pipeline.root / "protocol.json").read_bytes(), frozen)
                (self.model / name).write_bytes(original)

    def test_versioned_file_change_inside_tokenizer_call_stops_model_load(self):
        name, auth_id = self.versioned_tokenizer()
        expected = capture_model_identity(self.model)
        real_load = AutoTokenizer.from_pretrained

        def load_then_change(*args, **kwargs):
            tokenizer = real_load(*args, **kwargs)
            self.assertEqual(tokenizer.convert_tokens_to_ids("investigate_auth"), auth_id)
            self.change_tokenizer(name)
            return tokenizer

        with patch.object(AutoTokenizer, "from_pretrained", side_effect=load_then_change) as tokenizer:
            with patch.object(AutoModelForCausalLM, "from_pretrained") as model:
                with self.assertRaises(ModelIdentityError):
                    LocalLM(self.model, expected_model_files=expected)
                tokenizer.assert_called_once()
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
