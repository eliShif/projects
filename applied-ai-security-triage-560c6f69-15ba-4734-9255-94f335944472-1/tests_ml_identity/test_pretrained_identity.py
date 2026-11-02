"""Real local tensor/queue integration; missing ML dependencies are errors.

Tiny Llama weights are randomly initialized test fixtures, never pretrained
research results. These tests reuse the inherited compatible checkpoint builder.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from portfolio.local_lm import LocalLM
from portfolio.model_identity import ModelIdentityError, capture_model_identity, verify_model_identity
from portfolio.security_models import PretrainedPolicy
from portfolio.security_pipeline import Pipeline, audit, verified_path
from tests_ml import test_security_torch as inherited_fixture
from tests_security.test_model_identity import change_same_size


class TensorIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = tempfile.TemporaryDirectory(dir=Path.cwd())
        cls.addClassCleanup(cls.template.cleanup)
        inherited_fixture.TorchSecurityTests().model_directory(cls.template.name)
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        shutil.copytree(self.template.name, self.model)

    def pipeline(self, suffix="run", clock=None):
        return Pipeline(self.root / suffix, "pretrained", 1, self.model, clock=clock)

    def prepare(self, pipeline):
        pipeline.work_once(pipeline.enqueue("prepare"))

    def train(self, pipeline):
        self.prepare(pipeline)
        pipeline.work_once(pipeline.enqueue("train"))

    @contextmanager
    def loads(self):
        with patch.object(AutoTokenizer, "from_pretrained", wraps=AutoTokenizer.from_pretrained) as tokenizer:
            with patch.object(AutoModelForCausalLM, "from_pretrained", wraps=AutoModelForCausalLM.from_pretrained) as model:
                yield tokenizer, model

    def assert_failed(self, pipeline, job_id, stage, attempts=1):
        job = pipeline.store.get(job_id)
        self.assertIsNone(job["result"])
        self.assertEqual(job["attempts"], attempts)
        self.assertEqual(job["error"], "stage_error:ModelIdentityError")
        self.assertNotEqual(job["status"], "succeeded")
        for path in pipeline.root.glob("attempts/" + job_id + "-*"):
            self.assertEqual({p.name for p in path.iterdir()}, {"failure.json"})
            self.assertEqual(json.loads((path / "failure.json").read_text()),
                             {"error_type": "ModelIdentityError", "stage": stage})
        self.assertFalse((pipeline.root / "manifest.json").exists())

    def test_unchanged_checkpoint_trains_loads_both_arms_and_replays(self):
        pipeline = self.pipeline()
        expected = copy.deepcopy(pipeline.protocol["model_files_sha256"])
        # Enqueue first, then reopen, preserving identities across queue delay.
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        resumed = self.pipeline()
        with self.loads() as (tokenizer, model):
            manifest = resumed.run()
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(model.call_count, 3)
            self.assertEqual(self.pipeline().run(), manifest)
            self.assertEqual(model.call_count, 3)
        self.assertEqual(audit(pipeline.root)["verified_rows"], 72)
        self.assertEqual(manifest["queue_metrics"]["jobs_by_status"], {"succeeded": 3})
        verify_model_identity(self.model, expected)
        self.assertEqual(pipeline.protocol["model_files_sha256"], expected)

    def test_queued_training_mutations_stop_actual_loaders(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.pipeline("train-" + name)
                self.prepare(pipeline)
                job_id = pipeline.enqueue("train")
                original = change_same_size(self.model, name)
                with self.loads() as (tokenizer, model):
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                self.assert_failed(pipeline, job_id, "train")
                (self.model / name).write_bytes(original)
                # Restoring the exact bytes allows a real training retry.
                with self.loads() as (tokenizer, model):
                    pipeline.work_once(job_id)
                    tokenizer.assert_called_once()
                    model.assert_called_once()
                self.assertEqual(pipeline.store.get(job_id)["status"], "succeeded")

    def test_between_training_and_base_evaluation_mutations_stop_actual_loaders(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.pipeline("base-" + name)
                self.train(pipeline)
                job_id = pipeline.enqueue("evaluate")
                original = change_same_size(self.model, name)
                with self.loads() as (tokenizer, model):
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job_id)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                self.assert_failed(pipeline, job_id, "evaluate")
                (self.model / name).write_bytes(original)

    def test_between_real_evaluation_arms_mutations_stop_adapted_load(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                pipeline = self.pipeline("adapted-" + name)
                self.train(pipeline)
                job_id = pipeline.enqueue("evaluate")
                original = (self.model / name).read_bytes()
                def load_then_change(*args, **kwargs):
                    policy = PretrainedPolicy(*args, **kwargs)
                    if len(args) == 1:
                        change_same_size(self.model, name)
                    return policy
                with self.loads() as (tokenizer, model):
                    with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
                        with self.assertRaises(ModelIdentityError):
                            pipeline.work_once(job_id)
                    tokenizer.assert_called_once()
                    model.assert_called_once()
                self.assert_failed(pipeline, job_id, "evaluate")
                (self.model / name).write_bytes(original)

    def test_retry_and_resume_of_training_and_evaluation_keep_original_hashes(self):
        for stage in ("train", "evaluate"):
            for name in ("model.safetensors", "config.json"):
                with self.subTest(stage=stage, name=name):
                    pipeline = self.pipeline("retry-" + stage + name)
                    (self.prepare if stage == "train" else self.train)(pipeline)
                    job_id = pipeline.enqueue(stage)
                    frozen = (pipeline.root / "protocol.json").read_bytes()
                    resumed = Pipeline(pipeline.root, "pretrained", 1, self.model)
                    original = change_same_size(self.model, name)
                    with self.loads() as (tokenizer, model):
                        for worker, attempts in ((pipeline, 1), (resumed, 2)):
                            with self.assertRaises(ModelIdentityError):
                                worker.work_once(job_id)
                            self.assert_failed(worker, job_id, stage, attempts)
                        with self.assertRaises(ModelIdentityError):
                            Pipeline(pipeline.root, "pretrained", 1, self.model)
                        tokenizer.assert_not_called()
                        model.assert_not_called()
                    self.assertEqual((pipeline.root / "protocol.json").read_bytes(), frozen)
                    (self.model / name).write_bytes(original)
                    with self.loads() as (tokenizer, model):
                        resumed.work_once(job_id)
                        self.assertEqual(model.call_count, 1 if stage == "train" else 2)
                    self.assertEqual(resumed.store.get(job_id)["status"], "succeeded")

    def test_real_tokenizer_then_mutation_blocks_model_loading(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                expected = capture_model_identity(self.model)
                original = (self.model / name).read_bytes()
                real_tokenizer = AutoTokenizer.from_pretrained
                def load_then_change(*args, **kwargs):
                    tokenizer = real_tokenizer(*args, **kwargs)
                    change_same_size(self.model, name)
                    return tokenizer
                with patch.object(AutoTokenizer, "from_pretrained", side_effect=load_then_change) as tokenizer:
                    with patch.object(AutoModelForCausalLM, "from_pretrained") as model:
                        with self.assertRaises(ModelIdentityError):
                            LocalLM(self.model, expected_model_files=expected)
                        tokenizer.assert_called_once()
                        model.assert_not_called()
                (self.model / name).write_bytes(original)

    def test_sharded_real_checkpoint_load_and_shard_mutation(self):
        # Convert the existing tiny fixture into a genuine Transformers shard set.
        loaded = LocalLM(self.model)
        sharded = self.root / "sharded"
        loaded.tokenizer.save_pretrained(sharded)
        loaded.model.save_pretrained(sharded, safe_serialization=True, max_shard_size="1KB")
        expected = capture_model_identity(sharded)
        self.assertIn("model.safetensors.index.json", expected)
        shards = sorted(sharded.glob("*.safetensors"))
        self.assertGreater(len(shards), 1)
        with self.loads() as (tokenizer, model):
            PretrainedPolicy(sharded, expected_model_files=expected)
            model.assert_called_once()
        for path in (shards[-1], sharded / "model.safetensors.index.json"):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.loads() as (tokenizer, model):
                with self.assertRaises(ModelIdentityError):
                    PretrainedPolicy(sharded, expected_model_files=expected)
                tokenizer.assert_not_called()
                model.assert_not_called()
            path.write_bytes(original)

    def test_adapter_receipt_corruption_rejects_without_loading(self):
        pipeline = self.pipeline()
        self.train(pipeline)
        receipt = pipeline.predecessor("train")
        path = verified_path(pipeline.root, receipt, "adapters.safetensors")
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.loads() as (tokenizer, model):
            with self.assertRaisesRegex(ValueError, "checksum"):
                pipeline.work_once(pipeline.enqueue("evaluate"))
            tokenizer.assert_not_called()
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
