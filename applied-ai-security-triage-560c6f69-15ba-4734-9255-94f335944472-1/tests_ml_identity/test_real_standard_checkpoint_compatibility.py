"""Real tiny-model loading, one-step training and configuration-selection regressions."""
import json
from pathlib import Path
from unittest.mock import patch
from transformers import AutoConfig
from portfolio.model_identity import ModelIdentityError
from portfolio.security_models import PretrainedPolicy
from tests_ml_identity.test_checkpoint_layouts import CheckpointCase


class StandardCheckpointLoading(CheckpointCase):
    def test_real_hf_cache_style_external_links_and_unused_index_load(self):
        checkpoint = self.linked_checkpoint()
        blobs = self.root / "shared-blobs"
        (checkpoint / "payloads").rename(blobs)
        for link in checkpoint.iterdir():
            if link.is_symlink():
                link.unlink()
                link.symlink_to("../shared-blobs/" + link.name)
        (checkpoint / "model.safetensors.index.json").write_text("{unused invalid JSON")
        pipeline = self.pipeline("shared-cache")
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        with self.loads() as (tokenizer, model):
            pipeline.run()
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(model.call_count, 3)
        self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})

    def test_selected_configuration_is_really_loaded_and_bound_at_each_boundary(self):
        variant = self.model / "config.4.0.0.json"
        original_config = json.loads((self.model / "config.json").read_text())
        selected = dict(original_config, rms_norm_eps=0.000012)
        variant.write_text(json.dumps(selected))
        metadata = dict(original_config, configuration_files=[variant.name])
        (self.model / "config.json").write_text(json.dumps(metadata))
        actual = AutoConfig.from_pretrained(str(self.model), local_files_only=True, trust_remote_code=False)
        self.assertEqual(actual.rms_norm_eps, 0.000012)
        for boundary in ("train", "base", "adapted"):
            with self.subTest(boundary=boundary):
                pipeline = self.pipeline("configuration-" + boundary)
                self.assertIn(variant.name, pipeline.protocol["model_files_sha256"])
                (self.prepare if boundary == "train" else self.train)(pipeline)
                stage = "train" if boundary == "train" else "evaluate"
                job = pipeline.enqueue(stage)
                original = variant.read_bytes()
                changed = json.dumps(dict(selected, rms_norm_eps=0.000024)).encode()
                def load_then_change(*args, **kwargs):
                    policy = PretrainedPolicy(*args, **kwargs)
                    self.assertEqual(policy.lm.model.config.rms_norm_eps, 0.000012)
                    variant.write_bytes(changed)
                    return policy
                if boundary != "adapted":
                    variant.write_bytes(changed)
                with self.loads() as (tokenizer, model):
                    with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
                        with self.assertRaises(ModelIdentityError):
                            pipeline.work_once(job)
                    self.assertEqual(tokenizer.call_count, int(boundary == "adapted"))
                    self.assertEqual(model.call_count, int(boundary == "adapted"))
                variant.write_bytes(original)
                with self.loads() as (tokenizer, model):
                    pipeline.work_once(job)
                    self.assertEqual(model.call_count, 1 if stage == "train" else 2)
                self.assertEqual(pipeline.store.get(job)["status"], "succeeded")
