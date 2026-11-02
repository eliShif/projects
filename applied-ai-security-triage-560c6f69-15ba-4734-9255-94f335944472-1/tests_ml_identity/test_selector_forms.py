"""Real loader coverage for accepted version selector containers and missing links."""
import json

from portfolio.local_lm import LocalLM
from portfolio.model_identity import ModelIdentityError
from portfolio.security_models import capture_model_identity, verify_model_identity
from tests_ml_identity.test_checkpoint_layouts import CheckpointCase


class SelectorForms(CheckpointCase):
    def selections(self, object_keys=False, unavailable_self_links=False):
        selected, token = self.versioned_tokenizer()
        path = self.model / "tokenizer_config.json"
        config = json.loads(path.read_text())
        if object_keys:
            config["fast_tokenizer_files"] = {name: None for name in config["fast_tokenizer_files"]}
        path.write_text(json.dumps(config))
        path = self.model / "config.json"
        config = json.loads(path.read_text())
        chosen = "config.4.0.0.json"
        (self.model / chosen).write_text(json.dumps({**config, "identity_selector_marker": "selected"}))
        versions = [chosen, "config.99.0.0.json"]
        config["configuration_files"] = {name: None for name in versions} if object_keys else versions
        path.write_text(json.dumps(config))
        if unavailable_self_links:
            for name in ("tokenizer.99.0.0.json", "config.99.0.0.json"):
                (self.model / name).symlink_to(name)
        return chosen, selected, token

    def test_object_key_selectors_bind_actual_consumed_inputs_and_all_queue_boundaries(self):
        config_name, tokenizer_name, token = self.selections(object_keys=True)
        loaded = LocalLM(self.model)
        self.assertEqual(loaded.model.config.identity_selector_marker, "selected")
        self.assertEqual(loaded.tokenizer.convert_tokens_to_ids("investigate_auth"), token)
        expected = capture_model_identity(self.model)
        self.assertIn(config_name, expected)
        self.assertIn(tokenizer_name, expected)
        for name in (config_name, tokenizer_name):
            target = self.model / name
            original = target.read_bytes()
            for boundary in ("train", "base", "adapted"):
                with self.subTest(artifact=name, boundary=boundary):
                    pipeline = self.pipeline("object-" + name + "-" + boundary)
                    (self.prepare if boundary == "train" else self.train)(pipeline)
                    stage = "train" if boundary == "train" else "evaluate"
                    job = pipeline.enqueue(stage)
                    from unittest.mock import patch
                    from portfolio.security_models import PretrainedPolicy
                    def changed_policy(*args, **kwargs):
                        policy = PretrainedPolicy(*args, **kwargs)
                        target.write_bytes(original + b" ")
                        return policy
                    if boundary != "adapted":
                        target.write_bytes(original + b" ")
                    with self.loads() as (tokenizer, model):
                        with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=changed_policy):
                            with self.assertRaises(ModelIdentityError):
                                pipeline.work_once(job)
                        self.assertEqual(tokenizer.call_count, int(boundary == "adapted"))
                        self.assertEqual(model.call_count, int(boundary == "adapted"))
                    self.assert_failed(pipeline, job, stage)
                    target.write_bytes(original)
                    with self.loads() as (_, model):
                        pipeline.work_once(job)
                        self.assertEqual(model.call_count, 1 if stage == "train" else 2)
                    self.assertEqual(pipeline.store.get(job)["status"], "succeeded")
        verify_model_identity(self.model, expected)

    def test_unavailable_unselected_self_links_do_not_block_actual_queue_loading(self):
        config_name, tokenizer_name, token = self.selections(unavailable_self_links=True)
        loaded = LocalLM(self.model)
        self.assertEqual(loaded.model.config.identity_selector_marker, "selected")
        self.assertEqual(loaded.tokenizer.convert_tokens_to_ids("investigate_auth"), token)
        expected = capture_model_identity(self.model)
        self.assertIn(config_name, expected)
        self.assertIn(tokenizer_name, expected)
        for name in ("tokenizer.99.0.0.json", "config.99.0.0.json"):
            self.assertNotIn(name, expected)
        pipeline = self.pipeline("unselected-self-links")
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        with self.loads() as (_, model):
            pipeline.run()
            self.assertEqual(model.call_count, 3)
        self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})
        for name in ("tokenizer.99.0.0.json", "config.99.0.0.json"):
            target = self.model / name
            target.unlink()
            target.write_text("{}")
            with self.assertRaises(ModelIdentityError):
                verify_model_identity(self.model, expected)
            target.unlink()
            target.symlink_to(name)
            verify_model_identity(self.model, expected)
