"""Actual tiny-model compatibility and effective-input identity checks.

PEFT is an optional validation dependency, not a requirement for core loading.
No pretrained quality result is implied by random-fixture adapter integration.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_peft_available
from portfolio.local_lm import LocalLM
from portfolio.model_identity import ModelIdentityError
from portfolio.security_models import PretrainedPolicy, capture_model_identity, verify_model_identity
from tests_ml_identity.test_checkpoint_layouts import CheckpointCase


class TokenizerReferences(CheckpointCase):
    def test_relative_checkpoint_override_capture_and_actual_loading_use_same_keys(self):
        from portfolio.model_identity import capture_model_identity as strict_capture
        self.override()
        relative = self.model.relative_to(Path.cwd())
        expected = strict_capture(relative)
        self.assertIn("custom-tokenizer.json", expected)
        self.assertEqual(expected, capture_model_identity(relative))
        self.assertEqual(LocalLM(relative, expected_model_files=expected).model.config.hidden_size, 16)
        target = self.model / "custom-tokenizer.json"
        original = self.change_tokenizer(target.name)
        with self.loads() as (tokenizer, model):
            with self.assertRaises(ModelIdentityError):
                LocalLM(relative, expected_model_files=expected)
            tokenizer.assert_not_called()
            model.assert_not_called()
        target.write_bytes(original)
        self.assertEqual(LocalLM(relative, expected_model_files=expected).model.config.hidden_size, 16)

    def test_dangling_unused_versions_remain_absent_while_real_selected_inputs_load(self):
        selected, auth_id = self.versioned_tokenizer()
        (self.model / "tokenizer.99.0.0.json").symlink_to("missing-future-tokenizer.json")
        config = json.loads((self.model / "config.json").read_text())
        (self.model / "config.4.0.0.json").write_text(json.dumps(config))
        config["configuration_files"] = ["config.4.0.0.json", "config.99.0.0.json"]
        (self.model / "config.json").write_text(json.dumps(config))
        (self.model / "config.99.0.0.json").symlink_to("missing-future-config.json")
        (self.model / "unused.safetensors").symlink_to("missing-unused-weights")
        expected = capture_model_identity(self.model)
        self.assertIn(selected, expected)
        self.assertIn("config.4.0.0.json", expected)
        for name in ("tokenizer.99.0.0.json", "config.99.0.0.json", "unused.safetensors"):
            self.assertNotIn(name, expected)
        actual = LocalLM(self.model, expected_model_files=expected)
        self.assertEqual(actual.tokenizer.convert_tokens_to_ids("investigate_auth"), auth_id)
        pipeline = self.pipeline("dangling-unselected")
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        with self.loads() as (_, model):
            pipeline.run()
            self.assertEqual(model.call_count, 3)
        self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})
        # A previously unavailable referenced input becoming available must still
        # change the frozen identity set, even if its version remains unselected.
        target = self.model / "missing-future-tokenizer.json"
        target.write_text("{}")
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.model, expected)
        target.unlink()
        verify_model_identity(self.model, expected)

    def override(self, relative=False):
        value = json.loads((self.model / "tokenizer.json").read_text())
        vocab = value["model"]["vocab"]
        vocab["investigate_auth"], vocab["investigate_access"] = vocab["investigate_access"], vocab["investigate_auth"]
        target = self.model / "custom-tokenizer.json"
        target.write_text(json.dumps(value))
        config_path = self.model / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        config["tokenizer_file"] = str(target.relative_to(Path.cwd()) if relative else target.resolve())
        config_path.write_text(json.dumps(config))
        return target, vocab["investigate_auth"]

    def test_absolute_and_cwd_relative_overrides_are_really_selected_and_bound(self):
        for relative in (False, True):
            with self.subTest(relative=relative):
                target, token = self.override(relative)
                actual = AutoTokenizer.from_pretrained(str(self.model), local_files_only=True, trust_remote_code=False)
                self.assertEqual(actual.convert_tokens_to_ids("investigate_auth"), token)
                expected = capture_model_identity(self.model)
                self.assertIn(target.name, expected)
                original = self.change_tokenizer(target.name)
                with self.assertRaises(ModelIdentityError):
                    verify_model_identity(self.model, expected)
                target.write_bytes(original)
                verify_model_identity(self.model, expected)

    def test_changed_override_blocks_each_queued_loading_boundary_and_retry_restores(self):
        for relative in (False, True):
            target, _ = self.override(relative)
            for boundary in ("train", "base", "adapted"):
                with self.subTest(relative=relative, boundary=boundary):
                    pipeline = self.pipeline("override-" + str(relative) + "-" + boundary)
                    (self.prepare if boundary == "train" else self.train)(pipeline)
                    stage = "train" if boundary == "train" else "evaluate"
                    job = pipeline.enqueue(stage)
                    original = target.read_bytes()
                    def load_then_change(*args, **kwargs):
                        policy = PretrainedPolicy(*args, **kwargs)
                        self.change_tokenizer(target.name)
                        return policy
                    if boundary != "adapted":
                        self.change_tokenizer(target.name)
                    with self.loads() as (tokenizer, model):
                        with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
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

    def test_unused_sidecar_retains_actual_non_peft_loading_and_is_hashed(self):
        (self.model / "adapter_config.json").write_text("{unused invalid metadata")
        # Select the real Transformers branch used before PEFT is installed;
        # loading, fitting and evaluation still use actual tensors and loaders.
        with ExitStack() as stack:
            stack.enter_context(patch("portfolio.model_identity.importlib.util.find_spec", return_value=None))
            for name in ("transformers.models.auto.auto_factory.is_peft_available",
                         "transformers.modeling_utils.is_peft_available"):
                stack.enter_context(patch(name, return_value=False))
            self.assertEqual(LocalLM(self.model).model.config.hidden_size, 16)
            pipeline = self.pipeline("unused-sidecar")
            self.assertIn("adapter_config.json", pipeline.protocol["model_files_sha256"])
            for stage in ("prepare", "train", "evaluate"):
                pipeline.enqueue(stage)
            with self.loads() as (tokenizer, model):
                pipeline.run()
                self.assertEqual(tokenizer.call_count, 3)
                self.assertEqual(model.call_count, 3)
            self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})


@unittest.skipUnless(is_peft_available(), "Install requirements-identity-validation.txt for optional real PEFT checks")
class AdapterReferences(CheckpointCase):
    def test_relative_dot_peft_base_has_absolute_distinct_keys_and_real_queue_loading(self):
        base = self.adapter()
        config = self.model / "adapter_config.json"
        value = json.loads(config.read_text())
        value["base_model_name_or_path"] = "."
        config.write_text(json.dumps(value))
        caller = self.model / "config.json"
        caller.write_bytes(caller.read_bytes() + b" ")
        cwd = Path.cwd()
        try:
            os.chdir(base)
            expected = capture_model_identity(self.model)
            external = str(base.resolve() / "config.json")
            self.assertIn(external, expected)
            self.assertNotEqual(expected[external], expected["config.json"])
            self.assertTrue(LocalLM(self.model, expected_model_files=expected).model.peft_config)
            pipeline = self.pipeline("relative-peft-base")
            for stage in ("prepare", "train", "evaluate"):
                pipeline.enqueue(stage)
            with self.loads() as (_, model):
                pipeline.run()
                self.assertEqual(model.call_count, 3)
            self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})
            weights = base / "model.safetensors"
            original = weights.read_bytes()
            weights.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.loads() as (tokenizer, model):
                with self.assertRaises(ModelIdentityError):
                    PretrainedPolicy(self.model, expected_model_files=expected)
                tokenizer.assert_not_called()
                model.assert_not_called()
            weights.write_bytes(original)
            self.assertTrue(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.peft_config)
        finally:
            os.chdir(cwd)

    def test_unused_malformed_caller_config_keeps_real_redirect_and_all_queue_guards(self):
        self.adapter()
        config = self.model / "config.json"
        config.write_text("{unused malformed caller configuration")
        self.assertTrue(LocalLM(self.model).model.peft_config)
        expected = capture_model_identity(self.model)
        self.assertIn("config.json", expected)
        self.assertTrue(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.peft_config)
        # Without an actual separate-base redirect this malformed input remains
        # invalid. The closed-bundle policy is not relaxed by the compatibility fix.
        with patch("portfolio.model_identity._effective_adapter_base", return_value=None):
            with self.assertRaises(ModelIdentityError):
                capture_model_identity(self.model)
        original = config.read_bytes()
        for boundary in ("train", "base", "adapted"):
            with self.subTest(boundary=boundary):
                pipeline = self.pipeline("unused-caller-config-" + boundary)
                (self.prepare if boundary == "train" else self.train)(pipeline)
                stage = "train" if boundary == "train" else "evaluate"
                job = pipeline.enqueue(stage)
                def change():
                    config.write_bytes(original + b" ")
                def load_then_change(*args, **kwargs):
                    policy = PretrainedPolicy(*args, **kwargs)
                    change()
                    return policy
                if boundary != "adapted":
                    change()
                with self.loads() as (tokenizer, model):
                    with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
                        with self.assertRaises(ModelIdentityError):
                            pipeline.work_once(job)
                    self.assertEqual(tokenizer.call_count, int(boundary == "adapted"))
                    self.assertEqual(model.call_count, int(boundary == "adapted"))
                self.assert_failed(pipeline, job, stage)
                config.write_bytes(original)
                with self.loads() as (_, model):
                    pipeline.work_once(job)
                    self.assertEqual(model.call_count, 1 if stage == "train" else 2)
                self.assertEqual(pipeline.store.get(job)["status"], "succeeded")

    def nested_self_base(self, binary=False):
        from safetensors.torch import load_file
        import torch
        self.model = self.root / ("nested-bin" if binary else "nested-safe")
        shutil.copytree(self.template.name, self.model)
        original = AutoModelForCausalLM.from_pretrained(str(self.model), local_files_only=True, trust_remote_code=False)
        original.save_pretrained(self.model, safe_serialization=True, max_shard_size="1KB")
        (self.model / "model.safetensors").unlink(missing_ok=True)
        self.adapter(same_directory=True)
        nested = self.model / "nested"
        nested.mkdir()
        for file in self.model.glob("model-*.safetensors"):
            file.rename(nested / file.name)
        index = self.model / "model.safetensors.index.json"
        value = json.loads(index.read_text())
        value["weight_map"] = {key: "nested/" + name for key, name in value["weight_map"].items()}
        index.write_text(json.dumps(value))
        if binary:
            torch.save(load_file(str(self.model / "adapter_model.safetensors")), self.model / "adapter_model.bin")
            (self.model / "adapter_model.safetensors").unlink()
        return next(nested.glob("*.safetensors"))

    def adapter(self, same_directory=False):
        from peft import LoraConfig, get_peft_model
        base = self.model
        model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
        wrapped = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        target = base if same_directory else self.root / "adapter"
        wrapped.save_pretrained(target, safe_serialization=True)
        if not same_directory:
            # Preserve the actual tokenizer artifacts too (including the saved
            # Jinja chat template); both directories are full checkpoints.
            shutil.copytree(base, target, dirs_exist_ok=True)
        config = json.loads((target / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = str(base.resolve())
        (target / "adapter_config.json").write_text(json.dumps(config))
        self.model = target
        return base

    def test_real_local_peft_base_and_adapter_train_evaluate_and_resume(self):
        base = self.adapter()
        expected = capture_model_identity(self.model)
        self.assertIn(str(base / "model.safetensors"), expected)
        self.assertTrue(PretrainedPolicy(self.model).lm.model.peft_config)
        pipeline = self.pipeline("peft-local")
        for stage in ("prepare", "train", "evaluate"):
            pipeline.enqueue(stage)
        with self.loads() as (tokenizer, model):
            self.pipeline("peft-local").run()
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(model.call_count, 3)
        self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})

    def test_redirected_base_mutations_stop_train_and_both_eval_arms_before_load(self):
        base = self.adapter()
        for filename in ("config.json", "model.safetensors"):
            for boundary in ("train", "base", "adapted"):
                with self.subTest(filename=filename, boundary=boundary):
                    pipeline = self.pipeline("peft-" + filename + "-" + boundary)
                    (self.prepare if boundary == "train" else self.train)(pipeline)
                    stage = "train" if boundary == "train" else "evaluate"
                    job = pipeline.enqueue(stage)
                    target = base / filename
                    original = target.read_bytes()
                    def mutate():
                        value = original + b" " if filename.endswith("json") else original[:-1] + bytes([original[-1] ^ 1])
                        target.write_bytes(value)
                    def load_then_change(*args, **kwargs):
                        policy = PretrainedPolicy(*args, **kwargs)
                        mutate()
                        return policy
                    if boundary != "adapted":
                        mutate()
                    with self.loads() as (tokenizer, model):
                        with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
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

    def test_unused_base_tokenizer_does_not_narrow_model_only_redirect(self):
        base = self.adapter()
        (base / "tokenizer_config.json").write_text("{unused malformed base tokenizer")
        expected = capture_model_identity(self.model)
        self.assertNotIn(str(base / "tokenizer_config.json"), expected)
        self.assertTrue(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.peft_config)

    def test_adapter_sidecar_can_select_its_own_full_checkpoint_without_recursion(self):
        self.adapter(same_directory=True)
        expected = capture_model_identity(self.model)
        self.assertIn("adapter_config.json", expected)
        self.assertIn("model.safetensors", expected)
        self.assertTrue(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.peft_config)

    def test_actual_cached_hub_base_is_loaded_offline_and_its_bytes_are_bound(self):
        base = self.adapter()
        cache = self.root / "hub-cache"
        snapshot = cache / "models--controlled--tiny-cache/snapshots" / ("a" * 40)
        shutil.copytree(base, snapshot)
        refs = cache / "models--controlled--tiny-cache/refs"
        refs.mkdir()
        (refs / "main").write_text("a" * 40)
        config = json.loads((self.model / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = "controlled/tiny-cache"
        (self.model / "adapter_config.json").write_text(json.dumps(config))
        program = r'''
from pathlib import Path
from unittest.mock import patch
from transformers import AutoTokenizer,AutoModelForCausalLM
from portfolio.security_models import capture_model_identity,verify_model_identity,PretrainedPolicy
from portfolio.model_identity import ModelIdentityError
import sys
root,base=Path(sys.argv[1]),Path(sys.argv[2])
expected=capture_model_identity(root)
assert str(base/'model.safetensors') in expected
assert PretrainedPolicy(root,expected_model_files=expected).lm.model.peft_config
original=(base/'config.json').read_bytes()
(base/'config.json').write_bytes(original+b' ')
with patch.object(AutoTokenizer,'from_pretrained') as tokenizer,patch.object(AutoModelForCausalLM,'from_pretrained') as model:
    try: PretrainedPolicy(root,expected_model_files=expected)
    except ModelIdentityError: pass
    else: raise AssertionError('Changed cached base was accepted')
    tokenizer.assert_not_called();model.assert_not_called()
(base/'config.json').write_bytes(original)
verify_model_identity(root,expected)
assert PretrainedPolicy(root,expected_model_files=expected).lm.model.peft_config
'''
        env = dict(os.environ, HF_HOME=str(self.root / "isolated-hf"), HF_HUB_CACHE=str(cache),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        completed = subprocess.run([sys.executable, "-c", program, str(self.model), str(snapshot)],
                                   env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_legacy_adapter_weight_file_is_bound_with_safetensors_base(self):
        import torch
        from safetensors.torch import load_file
        base = self.adapter()
        state = load_file(str(self.model / "adapter_model.safetensors"))
        torch.save(state, self.model / "adapter_model.bin")
        (self.model / "adapter_model.safetensors").unlink()
        (self.model / "model.safetensors").unlink()
        expected = capture_model_identity(self.model)
        self.assertIn("adapter_model.bin", expected)
        self.assertIn(str(base / "model.safetensors"), expected)
        self.assertTrue(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.peft_config)
        pipeline = self.pipeline("peft-binary-adapter")
        self.prepare(pipeline)
        job = pipeline.enqueue("train")
        target = self.model / "adapter_model.bin"
        original = target.read_bytes()
        target.write_bytes(original + b" ")
        with self.loads() as (tokenizer, model):
            with self.assertRaises(ModelIdentityError):
                pipeline.work_once(job)
            tokenizer.assert_not_called()
            model.assert_not_called()
        self.assert_failed(pipeline, job, "train")
        target.write_bytes(original)
        pipeline.work_once(job)
        pipeline.work_once(pipeline.enqueue("evaluate"))
        self.assertEqual(pipeline.store.metrics()["jobs_by_status"], {"succeeded": 3})

    def test_same_directory_peft_nested_shards_are_bound_at_all_queued_boundaries(self):
        for binary in (False, True):
            target = self.nested_self_base(binary)
            self.assertTrue(LocalLM(self.model).model.peft_config)
            expected = capture_model_identity(self.model)
            self.assertIn("nested/" + target.name, expected)
            for boundary in ("train", "base", "adapted"):
                with self.subTest(binary=binary, boundary=boundary):
                    pipeline = self.pipeline("nested-" + str(binary) + "-" + boundary)
                    (self.prepare if boundary == "train" else self.train)(pipeline)
                    stage = "train" if boundary == "train" else "evaluate"
                    job = pipeline.enqueue(stage)
                    original = target.read_bytes()
                    def mutate():
                        target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                    def load_then_change(*args, **kwargs):
                        policy = PretrainedPolicy(*args, **kwargs)
                        mutate()
                        return policy
                    if boundary != "adapted":
                        mutate()
                    with self.loads() as (tokenizer, model):
                        with patch("portfolio.security_pipeline.PretrainedPolicy", side_effect=load_then_change):
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
