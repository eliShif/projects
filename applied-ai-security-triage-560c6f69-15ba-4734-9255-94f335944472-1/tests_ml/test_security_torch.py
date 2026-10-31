"""Optional real tensor integration tests; requires the ML dependencies.

Weights are locally initialized tiny Llama tensors, not pretrained-model results.
"""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from portfolio.security_data import build_dataset, observations
from portfolio.security_models import PretrainedPolicy, attach_adapters, encode_plan, train_pretrained
from portfolio.security_triage import OPERATIONS


class TorchSecurityTests(unittest.TestCase):
    def model_directory(self, root):
        vocabulary = {s: i for i, s in enumerate(["[UNK]", "[BOS]", "[EOS]", *OPERATIONS, "operation", "alert_id"])}
        tokenizer_impl = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer_impl.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        tokenizer.save_pretrained(root)
        torch.manual_seed(5)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=4096, bos_token_id=1, eos_token_id=2))
        model.save_pretrained(root, safe_serialization=True)

    def test_actual_tensors_mask_gradients_freeze_and_reload(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as trained:
            self.model_directory(root)
            policy = PretrainedPolicy(root)
            observation = observations(build_dataset()["train"])[0]
            encoded = encode_plan(policy.lm.tokenizer, observation, "investigate_auth")
            mask = encoded["labels"] == -100
            self.assertTrue(mask.any())
            self.assertTrue((~mask).any())
            frozen = {n: p.detach().clone() for n, p in policy.lm.model.named_parameters()}
            attach_adapters(policy.lm.model)
            optimizer = torch.optim.AdamW([p for p in policy.lm.model.parameters() if p.requires_grad], lr=.01)
            policy.lm.model(**encoded).loss.backward()
            optimizer.step()
            for name, parameter in policy.lm.model.named_parameters():
                if not parameter.requires_grad:
                    original = name.replace(".base.", ".")
                    torch.testing.assert_close(parameter, frozen[original])
                    self.assertIsNone(parameter.grad)
            data = build_dataset()
            metadata = train_pretrained(root, observations(data["train"][:2]), data["train"][:2],
                                       observations(data["validation"][:2]), data["validation"][:2], trained, steps=2)
            self.assertEqual(metadata["steps"], 2)
            state = load_file(str(Path(trained)/"adapters.safetensors"))
            self.assertTrue(state)
            self.assertTrue(all(name.endswith((".a", ".b")) for name in state))
            restored = PretrainedPolicy(root, Path(trained)/"adapters.safetensors")
            response = restored.plan(observation)
            self.assertIn(json.loads(response["text"])["operation"], OPERATIONS)
            self.assertEqual(set(response["candidate_answer_nll"]), set(OPERATIONS))

    def test_token_budget_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            self.model_directory(root)
            policy = PretrainedPolicy(root)
            observation = observations(build_dataset()["train"])[0]
            observation["alert"]["note"] = "overflow " * 5000
            with self.assertRaisesRegex(ValueError, "budget"):
                encode_plan(policy.lm.tokenizer, observation, "abstain")
