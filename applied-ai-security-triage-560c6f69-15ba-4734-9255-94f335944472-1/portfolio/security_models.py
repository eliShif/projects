"""Trainable reference policy and optional local pretrained causal-LM adapter.

The reference policy is a small linear classifier, explicitly not an LLM. Its
low-rank update provides an offline executable control for lifecycle mechanics.
The pretrained backend reuses the existing PyTorch LoRA and answer-token loss.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

from .common import sha256_json, write_json
from .security_triage import OPERATIONS, SYSTEM

DIMENSIONS = 8
RANK = 2


def features(observation):
    anchor = observation["anchor"]
    kind = anchor["kind"] if anchor else "missing"
    return [1., float(kind == "login_success"), float(kind == "secret_read"),
            float(kind == "login_failure"), float(anchor is None),
            float(bool(observation["blocked_reason"])),
            min(1., len(observation["events"]) / 64), float(observation["alert"]["complete"])]


def softmax(values):
    exps = [math.exp(v - max(values)) for v in values]
    return [v / sum(exps) for v in exps]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class ReferencePolicy:
    backend = "reference_linear_policy_not_llm"

    def __init__(self, seed=2026, state=None):
        rng = random.Random(seed)
        self.state = copy.deepcopy(state) if state is not None else {
            "format": 1, "seed": seed,
            "base": [[rng.uniform(-.2, .2) for _ in range(DIMENSIONS)] for _ in OPERATIONS],
            "a": [[rng.uniform(-.2, .2) for _ in range(DIMENSIONS)] for _ in range(RANK)],
            "b": [[0.] * RANK for _ in OPERATIONS]}
        if set(self.state) != {"format", "seed", "base", "a", "b"} or self.state["format"] != 1:
            raise ValueError("Unsupported reference checkpoint")
        for key, rows, cols in (("base", 3, DIMENSIONS), ("a", RANK, DIMENSIONS), ("b", 3, RANK)):
            matrix = self.state[key]
            if not isinstance(matrix, list) or len(matrix) != rows:
                raise ValueError("Checkpoint shape mismatch")
            for row in matrix:
                if not isinstance(row, list) or len(row) != cols or any(
                        type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e4 for v in row):
                    raise ValueError("Invalid checkpoint weights")

    def logits(self, x):
        z = [dot(row, x) for row in self.state["a"]]
        return [dot(w, x) + dot(b, z) for w, b in zip(self.state["base"], self.state["b"])]

    def gradients(self, x, target):
        probabilities = softmax(self.logits(x))
        error = [p - float(i == target) for i, p in enumerate(probabilities)]
        z = [dot(row, x) for row in self.state["a"]]
        grad_b = [[e * v for v in z] for e in error]
        grad_a = [[sum(error[c] * self.state["b"][c][r] for c in range(3)) * v for v in x] for r in range(RANK)]
        return grad_a, grad_b

    def update(self, x, target, lr=.4):
        ga, gb = self.gradients(x, target)
        norm = math.sqrt(sum(v*v for matrix in (ga, gb) for row in matrix for v in row))
        scale = min(1., 1. / max(norm, 1e-12))
        for key, gradient in (("a", ga), ("b", gb)):
            for row, grad in zip(self.state[key], gradient):
                for i, value in enumerate(grad):
                    row[i] -= lr * scale * value

    def loss(self, observations, targets):
        return sum(-math.log(max(1e-15, softmax(self.logits(features(o)))[t]))
                   for o, t in zip(observations, targets)) / len(targets)

    def plan(self, observation):
        values = self.logits(features(observation))
        index = max(range(3), key=lambda i: values[i])
        return {"text": json.dumps({"operation": OPERATIONS[index], "alert_id": observation["alert"]["id"]}),
                "logits": dict(zip(OPERATIONS, values)), "backend": self.backend}


def targets(cases):
    return [OPERATIONS.index(c["expected"]["operation"]) for c in cases]


def train_reference(train_obs, train_cases, valid_obs, valid_cases, out, steps=96, heartbeat=lambda: None):
    if type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError("Training step budget must be 1..512")
    model = ReferencePolicy()
    frozen_hash = sha256_json(model.state["base"])
    train_y, valid_y = targets(train_cases), targets(valid_cases)
    history = [{"step": 0, "validation_nll": model.loss(valid_obs, valid_y)}]
    best_loss, best_step, best = history[0]["validation_nll"], 0, copy.deepcopy(model.state)
    order = list(range(len(train_y)))
    random.Random(2026).shuffle(order)
    for step in range(1, steps + 1):
        index = order[(step-1) % len(order)]
        model.update(features(train_obs[index]), train_y[index])
        if step % 12 == 0 or step == steps:
            heartbeat()
            loss = model.loss(valid_obs, valid_y)
            history.append({"step": step, "validation_nll": loss})
            if loss < best_loss:
                best_loss, best_step, best = loss, step, copy.deepcopy(model.state)
    if sha256_json(model.state["base"]) != frozen_hash:
        raise RuntimeError("Frozen base changed")
    write_json(Path(out)/"adapter.json", best)
    metadata = {"backend": model.backend, "selection": "minimum validation NLL, including step zero",
                "rank": RANK, "trainable_parameters": RANK * (DIMENSIONS + len(OPERATIONS)),
                "seed": 2026, "steps": steps, "selected_step": best_step, "lr": .4,
                "frozen_base_sha256": frozen_hash, "validation_history": history,
                "base_origin": "seeded random linear weights; no pretrained language model"}
    write_json(Path(out)/"training.json", metadata)
    return metadata


def model_files(model_path):
    """Hash only explicit model artifacts; do not scan account/config directories."""
    path = Path(model_path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError("Local pretrained model config.json required")
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("Local pretrained safetensors weights required")
    files += [p for p in sorted(path.glob("*.json")) if p.name in {
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "model.safetensors.index.json"}]
    files += [path / name for name in ("tokenizer.model", "vocab.json", "merges.txt") if (path / name).is_file()]
    return files


def prompt_messages(observation):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(observation, sort_keys=True)}]


def encode_plan(tokenizer, observation, operation):
    import torch
    prefix = tokenizer.apply_chat_template(prompt_messages(observation), tokenize=False, add_generation_prompt=True)
    # Encode segments separately so a token that straddles the boundary cannot
    # accidentally turn a prompt token into a supervised answer token.
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer = json.dumps({"operation": operation, "alert_id": observation["alert"]["id"]})
    suffix_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if not prefix_ids or not suffix_ids or len(prefix_ids) + len(suffix_ids) > 4096:
        raise ValueError("Token budget exceeded or empty prompt")
    ids = torch.tensor([prefix_ids + suffix_ids])
    labels = ids.clone()
    labels[:, :len(prefix_ids)] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def attach_adapters(model):
    from .adapter_tuning import LoRALinear
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    if len(layers) < 2:
        raise ValueError("Need at least two Llama-style decoder layers")
    names = []
    for i in range(len(layers)-2, len(layers)):
        for name in ("q_proj", "v_proj"):
            attention = layers[i].self_attn
            setattr(attention, name, LoRALinear(getattr(attention, name), rank=4, alpha=8))
            names.append(f"model.layers.{i}.self_attn.{name}")
    return names


class PretrainedPolicy:
    backend = "local_pretrained_causal_lm_candidate_nll"

    def __init__(self, model_path, adapter_path=None):
        model_files(model_path)
        from .local_lm import LocalLM
        self.lm = LocalLM(model_path)
        if adapter_path:
            from safetensors.torch import load_file
            attach_adapters(self.lm.model)
            state = load_file(str(adapter_path))
            parameters = {n: p for n, p in self.lm.model.named_parameters() if p.requires_grad}
            if set(state) != set(parameters):
                raise ValueError("Adapter parameter mismatch")
            with self.lm.torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(state[name])
        self.lm.model.eval()

    def plan(self, observation):
        from .adapter_tuning import answer_loss
        scores = [answer_loss(self.lm.model, [encode_plan(self.lm.tokenizer, observation, op)]) for op in OPERATIONS]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("Nonfinite model scores")
        operation = OPERATIONS[min(range(3), key=lambda i: scores[i])]
        return {"text": json.dumps({"operation": operation, "alert_id": observation["alert"]["id"]}),
                "candidate_answer_nll": dict(zip(OPERATIONS, scores)), "backend": self.backend}


def train_pretrained(model_path, train_obs, train_cases, valid_obs, valid_cases, out, steps=96, heartbeat=lambda: None):
    if type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError("Training step budget must be 1..512")
    import torch
    from safetensors.torch import save_file
    from .adapter_tuning import answer_loss
    torch.manual_seed(2026)
    policy = PretrainedPolicy(model_path)
    lm = policy.lm
    names = attach_adapters(lm.model)
    training = [encode_plan(lm.tokenizer, o, c["expected"]["operation"]) for o, c in zip(train_obs, train_cases)]
    validation = [encode_plan(lm.tokenizer, o, c["expected"]["operation"]) for o, c in zip(valid_obs, valid_cases)]
    parameters = {n: p for n, p in lm.model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=.001)
    best_loss, best_step = answer_loss(lm.model, validation), 0
    best = {n: p.detach().clone() for n, p in parameters.items()}
    history = [{"step": 0, "validation_nll": best_loss}]
    for step in range(1, steps+1):
        heartbeat()
        lm.model.train()
        optimizer.zero_grad()
        loss = lm.model(**training[(step-1) % len(training)]).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.)
        optimizer.step()
        if step % 12 == 0 or step == steps:
            value = answer_loss(lm.model, validation)
            history.append({"step": step, "validation_nll": value})
            if value < best_loss:
                best_loss, best_step = value, step
                best = {n: p.detach().clone() for n, p in parameters.items()}
    Path(out).mkdir(parents=True, exist_ok=True)
    save_file(best, str(Path(out)/"adapters.safetensors"))
    metadata = {"backend": policy.backend, "rank": 4, "alpha": 8, "targets": names,
                "trainable_parameters": sum(p.numel() for p in parameters.values()),
                "seed": 2026, "steps": steps, "lr": .001, "selected_step": best_step,
                "selection": "minimum validation answer-token NLL, including step zero", "validation_history": history}
    write_json(Path(out)/"training.json", metadata)
    return metadata
