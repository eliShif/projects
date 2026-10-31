"""Frozen, local Transformers encoders behind the existing retrieval interface.

No downloads, remote code, pickle weights, or synthetic inference fallback.
Test checkpoints can exercise this contract but are not pretrained-model evidence.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import torch

from .common import sha256_json


def positive_integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def normalize_embeddings(vectors):
    """Return finite, unit-length rows; zero vectors do not define a cosine."""
    if (not isinstance(vectors, torch.Tensor) or vectors.ndim != 2
            or 0 in vectors.shape or not vectors.is_floating_point()):
        raise ValueError("Expected a nonempty floating [batch, dimension] tensor")
    vectors = vectors.float()
    if not torch.isfinite(vectors).all():
        raise ValueError("Non-finite embeddings")
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Zero or non-finite embedding norm")
    return vectors / norms


def masked_pool(hidden, attention_mask, pooling="mean"):
    """[B, T, D] -> [B, D]; padding contributes neither sum nor count.

    Special tokens remain attended, as in the usual sentence-transformer mean
    pooling recipe. CLS is an explicit ablation, requiring right padding.
    """
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
            or 0 in hidden.shape or not hidden.is_floating_point()):
        raise ValueError("Expected nonempty floating hidden states [batch, tokens, dimension]")
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2
            or attention_mask.shape != hidden.shape[:2]
            or attention_mask.device != hidden.device):
        raise ValueError("Attention mask must match hidden batch and token dimensions")
    if not torch.isfinite(hidden).all():
        raise ValueError("Non-finite hidden states")
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("Attention mask must be binary")
    mask = attention_mask.to(torch.float32)
    counts = mask.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("A sequence has no attended tokens")
    if pooling == "mean":
        pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts
    elif pooling == "cls":
        if not (attention_mask[:, 0] == 1).all():
            raise ValueError("CLS pooling requires an attended first token")
        pooled = hidden[:, 0, :]
    else:
        raise ValueError("pooling must be mean or cls")
    return normalize_embeddings(pooled)


def local_model_manifest(path):
    """Hash only model/tokenizer assets, never arbitrary directory contents.

    Content hashes give an exact local version even when the upstream revision
    is unavailable. They do not establish how the checkpoint was trained.
    """
    path = Path(path).resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError("Supply a local encoder directory containing config.json")
    names = {"config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.txt",
             "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.bpe.model",
             "model.safetensors", "model.safetensors.index.json"}
    names.update(p.name for p in path.glob("model-*.safetensors")
                 if re.fullmatch(r"model-\d+-of-\d+\.safetensors", p.name))
    files = {}
    for name in sorted(names):
        asset = path / name
        if not asset.exists():
            continue
        if not asset.is_file() or not asset.resolve().is_relative_to(path):
            raise ValueError("Model assets must be files within the local model directory")
        digest = hashlib.sha256()
        with asset.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[name] = {"sha256": digest.hexdigest(), "bytes": asset.stat().st_size}
    if not any(name.endswith(".safetensors") for name in files):
        raise FileNotFoundError("Local safetensors weights are required (no pickle fallback)")
    # Prevent a sharded index from directing the loader outside the supplied path.
    if "model.safetensors.index.json" in files:
        import json
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise ValueError("Invalid local safetensors index")
        shards = index.get("weight_map", {})
        if (not isinstance(shards, dict) or not shards
                or any(not isinstance(s, str) or s not in files or not s.endswith(".safetensors")
                       for s in shards.values())):
            raise ValueError("Invalid or missing local safetensors shard")
    return {"version": "sha256:" + sha256_json(files), "files": files}


class FrozenEncoder:
    """Bounded CPU inference from locally supplied AutoModel-compatible weights."""

    def __init__(self, model_path, *, batch_size=8, max_length=128, threads=2,
                 max_texts=1024, max_forward_calls=1024):
        self.batch_size = positive_integer(batch_size, "batch_size", 32)
        self.max_length = positive_integer(max_length, "max_length", 512)
        if self.max_length < 4:
            raise ValueError("max_length must leave room for text and special tokens (at least 4)")
        positive_integer(threads, "threads", 8)
        self.max_texts = positive_integer(max_texts, "max_texts", 4096)
        self.max_forward_calls = positive_integer(max_forward_calls, "max_forward_calls", 4096)
        self.manifest = local_model_manifest(model_path)
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        path = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Encoder tokenizer needs a padding token")
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32).to("cpu")
        if (getattr(self.model.config, "is_decoder", False)
                or getattr(self.model.config, "is_encoder_decoder", False)):
            raise ValueError("Supply an encoder-only checkpoint")
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(limit, int) and max_length > limit:
            raise ValueError("max_length exceeds the encoder position limit")
        self.model.requires_grad_(False)
        self.model.eval()
        self.stats = {"texts": 0, "forward_calls": 0, "attended_tokens": 0,
                      "padded_token_slots": 0}
        self.dimension = None

    def encode(self, texts, *, pooling="mean"):
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Need a nonempty list of texts")
        if any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in texts):
            raise ValueError("Texts must be nonblank strings of at most 4000 characters")
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be mean or cls")
        calls = math.ceil(len(texts) / self.batch_size)
        if (self.stats["texts"] + len(texts) > self.max_texts
                or self.stats["forward_calls"] + calls > self.max_forward_calls):
            raise RuntimeError("Encoder inference budget exhausted")
        rows = []
        self.model.eval()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt",
                                    return_attention_mask=True)
            mask = inputs.get("attention_mask")
            if (mask is None or mask.ndim != 2 or mask.shape[0] != len(batch)
                    or not 1 <= mask.shape[1] <= self.max_length):
                raise ValueError("Tokenizer attention mask violates the batch or token budget")
            self.stats["texts"] += len(batch)
            self.stats["forward_calls"] += 1
            self.stats["attended_tokens"] += int(mask.sum())
            self.stats["padded_token_slots"] += mask.numel()
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                if hidden.shape[0] != len(batch):
                    raise ValueError("Encoder returned the wrong batch dimension")
                vectors = masked_pool(hidden, mask, pooling).cpu()
            if self.dimension is not None and self.dimension != vectors.shape[1]:
                raise ValueError("Encoder dimension changed between batches")
            self.dimension = vectors.shape[1]
            rows.append(vectors)
        return torch.cat(rows, dim=0)


class DenseRetriever:
    """The same Document/search(query, k) contract as retrieval.Retriever."""

    def __init__(self, documents, encoder, *, pooling="mean"):
        self.documents = list(documents)
        if (not self.documents or len({d.id for d in self.documents}) != len(self.documents)
                or any(not isinstance(d.id, str) or not d.id.strip() for d in self.documents)):
            raise ValueError("Need a nonempty corpus with unique, nonblank IDs")
        self.encoder, self.pooling = encoder, pooling
        self.matrix = normalize_embeddings(encoder.encode(
            [d.text for d in self.documents], pooling=pooling))
        if self.matrix.shape[0] != len(self.documents):
            raise ValueError("Encoder returned the wrong number of documents")

    def search(self, query, k=3):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Invalid query")
        positive_integer(k, "k", 10)
        vector = normalize_embeddings(self.encoder.encode([query], pooling=self.pooling))
        if vector.shape != (1, self.matrix.shape[1]):
            raise ValueError("Query and corpus embedding dimensions differ")
        scores = self.matrix @ vector[0]
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite cosine similarities")
        order = torch.argsort(scores, descending=True, stable=True)[:k].tolist()
        return [{"id": self.documents[i].id, "text": self.documents[i].text,
                 "score": float(scores[i])} for i in order]
