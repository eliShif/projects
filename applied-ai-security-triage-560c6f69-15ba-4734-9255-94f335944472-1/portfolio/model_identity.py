"""SHA-256 identities for self-contained local Transformers model inputs.

The closed-bundle helper policy confines references to its checkpoint. Existing
loading APIs opt into compatible trusted-local layouts, including shared-cache
symlink targets, dot-relative shards and advertised configuration versions.
Only explicit model artifacts/references are inventoried, not account directories.
Hashes enforce an earlier byte-content protocol; they are neither filesystem
permissions, signatures nor locks. Loader sandboxing remains a separate boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re


class ModelIdentityError(ValueError):
    """Local model inputs no longer match the frozen experiment protocol."""


def _checkpoint_root(model_path):
    try:
        path = Path(model_path)
        # Preserve the caller's directory alias and inner symlink/parent
        # semantics while making relative and absolute calls name inputs alike.
        return path if path.is_absolute() else Path.cwd() / path
    except (OSError, RuntimeError) as exc:
        raise ModelIdentityError("Invalid local checkpoint root") from exc


_TOKENIZER_FILES = {"tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
                    "merges.txt", "vocab.txt", "chat_template.jinja"}
_CONFIG_FILES = {"config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                 "vocab.json", "preprocessor_config.json", "processor_config.json",
                 "chat_template.json"}


def _local_path(root, name, compatible=False):
    """Keep logical names for protocol keys, confine links to the real root.

    The model directory itself may be a symlink supplied by the caller. Check
    each relative component before opening files or enumerating directories so
    an artifact link cannot redirect inspection outside that checkpoint.
    Missing optional tokenizer versions can be checked without opening them.
    """
    if compatible:
        # Match the legacy loading APIs: caller-selected local artifacts can be
        # links to shared cache blobs. Identity is byte content, not authorization.
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ModelIdentityError("Invalid local model artifact path")
        path = root / name
        try:
            path.resolve()
        except (OSError, RuntimeError) as exc:
            raise ModelIdentityError("Invalid local model artifact link: " + name) from exc
        return path
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or
            "\\" in name or relative.as_posix() != name):
        raise ModelIdentityError("Unsafe local model artifact path")
    path = root
    try:
        boundary = root.resolve(strict=True)
        for part in relative.parts:
            path = path / part
            if not path.resolve().is_relative_to(boundary):
                raise ModelIdentityError("Model artifact escapes local checkpoint: " + name)
    except (OSError, RuntimeError) as exc:
        raise ModelIdentityError("Invalid local model artifact link: " + name) from exc
    return path


def _regular_file(root, name, compatible=False):
    path = _local_path(root, name, compatible)
    if not path.is_file():
        raise ModelIdentityError("Missing local model artifact: " + name)
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelIdentityError("Duplicate model-input JSON field")
        result[key] = value
    return result


def _effective_adapter_base(root):
    """Match AutoModel's optional PEFT redirect without fetching remote files.

    Without PEFT the sidecar is unused and must not narrow normal loading.
    Local/cache-only base selection follows the same default revision as the
    existing LocalLM call. The base model is inventoried separately from the
    tokenizer, which AutoTokenizer still loads from the caller's directory.
    """
    sidecar = root / "adapter_config.json"
    if not sidecar.is_file() or importlib.util.find_spec("peft") is None:
        return None
    from transformers.utils import is_peft_available
    if not is_peft_available():
        return None
    try:
        config = json.loads(sidecar.read_text())
        name = config["base_model_name_or_path"]
        if not isinstance(name, str) or not name:
            raise ValueError("Missing PEFT base model")
        candidate = Path(name)
        if not candidate.is_dir():
            from transformers.utils.hub import cached_file
            resolved = cached_file(name, "config.json", local_files_only=True)
            if not resolved:
                raise ValueError("PEFT base is not available locally")
            candidate = Path(resolved).parent
        return candidate.resolve()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelIdentityError("Invalid or unavailable local/cache-only PEFT base") from exc


def model_files(model_path, *, compatible=False, _model_only=False):
    """Inventory JSON/configuration, tokenizer inputs, weights and index shards.

    Root-level safetensors (including alternative variants) are all bound.
    Nested shards are supported only through safe relative index references.
    Tokenizer versions advertised in tokenizer_config.json are bound when
    present, including versions not selected by the installed Transformers.
    Unrelated notes are ignored. Local symlinks retain their logical names.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError("Local pretrained model config.json required")
    if not compatible and ((root / "adapter_config.json").exists() or (root / "adapter_config.json").is_symlink()):
        raise ModelIdentityError("Implicit PEFT base-model redirects are unsupported")
    base = _effective_adapter_base(root) if compatible and not _model_only else None
    config_files = {"config.json", "generation_config.json"} if _model_only else _CONFIG_FILES
    names = {p.name for p in root.iterdir()
             if (not compatible or p.is_file()) and
             (p.suffix == ".safetensors" or p.name.endswith(".safetensors.index.json")
              or p.name in config_files | (set() if _model_only else _TOKENIZER_FILES))}
    if not _model_only and compatible and (root / "adapter_config.json").is_file():
        names.add("adapter_config.json")
    if base is not None and (root / "adapter_model.bin").is_file():
        names.add("adapter_model.bin")
    if "config.json" not in names:
        raise FileNotFoundError("Local pretrained model config.json required")
    templates = root / "chat_templates"
    if not _model_only and (templates.is_dir() if compatible else templates.exists() or templates.is_symlink()):
        templates = _local_path(root, "chat_templates", compatible)
        if not templates.is_dir():
            raise ModelIdentityError("Invalid local chat template directory")
        names.update("chat_templates/" + p.name for p in templates.iterdir()
                     if p.suffix == ".jinja" and (not compatible or p.is_file()))
    # Validate paths before opening even the index, never following its references
    # into arbitrary configuration/account directories.
    for name in sorted(names):
        _regular_file(root, name, compatible)
    # AutoConfig can select a versioned configuration from this advertised
    # list. Bind every existing matching candidate, including nested references;
    # absent unselected versions may legitimately remain absent.
    try:
        config = json.loads((root / "config.json").read_text(),
                            object_pairs_hook=None if compatible else _unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        if compatible and base is not None and base.resolve() != root.resolve():
            # With a complete caller tokenizer, AutoModel redirects to the
            # separate PEFT base without parsing the caller's config.json.
            # Its raw bytes remain bound; the actual loader still rejects an
            # invalid configuration if tokenizer selection needs to consume it.
            config = {}
        else:
            raise ModelIdentityError("Invalid model configuration") from exc
    versions = config.get("configuration_files", []) if isinstance(config, dict) else []
    if isinstance(versions, (list, dict) if compatible else list):
        for name in versions:
            if not isinstance(name, str) or re.search(r"config\.(.*)\.json", name) is None:
                continue
            if compatible and not (root / name).is_file():
                continue  # Unavailable, unselected references are not loader inputs.
            path = _local_path(root, name, compatible)
            if path.is_file() or (not compatible and (path.exists() or path.is_symlink())):
                _regular_file(root, name, compatible)
                names.add(name)
    if "tokenizer_config.json" in names:
        try:
            config = json.loads((root / "tokenizer_config.json").read_text(), object_pairs_hook=None if compatible else _unique_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ModelIdentityError("Invalid tokenizer configuration") from exc
        if not isinstance(config, dict):
            raise ModelIdentityError("Invalid tokenizer configuration")
        # These initialization kwargs can override the resolved standard files.
        # Transformers passes their strings to constructors unchanged: relative
        # values are relative to process cwd, not to the checkpoint directory.
        for key in ("tokenizer_file", "vocab_file", "merges_file"):
            value = config.get(key)
            if not isinstance(value, str) or not value:
                continue
            override = Path(value)
            if not override.is_absolute():
                override = Path.cwd() / override
            if not override.is_file():
                continue  # Unused/missing slow-tokenizer kwargs remain the loader's decision.
            if compatible:
                names.add(override.as_posix())
            else:
                try:
                    names.add(override.resolve().relative_to(root.resolve()).as_posix())
                except ValueError as exc:
                    raise ModelIdentityError("Tokenizer override escapes closed bundle") from exc
        versions = config.get("fast_tokenizer_files", [])
        if not isinstance(versions, (list, dict) if compatible else list) or any(not isinstance(name, str) for name in versions):
            if not compatible:
                raise ModelIdentityError("Invalid fast_tokenizer_files references")
            versions = []  # Let the actual legacy tokenizer reject invalid metadata.
        for name in versions:
            # Transformers get_fast_tokenizer_file uses this search pattern.
            # Bind every available advertised version rather than guessing its
            # version-dependent selection. Absent unselected versions are valid
            # in Transformers; their later appearance changes the identity set.
            if re.search(r"tokenizer\.(.*)\.json", name) is None:
                continue
            if compatible and not (root / name).is_file():
                continue  # Includes unavailable self-links on older Python versions.
            path = _local_path(root, name, compatible)
            if path.is_file() or (not compatible and (path.exists() or path.is_symlink())):
                _regular_file(root, name, compatible)
                names.add(name)
    indexes = [name for name in sorted(names) if name.endswith(".safetensors.index.json")]
    if compatible:
        # Transformers selects the monolithic standard file before its default
        # shard index. Unused indexes remain hashed but are not parsed as inputs.
        indexes = (["model.safetensors.index.json"]
                   if not (root / "model.safetensors").is_file()
                   and "model.safetensors.index.json" in names
                   and (base is None or base.resolve() == root.resolve()) else [])
    for name in indexes:
        try:
            index = json.loads((root / name).read_text(), object_pairs_hook=None if compatible else _unique_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ModelIdentityError("Invalid safetensors shard index") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelIdentityError("Missing safetensors shard map")
        for tensor, shard in weight_map.items():
            if (not isinstance(tensor, str) or not tensor or not isinstance(shard, str)
                    or not shard.endswith(".safetensors")):
                raise ModelIdentityError("Invalid safetensors shard reference")
            _regular_file(root, shard, compatible)
            names.add(shard)
    files = [_regular_file(root, name, compatible) for name in sorted(names)]
    if base is not None and base.resolve() != root.resolve():
        # AutoModel peels the caller's adapter once; the actual base load receives
        # that caller adapter explicitly. A base's own sidecar is not another redirect.
        files.extend(model_files(base, compatible=True, _model_only=True))
    if not any(file.name.endswith(".safetensors") for file in files):
        raise FileNotFoundError("Local pretrained safetensors base weights required")
    return files


def capture_model_identity(model_path, *, compatible=False):
    """Capture once at protocol creation, never as a replacement during work."""
    root = _checkpoint_root(model_path)
    result = {}
    for path in model_files(root, compatible=compatible):
        with path.open("rb") as stream:
            # Do not lexically collapse '..': after a symlink, it may resolve
            # to a different file than that lexical normalization would imply.
            try:
                name = path.relative_to(root).as_posix()
            except ValueError:
                if not compatible:
                    raise
                try:
                    # Absolute tokenizer overrides may spell the same internal
                    # file through another directory alias. Keep that consistent
                    # with the closed-bundle capture without collapsing logical
                    # shard references already retained by the branch above.
                    name = path.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    name = path.as_posix()  # Explicit external absolute reference.
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if name in result and result[name] != digest:
                raise ModelIdentityError("Ambiguous local model identity key")
            result[name] = digest
    return result


def verify_model_identity(model_path, expected, *, compatible=False):
    """Re-read bytes and reject changed, missing or added relevant inputs."""
    if (not isinstance(expected, Mapping) or not expected or "config.json" not in expected or
            not any(isinstance(name, str) and name.endswith(".safetensors") for name in expected) or
            any(not isinstance(name, str) or not isinstance(digest, str) or
                re.fullmatch(r"[0-9a-f]{64}", digest) is None for name, digest in expected.items())):
        raise ModelIdentityError("Invalid frozen model identity")
    try:
        actual = capture_model_identity(model_path, compatible=compatible)
    except (OSError, ValueError) as exc:
        raise ModelIdentityError("Model identity check failed: missing or invalid inputs") from exc
    if set(actual) != set(expected):
        raise ModelIdentityError("Model identity file set changed")
    if actual != dict(expected):
        raise ModelIdentityError("Model identity SHA-256 mismatch")
