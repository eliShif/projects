# Local checkpoint layout and tokenizer identity

This focused repair extends the existing model-input verifier. The queue,
training loop, two evaluation arms, adapter receipts and historical experiment
artifacts remain unchanged. The earlier [identity contract](model-identity.md)
describes the original loading boundaries; this document supersedes its blanket
symlink restriction and adds versioned tokenizer inputs.

## Supported links

An explicitly supplied model directory can be a symlink. That directory's
resolved target is the checkpoint boundary. Configuration, safetensors,
tokenizer files, indexed shard directories and chat templates may use relative
or absolute links whose targets stay inside that boundary. Internal link chains
are supported. Protocol keys remain logical paths, for example `config.json`
and `shards/part.safetensors`, and SHA-256 covers the target bytes.

Every referenced path component is checked for containment before its file is
opened or directory enumerated. Escaping links, traversal, dangling required
files and loops fail. An escaping artifact is rejected even if its bytes match.
Retargeting a link to different bytes fails the original identity check. Moving
identical bytes between safe internal targets does not change content identity;
the protocol does not pin inodes or link text. A directory alias similarly binds
the checkpoint's logical files and bytes, rather than the alias's destination
name. Implicit PEFT base-model redirects remain unsupported.

## Versioned fast tokenizers

Transformers reads `fast_tokenizer_files` in `tokenizer_config.json` and selects
a `tokenizer.<version>.json` file. Hashing only the default `tokenizer.json`
misses that consumed input. The inventory now reads the configuration and binds
every present advertised version matching Transformers' filename pattern. It
also keeps the default tokenizer and previously supported configuration and
vocabulary inputs in the inventory.

This conservative inventory avoids duplicating the library's version selection
algorithm. Changes to any bound version fail, including a version not selected
by the current library. Absent, unselected advertised versions remain allowed,
as in Transformers; their later appearance changes the file set and fails.
Removal of a previously captured version also fails. Paths are validated before
inspection, including nested references and links. Malformed reference lists,
duplicate JSON keys and paths outside the checkpoint fail explicitly. Arbitrary
custom tokenizer formats and implicit external references remain outside scope.

The real loading tests construct an older tokenizer, a selected tokenizer with
different token IDs, a default tokenizer and an absent future version. They
verify selection through `AutoTokenizer.from_pretrained`, then mutate the selected
file with valid JSON of the same size and modification time. The tests check
that this actually changes tokenization when loaded directly, while queued work
rejects it before the affected loader.

## Loading and failure behavior

The original expected map still flows from `Pipeline.protocol` through
`train_pretrained` and each `PretrainedPolicy` to `LocalLM`. The verifier runs
before both actual `from_pretrained` calls, separately for training, base
evaluation and adapted evaluation. Queue delay, same-instance reuse, reopened
queues, expired leases and retries retain the original map. No retry refreshes
expected hashes. Adapter-receipt validation remains independent.

Identity rejection writes a failure category using the existing queue mechanism.
It publishes no successful receipt or evaluation metrics for that attempt.
Exact restoration permits another attempt within the existing three-attempt
budget. An exhausted job remains failed. Deliberate model or source changes need
a new experiment directory.

## Executable coverage

| Requirement | Implementation | Tests |
|---|---|---|
| R02 | [Python inventory and SHA-256 verifier](../portfolio/model_identity.py) | [Layout inventory regressions](../tests_security/test_model_layouts.py) |
| R03 | [Frozen protocol and lifecycle](../portfolio/security_pipeline.py), [training and validation selection](../portfolio/security_models.py) | [Existing reproducibility tests](../tests_security/test_models.py), [real checkpoint lifecycle](../tests_ml_identity/test_checkpoint_layouts.py) |
| R04 (partial) | [Shared verifier](../portfolio/model_identity.py), [actual loading boundaries](../portfolio/local_lm.py), [policy/training integration](../portfolio/security_models.py) | [Real symlink and versioned-tokenizer loads](../tests_ml_identity/test_checkpoint_layouts.py), [original weight/configuration regressions](../tests_ml_identity/test_pretrained_identity.py); `configuration_files` redirects remain uncovered. |
| R05 | [Queue failure and receipt handling](../portfolio/security_pipeline.py), [per-load checks](../portfolio/local_lm.py) | [Real rejection and restoration controls](../tests_ml_identity/test_checkpoint_layouts.py), [inventory success/failure cases](../tests_security/test_model_layouts.py) |
| R06 | [Original preservation checker](../scripts/check_preservation.py), [repair preservation checker](../scripts/check_layout_preservation.py) | [Original preservation tests](../tests_security/test_preservation.py), [repair preservation tests](../tests_security/test_layout_preservation.py) |
| P01, D01 | [Existing Python queue](../portfolio/reliable_service.py), [experiment stages](../portfolio/security_pipeline.py) | [Original lifecycle tests](../tests_security/test_pipeline.py), [retry/resume/lease regressions](../tests_ml_identity/test_checkpoint_layouts.py) |
| P02 | [Report and replay implementation](../portfolio/security_eval.py), [artifact auditing](../portfolio/security_pipeline.py) | [Existing report consistency tests](../tests_security/test_pipeline.py), [real replay and failed-attempt assertions](../tests_ml_identity/test_checkpoint_layouts.py) |

Run the complete affected security and tensor suites with:

```sh
.venv/bin/python scripts/test_model_identity.py
.venv/bin/python scripts/check_preservation.py
.venv/bin/python scripts/check_layout_preservation.py
```

The targeted layout command is:

```sh
.venv/bin/python -m unittest tests_security.test_model_layouts tests_ml_identity.test_checkpoint_layouts -v
```

See the [repair validation record](../results/model_identity_layout_repair/REPORT.md)
for actual outcomes, commands and preserved failures. All tensor models are tiny
randomly initialized local fixtures built by the original test helper. They
exercise real safetensors, Transformers and PyTorch LoRA without model downloads.
They do not establish pretrained-model quality.

## Remaining limits

The supplied review found incomplete configuration coverage: `model_files` neither
inventories nor rejects `configuration_files` redirects in `config.json`.
Transformers can select a `config.<version>.json` whose contents are not bound
by the frozen identity. Changing that file can therefore avoid an identity
mismatch. R04 remains partially supported pending binding or rejection of those
redirects and a real-loader regression.

Checksums require a trusted protocol and process. They are not signatures,
immutable snapshots or filesystem locks. Concurrent writes can still race after
verification and during library reads; a link can also be retargeted in that
window. Hashing overhead has not been benchmarked. External artifact links,
including cache layouts that point outside the checkpoint, are rejected.

The full pretrained-security research experiment remains unrun. Historical
synthetic-template and one-seed limits, the reference policy's tie with the rule
control, model-comparison confounds, negative efficiency/detection results and
deployment limitations remain in the retained reports. The repair adds input
integrity and compatibility evidence, not a new model-quality experiment.
