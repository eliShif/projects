# Execution-time local model identities

The extension closes a specific reproducibility gap: the previous protocol
recorded model hashes at construction, while training and both evaluation arms
later loaded files from the same mutable directory. A changed checkpoint could
therefore run under an outdated fingerprint. The original research results and
methodological limitations remain in their existing directories.

## Contract and loading boundaries

`capture_model_identity` runs when a new protocol is created.
`verify_model_identity` reads the files again and compares both names and SHA-256
values with that original map. No size/mtime cache is used. A retry never replaces
the expected hashes with current hashes.

| Execution boundary | Check location | Expected identity source |
|---|---|---|
| Reopening a pretrained run | `security_pipeline.protocol` | Existing `protocol.json` |
| Queued training, including retry/reclaimed lease | `security_models.train_pretrained` | Pipeline's frozen protocol |
| Training's model construction | `security_models.PretrainedPolicy` | Forwarded original map |
| Base evaluation model | `security_models.PretrainedPolicy` | Pipeline's frozen protocol |
| Adapted evaluation model | A second `security_models.PretrainedPolicy` call | Same original map |
| Tokenizer and causal-LM library calls | Separate checks in `local_lm.LocalLM` | Forwarded original map |

The underlying library calls retain `local_files_only=True`,
`trust_remote_code=False` and `use_safetensors=True` for model weights.
Adapter outputs still pass `verified_path` against their training receipt before
loading. Model-identity checks do not replace adapter-receipt checks.

`ModelIdentityError` is persisted using the existing failure-category mechanism.
The failed attempt has no successful receipt, training output or evaluation
metrics. Previous successful stages remain usable. A run can retry after exact
input restoration, within the existing three-attempt budget; exhausted jobs
remain failed. An intentionally different model needs a new output directory.

## Inventory scope

Only the explicit local model directory is inspected. The inventory includes:

- Every root-level `.safetensors` file and `.safetensors.index.json` index.
- All safely referenced shard files, including nested relative shard paths.
  Missing shards, duplicate index keys, invalid maps and path traversal fail.
- `config.json`, `generation_config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`,
  `vocab.json`, `preprocessor_config.json`, `processor_config.json` and
  `chat_template.json` when present.
- Standard tokenizer model/vocabulary inputs (`tokenizer.model`, `spiece.model`,
  `sentencepiece.bpe.model`, `merges.txt`, `vocab.txt`) and chat-template Jinja files.

Relevant additions and removals fail even if the old files are unchanged. All
bound bytes matter, including configuration whitespace. Notes and unrelated JSON
are not inspected. Symlinked artifacts/directories and implicit PEFT base-model
redirects are unsupported. This contract targets self-contained standard local
Llama-style checkpoints; arbitrary tokenizer formats/external references are
outside its supported scope.

Standalone callers may supply an earlier `expected_model_files` map to
`PretrainedPolicy` or `train_pretrained`. Omitting it retains immediate-call
compatibility by capturing on entry; that alone cannot bind an earlier queue
decision. `Pipeline` always supplies the frozen map. Generic inherited LocalLM
uses outside this pipeline are unchanged unless the caller supplies identities.

## Executable evidence

| Requirement | Implementation | Test path | Evidence scope |
|---|---|---|---|
| R02 | `portfolio/model_identity.py`, `portfolio/security_pipeline.py` | `tests_security/test_model_identity.py` | Python hashing, validation and queued failure handling |
| R03 | `portfolio/security_pipeline.py`, `portfolio/security_models.py` | `tests_security/test_models.py`, `tests_security/test_pipeline.py`, `tests_security/test_model_identity.py` | Original deterministic training/validation protocol plus enforcement of model-input identity |
| R04 | `portfolio/model_identity.py`, `portfolio/security_models.py`, `portfolio/local_lm.py`, `portfolio/security_pipeline.py` | `tests_security/test_model_identity.py`, `tests_ml_identity/test_pretrained_identity.py` | Every security training/evaluation load receives the original map; eight real tensor identity tests now pass |
| R05 | `portfolio/security_pipeline.py`, `portfolio/local_lm.py` | `tests_security/test_model_identity.py`, `tests_ml_identity/test_pretrained_identity.py` | Both loader doubles and real tensor loaders verify rejection and unchanged-input success, including restoration and retries |
| R06 | `scripts/check_preservation.py`, `portfolio/security_pipeline.py` | `tests_security/test_preservation.py`, `tests_security/test_preservation_packaging.py`, `tests_security/test_pipeline.py` | Original protected inventory preserved outside seven allowed edits; generated packaging metadata excluded; historical audit compatibility retained |
| P01 | `portfolio/reliable_service.py`, `portfolio/security_pipeline.py` | `tests_security/test_pipeline.py`, `tests_security/test_model_identity.py` | Delayed queued work, retries, reopening and expired leases |
| P02 | `portfolio/security_eval.py`, `portfolio/security_pipeline.py`, `results/model_identity_recovery_validation/REPORT.md` | `tests_security/test_pipeline.py`, `tests_security/test_model_identity.py`, `tests_security/test_preservation.py` | Observed results, unsuccessful checks and preserved earlier evidence documented separately |
| D01 | `portfolio/security_pipeline.py`, `portfolio/reliable_service.py` | `tests_security/test_pipeline.py`, `tests_security/test_model_identity.py` | Existing Python queue extended without replacing lifecycle or receipt behavior |

The offline suite uses one genuine safetensors scalar as a hashing fixture, not
a compatible language model. Transformers calls are explicit test doubles in
those boundary tests, and evaluation prerequisites use labeled fixture receipts.
The separate real tensor suite reuses the inherited compatible tiny Llama
builder, trains through the existing PyTorch/LoRA path, loads both evaluation arms,
replays results, and tests successful restoration and genuine sharded checkpoints.
All eight real identity tests and both inherited tensor tests now execute
successfully in the recovered virtual environment. Their earlier import failures
remain in [the previous validation record](../results/model_identity_validation/REPORT.md).
No randomly initialized fixture is presented as a pretrained research experiment.
Exact current commands/outcomes are in the
[recovery validation record](../results/model_identity_recovery_validation/REPORT.md).

The recovery also fixes the preservation checker: package installation rewrites
generated `.egg-info`, which previously failed the source-preservation assertion.
The authoritative 148-entry baseline is unchanged. Five generated entries are
reported separately; 143 source/documentation/result files remain protected.
Only root build/dependency output is excluded, so a similarly named directory
inside saved results is still checked. New regressions reproduce the packaging
failure and verify changed/missing protected files still fail.

## Remaining limitations

Hashes are integrity checks, not signatures. The process and frozen protocol
must be trusted. A concurrent writer can race after verification and before or
during a library read; these checks do not provide immutable snapshots or locking.
Repeated hashing costs disk I/O, which has not been benchmarked. SHA-256 equality
does not establish model provenance, safe behavior or generalization.

Real tensor integration is verified with tiny randomly initialized local models.
A full pretrained-security experiment has not been run. The original synthetic reference results, rule-baseline tie,
small-sample/one-seed limits, model-comparison confounds, missed attacks and
negative efficiency findings remain unchanged. See [historical validation](validation.md)
and the retained experiment reports for their separate evidence.
