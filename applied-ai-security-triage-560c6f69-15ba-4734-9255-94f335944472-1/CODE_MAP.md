# Code map

## Execution-time model identity extension

| Boundary | Implementation | Verification |
|---|---|---|
| Capture and verify | [model_identity.py](portfolio/model_identity.py) inventories explicit model inputs, validates shard references and compares fresh SHA-256 values with the original map. | [IdentityInventoryTests](tests_security/test_model_identity.py): unchanged inputs, same-size/same-mtime mutations, missing/added files, shards, invalid indexes and symlinks. |
| Local layout compatibility | [model_identity.py](portfolio/model_identity.py) preserves logical protocol keys, supports directory aliases and trusted shared-cache links in guarded loading, and retains an explicit closed-bundle helper policy. | [Inventory layout tests](tests_security/test_model_layouts.py), [real linked-checkpoint tests](tests_ml_identity/test_checkpoint_layouts.py), [compatibility regressions](tests_security/test_standard_checkpoint_compatibility.py) and [real compatibility tests](tests_ml_identity/test_real_standard_checkpoint_compatibility.py): supported links, shards, changed targets, queued loading and restoration. |
| Versioned tokenizer inputs | `model_files` in [model_identity.py](portfolio/model_identity.py) parses `fast_tokenizer_files` and binds all present advertised versions. | [Version inventory tests](tests_security/test_model_layouts.py) and [actual Transformers selection/mutation tests](tests_ml_identity/test_checkpoint_layouts.py), including queue delays, each evaluation arm, retries and tokenizer-to-model changes. |
| Versioned configurations and standard checkpoints | [model_identity.py](portfolio/model_identity.py) binds advertised configuration inputs, supports dot-relative shards, ignores unused monolithic indexes and preserves distinct symlink/parent shard identities. | [Compatibility regressions](tests_security/test_standard_checkpoint_compatibility.py) and [actual configuration selection/loading tests](tests_ml_identity/test_real_standard_checkpoint_compatibility.py), including mutation rejection at affected boundaries. |
| Frozen queue protocol | [security_pipeline.py](portfolio/security_pipeline.py) captures once, validates saved identities on resume and passes them to every training/evaluation call. | [QueuedIdentityTests](tests_security/test_model_identity.py): delay, two evaluation boundaries, retry, reopen, expired leases and unpublished failures. |
| Training and policy loading | [security_models.py](portfolio/security_models.py) verifies before optional imports/loading and preserves the separate adapter-receipt path. | [Queue regressions](tests_security/test_model_identity.py) and [real tensor integration](tests_ml_identity/test_pretrained_identity.py). All eight original real identity tests also passed within the latest 124-test worker run. |
| Actual library calls | [local_lm.py](portfolio/local_lm.py) checks immediately before tokenizer and model `from_pretrained` when given expected identities. | [LoaderBoundaryTests](tests_security/test_model_identity.py) uses explicit loader doubles. The [real identity tests](tests_ml_identity/test_pretrained_identity.py) and [layout tests](tests_ml_identity/test_checkpoint_layouts.py) passed unchanged-input controls and tokenizer-to-model mutation rejection in the latest 124-test combined worker run. |
| Preservation | [check_preservation.py](scripts/check_preservation.py) checks the normalized 141-entry authoritative inventory against an explicit change allowlist; private worker output and generated packaging files are excluded. The earlier 148/143 totals describe the 1 November 2026 pre-publication workspace. | [test_preservation.py](tests_security/test_preservation.py) checks deletion/replacement; [packaging regressions](tests_security/test_preservation_packaging.py) allow regenerated/missing build output while rejecting source, test and result changes. |
| Repair checkpoint preservation | [check_layout_preservation.py](scripts/check_layout_preservation.py) checks the 235-file recovery checkpoint using the current nine-file allowlist. The ephemeral project-root CPU pip constraint is excluded from source preservation; historical tests and results remain protected. | [Repair preservation tests](tests_security/test_layout_preservation.py) and [compatibility preservation regressions](tests_security/test_standard_checkpoint_compatibility.py) check protected changes and the narrowly scoped host-constraint exclusion. The five-edit count belongs to the earlier 1 November 2026 layout repair. |
| Combined regression command | [test_model_identity.py](scripts/test_model_identity.py) runs the security and both tensor suites, including real tiny local checkpoints. | Uses the original tests plus the added identity/preservation tests; missing dependencies fail explicitly. |

The security call chain is `Pipeline.work_once → train_pretrained →
PretrainedPolicy → LocalLM` for fitting, and two independent `PretrainedPolicy →
LocalLM` chains for evaluation. Both chains receive the same original identity
map. Rehashing at work time never updates the protocol. Model-identity failure
uses the existing attempt failure record and bounded queue retry mechanism.

The generic inherited LocalLM API remains usable without an identity map; the
security queue always supplies one. The checks do not lock files across library
reads. See [current scope, evidence and limitations](docs/model-input-layouts.md).

## Retained investigation and experiment components

The default path is `security_pipeline.run → JobStore → prepare → train → evaluate`.
`security_smoke` exercises the same path in a temporary workspace and exits.

| Component | Implementation | Behavior and verification |
|---|---|---|
| Alert and event boundary | [security_triage.py](portfolio/security_triage.py), [telemetry.py](portfolio/telemetry.py) | Validates identity, time, completeness, event schema and budget; exact actor/time retrieval. [Triage tests](tests_security/test_triage.py). |
| Versioned policy execution | [evidence_agent.py](portfolio/evidence_agent.py), `policy_values` in [security_triage.py](portfolio/security_triage.py) | Existing typed lookups resolve latest policy or abstain on missing/conflicting facts. [Inherited evidence tests](tests/test_evidence_agent.py), [triage tests](tests_security/test_triage.py). |
| Bounded planner/executor | [security_triage.py](portfolio/security_triage.py) | Rule, JSON-model and learned planners choose among two investigations and abstention. Scope and semantic checks precede execution. Findings and citations are constructed from evidence. No response tools. [Triage tests](tests_security/test_triage.py). |
| Evidence and JSON boundaries | [security_triage.py](portfolio/security_triage.py), [common.py](portfolio/common.py), [reliable_service.py](portfolio/reliable_service.py) | Separate planner/executor snapshots; duplicate keys, nonfinite numbers, nesting and incomplete bodies rejected. [Triage tests](tests_security/test_triage.py), [boundary tests](tests_security/test_acceptance.py), [HTTP tests](tests_security/test_http_contract.py). |
| Data preparation | [security_data.py](portfolio/security_data.py) | 32 train, 16 validation and 24 test cases; disjoint actors/events; independent label checks. [Data tests](tests_security/test_triage.py). |
| Reference adaptation | [security_models.py](portfolio/security_models.py) | Real analytic gradients train a rank-2 update of a frozen seeded linear policy. This backend is not an LLM. [Gradient and checkpoint tests](tests_security/test_models.py). |
| Pretrained adaptation | [security_models.py](portfolio/security_models.py), [adapter_tuning.py](portfolio/adapter_tuning.py), [local_lm.py](portfolio/local_lm.py) | Local safetensors, masked answer-token loss, frozen base, query/value LoRA, validation selection and candidate NLL inference. Both [inherited tensor tests](tests_ml/test_security_torch.py) now pass using a random tiny fixture. Historical missing-dependency failures remain recorded; no full pretrained-security research run has been added. |
| Evaluation and writing | [security_eval.py](portfolio/security_eval.py) | Identical observations across arms, raw outputs and failed calls, citation-aware scoring, family counts, paired changes, report generation and replay of successful and rejected plans. [Evaluation tests](tests_security/test_pipeline.py). |
| Durable lifecycle | [security_pipeline.py](portfolio/security_pipeline.py), [reliable_service.py](portfolio/reliable_service.py) | Idempotent stages, prerequisite receipts, renewable leases, bounded retries and attempt-specific artifacts. Only fenced completion publishes a receipt. [Pipeline and queue tests](tests_security/test_pipeline.py). |
| HTTP contract | [reliable_service.py](portfolio/reliable_service.py) | Structured `security_triage` payloads, authorization, body limits and ambiguous-length rejection. Training stages are internal and rejected by the HTTP validator. [Byte-stream handler tests](tests_security/test_http_contract.py); [socket tests](tests/test_service.py) need socket access. |
| Combined service acceptance | [security_acceptance.py](portfolio/security_acceptance.py) | 32 jobs, four workers, duplicate HTTP submissions, invalid requests, database reopen, expired-lease recovery, stale-write rejection and five adversarial plan fixtures. Saves raw outcomes and checks them during audit. [Acceptance tests](tests_security/test_acceptance.py). |
| Finite smoke | [security_smoke.py](portfolio/security_smoke.py) | Runs all stages, verifies resume/replay, submits a hostile-note alert and rejects stale completion. |

`protocol.json` records source/data/model digests and budgets before processing.
`manifest.json` references completed stage receipts. Each receipt names files in
an attempt directory and their hashes; `audit` checks these and replays findings
and scores. The database is needed for resuming work, but saved receipts and
artifacts suffice for offline audit. Hashes are integrity checks, not signatures.
Format-2 runs also retain `acceptance.json` and verify that the research report
matches the recorded metrics. Format-1 checkpoint artifacts remain auditable.

The inherited vision, language-model efficiency, telemetry and Qwen experiments
remain in their original modules and result directories. Their methodology and
negative results are retained in `docs/` and linked from the root README.
