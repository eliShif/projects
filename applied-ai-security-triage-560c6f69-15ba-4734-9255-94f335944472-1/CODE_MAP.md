# Code map

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
| Pretrained adaptation | [security_models.py](portfolio/security_models.py), [adapter_tuning.py](portfolio/adapter_tuning.py), [local_lm.py](portfolio/local_lm.py) | Local safetensors, masked answer-token loss, frozen base, query/value LoRA, validation selection and candidate NLL inference. [Optional tensor tests](tests_ml/test_security_torch.py) did not reach test bodies because `torch` could not be imported in the restricted builder environment on 31 October 2026; the later worker installed CPU PyTorch `2.15.0+cpu` but supplied no successful tensor-suite rerun. |
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
