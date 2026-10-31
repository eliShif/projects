# Implementation and verification coverage

The table maps this extension's scope to executable paths. It describes project
behavior and verification limits. The reference experiment is a trained linear
policy; it does not supply a pretrained-LLM result. Detailed commands and observed
execution outcomes are in [validation.md](validation.md).

| ID | Implemented behavior | Implementation | Tests | Verification boundary |
|---|---|---|---|---|
| M02 | Structured validation, constrained execution, gradients and durable orchestration | `portfolio/security_triage.py`, `portfolio/security_models.py`, `portfolio/security_pipeline.py` | `tests_security/test_triage.py`, `tests_security/test_models.py`, `tests_security/test_pipeline.py` | Executed standard-library path |
| M04 | Local causal-LM plan scoring, answer-token LoRA adaptation and paired evaluation | `portfolio/security_models.py`, `portfolio/security_eval.py`, `portfolio/adapter_tuning.py` | `tests_ml/test_security_torch.py`, `tests_security/test_pipeline.py` | Evaluation/replay executed with reference policy; new pretrained security experiment and tensor tests unrun |
| M05 | Planner selects a bounded investigation; executor validates scope and derives findings with citations | `portfolio/security_triage.py` | `tests_security/test_triage.py` | Rule/reference and hostile-output fixtures executed; JSON-model adapter connected, new LLM inference unrun |
| M06 | Idempotency, serialized claims, renewal, stale-worker fencing, retries and HTTP validation in one bounded scenario | `portfolio/reliable_service.py`, `portfolio/security_acceptance.py`, `portfolio/security_pipeline.py` | `tests_security/test_acceptance.py`, `tests_security/test_pipeline.py`, `tests_security/test_http_contract.py`, `tests/test_service.py` | 32 jobs/four workers, duplicate HTTP requests, database reopen and expired-lease recovery executed together; socket tests failed in the restricted builder environment on 31 October 2026, with no successful worker socket rerun supplied; production capacity, cloud/TLS unmeasured |
| D01 | Fits a low-rank policy for defined synthetic security investigation procedures | `portfolio/security_models.py`, `portfolio/security_data.py` | `tests_security/test_models.py`, `tests_security/test_triage.py`, `tests_ml/test_security_torch.py` | Real reference gradients/checkpoints measured; no new pretrained security or live-threat result |
| D02 | One queued lifecycle connects preparation, adaptation, evaluation, artifacts and audit | `portfolio/security_pipeline.py` | `tests_security/test_pipeline.py` | Full reference lifecycle, resume and failure recovery executed |
| D03 | Actor/time-scoped alert investigation, policy lookup, correlation, abstention and cited summary | `portfolio/security_triage.py`, `portfolio/reliable_service.py` | `tests_security/test_triage.py`, `tests_security/test_http_contract.py` | Read-only; no response actions or measured operational improvement |
| D04 | Generates research tables from recorded outcomes and links the method to a technical draft | `portfolio/security_eval.py`, `docs/security-whitepaper.md` | `tests_security/test_pipeline.py` | Report text checked against raw metrics, including after rehashing a modified report; no publication outcome asserted |
| P01 | Maps indirect prompt injection and low-rank adaptation literature to explicit controls and an experiment | `portfolio/security_triage.py`, `portfolio/security_models.py`, `docs/security-whitepaper.md` | `tests_security/test_triage.py`, `tests_security/test_models.py` | Concrete paper-to-prototype mapping; narrow threat and synthetic distribution |
| P04 | Reuses PyTorch nn.Module/AdamW, frozen parameters, LoRA gradients, safetensors and masked loss | `portfolio/adapter_tuning.py`, `portfolio/security_models.py` | `tests/test_adapters.py`, `tests_ml/test_security_torch.py` | Implementation and tensor tests provided; optional dependencies were unavailable in the restricted builder environment on 31 October 2026; later worker installed CPU PyTorch `2.15.0+cpu`, but no successful optional tensor-suite execution is supplied |
| I01 | Entity/event split checks, common observations, seeded training, validation selection, version hashes | `portfolio/security_data.py`, `portfolio/security_models.py`, `portfolio/security_pipeline.py` | `tests_security/test_models.py`, `tests_security/test_pipeline.py` | Reference reproducibility checked; cross-platform bitwise equivalence not promised |
| I02 | Independent label oracle, citation-aware scoring, artifact hashes and replay of successful and rejected plans | `portfolio/security_data.py`, `portfolio/security_eval.py`, `portfolio/security_pipeline.py` | `tests_security/test_triage.py`, `tests_security/test_pipeline.py` | False labels, budget-inconsistent labels, missing rows, altered findings/traces and scores rejected |
| I03 | Allowlisted procedures, exact alert binding, semantic validation, separate evidence copies, strict JSON and no response tools | `portfolio/security_triage.py`, `portfolio/common.py`, `portfolio/security_acceptance.py`, `portfolio/reliable_service.py` | `tests_security/test_triage.py`, `tests_security/test_acceptance.py`, `tests_security/test_http_contract.py` | Hostile notes, forbidden operations, fabricated sources, planner mutations, duplicate JSON keys and privilege escalation tested; no universal injection or Python-sandbox claim |

## Retained methodological limits

The original 135M agent failed its usefulness gate despite successful retrieval.
Original LoRA improved NLL without improving exact match. Typed Qwen versus RAG
uses different evidence-access architectures; it cannot isolate model quality.
Those experiments use small synthetic suites, shared task families and limited
entity clusters. Early baseline timing was affected by concurrent CPU work.

The retained vision/INT8 and small-LM reports include slower tiny-model latency,
worse language-model perplexity and higher process RSS despite smaller serialized
weights. Power, native edge devices, Windows/NPU deployment and production MLOps
were not measured. The telemetry lab retains slow-attack misses and maintenance
false positives. KQL, native cloud ingestion and real-incident performance remain
unexecuted. Hosted-provider behavior has contract fixtures only; no new hosted
inference or paid API calls are part of this extension.
