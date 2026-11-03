# Code map

The Python workflow compares a TF-IDF text-RAG baseline with a typed agent using Qwen3-4B GGUF CPU inference. The planner receives the question and operation schema without source values or record notes. A deterministic executor validates operations, retrieves versioned facts, checks units and records source IDs, revisions, fields, operands and operations. Missing facts, conflicting current records and unknown services trigger abstention. The evaluation runner records protocol hashes and actual outputs, separates development/test entities and estimates paired uncertainty by service entity. Supporting components include bounded search/calculation actions, restricted AST arithmetic and custom PyTorch LoRA adapters with frozen base weights and validation-based checkpoint selection. A documented HTTP/SQLite integration connects research requests to durable jobs and inspectable traces.

## Source files

- `portfolio/__init__.py`
- `portfolio/acceptance.py`
- `portfolio/adapter_tuning.py`
- `portfolio/agent_eval.py`
- `portfolio/common.py`
- `portfolio/download_model.py`
- `portfolio/edge_vision.py`
- `portfolio/evidence_agent.py`
- `portfolio/evidence_eval.py`
- `portfolio/gguf_lm.py`
- `portfolio/local_lm.py`
- `portfolio/providers.py`
- `portfolio/reliable_service.py`
- `portfolio/research_cases.py`
- `portfolio/retrieval.py`
- `portfolio/small_lm.py`
- `portfolio/telemetry.py`
- `results/evidence_agent_test/frozen_source/evidence_agent.py`
- `results/evidence_agent_test/frozen_source/evidence_eval.py`
- `results/evidence_agent_test/frozen_source/gguf_lm.py`
- `results/evidence_agent_test/frozen_source/research_cases.py`
- `results/evidence_agent_test/recovery/resumed_runner.py`
- `scripts/resume_frozen_evaluation.py`
- `tests/test_adapters.py`
- `tests/test_evidence_agent.py`
- `tests/test_providers.py`
- `tests/test_service.py`
- `tests/test_statistics_retrieval.py`
- `tests/test_telemetry.py`
