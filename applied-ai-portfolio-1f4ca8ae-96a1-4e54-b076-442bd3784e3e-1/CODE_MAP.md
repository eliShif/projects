# Code map

## Extended experiment path

1. `portfolio/edge_protocol.py`: validates budgets and seeds, builds four-position Latin-square schedules, and selects the smallest feasible model using validation quality and measured resources.
2. `portfolio/edge_benchmark.py`: saves the protocol, launches fresh Python workers, captures commands/output/exit codes, freezes the decision, runs held-out evaluation and renders the tradeoff report.
3. `portfolio/edge_vision.py`: reused digits loading/normalization, fixed stratified splits, MLPs, Adam training, validation checkpoint restoration, supervised/KL distillation, dynamic qint8 conversion, warmups and raw CPU timings. The original `portfolio.edge_vision` entry point remains available.
4. `portfolio/measurement.py`: records allowlisted host/package metadata, linear timing quantiles, Linux/macOS process high-water RSS, Windows peak working set, file hashes and run lifecycle. Unavailable power is represented by null values and a reason.
5. `portfolio/common.py`: retained JSON, hashing, Wilson intervals and bootstrap utilities. NumPy loads only when the bootstrap is called, allowing the queue and protocol paths to run without ML packages.

`protocol.json` → training workers → `models/` and training histories → isolated `trials/` → `selection.json` → held-out `test/` → `summary.json`, `REPORT.md`, `manifest.json`.

`status.json` remains `failed` on an exception; partial evidence survives and successful summary/report files are removed. Each run requires an empty output directory. The worker's stored model hash is checked before `weights_only=True` loading.

## Service and verification

| File | Responsibility | Primary tests |
|---|---|---|
| `portfolio/reliable_service.py` | Existing SQLite WAL transactions, idempotency keys, lease fencing, retries and HTTP routing; retrieval imports are deferred until model-engine use | `tests/test_service.py` |
| `portfolio/service_load.py` | Finite HTTP or direct-store load, raw latency samples, expected conflicts, receipt checks and fault probes | `tests/core/test_queue_load.py`, `tests/test_service_load_http.py` |
| `portfolio/smoke.py` | Temporary-store run plus manifest verification; explicit optional real-vision and HTTP checks | Invoked by `scripts/validate.py`; uses the tested load and benchmark implementations |
| `scripts/validate.py` | Executes test/build/smoke commands without a shell, records logs and actual return codes | `results/validation_core_final/commands.json`, `results/validation_all_final/commands.json` |
| `tests/core/test_measurement.py` | Numerical quantiles, peak-memory units/availability, manifest correctness and failed-run cleanup | Standard-library test suite |
| `tests/core/test_edge_protocol.py` | Balanced order, input rejection, test-independent selection, quality/resource rejection and missing trials | Standard-library test suite |
| `tests/core/test_edge_runner.py` | Child-process failure, timeout and runtime-control contracts | Standard-library test suite |
| `tests/test_edge_models.py` | Real gradients, unchanged teacher, selected checkpoint, qint8 kernels, isolated pipeline and corrupted artifacts | Requires installed PyTorch/NumPy/scikit-learn; passed in later worker validation, including one-seed, two-epoch isolated integration; full three-seed benchmark remains unverified |

## Retained components

- `portfolio/agent_eval.py`, `retrieval.py`, `local_lm.py`: original controlled LLM evaluation, TF-IDF retrieval and local Transformers inference.
- `portfolio/adapter_tuning.py`: frozen-base LoRA training and validation-selected adapters.
- `portfolio/evidence_agent.py`, `research_cases.py`, `evidence_eval.py`, `gguf_lm.py`: typed evidence plans, synthetic data, evaluation and local GGUF inference.
- `portfolio/small_lm.py`: original isolated FP32/INT8 language-model comparison.
- `portfolio/telemetry.py`, `queries/detections.kql`: synthetic-event evaluation and unexecuted KQL reference queries.
- `portfolio/providers.py`: optional hosted-model adapter; only contract fixtures are tested, with no paid API execution.
- `results/evidence_agent_test/frozen_source/`, `results/evidence_agent_test/recovery/`: retained evaluation source and interruption checkpoints.

See [coverage](docs/coverage.md) for implementation and test paths per requirement. [UPSTREAM.md](UPSTREAM.md) identifies the original source; [VALIDATION.md](VALIDATION.md) distinguishes historical results from current execution.
