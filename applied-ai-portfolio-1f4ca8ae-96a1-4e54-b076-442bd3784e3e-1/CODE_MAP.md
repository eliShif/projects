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

## Classical vision and signal-processing extension

Details and limitations: [docs/application-bff5013d.md](docs/application-bff5013d.md).

| File | Responsibility | Primary tests |
|---|---|---|
| `portfolio/classical_vision.py` | NumPy reference: correlation, Gaussian/Sobel, threshold edges, FIR design/filtering, FFT/STFT, edge scoring, seeded synthetic generators, ROI time-activity curves. Imports only NumPy | `tests/test_classical_vision.py` (also checked against `scipy.signal.correlate2d`) |
| `cpp/filters.cpp` | C++17 kernels behind a stdin/stdout text protocol; validates input and exits with code 2 on rejection; reports compute-loop time | `tests/test_classical_native.py` |
| `portfolio/classical_native.py` | Locates or compiles the binary with `g++`, runs it without a shell, parses results, exposes `NATIVE` backend with the same interface as `classical_vision.NUMPY` | `tests/test_classical_native.py` |
| `portfolio/classical_benchmark.py` | Equivalence, edge-quality and timing check; reuses `measurement.distribution`, `RunRecord` and `environment` | `tests/test_classical_benchmark.py` |

Data flow: `synthetic_image` / `synthetic_signal` → reference function and (optionally) native function → tolerance comparison and timing distributions → console JSON or a recorded run directory. The prebuilt binary is `cpp/classical_filters.bin` (ignored by the existing `*.bin` rule); a missing or stale binary is rebuilt into a temporary directory.

Update: no binary is stored in the repository any more. `portfolio/classical_native.py` always compiles `cpp/filters.cpp` into a per-process temporary directory, and `python -m portfolio.classical_native` is the build check (it also removes a leftover `cpp/classical_filters.bin`). Regression tests: `tests/test_classical_artifacts.py`.

## LLM quality operations extension

Details and limitations: [docs/application-54a5d041.md](docs/application-54a5d041.md). One new module and one new test file; no existing file is modified.

| Area in `portfolio/llm_quality_ops.py` | Responsibility | Reuses |
|---|---|---|
| `make_cases`, `Case`, `ATTRIBUTES` | Deterministic synthetic evidence questions; a configurable number (default 6 of 24) cannot be answered from the supplied passages | none |
| `PROMPT_VERSIONS`, `render_messages` | Three prompt versions (free-form, JSON with citations, JSON with citations and abstention) | none |
| `parse_output`, `score_case` | Strict JSON/schema parsing; grounded-correct, invalid-citation, ungrounded, abstained flags | none |
| `StubGenerator`, `LocalLMGenerator` | Deterministic rule-based stand-in (simulated latency) and an adapter for `LocalLM.generate` | `portfolio/local_lm.py` |
| `evaluate_version`, `compare_evaluations`, `compare_versions` | Same cases and seeds for every version; paired case-clustered bootstrap and a decision (`a_better`, `b_better`, `inconclusive`) | `common.clustered_difference` |
| `request_spans`, `export_traces_jsonl`, `load_traces_jsonl`, `validate_spans`, `records_from_spans`, `spans_from_records` | Span-style JSONL export, strict loading and conversion to monitor records | none |
| `percentile`, `latency_stats` | p50/p95 from the existing distribution function; p99 computed here | `measurement.distribution` |
| `psi`, `ks_statistic`, `ks_critical`, `MonitorConfig`, `monitor` | Window statistics, rate-shift test (Wilson interval vs reference rate plus margin), PSI/KS drift, alert records | `common.wilson` |
| `synthetic_records`, `SCENARIOS`, `run_demo`, `main` | Seeded request streams with injectable faults, the finite check and the CLI (`--out`, `--replay`, `--model-dir`) | `measurement.RunRecord`, `common.write_json` |

Data flow: `make_cases` + prompt version + generator → parsed and scored rows → outcome matrix → paired comparison. In parallel, `synthetic_records` (or a loaded trace file) → `monitor` → windows and alerts. Spans from the comparison run are exported and reloaded as the trace round-trip check. Tests: `tests/test_llm_quality_ops.py` (`CaseTests`, `ParsingAndScoringTests`, `PromptComparisonTests`, `StatisticsTests`, `MonitorTests`, `TraceTests`, `CommandLineTests`).
