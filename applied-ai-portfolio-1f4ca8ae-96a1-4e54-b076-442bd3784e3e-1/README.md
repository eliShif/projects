# Applied AI Portfolio: measured model and service tradeoffs

Python experiments for image-model training, distillation, INT8 inference, language-model evaluation, and reliable job processing. This extension reuses the existing digits models and SQLite service from the [pinned upstream source](UPSTREAM.md).

The digits benchmark adds isolated measurement processes, balanced variant order, raw timing samples, OS peak-memory accounting, hardware/runtime metadata, and a validation-only quality/resource decision. The service harness adds bounded concurrent load, content-conflict checks, lease-boundary failures, and retry verification. No external model service is needed for these paths.

## Setup

Use Python 3.11+ and a CPU build of PyTorch. Run commands from the repository root. The core queue and protocol checks use only the standard library; full model checks require the existing dependencies. No activation script is needed.

```text
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
```

The Linux/macOS environment uses `.venv/bin/python`; on Windows use the equivalent virtual-environment Python executable. Native Windows execution and its memory API have not been validated in this workspace. NumPy, SciPy and scikit-learn retain the supplied pins. PyTorch and other inherited ranges resolve at install time; each run records the installed versions.

## Verify and run

The following commands work without ML packages, model downloads, or sockets:

```text
python3 -m unittest discover -s tests/core -v
python3 -m unittest tests.test_service.StoreTests -v
python3 -m compileall -q portfolio tests scripts
python3 -m portfolio.smoke
python3 -m portfolio.service_load --transport store --jobs 24 --clients 4 --copies 3 --out results/store_load_new
```

`smoke` is finite: it creates a temporary database, exercises concurrent retries, conflicting content, stale leases and processing recovery, verifies the artifact manifest, and exits. It explicitly reports `vision_executed: false` and `service_transport: store`.

After dependencies are installed and loopback sockets are permitted:

```text
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m portfolio.smoke --vision --http
.venv/bin/python -m portfolio.edge_benchmark --epochs 100 --rounds 4 --repetitions 100 --threads 1 --device-label development-host --power-condition unknown --out results/edge_isolated_new
.venv/bin/python -m portfolio.service_load --transport http --jobs 24 --clients 4 --copies 3 --out results/http_load_new
```

The full digits run trains the original teacher, supervised control and distilled student at seeds 7, 19 and 43, converts the student to actual dynamic INT8, and runs 48 separate measurement workers. It preserves the original fixed splits, 10 warmups, and batch sizes 1 and 32. Worker processes time out after 600 seconds each. `--vision` smoke uses one seed and two epochs; its purpose is integration correctness, not model-quality evidence.

Use a **new, empty output directory** for each run. Failed runs keep their protocol, status and available logs; they do not publish a successful summary. Existing upstream records and recovery checkpoints remain intact.

## Current verification

[VALIDATION.md](VALIDATION.md) records worker verification at **2026-10-31T14:15:10.319Z**. Full test discovery passed **75 tests in 139.882 seconds**, including real-model checks and concurrent HTTP retries, conflicts, receipts and fault handling. Python compilation and the documented finite store smoke also exited successfully. The isolated model integration used one seed and two epochs; a completed full three-seed benchmark and its model recommendation remain unverified. The documented smoke used direct SQLite and reported `vision_executed: false` and `power_measured: false`.

Historical builder sandbox, before that worker verification (exact builder timestamp not supplied): the 27 new core checks, nine inherited store tests, compilation and finite store smoke passed. Full discovery recorded 10 dependency-import/socket-permission errors; dependency installation failed, the vision attempt stopped at missing PyTorch, and the HTTP-load attempt was blocked by prohibited sockets. The later worker successfully executed the previously blocked model and HTTP tests.

The [executed SQLite load report](results/store_load_verified/REPORT.md) records 24 unique jobs, 72 submissions, four threads and 24 rejected content conflicts. These are direct queue measurements with a deterministic receipt worker, not HTTP latency or AI inference throughput.

## Interpreting model results

Default experimental gates require validation accuracy of at least 95% for every seed, at most two percentage points below the teacher, state size at most 16 KiB, batch-1 p95 at most 1 ms in every trial, and process peak memory at most 1,024 MiB. The smallest feasible state is selected, with median latency breaking ties. `selection.json` is saved before held-out evaluation. These budgets are configurable research criteria, not product requirements. A run can validly recommend no variant.

The [inherited vision results](results/edge_vision/REPORT.md) report INT8 storage reduction alongside **slower** batch-1 inference. They were not rerun here. Until the new benchmark runs on an identified target, retain that negative latency finding and do not infer speed, RAM or energy savings from smaller files. Power and energy remain explicitly unavailable; no sensor has been instrumented. A development-host run does not verify an edge device or Windows deployment.

## Project references

- [Code map](CODE_MAP.md), [learning exercises](EXERCISES.md), [measurement protocol](docs/edge-measurement.md), and [coverage and limits](docs/coverage.md).
- [Original vision experiment](docs/edge-vision.md), [small-LM efficiency](docs/small-lm.md), and [service operations](docs/operations.md).
- [Agent evaluation](docs/agent-evaluation.md), [typed evidence agent](docs/evidence-agent.md), [failure analysis](docs/evidence-failure-analysis.md), and [synthetic telemetry](docs/security-telemetry.md).

Inherited negative results remain: the initial 135M agent failed its task gate; LoRA reduced answer-token NLL without improving exact match; small-LM INT8 worsened perplexity and increased post-load RSS. The typed agent's 88/96 result is limited to its synthetic suite and differs from text RAG in evidence access. Hosted-model inference, KQL execution, cloud deployment, customer impact and production capacity remain unverified. The upstream documentation reports a historical 43-test GitHub Actions run; that is separate from this workspace's checks. Dataset and model attribution remain in the linked guides; no model weights are included in the source snapshot.
