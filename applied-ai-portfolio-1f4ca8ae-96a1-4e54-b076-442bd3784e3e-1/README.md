# applied-ai-portfolio

Compact models and AI services require measured tradeoffs between task accuracy, latency, storage, memory and reliable execution. Smaller files or lower training loss can conceal slower inference, higher memory use or unchanged answer accuracy. Concurrent submissions and worker failures also require consistent job handling.

## What it does

An applied-AI portfolio combining model-compression benchmarks, typed evidence retrieval, synthetic telemetry evaluation and a durable SQLite/HTTP job queue. Completed experiments document quality, latency, storage and memory tradeoffs; host validation verifies real digit-model integration, artifact integrity, concurrent retries and lease fencing.

Compare teacher, supervised-control, distilled and dynamic INT8 classifiers using validation quality and resource budgets before held-out evaluation. Evaluate retrieval, typed execution, adapter tuning and synthetic telemetry with explicit scoring. Retain raw measurements and traceable artifacts, and verify queue idempotency, conflict rejection, lease fencing and retry recovery under bounded concurrent load.

## Solution and technologies

Python, NumPy, scikit-learn and PyTorch train classifiers on 1,797 normalized UCI digit images using fixed stratified training, validation and test partitions. The extended benchmark launches fresh worker processes, balances measurement order, restores validation-selected checkpoints, records raw CPU timings and whole-process peak memory, verifies artifact hashes, and saves model selection before test evaluation.

Language-model components provide TF-IDF retrieval, bounded tool use, frozen-base LoRA and isolated FP32/INT8 comparisons. A Qwen3-4B planner produces typed operations executed against versioned evidence. Synthetic telemetry compares temporal rules with Isolation Forest. The service combines SQLite WAL transactions, idempotency keys, fenced leases, bounded retries and authenticated HTTP routes. Protocols, logs, manifests and reports connect configuration to recorded outputs.

Python · PyTorch · NumPy · scikit-learn · SQLite · HTTP · unittest

## Quick start

```text
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
```

```sh
.venv/bin/python -m portfolio.smoke
```

## Results

Recorded extension measurements from 2026-10-31 cover two direct-SQLite runs. Each processed 72 submissions for 24 unique jobs using four threads, rejected 24 expected content conflicts and recorded zero unexpected errors. All recorded invariants passed; store_load_verified additionally records receipt-content preservation after conflicts.

| Report | Enqueue p50 / p95 (ms) | Submission operations/s | Unique enqueue jobs/s | Separate drain jobs/s |
|---|---:|---:|---:|---:|
| results/store_load/REPORT.md | 0.462 / 10.217 | 828.66 | 276.22 | 193.75 |
| results/store_load_verified/REPORT.md | 1.482 / 18.820 | 1,065.03 | 355.01 | 253.06 |

These are short receipt-processing workloads with separate enqueue and drain phases. Later worker HTTP tests passed, but no standalone HTTP performance report is supplied.

The completed original digits experiment, recorded on 2026-10-22 with three seeds and 100 epochs, reports:

| Variant | Mean test accuracy | Mean state KiB | Median batch-1 CPU latency ms |
|---|---:|---:|---:|
| Teacher FP32 | 97.87% | 39.7 | 0.0105 |
| Supervised student FP32 | 97.41% | 9.3 | 0.0098 |
| Distilled student FP32 | 97.59% | 9.3 | 0.0100 |
| Dynamic INT8 student | 97.69% | 5.4 | 0.0312 |

INT8 reduced serialized storage while increasing latency. These inherited measurements are separate from the later integration checks and do not establish a recommendation for the full extended benchmark.

Other completed inherited experiments document the following outcomes:

- Initial agent evaluation: retrieval exact match was 19.44%; direct and tool-agent policies scored 0%. Retrieval recovered the expected source in every retrieval run, identifying generation and action selection as remaining bottlenecks.
- LoRA adaptation: held-out answer-token NLL fell from 6.153 to 0.521, while exact match remained 33.3% on 12 cases.
- Typed evidence agent: 88/96 answers were correct with expected evidence, versus 42/96 for text RAG. Correct abstentions were 36/36 versus 18/36, with no invalid/failed outputs for either system. The typed system passed its predefined synthetic-task gate; six outputs reversed operands and two misinterpreted a service name.
- SmolLM2-135M-Instruct: dynamic INT8 reduced state size from 513.2 to 236.6 MiB and increased generation from 14.88 to 47.23 tokens/s. Authored-text perplexity worsened from 120.26 to 258.58, and post-load RSS increased from 960.7 to 1540.6 MiB.
- Synthetic telemetry: Isolation Forest increased recall from 73.33% to 93.33%, but increased false positives from 7.78% to 21.11% and reduced F1 from 0.746 to 0.727. Temporal rules missed slow events and flagged legitimate maintenance.

Inherited loopback HTTP acceptance from 2026-10-22 completed actual local-model research and synthetic-triage jobs with evidence and traces. It verified workflow, authorization and duplicate handling; generated factual accuracy was not certified.

## Scope and limitations

A completed full new three-seed isolated benchmark, its resource-gated model recommendation and replacement performance measurements are not evidenced. The original three-seed experiment is complete; the later integration run is a separate execution check.

Target-edge performance, physical power/energy measurements and native Windows/macOS execution remain unverified. Host CPU and Linux memory checks do not establish those results. Background load and thermal conditions were uncontrolled; serialized size, post-load RSS and whole-process peak memory measure different resources.

Queue rates describe short, closed-loop direct-store receipt workloads with separate enqueue/drain phases. They do not establish representative HTTP capacity, inference throughput, saturation, end-to-end client latency or customer impact. Controlled-clock lease probes do not establish process-crash or power-loss durability.

The service remains a single-workspace SQLite prototype with at-least-once processing, bounded attempts and no lease heartbeat or supervised long-running worker. TLS, quotas, backups and schema migrations remain deployment work; external side effects would require independent idempotency or an outbox.

Digits seeds share one test set; the small-language-model study uses six authored passages; adapter results use one seed and 12 test cases. Gold-document greedy adapter evaluation is not causally comparable with stochastic retrieval evaluation. Artifact hashes support content traceability, not independent replication.

Agent and telemetry results concern authored or synthetic tasks with shared task/scenario families. The typed-agent comparison uses six held-out entity clusters and different evidence access from RAG. Evidence-ID scoring is not complete claim-level groundedness or broad injection-robustness evidence. Concurrent CPU work and interrupted/resumed execution limit agent timing comparisons.

Hosted-model inference, KQL execution, native cloud-event ingestion, Kubernetes operation, cloud deployment and production availability remain unverified. The inherited GitHub Actions record is separate from later local worker validation.

The supplied report references earlier lease-handling and failure-evidence concerns without detailed findings or a closure record. Passing the documented cases does not establish resolution of unspecified concerns.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.

## Additions in this extension

# applied-ai-portfolio (classical vision and signal-processing extension)
Classical image and signal-processing algorithms (filtering, edge detection, spectral analysis) are easy to get subtly wrong at borders, on flat inputs and across implementations. A second implementation in another language is only credible if it is checked against a reference on boundary cases and malformed input. This extension implements these algorithms in NumPy and in C++17 and verifies the two against each other with a finite, reproducible check.
Classical vision and signal-processing extension of the applied-ai portfolio. It has a NumPy reference (2-D correlation, Gaussian blur, Sobel edges, FIR, FFT, STFT, synthetic ROI time-activity curves) and C++17 kernels behind a validated text protocol, called from Python without a shell. A seeded benchmark checks C++ against NumPy at 1e-9 tolerance and records raw timings. On the host, the full discovery run (121 tests) and the C++ build passed. A one-repeat smoke run on synthetic data matched C++ to NumPy within 3.4e-14 and gave NumPy-pipeline edge F1 1.0 at 2 px tolerance. Scope is small synthetic data. The C++ is unoptimized, timings are not a ranking, and there is no video, detection or PET/fMRI evidence.
1) Implement 2-D cross-correlation, Gaussian blur, Sobel gradients and a thresholded edge pipeline, scored against synthetic ground truth. 2) Implement windowed-sinc FIR design and filtering, FFT, STFT, dominant-frequency detection and a synthetic ROI-versus-background time-activity comparison. 3) Provide a C++17 kernel program behind a validated stdin/stdout text protocol, called from Python without a shell. 4) Check C++ against NumPy within a stated tolerance (1e-9) and record raw timing samples with a documented timing method. 5) Add new source, test and result-free modules without modifying existing source, test or result files. Documentation sections were only appended.
portfolio/classical_vision.py is the NumPy reference. It covers zero-padded odd-kernel correlation, Gaussian kernel, Sobel magnitude, a relative-threshold edge map with a flat-image floor, an edge pipeline that edge-replicates the image before filtering and crops the margin afterwards, edge precision/recall/F1 with pixel tolerance, causal FIR filtering, windowed-sinc low-pass design, power-of-two real FFT, Hann STFT magnitude, dominant frequency, seeded synthetic generators and ROI time-activity curves.
cpp/filters.cpp holds C++17 kernels for direct correlation, Sobel, FIR, iterative radix-2 FFT and STFT. They sit behind a text protocol that checks sizes, rejects non-finite, truncated and trailing input, bounds repeat counts, exits with code 2 on rejection and reports its own compute-loop time.
portfolio/classical_native.py finds or compiles the binary with g++, runs it without a shell and with a timeout, validates output shapes and exposes a NATIVE backend with the same interface as the NumPy backend. Both backends receive the same padded array, so edge maps are identical.
portfolio/classical_benchmark.py is a seeded finite CLI check. It compares C++ and NumPy within tolerance, scores edge quality, records raw timings and exits nonzero on any failed check. It reuses the existing RunRecord, distribution and environment helpers, and with --out it records a protocol, source hashes and a manifest in a new empty directory. If g++ and the binary are both absent it falls back to NumPy-only checks unless --require-native is given.
Data flow: synthetic_image / synthetic_signal → NumPy reference and (optionally) C++ → tolerance comparison and timing distributions → console JSON or a recorded run directory.
Python · NumPy · SciPy (reference checks in tests) · C++17 · g++ · unittest
.venv/bin/python -m pip install --no-cache-dir 'torch>=2.9,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m portfolio.classical_benchmark --require-native --repeats 1
All results below come from the latest worker run on the host, with a single repeat and seed 11.
Smoke command (`--require-native --repeats 1`; exit 0, all_passed true, native_executed true):
- Data: synthetic 64x64 image (rectangle and disc, noise sigma 0.02) and a 256-sample signal at 256 Hz (12 Hz amplitude 1.0 plus 40 Hz amplitude 0.5, noise sigma 0.05).
- Edge quality, NumPy pipeline against synthetic ground truth: precision 1.0, recall 1.0, F1 1.0 at 2 px tolerance (563 predicted pixels, 148 truth pixels). This is one image and not a general accuracy estimate. The predicted-pixel count is larger than the truth count because scoring is tolerance-based.
- Dominant frequency: 12.0 Hz from both NumPy and C++.
- C++ versus NumPy equivalence (tolerance 1e-9): max absolute difference 0.0 for 5x5 Gaussian correlation, 0.0 for Sobel magnitude, 6.7e-16 for 63-tap FIR, 3.4e-14 for 256-point rFFT and 7.1e-15 for STFT (frame 64, hop 32). All are within tolerance, and the edge maps are identical.
- Timings (one sample each, so p50 equals p95; NumPy is one call, C++ wall time includes subprocess start and text parsing, kernel time is the in-program compute loop). The values are in milliseconds, given as NumPy / C++ wall / C++ kernel:
  - correlation 0.435 / 21.458 / 0.427
  - Sobel 0.617 / 19.353 / 0.278
  - FIR 4.816 / 4.318 / 0.017
  - rFFT 0.0745 / 3.566 / 0.047
  - STFT 0.284 / 4.298 / 0.156
C++ wall time was higher than NumPy for four of five operations, which mostly reflects process start and text I/O. These timings are not an algorithm or language comparison. No result files are shipped.
Integration status: the full discovery run (121 tests) passed. No new training experiment was run, and no model or performance numbers are inherited into this card.
Data are seeded synthetic images (64x64) and signals (256 samples); there is no video, object detection, camera geometry, driving or clinical data.
The ROI time-activity comparison is a synthetic stand-in and not a PET or fMRI pipeline.
The edge F1 result is a single smoke case (seed 11, one image). The F1 >= 0.75 criterion applies only to the synthetic shapes and noise levels in the tests.
The C++ kernels are direct, single-threaded and unoptimized, with no SIMD and no production-C++ claim. Timings come from one repeat on the host under uncontrolled background load, and NumPy time, C++ wall time and C++ kernel time measure different things. They are not a performance or language ranking.
The FFT and STFT require power-of-two lengths, kernels must be odd, and the text protocol may reject subnormal-range numbers.
Without g++ and a prebuilt binary the native tests are skipped and the benchmark runs NumPy-only checks, reporting native_executed false unless --require-native is set.
No measurement from this extension is shipped with the code, so timings must be reproduced by running the command. Target-device, power and native Windows/macOS measurements were not made.

### Native build check (no stored binary)

The C++ program is compiled into a temporary directory on first use; no binary is kept in the repository, and the statements above about a prebuilt binary are superseded. To verify the compiler path and remove any leftover `cpp/classical_filters.bin` from earlier revisions:

```
.venv/bin/python -m portfolio.classical_native
```

It prints `native build ok ...` and exits 0, or exits nonzero when `g++` is missing or compilation fails. Tests: `.venv/bin/python -m unittest discover -s tests -v` (includes `tests/test_classical_artifacts.py`).

## Current verified documentation (this version)

- [README.md](README.verified-66cc6aca0583.md)
- [CODE_MAP.md](CODE_MAP.verified-fdfa76eca15f.md)
- [EXERCISES.md](EXERCISES.verified-21242e3be84c.md)

## Current verified documentation (this version)

- [VALIDATION.md](VALIDATION.verified-a5825e3274d3.md)

## Current version

The current version of this document is [README.verified-4d79712c9997.md](README.verified-4d79712c9997.md).
