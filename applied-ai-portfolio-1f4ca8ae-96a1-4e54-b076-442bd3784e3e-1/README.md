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
