# Implementation coverage and limits

The extension retains Python, PyTorch, NumPy, scikit-learn, SQLite, the existing HTTP service and unittest. Requirement identifiers below index technical behavior. Execution status is detailed in [VALIDATION.md](../VALIDATION.md).

| IDs | Implemented behavior | Implementation | Tests | Current evidence boundary |
|---|---|---|---|---|
| R02, R04, P01 | Original normalized digits data, fixed stratified splits, teacher/control/student training and evaluation | `portfolio/edge_vision.py` | `tests/test_edge_models.py` | Inherited results retained; worker model checks and one-seed, two-epoch isolated integration passed; full three-seed experiment remains unverified |
| R03 | Executable Python orchestration and strict configuration validation | `portfolio/edge_benchmark.py`, `portfolio/edge_protocol.py` | `tests/core/test_edge_protocol.py`, `tests/core/test_edge_runner.py` | Protocol and worker-failure checks passed |
| R05 | Transactional retries, concurrent conflicts, stale-owner rejection and bounded queue/HTTP load | `portfolio/reliable_service.py`, `portfolio/service_load.py` | `tests/test_service.py`, `tests/core/test_queue_load.py` | Direct SQLite load measured; worker HTTP and concurrent HTTP-load checks passed; representative inference load, availability, customer impact and production capacity remain unverified |
| R06 | Raw timings, process isolation, balanced scheduling, runtime metadata and OS peak memory | `portfolio/edge_benchmark.py`, `portfolio/measurement.py` | `tests/core/test_measurement.py`, `tests/core/test_edge_runner.py`, `tests/test_edge_models.py` | Host model timing and Linux memory API checks passed; full new three-seed benchmark remains unverified; no edge-target or power measurements; Windows/macOS native paths unrun |
| R07, P02 | Existing KL/hard-label distillation and actual qint8 Linear kernels | `portfolio/edge_vision.py`, `portfolio/edge_benchmark.py` | `tests/test_edge_models.py` | Real-model checks passed in the worker, including distillation, teacher preservation, checkpoint restoration, actual qint8 layers and isolated pipeline/artifact integrity |
| R08, I01 | Save validation-only decision before test; reject incomplete/changed trial sets | `portfolio/edge_protocol.py`, `portfolio/edge_benchmark.py` | `tests/core/test_edge_protocol.py`, `tests/test_edge_models.py` | Validation-only selection checks and one-seed, two-epoch isolated pipeline passed; full new three-seed experiment remains unverified |
| R10 | Reports explain selection/rejection, quality/resource constraints and local load limits | `portfolio/edge_benchmark.py`, `portfolio/service_load.py` | `tests/core/test_edge_protocol.py`, `tests/core/test_queue_load.py` | Direct-store queue report measured; model integration passed; a model recommendation from the full new benchmark remains unverified; inherited negative tradeoffs preserved |
| P03, I02 | Source/data/model hashes, command logs, version metadata and failed-run lifecycle | `portfolio/measurement.py`, `portfolio/edge_benchmark.py` | `tests/core/test_measurement.py`, `tests/core/test_edge_runner.py`, `tests/test_edge_models.py` | Lifecycle/manifest checks passed; hashes identify contents, not training history or independent replication |
| I03 | Numerical, malformed-input, concurrent-conflict, lease, artifact and worker-failure tests | `tests/core/`, `tests/test_edge_models.py`, `tests/test_service.py` | Same executable test paths | Full worker suite passed 75 tests in 139.882 seconds, including model, HTTP, concurrency, lease, artifact and worker-failure checks |

## Preserved experimental findings

- Original agent evaluation: 108 generations; retrieval exact match 19.44%, direct/tool policies 0%. Retrieval found the expected source, so generation and action selection remained bottlenecks. Concurrent CPU work limits the original timing comparison.
- LoRA: 15,360 trainable parameters, 48 updates; test NLL 6.153 to 0.521 while exact match stayed 33.3%. Gold-document greedy decoding is not causally comparable to stochastic top-three RAG.
- Typed agent: 88/96 correct with expected evidence versus 42/96 for text RAG. Shared task families, six test-entity clusters, different evidence access and resumed execution limit generalization and timing interpretation. The source-ID score is not full claim-level groundedness.
- Vision: inherited INT8 reduces storage but increases batch-1 latency. Three seeds use one test set. This small digits task does not establish a universal distillation benefit or advanced video performance.
- Small LM: INT8 worsens authored-text perplexity and increases post-load RSS despite smaller state and faster inference. Six passages are not a standard language benchmark; post-load RSS is not peak memory.
- Synthetic telemetry: rules miss slow events and flag legitimate maintenance; Isolation Forest improves recall with more false positives. Actor holdout does not remove shared scenario-template bias.

## Integrations and remaining experiments

Hosted-model adapters have contract tests only; no hosted inference score is available. KQL translations and native cloud-event ingestion have not been executed. Cloud hosting, Kubernetes operation, TLS termination, quotas, backups, schema migrations and a supervised long-running worker remain separate engineering work. The existing service is a single-workspace SQLite prototype with bounded attempts, no heartbeat, and at-least-once processing; external side effects would require independent idempotency or an outbox.

The upstream record reports a 43-test GitHub Actions run, while this extension has only the local receipts linked above. Deployment and production load were not performed. Target-device measurements, real power instrumentation, realistic arrival rates, representative model tasks and independent replication remain necessary before deployment decisions. No local timing or queue result establishes customer impact.
