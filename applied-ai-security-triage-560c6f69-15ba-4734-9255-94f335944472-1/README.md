# Applied AI: bounded security triage

A Python extension of the pinned Applied AI Portfolio that connects structured
alert investigation, low-rank adaptation and reproducible evaluation through the
existing SQLite job queue. It reuses versioned evidence execution, telemetry
normalization, local model loading and PyTorch LoRA. [Source provenance](UPSTREAM.md).

The workflow retrieves events within an alert's actor/time scope, reads versioned
policy, validates a planner's procedure, computes a finding and cites its sources.
It abstains on missing or conflicting evidence. It exposes no response actions.
Preparation, training and evaluation are separate durable jobs with bounded
retries, renewable leases and fenced artifact publication. Evaluation also saves
a combined service acceptance record covering request validation, concurrent
workers, duplicate requests, lease recovery and adversarial plan fixtures.

## Setup

Python 3.11+ is required. The default reference experiment uses only the standard
library. Packaging requires setuptools 77+; the supplied environment has it in
system site packages. These commands install the actual local package without
network access:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-cache-dir --no-build-isolation --no-deps .
```

On a clean machine without setuptools, install the packaging dependency first:

```sh
.venv/bin/python -m pip install "setuptools>=77"
```

To install the existing ML stack for PyTorch, local pretrained models and the
inherited experiments, use the real dependency file:

```sh
.venv/bin/python -m pip install -r requirements.txt
```

In the restricted builder environment on 31 October 2026 (Python 3.14.4),
that ML installation failed because the package index supplied no installable
NumPy distribution. Earlier checkpoint attempts also recorded DNS failures.
PyTorch, NumPy and local model weights were unavailable in that builder run.
The subsequent worker installation record confirms CPU PyTorch `2.15.0+cpu`
and local package installation. It does not establish installation of the full
ML stack or availability of pretrained weights. The worker passed the 44-test
security suite, wheel build and finite reference smoke; no successful optional
tensor suite or pretrained security experiment is supplied. The reference path
is separately implemented and does not silently stand in for the pretrained backend.

## Test, build and finite smoke

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests_security -v
.venv/bin/python -m pip wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m portfolio.security_smoke
```

The smoke command runs preparation, real reference-policy training, 72
held-out arm/case evaluations and the 32-job/four-worker service scenario. It
verifies replay and idempotent resume, then checks a hostile-note triage case
and stale-worker rejection. It removes its temporary
run and exits. It requires no socket, credentials, download or external service.

## Run and inspect a persistent experiment

```sh
.venv/bin/python -m portfolio.security_pipeline run --out results/local_security_run --steps 96
.venv/bin/python -m portfolio.security_pipeline audit --out results/local_security_run
```

Repeat the same `run` command to resume or reuse completed stages. A source,
configuration, dependency-version or model change requires a new output directory.
The documented run produced the retained [research report](results/security_triage_v2/REPORT.md),
[protocol](results/security_triage_v2/protocol.json) and [manifest](results/security_triage_v2/manifest.json).
The manifest locates raw decisions, failures, service acceptance, checkpoint
history and data by hash. Audit checks successful and rejected plans, execution
traces, service outcomes and the generated report against those artifacts.
Do not replace inherited experiment directories with new runs.

The reference backend is a small seeded linear policy with a trained rank-2
update, **not an LLM**. It compares a deterministic rule, the unadapted random
reference policy and its adapted version. All arms receive the same label-free
observations and use the same evidence executor. Checkpoint selection uses only
training/validation cases. The synthetic dataset contains 32 training, 16
validation and 24 test cases with disjoint actors/events and shared task templates.
In the recorded 96-step run, the adapted policy achieved 24/24 grounded-correct
outcomes, tying the rule baseline; the random base achieved 6/24 with 18 rejected
plans. This measures procedure selection on these templates, not threat detection
or LLM quality. The earlier checkpoint's results remain in `results/security_triage`.

## Optional local pretrained experiment

Supply an explicit local Llama-style causal-LM directory with tokenizer, config
and safetensors. Loading is local-only, remote model code is disabled, and there
is no hosted-provider path in this workflow:

```sh
.venv/bin/python -m portfolio.security_pipeline run --backend pretrained --model-path models/SmolLM2-135M-Instruct --steps 96 --out results/local_pretrained_security
.venv/bin/python -m unittest discover -s tests_ml -v
.venv/bin/python -m unittest discover -s tests -v
```

The pretrained branch trains query/value LoRA in the last two decoder layers,
using the existing PyTorch layer and masked answer-token loss. It stores adapters
in safetensors and selects a checkpoint by validation NLL, including step zero.
Inference compares three permitted plans by answer-token NLL; this is constrained
candidate selection, not free-form generation. Missing dependencies or weights
cause an explicit error. The optional tensor tests use a locally initialized tiny
Llama and are distinct from pretrained-model evaluation. In the restricted
builder environment on 31 October 2026, the tensor suite failed importing
`torch` before test bodies executed. The later worker installed CPU PyTorch
`2.15.0+cpu`, but supplied no tensor-suite rerun or pretrained security experiment.
Reference training exercised by tests and finite smoke verifies integration;
it does not establish pretrained-model quality.

## Service and safety boundaries

`SecurityTriageEngine` accepts `{mode: "security_triage", alert: {...}}` through
`JobStore` and `work_once`. The existing HTTP handler accepts the same payload;
internal pipeline stages cannot be submitted through that API. `JSONPlanner`
connects the existing LocalLM/GGUF interfaces to the bounded executor. The older
question-based triage route remains a prose summary requiring review.

Planners receive separate evidence copies; changing a planner's copy cannot alter
the executor's evidence or another evaluation arm. JSON boundaries reject duplicate
keys, nonfinite numbers and excessive nesting. Request bodies must match their
declared length. These are data-integrity controls, not isolation of arbitrary
Python code loaded into the process.

The test suite and recorded service scenario exercise authentication and body
limits through the actual HTTP parser over in-memory byte streams. In the restricted builder environment on 31 October 2026 (Python 3.14.4),
four inherited live HTTP tests failed at socket creation with `PermissionError`.
No successful live-socket rerun is supplied by the subsequent worker; its smoke
reports `direct queue; no sockets`. These records do not establish transport
verification. The combined scenario recovers one expired lease after
reopening SQLite and rejects seven invalid requests, five hostile fixture plans
and three stale lease operations. These are bounded local correctness checks,
not throughput or production-capacity measurements. To run this scenario alone:

```sh
.venv/bin/python -m portfolio.security_acceptance --out results/local_security_acceptance
```

See [validation](docs/validation.md) for executed checks and environment failures.

Collection completeness is a caller assertion. Temporal correlations do not
establish malicious intent. Hostile-note tests and operation restrictions do not
establish broad prompt-injection robustness. No cloud deployment, live incident
response, analyst-productivity improvement or production capacity was measured.

## Guides and retained experiments

- [Code map](CODE_MAP.md), [learning exercises](EXERCISES.md) and [requirement evidence](docs/coverage.md).
- [Draft technical note](docs/security-whitepaper.md) maps the security paper to the experiment.
- [Original agent evaluation](docs/agent-evaluation.md): reported retrieval exact match 19.44%, direct/tool-agent exact match 0%; concurrent CPU work confounded initial timings.
- [Original LoRA report](results/adapter_tuning/REPORT.md): reported answer NLL 6.153 → 0.521 while exact match remained 33.3%. Its gold-document greedy setup differs from stochastic RAG.
- [Typed Qwen evaluation](results/evidence_agent_test/REPORT.md): retained upstream results, 88/96 grounded correct versus 42/96 text RAG. Structured-corpus versus top-six-document access compares whole systems; six held-out entities and shared templates limit generalization.
- [Vision](docs/edge-vision.md), [small-LM efficiency](docs/small-lm.md) and [telemetry](docs/security-telemetry.md) retain negative latency, perplexity, memory and detection findings. KQL, cloud deployment and power measurements remain unverified.

These inherited records were preserved rather than rerun in this environment.
The earlier README links a 43-test [upstream CI run](https://github.com/eliShif/applied-ai-portfolio/actions/runs/39810558301)
from 22 October 2026; that historical result does not validate this extension.
