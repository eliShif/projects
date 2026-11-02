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

The current extension binds inventoried local model inputs to the SHA-256 identities
saved in the experiment protocol. Training and each base/adapted evaluation load
recheck those identities, including after queue delay, retry and resume. See the
[current layout and identity contract](docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.verified-fb95b96e2245.verified-6b6a9a9dcad3.md) and the separate
[repair validation record](results/model_identity_layout_repair/REPORT.md).
Earlier experiments and validation records remain in their existing directories.
The focused repair restores safe local symlink layouts and binds versioned
tokenizer files selected through `fast_tokenizer_files`. It reuses all existing
training, evaluation, retry and adapter-receipt checks.

The earlier layout review identified an uncovered `configuration_files` redirect.
The follow-up now binds advertised versioned configurations and tests actual
Transformers selection and mutation rejection at the affected loading boundaries.
It also accepts trusted shared-cache links and dot-relative shards, ignores unused
indexes for monolithic checkpoints, and preserves distinct symlink/parent shard
identity keys. R04 is demonstrated for the guarded Security Triage workflow;
direct `LocalLM` calls without expected identities remain unchecked.

The latest supplied worker validation passed 124 tests in 121.994 seconds,
built the 74,945-byte wheel and passed the finite reference smoke. The earlier
worker results of 122 tests in 157.346 seconds with a 74,807-byte wheel and
119 tests in 141.903 seconds with a 74,634-byte wheel are historical.
The latest suite includes actual-loader regressions for unused non-file `.jinja`
children, regular-file mutation rejection and strict closed-bundle helper checks.

The separate
[2 November 2026 isolated-copy selector/template verification](results/operator_selector_forms/verification.json)
records 122 passing combined tests and all 43 inherited HTTP/portfolio tests
passing in a separate command, including the previously blocked HTTP cases.
Those isolated-copy executions are separate from the three latest worker checks.
No separate inherited-suite rerun appears in the latest supplied worker manifest.
Tiny random-model loading, one-step fitting and evaluation verify software
integration; a full pretrained-security experiment remains unrun.

## Setup

Python 3.11+ is required. The default reference experiment uses only the standard
library. Packaging requires setuptools 77+; the recovered virtual environment
already contains it. These commands install the actual local package without
network access when the packaging dependency is available:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir --no-build-isolation --no-deps .
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

The recovered workspace already contains these ML dependencies. Its offline
installation check is reproducible with:

```sh
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir -r requirements.txt
```

In the restricted builder environment on 31 October 2026 (Python 3.14.4),
that ML installation failed because the package index supplied no installable
NumPy distribution. Earlier checkpoint attempts also recorded DNS failures.
PyTorch, NumPy and local model weights were unavailable in that builder run.
The subsequent worker installation record confirms CPU PyTorch `2.15.0+cpu`
and local package installation, and passed the original 44-test security suite,
wheel build and finite reference smoke. The current recovery environment also
contains Transformers `4.57.6` and safetensors `0.8.0`: all eight real model-identity
tests and both inherited tensor tests now pass. The earlier dependency failures
remain in [the previous validation record](results/model_identity_validation/REPORT.md).
A full pretrained-security research experiment remains unrun. The reference path
is separately implemented and does not silently stand in for the pretrained backend.

## Test, build and finite smoke

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests_security -v
.venv/bin/python scripts/test_model_identity.py
.venv/bin/python -m pip wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m portfolio.security_smoke
.venv/bin/python scripts/check_preservation.py
.venv/bin/python scripts/check_layout_preservation.py
```

The smoke command runs preparation, real reference-policy training, 72
held-out arm/case evaluations and the 32-job/four-worker service scenario. It
verifies replay and idempotent resume, then checks a hostile-note triage case
and stale-worker rejection. It removes its temporary
run and exits. It requires no socket, credentials, download or external service.
The security suite includes the new identity inventory, loading-boundary and
queue regressions. These offline tests use explicit loader doubles for model
boundaries; they do not execute tensor training. `scripts/test_model_identity.py`
additionally runs both real tensor suites; it requires the ML stack and treats
missing imports as errors. All tensor checkpoints are randomly initialized local
fixtures, with no downloads or hosted inference.

To run only the layout repair regressions:

```sh
.venv/bin/python -m unittest tests_security.test_model_layouts tests_ml_identity.test_checkpoint_layouts -v
```

These exercise directory aliases, internal artifact and shard links, escape and
retarget rejection, and actual Transformers versioned-tokenizer selection. The
selected tokenizer is changed after enqueue at training, base evaluation,
adapted evaluation and retry/resume boundaries. Rejected attempts must stop
before the affected loader and publish no successful receipt or metrics;
unchanged/restored fixtures must train and evaluate successfully.

`check_preservation.py` checks the normalized [original-source inventory](results/model_identity_validation/baseline.json),
which contains 141 authoritative entries, including 13 original test files and
81 original result files. Private worker output and generated packaging files
are excluded; every authoritative source entry is retained. It rejects
protected-file deletions or changes outside the seven permitted
implementation/documentation files. Root `.venv`, `build`, `dist` and `.egg-info`
output is disposable; reinstalling or rebuilding the package must not fail
preservation. The [packaging regression](tests_security/test_preservation_packaging.py)
also proves that source, tests and saved results remain protected.

The 148-entry inventory, 143 protected files and five generated entries in the
1 November 2026 recovery report describe the historical pre-publication workspace.
The repair adds a separate [235-file checkpoint inventory](results/model_identity_layout_repair/repair_baseline.json).
`check_layout_preservation.py` protects all 156 previous result files and all
17 supplied test files. The current preservation allowlist contains nine files;
the verifier and four root guides were the five permitted edits at the earlier
1 November 2026 layout-repair stage. The ephemeral project-root
`.report-cpu-constraints.txt` is excluded from source preservation, while
historical test and result files remain protected. Previous validation outputs
are retained without rewriting.

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
.venv/bin/python -m unittest discover -s tests_ml_identity -v
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
`torch` before test bodies executed. The current recovery rerun executes both
inherited tensor tests successfully; it does not run a pretrained research model.
Reference training exercised by tests and finite smoke verifies integration;
it does not establish pretrained-model quality.

The `tests_ml_identity` suite reuses the inherited tiny Llama checkpoint
builder. It tests real safetensors training, both evaluation loads, restoration
after failure, shard loading, and mutations with spies on the actual Transformers
loaders. Its randomly initialized checkpoints are test fixtures. On 1 November
2026, an earlier attempt failed importing `torch` before test bodies ran; the
package index supplied neither the requested NumPy distribution nor PyTorch.
Those failed installation logs remain intact. The preceding recovery rerun passed
its eight tests with real Transformers loaders, one-step LoRA training, both
evaluation arms and artifact replay. See the
[preceding validation record](results/model_identity_recovery_validation/REPORT.md).
The 1 November 2026 layout-repair outcomes remain in the
[repair validation record](results/model_identity_layout_repair/REPORT.md).
Loading-reference and nested-PEFT checks remain recorded in
[the earlier 2 November 2026 isolated-copy validation](results/operator_nested_peft/verification.json).
The historical [enumeration-contract verification](results/operator_enumeration_contract/verification.json)
records 119 passing tests and all 43 inherited HTTP/portfolio tests passing in a
separate command. The subsequent [selector/template verification](results/operator_selector_forms/verification.json)
records 122 passing combined tests and 43 passing inherited tests in separate
commands on 2 November 2026. These isolated-copy executions are separate from
the latest supplied 124-test worker run.

For an existing queued pretrained run, changing any bound file causes
`ModelIdentityError` before the affected loader. The attempt records only a
failure category and publishes no successful receipt or evaluation metrics.
Restore the exact original files to retry within the queue's three-attempt
budget, or create a new output directory for an intentional model change.
Do not edit `protocol.json` to accept replacement inputs.

The identity inventory covers supported safetensors, configuration/tokenizer
inputs, active shard indexes and referenced shards, including advertised
versioned tokenizer and configuration files. The guarded workflow supports
model-directory aliases and trusted shared-cache artifact links. Dot-relative
shards are supported, unused indexes do not prevent monolithic loading, and
distinct symlink/parent shard references retain separate identity keys. The
closed-bundle helper policy still rejects links outside its checkpoint boundary.
Guarded loading binds effective absolute/CWD-relative tokenizer overrides and,
when optional `peft==0.21.2` is active, the caller adapter and its actual local or
already-cached safetensors base. Both safetensors and supported legacy adapter
weights are covered, including same-directory PEFT with nested base shards.
Executed tests cover queued training, both evaluation arms, mutation rejection
and successful restoration/retry. The protocol records the PEFT version.
Arbitrary remote/custom-code tokenizer formats remain outside the verified
scope. This validation used no network model downloads.

Tokenizer and model `from_pretrained` calls each get a fresh check when expected
identities are supplied. These are content checks, not filesystem locks: a
concurrent writer can still race between verification and a library opening the
files. Use stable, trusted model inputs and protocol identities. Standalone
`PretrainedPolicy`/`train_pretrained` callers should pass `expected_model_files`
captured earlier when they need delayed-work protection. Without it, these APIs
capture identities on entry. Direct `LocalLM` calls without that map remain
unchecked. The queue always supplies its original protocol identities.

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
limits through the actual HTTP parser over in-memory byte streams. In the
restricted builder environment on 31 October 2026 (Python 3.14.4), four inherited
live HTTP tests failed at socket creation with `PermissionError`. The
1 November 2026 recovery and layout-repair records also retain 39 passes and
four socket-creation errors. Those outcomes describe their original environments.

The [2 November 2026 isolated-copy selector/template verification](results/operator_selector_forms/verification.json)
records all 43 inherited HTTP/portfolio tests passing, including the previously
blocked HTTP cases. This is separate execution evidence from the latest 124-test
worker run. The finite worker smoke reports `direct queue; no sockets`. The combined acceptance scenario recovers
one expired lease after reopening SQLite and rejects seven invalid requests,
five hostile fixture plans and three stale lease operations. These are bounded
correctness checks; throughput and production capacity remain unmeasured. To run
this scenario alone:

```sh
.venv/bin/python -m portfolio.security_acceptance --out results/local_security_acceptance
```

See [validation](docs/validation.verified-caa87f5df67b.verified-53af02ec2d3d.verified-01eb59e00e90.md) for executed checks and environment failures.

Collection completeness is a caller assertion. Temporal correlations do not
establish malicious intent. Hostile-note tests and operation restrictions do not
establish broad prompt-injection robustness. No cloud deployment, live incident
response, analyst-productivity improvement or production capacity was measured.

## Guides and retained experiments

- [Code map](CODE_MAP.md), [learning exercises](EXERCISES.md) and [requirement evidence](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.verified-bbfe3cea5f1e.verified-4e1bef13c95a.verified-2766e7764ba1.md).
- [Current identity layout contract and requirement mapping](docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.verified-fb95b96e2245.verified-6b6a9a9dcad3.md), with the [earlier loading-boundary record](docs/model-identity.verified-b82ad070789c.md) retained separately.
- [Draft technical note](docs/security-whitepaper.verified-1bfb5a839d56.verified-15187ad48f78.md) maps the security paper to the experiment.
- [Original agent evaluation](docs/agent-evaluation.md): reported retrieval exact match 19.44%, direct/tool-agent exact match 0%; concurrent CPU work confounded initial timings.
- [Original LoRA report](results/adapter_tuning/REPORT.md): reported answer NLL 6.153 → 0.521 while exact match remained 33.3%. Its gold-document greedy setup differs from stochastic RAG.
- [Typed Qwen evaluation](results/evidence_agent_test/REPORT.md): retained upstream results, 88/96 grounded correct versus 42/96 text RAG. Structured-corpus versus top-six-document access compares whole systems; six held-out entities and shared templates limit generalization.
- [Vision](docs/edge-vision.md), [small-LM efficiency](docs/small-lm.md) and [telemetry](docs/security-telemetry.md) retain negative latency, perplexity, memory and detection findings. KQL, cloud deployment and power measurements remain unverified.

These inherited research experiments were not rerun in this environment. The
two security experiment directories were checked by artifact replay; that audit
does not rerun model inference.
The earlier README links a 43-test [upstream CI run](https://github.com/eliShif/applied-ai-portfolio/actions/runs/39810558301)
from 22 October 2026; that historical result does not validate this extension.

<!-- verified-documentation:start -->
## Current verified documentation

- [VALIDATION.md](VALIDATION.verified-edcbabe8ade5.md)
- [docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.verified-bbfe3cea5f1e.verified-4e1bef13c95a.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.verified-bbfe3cea5f1e.verified-4e1bef13c95a.verified-2766e7764ba1.md)
- [docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.verified-bbfe3cea5f1e.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.verified-bbfe3cea5f1e.verified-4e1bef13c95a.md)
- [docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.verified-fb95b96e2245.md](docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.verified-fb95b96e2245.verified-6b6a9a9dcad3.md)
- [docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.verified-bbfe3cea5f1e.md)
- [docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.md](docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.verified-fb95b96e2245.md)
- [docs/model-input-layouts.verified-9feedecf033e.verified-06a534f46b96.verified-17af42a40b09.md](docs/model-input-layouts.verified-9feedecf033e.verified-06a534f46b96.verified-17af42a40b09.verified-b5764ab14405.md)
- [docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.verified-38cb251a2bed.md)
- [docs/model-input-layouts.verified-161f084d4d58.md](docs/model-input-layouts.verified-161f084d4d58.verified-aa197a1dfeb9.md)
- [docs/model-input-layouts.verified-9feedecf033e.verified-06a534f46b96.md](docs/model-input-layouts.verified-9feedecf033e.verified-06a534f46b96.verified-17af42a40b09.md)
- [docs/operations.verified-e3c443ea8c3d.md](docs/operations.verified-e3c443ea8c3d.verified-7f3437750462.md)
- [docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.verified-6189e3ede380.md)
- [docs/model-input-layouts.md](docs/model-input-layouts.verified-161f084d4d58.md)
- [docs/model-input-layouts.verified-9feedecf033e.md](docs/model-input-layouts.verified-9feedecf033e.verified-06a534f46b96.md)
- [docs/operations.md](docs/operations.verified-e3c443ea8c3d.md)
- [docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.verified-572edc0a504c.md)
- [docs/validation.verified-caa87f5df67b.verified-53af02ec2d3d.md](docs/validation.verified-caa87f5df67b.verified-53af02ec2d3d.verified-01eb59e00e90.md)
- [docs/coverage.verified-e0b29a65529a.md](docs/coverage.verified-e0b29a65529a.verified-eb319bb4e93f.md)
- [docs/model-identity.md](docs/model-identity.verified-b82ad070789c.md)
- [docs/security-whitepaper.verified-1bfb5a839d56.md](docs/security-whitepaper.verified-1bfb5a839d56.verified-15187ad48f78.md)
- [docs/validation.verified-caa87f5df67b.md](docs/validation.verified-caa87f5df67b.verified-53af02ec2d3d.md)
- [VALIDATION.verified-43a6c0f12ca1.md](VALIDATION.verified-43a6c0f12ca1.verified-28cc84b19773.md)
- [docs/coverage.md](docs/coverage.verified-e0b29a65529a.md)
- [docs/security-whitepaper.md](docs/security-whitepaper.verified-1bfb5a839d56.md)
- [docs/validation.md](docs/validation.verified-caa87f5df67b.md)
<!-- verified-documentation:end -->
