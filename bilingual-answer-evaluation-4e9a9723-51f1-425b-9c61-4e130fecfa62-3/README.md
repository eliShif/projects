# Applied AI Portfolio

## Junior AI Researcher adaptation: bilingual answer evaluation

`portfolio.bilingual_eval` adds an offline synthetic Hebrew/English evaluation to
the existing evidence scorer and TF-IDF retriever. Each language asks the same
three fact or abstention questions. A lexical retrieval baseline runs without
model weights or an API. Its output is a real baseline result, not a language
model result. Numeric correctness, abstention and expected citation rules are
identical in both languages. The saved runs identify errors separately by
language. This small synthetic suite does not establish general Hebrew ability
or semantic search; TF-IDF here matches tokens.

With an existing local GGUF weight file, `--model` runs real CPU inference on
each paired question. The model and baseline receive the same top-one TF-IDF
record; the model must return a numeric value (or null) and record citations.
The run saves raw model responses, retrieval IDs, failures, model file hash,
fixed seeds, protocol, per-language comparison and concrete error cases. This
compares two answer systems on six authored cases; it cannot isolate a language
effect or establish general model quality. No model weights are included, and
the baseline smoke run is not a model result.

`retrieval_trace.json` records the top-one record shown for each case.
`case_diagnostics.json` pairs baseline and evaluated outcomes and flags when a
required source never reached the answerer. For abstention questions, source
sufficiency is `null`: a single retrieved record cannot prove that a field is
absent from the whole corpus. A local model run also creates
`review_queue.json` with raw responses and `human_review: pending`; a reader
must inspect these before claiming any human validation. For supplied answers,
the trace describes this evaluator's retrieval, not independently verified
context from the system that produced the supplied file.
`paired_audit.json` joins each English/Hebrew question pair, records outcome
and retrieval disagreements, and counts observable failure causes by language.
It rejects unpaired or answer-inconsistent cases. A disagreement is a case
finding, not evidence that language caused the difference.

Run from the repository root with Python 3.11+:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-bilingual.txt
.venv/bin/python -m unittest discover -s tests -p test_bilingual_eval.py -v
.venv/bin/python -m compileall -q portfolio/bilingual_eval.py
.venv/bin/python -m portfolio.bilingual_eval --out /tmp/bilingual-eval
```

For an optional local model run, install the existing GGUF runtime with
`.venv/bin/pip install -r requirements-gguf.txt`, then run
`.venv/bin/python -m portfolio.bilingual_eval --model /path/to/model.gguf --out /tmp/bilingual-model-eval`.
The command requires real weights on disk and fails if they are absent. It
does not download weights or call an API.

The last command finishes and writes `protocol.json`, `corpus.json`,
`cases.json`, `predictions.json`, `runs.json`, `summary.json`,
`baseline_predictions.json`, `comparison.json`, `errors.json`,
`retrieval_trace.json` and `case_diagnostics.json` under the chosen
output directory, plus `paired_audit.json`. To score externally saved structured answers, supply a JSON array with one
entry per `case_id` from `cases.json` using `--predictions path/to/answers.json`.
Each entry has `case_id`, `status` (`answered`, `abstained` or `failed`), numeric
`value` or `null`, and `citations` containing corpus record IDs. Invalid or
incomplete prediction sets fail. Supplied predictions have no independently
verified model provenance; use `--model` for a directly executed local run.
See `CODE_MAP.md` and `EXERCISES.md` for code
navigation and learning exercises. AI-assisted code and its tests do not prove
personal mastery.

Five runnable, AI-assisted portfolio projects, selected **per Microsoft opening**.
These are new personal projects, not employment history, a publication, or claims
of prior production deployment. Run records distinguish real models, synthetic
datasets, unit-test fixtures and unexecuted infrastructure integrations.

| Opening | Project 1 | Project 2 |
|---|---|---|
| 200062584 - Data Science: LLMs and agents | `agent_eval` - controlled LLM/agent experiments | `reliable_service` - retrieval, failure recovery and traceable AI workflows |
| 200047371 - Software Engineering: AI | `reliable_service` - typed HTTP API, durable jobs and observability | `agent_eval` - tools, retrieval, quality gates and regression evaluation |
| 200062523 - Data Science: efficient models | `edge_vision` - training, distillation and INT8 vision inference | `small_lm` - language-model quality, latency, memory and quantization |
| 200028527 - Security Research | `telemetry` - detection engineering and evaluation | `reliable_service` security-triage adaptation - bounded evidence retrieval and auditable investigations |

See `docs/coverage.md` for requirement-by-requirement coverage and limitations.
Project guides, executable entry points and saved results are linked below. The
test suite covers queue/HTTP correctness, statistics, detection invariants,
retrieval safety and adapter gradients. Vision/LM performance uses actual runs.

| Project | Guide | Evidence |
|---|---|---|
| Agent Evaluation & Adaptation | [Typed-agent design](docs/evidence-agent.md) | [Held-out typed-agent evaluation](results/evidence_agent_test/REPORT.md), [original baseline](results/agent_eval/REPORT.md), [fine-tuning](results/adapter_tuning/REPORT.md) |
| Reliable Research Service | [Guide](docs/operations.md) | [Real-model HTTP acceptance](results/service/acceptance.json) |
| Edge Vision Lab | [Guide](docs/edge-vision.md) | [Three-seed results](results/edge_vision/REPORT.md) |
| Small-LM Efficiency Lab | [Guide](docs/small-lm.md) | [Measured tradeoffs](results/small_lm/REPORT.md) |
| Security Telemetry Lab | [Guide](docs/security-telemetry.md) | [Held-out detection results](results/telemetry/REPORT.md) |

**Readiness:** local prototypes and experiments run. The original small-model
agent failed its usefulness bar; a new typed agent separates planning from
evidence execution (see `docs/evidence-agent.md`). Hosted-model integration has contract tests only:
no API credential was available, so no frontier-model score is claimed. KQL and cloud deployment remain unverified. The 43-test suite also passed in [GitHub Actions](https://github.com/eliShif/applied-ai-portfolio/actions/runs/39810558301) on 22 October 2026. See the coverage matrix.

## Current agent iteration

Versioned evidence, dimensional checks, safe abstention and schema-constrained
Qwen3-4B CPU inference. [Design and reproduction](docs/evidence-agent.md).
Original negative results are retained alongside the new experiment.

The completed held-out synthetic suite has 48 questions and two repetitions per
system (96 outputs each). The typed agent scores **88/96 (91.7%)** with expected
evidence, versus **42/96 (43.8%)** for text RAG using the same Qwen3-4B model.
Correct abstention is 36/36 versus 18/36; no malformed outputs occurred.
This passes the predeclared suite gate, not a general-purpose or production gate.
See [full results](results/evidence_agent_test/REPORT.md) and
[failure analysis](docs/evidence-failure-analysis.md).

## Reproduce

Python 3.11+; CPU supported. Install PyTorch's CPU wheel separately if desired.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m portfolio.edge_vision --out results/edge_vision
python -m portfolio.telemetry --out results/telemetry
python -m portfolio.download_model
python -m portfolio.agent_eval --model-path /path/to/SmolLM2-135M-Instruct --out results/agent_eval
python -m portfolio.small_lm --model-path /path/to/SmolLM2-135M-Instruct --out results/small_lm
python -m portfolio.adapter_tuning --model-path models/SmolLM2-135M-Instruct
python -m portfolio.acceptance --model-path models/SmolLM2-135M-Instruct
```

The first two experiments work offline. LLM experiments require actual local
model weights; they **fail** when weights are absent rather than fabricate scores.
Never submit personal CVs, credentials or private company data to this repository.

## Scope

This repository was built with AI assistance at the repository owner's direction.
The code and measured outcomes are portfolio evidence. A source listing does not
prove a person's independent mastery; interview preparation should include running,
modifying and explaining the experiments and their limitations.

No third-party model weights are committed. Dataset and model provenance is in
the project documentation. Synthetic research cases and telemetry are authored
specifically for this repository and are not real customer or attack data.
