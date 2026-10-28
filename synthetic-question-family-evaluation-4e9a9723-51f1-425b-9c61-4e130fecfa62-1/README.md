# Applied AI Portfolio

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

## Junior AI researcher adaptation: grouped question classification

[`portfolio/research_lab.py`](portfolio/research_lab.py) extends the existing
synthetic research cases into a small **supervised** experiment. It asks whether
question wording predicts one of eight task families for unseen service entities.
Three entities train the models, one selects the model, and two are held out for
one final evaluation. All cases for an entity stay in one split. A majority-class
baseline is compared with unigram and unigram-plus-bigram multinomial Naive Bayes
models. The script checks missing fields, duplicate IDs, label coverage and entity
leakage, then saves the input, frozen protocol, metrics, confusion matrices,
per-entity variance, failure examples and a readable report.
For each family the saved test metrics include support, predicted count,
precision, recall and F1. The report puts the majority baseline beside the
validation-selected method, so missed families are visible even when an
aggregate accuracy looks acceptable. The formulas and the English technical
source that motivates this choice are in `CODE_MAP.md`.
The distribution review converts label counts to fractions so the 24/8/16
split sizes are comparable. It reports each family's mean and population
variance across the three fixed splits, plus total variation from training.
These are descriptive checks of class mix, not estimates of sampling error.
The score-margin review also records the top-two class log-score gap for each
validation, test and alternate-wording case under both Naive Bayes methods.
It reports means, population variances and separate correct/error summaries.
These unnormalized gaps are not calibrated probabilities. In the regenerated
run, the selected bigram model's 16 alternate-wording cases have mean margin
3.378 and population variance 7.416; six are wrong despite a positive margin.
It also rejects conflicting labels for identical questions, checks a fixed
method across six entity holdouts, downsamples one training label, and records
unseen features and label counts for challenge errors. These diagnostics are
exploratory and never affect model selection.
For generated case IDs, a source-ledger audit checks that the entity and family
label still match the generator. Custom IDs are listed as unmatched, and this
provenance check does not replace an independent annotation review. The grouped
fold report includes the mean and population variance of accuracy across its
six holdouts; it is descriptive, not a confidence interval.
The validation-selected method is also refit on 12 fixed-seed, label-stratified
resamples of the 24 training cases. Original test and alternate wording sets
stay frozen. The report gives the range, mean and population variance of
accuracy; `summary.json` records sampled case IDs and per-case error counts.
These fits probe sensitivity to training composition, not uncertainty over
new data or independent test performance.
The paired wording review matches each held-out question to its alternate by
case ID, checks that its entity and label are unchanged, and records all four
correct/wrong transitions. Its per-family counts show which labels lose correct
predictions when wording changes; incomplete or mismatched pairs are rejected.

Python 3.11+ is enough for this new experiment. From the repository root:

```sh
python3 -m unittest discover -s tests -p test_research_lab.py -v
python3 -m compileall -q portfolio tests
python3 -m portfolio.research_lab --out results/research_lab
```

No package installation is needed for this standard-library experiment. To
reproduce the repository's other research modules in an isolated environment,
run `python3 -m venv .venv` followed by
`.venv/bin/pip install -r requirements-research.txt`.

The last command is a finite offline smoke run. Inspect
[`results/research_lab/REPORT.md`](results/research_lab/REPORT.md) and
[`results/research_lab/summary.json`](results/research_lab/summary.json).
The run also writes `challenge_cases.json`: one alternate question wording for
each of the 16 held-out entity/family pairs. The selected bigram model scores
16/16 on the repeated templates and 10/16 on this paired wording diagnostic;
the six failures are listed in `REPORT.md`. The model is still selected only
on the original validation split. Since the alternate wording was authored
after seeing the original templates, this is an exploratory sensitivity check,
not an independent estimate of performance on unseen writing styles.

The regenerated report shows all six original-template entity folds at 1.0
accuracy, reflecting their shared wording. Downsampling `lookup` from three
to one training example lowers held-out macro F1 from 1.000 to 0.833 and
lookup recall from 1.000 to 0.000 in this post-hoc check. The six challenge
errors include unseen-feature fractions. No independent annotation audit
exists, so these clues do not establish an error's cause.
The paired review records six correct-to-wrong transitions across the 16
held-out pairs. `summary.json` contains every pair and per-family counts;
`REPORT.md` shows the count table.

The template-reuse audit masks the synthetic service names while preserving
`Sandbox` and `Mirror` suffixes. It finds exact training-template matches for
5/8 validation cases and 16/16 original test cases, versus 0/16 alternate
wording cases. These counts explain why entity separation alone does not test
new phrasing. The audit checks exact normalized strings, not semantic
similarity; the challenge remains an exploratory diagnostic.
In this run, each of the eight families has fraction 0.125 in every split;
each per-family split variance and train-to-evaluation total variation is 0.
This balance does not resolve template reuse: the original test still has
16/16 exact normalized training-template matches.

The input workspace is the pinned Applied AI Portfolio named in `UPSTREAM.md`,
not the separately described GPT-2/CoLA project. This adaptation reuses its
existing synthetic question-family classifier and makes no claim that the
GPT-2/CoLA code or results were available or reproduced.

To evaluate a modified case file, use
`python3 -m portfolio.research_lab --cases results/research_lab/cases.json --out results/research_lab_custom`.
This reads JSON from disk and rejects malformed cases before writing results.
The focused test and finite smoke commands above ran in this workspace; the
full repository suite was not rerun here. The focused suite has 12 tests,
including hand-calculated score-margin arithmetic and invalid-input checks.

The held-out entities share question templates with training. A perfect
original-test score on this tiny synthetic set is not evidence of general language understanding,
Hebrew performance, or a person's independent mastery. See
[`CODE_MAP.md`](CODE_MAP.md) and [`EXERCISES.md`](EXERCISES.md) for a code guide and
learning exercises. This adaptation is AI-assisted; individual contribution and
understanding need direct review.
