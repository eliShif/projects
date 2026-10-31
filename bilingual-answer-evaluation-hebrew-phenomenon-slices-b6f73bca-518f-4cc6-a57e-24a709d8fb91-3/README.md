# Bilingual Answer Evaluation — Hebrew Phenomenon Slices

This extends the existing Python evaluator with a small, explicitly selected Hebrew challenge corpus. It asks: **when a question changes through possessive inflection, full/deficient spelling, or a Hebrew–English content-word substitution, does the lexical baseline retrieve the same supporting fact?**

The implementation reuses source validation, validation-only threshold selection, word-overlap retrieval, exact answer-and-citation scoring, paired comparisons, and the optional local GGUF adapter. New code validates the added grammar and annotations, keeps related variants together, and reports concrete failures for each phenomenon. The original experiment remains the default dataset.

This is AI-assisted implementation and preparation material. It does not establish anyone's mastery, language proficiency, completed reading, presentation, learning progress, or team review. No model weights, credentials, paid APIs, or external services are required. Source lineage is recorded in [UPSTREAM.md](UPSTREAM.md).

## Setup, tests and build

Use Python 3.10+. The evaluator and tests use the standard library; there are no third-party runtime dependencies. Setuptools 68+ and wheel are packaging tools. In the supplied worker, those tools are available locally:

```text
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip wheel --no-index --no-build-isolation --no-deps --wheel-dir dist .
```

The install command installs this project; the build produces an installable wheel containing the corpus. On another machine without packaging tools, use the ordinary network-enabled installation instead:

```text
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Run and verify

Run from this project directory. Choose a new output directory for each experiment; existing answer artifacts are never overwritten.

```text
.venv/bin/python -m cola_lab answers --dataset hebrew-phenomena --output-dir artifacts/hebrew-study
.venv/bin/python -m cola_lab answers-verify --output-dir artifacts/hebrew-study
.venv/bin/python -m cola_lab phenomena-smoke
```

`answers` saves the exact raw/cleaned dataset and a JSON report. `answers-verify` recomputes the baseline from that saved dataset and rejects changed scores, diagnostics, source hashes or policies. It performs no model inference and is intended for baseline-only reports.

The finite `phenomena-smoke` runs the same baseline in a temporary directory, replays it, checks all three slices and shared inputs, freezes 48 requests for two repeats, exercises an unavailable-model failure, verifies the failure archive, and exits. Its success is a software check, not a model-quality result. It does not download anything.

Optional packaging check after the build:

```text
.venv/bin/python scripts/wheel_smoke.py dist/bilingual_answer_evaluation-0.4.0-py3-none-any.whl
```

This executes the smoke from the unpacked wheel and checks the imported package's origin, so an editable source installation cannot hide a missing resource.

To capture real command output, exit codes and source hashes for the complete workflow:

```text
python3 scripts/validate.py --output-dir validation/my-run
```

Use a new directory. The runner includes install, tests, wheel build, packaged smoke, both datasets, baseline replay, frozen challenge plan, missing-model audit and retained smoke commands. The unavailable-model command is expected to exit 2; verifying its consistent failure archive exits 0 and still reports `model_quality_evidence: false`. See [VALIDATION.md](VALIDATION.md) for recorded execution, not the inherited checkpoint logs.

## Corpus and experiment design

[DATA_CARD.md](DATA_CARD.md) specifies the source, examples, labels, grammar and cleaning rules. The packaged resource is [hebrew_phenomena.json](cola_lab/resources/hebrew_phenomena.json).

The challenge corpus has 48 cases and 32 source documents. Validation and test each contain four disjoint fictional entities, 24 questions, both languages, and both answerable and unanswerable cases. Each entity's two question forms and English/Hebrew translations share one variant group and one partition. Controls and variants use the same reference fact within each language. No model is trained; the only selected parameter is the overlap threshold, chosen on validation before test scoring.

| Slice | Hebrew control | Hebrew variant | Expected answer and source |
| --- | --- | --- | --- |
| inflection | מה הצבע של Ember? | מה צבעו של Ember? | כחול; הצבע של Ember הוא כחול. |
| spelling | מה התיאור של Fjord? | מה התאור של Fjord? | מרובע; התיאור של Fjord הוא מרובע. |
| mixed_language | מה הצבע של Grove? | מה ה-color של Grove? | אדום; הצבע של Grove הוא אדום. |

Cleaning applies NFC and whitespace normalization only. Raw strings remain in `raw_question` and `raw_text`; the validator verifies their correspondence to the cleaned inputs. It preserves the inflection, the missing letter, the English word, punctuation and vowel marks. Retrieval retains its own existing tokenization. Duplicate normalized questions/documents are rejected within each language, including across partitions. This is exact duplicate detection, not semantic deduplication.

The validator explicitly parses each new form, checks its declared phenomenon and variant, resolves its entity and attribute, and verifies the answer and evidence against source text. It also requires both forms and both translations in every group. Incorrect source values, wrong attributes/entities, false abstention, forged groups, cross-split variants, unknown grammar and incomplete groups are rejected before publishing a report or creating generation requests. Source parsing is a closed grammar, not a general Hebrew semantic validator.

The lexical baseline now also extracts the new description fact and feminine shape-source template. It does not use the reference parser or annotations to repair the query. Inputs for lexical retrieval and the optional model contain only the same question and selected document ID/text in `shared-document` mode. Labels, raw text, phenomenon tags and group annotations are excluded. `all-documents` deliberately gives the model more sources and remains labeled as a different-information system comparison.

Budget: 24 validation cases, 24 test cases, and a finite threshold grid `0..maximum_validation_overlap+1`, recorded as `threshold_candidates`. The local-model plan defaults to two repeats (48 requests), CPU, one thread, a 2048-token context and at most 96 output tokens per request. No performance, compute consumption or model result is inferred from these settings.

## Reading the report

| Field | Interpretation |
| --- | --- |
| protocol.challenge_corpus | Cleaning/reference/slice versions, definitions, group partitions, packaged resource/code hashes, and `independent_linguistic_review: false`. |
| protocol.dataset_sha256 | Hash of the actual saved input, including raw and cleaned strings. The packaged-resource hash separately identifies the bundled corpus. |
| protocol.information_control | Actual baseline input hashes and document IDs per case. Imported predictions have unverified model exposure. |
| methods.*.overall / by_language / by_entity / by_answerability | Retained exact answer-and-citation accuracy, grounded-answer rate, abstention accuracy, denominators and errors. |
| methods.*.by_phenomenon | Three named Hebrew-only slices. `metrics` scores variants; `control_metrics` scores their matched controls. English paraphrases are excluded from these counts. |
| by_phenomenon.*.pairs / failures | Questions, raw variant text, references, predictions, control outcomes, selected and expected source texts, and transitions. `failures` contains only incorrect variants. |
| retrieval_status / error_stage | Was the gold source selected? For the lexical baseline only, a failure is `source_not_retrieved` or `answer_error_with_source`. No stage is inferred for model outputs. |
| paired_audit / comparisons | Retained bilingual pairs and method comparisons, with whole-entity descriptive resampling. |

A nonempty answerable-only slice has `abstention_accuracy: null` with an explanation, not zero. An absent phenomenon has `count: 0`, `metrics: null`, `control_metrics: null`, an `empty_reason`, and no failures. No cases are fabricated to fill empty slices. Low counts are displayed explicitly.

The current baseline observations are recorded in [VALIDATION.md](VALIDATION.md). The challenge has deliberately difficult word-overlap cases; failure is useful evidence about this bounded baseline. A phenomenon tag and a failure together do not prove that phenomenon caused the failure. In particular, inspect retrieval before attributing an error to answer generation.

## Optional local model and audit

The existing native adapter uses `llama-cpp-python`. It is optional and is not needed for the primary workflow. If authorized local weights are already available, install the backend on a suitable machine:

```text
.venv/bin/python -m pip install ".[gguf]"
```

Freeze a new challenge protocol before inference, then run and replay it:

```text
.venv/bin/python -m cola_lab gguf-plan --dataset hebrew-phenomena --output-dir artifacts/challenge-plan --context-mode shared-document --seed 17 --repeats 2
.venv/bin/python -m cola_lab gguf --model models/authorized-model.gguf --protocol artifacts/challenge-plan/gguf_protocol.json --output-dir artifacts/challenge-model
.venv/bin/python -m cola_lab gguf-verify --output-dir artifacts/challenge-model --model models/authorized-model.gguf
```

No command downloads weights. Missing weights cause exit 2 with a protocol, attempt journal and failed status; `gguf-verify` without `--model` can inspect that failure. There is no final manifest or quality score for a failed run. Do not supply `--dataset`, `--seed`, `--repeats` or `--context-mode` alongside a frozen `--protocol`; it already specifies them. All plan/run directories must be new or empty.

The version-2 protocol records the selected dataset, exact requests, pair/repeat seeds, corpus and code hashes. Both translations of each control or variant receive the same seed. Distinct forms have distinct pair IDs. All repeats are retained; test results never select a threshold or best repeat. Changed code/data require a new protocol. Earlier checkpoint plans cannot be replayed under this revision.

Successful GGUF archives retain raw responses, parsed predictions, settings, actual input exposures, weight hashes, backend provenance and every repeat's report. Replay verifies all new slices and challenge policies as well as the retained metrics. Fixture backends used by tests are explicitly labeled `injected-test-backend` with `model_quality_evidence: false`. No real-model inference was performed for this adaptation.

For externally saved predictions, the retained import path accepts `answers --dataset hebrew-phenomena --predictions local_predictions.json --output-dir artifacts/imported`. The file must contain `model_id`, `model_version` and one `{case_id, answer, citations}` prediction per test case. Abstention is null with no citations. Unknown/missing/duplicate cases and invalid citations are rejected. Imported outputs do not verify generation provenance; `answers-verify` is not an audit of imported model runs.

## Retained experiments and limitations

```text
.venv/bin/python -m cola_lab answers --output-dir artifacts/original-study
.venv/bin/python -m cola_lab study-smoke
.venv/bin/python -m cola_lab smoke
```

The first two retain the original 48-case color/shape/missing-weight experiment. The last checks the earlier synthetic bigram experiment in `data.py` and `experiment.py`.

The challenge uses constructed examples, only four held-out entities and one narrow pattern for each phenomenon. Inflection covers a possessive suffix, not all morphology; spelling covers one full/deficient pair; mixing covers one English attribute inside Hebrew syntax. The source templates and labels have no independent linguistic review. English alternatives are paraphrases, so aggregate language differences cannot isolate a causal Hebrew effect. Exact matching can reject reasonable answers. Word overlap is lexical retrieval, not learned semantic search or general information extraction. Resampling over four entities is descriptive, not a population confidence claim. No new training, model improvement or real-model quality is established.

See [CODE_MAP.md](CODE_MAP.md) for implementation/test evidence and [EXERCISES.md](EXERCISES.md) for future walkthroughs and change exercises.
