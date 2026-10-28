# Bilingual Answer Evaluation

This extends the existing Python evaluator and optional local GGUF inference path. The question is: **how do a source-extraction baseline and a local model answer the same English/Hebrew questions when given the same retrieved document?** It retains the fictional dataset, validation threshold selection, scorer, paired error audit, frozen predictions and provenance manifest.

The default GGUF mode selects one document using the existing word-overlap rule and gives its ID and source text to both answerers. The baseline extracts its answer from the source sentence; it no longer reads the structured answer label. The former full-document prompt remains available as a separately labeled system comparison.

This is AI-assisted preparation material, not evidence of anyone's independent implementation, mastery, reading, presentation or supervised research experience. No model weights or real-model results are included. No credentials, network service or paid API are needed for the baseline and tests. See [UPSTREAM.md](UPSTREAM.md) for the pinned source.

## Setup and commands

Python 3.10+ is required. The evaluator and tests have **no third-party runtime dependencies**. Packaging uses setuptools. In this worker, setuptools 78.1.1 is already available; these are the verified offline setup commands:

~~~text
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -e .
~~~

The environment inherits locally installed tooling to avoid a network fetch. On a fresh machine without setuptools, use the normal network-enabled setup instead:

~~~text
python3 -m venv .venv
.venv/bin/python -m pip install -e .
~~~

Run from the project directory:

~~~text
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip wheel --no-index --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m cola_lab answers --output-dir artifacts/baseline
.venv/bin/python -m cola_lab answers-smoke
~~~

The wheel command builds an installable package. The finite smoke runs the documented generate → JSON read → validation → evaluation → report path in a temporary directory, checks 24 held-out cases, 12 language pairs, comparison outputs and selected-source exposure, then exits. It prints **baseline only, no model run**. GGUF tests use injected fixtures and a backend adapter spy, never real weights. [VALIDATION.md](VALIDATION.md) records actual commands and outcomes; inherited validation claims do not certify this extension.

## Dataset, retrieval and scoring

The deterministic dataset has 48 questions over fictional color, shape and missing weight facts. Validation and test each contain four different entities, both languages and both answerable/unanswerable cases. Labels are generated, not independently annotated. Validation rejects missing translations, entity leakage, invalid citations, missing fields and inconsistent source labels.

The fixed retrieval rule:

1. Restrict candidates to the question's split and language.
2. Count shared case-folded word tokens after the existing stopword removal.
3. Choose the highest overlap; the lexicographically highest document ID wins ties.

Retrieval uses question and source text, never the expected answer or gold evidence. The selected document is supplied even when overlap is low. For the deliberately paraphrased Ember color question, a tie selects the shape document: **shared evidence does not guarantee correct evidence**.

The lexical answerer receives only the question and one document's ID/text. It extracts the final answer word from the known English or Hebrew fact template and abstains below the validation-selected overlap threshold or for an unrecognized template. Validation chooses highest exact-answer-and-evidence accuracy; the highest threshold wins ties. Test labels are used only for scoring. Always-abstain receives no documents.

A correct answer needs the expected short answer and exactly its evidence IDs; a missing fact requires abstention. Accuracy, grounded-answer rate, abstention accuracy, error categories, entity slices and matched English/Hebrew outcomes use the same scorer. Exact-match scoring may reject reasonable paraphrases.

## Inspecting the baseline report

| Artifact or JSON field | What to inspect |
| --- | --- |
| answer_dataset.json | Source texts, split membership, paired cases and scoring labels. |
| answer_report.json → protocol | Dataset/evaluator hashes, Python version, threshold and data-quality counts. |
| protocol.information_control | baseline_only status, retrieval rule, and actual document IDs/input hashes per baseline and case. |
| methods | Overall, language and entity metrics, error types and concrete failures. |
| paired_audit | Both translations, answers/citations/errors, and improvements/regressions relative to lexical retrieval. |
| comparisons | Paired differences from always-abstain and whole-entity resampling. |
| comparisons_with_lexical | Model-versus-lexical comparison when model outputs exist. |

Values are computed during each run. The four-entity resampling interval describes sensitivity to this sample; it is not a population confidence interval. Shared entities and templates limit claims of general language parity.

## Optional local GGUF run

Only use an existing local model that you are authorized to use. Nothing downloads weights. Install the actual optional backend when needed; this may require network access and a C++ build toolchain:

~~~text
.venv/bin/python -m pip install ".[gguf]"
~~~

Then run:

~~~text
.venv/bin/python -m cola_lab gguf --model /absolute/path/to/model.gguf --output-dir artifacts/shared-run --context-mode shared-document --seed 17 --repeats 2
~~~

Use a new or empty output directory. Each repeat evaluates all 24 test cases. Repeat zero uses the root directory; additional reports and predictions go into repeat-1/, repeat-2/, etc. The exact same dataset is passed to generation and evaluation. Test outcomes do not select thresholds or the best repeat.

Generation is fixed to CPU, context size 2048, one thread for both prompt evaluation and generation, and batch size 128. Sampling uses temperature 0, at most 96 output tokens, top-p 1, top-k 0, min-p 0, typical-p 1, repetition penalty 1, frequency/presence penalties 0, disabled Mirostat and no stop strings. Every call resets backend state and passes its own seed explicitly. Context overflow or malformed output aborts the run; there is no silent prompt truncation or partial quality score.

The seed policy is versioned as **pair-repeat-v1**:

- Pair ID is split/entity/intent, shared by English and Hebrew versions.
- Hash compact UTF-8 JSON [base_seed, pair_id] with SHA-256.
- Interpret the first four bytes as an unsigned big-endian integer, add the zero-based repeat index, and reduce modulo 2147483647.

Both translations receive identical generation settings within each repeat, regardless of case ordering. Repeats change only the seed; prompts and evaluation cases stay fixed. At temperature zero, repeated outputs may be identical. Equal seeds do not guarantee identical sampling across languages or bitwise reproducibility across backend versions/hardware. Repeats are not confidence intervals.

| Artifact | Provenance |
| --- | --- |
| gguf_predictions.json | Frozen answers/citations for all test cases; scoring input. |
| answer_dataset.json, answer_report.json | Existing data, metrics and paired audit, with information-control status and per-case generation settings. |
| gguf_run.json | Schema version 2; weight/data/prediction/evaluator/generation-code hashes; Python/platform/backend versions; configuration and seed policy; every prompt/hash, raw generation, parsed prediction and per-method input/document exposure. |
| gguf_run.json → repeat_artifacts | Paths and hashes for every complete repeat, with measured accuracy identified by backend provenance. |

The manifest's settings.seed is the **base** seed; generations[*].settings.seed is the seed actually passed. Records are recomputed against the declared protocol before scoring. The saved report's baseline input hashes must agree with generation records. A complete run has a final gguf_run.json.

**shared_document_verified** describes a programmatically checked input protocol, not independent review or model quality. Injected tests carry backend **injected-test-backend** and model_quality_evidence **false** in manifest and report. Only the normal local backend can produce a real-model run; successful execution alone does not establish useful quality.

The former exposure mode remains available:

~~~text
.venv/bin/python -m cola_lab gguf --model /absolute/path/to/model.gguf --output-dir artifacts/system-run --context-mode all-documents --seed 17 --repeats 2
~~~

The model receives all eight same-split/same-language documents; the lexical answerer receives one. The report labels this **different_evidence_system_comparison**. Do not attribute a score difference to the model or language alone.

## Existing frozen-output workflow

~~~text
.venv/bin/python -m cola_lab answers --output-dir artifacts/imported --predictions local_predictions.json
~~~

UTF-8 JSON must have exactly model_id, model_version and predictions. Every test case must occur once; each prediction contains exactly case_id, answer (text or null), and citations (distinct document IDs). Abstention uses null and an empty list. Validation rejects unknown/duplicate/missing cases and cross-language or cross-split citations.

This format does not prove how outputs were generated. Imported reports explicitly use **unverified_frozen_outputs** and a null local-model exposure, even if the filename resembles a GGUF output. Preserve the original manifest and report when reviewing a generated experiment.

## Scope and limitations

- The change controls document exposure per case. It does not equalize pretraining, template knowledge, prompt token lengths or language-specific retrieval. Matched translations may select different facts.
- Source extraction is specific to fictional one-word facts. Word overlap is neither TF-IDF nor semantic retrieval. No large model is trained.
- Generated labels, shared templates, four test entities and exact-match scoring limit interpretation. Error categories describe observations, not causes.
- Invalid JSON or an unexposed citation aborts GGUF generation instead of counting as an ordinary answer error. Completed runs alone cannot estimate malformed-output rates.
- No local weights were available. Real English/Hebrew model quality remains unverified (R16 partial). The optional backend was checked through fixtures/adapter tests, not real inference.
- Python and experiment records are exercised. This does not demonstrate every tool in a broader research environment, personal learning progress, or work with a supervisor.

## Retained acceptability experiment

The earlier synthetic bigram study is unchanged. It compares sum and mean token log probabilities on generated agreement sentences, fits grammatical training rows, chooses thresholds on validation and scores the same test rows plus a train-majority baseline. Whole groups stay within splits. This is not CoLA or GPT-2.

~~~text
.venv/bin/python -m cola_lab run --output-dir artifacts/acceptability
.venv/bin/python -m cola_lab smoke
~~~

It writes CSV, JSON and Markdown, including length slices, score population variance and training-group resampling with unchanged validation/test data. These describe the synthetic sample, not population confidence or causal effects. See [CODE_MAP.md](CODE_MAP.md) for implementation/test paths and [EXERCISES.md](EXERCISES.md) for future walkthroughs.
