# Bilingual Answer Evaluation — Reference and Metric Validation

This extends the pinned Python evaluator and its local GGUF inference path. The question is: **how do a source-extraction baseline and a local model answer the same English/Hebrew questions when given the same retrieved document?** It reuses the fictional dataset, validation threshold selection, scorer, paired error audit and baseline comparison.

This revision fixes two evaluator defects: accepting a source word that does not answer the question, and dividing by zero when a nonempty subset contains only answerable or only unanswerable cases. It validates the question's entity and attribute against parsed synthetic facts, including whether a claimed missing answer is actually absent. Undefined conditional rates now serialize as JSON `null`, with case counts and an explanation. Reports include answerability slices so these cases are visible in normal baseline runs.

The existing GGUF workflow saves a protocol before inference, preserves raw failed attempts, and verifies completed reports by replaying the same scorer. The new reference/metric policies and subset summaries participate in replay verification; the journal and inference adapter are reused.

The default GGUF mode selects one document using the existing word-overlap rule and gives its ID and source text to both answerers. The baseline extracts its answer from the source sentence; it no longer reads the structured answer label. The former full-document prompt remains available as a separately labeled system comparison.

This is AI-assisted preparation material, not evidence of anyone's independent implementation, mastery, reading, presentation or supervised research experience. No model weights or real-model results are included. No credentials, network service or paid API are needed for the baseline and tests. See [UPSTREAM.md](UPSTREAM.md) for the pinned source.

## Setup and commands

Python 3.10+ is required. The evaluator and tests have **no third-party runtime dependencies**. Packaging uses setuptools (68+). The worker provides setuptools locally; install the actual project with these offline commands:

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
.venv/bin/python -m cola_lab study-smoke
~~~

The wheel command builds an installable package. The finite `study-smoke` runs the documented baseline, checks the null metrics/counts/explanations for both answerability slices, freezes two repeats (48 requests), exercises the missing-model failure path, verifies its saved records, and exits. It checks 24 held-out cases and 12 language pairs and prints **no model quality result**. The retained `answers-smoke` additionally checks selected-source exposure and paired comparison fields. GGUF success tests use injected fixtures and a backend adapter spy, never real weights.

To save fresh, source-linked command output for the entire sequence, use a new output directory:

~~~text
python3 scripts/validate.py --output-dir validation/recheck
~~~

This runs actual install/test/build/baseline/plan/smoke commands using argument lists, captures stdout and stderr separately, records exit codes and hashes, and checks that source files did not change. The unavailable-file check expects exit 2; verification of that failed run expects exit 0. [VALIDATION.md](VALIDATION.md) links this revision's actual output. Older validation files are inherited checkpoints, not certification of this revision.

## Dataset, retrieval and scoring

The deterministic dataset has 48 questions over fictional color, shape and missing weight facts. Validation and test each contain four different entities, both languages and both answerable/unanswerable cases. Labels are generated, not independently annotated. Validation rejects missing translations, entity leakage, invalid citations, missing fields and incorrect references.

`validate_dataset` parses every source into an entity, attribute and value using the corpus's exact English/Hebrew templates. It separately parses ordinary color/shape/weight questions and the existing hue paraphrases. The case ID's intent and entity metadata must match the parsed question. It looks up the fact by split, language, entity and attribute, checks the exact gold value and its evidence ID, and accepts `null` with empty evidence only when that fact is absent from **all** same-split/same-language sources. An irrelevant retrieval result never makes a source-supported question unanswerable.

For example, in `The color of Ember is blue.`, the word `color` occurs but cannot be the gold value; `round` from Ember's shape sentence also cannot answer its color question. A question asking about Fjord cannot use Ember's answer. Changing both translations to `null` with empty evidence is rejected when either queried fact exists. Tests in [test_reference_validation.py](tests/test_reference_validation.py) use [hand-written source/answer fixtures](tests/fixtures/reference_cases.json) independent of the generator, parser and lexical answerer.

Unsupported question/source templates and duplicate or conflicting facts are rejected. This is a closed synthetic grammar with one fact per entity/attribute/language/split, single-word values, and canonical case IDs; it is not a semantic validator for arbitrary documents or an independent audit of translation correctness. Adding a new template requires explicit parser rules and separately specified test expectations.

The fixed retrieval rule:

1. Restrict candidates to the question's split and language.
2. Count shared case-folded word tokens after the existing stopword removal.
3. Choose the highest overlap; the lexicographically highest document ID wins ties.

Retrieval uses question and source text, never the expected answer or gold evidence. The selected document is supplied even when overlap is low. For the deliberately paraphrased Ember color question, a tie selects the shape document: **shared evidence does not guarantee correct evidence**.

The lexical answerer receives only the question and one document's ID/text. It extracts the final answer word from the known English or Hebrew fact template and abstains below the validation-selected overlap threshold or for an unrecognized template. Validation chooses highest exact-answer-and-evidence accuracy; the highest threshold wins ties. Test labels are used only for scoring. Always-abstain receives no documents.

A correct answer needs the expected short answer and exactly its evidence IDs; a missing fact requires abstention. Accuracy, grounded-answer rate, abstention accuracy, error categories, entity slices and matched English/Hebrew outcomes use the same scorer. Exact-match scoring may reject reasonable paraphrases.

Every nonempty summary includes `count`, `answerable_count`, `unanswerable_count` and `undefined_metrics` (a metric-to-reason map). Accuracy divides all correct cases by `count`; grounded-answer rate divides correct answer-and-citation cases by `answerable_count`; abstention accuracy divides correct abstentions by `unanswerable_count`. If a conditional denominator is zero, that rate is `null` and the map explains why. A defined rate of zero remains numeric zero. Empty input and prediction-count mismatches still raise `ValueError`.

`methods.<name>.by_answerability` summarizes each present case type; an absent type is omitted rather than passed as an empty slice. `summarize` also supports a nonempty single-language subset; the full `method_report` continues to require paired English/Hebrew cases. These semantics apply to overall, language, entity and answerability summaries and are checked numerically in [test_summary_subsets.py](tests/test_summary_subsets.py).

## Inspecting the baseline report

| Artifact or JSON field | What to inspect |
| --- | --- |
| answer_dataset.json | Source texts, split membership, paired cases and scoring labels. |
| answer_report.json → protocol | Dataset/evaluator hashes, Python version, threshold, data-quality counts, reference_validation and metric_policy versions. |
| protocol.information_control | baseline_only status, retrieval rule, and actual document IDs/input hashes per baseline and case. |
| methods | Overall, language, entity and answerability metrics, support counts, undefined reasons, error types and concrete failures. |
| paired_audit | Both translations, answers/citations/errors, and improvements/regressions relative to lexical retrieval. |
| comparisons | Paired differences from always-abstain and whole-entity resampling. |
| comparisons_with_lexical | Model-versus-lexical comparison when model outputs exist. |

Values are computed during each run. The four-entity resampling interval describes sensitivity to this sample; it is not a population confidence interval. Shared entities and templates limit claims of general language parity.

## Optional local GGUF run

Only use an existing local model that you are authorized to use. Nothing downloads weights. Install the actual optional backend when needed; this may require network access and a C++ build toolchain:

~~~text
.venv/bin/python -m pip install ".[gguf]"
~~~

Freeze the protocol **before** inspecting model outputs (this does not need weights):

~~~text
.venv/bin/python -m cola_lab gguf-plan --output-dir artifacts/planned-study --context-mode shared-document --seed 17 --repeats 2
~~~

This saves the existing dataset, every label-free request and prompt, pair/repeat seeds, settings, failure policy and source hashes. Labels remain in the dataset for scoring; they never enter answerer inputs. The selected weight file is bound by hash when the run starts. `gguf-plan` defaults to two repeats; an unplanned direct `gguf` run retains the previous one-repeat default. Explicit settings in the commands avoid that distinction.

Once an authorized local model is available, run the saved protocol and audit the result:

~~~text
.venv/bin/python -m cola_lab gguf --model models/authorized-model.gguf --protocol artifacts/planned-study/gguf_protocol.json --output-dir artifacts/shared-run
.venv/bin/python -m cola_lab gguf-verify --output-dir artifacts/shared-run --model models/authorized-model.gguf
~~~

Use new or empty plan and run directories. A frozen protocol cannot be combined with `--seed`, `--repeats` or `--context-mode` overrides. Changed code, inputs or settings require a new plan. The default selected mode is `shared-document`. Each repeat evaluates all 24 test cases; all repeats are retained. Repeat zero uses the root directory; additional reports and predictions go into repeat-1/, repeat-2/, etc. Test outcomes do not select thresholds or the best repeat.

Generation is fixed to CPU, context size 2048, one thread for both prompt evaluation and generation, and batch size 128. Sampling uses temperature 0, at most 96 output tokens, top-p 1, top-k 0, min-p 0, typical-p 1, repetition penalty 1, frequency/presence penalties 0, disabled Mirostat and no stop strings. Every call resets backend state and passes its own seed explicitly. Context overflow or malformed output stops at the first failure; the raw attempts remain on disk. There is no silent prompt truncation or partial quality score.

The seed policy is versioned as **pair-repeat-v1**:

- Pair ID is split/entity/intent, shared by English and Hebrew versions.
- Hash compact UTF-8 JSON [base_seed, pair_id] with SHA-256.
- Interpret the first four bytes as an unsigned big-endian integer, add the zero-based repeat index, and reduce modulo 2147483647.

Both translations receive identical generation settings within each repeat, regardless of case ordering. Repeats change only the seed; prompts and evaluation cases stay fixed. At temperature zero, repeated outputs may be identical. Equal seeds do not guarantee identical sampling across languages or bitwise reproducibility across backend versions/hardware. Repeats are not confidence intervals.

| Artifact | Provenance |
| --- | --- |
| gguf_protocol.json | Saved before opening a model: dataset, ordered requests, exact shared inputs, prompts, settings, repeats, source hashes and failure policy. |
| gguf_status.json | running / failed / completed; backend and environment, weight hash when available, protocol/journal hashes, counts, failure phase/error. Only completion can claim model quality evidence. |
| gguf_attempts.jsonl | One flushed record per completed or failed call, including language, repeat, exact raw response, parsed output or error. A backend exception has no response. |
| gguf_predictions.json | Frozen answers/citations for all test cases; scoring input. |
| answer_dataset.json, answer_report.json | Existing data, metrics and paired audit, with information-control status and per-case generation settings. |
| gguf_run.json | Schema version 3; weight/data/prediction/code/protocol/journal hashes; environment versions; every prompt/hash, raw generation, parsed prediction and per-method input/document exposure. Written after all repeats score. |
| gguf_run.json → repeat_artifacts | Paths and hashes for every complete repeat, with measured accuracy identified by backend provenance. |

The manifest's settings.seed is the **base** seed; generations[*].settings.seed is the seed actually passed. Records are recomputed against the declared protocol before scoring. Weight/source hashes are rechecked after generation. The saved report's baseline input hashes must agree with generation records. A complete run requires both `status: completed` and a final manifest.

`gguf-verify` checks file hashes, journal order, full repeat coverage, model provenance, actual exposure and settings. It then replays raw outputs through the existing parser and scorer and compares saved metrics (including nulls, denominators and reasons), errors, paired comparisons and the declared reference/metric policies. Its JSON includes **every repeat's** English/Hebrew accuracy, lexical accuracy, differences and error counts. Optional `--model` checks the weight hash; omitting it never claims the weights were checked or reloaded. Verification needs the same source revision, uses no inference and rejects unsupported older manifests. This evaluator change invalidates previous frozen plans by design; make a new plan for this revision and retain old records with their original source.

## Inspecting an unavailable or failed model run

Without weights, the documented `gguf` command exits 2 after saving the protocol, an empty journal and a failed status. Run the same `gguf-verify` command **without `--model`** to inspect it. Verification exits 0 when that failure record is internally consistent: the JSON still says `status: failed`, `model_quality_evidence: false`, and `quality_summary: null`. Exit 0 means the archive was checked, not that inference succeeded.

For malformed responses, the journal retains the response verbatim and the first failed case/language/repeat. The audit counts planned, attempted, parsed, invalid, backend-failed and unattempted requests by language. It reports no partial accuracy. Early stopping means these counts cannot estimate overall malformed-output rates or compare language reliability. Even if scoring fails after one repeat, partial reports/predictions are removed and the journal remains.

Status JSON is atomically replaced and journal records are flushed after calls. A forced process kill may leave `running` status or an incomplete final journal line; the verifier rejects it. Inspect that journal manually and use a new run directory. There is no automatic resume, retry or cherry-picking of completed cases.

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
- Source-based reference checks validate the supported grammar, not independently annotated truth or translation quality. Generated labels, shared templates, four test entities and exact-match scoring limit interpretation. Error categories describe observations, not causes.
- Invalid JSON or an unexposed citation aborts GGUF generation instead of counting as an ordinary answer error. Completed runs alone cannot estimate malformed-output rates.
- No GGUF weights or optional native backend were available in this workspace. Real English/Hebrew model quality remains unverified (**R18 partial**). The backend adapter was checked through fixtures/spy tests, not real inference; no weights were downloaded and no paid API was used.
- Protocols and hashes provide local consistency checks, not independent proof of execution, authorship or a timestamped external preregistration. A user controlling all files can replace them together. Source-version verification does not guarantee identical inference across hardware or backend versions.
- Python and experiment records are exercised. This does not demonstrate every tool in a broader research environment, personal learning progress, or work with a supervisor.

## Retained acceptability experiment

The earlier synthetic bigram study is unchanged. It compares sum and mean token log probabilities on generated agreement sentences, fits grammatical training rows, chooses thresholds on validation and scores the same test rows plus a train-majority baseline. Whole groups stay within splits. This is not CoLA or GPT-2.

~~~text
.venv/bin/python -m cola_lab run --output-dir artifacts/acceptability
.venv/bin/python -m cola_lab smoke
~~~

It writes CSV, JSON and Markdown, including length slices, score population variance and training-group resampling with unchanged validation/test data. These describe the synthetic sample, not population confidence or causal effects. See [CODE_MAP.md](CODE_MAP.md) for implementation/test paths and [EXERCISES.md](EXERCISES.md) for future walkthroughs.
