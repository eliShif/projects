# Code map

The existing project was inspected and extended locally. Python, unittest, JSON, the original dataset and scorer, and the optional llama-cpp-python adapter are retained. This map describes code and executable checks, not personal competence or independent review.

## Execution paths

Challenge baseline:

`__main__.py answers --dataset hebrew-phenomena` → `dataset_named` → `phenomena.make_dataset` (packaged JSON, raw + cleaned text) → `validate_dataset` (text hygiene, explicit grammar, source references, complete isolated groups) → validation-only threshold selection → retained shared `method_inputs` and lexical extraction → retained metrics/comparisons + `phenomenon_slices` → dataset/report files.

Baseline replay:

`answers-verify` → load saved input bytes → validate and evaluate again → compare the entire report, including source policies, nulls, selected-source diagnostics and failures. A mismatch fails; replay does not call a language model.

Optional GGUF:

`prepare` freezes the chosen dataset, label-free inputs, seeds and code/resource hashes → `run` loads that protocol, checks weights, records each raw attempt, validates records and scores every repeat with the same evaluator → completed status/final manifest. On failure, attempts remain but partial scores and final manifest do not. `verify_bundle` replays raw outputs and checks all report slices and policies. Injected fixture success is explicitly distinguished from a real backend.

## Implementation and tests

| Path | Responsibility | Executable checks |
| --- | --- | --- |
| `cola_lab/resources/hebrew_phenomena.json` | Literal fictional sources, expected values/evidence, both languages, controls/variants, split and group annotations. Included in the wheel. | `tests/test_phenomena_data.py`; `scripts/wheel_smoke.py` |
| `cola_lab/phenomena.py` | Definitions; NFC/whitespace cleaning; raw/clean correspondence; duplicate detection; explicit inflection/spelling/mixed question grammar; description/feminine-shape sources; complete same-split variant groups. | `tests/test_phenomena_data.py` |
| `cola_lab/answer_eval.py` | Retained source-reference validation, retrieval, shared inputs, threshold, scoring and comparisons; extended schema/grammar; Hebrew slices, matched controls, failures, lexical retrieval diagnostics; baseline replay and overwrite prevention. | `tests/test_phenomena_data.py`, `tests/test_phenomena_reports.py`; retained reference/metric/shared-input tests |
| `cola_lab/local_gguf.py` | Retained local adapter and attempt journal; selected corpus in version-2 frozen protocols; resource/code hashes; rejects dataset overrides. | `tests/test_phenomena_protocol.py`, `tests/test_local_gguf.py`, `tests/test_gguf_audit.py` |
| `cola_lab/gguf_audit.py` | Retained artifact verification and scoring replay; also compares challenge policies and threshold grid; method replay covers new slices and diagnostics. | `tests/test_phenomena_protocol.py` rehashed-edit test; `tests/test_gguf_audit.py` |
| `cola_lab/__main__.py` | Corpus selector; baseline replay command; finite `phenomena-smoke`; retained commands and argument validation. | `tests/test_phenomena_protocol.py` command tests; retained command tests |
| `cola_lab/data.py`, `cola_lab/experiment.py` | Unchanged earlier synthetic bigram experiment. | `tests/test_experiment.py` |
| `tests/fixtures/reference_cases.json` | Retained independent literal fixture for original source-reference tests. | `tests/test_reference_validation.py` |
| `scripts/validate.py` | Runs actual commands, captures stdout/stderr/exits/hashes, checks source stability, saves actual observations from both corpora. | Recorded execution in `validation/hebrew-phenomena/commands.json` |
| `scripts/wheel_smoke.py`, `pyproject.toml` | Build version 0.4.0 with corpus package data; finite smoke from unpacked wheel with import-origin check. | Recorded wheel build and packaged smoke output |
| `README.md`, `DATA_CARD.md`, `EXERCISES.md` | Setup, protocol/data limits, future learning walkthrough and bounded changes. | Documented CLI integration tests and smoke |

## Requirement evidence and boundaries

These IDs refer only to the current assignment. Coverage is limited to the implementation and observed software checks below; personal skills and future workplace activities are not inferred.

| ID | Implemented support | Implementation path | Success/failure test path |
| --- | --- | --- | --- |
| R10 | Raw-preserving cleaning, exact duplicate rejection, source-checked labels, entity/group isolation across validation/test, complete variants/translations. No training partition is needed by this threshold-only evaluator. | `cola_lab/phenomena.py`; `cola_lab/answer_eval.py` → validate_dataset | `tests/test_phenomena_data.py` |
| R14 | Retained task-appropriate exact answer/citation and abstention metrics, denominators/null reasons; three Hebrew slices with measured support and matched controls. | `cola_lab/answer_eval.py` → summarize, phenomenon_slices | `tests/test_phenomena_reports.py`; `tests/test_summary_subsets.py` |
| R15 | Explicit possessive inflection, one full/deficient spelling pair and one English content-word substitution; concrete failures, reference/selected text and source-miss versus answer-error diagnostics. Bounded synthetic examples only. | `cola_lab/phenomena.py`; `cola_lab/answer_eval.py` → phenomenon_slices | `tests/test_phenomena_data.py`; `tests/test_phenomena_reports.py` |
| R17 | Raw and cleaned inputs, group partitions, corpus/code hashes, selected threshold/grid, frozen requests, baseline replay and retained GGUF replay including new slices/policies. | `cola_lab/answer_eval.py` → challenge_protocol, verify_answers; `cola_lab/local_gguf.py`; `cola_lab/gguf_audit.py` | `tests/test_phenomena_reports.py`; `tests/test_phenomena_protocol.py` |
| R19 | Reuses bilingual lexical retrieval and template answer extraction on the new forms. This is not learned semantic search, general NER or a new classifier. | `cola_lab/answer_eval.py` → retrieve, predict_from_input, method_inputs | `tests/test_phenomena_reports.py`; `tests/test_phenomena_data.py` |
| R21 | Reproducible existing experiment, finite run/replay path and explanation exercises. No personal explanation or employer-data familiarity is claimed. | `cola_lab/__main__.py` → phenomena-smoke; `EXERCISES.md` | `tests/test_phenomena_protocol.py` → PhenomenaCommandTests |
| R22 | Focused question, retained always-abstain/lexical alternatives, optional local-model route, computed comparisons and documented findings/limits; no real-model comparison executed. | `cola_lab/answer_eval.py` → evaluate, compare_methods; `README.md` | `tests/test_phenomena_reports.py`; `tests/test_answer_eval.py` |
| R24 | Negative findings retained, empty slices explicit, failed experiments cannot publish quality, changed reports rejected, existing successful artifacts cannot be overwritten. No learning progress or team assessment claimed. | `cola_lab/answer_eval.py` → run, verify_answers; `cola_lab/local_gguf.py`; `cola_lab/gguf_audit.py` | `tests/test_phenomena_reports.py`; `tests/test_phenomena_protocol.py` |
| P01 | Same NLP evaluation project now provides per-phenomenon controls, comparisons and concrete errors. | `cola_lab/answer_eval.py` → phenomenon_slices, evaluate | `tests/test_phenomena_reports.py` |
| P04 | Hebrew examples, explicit annotation definitions, source quality and cleaning checks. No human linguistic annotation/review is asserted. | `cola_lab/phenomena.py`; `cola_lab/resources/hebrew_phenomena.json` | `tests/test_phenomena_data.py` |
| I01 | Same label-free selected source/question per answerer, matching seeds within translation pairs, held-out test references do not choose the threshold. Control/variant references remain fixed within language. | `cola_lab/answer_eval.py` → method_inputs, evaluate; `cola_lab/local_gguf.py` → generation_requests | `tests/test_phenomena_data.py`; `tests/test_phenomena_protocol.py` |
| I02 | Literal reference values, wrong labels/entities/attributes, false absence, text duplicates, split leakage, missing variants, zero/absent slices, source-hit errors, changed artifacts and CLI failure cases. | `cola_lab/phenomena.py`; `cola_lab/answer_eval.py`; `cola_lab/gguf_audit.py` | `tests/test_phenomena_data.py`; `tests/test_phenomena_reports.py`; `tests/test_phenomena_protocol.py` |

Earlier validation directories are inherited checkpoints and use earlier source hashes and requirement mappings. Only the new execution records describe this implementation. Hashes and replay establish local consistency, not authorship or independent proof that inference occurred.
