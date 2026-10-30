# Code map

The source was inspected and extended in place from the revision in [UPSTREAM.md](UPSTREAM.md). No original remote was changed. Python, unittest, the existing data, retrieval, scoring and optional llama-cpp-python adapter are retained.

| Path | Responsibility | Verification |
| --- | --- | --- |
| cola_lab/answer_eval.py | _source_fact and _question_target parse the bounded grammar; validate_dataset joins question targets to source facts, rejects false abstention and wrong references. summarize exposes denominators/nulls/reasons; method_report adds answerability slices. Retains generation, retrieval, label-free inputs, scoring and paired comparisons. | tests/test_reference_validation.py, tests/test_summary_subsets.py, tests/test_answer_eval.py, tests/test_shared_protocol.py |
| tests/fixtures/reference_cases.json | Literal English/Hebrew facts and gold answers specified separately from the generator and parser. Tests mutate them to show semantic reference failures, including both translations at once. No human annotation or model output is claimed. | tests/test_reference_validation.py |
| cola_lab/local_gguf.py | Retained GGUF adapter, prompts, pair seeds, frozen plans, durable attempts, failure status, weight/code drift checks and transactional scoring. generation_requests calls the strengthened validator; _publish_results calls the same evaluator. | tests/test_local_gguf.py, tests/test_gguf_audit.py, tests/test_reference_validation.py |
| cola_lab/gguf_audit.py | Retained bundle verification and scoring replay; now also checks reference_validation and metric_policy. Existing methods replay checks the new subset metrics, counts and reasons. | tests/test_gguf_audit.py → test_subset_nulls_and_denominators_survive_replay_and_reject_tampering, test_replay_rejects_changed_reference_and_metric_policies |
| cola_lab/__main__.py | Existing commands; study-smoke now verifies both undefined subset metrics in the actual baseline report, then exercises the retained unavailable-model audit. | tests/test_gguf_audit.py → CommandTests, tests/test_shared_protocol.py → CommandTests |
| cola_lab/data.py, cola_lab/experiment.py | Unchanged earlier synthetic bigram experiment. | tests/test_experiment.py |
| scripts/validate.py | Reused command runner; source manifest now includes the reference JSON fixture and observations include metric policies and answerability slices. Captures actual exits/output/hashes. | CLI integration tests above; actual records in validation/reference-fix/commands.json |
| pyproject.toml | Installable version 0.3.1, setuptools build, no baseline runtime dependencies; retained optional native GGUF dependency. | Raw installation/build output in validation/reference-fix/ |
| README.md, EXERCISES.md, VALIDATION.md | Setup/protocol, future learning prompts and revision-specific execution evidence. | Documented CLI tests and finite smoke |

## Execution boundaries

Baseline: make_dataset → JSON file → validate_dataset (parse all sources, parse questions, validate gold/absence) → validation-only threshold → shared method_inputs → source extraction → summarize (support counts and nulls) → method_report (language/entity/answerability slices) → paired audit/comparisons → JSON report. Reference parsing is confined to validation; answerers still see only question and document ID/text.

Planned GGUF: prepare saves dataset + requests + code hashes → load_protocol checks against current code → run writes protocol/status **before opening the model** → hash weights → initialize/reset native backend → append each raw attempt → validate all records and recheck code/weights → _publish_results calls answer_eval.run for every repeat → final manifest/completed status.

Failure: run persists the phase/error and journal, removes any partial reports/predictions and propagates the exception. The verifier accepts a consistent failed archive without producing a quality score. A forcibly terminated process remains incomplete and fails verification.

Replay: verify_bundle checks fixed artifact paths and hashes → compares journal order and full request/settings/input provenance → parses raw outputs → loads frozen predictions → calls the retained evaluate → compares metrics, errors, paired audit and report exposure. It does not load weights or regenerate responses. The optional model check hashes the supplied file.

## Requirement evidence and limits

IDs below refer to this assignment. They describe executable preparation material and observed worker checks, not personal competence, supervision or learning progress.

| ID | Implemented support | Implementation path | Success and failure test path |
| --- | --- | --- | --- |
| R11 | Preserved baseline comparisons and concrete bilingual failures; new answerability slices distinguish answering from abstaining without assigning false zero scores. | cola_lab/answer_eval.py → summarize, method_report, paired_case_audit | tests/test_summary_subsets.py → saved baseline and per-language numerical tests; tests/test_answer_eval.py → test_comparison_keeps_paired_cases_and_exposes_no_gain |
| R13 | Retained frozen protocols, journal and scoring replay; explicit reference/metric policies, including detection of edited policies or subset results even after hashes are updated. Fresh command records include fixture/source hashes. | cola_lab/gguf_audit.py → _verify_bundle; cola_lab/local_gguf.py; scripts/validate.py | tests/test_gguf_audit.py → test_replay_rejects_changed_reference_and_metric_policies, test_subset_nulls_and_denominators_survive_replay_and_reject_tampering |
| R17 | Source validation checks the actual queried entity/attribute/value and evidence; rejects false abstention, in-source nonanswers, wrong-attribute/other-entity references and ambiguous sources. | cola_lab/answer_eval.py → _source_fact, _question_target, validate_dataset | tests/test_reference_validation.py; independent literal fixture at tests/fixtures/reference_cases.json |
| R18 | Retained optional local GGUF path with bilingual inputs; strengthened validation runs before requests. **Partial: no weights, native backend or real-model quality run here.** | cola_lab/local_gguf.py → generation_requests, create_backend, run | tests/test_reference_validation.py → test_invalid_references_fail_before_generation_and_report_publication; tests/test_local_gguf.py → backend spy; tests/test_gguf_audit.py → missing-model and injected-success tests |
| R29 | Focused success/failure tests, defined-versus-undefined metrics, source-linked command records and explicit limitations; no claim about personal progress. | cola_lab/answer_eval.py; cola_lab/gguf_audit.py; cola_lab/__main__.py → study-smoke | tests/test_reference_validation.py; tests/test_summary_subsets.py; tests/test_gguf_audit.py → subset replay and finite smoke tests |
| I01 | Same retrieved, label-free question/document inputs for both answerers; invalid gold rejected at experiment boundary without giving labels to answerers. | cola_lab/answer_eval.py → method_inputs; cola_lab/local_gguf.py → generation_requests | tests/test_shared_protocol.py → test_label_free_inputs_ignore_changed_labels_but_protocol_rejects_them, test_shared_input_is_selected_by_fixed_overlap_and_not_gold_evidence |
| I02 | Known numerical expectations for mixed/unequal/single-type subsets, per-language and saved-report behavior; empty input and mismatched predictions remain errors. | cola_lab/answer_eval.py → summarize, method_report, run | tests/test_summary_subsets.py; tests/test_reference_validation.py; tests/test_gguf_audit.py → subset/policy tampering tests |

The earlier validation/checkpoint files remain for continuity. This revision's execution evidence is isolated under validation/reference-fix/; hashes link it to the implementation, literal fixture and test files actually run. Requirement IDs in older cards/checkpoints use an older mapping; the table above uses this assignment's IDs. No personal or logistical requirements are inferred from this code.
