# Code map

The source was inspected and extended in place from the revision in [UPSTREAM.md](UPSTREAM.md). No original remote was changed. Python, unittest, the existing data, retrieval, scoring and optional llama-cpp-python adapter are retained.

| Path | Responsibility | Verification |
| --- | --- | --- |
| cola_lab/answer_eval.py | Unchanged bilingual dataset, validation, label-free shared inputs, source extraction, validation threshold, scorer, language slices, paired errors and entity comparisons. | tests/test_answer_eval.py, tests/test_shared_protocol.py |
| cola_lab/local_gguf.py | Existing GGUF adapter, prompts, pair seeds and repeat reports; extended with make_protocol, prepare, load_protocol, durable attempts, failure status, weight/code drift detection and cleanup of partial reports. _publish_results still calls the existing evaluator. | tests/test_local_gguf.py, tests/test_gguf_audit.py |
| cola_lab/gguf_audit.py | Verifies completed/failed bundles; reconstructs requests and predictions; reuses evaluate and load_model_predictions for scoring replay; returns each repeat's baseline comparison and errors by language. No new scoring rule or inference engine. | tests/test_gguf_audit.py |
| cola_lab/__main__.py | Retained commands plus gguf-plan, frozen gguf --protocol, gguf-verify and finite study-smoke; rejects overrides and misplaced flags. | tests/test_gguf_audit.py → CommandTests, tests/test_shared_protocol.py → CommandTests |
| cola_lab/data.py, cola_lab/experiment.py | Unchanged earlier synthetic bigram experiment. | tests/test_experiment.py |
| scripts/validate.py | Runs real offline setup, tests, build and documented commands; captures stdout/stderr, expected versus observed exits and source hashes. The missing-file command is an expected failure, never a model score. | CLI integration tests above; actual records in validation/current/commands.json |
| pyproject.toml | Installable version 0.3.0, setuptools build, no baseline runtime dependencies; retained optional native GGUF dependency. | Raw installation/build output in validation/current/ |
| README.md, EXERCISES.md, VALIDATION.md | Setup/protocol, future learning prompts and revision-specific execution evidence. | Documented CLI tests and finite smoke |

## Execution boundaries

Baseline: make_dataset → JSON file → validate_dataset → validation-only threshold → shared method_inputs → source extraction → existing metrics/audit/report.

Planned GGUF: prepare saves dataset + requests + code hashes → load_protocol checks against current code → run writes protocol/status **before opening the model** → hash weights → initialize/reset native backend → append each raw attempt → validate all records and recheck code/weights → _publish_results calls answer_eval.run for every repeat → final manifest/completed status.

Failure: run persists the phase/error and journal, removes any partial reports/predictions and propagates the exception. The verifier accepts a consistent failed archive without producing a quality score. A forcibly terminated process remains incomplete and fails verification.

Replay: verify_bundle checks fixed artifact paths and hashes → compares journal order and full request/settings/input provenance → parses raw outputs → loads frozen predictions → calls the retained evaluate → compares metrics, errors, paired audit and report exposure. It does not load weights or regenerate responses. The optional model check hashes the supplied file.

## Requirement evidence and limits

IDs below refer to this assignment. They describe executable preparation material and observed worker checks, not personal competence, supervision or learning progress.

| ID | Implemented support | Implementation path | Success and failure test path |
| --- | --- | --- | --- |
| R09 | Runnable baseline and bounded GGUF experiments with raw command/output records. | cola_lab/local_gguf.py, scripts/validate.py | tests/test_local_gguf.py, tests/test_gguf_audit.py → test_documented_plan_attempt_and_verify_commands |
| R11 | Each repeat's language-specific baseline differences/errors; preserves malformed responses and backend failures instead of discarding them. | cola_lab/gguf_audit.py, cola_lab/local_gguf.py | tests/test_gguf_audit.py → test_saved_protocol_exists_before_backend_and_is_used_without_overrides, test_malformed_hebrew_response_retains_english_calls_and_raw_failure |
| R13 | Frozen data/inputs/settings, model/code/output hashes, environment, full journal and consistency replay. | cola_lab/local_gguf.py, cola_lab/gguf_audit.py | tests/test_gguf_audit.py → test_protocol_rejects_drift_and_overrides_before_starting_a_run, test_verifier_rejects_artifact_drift_missing_repeat_and_wrong_weights |
| R18 | Existing local GGUF adapter and bilingual pairs are runnable when authorized weights/backend are supplied. **Partial: no real-model quality run here.** | cola_lab/local_gguf.py → create_backend, run | tests/test_local_gguf.py → backend adapter spy; tests/test_gguf_audit.py → missing weights and injected success tests only |
| R25 | Reproduce the baseline, freeze a study and replay saved scoring; exercises ask the learner to explain metrics. | cola_lab/__main__.py → study-smoke; cola_lab/gguf_audit.py | tests/test_gguf_audit.py → test_finite_smoke_and_rejected_flags, test_replay_rejects_rehashed_false_metrics_and_exposure |
| R26 | A defined same-source experiment, complete repeat reporting, lexical comparison and explicit failure/interpretation limits. | cola_lab/gguf_audit.py, cola_lab/answer_eval.py | tests/test_gguf_audit.py → saved-protocol/language comparison test; tests/test_answer_eval.py → paired errors and no-gain comparison |
| R27 | Meaningful success/rejection tests, current raw validation output and source-linked records; no claim about personal progress. | cola_lab/gguf_audit.py, scripts/validate.py | tests/test_gguf_audit.py → artifact drift, interrupted generation, failed scoring and incomplete status tests |
| R36 | Retains shared label-free source inputs and pair/repeat settings; verification rejects altered exposure and keeps legacy full-context comparisons distinct. | cola_lab/answer_eval.py → method_inputs; cola_lab/local_gguf.py → generation_requests; cola_lab/gguf_audit.py | tests/test_shared_protocol.py, tests/test_gguf_audit.py → exposure drift and legacy-context tests |
| R37 | Numerical expectations for per-language accuracy/differences/errors; rejects missing repeats, changed metrics, reordered attempts, invalid protocol and partial completion. | cola_lab/gguf_audit.py, cola_lab/local_gguf.py | tests/test_gguf_audit.py, tests/test_answer_eval.py |

The earlier validation/checkpoint files remain for continuity. Current execution evidence is isolated under validation/current/; hashes link it to the implementation and test files that were actually run.
