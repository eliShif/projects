# Code map

The extension reuses the pinned Python source in [UPSTREAM.md](UPSTREAM.md). The original remote is not modified.

| Path | Responsibility | Tests |
| --- | --- | --- |
| cola_lab/answer_eval.py | Existing bilingual data, validation, retrieval, threshold selection, scorer, paired audit and entity comparisons. New method_inputs strips labels; predict_from_input extracts source text; reports expose inputs and model-versus-lexical comparisons. | tests/test_answer_eval.py, tests/test_shared_protocol.py |
| cola_lab/local_gguf.py | Existing local inference/manifest, extended with shared/full context, frozen requests, pair/repeat seeds, per-call reset, exposed-citation checks, record validation and repeat outputs. | tests/test_local_gguf.py, tests/test_shared_protocol.py |
| cola_lab/__main__.py | Baseline, GGUF and retained commands; rejects misplaced flags; finite smoke checks the documented path and shared-input contract. | tests/test_shared_protocol.py → CommandTests |
| cola_lab/data.py | Retained synthetic sentences, CSV round trip, label/group checks. | tests/test_experiment.py |
| cola_lab/experiment.py | Retained bigram fitting, thresholds, metrics, descriptive statistics, errors and training-group sensitivity. | tests/test_experiment.py |
| pyproject.toml | Installable Python package and optional real llama-cpp-python extra. No new baseline runtime dependencies. | Installation/wheel commands in VALIDATION.md |
| README.md, EXERCISES.md | Setup, protocol, interpretation and future reading/change exercises. | CLI smoke and regression tests above |
| VALIDATION.md, validation/ | Current commands and raw results, separating baseline outcomes from fixture checks. | Actual executions |

## Execution boundaries

Baseline: make_dataset → saved JSON → validate_dataset → validation-only threshold → method_inputs → predict_from_input → existing metrics/audit → report.

Shared GGUF: generation_requests freezes inputs/settings → create_backend resets and receives explicit seeds → parse_generation checks actual selected evidence → validate_records recomputes requests → frozen predictions → existing answer_eval.run with the exact same dataset → checked input-exposure report → final gguf_run.json.

In all-documents, only the model input changes to the full language/split pool. Reports identify unequal evidence. Imported predictions without generation records are marked unverified; their exposure is not inferred.

The baseline no longer reads doc["answer"] to answer. Labels remain for validation/scoring. Both answerers see public question/source inputs, although the baseline knows the synthetic fact grammar. settings_for excludes language and list index from the seed key.

## Requirement evidence and limits

These are executable preparation materials, not personal ability or completed supervised work.

| ID | Implemented support | Implementation | Success/failure tests |
| --- | --- | --- | --- |
| R07 | Small Python experiment using the retained baseline and scorer. | cola_lab/answer_eval.py, cola_lab/local_gguf.py | tests/test_answer_eval.py, tests/test_local_gguf.py |
| R09 | Paired errors, improvements/regressions and direct model-versus-lexical differences. | cola_lab/answer_eval.py → paired_case_audit, compare_methods, evaluate | tests/test_answer_eval.py, tests/test_local_gguf.py |
| R11 | Weight/data/code/input/prompt/output hashes, versions, raw outputs and repeat files. | cola_lab/local_gguf.py → run, validate_records | tests/test_local_gguf.py → provenance, repeats and drift |
| R16 | Optional real local GGUF path in both languages; partial without authorized weights and real outputs. | cola_lab/local_gguf.py → create_backend, run | tests/test_local_gguf.py → injected integration and adapter spy; no real-model quality test |
| R21 | Python, packaging and experiment records; not the entire broader tool list. | cola_lab/__main__.py, cola_lab/local_gguf.py, pyproject.toml | tests/test_shared_protocol.py, tests/test_local_gguf.py |
| R22 | Existing reproducible experiment and finite verification for a future reader. | cola_lab/__main__.py → answers-smoke; EXERCISES.md | tests/test_shared_protocol.py → smoke; tests/test_answer_eval.py → repeatability |
| R23 | Bounded same-source baseline comparison, audit, limits and future research-note exercise. | cola_lab/answer_eval.py, cola_lab/local_gguf.py; EXERCISES.md | tests/test_local_gguf.py → shared/full mode comparison |
| R25 | Auditable records and rejected invalid protocols; personal learning remains unassessed. | cola_lab/local_gguf.py → validate_records, run | tests/test_local_gguf.py → drift, bad responses, nonempty output and invalid options |
| R33 | Same selected text, actual exposure, paired settings and order-independent repeat seeds. | cola_lab/answer_eval.py → method_inputs, predict_from_input; cola_lab/local_gguf.py → settings_for, generation_requests | tests/test_shared_protocol.py, tests/test_local_gguf.py → actual input equality, seed equality, unexposed citations |
| R34 | Focused correctness/rejection checks for cases, repeats, input/settings drift, frozen provenance and CLI misuse. | cola_lab/answer_eval.py, cola_lab/local_gguf.py, cola_lab/__main__.py | tests/test_answer_eval.py, tests/test_local_gguf.py, tests/test_shared_protocol.py |
