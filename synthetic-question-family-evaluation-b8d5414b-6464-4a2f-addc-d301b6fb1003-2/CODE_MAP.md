# Code map

This extension reuses the pinned Python implementation. It does not substitute
a new framework or claim that a person has already learned the material.

| Path | Executable responsibility |
|---|---|
| `portfolio/research_cases.py` | Existing facts, questions and expected arithmetic answers; `entity_assignments` validates disjoint pools and alias ownership, and optional `entity_groups` restricts comparison partners before question generation. Default datasets remain unchanged. |
| `portfolio/research_lab.py` | Fixed pools and version, JSON validation, leakage audit, family counts, features, majority and Naive Bayes models, training/validation metrics, `train_validation_review`, validation selection, persisted decision, diagnostics and CLI/report output. |
| `tests/test_research_generalization.py` | Known confusion results with signed gaps, invalid/missing scores, report integration, selection independent of training metrics and changed test wording, and tampered-gap/metric rejection. |
| `tests/test_research_cases.py` | Original-output fingerprints, within-pool operands with correct answers, and rejected undersized/overlapping/ambiguous pools. |
| `tests/test_research_integrity.py` | Original leak regression, secondary mentions/aliases, missing-family success/failure policy, absent-class metric arithmetic, validation freeze order, forced validation choice, malformed CLI input and artifact-tampering detection. |
| `tests/test_research_lab.py` | Reproduction, word-order representation comparison, hand-calculated metrics, paired failures, purged entity folds, label distributions, margins, resampling and invalid-input checks. |
| `tests/test_evidence_agent.py` | Existing shared-generator caller and typed arithmetic regression suite, run to check compatibility. |
| `scripts/smoke_research_lab.py` | Finite CLI execution, hash/count checks, independent metric arithmetic from confusion matrices, signed-gap checks, exact input replay, installed-package replay and rejected-data runs. |
| `scripts/verify_research_lab.py` | Records real installation/test/build/smoke commands, outputs, versions, artifact digests and return codes; generates VALIDATION.md. Stops after a failed step. |
| `pyproject.toml` | Installs the local Python package; setuptools is a build tool, with no third-party runtime requirements for this lab. |
| `results/research_lab/` | Regenerated cases, challenge, protocol, frozen selection, metrics and report. |
| `results/research_validation/` | Actual command manifest and raw output from this extension's verification. |

## Execution path

`SPLITS` → `entity_assignments` → `build_cases(entity_groups=SPLITS)` →
`validate_cases` → complete-family check → generator label ledger → paired
challenge → training fits → training and validation metrics → `selection.json`
→ descriptive train–validation gaps → original test and challenge evaluation
→ sensitivity checks → `summary.json` and report.

`mentioned_entities` scans whole known service names case-insensitively and maps
Sandbox/Mirror aliases to their base pool. It checks question text rather than
trusting only primary-entity metadata. `quality.label_counts` always has eight
keys; `missing_labels` makes absence explicit. An absent evaluation class makes
the audit's imbalance ratio null. `run` requires complete family support and
rejects absent classes before fitting or writing artifacts.

`TextBaseline.fit` learns vocabulary, class priors and token counts from training
only. The unigram model discards word order; bigram adds adjacent pairs while
retaining unigrams. Both use add-one smoothing and ignore unseen features.
`evaluate` reports accuracy, an eight-label macro F1, per-family support,
predicted count, precision, recall and F1, confusion matrices, per-entity
accuracy/variance and every failure. Undefined metric denominators map to zero.

`selection.json` records the validation-only ranking inputs, selected method,
fit/validation IDs, hashes and smoothing/tie-break rules. It is written before
test predictions. The control-flow tests observe that ordering and force a
majority validation winner even though its actual test accuracy is lower. Such
controlled test fixtures are not experiment results.

`train_validation_review(training, validation)` consumes the existing `evaluate`
outputs for all three candidates and checks that both aggregate metrics are
finite numbers in [0, 1]. For each metric it saves both scores, their signed
difference (train minus validation), and the observed direction. Negative gaps
are retained and no rounding occurs before serialization. The full training
results, including confusion matrices and failures, are saved separately under
`summary.training`. The report connects the gaps to template overlap and label
proportions; it does not infer an overfitting cause or a confidence interval.
This helper receives no test scores and cannot refit or replace the winner.

`check_metrics` in the smoke script independently recalculates accuracy and
macro F1 from confusion counts for each candidate and split. `check_artifacts`
then checks the duplicated train/validation scores, signed gaps and directions.
The integration test changes only held-out question wording and verifies that
the selection artifact and the train–validation review stay identical despite
different test results.

The retained diagnostics operate after selection:

- `grouped_stability`: hold out one primary entity's cases, then purge every
  remaining case sharing **any** held-out reference. Save excluded IDs and both
  disjoint mention sets. Seven folds reuse the primary data; no model selection
  happens within them.
- `resample_stability`: twelve seeded, label-stratified training resamples with
  replacement; method and evaluation sets stay fixed. Record sampled IDs, error
  counts, range, mean and population variance.
- `imbalance_stress`: downsample one training family to one example, then compare
  scores on the unchanged test set.
- `paired_wording_review`: verify ID/entity/label/split correspondence and count
  all four correct/wrong transitions.
- `diagnostic_review`: show unseen features and training label counts for errors
  as observations, without assigning a cause.
- `template_overlap`: mask service names and compare exact normalized strings;
  detect template reuse without claiming semantic matching.
- `label_distribution_review` and `score_margin_review`: describe label fractions
  and unnormalized top-two log-score margins. Neither establishes uncertainty,
  calibrated confidence or causality.

## Requirement evidence

These paths demonstrate executable preparation material, not personal mastery,
reading, mentorship or an already-completed research review.

| ID | Implementation path and behavior | Test path and check |
|---|---|---|
| R01 | `portfolio/research_lab.py`: functions, mappings, file input and runnable CLI | `tests/test_research_integrity.py`: malformed-file CLI rejection and reproducible artifact inspection |
| R02 | `portfolio/research_lab.py`: supervised fits, fixed-label metrics and explicit train–validation gap/overfitting analysis | `tests/test_research_generalization.py`: hand-derived gaps, signed directions, report integration and selection controls; `tests/test_research_integrity.py`: training-only fits and validation freeze |
| R08 | `portfolio/research_cases.py`: seeded synthetic generation; `portfolio/research_lab.py`: quality audit | `tests/test_research_cases.py`: reproducible generator, valid/invalid pools and correct comparison answers |
| R09 | `portfolio/research_lab.py`: executable majority, unigram and bigram experiment | `tests/test_research_lab.py`: reproducible run and baseline metrics |
| R10 | `portfolio/research_lab.py`: all-mention split checks and validation-only frozen selection | `tests/test_research_integrity.py`: split isolation and fit/selection ordering; `tests/test_research_generalization.py`: test-wording independence |
| R11 | `portfolio/research_lab.py`: baseline comparisons, paired failures, gap interpretation and diagnostic review | `tests/test_research_lab.py`: failure counts, paired consistency and rejected mismatched pairs |
| R13 | `portfolio/research_lab.py`: versioned input/source hashes, frozen selection, metrics and report; `scripts/verify_research_lab.py`: command/artifact provenance | `tests/test_research_lab.py`: reproducible runs; `tests/test_research_integrity.py`: hash checks and tampering rejection |
| R16 | `portfolio/research_lab.py`: lexical text classification and unigram/bigram representations; no semantic search claim | `tests/test_research_lab.py`: reversed-word-order case separates bigram from unigram |
| R17 | `portfolio/research_lab.py`: per-family metrics and purged/resampled stability checks | `tests/test_research_lab.py`: hand-calculated majority result, purged folds and seeded resampling |
| R18 | `portfolio/research_cases.py`: pool-constrained generation; `portfolio/research_lab.py`: label and missing-data audits | `tests/test_research_integrity.py`: secondary-only leakage, aliases and missing families; `tests/test_research_lab.py`: label conflicts and missing fields |
| R21 | `portfolio/research_lab.py`: basic NLP token counts and multinomial Naive Bayes | `tests/test_research_lab.py`: word-order representation behavior and unknown candidate rejection |
| I01 | `portfolio/research_lab.py`: candidates share training and evaluation cases; matched original/alternate wording review | `tests/test_research_lab.py`: paired case correspondence, outcome counts and mismatch rejection |
| I02 | `portfolio/research_lab.py`: input/metric validation; `scripts/smoke_research_lab.py`: independent count-to-metric arithmetic | `tests/test_research_generalization.py`: known gaps, invalid scores and corrupted artifacts; `tests/test_research_integrity.py`: missing labels, leakage and malformed files |

IDs above follow this adaptation's assigned requirements. The new R02 work is
implemented and tested; the other entries identify retained executable behavior.
These mappings describe technical artifacts, not an assessment of personal skill.

## Technical reading for a future review

The [MultinomialNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
describes discrete feature counts and additive smoothing. This small standard-
library implementation uses alpha=1; it is not a complete scikit-learn replica.
The [precision/recall/F-score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
explains macro averaging and zero denominators. Here every named family counts
once in macro F1, even when its F1 is zero. These are prepared reading links and
implementation notes, not evidence that anyone has already read or summarized
the sources. No scikit-learn dependency is introduced for these references.
