# Code map

This extension reuses the pinned Python implementation. It does not substitute
a new framework or claim that a person has already learned the material.

| Path | Executable responsibility |
|---|---|
| `portfolio/research_cases.py` | Existing facts, questions and expected arithmetic answers; `entity_assignments` validates disjoint pools and alias ownership, and optional `entity_groups` restricts comparison partners before question generation. Default datasets remain unchanged. |
| `portfolio/research_lab.py` | Fixed pools and version, JSON validation, all-mention leakage audit, full family counts, features, majority and Naive Bayes models, signed metric gaps, nested train–validation learning curves, validation selection, persisted decision, diagnostic analyses and CLI/report output. |
| `tests/test_research_cases.py` | Original-output fingerprints, within-pool operands with correct answers, and rejected undersized/overlapping/ambiguous pools. |
| `tests/test_research_integrity.py` | Original leak regression, secondary mentions/aliases, missing-family success/failure policy, absent-class metric arithmetic, validation freeze order, forced validation choice, malformed CLI input and artifact-tampering detection. |
| `tests/test_research_lab.py` | Reproduction, word-order representation comparison, hand-calculated metrics, paired failures, purged entity folds, label distributions, margins, resampling and invalid-input checks. |
| `tests/test_research_learning_curve.py` | Hand-derived signed gaps, actual label-shift control, deterministic nested samples, all-family/secondary-mention isolation, infeasible size/seed rejection before fitting or output, real test-label permutation, custom CLI settings, and curve artifact tampering. |
| `tests/test_evidence_agent.py` | Existing shared-generator caller and typed arithmetic regression suite, run to check compatibility. |
| `scripts/smoke_research_lab.py` | Finite CLI execution, hash/count checks, curve metric recomputation from confusion matrices, nested sample audit, exact input replay, installed-package replay and rejected-data/size runs. |
| `scripts/verify_research_lab.py` | Records real installation/test/build/smoke commands, outputs, versions and return codes. Stops after a failed step. |
| `pyproject.toml` | Installs the local Python package; setuptools is a build tool, with no third-party runtime requirements for this lab. |
| `results/research_lab/` | Regenerated cases, challenge, protocol, frozen selection, metrics and report. |
| `results/research_validation/` | Actual command manifest and raw output from this extension's verification. |

## Execution path

`SPLITS` → `entity_assignments` → `build_cases(entity_groups=SPLITS)` →
`validate_cases` → complete-family check → generator label ledger → paired
challenge → curve subset preflight and saved protocol → full-training fits →
validation metrics → `selection.json` → original test and challenge evaluation →
train–validation learning curve and sensitivity checks → `summary.json` and report.

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

## Train–validation extension

- `learning_curve_protocol`: saves the declared sizes, seed, sampling rule,
  metric direction and diagnostic-only role. The main run writes this protocol
  before any model fit. Defaults are 8/16/24 cases and seed 19.
- `learning_curve_subsets`: reuses `validate_cases(development_only=True)` to
  reject test rows and foreign secondary mentions. It requires validation
  family coverage and at least two feasible, strictly increasing integer sizes.
  ID-sorted, seeded family queues are interleaved into nested prefixes without
  replacement. Each prefix contains every family. Input rows are not mutated.
- `learning_curve`: creates fresh existing `TextBaseline` objects at every size.
  Each candidate sees the same training IDs and complete validation set. It
  saves compact scores, confusion matrices, class counts, all mentioned roots,
  subset IDs and canonical input hashes in `summary.learning_curve`.
- `generalization_gap`: checks finite scores in [0, 1] and subtracts validation
  from training separately for accuracy and macro F1. Negative gaps survive.
  It makes no automatic causal or significance claim.
- The main report renders all declared points and interprets their limits. The
  separate alternate-wording diagnostic measures wording sensitivity; it is
  not used as a substitute for direct train–validation measurement.

`test_permuting_real_test_labels_cannot_change_selection_or_curve` performs two
real runs. The second uses custom test IDs and cyclically permuted labels with
all families still present per entity. Test accuracy changes, while the curve
and frozen selection are identical. This supplements the existing ordering test
that observes `selection.json` before test predictions. The learning curve has
no test argument and its models cannot overwrite the full-training candidates.

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

These paths identify executable preparation material for a future research
review. IDs below refer to the current assignment. R02 and R06 provide
explanation preparation only; R20 implements lexical classification and
representation comparison, not learned semantic search.

| ID | Implementation path and behavior | Test path and check |
|---|---|---|
| R01 | `portfolio/research_lab.py`: functions, mappings, JSON file input and runnable CLI | `tests/test_research_integrity.py`: malformed-file rejection; `tests/test_research_learning_curve.py`: valid and invalid CLI schedules |
| R02 | `portfolio/research_lab.py`: explained sampling and metric functions; `EXERCISES.md`: trace and hand-calculation prompts | `tests/test_research_learning_curve.py`: numerical and control-flow examples to explain; no personal explanation assessed |
| R03 | `portfolio/research_lab.py`: supervised fitting, fixed validation, signed learning-curve gaps and qualified overfitting interpretation | `tests/test_research_learning_curve.py`: hand-derived accuracy/F1 gaps, real label-shift control, test-label independence and failure cases |
| R06 | `portfolio/research_lab.py`: generated question/protocol/choices/limits in the report; `README.md` and `EXERCISES.md`: explanation preparation | `tests/test_research_lab.py`: report and run checks; `tests/test_research_learning_curve.py`: saved settings and report checks; no personal explanation assessed |
| R09 | `portfolio/research_cases.py`: synthetic generation; `portfolio/research_lab.py`: input and label-distribution audits | `tests/test_research_cases.py`: valid/invalid pools and correct comparison answers; `tests/test_research_lab.py`: label-distribution arithmetic |
| R10 | `portfolio/research_lab.py`: executable majority, unigram and bigram experiments and repeated subset fits | `tests/test_research_lab.py`: reproducible run; `tests/test_research_learning_curve.py`: re-evaluated subset scores |
| R11 | `portfolio/research_lab.py`: all-mention split isolation and saved validation-only selection | `tests/test_research_integrity.py`: fit/freeze order; `tests/test_research_learning_curve.py`: real test-label permutation leaves choice unchanged |
| R12 | `portfolio/research_lab.py`: method/size comparisons, paired wording failures, feature clues and limitations | `tests/test_research_lab.py`: error counts, paired consistency and rejected mismatched pairs |
| R14 | `portfolio/research_lab.py`: versions, input/source hashes, seeded subset IDs, declared sizes and selection record; `scripts/verify_research_lab.py`: actual command logs | `tests/test_research_learning_curve.py`: deterministic replay and artifact tampering; `tests/test_research_integrity.py`: source/hash integrity |
| R17 | `portfolio/research_lab.py`: tokenization, count features and Naive Bayes (NLP alternative) | `tests/test_research_lab.py`: word-order behavior and unknown candidate rejection |
| R20 | `portfolio/research_lab.py`: lexical classification with unigram and unigram-plus-bigram features | `tests/test_research_lab.py`: reversed-word-order comparison; no semantic-search claim |
| R21 | `portfolio/research_lab.py`: per-family metrics, baselines, error review, purged folds and training resamples | `tests/test_research_lab.py`: hand-calculated metrics, fold isolation and resampling controls |
| R22 | `portfolio/research_lab.py`: schema/label audits, zero-aware counts, imbalance and all-mention isolation | `tests/test_research_integrity.py`: secondary/alias leakage and missing-family rejection; `tests/test_research_lab.py`: provenance mismatch |
| I01 | `portfolio/research_lab.py`: all candidates use the same subset and fixed validation rows; primary comparisons use the same test cases | `tests/test_research_learning_curve.py`: recorded fitting IDs and independently re-evaluated metrics; `tests/test_research_lab.py`: paired-case checks |
| I02 | `portfolio/research_lab.py`: finite-score, schedule, schema and split checks; `scripts/smoke_research_lab.py`: recomputed curve counts/gaps | `tests/test_research_learning_curve.py`: nonfinite/missing scores, invalid sizes/seeds, leakage and tampering; `tests/test_research_integrity.py`: existing edge cases |

## Technical reading for a future review

The [MultinomialNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
describes discrete feature counts and additive smoothing. This small standard-
library implementation uses alpha=1; it is not a complete scikit-learn replica.
The [precision/recall/F-score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
explains macro averaging and zero denominators. Here every named family counts
once in macro F1, even when its F1 is zero. These are prepared reading links and
implementation notes, not evidence that anyone has already read or summarized
the sources. No scikit-learn dependency is introduced for these references.
