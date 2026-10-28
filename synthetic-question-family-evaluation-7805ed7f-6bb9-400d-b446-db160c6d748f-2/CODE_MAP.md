# Code map

This extension reuses the pinned Python implementation. It does not substitute
a new framework or claim that a person has already learned the material.

| Path | Executable responsibility |
|---|---|
| `portfolio/research_cases.py` | Existing facts, questions and expected arithmetic answers; `entity_assignments` validates disjoint pools and alias ownership, and optional `entity_groups` restricts comparison partners before question generation. Default datasets remain unchanged. |
| `portfolio/research_lab.py` | Fixed pools and version, JSON validation, all-mention leakage audit, full family counts, features, majority and Naive Bayes models, metrics, validation selection, persisted decision, diagnostic analyses and CLI/report output. |
| `tests/test_research_cases.py` | Original-output fingerprints, within-pool operands with correct answers, and rejected undersized/overlapping/ambiguous pools. |
| `tests/test_research_integrity.py` | Original leak regression, secondary mentions/aliases, missing-family success/failure policy, absent-class metric arithmetic, validation freeze order, forced validation choice, malformed CLI input and artifact-tampering detection. |
| `tests/test_research_lab.py` | Reproduction, word-order representation comparison, hand-calculated metrics, paired failures, purged entity folds, label distributions, margins, resampling and invalid-input checks. |
| `tests/test_evidence_agent.py` | Existing shared-generator caller and typed arithmetic regression suite, run to check compatibility. |
| `scripts/smoke_research_lab.py` | Finite CLI execution, hash/count checks, exact input replay, installed-package replay and rejected-data runs. |
| `scripts/verify_research_lab.py` | Records real installation/test/build/smoke commands, outputs, versions and return codes. Stops after a failed step. |
| `pyproject.toml` | Installs the local Python package; setuptools is a build tool, with no third-party runtime requirements for this lab. |
| `results/research_lab/` | Regenerated cases, challenge, protocol, frozen selection, metrics and report. |
| `results/research_validation/` | Actual command manifest and raw output from this extension's verification. |

## Execution path

`SPLITS` → `entity_assignments` → `build_cases(entity_groups=SPLITS)` →
`validate_cases` → complete-family check → generator label ledger → paired
challenge → training fits → validation metrics → `selection.json` → original
test and challenge evaluation → sensitivity checks → `summary.json` and report.

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
| R02 | `portfolio/research_lab.py`: training-only fits, validation selection and fixed-label metrics | `tests/test_research_integrity.py`: selection ordering, forced validation choice and absent-family arithmetic |
| R06 | `portfolio/research_cases.py`: synthetic generation; `portfolio/research_lab.py`: quality audit | `tests/test_research_cases.py`: valid/invalid pools and correct comparison answers |
| R07 | `portfolio/research_lab.py`: executable majority, unigram and bigram experiment | `tests/test_research_lab.py`: reproducible run and baseline metrics |
| R08 | `portfolio/research_lab.py`: all-mention split checks and frozen selection | `tests/test_research_integrity.py`: split isolation, legacy leak rejection and fit/selection ordering |
| R09 | `portfolio/research_lab.py`: baseline comparisons, paired failures and diagnostic review | `tests/test_research_lab.py`: failure counts, paired consistency and rejected mismatched pairs |
| R13 | `portfolio/research_lab.py`: lexical feature representations and text classification | `tests/test_research_lab.py`: reversed-word-order case separates bigram from unigram |
| R14 | `portfolio/research_lab.py`: per-family metrics and purged/resampled stability checks | `tests/test_research_lab.py`: hand-calculated majority result and disjoint purged-fold mentions |
| R15 | `portfolio/research_cases.py`: pool-constrained generation; `portfolio/research_lab.py`: zero-aware counts | `tests/test_research_integrity.py`: secondary-only leakage, aliases, missing training/validation/test families |
| R18 | `portfolio/research_lab.py`: basic NLP counts and Naive Bayes | `tests/test_research_lab.py`: word-order representation behavior and unknown candidate rejection |
| R22 | `portfolio/research_lab.py`: versioned reproducible run and metric report | `tests/test_research_lab.py`: two identical runs; `tests/test_research_integrity.py`: artifact integrity |
| R23 | `portfolio/research_lab.py`: bounded baseline comparison, failures and limitations | `tests/test_research_lab.py`: measured baseline/family diagnostics; supports a future supervised exercise only |
| R25 | `portfolio/research_lab.py`: source/input hashes and validation record | `tests/test_research_integrity.py`: source/hash checks and rejected artifact tampering; no personal learning outcome asserted |
| R34 | `portfolio/research_lab.py`: input and metric edge-case handling | `tests/test_research_integrity.py`: zero-count, leakage, malformed input and pre-evaluation freeze regressions |

## Technical reading for a future review

The [MultinomialNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
describes discrete feature counts and additive smoothing. This small standard-
library implementation uses alpha=1; it is not a complete scikit-learn replica.
The [precision/recall/F-score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
explains macro averaging and zero denominators. Here every named family counts
once in macro F1, even when its F1 is zero. These are prepared reading links and
implementation notes, not evidence that anyone has already read or summarized
the sources. No scikit-learn dependency is introduced for these references.
