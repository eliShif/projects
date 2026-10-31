# Synthetic question-family experiment

Can lexical features classify a synthetic research question's family for unseen service entities?

Experiment version: 2.1.0. Python: 3.14.4. Source and input hashes: `protocol.json`.

Entity pools are allocated before generation. Every recognized mention, including secondary services and Sandbox/Mirror aliases, must belong to its row's split. `summary.json` lists the audited mentions for each case.

## Data quality

| Split | Cases | Entities | Largest/smallest label count |
|---|---:|---:|---:|
| train | 24 | 3 | 1.00 |
| validation | 16 | 2 | 1.00 |
| test | 16 | 2 | 1.00 |

## Label distribution across splits

Fractions account for the different split sizes. Total variation is half the sum of absolute label-fraction differences from training; zero means identical proportions.

| Family | Train | Validation | Test | Mean split fraction | Population variance |
|---|---:|---:|---:|---:|---:|
| lookup | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| conversion | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| difference | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| sum | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| ratio | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| missing_field | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| unknown_service | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |
| conflict | 0.125 | 0.125 | 0.125 | 0.125 | 0.0000 |

Total variation from training: validation 0.000; test 0.000.
The mean and variance treat each fixed split equally; they are descriptive and do not estimate population uncertainty. Even identical label proportions cannot rule out wording shift: the separate template audit checks one form of that shift.

Generator label/entity ledger: 56 matching IDs; 0 custom IDs without a source match. This checks provenance, not independent annotation quality.

## Template reuse audit

The audit masks known service names, retaining Sandbox/Mirror suffixes, then checks exact normalized wording against training. It detects one constructed shortcut, not semantic similarity.

| Evaluation set | Training templates seen | Cases | Fraction |
|---|---:|---:|---:|
| validation | 16 | 16 | 1.000 |
| test | 16 | 16 | 1.000 |
| challenge | 0 | 16 | 0.000 |

A high overlap can make entity-held-out accuracy optimistic; a lower challenge score is consistent with wording sensitivity, without identifying the cause of each error.

## Train–validation learning curve

Predeclared training sizes: [8, 16, 24]; seed: 19. Each nested subset retains all eight families, with no replacement. The same validation cases are used at every point; every known entity mention stays in its original pool.

Each method is fit afresh on a subset and evaluated on that same subset (resubstitution) and on validation. These diagnostic refits do not change the full-training validation choice. Exact subset IDs, class counts, confusion matrices and input hashes are in `summary.learning_curve`.

Gaps are signed train minus validation scores, in score units (0.10 = 10 percentage points). A positive gap means training scored higher; negative values are retained.

| Training cases | Method | Train accuracy | Validation accuracy | Accuracy gap | Train macro F1 | Validation macro F1 | Macro F1 gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 8 | majority | 0.125 | 0.125 | +0.000 | 0.028 | 0.028 | +0.000 |
| 8 | unigram | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 8 | bigram | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 16 | majority | 0.125 | 0.125 | +0.000 | 0.028 | 0.028 | +0.000 |
| 16 | unigram | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 16 | bigram | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 24 | majority | 0.125 | 0.125 | +0.000 | 0.028 | 0.028 | +0.000 |
| 24 | unigram | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 24 | bigram | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |

Observed: 0 of 9 method/size pairs have a positive gap in at least one metric.

Overfitting means fitting training-specific patterns that fail to generalize. A high training score with a lower validation score is consistent with it, but a gap alone cannot distinguish overfitting from sampling variation or distribution shift. Shared templates can hide a gap; zero gaps do not demonstrate broad generalization. Low scores on both sets can suggest underfitting.

The separate alternate-wording challenge changes phrasing and measures sensitivity to that change. Its score drop alone is not an overfitting diagnosis. This small, fixed validation set and nested training samples do not support significance or causal claims; no improvement or gap was required by the protocol.

## Results

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| majority | 0.028 | 0.125 | 0.028 |
| unigram | 1.000 | 1.000 | 1.000 |
| bigram | 1.000 | 1.000 | 1.000 |

Selected on validation: **bigram**. The choice and validation scores were written to `selection.json` before any test prediction; primary models were fit on training only and were not refit after selection.

All eight families must occur in every split for this run. The audit reports absent families explicitly and uses null for an undefined largest/smallest ratio. Metrics retain zero for undefined precision/recall/F1 denominators.

Test errors for selected model:

- None in this small synthetic test.

## Per-family held-out evaluation

Support counts true cases; predicted counts assigned cases. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. An undefined denominator is reported as zero. Macro F1 averages all eight family F1 values, including zero for missed families; accuracy weights individual cases.

| Model | Family | Support | Predicted | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| majority | lookup | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | conversion | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | difference | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | sum | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | ratio | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | missing_field | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | unknown_service | 2 | 0 | 0.000 | 0.000 | 0.000 |
| majority | conflict | 2 | 16 | 0.125 | 1.000 | 0.222 |
| bigram | lookup | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | conversion | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | difference | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | sum | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | ratio | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | missing_field | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | unknown_service | 2 | 2 | 1.000 | 1.000 | 1.000 |
| bigram | conflict | 2 | 2 | 1.000 | 1.000 | 1.000 |

The majority baseline shows why one aggregate accuracy cannot describe which families are missed. These fixed synthetic cases do not estimate population performance.

## Class score margin distribution

For each Naive Bayes model, the margin is the largest minus second-largest unnormalized class log score. The table describes fixed cases; larger margins are not calibrated probabilities or proof that a prediction is correct. The majority baseline has no class log scores.

| Model | Set | Cases | Mean margin | Population variance | Wrong cases | Wrong-case mean |
|---|---|---:|---:|---:|---:|---:|
| unigram | validation | 16 | 7.529 | 11.606 | 0 | n/a |
| unigram | test | 16 | 7.529 | 11.606 | 0 | n/a |
| unigram | challenge | 16 | 2.307 | 1.770 | 6 | 2.843 |
| bigram | validation | 16 | 14.687 | 31.459 | 0 | n/a |
| bigram | test | 16 | 14.687 | 31.459 | 0 | n/a |
| bigram | challenge | 16 | 3.380 | 7.421 | 6 | 4.332 |

Per-case margins and correctness flags are in `summary.json`. These scores reflect constructed feature-label associations; this comparison does not identify a causal effect of wording changes. Because question lengths and vocabularies vary, compare margins within a fitted model cautiously.

## New-template diagnostic (same held-out entities and labels)

| Model | Original test accuracy | New-template accuracy | Change |
|---|---:|---:|---:|
| majority | 0.125 | 0.125 | +0.000 |
| unigram | 1.000 | 0.625 | -0.375 |
| bigram | 1.000 | 0.625 | -0.375 |

Paired outcomes for the validation-selected model:

| Original → alternate | Cases |
|---|---:|
| correct → correct | 10 |
| correct → wrong | 6 |
| wrong → correct | 0 |
| wrong → wrong | 0 |

| Family | Original-correct → alternate-wrong |
|---|---:|
| lookup | 2 |
| conversion | 0 |
| difference | 2 |
| sum | 0 |
| ratio | 0 |
| missing_field | 0 |
| unknown_service | 0 |
| conflict | 2 |

New-template errors for selected model:

- `challenge-test-Garnet-lookup`: expected lookup, predicted unknown_service; State the audit log storage duration, in days, for Garnet.
- `challenge-test-Garnet-difference`: expected difference, predicted unknown_service; In days, what gap separates the log storage durations of Garnet and Helix?
- `challenge-test-Garnet-conflict`: expected conflict, predicted unknown_service; Resolve the log storage duration for GarnetMirror despite two equally current records.
- `challenge-test-Helix-lookup`: expected lookup, predicted unknown_service; State the audit log storage duration, in days, for Helix.
- `challenge-test-Helix-difference`: expected difference, predicted unknown_service; In days, what gap separates the log storage durations of Helix and Garnet?
- `challenge-test-Helix-conflict`: expected conflict, predicted unknown_service; Resolve the log storage duration for HelixMirror despite two equally current records.

## Group and training-composition sensitivity

The selected method stays fixed. Each fold holds one primary entity's cases out and removes every remaining training case mentioning any service used by that holdout. Excluded IDs and disjoint mention sets are recorded in the summary. These folds reuse the primary data and are exploratory.

| Held-out entity | Accuracy | Macro F1 |
|---|---:|---:|
| Cobalt | 1.000 | 1.000 |
| Delta | 1.000 | 1.000 |
| Ember | 1.000 | 1.000 |
| Fjord | 1.000 | 1.000 |
| Garnet | 1.000 | 1.000 |
| Helix | 1.000 | 1.000 |
| Ibis | 1.000 | 1.000 |

Accuracy range: 1.000–1.000; mean: 1.000; population variance: 0.0000.

Twelve seeded, label-stratified training resamples refit the validation-selected method while keeping test and alternate wording fixed. Duplicate training cases can occur; these are sensitivity runs, not independent samples or a confidence interval.

| Evaluation | Accuracy min | Mean | Max | Population variance |
|---|---:|---:|---:|---:|
| Original test | 1.000 | 1.000 | 1.000 | 0.0000 |
| Alternate wording | 0.625 | 0.625 | 0.625 | 0.0000 |

Per-case error counts and sampled training IDs for each seed are in `summary.json`.

Downsampling `lookup` from 3 to 1 training case changes held-out macro F1 from 1.000 to 0.833; minority recall changes from 1.000 to 0.000.
This post-hoc stress check does not change model selection or the primary score.

## Error review: distinguish observations from possible causes

Labels come from the synthetic family generator; there is no independent annotation audit. The held-out challenge label counts and training counts are recorded in `summary.json`. Unseen feature fractions can indicate wording shift, but cannot establish whether a failure is due to label quality, data composition or the classifier.

- `challenge-test-Garnet-lookup`: lookup → unknown_service; unseen feature fraction 0.47; training examples for true label 3. Annotation unverified independently.
- `challenge-test-Garnet-difference`: difference → unknown_service; unseen feature fraction 0.68; training examples for true label 3. Annotation unverified independently.
- `challenge-test-Garnet-conflict`: conflict → unknown_service; unseen feature fraction 0.83; training examples for true label 3. Annotation unverified independently.
- `challenge-test-Helix-lookup`: lookup → unknown_service; unseen feature fraction 0.47; training examples for true label 3. Annotation unverified independently.
- `challenge-test-Helix-difference`: difference → unknown_service; unseen feature fraction 0.68; training examples for true label 3. Annotation unverified independently.
- `challenge-test-Helix-conflict`: conflict → unknown_service; unseen feature fraction 0.83; training examples for true label 3. Annotation unverified independently.

## Limits

- The alternate wording was authored after inspecting the original templates; this diagnostic is exploratory, not an independent benchmark.
- Learning-curve training scores reuse fitted examples; nested subsets and a reused tiny validation set are not independent trials or confidence intervals.
- A positive train-minus-validation gap is consistent with overfitting, but sampling variation or distribution differences can also contribute; a zero gap does not prove generalization.
- Alternate wording measures sensitivity to a changed input distribution, not by itself overfitting; the curve compares training and validation directly without tuning on test.
- Two validation and two test entities; variance describes these entities, not a confidence interval.
- Training and original evaluation share authored templates; entity separation does not test unseen phrasing.
- Entity detection uses a closed synthetic catalogue, not general named-entity recognition; external corpora need an explicit audited entity inventory.
- Label-fraction statistics use three fixed synthetic splits; equal proportions do not imply equal wording or representative sampling.
- English synthetic questions only; no claim about Hebrew quality or language models.
- Stability folds reuse all entities after the primary selection; they are a sensitivity analysis, not a second independent test.
- Seeded resamples reuse the same 24 constructed training cases and frozen evaluation sets; their spread is not population uncertainty.
- The downsampling check changes training composition and is post-hoc; any score change cannot establish a general effect size.
- Generated labels have no independent human audit; feature and frequency clues cannot prove error cause.
- Feature-label association does not establish causality.
