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

## Train–validation gaps and overfitting

gap = training score - validation score, in fraction units (0.10 = 10 percentage points). Display rounded to six decimals; JSON retains full precision.

All candidates use the same 24 fitting cases and 16 validation cases. Training metrics reuse those fitting cases, without refitting.

| Model | Metric | Training | Validation | Train minus validation | Direction |
|---|---|---:|---:|---:|---|
| majority | accuracy | 0.125000 | 0.125000 | +0.000000 | equal |
| majority | macro_f1 | 0.027778 | 0.027778 | +0.000000 | equal |
| unigram | accuracy | 1.000000 | 1.000000 | +0.000000 | equal |
| unigram | macro_f1 | 1.000000 | 1.000000 | +0.000000 | equal |
| bigram | accuracy | 1.000000 | 1.000000 | +0.000000 | equal |
| bigram | macro_f1 | 1.000000 | 1.000000 | +0.000000 | equal |

Descriptive only; model selection still uses validation macro F1, validation accuracy and name. No refitting or test/challenge scores enter the gap calculation.

- Overfitting means learning training-specific patterns that do not generalize. High training scores with lower validation scores are consistent with it when the sets measure comparable tasks; a positive gap alone is not a diagnosis.
- Shared templates can make both scores high and conceal poor generalization to new wording. A zero gap does not rule out overfitting; equally low scores can instead indicate limited features or underfitting.
- A negative gap means validation scored higher on these cases, not that generalization is guaranteed. Accuracy and macro F1 can have different gap signs because they weight classes differently.
- Alternative explanations include wording/template shift, different label proportions or difficulty, noisy labels and small samples. Inspect template_overlap, label_distribution and failure examples before attributing a cause.
- Training scores reuse the fitting cases and are optimistic. Validation scores also serve model selection. These descriptive differences are neither confidence intervals nor significance tests; test and alternate wording remain post-selection diagnostics.

Context for this run: 16/16 validation questions reuse a training template after masking entities; validation label total variation from training is 0.000000. Exact template overlap and label proportions describe only two possible sources of similarity or shift.

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
- Two validation and two test entities; variance describes these entities, not a confidence interval.
- Training and original evaluation share authored templates; entity separation does not test unseen phrasing.
- Train-minus-validation gaps are descriptive; zero gaps cannot exclude overfitting and positive gaps cannot distinguish it from distribution shift or small-sample effects.
- Entity detection uses a closed synthetic catalogue, not general named-entity recognition; external corpora need an explicit audited entity inventory.
- Label-fraction statistics use three fixed synthetic splits; equal proportions do not imply equal wording or representative sampling.
- English synthetic questions only; no claim about Hebrew quality or language models.
- Stability folds reuse all entities after the primary selection; they are a sensitivity analysis, not a second independent test.
- Seeded resamples reuse the same 24 constructed training cases and frozen evaluation sets; their spread is not population uncertainty.
- The downsampling check changes training composition and is post-hoc; any score change cannot establish a general effect size.
- Generated labels have no independent human audit; feature and frequency clues cannot prove error cause.
- Feature-label association does not establish causality or independent human mastery.
