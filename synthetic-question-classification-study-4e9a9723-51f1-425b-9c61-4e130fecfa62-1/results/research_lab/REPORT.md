# Synthetic question-family experiment

Can lexical features classify a synthetic research question's family for unseen service entities?

## Data quality

| Split | Cases | Entities | Largest/smallest label count |
|---|---:|---:|---:|
| train | 24 | 3 | 1.00 |
| validation | 8 | 1 | 1.00 |
| test | 16 | 2 | 1.00 |

## Results

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| majority | 0.028 | 0.125 | 0.028 |
| unigram | 1.000 | 1.000 | 1.000 |
| bigram | 1.000 | 1.000 | 1.000 |

Selected on validation: **bigram**.

Test errors for selected model:

- None in this small synthetic test.

## New-template diagnostic (same held-out entities and labels)

| Model | Original test accuracy | New-template accuracy | Change |
|---|---:|---:|---:|
| majority | 0.125 | 0.125 | +0.000 |
| unigram | 1.000 | 0.625 | -0.375 |
| bigram | 1.000 | 0.625 | -0.375 |

New-template errors for selected model:

- `challenge-test-Garnet-lookup`: expected lookup, predicted unknown_service; State the audit log storage duration, in days, for Garnet.
- `challenge-test-Garnet-difference`: expected difference, predicted unknown_service; In days, what gap separates the log storage durations of Garnet and Helix?
- `challenge-test-Garnet-conflict`: expected conflict, predicted unknown_service; Resolve the log storage duration for GarnetMirror despite two equally current records.
- `challenge-test-Helix-lookup`: expected lookup, predicted unknown_service; State the audit log storage duration, in days, for Helix.
- `challenge-test-Helix-difference`: expected difference, predicted unknown_service; In days, what gap separates the log storage durations of Helix and Garnet?
- `challenge-test-Helix-conflict`: expected conflict, predicted unknown_service; Resolve the log storage duration for HelixMirror despite two equally current records.

## Group and training-composition sensitivity

The selected method stays fixed. Leave-one-entity-out folds retrain it on the other five entities; these folds reuse the primary data and are exploratory.

| Held-out entity | Accuracy | Macro F1 |
|---|---:|---:|
| Cobalt | 1.000 | 1.000 |
| Delta | 1.000 | 1.000 |
| Ember | 1.000 | 1.000 |
| Fjord | 1.000 | 1.000 |
| Garnet | 1.000 | 1.000 |
| Helix | 1.000 | 1.000 |

Accuracy range: 1.000–1.000.

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
- One validation and two test entities; variance describes these entities, not a confidence interval.
- English synthetic questions only; no claim about Hebrew quality or language models.
- Stability folds reuse all entities after the primary selection; they are a sensitivity analysis, not a second independent test.
- The downsampling check changes training composition and is post-hoc; any score change cannot establish a general effect size.
- Generated labels have no independent human audit; feature and frequency clues cannot prove error cause.
- Feature-label association does not establish causality or independent human mastery.
