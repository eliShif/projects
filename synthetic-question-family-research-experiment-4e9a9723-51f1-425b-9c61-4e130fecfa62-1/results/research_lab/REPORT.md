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

## Limits

- The alternate wording was authored after inspecting the original templates; this diagnostic is exploratory, not an independent benchmark.
- One validation and two test entities; variance describes these entities, not a confidence interval.
- English synthetic questions only; no claim about Hebrew quality or language models.
- Feature-label association does not establish causality or independent human mastery.
