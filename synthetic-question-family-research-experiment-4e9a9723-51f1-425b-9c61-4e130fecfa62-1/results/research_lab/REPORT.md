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

## Limits

- Authored templates repeat across entities; holdout does not establish new-template generalization.
- One validation and two test entities; variance describes these entities, not a confidence interval.
- English synthetic questions only; no claim about Hebrew quality or language models.
- Feature-label association does not establish causality or independent human mastery.
