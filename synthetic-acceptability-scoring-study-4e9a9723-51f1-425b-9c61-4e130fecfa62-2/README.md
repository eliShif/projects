# Synthetic acceptability scoring study

This is an executable learning exercise related to grammatical acceptability classification and length bias in language-model scores. It does **not** use CoLA, GPT-2, a downloaded model, or any material from the previously described project. The workspace contained no project source or dataset at inspection time, so earlier project claims remain unverified.

## Research question and design

For a small, generated subject–verb agreement task, how do summed and average token log probabilities compare as classification scores? A Laplace-smoothed bigram model is fit only to grammatical **training** sentences. For each score, a threshold is chosen using **validation** balanced accuracy. The same held-out **test** examples are then scored by both methods and a train-majority baseline. The report includes accuracy, balanced accuracy, ROC AUC, confusion counts, results by short/long sentence, mean scores by length and label, and individual errors. Equal results are possible; the experiment does not assume a winner.

The generator creates 24 noun–verb groups and four labeled sentences per group. Groups are shuffled with seed 17, then allocated 12/6/6 to train/validation/test, so no group crosses splits. A CSV validator checks labels, missing values, duplicate sentences, length buckets, both labels per split, and group leakage. The actual experiment reads the saved CSV. Labels come from the generation rule, not human annotation.

## Setup, run, and verify

Python 3.10+ is sufficient. There are no third-party dependencies or downloads.

```text
python3 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q cola_lab tests
.venv/bin/python -m cola_lab run --output-dir artifacts
.venv/bin/python -m cola_lab smoke
```

`run` writes `artifacts/sentences.csv` and `artifacts/report.json`. Inspect the report's `protocol`, `baseline`, `methods.sum`, and `methods.mean` fields. `smoke` runs the same generate → CSV read → evaluate → report path in a temporary directory and exits. The test suite checks success and rejected input cases. A repeated run with the same Python version should produce the same data and scores; the report records the Python version and SHA-256 of the generated CSV.

## Reading the results

Higher log probability is treated as stronger evidence for a grammatical sentence. Summing penalizes extra tokens because each log probability is nonpositive; averaging changes that length dependence. The two methods use identical token probabilities, train data, validation data, and test data. Only score aggregation and its validation-selected threshold differ. Compare `by_length.*.mean_score` alongside accuracy and errors. Accuracy on 24 synthetic test items is a demonstration, not a performance estimate for CoLA or English generally. Score-length association alone does not prove length caused a classification error.

The bigram model has no syntax representation beyond adjacent token counts. The templates are narrow, the labels are generated, and one small split cannot establish stability. This artifact supports explaining a reproducible protocol; it is not evidence of anyone's personal mastery, prior contribution, or supervised research experience.
