# Synthetic acceptability scoring study

This is an executable learning exercise related to grammatical acceptability classification and length bias in language-model scores. It does **not** use CoLA, GPT-2, or a downloaded model. Earlier project claims remain unverified.

This workspace also contains a **bilingual answer and evidence evaluation** added for a junior AI research exercise. It uses fictional English and Hebrew facts and a transparent lexical retrieval rule. It is not the previously described portfolio repository or an evaluation of a language model. The code was prepared with AI assistance; running it does not establish anyone's independent authorship or mastery.

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
.venv/bin/python -m cola_lab answers --output-dir artifacts
.venv/bin/python -m cola_lab answers-smoke
```

`run` writes `artifacts/sentences.csv` and `artifacts/report.json`. Inspect the report's `protocol`, `baseline`, `methods.sum`, and `methods.mean` fields. `smoke` runs the same generate → CSV read → evaluate → report path in a temporary directory and exits. The test suite checks success and rejected input cases. A repeated run with the same Python version should produce the same data and scores; the report records the Python version and SHA-256 of the generated CSV.

`answers` writes `artifacts/answer_dataset.json` and `artifacts/answer_report.json`; `answers-smoke` exercises that same path in a temporary directory. The dataset contains 48 questions over fictional facts, with separate validation and test entities, both languages in each split, and answerable and unanswerable cases. No download or external data permission is needed. The evaluator selects one word-overlap threshold on validation, then applies the same retrieval, answer, citation, and abstention rules to both test languages. It compares retrieval with an always-abstain baseline. A correct answer requires the exact expected answer and its evidence ID; an unanswerable case requires abstention. The report stores the data hash, implementation hash, Python version, threshold, per-language results, error types, and individual errors.

### Optional offline model comparison

If a locally available model has already produced answers for these fictional cases, save its **frozen outputs** to a UTF-8 JSON file and run:

```text
.venv/bin/python -m cola_lab answers --output-dir artifacts --predictions local_predictions.json
```

The file must contain `model_id` and `model_version` strings and a `predictions` array. Each entry has exactly `case_id`, `answer` (string or `null`), and `citations` (list of document IDs). Include each **test** case ID from `answer_dataset.json` exactly once, in any order. Use `null` and an empty citation list for abstention. For example, a single entry can be `{"case_id":"test-en-Ember-color-q","answer":"blue","citations":["test-en-Ember-color"]}`; this illustrates the format, not an observed model output. The evaluator rejects incomplete files, duplicate IDs, invalid answer fields, and citations to validation or the other language. It then adds `methods.local_model` using the same scoring as the baselines and records the input file hash and model identifiers in `protocol.model_outputs`. It does not call or download a model. Prepare the outputs before reading test labels if using them for a genuine held-out comparison; the code cannot verify how they were produced.

In the verified local run, lexical retrieval scored 22/24 on test versus 8/24 for always-abstain. Each language had one false abstention on a deliberately paraphrased color question. These numbers describe only this generated sample. The two languages use parallel templates, so equal scores here do not establish equal quality on natural questions. Word overlap cannot handle paraphrases reliably, and document answer fields make this a rule-based baseline rather than a language model.

## Reading the results

Higher log probability is treated as stronger evidence for a grammatical sentence. Summing penalizes extra tokens because each log probability is nonpositive; averaging changes that length dependence. The two methods use identical token probabilities, train data, validation data, and test data. Only score aggregation and its validation-selected threshold differ. Compare `by_length.*.mean_score` alongside accuracy and errors. Accuracy on 24 synthetic test items is a demonstration, not a performance estimate for CoLA or English generally. Score-length association alone does not prove length caused a classification error.

The bigram model has no syntax representation beyond adjacent token counts. The templates are narrow, the labels are generated, and one small split cannot establish stability. This artifact supports explaining a reproducible protocol; it is not evidence of anyone's personal mastery, prior contribution, or supervised research experience.
