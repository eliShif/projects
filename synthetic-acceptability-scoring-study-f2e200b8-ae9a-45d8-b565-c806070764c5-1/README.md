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

`run` writes `artifacts/sentences.csv`, `artifacts/report.json`, and a generated `artifacts/REPORT.md`. Inspect `data_quality`, `baseline`, `methods.sum`, `methods.mean`, and `training_sensitivity` in the JSON, then read the generated report for the same results in one place. `smoke` runs the same generate → CSV read → evaluate → report path in a temporary directory and exits after checking the new analysis is present. The test suite checks success and rejected input cases. A repeated run with the same Python version should produce the same data and scores; the report records the Python version and SHA-256 of the generated CSV.

The extension reports label and length distributions per split, population mean and variance of test scores by method, length, and label, held-out errors, and a 24-draw training-composition sensitivity check. `methods.*.score_histogram` adds five equal-width score bins per method, with counts for each length and label combination. The bin edges use the minimum and maximum of that method's held-out scores, shared across its four slices; the final bin includes the maximum. Compare counts within one method because sum and mean have different score scales. Each seeded draw samples whole training groups with replacement, refits the bigram model, selects a threshold on the unchanged validation split, and scores the unchanged test split. The JSON records sampled groups, accuracy and balanced accuracy by draw, aggregate statistics, and per-sentence error frequency. The readable report includes the same observations and a source note based on Python's `statistics.pvariance` documentation. These are descriptive values over a tiny generated sample, not confidence intervals. The score-length relationship does not establish that length caused a failure.

`answers` writes `artifacts/answer_dataset.json` and `artifacts/answer_report.json`; `answers-smoke` exercises that same path in a temporary directory. The dataset contains 48 questions over fictional facts, with separate validation and test entities, both languages in each split, and answerable and unanswerable cases. No download or external data permission is needed. The evaluator selects one word-overlap threshold on validation, then applies the same retrieval, answer, citation, and abstention rules to both test languages. It compares retrieval with an always-abstain baseline. A correct answer requires the exact expected answer and its evidence ID; an unanswerable case requires abstention. The report stores the data hash, implementation hash, Python version, threshold, per-language results, error types, and individual errors.

The answer report also records `protocol.data_quality` (answerable/unanswerable counts and entity membership by split and language). Each method reports `by_entity` results and `paired_languages`: counts of matched English/Hebrew questions where both answers are correct, both are wrong, or only one language is correct. Validation rejects a missing language counterpart even if the remaining cases still have both answer types. These slices expose imbalances and failures; four test entities and parallel templates are too small to establish model stability or natural-language parity.

The validator also rejects a document whose recorded answer is absent from its source sentence, including when a case label has been changed to agree with the bad document label. This is a narrow consistency check for the generated facts, not independent human annotation review. `comparisons.lexical_retrieval` reports the matched accuracy difference from always abstaining, one difference per test entity, and a descriptive 95% percentile interval from all 256 resamples of the four whole entities. Whole-entity resampling keeps each entity's English and Hebrew cases together. On the fixed sample the observed difference is 14/24; the interval describes sensitivity to these four entities only. It is not a population confidence interval or evidence of language parity. If frozen local-model outputs are supplied, `comparisons.local_model` uses the same reference and resampling rule.

### Optional offline model comparison

If a locally available model has already produced answers for these fictional cases, save its **frozen outputs** to a UTF-8 JSON file and run:

```text
.venv/bin/python -m cola_lab answers --output-dir artifacts --predictions local_predictions.json
```

The file must contain `model_id` and `model_version` strings and a `predictions` array. Each entry has exactly `case_id`, `answer` (string or `null`), and `citations` (list of document IDs). Include each **test** case ID from `answer_dataset.json` exactly once, in any order. Use `null` and an empty citation list for abstention. For example, a single entry can be `{"case_id":"test-en-Ember-color-q","answer":"blue","citations":["test-en-Ember-color"]}`; this illustrates the format, not an observed model output. The evaluator rejects incomplete files, duplicate IDs, invalid answer fields, and citations to validation or the other language. It then adds `methods.local_model` using the same scoring as the baselines and records the input file hash and model identifiers in `protocol.model_outputs`. It does not call or download a model. Prepare the outputs before reading test labels if using them for a genuine held-out comparison; the code cannot verify how they were produced.

In the verified local run, lexical retrieval scored 22/24 on test versus 8/24 for always-abstain. Each language had one false abstention on a deliberately paraphrased color question. These numbers describe only this generated sample. The two languages use parallel templates, so equal scores here do not establish equal quality on natural questions. Word overlap cannot handle paraphrases reliably, and document answer fields make this a rule-based baseline rather than a language model.

## Reading the results

Higher log probability is treated as stronger evidence for a grammatical sentence. Summing penalizes extra tokens because each log probability is nonpositive; averaging changes that length dependence. The two methods use identical token probabilities, train data, validation data, and test data. Only score aggregation and its validation-selected threshold differ. Compare `by_length.*.mean_score`, the histogram counts, accuracy, and errors. Overlap between label counts in a bin shows that those scores are not separated at that resolution; a bin does not prove individual predictions are wrong. Accuracy on 24 synthetic test items is a demonstration, not a performance estimate for CoLA or English generally. The 24 training refits show sensitivity to training composition on unchanged validation and test cases, not population uncertainty. Score-length association alone does not prove length caused a classification error.

The bigram model has no syntax representation beyond adjacent token counts. The templates are narrow, the labels are generated, and one small split cannot establish stability. This artifact supports explaining a reproducible protocol; it is not evidence of anyone's personal mastery, prior contribution, or supervised research experience.
