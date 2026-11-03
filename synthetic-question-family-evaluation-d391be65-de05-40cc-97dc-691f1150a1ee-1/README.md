# Synthetic question-family evaluation

Evaluate whether lexical classifiers can assign synthetic English questions to eight task families for held-out service entities, while detecting entity leakage, wording sensitivity and train–validation performance gaps.

## What it does

An offline Python evaluation lab for eight synthetic question families, combining lexical baseline comparisons, reproducible learning curves, entity-leakage checks and reports on accuracy, macro-F1 and wording sensitivity.

Compare majority, unigram and unigram-plus-bigram baselines; measure accuracy and macro-F1 across reproducible nested training subsets; preserve validation-only selection; and document template reuse, classification failures and sensitivity to training composition.

## Solution and technologies

An offline Python standard-library pipeline generates 56 questions across disjoint entity pools: 24 training, 16 validation and 16 test cases. JSON validation checks schema, family coverage, duplicates and primary, secondary and alias entity mentions. Models comprise a majority baseline and add-one-smoothed multinomial Naive Bayes using unigram or unigram-plus-bigram counts. Validation metrics and deterministic tie-breaking select the model, with the decision saved before test predictions. Seed 19 defines nested 8/16/24-case learning curves against fixed validation data. JSON and Markdown artifacts retain protocols, source/input hashes, metrics and diagnostic failures for reproducible replay.

Python · Python standard library · JSON · unittest

## Quick start

```text
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps .
```

```text
.venv/bin/python -m unittest discover -s tests -p "test_research*.py" -v
.venv/bin/python -m unittest discover -s tests -p test_evidence_agent.py -v
.venv/bin/python -m compileall -q portfolio tests scripts
.venv/bin/python -m portfolio.research_lab --out results/research_lab
.venv/bin/python scripts/smoke_research_lab.py --out results/research_lab --installed
```

## Results

The completed version 2.1.0 experiment selected unigram-plus-bigram Naive Bayes by deterministic tie-break after both lexical models reached validation macro-F1 1.0. Both achieved test accuracy and macro-F1 1.0, classifying 16/16 questions correctly; majority achieved 2/16 with macro-F1 1/36. Alternate wording reduced both lexical models to 10/16 correct (62.5%). The selected model’s six failures were lookup, difference and conflict questions classified as unknown_service.

At all three training sizes, both lexical models achieved training and validation accuracy/macro-F1 1.0; all methods had zero signed gaps. Every original validation and test template also appeared in training after entity masking. Seven purged entity folds scored 1.0 accuracy and macro-F1; twelve training resamples retained 100% original-test and 62.5% alternate-wording accuracy. Post-hoc lookup downsampling from three examples to one reduced held-out macro-F1 from 1.0 to 0.833 and lookup recall from 1.0 to 0.0.

These are stored experiment measurements. This documentation refresh produced no new measurements.

## Scope and limitations

The small synthetic English dataset contains only two validation and two test entities. Shared templates limit generalization claims; perfect scores and zero gaps do not rule out overfitting.

Alternate wording was authored after inspecting the original templates, making it an exploratory diagnostic rather than an independent benchmark.

Training scores reuse fitted examples. Nested subsets, reused validation cases, entity folds and training resamples do not constitute independent trials or population confidence intervals.

Entity checks recognize a fixed synthetic catalogue and its aliases. External corpora require an independently audited entity inventory.

Generated labels have no independent human annotation review. Feature diagnostics do not establish error causes, and class-score margins are not calibrated probabilities.

The implementation evaluates lexical classification; it does not establish learned semantic-search, Hebrew or language-model performance.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
