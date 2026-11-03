# Bilingual Answer Evaluation — Reference and Metric Validation

The evaluator could accept source words that did not answer the queried entity or attribute, and could divide by zero when a subset lacked answerable or unanswerable cases. These defects weakened reference validation and metric interpretation.

## What it does

Python toolkit for evaluating English/Hebrew answers, citations and abstention with source-checked references, explicit undefined metrics and replayable experiment records. Includes shared-document comparisons, an optional GGUF workflow and a documented synthetic bigram study.

Validate answers, citations and abstention labels against source facts; report undefined conditional metrics with denominators and explanations; support controlled bilingual comparisons using shared, label-free evidence and reproducible scoring.

## Solution and technologies

A Python CLI evaluates 48 synthetic English/Hebrew questions across separate validation and test entities. Template parsers check references against source facts. Word-overlap retrieval selects a document, and validation selects the lexical answerer's abstention threshold. JSON reports contain answer-and-citation scores, language/entity/answerability slices and paired error analyses. The optional GGUF adapter records frozen protocols, shared inputs, generation settings and raw attempts; audit replay checks hashes and recomputes saved results. A separate inherited bigram workflow trains on grammatical sentences, selects thresholds on validation data and compares sum and mean log-probability scoring on held-out groups.

Python · unittest · JSON

## Quick start

~~~text
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -e .
~~~

~~~text
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip wheel --no-index --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m cola_lab answers --output-dir artifacts/baseline
.venv/bin/python -m cola_lab study-smoke
~~~

## Results

Reviewed verification covers incorrect references, false abstention labels, undefined-rate reporting and rejection of altered metrics or policies even after artifact hashes are updated. The documented bilingual baseline retains a paired false abstention on Ember's color paraphrase after retrieval selects its shape source.

The inherited completed experiment in artifacts/REPORT.md evaluated 24 held-out sentences. Sum and mean bigram scoring each achieved 0.667 accuracy and balanced accuracy, with AUC 0.694; train-majority accuracy was 0.500. Each scoring method produced eight false negatives and no false positives, with different error examples. Across 24 training-group refits, mean accuracy was 0.606 for sum and 0.602 for mean, with population variances of 0.0038 and 0.0042; both ranged from 0.500 to 0.750. These are inherited synthetic measurements, not fresh runs.

## Scope and limitations

Reference validation supports a closed synthetic grammar with single-word facts. It does not establish arbitrary-language semantics or independently verified translation quality.

Generated templates, four bilingual test entities and exact-match scoring limit generalization. The bigram refit statistics are descriptive, and observed length effects do not establish causality.

No completed real-model English/Hebrew experiment is supplied. Fixture and adapter checks establish software behavior without establishing model quality.

Shared retrieval can select irrelevant evidence. The all-documents mode gives methods different evidence and cannot isolate model or language effects.

GGUF execution stops at the first invalid output or backend failure. Partial attempts cannot establish overall or language-specific failure rates.

Hashes and replay establish internal consistency, not external execution attestation or identical inference across hardware and backend versions. Source changes require fresh frozen protocols.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
