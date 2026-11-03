# Bilingual Answer Evaluation

Aggregate answer scores can conceal whether equivalent Hebrew and English questions received different evidence or produced different answers. This component makes paired outcomes, citations and required-source availability inspectable.

## What it does

A Python evaluator for paired Hebrew and English answers, combining a TF-IDF baseline, shared numeric and citation checks, retrieval diagnostics and reproducible run artifacts.

Apply consistent numeric, abstention and citation criteria across languages; compare answers with a lexical baseline; expose retrieval and outcome disagreements; and retain reproducible records for failure analysis and review.

## Solution and technologies

The Python CLI extends the existing evidence scorer and NumPy/scikit-learn TF-IDF retriever. Three synthetic records support three English/Hebrew question pairs. Top-one retrieval feeds a record-copying baseline or optional local GGUF inference; imported structured predictions can also be evaluated. Validation checks prediction coverage, values, statuses, citations and pair consistency. Per-language scoring feeds case diagnostics and paired_audit.json. Saved artifacts include inputs, predictions, comparisons, errors and protocol hashes for data, source code and prompts, alongside dependency versions. Model mode additionally records raw responses, model identity/settings and a pending human-review queue.

Python

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-bilingual.txt
.venv/bin/python -m unittest discover -s tests -p test_bilingual_eval.py -v
.venv/bin/python -m compileall -q portfolio/bilingual_eval.py
.venv/bin/python -m portfolio.bilingual_eval --out /tmp/bilingual-eval
```

## Results

Historical validation completed the lexical-baseline smoke run over six authored cases: two numeric lookups and one missing-field question in each language. The baseline retrieves one TF-IDF record, copies its value when the requested field matches, and otherwise abstains. Shared scoring checks numeric correctness, abstention and required citations. This fixed smoke suite has no training or held-out split, and the evaluated system is the baseline itself. The documented outcome is a runnable paired-audit workflow with retrieval traces and reproducibility metadata. Separate completed experiments and their negative findings remain in the [upstream experiment history](https://github.com/eliShif/applied-ai-portfolio/tree/f06d214375d45572c92dae6e7229d5f1d652f05b).

## Scope and limitations

The six authored cases form three paired smoke scenarios without an independent held-out split. Hebrew questions retain English service names and field identifiers, limiting conclusions about general bilingual ability.

No completed GGUF inference run or human-reviewed outcomes are supplied for this bilingual component. The inherited model experiments evaluate different systems and datasets.

The supplied validation contains exit codes and output hashes but no bilingual predictions, summaries or paired-audit outputs. Numerical correctness, disagreement and failure totals therefore cannot be reported.

TF-IDF matches tokens, and top-one retrieval can omit required records. Source-availability diagnostics describe observable conditions, not internal model causes or causal language effects; one retrieved record cannot establish corpus-wide absence.

Imported predictions lack independently verified model provenance and retrieval context. Their recorded trace describes this evaluator's retrieval only.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
