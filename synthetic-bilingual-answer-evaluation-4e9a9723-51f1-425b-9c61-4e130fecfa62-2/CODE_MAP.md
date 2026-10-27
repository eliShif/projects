# Code map

| Path | Role |
| --- | --- |
| `cola_lab/data.py` | Deterministic synthetic sentence generation, CSV write/read, and quality/leakage checks. |
| `cola_lab/experiment.py` | Add-one bigram probabilities, sum/mean scoring, validation threshold selection, held-out metrics, length slices, errors, and JSON report. |
| `cola_lab/__main__.py` | `run` and finite `smoke` command paths. |
| `tests/test_experiment.py` | Success, invalid-data, leakage, scoring, metrics, and repeatability tests. |
| `cola_lab/answer_eval.py` | Generate and validate fictional bilingual facts; retrieve, answer, cite, abstain, calibrate on validation, score by language, and write a reproducible JSON report. |
| `tests/test_answer_eval.py` | Reproducibility, bilingual success, missing facts, invalid evidence, leakage, scoring failures, and test-label isolation. |
| `README.md` | Setup, research question, protocol, output interpretation, limits. |
| `EXERCISES.md` | Walkthrough and change exercises for a future reader. |

Execution flow: `generate_rows` → `write_rows` → `read_rows` → `evaluate` → `report.json`. Inside `evaluate`, the model sees grammatical train rows only; thresholds see validation labels; the test labels are read only for final metrics and error examples. The train-majority baseline is calculated separately from train labels.

Answer path: `make_dataset` → JSON file → `validate_dataset` → validation-only threshold selection → test predictions and `score_case` → `answer_report.json`. `retrieve` uses word overlap after a fixed stopword list; `predict` returns an answer and citation only above the threshold. `summarize` applies the same criteria to English and Hebrew and records each error. No model is trained on this answer dataset.
