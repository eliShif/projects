# Code map

| Path | Role |
| --- | --- |
| `cola_lab/data.py` | Deterministic synthetic sentence generation, CSV write/read, and quality/leakage checks. |
| `cola_lab/experiment.py` | Add-one bigram probabilities, sum/mean scoring, validation threshold selection, held-out metrics, length slices, errors, and JSON report. |
| `cola_lab/__main__.py` | `run` and finite `smoke` command paths. |
| `tests/test_experiment.py` | Success, invalid-data, leakage, scoring, metrics, and repeatability tests. |
| `README.md` | Setup, research question, protocol, output interpretation, limits. |
| `EXERCISES.md` | Walkthrough and change exercises for a future reader. |

Execution flow: `generate_rows` → `write_rows` → `read_rows` → `evaluate` → `report.json`. Inside `evaluate`, the model sees grammatical train rows only; thresholds see validation labels; the test labels are read only for final metrics and error examples. The train-majority baseline is calculated separately from train labels.
