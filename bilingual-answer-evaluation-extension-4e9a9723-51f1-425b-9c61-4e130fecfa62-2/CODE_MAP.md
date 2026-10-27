# Code map

| Path | Role |
| --- | --- |
| `cola_lab/data.py` | Deterministic synthetic sentence generation, CSV write/read, and quality/leakage checks. |
| `cola_lab/experiment.py` | Add-one bigram probabilities, sum/mean scoring, validation threshold selection, held-out metrics, length slices, errors, and JSON report. |
| `cola_lab/__main__.py` | `run` and finite `smoke` command paths. |
| `tests/test_experiment.py` | Success, invalid-data, leakage, scoring, metrics, and repeatability tests. |
| `cola_lab/answer_eval.py` | Generate and validate fictional bilingual facts, source-answer consistency, and complete language pairs; retrieve, answer, cite, abstain, calibrate on validation, score by language/entity/matched pair, compare methods by entity resampling, validate optional frozen local-model outputs, and write a reproducible JSON report. |
| `tests/test_answer_eval.py` | Reproducibility, bilingual success, language-specific errors, rejected incomplete pairs and false source labels, missing facts, invalid evidence, leakage, scoring failures, paired method comparisons, test-label isolation, and accepted/rejected offline model outputs. |
| `README.md` | Setup, research question, protocol, output interpretation, limits. |
| `EXERCISES.md` | Walkthrough and change exercises for a future reader. |

Execution flow: `generate_rows` → `write_rows` → `read_rows` → `evaluate` → `report.json`. Inside `evaluate`, the model sees grammatical train rows only; thresholds see validation labels; the test labels are read only for final metrics and error examples. The train-majority baseline is calculated separately from train labels.

Answer path: `make_dataset` → JSON file → `validate_dataset` → validation-only threshold selection → test predictions and `score_case` → `method_report` and `compare_methods` → `answer_report.json`. If `--predictions` is given, `load_model_predictions` reads a frozen file, checks every test ID and citation, and adds a local-model arm to the same scoring path. `retrieve` uses word overlap after a fixed stopword list; `predict` returns an answer and citation only above the threshold. `summarize` records errors; `paired_language_analysis` matches outcomes by entity and question intent; `describe_dataset` records split balance. `compare_methods` enumerates whole-entity resamples for a descriptive accuracy-difference interval. No model is trained on this answer dataset.
