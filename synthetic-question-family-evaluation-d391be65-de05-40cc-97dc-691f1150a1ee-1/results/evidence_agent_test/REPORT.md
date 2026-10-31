# Evidence-preserving agent evaluation

Model: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. Real CPU inference.
Split: **test**, 48 questions, 2 repetitions per system.

| System | Correct | Correct with expected evidence | Abstention correct | Invalid/failed |
|---|---:|---:|---:|---:|
| rag | 43.8% | 43.8% | 50.0% | 0 |
| typed_agent | 91.7% | 91.7% | 100.0% | 0 |

Predeclared acceptance gate: **PASS**.

## Per-family grounded correctness

| Family | RAG | Typed agent |
|---|---:|---:|
| conflict | 0.0% | 100.0% |
| conversion | 0.0% | 83.3% |
| difference | 50.0% | 50.0% |
| lookup | 100.0% | 100.0% |
| missing_field | 100.0% | 100.0% |
| ratio | 0.0% | 100.0% |
| sum | 50.0% | 100.0% |
| unknown_service | 50.0% | 100.0% |

## Paired uncertainty
{"mean_difference": 0.4791666666666667, "ci95": [0.375, 0.5833333333333334], "unit": "service entity", "questions": 6, "repeats_per_question": 1}

## Limits

- Authored synthetic data; six held-out entities, not a public benchmark.
- Typed agent has structured corpus access; RAG receives top-six text records. This compares systems, not isolated model intelligence.
- Agent prevents document-note injection by keeping notes outside the planning boundary; no broad injection-robustness claim.
- Task families are shared across development/test; entity holdout does not prove task generalization.
- The evaluation does not establish frontier-model quality or production load capacity.
