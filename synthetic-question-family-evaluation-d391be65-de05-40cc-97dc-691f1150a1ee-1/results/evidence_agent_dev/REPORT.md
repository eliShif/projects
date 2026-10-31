# Evidence-preserving agent evaluation

Model: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. Real CPU inference.
Split: **dev**, 8 questions, 1 repetitions per system.

| System | Correct | Correct with expected evidence | Abstention correct | Invalid/failed |
|---|---:|---:|---:|---:|
| rag | 50.0% | 50.0% | 33.3% | 0 |
| typed_agent | 100.0% | 100.0% | 100.0% | 0 |

Predeclared acceptance gate: **PASS**.

## Per-family grounded correctness

| Family | RAG | Typed agent |
|---|---:|---:|
| conflict | 0.0% | 100.0% |
| conversion | 0.0% | 100.0% |
| difference | 100.0% | 100.0% |
| lookup | 100.0% | 100.0% |
| missing_field | 100.0% | 100.0% |
| ratio | 0.0% | 100.0% |
| sum | 100.0% | 100.0% |
| unknown_service | 0.0% | 100.0% |

## Paired uncertainty
null

## Limits

- Authored synthetic data; six held-out entities, not a public benchmark.
- Typed agent has structured corpus access; RAG receives top-six text records. This compares systems, not isolated model intelligence.
- Agent prevents document-note injection by keeping notes outside the planning boundary; no broad injection-robustness claim.
- Task families are shared across development/test; entity holdout does not prove task generalization.
- The evaluation does not establish frontier-model quality or production load capacity.
