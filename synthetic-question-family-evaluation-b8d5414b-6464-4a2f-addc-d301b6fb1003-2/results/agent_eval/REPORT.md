# Local Agent Evaluation

Real local model generations on an authored, synthetic QA set.

| Policy | Accuracy | Evidence recall | All repeats correct | Median seconds |
|---|---:|---:|---:|---:|
| direct | 0.00% | 0.00% | 0.00% | 2.09 |
| retrieval | 19.44% | 100.00% | 0.00% | 2.30 |
| tool_agent | 0.00% | 0.00% | 0.00% | 4.22 |

## Limits
- Small authored QA suite, not evidence of general agent capability.
- Retrieval is TF-IDF, not a neural embedding model.
- Correctness is strict numeric exact match; retrieval evidence recall is not a full groundedness judgment.
- Bootstrap resamples questions; repeated generations are clustered.
- Invalid tool actions and step exhaustion count as failures; no hidden repair or mock-model fallback.
