# Real low-rank adaptation

15,360 trainable parameters; 48 updates; validation-selected checkpoint.

| Metric | Frozen base | Adapter |
|---|---:|---:|
| Held-out answer-token NLL | 6.153 | 0.521 |
| Held-out exact match (12 cases) | 33.3% | 33.3% |

- Single seed and 12 authored test examples; not a general capability estimate.
- Fine-tuning learns an answer format on a narrow synthetic distribution.
- This arm uses one gold document and greedy decoding, unlike the three-document stochastic retrieval experiment; do not compare their scores causally.
