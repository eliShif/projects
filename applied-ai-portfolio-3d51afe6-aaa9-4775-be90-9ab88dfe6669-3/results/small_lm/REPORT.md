# Small Language Model Efficiency

Real CPU inference, isolated process per variant.

| Variant | State MiB | Resident MiB after load | Authored-text perplexity | Prefill p50 ms | Generated tokens/s |
|---|---:|---:|---:|---:|---:|
| fp32 | 513.2 | 960.7 | 120.26 | 145.49 | 14.88 |
| dynamic_int8 | 236.6 | 1540.6 | 258.58 | 39.73 | 47.23 |

## Limits
- Six authored prose passages are a smoke benchmark, not standard language-model perplexity.
- Same weights and tokenizer in both variants; no task-specific fine-tuning.
- Variants use separate processes. RSS includes Python, libraries and model; it is not weights-only memory.
- Dynamic INT8 targets Linear modules; embeddings remain floating point.
- No power meter or physical edge device was available. No energy-efficiency claim.
- This small model is a reproducible baseline, not a frontier-model capability claim.
