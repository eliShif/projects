# Edge Vision: measured results

Fixed held-out test set; validation-only checkpoint selection.

| Variant | Mean accuracy | Mean state KiB | Median batch-1 latency (ms) |
|---|---:|---:|---:|
| teacher_fp32 | 97.87% | 39.7 | 0.0105 |
| small_supervised_fp32 | 97.41% | 9.3 | 0.0098 |
| student_distilled_fp32 | 97.59% | 9.3 | 0.0100 |
| student_dynamic_int8 | 97.69% | 5.4 | 0.0312 |

## Limits
- Small educational image dataset, not a production vision benchmark.
- Model/checkpoint selection used validation only; test split fixed before training.
- Three training seeds share one test split; they are not independent datasets.
- No hardware power, phone or accelerator measurements; latency is local CPU only.
- Serialized state size is storage, not total process memory.
