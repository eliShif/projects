# Cloud Telemetry Lab

Dataset: **synthetic**. Test set held out from threshold selection.

| Detector | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|
| Temporal rules | 75.86% | 73.33% | 0.746 | 7.78% |
| Isolation Forest | 59.57% | 93.33% | 0.727 | 21.11% |

## Limits
- Synthetic scenario generator; these scores do not measure real attack detection.
- Disjoint actors, seeds and time ranges by split; scenario families are shared.
- No live cloud, Kubernetes cluster, nation-state investigation or KQL engine was used.
- KQL queries are reference translations and have not been executed on Azure.
