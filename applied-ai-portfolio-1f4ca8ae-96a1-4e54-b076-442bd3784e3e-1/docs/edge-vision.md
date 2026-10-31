# Edge Vision Lab

Question: what does a smaller distilled model actually trade against accuracy and
CPU time? The real UCI handwritten digits dataset bundled with scikit-learn has
1,797 grayscale 8x8 images. These are low-resolution digits, not a modern vision
deployment task. No claim about video, Windows NPU or customer-scale behavior.

The pipeline makes fixed stratified train/validation/test splits. Training uses
three seeds; validation picks checkpoints. Compare a 128-unit teacher, 24-unit
student trained on hard labels, same-size student trained with KL distillation
plus hard labels, and dynamically quantized student. The control distinguishes
the effect of distillation from reducing architecture size. Training histories,
split indices, seed results, confusion matrices and Wilson intervals are saved.

Measured means: teacher 97.87% accuracy, distilled 97.59%, INT8 student 97.69%.
Teacher state is 39.7 KiB versus 5.4 KiB for INT8. However batch-one median CPU
latency grows from about 0.0105 ms to 0.0312 ms. Serialization overhead matters
for such tiny models; parameter-file bytes are not process RAM or on-device size.
The small control scores 97.41%; this dataset does not establish a robust universal
benefit from distillation. No test-set threshold or model selection is performed.

Recommendation: do not choose INT8 merely because it is smaller. Test the target
hardware, realistic image sizes/batches and power before selecting a deployment
configuration. Current project demonstrates the training/evaluation/optimization
workflow and exposes a negative latency tradeoff.

Run `.venv/bin/python -m portfolio.edge_vision --out results/edge_vision`. CPU inference uses
warmups, fixed one-thread settings and median/p95 timings. Stored results document
the environment. Energy consumption, real edge hardware and MLOps deployment are
not measured here.

[Dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
and [PyTorch dynamic quantization](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html).

## Measurement extension

The historical figures above remain unchanged; the full original experiment was not rerun.
`portfolio.edge_benchmark` reuses these data/model/training functions and adds
isolated worker processes, raw timing records, peak memory and validation-only
resource selection. See [the protocol](edge-measurement.md) and
[current validation](../VALIDATION.md). The earlier builder-sandbox run stopped at
missing dependencies, before worker verification at 2026-10-31T14:15:10.319Z
(exact builder timestamp unavailable). Later worker checks passed real-model
training, distillation, quantization and isolated integration using one seed and
two epochs. A full new three-seed experiment remains unverified; the integration
checks supply no replacement performance table or model-quality estimate for
the full protocol.
