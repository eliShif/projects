# Isolated digits measurement protocol

This extension calls the original model/data/training functions. It adds measurement and decision controls without changing the 8x8 digit task, the four architectures/optimization variants, the fixed stratified partitions (1,149 train, 288 validation, 360 test), or the original default seeds.

## Execution sequence

1. Validate the configuration and save source hashes, command arguments, hardware/runtime metadata and the complete measurement schedule.
2. Train teacher, same-size supervised control and distilled student in a training process for each seed. Each model restores the checkpoint with minimum validation cross entropy. Save histories, state files, dataset hash and artifact hashes. Quantization converts the selected student to actual qint8 Linear kernels.
3. Measure one variant at a time in a fresh interpreter. Set intra-op/BLAS thread controls and one PyTorch inter-op thread. Verify artifact and dataset hashes, record actual runtime settings, score validation examples, warm up ten calls per batch size, and retain every timed forward-pass duration.
4. Rotate the four variants across four positions for each seed. Rounds must be a multiple of four. There are no simultaneous model workers. Default timing uses 100 samples for batch 1 and batch 32 in each trial.
5. Choose a variant from validation quality and measured resource constraints. The gate checks every seed's quality and every trial's batch-1 p95 and peak memory. Minimize maximum serialized state size, then median batch-1 latency. Save the decision before evaluating test data. No qualifying variant is a valid negative result.
6. Score all four frozen variants on test for descriptive comparison; do not update selection from these scores. Save predictions, labels, indices, confusion matrices and Wilson intervals, then a manifest of actual run artifacts.

## Scope of each metric

| Metric | Scope | Limit |
|---|---|---|
| Validation/test accuracy | Correct argmax labels, original fixed partitions | Small digits benchmark, shared test set across seeds |
| Raw milliseconds, p50/p95 | CPU forward pass after warmup, preprocessing/startup excluded | Sub-millisecond work is noise-sensitive; no frequency/thermal isolation |
| Load seconds | Construct network and restore the verified state; includes quantized skeleton construction | Excludes Python startup, imports and hash verification; not application cold start |
| Serialized bytes | Actual saved state file | Not RAM, installer size, executable size or deployment footprint |
| Linux peak memory | `getrusage(RUSAGE_SELF).ru_maxrss`, KiB to bytes | Whole worker lifetime including libraries, data and loading |
| macOS peak memory | `getrusage(RUSAGE_SELF).ru_maxrss`, native bytes | Native macOS path has not run in this workspace |
| Windows peak memory | `GetProcessMemoryInfo.PeakWorkingSetSize` | Whole-process working set; native Windows path has not run here |
| Power/energy | Null with explicit unavailable status | No instrumented sensor; CPU time/TDP are not substitutes |

Peak memory is read after validation scoring and timing, before report serialization. It is an OS high-water mark, not a sampled RSS maximum and not tensor-only allocation. Non-supported platforms report unavailable memory, which prevents passing the peak-memory budget. Different OS memory semantics should not be pooled.

The metadata records CPU model when available, architecture, OS/release, logical CPU count, affinity, Python/package versions, user-declared power condition and worker thread/backend settings. Device labels do not verify edge hardware. Background load and thermal conditions remain uncontrolled. A fixed seed does not guarantee identical numerics across hardware/runtime versions.

## Failure and interpretation

Output directories must be empty. Nonzero child exits, timeouts, absent models or hash mismatches stop the run. Protocols, training records and logs already written remain; a failed run does not retain a success summary/report. Model files are loaded with `weights_only=True` and their stored hashes are checked first. Hashes are audit identifiers, not a substitute for trusting a model source.

The earlier builder-sandbox dependency failure occurred before worker verification at 2026-10-31T14:15:10.319Z; its exact timestamp is not supplied. The later worker passed real-model checks, including distillation, qint8 layers, raw measurements and isolated model/artifact integration using one seed and two epochs. See [execution status](../VALIDATION.md). A completed full new three-seed benchmark and its model recommendation remain unverified. The original vision result remains inherited evidence of a negative storage/latency tradeoff, without establishing an outcome for the full new measurement protocol. Target-edge performance and physical power remain unmeasured; native Windows/macOS execution, production scalability and customer impact remain unverified.
