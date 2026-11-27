# Walkthrough and change exercises

## 1. Trace a durable queue operation

Run the finite store path:

```text
python3 -m portfolio.service_load --transport store --jobs 8 --clients 4 --copies 3 --out results/exercise_queue
```

Open `operations.json`, `receipts.json`, `faults.json` and `checks.json`. Follow `JobStore.enqueue`, `claim`, `complete` and `fail` in `portfolio/reliable_service.py`.

Questions:

- Why do 24 submissions produce eight jobs? Which SQL constraints and transaction boundaries make that true under concurrency?
- Why is a matching lease token insufficient once its deadline is reached?
- Why must a stale failure report be fenced as carefully as a stale successful completion?
- What does at-least-once processing imply if a future engine sends an external notification before committing its result?

Change exercise: add a test in `tests/core/test_queue_load.py` that reopens the store after a worker failure and completes the retried job. Verify that the attempt count and ordered trace persist. Use only generated fixture content.

## 2. Recompute the observed measurements

Read `portfolio.measurement.distribution` and recompute p50 and p95 from a run's `operations.json`. Compare them with `summary.json`. Distinguish total submission operations, unique jobs, and separately measured drain throughput.

Change exercise: compare one and four client threads in separate output directories. Keep job count and copies fixed. Report each raw run without selecting only the fastest one. Explain why a short closed-loop queue run cannot establish production capacity or end-to-end HTTP latency.

## 3. Follow the original model experiment

After installing the dependencies, read `digits_data`, `fit`, `train_variants` and `quantize` in `portfolio/edge_vision.py`. Run the real-model integration checks:

```text
.venv/bin/python -m unittest tests.test_edge_models -v
```

Questions:

- Which partitions are permitted to affect gradients, checkpoint selection, variant selection and final reporting?
- How does the supervised 24-unit control distinguish a distillation effect from an architecture-size effect?
- What do the temperature factor and KL term change? Why should teacher weights and gradients remain unchanged?
- How do the tests distinguish packed qint8 Linear kernels from merely rounded floating-point weights?

Change exercise: introduce a configurable distillation weight while retaining the existing default. Add boundary tests for invalid weights and confirm that the teacher remains unchanged. Select any new configuration on validation only; reserve a new test protocol if repeated development has used the old test outcomes.

## 4. Audit measurement isolation and selection

Read `schedule`, `choose_variant`, `launch_worker`, `peak_memory` and the measurement protocol. Run:

```text
python3 -m unittest discover -s tests/core -v
.venv/bin/python -m portfolio.smoke --vision
```

The first command is independent of ML packages. The second requires real models trained locally and fails if their dependencies are absent.

Questions:

- Which biases does a balanced four-position schedule reduce, and which clock/frequency/cache effects remain?
- Why is Linux `ru_maxrss` multiplied by 1,024 but macOS `ru_maxrss` is not?
- Why are state-file size, post-load RSS and whole-process peak memory different measurements?
- Why is `selection.json` saved before test evaluation? What should happen if every variant fails a declared budget?

Change exercise: add a batch-32 latency budget to `Config` and selection. Write one accepting and one rejecting case using clearly labeled test fixtures. Update the report and protocol; do not present those fixture values as measured model performance.

## 5. Investigate failure and reproducibility

Use a fresh output directory for each command. Inspect a failed run's `status.json` and logs, then compare them with a completed run's manifest. Do not overwrite the inherited recovery files.

Change exercise: copy a temporary trained artifact in a test, alter one byte and verify that loading fails before deserialization. Compare this with the existing corruption test. Explain what a content hash can establish and why it does not establish independent reproduction or prior training history.

## 6. Plan a target-device study

Before collecting new measurements, specify the device, OS, runtime, thread count, power source, arrival pattern, quality gates and measurement interval. Run all variants with the same declared conditions. If power is studied, use a real sensor or external meter, document its domain/resolution, and distinguish package energy from whole-device or per-model energy.

Until such instrumentation is available, keep energy and power null. Retain the inherited negative latency and quality results when discussing why a smaller model might still be the wrong deployment choice. Record Windows/macOS native validation separately from Linux unit checks. Write a recommendation with explicit eligibility constraints and unresolved deployment experiments.

## 7. Classical vision and signal processing (Python and C++)

Walkthrough. Verify the compiler path with `.venv/bin/python -m portfolio.classical_native` (the C++ program is compiled into a temporary directory on first use and no binary is stored), then run `.venv/bin/python -m portfolio.classical_benchmark --require-native --repeats 3`. Read `correlate2d`, `sobel_magnitude` and `detect_edges` in `portfolio/classical_vision.py`, then the matching functions in `cpp/filters.cpp`. Follow one call through `portfolio/classical_native.py` (`_call`, `run_raw`) and compare the printed `max_abs_diff` values with the tolerance in `portfolio/classical_benchmark.py`.

Questions:

- Why do both implementations restrict kernels to odd dimensions? What would an even kernel require you to decide about the anchor and about agreement with `scipy.signal.correlate2d`?
- Why is the threshold relative to the maximum magnitude, and which inputs make that choice fragile (for example a noise-only image)?
- Why does `test_blurring_is_needed_under_heavy_noise` compare two pipelines instead of asserting one absolute score?
- The three timing series in a run mean different things. Which one includes process start-up and text parsing, and why can none of them rank the two languages?
- Why does the FFT require power-of-two lengths here, and what would a Bluestein or mixed-radix extension change in the contract and in the tests?
- Why is `repr(float)` plus 17 significant digits enough for the exact round-trip test, and what inputs (for example subnormal values) are outside that guarantee?

Change exercises:

- Add a median filter to both implementations, wire it into the backend interface and add a reference test plus a native equivalence test that includes a one-pixel image and a kernel larger than the image.
- Add a Canny-style non-maximum suppression step in Python only. Define its success criterion on the synthetic shapes before measuring, and report its effect on precision and recall separately.
- Extend the C++ protocol with an `overlap-add` FIR command for long signals. Add rejection tests for malformed headers and an equivalence test against `numpy.convolve`.
- Replace the direct 2-D correlation with a separable implementation for separable kernels. Keep the existing equivalence tests unchanged and add a case where the kernel is not separable and must be rejected or fall back.
- Add a second synthetic ROI scenario (for example two regions with different peak frames) to `portfolio/classical_vision.py` and test that the curves separate; document that it remains synthetic.
