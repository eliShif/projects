# applied-ai-portfolio: classical vision and signal-processing extension

Classical image and signal-processing algorithms (2-D filtering, edge detection, spectral analysis) are easy to get subtly wrong at image borders, on flat inputs and across implementations. A second implementation in another language is only credible if it is checked against a reference on boundary cases and malformed input. This extension of the applied-ai portfolio addresses that on small synthetic data. It also had to avoid committing a compiled binary to the repository, which an earlier publication check rejected.

## What it does

Classical vision and signal-processing extension of the applied-ai portfolio. It has a NumPy reference (zero-padded 2-D correlation, Gaussian blur, Sobel edges, thresholded edge pipeline, FIR design, FFT, STFT, dominant frequency, synthetic ROI time-activity curves) and C++17 kernels behind a validated text protocol. The C++ is compiled with g++ into a temporary directory, so no binary is stored in the repository. On the latest worker run (seed 11, one repeat, synthetic 64x64 image and 256-sample signal) C++ matched NumPy within 3.4e-14 against a 1e-9 tolerance, edge maps were identical, and the full 126-test discovery run passed. Scope is small synthetic data with unoptimized C++ and single-sample timings that are not a ranking.

1) Provide a NumPy reference for zero-padded 2-D correlation, Gaussian blur, Sobel gradients and a thresholded edge pipeline, scored against synthetic ground truth. 2) Provide windowed-sinc FIR design and filtering, FFT, STFT, dominant-frequency detection and a synthetic ROI-versus-background time-activity comparison. 3) Provide C++17 kernels behind a validated stdin/stdout text protocol, called from Python without a shell. 4) Check C++ against NumPy at 1e-9 tolerance and record raw timing samples with a documented timing method. 5) Add the work as new modules and appended documentation, and keep compiled artifacts out of the repository.

## Solution and technologies

portfolio/classical_vision.py is the NumPy reference. It covers zero-padded odd-kernel correlation, Gaussian kernel, Sobel magnitude, a relative-threshold edge map with a flat-image floor, and edge precision/recall/F1 with pixel tolerance. The edge pipeline edge-replicates the image before filtering and crops the margin afterwards, so the border is not reported as an edge. It also covers causal FIR filtering, windowed-sinc low-pass design, power-of-two real FFT, Hann STFT magnitude, dominant frequency, seeded synthetic generators and ROI time-activity curves.

cpp/filters.cpp holds C++17 kernels for direct correlation, Sobel, FIR, iterative radix-2 FFT and STFT. They sit behind a text protocol with a bounds-checked reader. The protocol rejects non-finite, truncated and trailing input and bounds repeat counts. Rejections exit with code 2, and the program reports its own compute-loop time.

portfolio/classical_native.py compiles cpp/filters.cpp with g++ into a per-process temporary directory on first use. It runs the program with subprocess without a shell and with a timeout, validates output shapes and exposes a NATIVE backend with the same interface as the NumPy backend. Both backends receive the same padded array, so edge maps are identical. No binary is read or written under cpp/. `python -m portfolio.classical_native` is a build check: it removes a leftover cpp/classical_filters.bin, compiles outside the repository, runs one small FIR call and exits nonzero on failure.

portfolio/classical_benchmark.py is a seeded finite CLI. It compares C++ and NumPy at 1e-9 tolerance for five operations, scores edge quality, records raw timings and exits nonzero on any failed check. It falls back to NumPy-only checks when g++ is absent unless --require-native is given. With --out it records a protocol, source hashes and a manifest in a new directory through the existing RunRecord.

Data flow: synthetic_image / synthetic_signal → NumPy reference and (optionally) C++ → tolerance comparison and timing distributions → console JSON or a recorded run directory.

Python · NumPy · SciPy · C++17 · g++ · unittest

## Quick start

```sh
.venv/bin/pip install --no-cache-dir numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0
.venv/bin/pip install --no-cache-dir 'torch>=2.9,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m portfolio.classical_benchmark --require-native --repeats 1
```

## Results

All figures below come from one worker run on the host: seed 11, one repeat, synthetic data, `--require-native`; exit 0, all_passed true, native_executed true.
- Data: 64x64 synthetic image (rectangle and disc, noise sigma 0.02) and a 256-sample, 256 Hz signal (12 Hz amplitude 1.0 plus 40 Hz amplitude 0.5, noise sigma 0.05).
- Edge quality (NumPy pipeline vs synthetic ground truth, 2 px tolerance): precision 1.0, recall 1.0, F1 1.0 (563 predicted pixels, 148 truth pixels). This is one image and not a general accuracy estimate.
- Dominant frequency: 12.0 Hz from both NumPy and C++.
- C++ vs NumPy maximum absolute difference (tolerance 1e-9): 5x5 Gaussian correlation 0.0; Sobel magnitude 0.0; 63-tap FIR 6.7e-16; 256-point rFFT 3.4e-14; STFT (frame 64, hop 32) 7.1e-15. Edge maps were identical.
- Timings, one sample each so p50 equals p95, in ms as NumPy / C++ wall / C++ kernel: correlation 0.432 / 18.131 / 0.196; Sobel 0.659 / 23.153 / 0.242; FIR 0.161 / 6.207 / 0.019; rFFT 0.083 / 3.461 / 0.044; STFT 0.274 / 5.325 / 0.071. C++ wall time includes process start and text formatting/parsing. The three series measure different quantities under uncontrolled background load, so they are not an algorithm or language ranking and not a claim about optimized libraries.
- Build check: `native build ok (stale in-repo binary removed: False; compiled outside the repository: True)`. No stale binary existed in the worker run, so the removal path is covered by the regression test rather than by this build output.

## Scope and limitations

Data are small seeded synthetic images and signals (64x64 images, 256-sample signals, 24-frame 32x32 stacks). There is no video, object detection, camera geometry, driving or clinical data.

The edge F1 of 1.0 is a single synthetic case (seed 11, one image). The F1 >= 0.75 test criterion applies only to the synthetic shapes and noise levels in the tests.

The ROI time-activity comparison is a synthetic stand-in, not a PET or fMRI pipeline.

The C++ code is a direct, single-threaded, unoptimized implementation for equivalence checking, with no SIMD and no production-C++ claim. Timings come from one repeat under uncontrolled background load, and NumPy time, C++ wall time and C++ kernel time are different quantities. They do not rank the languages.

FFT and STFT require power-of-two lengths, kernels must be odd, and the text protocol may reject subnormal-range numbers.

Every Python process that uses the native backend compiles the C++ program once (a few seconds, not included in the timings). If g++ is absent, the native tests are skipped and the benchmark falls back to NumPy-only checks unless --require-native is set.

The removal of a leftover cpp/classical_filters.bin is covered by a regression test. The supplied worker build run found no stale file, and the supplied source inventory lists no binary.

Target-device, power and native Windows/macOS measurements were not made. Existing portfolio limitations (single-seed model integrations, SQLite prototype scope, no production deployment evidence) are unchanged and are documented in their own reports.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
