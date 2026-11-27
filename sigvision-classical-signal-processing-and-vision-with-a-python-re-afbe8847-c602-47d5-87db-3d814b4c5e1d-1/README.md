# sigvision: classical signal processing and vision with a Python reference and a C++17 core

Classical signal-processing and vision kernels (FFT, adaptive-threshold detection, filtering, convolution, corner detection) are often prototyped in Python and then re-implemented in C++. The port can silently diverge from the prototype, and the cost of the port is rarely measured. sigvision implements these algorithms by hand on seeded synthetic data. It checks that a C++17 core matches the Python reference, and it reports detection accuracy against known ground truth and host-side latency.

## What it does

sigvision is a Python (NumPy) and C++17 project that implements classical signal-processing and vision algorithms by hand on seeded synthetic data. The signal side has a radix-2 FFT, Hann-windowed spectra, a windowed-sinc FIR low-pass, SNR and CA-CFAR detection. The vision side has 2-D convolution, Sobel edges and Harris corners. A dependency-free C++17 core ports FFT, CA-CFAR and convolution, with native tests and a CLI that has a bench command. In the worker-verified quick run, the C++ output matched the Python reference (FFT error 1.5e-14, identical CFAR indices). The seeded synthetic trials had tone recall 1.0 over 10 trials and corner recall 1.0 over 5 trials. Host-CPU latency was also compared. All 73 unit tests passed. Limits: synthetic data only, small trial counts, host CPU only, and a three-kernel port.

1) Implement a Hann window, radix-2 FFT, power spectrum, windowed-sinc FIR low-pass, SNR measurement and CA-CFAR detection with guard and training cells. 2) Implement zero-padded 2-D convolution, Sobel gradients and edge map, and Harris corners with non-maximum suppression, scored against known rectangle corners. 3) Port FFT, CA-CFAR and 2-D convolution to a dependency-free C++17 core with native unit tests and a text-protocol CLI that includes a bench command. 4) Verify Python/C++ parity (FFT and conv2d within 1e-9, identical CFAR indices). 5) Report seeded synthetic detection accuracy and host-CPU p50/p95 latency, with the limits of each stated.

## Solution and technologies

Python package `sigvision/` (NumPy only):
- `signals.py`: seeded signal generator and Hann window.
- `spectral.py`: iterative radix-2 FFT, power spectrum, windowed-sinc FIR, SNR.
- `detection.py`: CA-CFAR using prefix sums, with a threshold multiplier derived from a target false-alarm probability. Also tone detection (Hann, FFT, CFAR).
- `vision.py`: conv2d, Sobel, edge map, Harris with non-maximum suppression, and synthetic rectangle scenes.
- `metrics.py`: matching of detections to ground truth.
- `perf.py`: nearest-rank p50/p95 and a tracemalloc helper.
- `cpp_bridge.py`: builds the binary with `make` and talks to it over a stdin/stdout text protocol using 17-digit floats.
- `experiment.py`: seeded Monte Carlo trials, parity run, timing run and CLI with a JSON report.

C++17 core `cpp/` (g++ and make, no third-party libraries):
- `sigcore.cpp`: FFT, CA-CFAR, conv2d and percentile. Invalid input throws `std::invalid_argument`.
- `main.cpp`: CLI commands `fft`, `cfar`, `conv2d` and `bench`, with bounded input sizes and number validation. Bench reports p50/p95, peak RSS and an output checksum.
- `test_sigcore.cpp`: native tests.

The Makefile builds with `-O2 -ffp-contract=off` and no fast-math, so CFAR arithmetic order matches Python. Sobel and Harris stay in Python. Of the Sobel path, only conv2d is ported.

Python 3 · NumPy · C++17 · make · unittest · CA-CFAR · FFT · FIR filtering · Sobel/Harris computer vision

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sigvision.experiment --quick --require-cpp
```

## Results

Evidence comes from one worker-run quick experiment (`--quick --require-cpp`, exit code 0). It uses seeded synthetic data, runs for a few seconds, and is a small harness run, not a statistical study. Each number is for the stated method and trial count only.

Detection accuracy against synthetic ground truth:
- Spectral tone detection (Hann window, FFT, CA-CFAR): tone_recall 1.000 over 10 trials, with 0.000 false alarms per trial.
- Harris corners on random synthetic rectangle scenes (match radius 3 px): corner_recall 1.000 over 5 trials, with 0.000 false corners per trial.
- FIR low-pass on one seeded noisy signal against a known clean reference: SNR went from -3.57 dB to 4.31 dB, an improvement of 7.89 dB.

Python/C++ parity (the Python implementation is the reference; tolerance 1e-9):
- FFT max abs error 1.465e-14.
- conv2d max abs error 0.
- Sobel max abs error 0.
- CFAR indices equal: True.

Host latency (single run on one machine, 5 repeats; CPU only, not an embedded target). p50 in microseconds, with Python p95 / C++ p95 in parentheses where reported:
- FFT, 4096 points: Python 1237.0 (2102.7), NumPy 87.4, C++ 503.7 (557.6).
- CFAR, 4096 points: Python 369.3 (415.7), C++ 24.4 (25.3).
- conv2d, 128x128 image with a 5x5 kernel: Python 1179.1 (1312.3), C++ 682.3 (751.6).

How to read the latency numbers:
- The C++ bench times the kernel in-process and excludes process start-up and text parsing, so it is not a figure for a cold call through the CLI bridge.
- The Python FFT is a hand-written radix-2 implementation, and NumPy's FFT is much faster. The differences reflect these particular implementations and are not a general language comparison.
- No Python-vs-C++ comparison is stated for memory, because Python reports a tracemalloc peak per call and C++ reports whole-process peak RSS.

Checks that run in the tests rather than as measurements:
- FFT is tested against NumPy and a naive DFT.
- conv2d is tested against a naive definition.
- Harris is tested to find all 8 corners of two synthetic rectangles, with no false corners.
- Two seeded tone bins [100, 237] are recovered exactly.

## Scope and limitations

All data is synthetic and seeded. There is no sensor, radar, lidar or real-image data, and no claim of generalisation to real measurements.

The experiment figures come from small seeded runs (10 tone trials, 5 corner trials, one FIR signal) and show that the harness works. They are not statistical estimates of detector performance.

Latency comes from a single 5-repeat quick run on one host CPU. There are no embedded, edge, power or Windows measurements. Python tracemalloc peaks and C++ peak RSS are different quantities and are not comparable.

The C++ port covers only FFT, CA-CFAR and 2-D convolution. Harris and Sobel are not ported; Sobel uses the ported conv2d.

Some tests assert exact seeded outcomes (tone bins [100, 237], 8/8 Harris corners with no false corners). They passed in the worker environment, but another NumPy version or platform could change them.

CFAR parity is exact because both implementations use the same arithmetic order, with `-ffp-contract=off` and no fast-math. Other compiler flags may break bit-for-bit equality.

CFAR edge cells use a truncated training window with the full-window multiplier, so border false-alarm rates are not nominal. The C++ CLI's stream parser may reject denormal numbers.

The project is a learning and prototyping artefact, not a production product, and it shows no novel algorithm development.

C++-dependent tests are skipped if make or g++ is missing. The smoke command fails in that case because it passes --require-cpp. Linux is assumed for the peak-RSS figure.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.

## Statistical estimation extension

`sigvision/estimation.py` adds seeded, synthetic-data estimation experiments with explicit error analysis. It reuses the existing Hann window, FFT, CA-CFAR and experiment CLI:

- noise-power estimators (mean-square and sample variance) with Monte Carlo bias, variance and RMSE against the analytic values, plus the CA-CFAR training-cell noise estimate and its false-alarm rate;
- sub-bin tone-frequency estimation (Hann FFT with parabolic interpolation) compared with the Cramer-Rao bound across SNR levels;
- a constant-velocity Kalman filter on synthetic 2-D tracks, scored by position RMSE against raw measurements and by NEES, including a deliberately mismatched process model.

Setup and run (NumPy only; the C++ build is needed only for the existing parity tests):

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make build
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m sigvision.estimation --quick
.venv/bin/python -m sigvision.experiment --quick --require-cpp --estimation
```

Figures from the latest quick estimation run (`python -m sigvision.estimation --quick`) are recorded in [docs/application-94735bca.md](docs/application-94735bca.md), with the scope of that run. The Results section above records an earlier, separate run (`sigvision.experiment --quick --require-cpp`) that was not repeated in the latest validation, so its latency, parity and detection figures are not part of that validation. Likewise, the combined `--require-cpp --estimation` command has not been run as part of the latest validation. Rerunning the commands regenerates all figures. The data is synthetic, the estimators are textbook methods, and no estimation kernel has been ported to C++. Method notes and limits are in the same document.

## Current verified documentation (this version)

- [README.md](README.verified-2e6c994d4a72.md)
- [docs/application-94735bca.md](docs/application-94735bca.verified-5c717a396fa5.md)
- [VALIDATION.md](VALIDATION.verified-754fd3708fc4.md)

## Current verified documentation (this version)

- [VALIDATION.md](VALIDATION.verified-061dd668f5eb.md)

## Current version

The current version of this document is [README.verified-c5cd811cee2c.md](README.verified-c5cd811cee2c.md).

## Current verified documentation (this version)

- [README.md](README.verified-5310fab634cf.md)
- [VALIDATION.md](VALIDATION.verified-107096347a95.md)

## Current verified documentation (this version)

- [VALIDATION.md](VALIDATION.verified-9d124bc56ec7.md)

## Current version

The current version of this document is [README.verified-21a21f0d16c7.md](README.verified-21a21f0d16c7.md).
