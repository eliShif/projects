# Project extension: classical signal/vision algorithms with a C++ core

This project adds classical signal-processing and computer-vision algorithms,
implemented in Python (NumPy) and C++17, with cross-language parity tests and a
host-side latency comparison. All data is synthetic and seeded.

## Requirement coverage

| Area | Implementation | Tests |
| --- | --- | --- |
| Computer vision techniques (2-D convolution, edges, corner features) | `sigvision/vision.py` | `tests/test_vision.py` |
| Signal processing (windowing, FFT, FIR filtering, adaptive-threshold detection) | `sigvision/spectral.py`, `sigvision/detection.py` | `tests/test_signals_spectral.py`, `tests/test_detection.py` |
| Python programming | `sigvision/` package | `tests/` (unittest) |
| C++ programming (numeric kernels in C++17, input validation, error handling) | `cpp/src/sigcore.cpp`, `cpp/src/main.cpp` | `cpp/tests/test_sigcore.cpp`, `tests/test_cpp_parity.py` |
| Classic algorithm development (radix-2 FFT, CA-CFAR, Harris with NMS) | `sigvision/spectral.py`, `sigvision/detection.py`, `sigvision/vision.py` | `tests/test_detection.py`, `tests/test_vision.py` |
| Prototype-to-core workflow (Python reference, C++ port, parity checks) | `sigvision/cpp_bridge.py`, `sigvision/experiment.py` | `tests/test_cpp_parity.py`, `tests/test_experiment.py` |
| Performance-aware implementation (latency, memory) | `sigvision/perf.py`, `run_timing` in `sigvision/experiment.py`, `bench` in `cpp/src/main.cpp` | `tests/test_perf.py`, `tests/test_cpp_parity.py` |
| Applying known methods to a new problem domain | `sigvision/experiment.py` (accuracy against synthetic ground truth) | `tests/test_experiment.py` |

## Boundaries of this extension

- Synthetic data only: no sensor, hardware, radar or lidar measurements.
- No claim of production deployment or maintenance of a shipped product.
- Latency is measured on the host CPU only; memory figures from Python and C++ are
  different measurements and are labelled as such by the experiment output.
- No measured results are recorded in this repository; run the commands in
  `README.md` to produce them.

## Commands

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make build
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m sigvision.experiment --quick --require-cpp
```
