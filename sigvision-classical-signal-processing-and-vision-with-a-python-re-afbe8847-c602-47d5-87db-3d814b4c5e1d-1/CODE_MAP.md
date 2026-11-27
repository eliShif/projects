# Code Map

## Python package `sigvision/`

| File | Contents |
| --- | --- |
| `signals.py` | `make_signal` (seeded sum of sinusoids plus Gaussian noise), `hann_window` |
| `spectral.py` | `fft_radix2` (iterative DIT FFT), `power_spectrum` (Hann/rect, one-sided), `design_lowpass_fir`, `apply_fir`, `snr_db` |
| `detection.py` | `cfar_alpha` (threshold multiplier from a target false-alarm probability), `cfar_ca_detect` (CA-CFAR + local-maximum rule), `detect_tones` (Hann + FFT + CFAR), `bins_to_hz` |
| `vision.py` | `conv2d`, `SOBEL_X`/`SOBEL_Y`, `sobel_gradients`, `edge_map`, `harris_response`, `harris_corners`, `make_scene`, `rect_corners` |
| `metrics.py` | `match_scalars`, `match_points`: detections vs. ground truth (matched truth, false detections) |
| `perf.py` | `nearest_rank_percentile`, `time_calls_us` (p50/p95), `peak_alloc_bytes` (`tracemalloc`) |
| `cpp_bridge.py` | `ensure_built` (runs `make build`), `CppCore` client for `build/sigcore`, payload builders, `CppCoreError` |
| `experiment.py` | Trial runners (spectral, FIR, vision), `run_parity`, `run_timing`, CLI `main` |

## C++ core `cpp/`

| File | Contents |
| --- | --- |
| `include/sigcore.hpp` | Public API: `fft_radix2`, `cfar_ca_detect`, `conv2d`, `nearest_rank_percentile` |
| `src/sigcore.cpp` | Implementations; throw `std::invalid_argument` on invalid input |
| `src/main.cpp` | CLI (`fft`, `cfar`, `conv2d`, `bench`), text parsing, peak-RSS reader |
| `tests/test_sigcore.cpp` | Dependency-free unit tests (`CHECK`, `CHECK_THROWS`); exit status 0 on success |
| `../Makefile` | `build`, `test-cpp`, `clean`; flags `-std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off` |

## Tests `tests/`

| File | Covers |
| --- | --- |
| `test_signals_spectral.py` | Signal generation, FFT vs. NumPy, spectrum scaling, window leakage, FIR design/response, SNR improvement, error cases |
| `test_detection.py` | CFAR alpha, spikes at edges, guard cells, adaptive threshold, false-alarm bound on noise, tone recovery, invalid parameters |
| `test_vision.py` | Convolution vs. a naive definition, Sobel directions, edge map, Harris corners (clean and noisy), scene/metric helpers, error cases |
| `test_perf.py` | Percentile definition, timing helper, allocation helper |
| `test_cpp_parity.py` | Native tests, FFT/CFAR/conv2d parity, protocol failures (truncated, garbage, unknown command, bad bench args), bench output |
| `test_experiment.py` | Trial runners, CLI exit codes (missing binary), end-to-end quick run with JSON report |

## Data flow for parity checks

1. Python builds an input (seeded NumPy arrays).
2. `cpp_bridge` serialises it with 17 significant digits and calls `build/sigcore`.
3. C++ output is parsed back and compared with the Python reference
   (`atol=1e-9` for floating-point outputs, exact equality for CFAR indices).

## Conventions worth knowing

- `conv2d` is true convolution (kernel flipped), odd kernel sizes only, zero padding.
  Sobel kernels are stored in convolution form so gradients are positive towards +x / +y.
- CFAR window bounds are half-open and clipped at the array ends; a detection also
  requires `x[i] > x[i-1]` and `x[i] >= x[i+1]`.
- Both sides use the same percentile definition (nearest rank) for p50/p95.

## Estimation extension

| File | Contents |
| --- | --- |
| `sigvision/estimation.py` | Noise-variance estimators and theory (`estimate_noise_variance`, `noise_estimator_theory`, `monte_carlo_noise_estimator`), CFAR training-cell check (`monte_carlo_cfar_false_alarm`), tone frequency (`estimate_tone_frequency`, `estimate_tone_frequencies`, `tone_frequency_crb_std_hz`, `monte_carlo_tone_frequency`), constant-velocity Kalman filter (`simulate_cv_track`, `kalman_cv_filter`, `nees`, `monte_carlo_tracking`), suite and CLI (`run_estimation_suite`, `print_estimation_report`, `main`) |
| `sigvision/experiment.py` | Opt-in `--estimation` flag that appends the estimation suite to the report |
| `tests/test_estimation.py` | Known values, Monte Carlo agreement with theory, seed dependence, invalid input, Kalman consistency (NEES), mismatched-model detection, CLI and JSON output |
| `docs/application-94735bca.md` | Requirement mapping, method notes and limits |

Reused from the existing code: `power_spectrum` (Hann + radix-2 FFT), `detect_tones`,
`cfar_alpha`. Conventions: the state vector is `[x, y, vx, vy]`; process noise comes
from a discrete white-noise-acceleration model (`Q = sigma_a^2 G G^T`); the tone
Monte Carlo uses `eta = A^2 / (2 sigma^2)` as the per-sample SNR.
