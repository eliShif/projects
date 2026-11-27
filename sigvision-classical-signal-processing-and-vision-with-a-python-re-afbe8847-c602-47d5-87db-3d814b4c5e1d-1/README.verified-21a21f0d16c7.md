# sigvision: classical signal processing and vision with a Python reference and a C++17 core

Classical signal-processing and tracking code is often judged only by whether it runs. Whether an estimator is biased, how its variance compares with theory, and whether a filter's reported uncertainty matches its real error are checked less often. The sigvision project already had hand-written signal and vision algorithms and a C++17 core checked against a Python reference. This component adds an error-analysis layer to it.

## What it does

sigvision is a Python/NumPy and C++17 project for classical signal processing and vision on seeded synthetic data. This card covers its estimation extension, sigvision/estimation.py. The extension compares noise-variance estimators with analytic bias, variance and RMSE. It checks the CA-CFAR training-cell noise estimate against its nominal false-alarm rate. It compares a Hann-FFT parabolic-interpolation tone-frequency estimator with the Cramér–Rao bound, where the RMSE/CRB ratio of 2.0 to 4.1 shows the estimator is not efficient. It also runs a Joseph-form constant-velocity Kalman filter scored by position RMSE and NEES. In one quick synthetic run the matched filter had mean NEES 3.99 (ideal 4) and a mismatched model had about 9,867. All 105 tests passed, make build exited 0 and the documented quick run exited 0. The evidence is synthetic, single-run and Python-only. No estimator was ported to C++.

1) Compare noise-variance estimators (known-mean mean-square, unbiased sample variance, ML sample variance) with analytic bias, variance and RMSE. 2) Check the CA-CFAR training-cell noise estimate and its false-alarm rate against theory. 3) Estimate sub-bin tone frequency with a Hann FFT and parabolic interpolation, and compare its RMSE with the Cramér–Rao bound across SNR. 4) Run a constant-velocity Kalman filter on simulated 2-D tracks, scored by position RMSE and NEES, including a deliberately mismatched process model. 5) Keep everything seeded and reproducible, runnable standalone or through an opt-in --estimation flag on the existing experiment CLI.

## Solution and technologies

sigvision/estimation.py is a NumPy-only module with four parts. (1) Noise-variance estimators, closed-form theory and a vectorised seeded Monte Carlo. (2) An empirical CA-CFAR false-alarm and training-cell noise-estimate check that uses the existing cfar_alpha. (3) Tone-frequency estimation by parabolic interpolation of the Hann log power spectrum, built on power_spectrum and detect_tones. It also has a refinement step for CFAR-detected tones, a Cramér–Rao bound function and a Monte Carlo over SNR. (4) A hand-written constant-velocity Kalman filter with a Joseph-form covariance update, a white-noise-acceleration track simulator and batched NEES. The CLI is `python -m sigvision.estimation --quick` (with --seed and --json). sigvision/experiment.py gained an opt-in --estimation flag, and "estimation" was added to __all__ in sigvision/__init__.py. The existing Python reference and the C++17 core (FFT, CA-CFAR, conv2d, with native tests and Python parity tests) are unchanged. No estimation kernel was ported to C++. Install and run commands: `python3 -m venv .venv`; `.venv/bin/pip install -r requirements.txt`; `make build`; `.venv/bin/python -m unittest discover -s tests -v`; `.venv/bin/python -m sigvision.estimation --quick`; `.venv/bin/python -m sigvision.experiment --quick --require-cpp --estimation`. The last command was not run in the latest validation.

Python 3 · NumPy · C++17 · make · unittest · statistical estimation · Monte Carlo · Cramér–Rao bound · Kalman filter · FFT · CA-CFAR

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sigvision.estimation --quick
```

## Results

All figures come from one run of `python -m sigvision.estimation --quick` (exit code 0). The data is seeded synthetic. Each figure is shown against its analytic theory or Cramér–Rao reference. Trial counts are small, so these are single-run figures with sampling error, not statistical performance estimates, and they are not a comparison with other methods.

Noise variance (sigma^2 = 1, n = 16, 500 trials; measured vs theory):
- Known-mean mean-square: bias +0.0055 (theory 0), variance 0.1163 (0.1250), RMSE 0.3407 (0.3536).
- Unbiased sample variance: bias +0.0019 (0), variance 0.1224 (0.1333), RMSE 0.3495 (0.3651).
- ML sample variance: bias -0.0607 (-0.0625), variance 0.1075 (0.1172), RMSE 0.3332 (0.3480).
- The variances are about 7–8% below theory, roughly 1.2 standard errors at 500 trials.

CA-CFAR training cells (32 cells, 5,000 trials, exponential power): empirical false-alarm rate 0.0118 (standard error 0.0014) against a nominal 0.0100. Relative spread of the noise estimate 0.1808 against a theoretical 0.1768.

Tone frequency (Hann FFT with parabolic interpolation, n = 256, 100 trials per SNR; RMSE against the Cramér–Rao bound std):
- 0 dB: RMSE 0.0704 Hz, bound 0.0345 Hz, ratio 2.04.
- 10 dB: RMSE 0.0256 Hz, bound 0.0109 Hz, ratio 2.35.
- 20 dB: RMSE 0.0140 Hz, bound 0.0035 Hz, ratio 4.06.
- The estimator is not efficient. The cause of the growing high-SNR ratio is an untested hypothesis (an offset-dependent interpolation bias).

Constant-velocity Kalman filter (40 synthetic 2-D tracks of 100 steps, drawn from the filter's own model):
- Matched: position RMSE 3.006 against 7.044 for raw measurements (theory 7.071), mean NEES 3.99 (consistent filter: 4).
- Mismatched process noise (accel 0.01): position RMSE 52.876, mean NEES 9866.95. NEES flags the mismatch, but this does not show robustness to real target motion.

Earlier latency, parity and detection figures from the separate `sigvision.experiment --quick --require-cpp` run stay in the README Results section and the historical records. They were not repeated in the latest validation and are not part of this card.

## Scope and limitations

The estimation figures come from one small quick run (500 noise trials, 5,000 CFAR trials, 100 tone trials per SNR, 40 tracks). They are not statistical performance estimates.

All data is synthetic and seeded. There is no sensor, video, radar, lidar or driving data, and the estimators are textbook methods, not new algorithms.

The tone-frequency estimator is not efficient (RMSE/CRB 2.04 to 4.06). The Cramér–Rao bound ignores the negative-frequency image and is meaningful only away from DC and Nyquist. The explanation for the high-SNR gap, an offset-dependent interpolation bias, is an untested hypothesis.

The Kalman filter is evaluated on tracks simulated from the model it assumes. NEES consistency therefore shows correct implementation under a matched model, not robustness to real target motion.

Seeded Monte Carlo tests use finite tolerances, and some are loose (RMSE/CRB between 0.8 and 10, mean NEES between 2.5 and 5.5). A different NumPy version or platform could change the draws and flip a borderline assertion.

No estimator was ported to C++. The C++ core covers only FFT, CA-CFAR and conv2d. The new estimation tests do not exercise the C++ core, so the Python/C++ claim rests on the existing parity tests.

The combined `sigvision.experiment --quick --require-cpp --estimation` command was not run. The earlier `--quick --require-cpp` latency, parity and detection figures were not repeated in the latest validation. Latency is host-CPU only, with no edge, power or Windows measurements.

The `make build` step reused an existing binary, so it did not show a fresh compile.

README.md ends with pipeline-generated 'Current verified documentation' and 'Current version' pointer blocks that name different side files. DOCUMENTATION.md and README.verified-c5cd811cee2c.md are the latest verified versions.

This is a personal project and not employment experience. It does not support R01, R04 or R13, and it supports R09, R11 and R12 only weakly or partially.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
