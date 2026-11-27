# sigvision: classical signal processing and vision with a Python reference and a C++17 core

Classical signal-processing and tracking code is often judged only by whether it runs. Whether an estimator is biased, how its variance compares with theory, and whether a filter's stated uncertainty is consistent with its actual error are checked less often. sigvision is a hand-written signal-processing and vision project with a Python reference and a C++17 core. Its estimation extension adds that kind of error analysis on seeded synthetic data.

## What it does

sigvision is a Python/NumPy and C++17 project that implements classical signal-processing and vision algorithms by hand on seeded synthetic data. Its estimation extension (sigvision/estimation.py) adds error analysis. It compares noise-variance estimators with analytic bias, variance and RMSE. It checks the CA-CFAR training-cell noise estimate against its nominal false-alarm rate. It compares a sub-bin tone-frequency estimator with the Cramér–Rao bound (RMSE/CRB 2.0 to 4.1, so not efficient). It runs a Joseph-form constant-velocity Kalman filter scored by position RMSE and NEES. In a quick seeded run the matched filter reached RMSE 3.01 against 7.04 for raw measurements and mean NEES 3.99. A mismatched process model gave mean NEES of about 9867. The module runs standalone (python -m sigvision.estimation --quick) or through an opt-in --estimation flag. Data is synthetic and the estimators are textbook methods. No estimator was ported to C++.

1) Compare noise-variance estimators (known-mean mean-square, unbiased sample variance, ML sample variance) with their analytic bias, variance and RMSE. 2) Check the CA-CFAR training-cell noise estimate and its false-alarm rate against theory. 3) Estimate sub-bin tone frequency by Hann FFT with parabolic log-spectrum interpolation and compare the RMSE with the Cramér–Rao bound across SNR. 4) Run a constant-velocity Kalman filter on simulated 2-D tracks, scored by position RMSE and NEES, including a deliberately mismatched process model. 5) Keep the module reproducible: seeded, runnable standalone and through an opt-in --estimation flag on the existing experiment CLI.

## Solution and technologies

The new module sigvision/estimation.py is NumPy-only and reuses the existing power_spectrum, detect_tones and cfar_alpha. It has four parts. (1) Noise-variance estimators with analytic theory and a vectorised seeded Monte Carlo. (2) An empirical CA-CFAR false-alarm and training-cell noise-estimate check. (3) Tone-frequency estimation by parabolic interpolation of the Hann log power spectrum, refinement of CFAR-detected tones, a Cramér–Rao bound function, and a Monte Carlo over SNR. (4) A hand-written constant-velocity Kalman filter with a Joseph-form covariance update, a 2-D track simulator using a white-noise-acceleration model, and batched NEES. A CLI (python -m sigvision.estimation --quick, with --seed and --json) runs the suite. sigvision.experiment gained an opt-in --estimation flag that appends the suite to its report. The existing C++17 core (FFT, CA-CFAR, conv2d), its native tests and its Python parity checks are unchanged. No estimator was ported to C++.

Python 3 · NumPy · C++17 · make · unittest · statistical estimation · Monte Carlo · Cramér–Rao bound · Kalman filter · FFT · CA-CFAR

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sigvision.estimation --quick
```

## Results

Method: python -m sigvision.estimation --quick, one run, exit code 0. Seeded synthetic data; small trial counts. Baselines are the analytic theory or the Cramér–Rao bound printed next to each figure.

Noise variance (sigma^2 = 1, n = 16, 500 trials; measured vs theory):
- Known-mean mean-square: bias +0.0055 (theory 0), variance 0.1163 (0.1250), RMSE 0.3407 (0.3536).
- Unbiased sample variance: bias +0.0019 (0), variance 0.1224 (0.1333), RMSE 0.3495 (0.3651).
- ML sample variance: bias -0.0607 (-0.0625), variance 0.1075 (0.1172), RMSE 0.3332 (0.3480).
All three variances are about 7–8% below theory, roughly 1.2 standard errors at 500 trials.

CA-CFAR training-cell noise estimate (32 cells, 5000 trials, exponential power): empirical false-alarm rate 0.0118 (standard error 0.0014) against a nominal 0.0100. Relative spread of the noise estimate 0.1808 against theory 0.1768.

Tone frequency (Hann FFT with parabolic interpolation, n = 256, 100 trials per SNR; RMSE vs Cramér–Rao bound std):
- 0 dB: RMSE 0.0704 Hz, bound 0.0345 Hz, ratio 2.04.
- 10 dB: RMSE 0.0256 Hz, bound 0.0109 Hz, ratio 2.35.
- 20 dB: RMSE 0.0140 Hz, bound 0.0035 Hz, ratio 4.06.
The mean biases (-0.0014 to -0.0005 Hz) are small relative to the std. The estimator is not efficient. The cause of the growing ratio at high SNR is an untested hypothesis (an offset-dependent interpolation bias that a mean over random offsets can hide).

Constant-velocity Kalman filter (40 synthetic 2-D tracks of 100 steps each; tracks drawn from the filter's own model):
- Matched filter: position RMSE 3.006 against 7.044 for raw measurements (theory 7.071), ratio 0.427, mean NEES 3.99 (consistent filter: 4).
- Mismatched process noise (accel 0.01): position RMSE 52.876 against raw 7.044, ratio 7.507, mean NEES 9866.95. NEES flags the mismatch.

These are single-run figures with sampling error. They are not general performance claims and are not a comparison with other methods.

## Scope and limitations

All data is synthetic and seeded. There is no sensor, video, radar, lidar or driving data, and the estimators are textbook methods, not new algorithms.

The reported figures come from one small quick run (500 noise trials, 5000 CFAR trials, 100 tone trials per SNR, 40 tracks). They are not statistical performance estimates.

The tone-frequency estimator is not efficient (RMSE/CRB 2.04 to 4.06). The Cramér–Rao bound ignores the negative-frequency image, so it is meaningful only away from DC and Nyquist. The cause of the high-SNR gap is an untested hypothesis.

The Kalman filter was evaluated on tracks simulated from the model it assumes, so NEES consistency shows correct implementation under a matched model, not robustness to real target motion.

Seeded Monte Carlo tests use finite tolerances. A different NumPy version or platform could change the draws and flip a borderline assertion.

No estimator was ported to C++. The C++ core still covers only FFT, CA-CFAR and conv2d.

The combined `sigvision.experiment --quick --require-cpp --estimation` command was not run. The experiment-CLI test passes a nonexistent binary, so it covers only the no-C++ path.

The latency and detection figures in the README Results section come from an earlier, separate `--quick --require-cpp` run. They were not rerun in this validation. Latency is host-CPU only, with no edge, power or Windows measurements.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
