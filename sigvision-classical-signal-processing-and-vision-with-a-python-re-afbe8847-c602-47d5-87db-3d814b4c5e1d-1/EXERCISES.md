# Exercises

These exercises are for studying and extending the project. Run the test suite
before and after each change.

## Walkthrough

1. Read `sigvision/spectral.py:fft_radix2` and `cpp/src/sigcore.cpp:fft_radix2`
   side by side. Identify the bit-reversal step and the butterfly loop in each.
2. Trace `detect_tones` for a two-tone signal: window, FFT, power spectrum,
   `cfar_alpha`, `cfar_ca_detect`. Print the intermediate arrays in a scratch script.
3. Follow one CFAR call through `cpp_bridge.cfar_payload`, `build/sigcore cfar`
   (`read_cfar` / `run_cfar` in `cpp/src/main.cpp`) and back to NumPy.
4. Run `.venv/bin/python -m sigvision.experiment --quick --require-cpp` and map each
   printed section to the function in `experiment.py` that produced it.

## Questions

1. Why does the Hann window reduce leakage compared with a rectangular window, and
   what does it cost in main-lobe width? Which test demonstrates the benefit?
2. `cfar_alpha` assumes exponentially distributed noise power. Which step of the
   pipeline makes that assumption reasonable for a windowed FFT of white noise?
   What changes for heavy-tailed noise?
3. Why do guard cells matter when two targets are adjacent? Which unit test
   shows masking without them?
4. At the array edges the training window is truncated. How does this change the
   effective false-alarm rate, and how could the multiplier be adapted per cell?
5. Why is CFAR parity exact while FFT parity needs a tolerance? What in the build
   flags (`-ffp-contract=off`) and the code (prefix-sum order) supports this?
6. Why does Harris give a negative response along an edge and a positive one at a
   corner? Relate this to the eigenvalues of the structure tensor.
7. The C++ `bench` command excludes process start-up and text parsing. What would
   the numbers look like if each call went through `CppCore.fft` instead, and why?
8. Python memory uses `tracemalloc` while C++ uses peak RSS. What does each
   include, and why can the two not be compared directly?

## Change exercises

1. **Inverse FFT:** add `ifft_radix2` in Python and C++ (conjugate trick or
   reversed twiddles), a round-trip test in both, and a parity test.
2. **OS-CFAR:** implement ordered-statistic CFAR (k-th smallest training cell) in
   Python, compare its false-alarm behaviour with CA-CFAR on noise containing a
   second strong target in the training window.
3. **Sub-bin frequency estimate:** add parabolic interpolation around each
   detected bin and test the estimate error for off-grid tones.
4. **Separable convolution:** exploit that Sobel kernels are outer products to
   implement `conv2d_separable`; assert equality with `conv2d` and compare latency.
5. **Rotated scenes:** extend `make_scene` with a rotated rectangle and decide
   how the Harris pipeline and the metric tolerance should change.
6. **Border handling:** add reflect padding as an option to `conv2d` in Python
   and C++, with parity tests for both modes.
7. **Non-power-of-two FFT:** add Bluestein or a mixed-radix path and relax the
   length check, keeping the failure tests for invalid lengths that remain invalid.
8. **Benchmark hygiene:** add a warm-up/outlier policy to `time_calls_us` and the
   C++ `bench`, keeping their percentile definitions identical.

## Estimation extension

### Walkthrough

1. Read `monte_carlo_noise_estimator` and `noise_estimator_theory` side by side, then
   run `.venv/bin/python -m sigvision.estimation --quick` and match each printed
   bias/variance/RMSE pair to its analytic value.
2. Trace `estimate_tone_frequency`: Hann `power_spectrum`, peak search, `_interpolate_bin`.
   Print the three log-power samples around the peak for a tone at 40.3 Hz.
3. Follow `kalman_cv_filter` for one step (predict, innovation covariance, gain,
   Joseph update) and compare the matrices with `_cv_matrices`.
4. Run `.venv/bin/python -m sigvision.experiment --quick --require-cpp --estimation`
   and locate where the estimation section is appended in `main`.

### Questions

1. Why is the maximum-likelihood variance estimator biased by `-sigma^2 / n` when the
   mean is unknown, and which test checks the sign and size of that bias?
2. Why does the training-cell average in CA-CFAR have a relative standard deviation of
   `1 / sqrt(2 * num_train)`, and how does that spread affect the realised threshold?
3. Which effects keep the tone-frequency RMSE above the Cramer-Rao bound at high SNR?
   How would a different window or interpolation formula change the bias?
4. What does a mean NEES near the state dimension indicate, and why does the matched
   test fail to say anything about real target motion?
5. In the mismatched run the filter underestimates process noise. What happens to the
   covariance, the gain and the NEES?

### Change exercises

1. **Bias correction:** measure the bias of the parabolic estimator as a function of the
   fractional bin offset and add a correction table; add a test that the corrected
   error is smaller.
2. **Other interpolators:** compare interpolation on magnitude, log-magnitude and
   Jacobsen's complex-ratio estimator for the same Monte Carlo setup.
3. **Mean-estimate error:** add an estimator for the mean of noisy data and report
   its variance against `sigma^2 / n`.
4. **Adaptive noise:** estimate the measurement noise from innovations and feed it
   back into the filter; test it on a track with a wrong initial `meas_std`.
5. **Consistency band:** replace the loose NEES interval in the tests with a
   chi-square confidence interval for the number of runs and steps used.
6. **C++ port:** port the Kalman filter to `cpp/` with a native test and a parity
   test against `kalman_cv_filter`.
