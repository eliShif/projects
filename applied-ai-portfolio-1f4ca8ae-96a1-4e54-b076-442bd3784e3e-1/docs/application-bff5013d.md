# Classical vision and signal-processing extension

This extension adds a small classical-vision and signal-processing module to the existing portfolio. It is purely additive: no existing source, test, result or documentation line was changed, and the earlier test suite is still part of the documented test command.

## What was added

| Path | Purpose |
|---|---|
| `portfolio/classical_vision.py` | NumPy reference: zero-padded 2-D cross-correlation, Gaussian kernel, Sobel gradient magnitude, relative-threshold edge map, causal FIR filtering, windowed-sinc low-pass design, real FFT, Hann-windowed STFT magnitude, dominant-frequency estimate, edge precision/recall/F1 with pixel tolerance, seeded synthetic image/signal/dynamic-stack generators and a region-of-interest time-activity comparison |
| `cpp/filters.cpp` | C++17 implementation of the same kernels (direct 2-D correlation, Sobel, FIR, iterative radix-2 FFT, STFT) behind a stdin/stdout text protocol with strict input validation and an in-program compute timer |
| `portfolio/classical_native.py` | Python bridge: locates or builds the binary with `g++`, formats inputs with exact `repr` round-trip, parses outputs, maps program rejections to `NativeError`, exposes a backend object with the same interface as the NumPy reference |
| `portfolio/classical_benchmark.py` | Finite check: runs both implementations on seeded synthetic data, compares them under a stated tolerance, scores the edge pipeline, times each path, and optionally records the run with the existing `RunRecord` lifecycle |
| `tests/test_classical_vision.py`, `tests/test_classical_native.py`, `tests/test_classical_benchmark.py` | Success and failure tests described below |

Reused from the existing code: `portfolio.measurement.distribution` (linear quantiles with raw samples retained), `RunRecord` (empty-output requirement, failure cleanup, manifest hashes), `environment` (host metadata with power explicitly unavailable), `portfolio.common.write_json`.

## Requirement coverage

| IDs | Behaviour | Implementation | Tests |
|---|---|---|---|
| R01 | Classical vision pipeline (blur, Sobel, threshold) scored against known synthetic ground truth; blur shown to matter under heavy noise; flat-image and invalid-input handling | `portfolio/classical_vision.py` | `tests/test_classical_vision.py` (`EdgeTests`, `CorrelationTests`) |
| R02 | FIR low-pass design and filtering, FFT/STFT spectra, dominant-frequency detection, ROI versus background time-activity curves on synthetic data | `portfolio/classical_vision.py`, `cpp/filters.cpp` | `tests/test_classical_vision.py` (`SignalTests`, `SyntheticDataTests`), `tests/test_classical_native.py` (`EquivalenceTests`) |
| R03 | Python implementation, validation and tests for the above, plus a CLI check | `portfolio/classical_vision.py`, `portfolio/classical_native.py`, `portfolio/classical_benchmark.py` | `tests/test_classical_vision.py`, `tests/test_classical_benchmark.py` |
| R04 | C++ core for image and signal kernels verified against the Python reference (tolerance 1e-9), including boundary, size-one, oversize-kernel and malformed-input cases | `cpp/filters.cpp`, `portfolio/classical_native.py` | `tests/test_classical_native.py` (`EquivalenceTests`, `RejectionTests`, `TimingTests`) |
| R05 | Extension of the existing project without altering existing files; earlier tests remain in the documented test command | all paths above; documentation appended only | existing `tests/` unchanged, run by the same discovery command |
| R06 | Seeded experiment with recorded protocol, source hashes, manifest, raw timing samples, failure cleanup and a documented timing method | `portfolio/classical_benchmark.py`, `portfolio/measurement.py` | `tests/test_classical_benchmark.py` |

## Run

See the README section "Classical vision extension" for the exact commands. The benchmark prints its own measured values; no result files are shipped with this extension.

## Design notes

- Both implementations use cross-correlation with zero padding and odd kernels only, so their border behaviour is identical by construction and the NumPy version is additionally checked against `scipy.signal.correlate2d`.
- The primitive filters (`correlate2d`, `sobel_magnitude`) zero-pad. The edge pipeline `detect_edges` instead edge-replicates the image by the blur radius plus the Sobel radius before filtering and crops that margin afterwards, so the image border is not reported as an edge. `threshold_edges` takes a `floor`: a peak gradient at or below it means "no edges", and `detect_edges` sets it to `1e-9` times the image scale so floating-point residue on flat regions is not amplified by the relative threshold. Both backends receive the same padded array, so the maps stay identical.
- The C++ program rejects even kernels, non-positive or oversized dimensions, truncated or trailing input, non-finite values, non-power-of-two FFT/frame lengths, frames longer than the signal, and invalid repeat counts, with exit code 2 and a message on stderr.
- Timing separates three quantities: NumPy call time, C++ wall time per subprocess call (includes process start and text formatting/parsing), and C++ compute-loop time. They are not interchangeable and the first two should not be read as an algorithm comparison.

## Defects found in validation and their regressions

A full discovery run reported 8 errors and 1 failure in the new tests. Two defects caused them:

- The native backend object was referenced as `cv.NATIVE` in `portfolio/classical_benchmark.py` and `tests/test_classical_native.py`, but it is defined as `classical_native.NATIVE` (the NumPy module cannot import the native bridge without a cycle). Call sites now use `native.NATIVE`. Regression: the benchmark and edge-pipeline equivalence tests exercise the native backend directly.
- `detect_edges` reported edges on a flat image because of border zero padding and rounding residue. Fixed as described in the design notes. Regressions: `test_flat_images_of_any_level_have_no_edges_with_or_without_blur`, `test_threshold_floor_suppresses_tiny_peaks_and_is_validated`, `test_image_border_is_not_reported_as_an_edge` (`tests/test_classical_vision.py`) and `test_flat_image_gives_no_edges_on_the_native_backend_too` (`tests/test_classical_native.py`).

## Limitations

- Data are small seeded synthetic images and signals (64x64 images, 256-sample signals, 24-frame 32x32 stacks). There is no video, object detection, camera geometry, driving or clinical data, and no result here speaks to such workloads.
- The time-activity function is a synthetic stand-in for ROI-versus-background analysis; it is not a PET or fMRI pipeline.
- The C++ code is a direct implementation for equivalence checking, not an optimized library. It is single-threaded, has no SIMD, and uses an O(n^2 k) direct correlation; timings carry no claim about production performance, other hardware or other compilers.
- Edge-quality thresholds (F1 of at least 0.75 at a 2-pixel tolerance) apply to the synthetic shapes and noise levels in the tests only.
- If `g++` and a prebuilt binary are both absent, the native tests are skipped and the benchmark falls back to NumPy-only checks unless `--require-native` is given; the report then states `native_executed: false`.
- The text protocol favours inspectability over speed. Subnormal-range numbers may be rejected by the parser.
- The existing limitations of the portfolio (single-seed model integrations, unverified target devices and power, SQLite prototype scope) are unchanged by this extension.

## Update: no compiled binary in the repository

A publication check rejected the compiled file `cpp/classical_filters.bin`, which an earlier build step had written into the source tree. The current behaviour supersedes the earlier statements about a prebuilt binary:

- `portfolio/classical_native.py` no longer reads or writes a binary under `cpp/`. The first native call in a process compiles `cpp/filters.cpp` with `g++` into a temporary directory that is removed at exit.
- `.venv/bin/python -m portfolio.classical_native` is the build check. It deletes a leftover `cpp/classical_filters.bin` from earlier revisions, compiles into a temporary directory, runs one small FIR call and prints a status line. It exits nonzero if compilation or the call fails.
- Regression tests: `tests/test_classical_artifacts.py` (stale-file removal, unavailable-compiler and unbuilt-program errors, binary location outside the repository, and no `*.bin` left in `cpp/` after the build check).
- Consequence: each Python process that uses the native backend pays one compilation (a few seconds); no timing in this extension includes it.
- The statement above that native tests are skipped when both `g++` and a prebuilt binary are absent now reads: skipped when `g++` is absent.
- The earlier validation records (`VALIDATION.verified-*.md`) describe the previous build command and are retained unchanged as history.
