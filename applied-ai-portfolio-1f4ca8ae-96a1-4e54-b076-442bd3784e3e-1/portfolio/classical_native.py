"""Bridge to the C++ kernels in ``cpp/filters.cpp`` (g++ required to build them).

The program is compiled into a per-process temporary directory; no binary is stored in the repository.
``python -m portfolio.classical_native`` performs a build check.

The program is driven through a text protocol on stdin/stdout; no shell is used.
Reals are written with ``repr`` (exact round-trip) and read back with 17 significant digits.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .classical_vision import (as_image, as_kernel, as_signal, require_fft_length,
                               require_stft_shape)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'cpp' / 'filters.cpp'
BINARY = ROOT / 'cpp' / 'classical_filters.bin'
COMPILE_FLAGS = ('-O2', '-std=c++17', '-Wall', '-Wextra')
RUN_TIMEOUT_SECONDS = 120


class NativeUnavailable(RuntimeError):
    """No usable binary and no compiler to build one."""


class NativeBuildError(RuntimeError):
    """The compiler exists but the source did not compile."""


class NativeError(RuntimeError):
    """The native program rejected its input or failed."""


@dataclass(frozen=True)
class NativeResult:
    value: np.ndarray
    kernel_ns_per_call: float  # compute loop only; excludes process start and text parsing
    repeats: int


_fallback_build: dict = {}


def compile_command(compiler: str, output: Path) -> list:
    return [compiler, *COMPILE_FLAGS, '-o', str(output), str(SOURCE)]


def remove_stale_artifact() -> bool:
    """Delete a leftover in-repository binary from earlier revisions; builds now go to a temp directory."""
    if BINARY.is_file():
        BINARY.unlink()
        return True
    return False


def binary_path(build_if_missing: bool = True) -> Path:
    """Path of the compiled kernel program, built once per process into a temporary directory.

    No binary is ever written into the repository tree.
    """
    if 'path' in _fallback_build:
        return _fallback_build['path']
    if not build_if_missing:
        raise NativeUnavailable('the native program has not been built in this process')
    compiler = shutil.which('g++')
    if compiler is None:
        raise NativeUnavailable('g++ was not found, so the native program cannot be built')
    temp = tempfile.mkdtemp(prefix='classical-filters-')
    atexit.register(shutil.rmtree, temp, ignore_errors=True)
    output = Path(temp) / 'classical_filters.bin'
    result = subprocess.run(compile_command(compiler, output), capture_output=True, text=True,
                            timeout=300, check=False)
    if result.returncode != 0:
        raise NativeBuildError('g++ failed: ' + result.stderr[-1000:])
    _fallback_build['path'] = output
    return output


def native_available() -> bool:
    try:
        binary_path()
    except NativeUnavailable:
        return False
    return True


def run_raw(command: str, text: str, repeat: int = 1) -> list:
    """Run one command with raw stdin text; return stdout tokens or raise ``NativeError``."""
    if int(repeat) != repeat or repeat < 1:
        raise ValueError('repeat must be a positive integer')
    argv = [str(binary_path()), command]
    if repeat != 1:
        argv += ['--repeat', str(int(repeat))]
    result = subprocess.run(argv, input=text, capture_output=True, text=True,
                            timeout=RUN_TIMEOUT_SECONDS, check=False)
    if result.returncode != 0:
        raise NativeError('exit %d: %s' % (result.returncode, result.stderr.strip()))
    return result.stdout.split()


def _numbers(array) -> str:
    return ' '.join(repr(float(v)) for v in np.ravel(array))


def _call(command: str, header, arrays, repeat: int):
    text = ' '.join(str(int(h)) for h in header) + '\n' + '\n'.join(_numbers(a) for a in arrays) + '\n'
    tokens = run_raw(command, text, repeat)
    if len(tokens) < 3 or tokens[-3] != 'timing_ns':
        raise NativeError('native output has no timing record')
    return tokens[:-3], int(tokens[-2]) / int(tokens[-1]), int(tokens[-1])


def _matrix(tokens, shape, command):
    rows, cols = int(tokens[0]), int(tokens[1])
    if (rows, cols) != shape or len(tokens) != 2 + rows * cols:
        raise NativeError('%s returned an unexpected shape' % command)
    return np.array(tokens[2:], dtype=np.float64).reshape(rows, cols)


def correlate2d(image, kernel, repeat: int = 1) -> NativeResult:
    img, ker = as_image(image), as_kernel(kernel)
    tokens, ns, reps = _call('conv2d', (*img.shape, *ker.shape), (img, ker), repeat)
    return NativeResult(_matrix(tokens, img.shape, 'conv2d'), ns, reps)


def sobel_magnitude(image, repeat: int = 1) -> NativeResult:
    img = as_image(image)
    tokens, ns, reps = _call('sobel', img.shape, (img,), repeat)
    return NativeResult(_matrix(tokens, img.shape, 'sobel'), ns, reps)


def fir_filter(signal, taps, repeat: int = 1) -> NativeResult:
    x, h = as_signal(signal), as_signal(taps)
    tokens, ns, reps = _call('fir', (x.size, h.size), (x, h), repeat)
    if int(tokens[0]) != x.size or len(tokens) != 1 + x.size:
        raise NativeError('fir returned an unexpected length')
    return NativeResult(np.array(tokens[1:], dtype=np.float64), ns, reps)


def rfft(signal, repeat: int = 1) -> NativeResult:
    x = as_signal(signal)
    require_fft_length(x.size)
    tokens, ns, reps = _call('fft', (x.size,), (x,), repeat)
    bins = x.size // 2 + 1
    if int(tokens[0]) != bins or len(tokens) != 1 + 2 * bins:
        raise NativeError('fft returned an unexpected length')
    flat = np.array(tokens[1:], dtype=np.float64)
    return NativeResult(flat[0::2] + 1j * flat[1::2], ns, reps)


def stft_magnitude(signal, frame: int, hop: int, repeat: int = 1) -> NativeResult:
    x = as_signal(signal)
    require_stft_shape(x.size, frame, hop)
    tokens, ns, reps = _call('stft', (x.size, frame, hop), (x,), repeat)
    frames, bins = 1 + (x.size - frame) // int(hop), frame // 2 + 1
    return NativeResult(_matrix(tokens, (frames, bins), 'stft'), ns, reps)


class NativeBackend:
    """C++ implementations behind the same interface as ``classical_vision.NUMPY``."""
    name = 'cpp'

    def correlate2d(self, image, kernel):
        return correlate2d(image, kernel).value

    def sobel_magnitude(self, image):
        return sobel_magnitude(image).value

    def fir_filter(self, signal, taps):
        return fir_filter(signal, taps).value

    def rfft(self, signal):
        return rfft(signal).value

    def stft_magnitude(self, signal, frame, hop):
        return stft_magnitude(signal, frame, hop).value


NATIVE = NativeBackend()


def main() -> int:
    """Build check: compile ``cpp/filters.cpp`` into a temporary directory and run one tiny call."""
    removed = remove_stale_artifact()
    path = binary_path()
    check = fir_filter([1.0, 2.0, 3.0], [1.0, 1.0]).value
    if not np.array_equal(check, [1.0, 3.0, 5.0]):
        raise NativeError('built program returned an unexpected FIR result')
    print('native build ok (stale in-repo binary removed: %s; compiled outside the repository: %s)'
          % (removed, ROOT not in path.parents))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
