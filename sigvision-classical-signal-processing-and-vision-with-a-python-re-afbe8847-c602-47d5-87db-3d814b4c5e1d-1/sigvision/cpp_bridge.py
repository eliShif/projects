"""Python wrapper around the ``build/sigcore`` C++ executable (text protocol).

The bridge exists so the C++ core can be compared with the Python reference on
identical inputs. Floats are exchanged with 17 significant digits, which
round-trips IEEE-754 doubles exactly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "build" / "sigcore"
TEST_BINARY = REPO_ROOT / "build" / "sigcore_tests"


class CppCoreError(RuntimeError):
    """The C++ core failed to build, was not found, or rejected its input."""


class CppToolchainMissing(CppCoreError):
    """``make`` or a C++ compiler is not available on this host."""


def ensure_built(timeout: float = 300.0) -> Path:
    """Run ``make build`` (incremental) and return the path of the CLI binary."""
    if shutil.which("make") is None or shutil.which("g++") is None:
        raise CppToolchainMissing("make and g++ are required to build the C++ core")
    proc = subprocess.run(
        ["make", "-s", "build"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise CppCoreError(f"make build failed:\n{proc.stdout}\n{proc.stderr}")
    if not DEFAULT_BINARY.exists():
        raise CppCoreError("make build succeeded but build/sigcore is missing")
    return DEFAULT_BINARY


def _fmt(values) -> str:
    out = []
    for v in values:
        f = float(v)
        if not np.isfinite(f):
            raise ValueError("non-finite value cannot be sent to the C++ core")
        out.append(format(f, ".17g"))
    return "\n".join(out)


def fft_payload(x) -> str:
    a = np.asarray(x, dtype=complex)
    if a.ndim != 1:
        raise ValueError("fft input must be 1-D")
    lines = [str(a.size)]
    lines.extend(f"{format(float(v.real), '.17g')} {format(float(v.imag), '.17g')}" for v in a)
    return "\n".join(lines) + "\n"


def cfar_payload(power, num_train: int, num_guard: int, alpha: float) -> str:
    x = np.asarray(power, dtype=float)
    if x.ndim != 1:
        raise ValueError("power must be 1-D")
    head = f"{x.size} {int(num_train)} {int(num_guard)} {format(float(alpha), '.17g')}"
    return head + "\n" + _fmt(x) + "\n"


def conv2d_payload(image, kernel) -> str:
    img = np.asarray(image, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    if img.ndim != 2 or ker.ndim != 2:
        raise ValueError("image and kernel must be 2-D")
    return (
        f"{img.shape[0]} {img.shape[1]} {ker.shape[0]} {ker.shape[1]}\n"
        + _fmt(img.reshape(-1))
        + "\n"
        + _fmt(ker.reshape(-1))
        + "\n"
    )


class CppCore:
    """Thin client for the ``sigcore`` executable."""

    def __init__(self, binary: Optional[Path] = None, timeout: float = 120.0):
        self.binary = Path(binary) if binary is not None else DEFAULT_BINARY
        self.timeout = timeout
        if not self.binary.exists():
            raise CppCoreError(f"C++ binary not found: {self.binary}")

    def run_raw(self, args: Sequence[str], payload: str) -> str:
        """Run the CLI with ``args`` and stdin ``payload``; raise on non-zero exit."""
        try:
            proc = subprocess.run(
                [str(self.binary), *args],
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise CppCoreError(f"sigcore timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise CppCoreError(proc.stderr.strip() or f"sigcore exited with {proc.returncode}")
        return proc.stdout

    def fft(self, x) -> np.ndarray:
        tokens = self.run_raw(["fft"], fft_payload(x)).split()
        n = int(tokens[0])
        vals = np.array(tokens[1:], dtype=float)
        if vals.size != 2 * n:
            raise CppCoreError("fft output has unexpected length")
        pairs = vals.reshape(n, 2)
        return pairs[:, 0] + 1j * pairs[:, 1]

    def cfar_detect(self, power, num_train: int, num_guard: int, alpha: float) -> np.ndarray:
        tokens = self.run_raw(["cfar"], cfar_payload(power, num_train, num_guard, alpha)).split()
        count = int(tokens[0])
        idx = np.array([int(t) for t in tokens[1:]], dtype=np.int64)
        if idx.size != count:
            raise CppCoreError("cfar output has unexpected length")
        return idx

    def conv2d(self, image, kernel) -> np.ndarray:
        tokens = self.run_raw(["conv2d"], conv2d_payload(image, kernel)).split()
        h, w = int(tokens[0]), int(tokens[1])
        vals = np.array(tokens[2:], dtype=float)
        if vals.size != h * w:
            raise CppCoreError("conv2d output has unexpected length")
        return vals.reshape(h, w)

    def bench(self, kind: str, payload: str, repeats: int) -> Dict[str, float]:
        """In-process C++ timing (excludes process start-up and text parsing).

        Returns ``p50_us``, ``p95_us``, ``peak_rss_kb`` (whole-process high-water
        mark, -1 if unavailable) and ``checksum``.
        """
        line = self.run_raw(["bench", kind, str(int(repeats))], payload).strip()
        fields = dict(item.split("=", 1) for item in line.split())
        try:
            return {
                "repeats": float(fields["repeats"]),
                "p50_us": float(fields["p50_us"]),
                "p95_us": float(fields["p95_us"]),
                "peak_rss_kb": float(fields["peak_rss_kb"]),
                "checksum": float(fields["checksum"]),
            }
        except KeyError as exc:
            raise CppCoreError(f"unexpected bench output: {line!r}") from exc
