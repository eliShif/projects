"""Host-side latency and allocation measurement helpers."""

from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable, Dict, Iterable


def nearest_rank_percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile (same definition as the C++ ``bench`` command)."""
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("no values")
    if not (0.0 < p <= 100.0):
        raise ValueError("p must lie in (0, 100]")
    rank = max(1, math.ceil(p / 100.0 * len(vals)))
    return vals[rank - 1]


def time_calls_us(fn: Callable[[], object], repeats: int, warmup: int = 1) -> Dict[str, float]:
    """Wall-clock latency of ``fn()`` in microseconds: p50/p95 over ``repeats`` calls."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(max(0, warmup)):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    return {
        "repeats": float(repeats),
        "p50_us": nearest_rank_percentile(samples, 50.0),
        "p95_us": nearest_rank_percentile(samples, 95.0),
    }


def peak_alloc_bytes(fn: Callable[[], object]) -> int:
    """Peak Python/NumPy heap allocation (via ``tracemalloc``) during one ``fn()`` call."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)
