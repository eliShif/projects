"""Host measurements and auditable local run records; no model dependencies."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

from .common import write_json


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def distribution(samples):
    """Linear quantiles, matching NumPy's default definition, retaining all samples."""
    values = list(samples)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Expected nonempty, finite, nonnegative measurements')
    ordered = sorted(values)
    point = .95 * (len(ordered) - 1)
    lo = math.floor(point)
    hi = math.ceil(point)
    return {'count': len(values), 'p50_ms': statistics.median(values),
            'p95_ms': ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo),
            'raw_ms': values}


def peak_memory():
    """OS high-water mark for this process, including interpreter/imports/load/inference."""
    scope = 'whole worker lifetime; includes Python, libraries, loading and inference'
    system = platform.system()
    try:
        if system in ('Linux', 'Darwin'):
            import resource
            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            scale = 1024 if system == 'Linux' else 1
            return {'status': 'measured', 'bytes': int(raw * scale),
                    'method': 'getrusage(RUSAGE_SELF).ru_maxrss',
                    'native_unit': 'KiB' if system == 'Linux' else 'bytes', 'scope': scope}
        if system == 'Windows':
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters(); counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError('GetProcessMemoryInfo failed')
            return {'status': 'measured', 'bytes': counters.PeakWorkingSetSize,
                    'method': 'GetProcessMemoryInfo.PeakWorkingSetSize',
                    'native_unit': 'bytes', 'scope': scope}
    except (OSError, ValueError, ImportError) as exc:
        return {'status': 'unavailable', 'bytes': None, 'reason': type(exc).__name__, 'scope': scope}
    return {'status': 'unavailable', 'bytes': None, 'reason': 'unsupported OS', 'scope': scope}


def environment(device_label='development-host', power_condition='unknown'):
    """Allowlisted hardware/runtime metadata; no environment dump, hostname or user data."""
    cpu = 'unknown'
    if platform.system() == 'Linux':
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip(); break
        except OSError:
            pass
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'utc': datetime.now(timezone.utc).isoformat(), 'device_label': device_label,
            'target_edge_device_verified': False,
            'os': platform.system(), 'os_release': platform.release(), 'machine': platform.machine(),
            'cpu': cpu, 'logical_cpus': os.cpu_count(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'python': sys.version.split()[0], 'packages': versions,
            'power_condition': power_condition,
            'power': {'status': 'unavailable', 'watts': None, 'joules': None,
                      'reason': 'No instrumented power sensor in this run; no estimates from TDP or CPU time.'},
            'thermal_and_background_load': 'not controlled; repeat on an idle identified target'}


class RunRecord:
    """Refuse overwrite; retain partial records and remove success claims on failure."""
    def __init__(self, out, protocol, sources):
        self.out = Path(out)
        self.protocol = protocol
        self.sources = sources

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if any(self.out.iterdir()):
            raise FileExistsError('Output must be empty; retain earlier runs and use a new directory')
        write_json(self.out / 'protocol.json', {**self.protocol,
                   'source_sha256': {str(p): file_hash(p) for p in self.sources}})
        write_json(self.out / 'status.json', {'status': 'running'})
        self.start = time.perf_counter()
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            for filename in ('summary.json', 'REPORT.md', 'manifest.json'):
                (self.out / filename).unlink(missing_ok=True)
            write_json(self.out / 'status.json', {'status': 'failed', 'error_type': kind.__name__,
                       'elapsed_seconds': time.perf_counter() - self.start,
                       'detail': 'Inspect retained worker logs or validation output; no successful summary.'})
            return False
        files = sorted(p for p in self.out.rglob('*') if p.is_file() and p.name != 'status.json')
        write_json(self.out / 'manifest.json', {str(p.relative_to(self.out)): file_hash(p) for p in files})
        write_json(self.out / 'status.json', {'status': 'complete',
                   'elapsed_seconds': time.perf_counter() - self.start})
        return False
