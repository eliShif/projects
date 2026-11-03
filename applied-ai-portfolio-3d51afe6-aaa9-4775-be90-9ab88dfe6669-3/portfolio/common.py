"""Shared statistics and provenance. No network or model side effects on import."""
from __future__ import annotations
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError('Require 0 <= successes <= total and total > 0')
    p = successes / total
    d = 1 + z*z/total
    c = (p + z*z/(2*total))/d
    w = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0., c-w), min(1., c+w)]


def clustered_difference(a, b, seed=73, draws=2000):
    """Paired bootstrap over question clusters, never treats repeats as independent."""
    a,b = np.asarray(a,float),np.asarray(b,float)
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError('Expected paired question x repeat matrices, >=2 questions')
    delta=(b-a).mean(axis=1)
    rng=np.random.default_rng(seed)
    boot=delta[rng.integers(0,len(delta),(draws,len(delta)))].mean(axis=1)
    return {'mean_difference':float(delta.mean()), 'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'unit':'question', 'questions':len(delta), 'repeats_per_question':a.shape[1]}


def sha256_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def provenance():
    return {'utc':datetime.now(timezone.utc).isoformat(), 'python':sys.version.split()[0],
            'platform':platform.platform(), 'machine':platform.machine(), 'processor':platform.processor()}
