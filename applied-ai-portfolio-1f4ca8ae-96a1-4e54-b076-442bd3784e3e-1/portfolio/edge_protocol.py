"""Dependency-free protocol validation and validation-only resource selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

VARIANTS = ('teacher_fp32', 'small_supervised_fp32',
            'student_distilled_fp32', 'student_dynamic_int8')


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (7, 19, 43)
    epochs: int = 100
    rounds: int = 4
    repetitions: int = 100
    threads: int = 1
    min_accuracy: float = .95
    max_accuracy_drop: float = .02
    max_state_kib: float = 16
    max_p95_ms: float = 1
    max_peak_mib: float = 1024
    device_label: str = 'development-host'
    power_condition: str = 'unknown'

    def validate(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(
                not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError('Require distinct seeds in [0, 2**32)')
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in
               (self.epochs, self.rounds, self.repetitions, self.threads)):
            raise ValueError('Epochs, rounds, repetitions and threads must be positive integers')
        if self.rounds % len(VARIANTS):
            raise ValueError('Rounds must be a multiple of four to balance variant positions')
        for value in (self.min_accuracy, self.max_accuracy_drop):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Quality constraints must be finite fractions in [0, 1]')
        for value in (self.max_state_kib, self.max_p95_ms, self.max_peak_mib):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Resource budgets must be finite and positive')
        if not self.device_label or len(self.device_label) > 120 or any(c in self.device_label for c in '\n\r'):
            raise ValueError('Use a short single-line device label')
        if self.power_condition not in ('unknown', 'ac', 'battery'):
            raise ValueError('Power condition must be unknown, ac or battery')
        return self

    def as_dict(self):
        return asdict(self)


def schedule(config):
    config.validate()
    # A Latin square balances each variant at every position within each seed.
    # Fresh processes eliminate persistent model/cache state between trials.
    return [{'seed': seed, 'round': trial, 'position': position, 'variant': variant}
            for trial in range(config.rounds)
            for seed in config.seeds
            for position, variant in enumerate(VARIANTS[trial % 4:] + VARIANTS[:trial % 4])]


def choose_variant(rows, config):
    """Freeze a family decision before any test evaluation; never uses test fields."""
    config.validate()
    keys = [(r['seed'], r['round'], r['variant']) for r in rows]
    expected = {(r['seed'], r['round'], r['variant']) for r in schedule(config)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate benchmark trials')
    for r in rows:
        acc = r['validation_accuracy']
        if not math.isfinite(acc) or not 0 <= acc <= 1:
            raise ValueError('Invalid validation accuracy')
        if r['serialized_state_bytes'] <= 0 or not math.isfinite(r['serialized_state_bytes']):
            raise ValueError('Invalid serialized size')
        for batch in ('latency_batch1', 'latency_batch32'):
            values = r[batch]
            if any(not math.isfinite(values[k]) or values[k] < 0 for k in ('p50_ms', 'p95_ms')):
                raise ValueError('Invalid latency')
    accuracy = {}
    for seed in config.seeds:
        for variant in VARIANTS:
            subset = [r for r in rows if r['seed'] == seed and r['variant'] == variant]
            if len({r['validation_accuracy'] for r in subset}) != 1 or len({r['artifact_sha256'] for r in subset}) != 1:
                raise ValueError('Artifact or quality changed between repetitions')
            accuracy[seed, variant] = subset[0]['validation_accuracy']
    candidates = []
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        accs = [accuracy[seed, variant] for seed in config.seeds]
        drops = [accuracy[seed, 'teacher_fp32'] - accuracy[seed, variant] for seed in config.seeds]
        size = max(r['serialized_state_bytes'] for r in group)
        worst_p95 = max(r['latency_batch1']['p95_ms'] for r in group)
        memory = [r['peak_memory']['bytes'] if r['peak_memory']['status'] == 'measured' else None for r in group]
        peak = max(memory) if all(v is not None and math.isfinite(v) and v > 0 for v in memory) else None
        reasons = []
        if min(accs) < config.min_accuracy: reasons.append('validation accuracy below floor')
        if max(drops) > config.max_accuracy_drop + 1e-12: reasons.append('validation loss of accuracy versus teacher exceeds budget')
        if size > config.max_state_kib * 1024: reasons.append('serialized footprint exceeds budget')
        if worst_p95 > config.max_p95_ms: reasons.append('batch-1 p95 exceeds budget in at least one trial')
        if peak is None: reasons.append('peak memory unavailable; budget cannot be checked')
        elif peak > config.max_peak_mib * 2**20: reasons.append('process peak memory exceeds budget')
        candidates.append({'variant': variant, 'eligible': not reasons, 'reasons': reasons,
                           'validation_accuracy_mean': statistics.mean(accs),
                           'validation_accuracy_min': min(accs), 'max_accuracy_drop': max(drops),
                           'state_bytes_max': size, 'peak_bytes_max': peak,
                           'batch1_p50_ms_median': statistics.median(r['latency_batch1']['p50_ms'] for r in group),
                           'batch1_p95_ms_max': worst_p95,
                           'batch32_p50_ms_median': statistics.median(r['latency_batch32']['p50_ms'] for r in group)})
    feasible = [r for r in candidates if r['eligible']]
    selected = min(feasible, key=lambda r: (r['state_bytes_max'], r['batch1_p50_ms_median'], r['variant'])) if feasible else None
    return {'selected_variant': selected['variant'] if selected else None, 'candidates': candidates,
            'rule': 'Smallest serialized state meeting every validation and resource constraint; median batch-1 latency breaks ties.',
            'decision_data': 'validation split and repeated validation-input measurements only',
            'recommendation': ('Evaluate the selected variant on the identified target hardware before deployment.' if selected
                               else 'No variant satisfies the declared constraints. Retain the baseline and investigate the recorded failures; do not relax gates after viewing test results.')}
