"""Offline LLM quality operations: prompt-version comparison, windowed monitoring, drift and trace export.

Three parts share one record format:

1. Paired prompt-version comparison. Every prompt version answers the same synthetic cases with the same
   seeds. Outputs are parsed against a strict JSON schema and scored as grounded-correct. Versions are
   compared with the existing paired, case-clustered bootstrap (``portfolio.common.clustered_difference``).
2. Windowed monitoring over a stream of request records: p50/p95 (``portfolio.measurement.distribution``)
   and p99 (computed here), citation/grounding/format/abstention rates as hallucination *proxies*, and
   PSI/KS drift against a reference window, each with explicit alert thresholds.
3. Span-style JSONL trace export and strict loading, so a recorded run can be replayed through the monitor.

Scope: this is offline replay on synthetic data. With the bundled rule-based stub generator the output
verifies the comparison and monitoring code, not the quality of any prompt or model. Nothing here
observes a live system.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile

import numpy as np

from .common import clustered_difference, wilson, write_json
from .measurement import RunRecord, distribution

SCOPE = ('Offline replay on synthetic cases and synthetic request streams; not live monitoring and not a '
         'production deployment. With the rule-based stub generator the results verify the comparison and '
         'monitoring code, not prompt or model quality. Hallucination figures are proxies (cited ids absent '
         'from the supplied passages, answers not found in the cited text, abstention rate), not a general '
         'hallucination measurement.')

REQUEST_SPAN = 'llm.request'
GENERATE_SPAN = 'llm.generate'

# ---------------------------------------------------------------------------------------------------
# Synthetic cases
# ---------------------------------------------------------------------------------------------------

ATTRIBUTES = {
    'port': ('{svc} listens on port {value}.', 'Which port does {svc} listen on?'),
    'owner': ('{svc} is owned by the {value} team.', 'Which team owns {svc}?'),
    'region': ('{svc} is deployed in region {value}.', 'In which region is {svc} deployed?'),
    'retries': ('{svc} retries failed calls {value} times.', 'How many times does {svc} retry failed calls?'),
}
SERVICES = ('ledger-api', 'billing-worker', 'auth-gateway', 'render-queue', 'image-cache',
            'search-index', 'notify-hub', 'report-batch', 'audit-log', 'metrics-sink')
TEAMS = ('Atlas', 'Borealis', 'Cobalt', 'Dune', 'Ember', 'Flint')
REGIONS = ('eu-north-1', 'us-east-2', 'ap-south-1', 'sa-east-1')


def _compile(template):
    out = ''
    for part in re.split(r'(\{svc\}|\{value\})', template):
        if part == '{svc}':
            out += r'(?P<svc>[\w-]+)'
        elif part == '{value}':
            out += r'(?P<value>[\w-]+)'
        else:
            out += re.escape(part)
    return re.compile('^' + out + '$')


_SENTENCE_RE = {a: _compile(s) for a, (s, _q) in ATTRIBUTES.items()}
_QUESTION_RE = {a: _compile(q) for a, (_s, q) in ATTRIBUTES.items()}


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    passages: tuple          # ((passage_id, text), ...)
    answerable: bool
    gold_answer: object      # str or None
    gold_id: object          # str or None
    attribute: str


def _value(rng, attribute):
    if attribute == 'port':
        return str(rng.randint(2000, 9999))
    if attribute == 'owner':
        return rng.choice(TEAMS)
    if attribute == 'region':
        return rng.choice(REGIONS)
    return str(rng.randint(2, 9))


def _distinct_value(rng, attribute, avoid):
    for _ in range(100):
        value = _value(rng, attribute)
        if value != avoid:
            return value
    raise RuntimeError('could not draw a distinct value')


def _sentence(attribute, svc, value):
    return ATTRIBUTES[attribute][0].format(svc=svc, value=value)


def make_cases(seed, n=24, unanswerable=6):
    """Deterministic synthetic cases with exactly ``unanswerable`` questions the passages cannot answer."""
    if int(n) != n or n < 2:
        raise ValueError('n must be an integer >= 2')
    if int(unanswerable) != unanswerable or not 0 <= unanswerable <= n:
        raise ValueError('unanswerable must be an integer between 0 and n')
    rng = random.Random(seed)
    no_answer = set(rng.sample(range(n), int(unanswerable)))
    attributes = list(ATTRIBUTES)
    cases = []
    for i in range(int(n)):
        svc = SERVICES[i % len(SERVICES)]
        other = rng.choice([s for s in SERVICES if s != svc])
        asked = rng.choice(attributes)
        spare = [a for a in attributes if a != asked]
        side = rng.choice(spare)
        side2 = rng.choice([a for a in spare if a != side])
        answerable = i not in no_answer
        gold = _value(rng, asked)
        sentences = []   # (text, is_gold)
        if answerable:
            sentences.append((_sentence(asked, svc, gold), True))
            sentences.append((_sentence(side, svc, _value(rng, side)), False))
        else:
            sentences.append((_sentence(side, svc, _value(rng, side)), False))
            sentences.append((_sentence(side2, svc, _value(rng, side2)), False))
        sentences.append((_sentence(asked, other, _distinct_value(rng, asked, gold)), False))
        sentences.append((_sentence(side, other, _value(rng, side)), False))
        rng.shuffle(sentences)
        passages = tuple((f'd{j + 1}', text) for j, (text, _g) in enumerate(sentences))
        gold_id = next((f'd{j + 1}' for j, (_t, g) in enumerate(sentences) if g), None)
        cases.append(Case(case_id=f'c{i:03d}', question=ATTRIBUTES[asked][1].format(svc=svc),
                          passages=passages, answerable=answerable,
                          gold_answer=gold if answerable else None, gold_id=gold_id, attribute=asked))
    return cases


# ---------------------------------------------------------------------------------------------------
# Prompt versions, output parsing and scoring
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptVersion:
    name: str
    system: str


_BASE = 'Answer the question using only the passages.'
_JSON = (' Reply with a JSON object with keys answer, citations, abstain, where citations lists the '
         'passage ids that support the answer.')
_ABSTAIN = ' If the passages do not contain the answer, set abstain to true and answer to null.'

PROMPT_VERSIONS = {
    'v0_freeform': PromptVersion('v0_freeform', _BASE),
    'v1_json': PromptVersion('v1_json', _BASE + _JSON),
    'v2_json_abstain': PromptVersion('v2_json_abstain', _BASE + _JSON + _ABSTAIN),
}


def get_version(name):
    try:
        return PROMPT_VERSIONS[name]
    except (KeyError, TypeError):
        raise ValueError(f'unknown prompt version {name!r}; known: {sorted(PROMPT_VERSIONS)}') from None


def render_messages(version, case):
    lines = ['Passages:'] + [f'[{pid}] {text}' for pid, text in case.passages] + [f'Question: {case.question}']
    return [{'role': 'system', 'content': version.system}, {'role': 'user', 'content': '\n'.join(lines)}]


def parse_output(text):
    """Return ``(parsed, error)``; error is ``'invalid_json'`` or ``'schema'`` and parsed is then None."""
    if not isinstance(text, str):
        return None, 'invalid_json'
    body = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        obj = json.loads(body)
    except (ValueError, RecursionError):
        return None, 'invalid_json'
    if not isinstance(obj, dict) or set(obj) != {'answer', 'citations', 'abstain'}:
        return None, 'schema'
    answer, citations, abstain = obj['answer'], obj['citations'], obj['abstain']
    if not isinstance(abstain, bool) or not isinstance(citations, list) \
            or not all(isinstance(c, str) for c in citations):
        return None, 'schema'
    if answer is not None and not isinstance(answer, str):
        return None, 'schema'
    if abstain != (answer is None):
        return None, 'schema'
    return obj, None


def _normalize(text):
    return ' '.join(text.lower().split()).strip('.')


def _contains_token(haystack, needle):
    if not needle:
        return False
    return re.search(r'(?<![\w-])' + re.escape(needle) + r'(?![\w-])', haystack) is not None


def score_case(case, parsed, error):
    """Per-request flags. ``grounded_correct`` needs a correct answer, valid citations that contain it,
    or an abstention on an unanswerable case."""
    flags = {'format_error': error is not None, 'invalid_format': error is not None,
             'abstained': False, 'invalid_citation': False, 'ungrounded': False,
             'correct': False, 'grounded_correct': False}
    if error is not None:
        return flags
    ids = {pid: text for pid, text in case.passages}
    flags['invalid_citation'] = any(c not in ids for c in parsed['citations'])
    if parsed['abstain']:
        flags['abstained'] = True
        flags['correct'] = not case.answerable
        flags['grounded_correct'] = flags['correct']
        return flags
    cited = ' '.join(ids[c] for c in parsed['citations'] if c in ids)
    answer = parsed['answer']
    flags['ungrounded'] = not _contains_token(cited, answer)
    flags['correct'] = bool(case.answerable and _normalize(answer) == _normalize(case.gold_answer))
    flags['grounded_correct'] = flags['correct'] and not flags['invalid_citation'] and not flags['ungrounded']
    return flags


def _tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def retrieval_score(case):
    """Best token-overlap (Jaccard) between the question and any passage; a lexical stand-in score."""
    q = _tokens(case.question)
    best = 0.0
    for _pid, text in case.passages:
        p = _tokens(text)
        if q | p:
            best = max(best, len(q & p) / len(q | p))
    return best


# ---------------------------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------------------------

def _unit(seed, key):
    return int(hashlib.sha256(f'{seed}|{key}'.encode()).hexdigest()[:8], 16) / 2 ** 32


class StubGenerator:
    """Deterministic rule-based reader standing in for an LLM; it is not a language model.

    Behaviour depends on the system prompt (JSON requested? abstention allowed?) and, for a fixed share of
    answerable requests chosen by a hash of (seed, question), it cites a passage id that does not exist.
    The ``seconds`` it reports are simulated from token counts, not measured.
    """
    simulated = True
    model_id = 'rule-based-stub'

    def __init__(self, slip_rate=0.1):
        if not 0 <= slip_rate <= 1:
            raise ValueError('slip_rate must be within [0, 1]')
        self.slip_rate = slip_rate

    def generate(self, messages, seed=1, max_tokens=64, temperature=0.0):
        if not messages:
            raise ValueError('messages must not be empty')
        system, user = messages[0]['content'], messages[-1]['content']
        passages = re.findall(r'^\[(d\d+)\] (.+)$', user, re.M)
        match = re.search(r'^Question: (.+)$', user, re.M)
        question = match.group(1) if match else ''
        attribute = svc = None
        for name, regex in _QUESTION_RE.items():
            m = regex.match(question)
            if m:
                attribute, svc = name, m.group('svc')
                break
        found = None
        if attribute:
            for pid, text in passages:
                m = _SENTENCE_RE[attribute].match(text)
                if m and m.group('svc') == svc:
                    found = (pid, m.group('value'))
                    break
        as_json = 'JSON object' in system
        may_abstain = 'set abstain to true' in system
        if found:
            pid, value = found
            cites = ['d9'] if _unit(seed, question) < self.slip_rate else [pid]
            payload = {'answer': value, 'citations': cites, 'abstain': False}
        elif may_abstain:
            payload = {'answer': None, 'citations': [], 'abstain': True}
        else:
            value = str(1000 + int(_unit(seed, 'fab|' + question) * 9000))
            payload = {'answer': value, 'citations': [passages[0][0]] if passages else [], 'abstain': False}
        if as_json:
            text = json.dumps(payload)
        elif payload['abstain']:
            text = 'The passages do not contain the answer.'
        else:
            text = f"The answer is {payload['answer']}."
        input_tokens = len(system.split()) + len(user.split())
        output_tokens = len(text.split())
        return {'text': text, 'input_tokens': input_tokens, 'output_tokens': output_tokens,
                'seconds': 0.040 + 0.0005 * input_tokens + 0.004 * output_tokens}


class LocalLMGenerator:
    """Adapter from ``portfolio.local_lm.LocalLM`` (or any object with the same ``generate``) to this module.

    The real model is only loaded by the CLI ``--model-dir`` option; unit tests use a fake object.
    """
    simulated = False

    def __init__(self, lm, max_tokens=96):
        self.lm = lm
        self.max_tokens = max_tokens
        self.model_id = str(getattr(lm, 'model_path', 'local-lm'))

    def generate(self, messages, seed=1, max_tokens=None, temperature=0.0):
        return self.lm.generate(messages, seed=seed, max_tokens=max_tokens or self.max_tokens,
                                temperature=temperature)


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_generation(out):
    if not isinstance(out, dict) or not isinstance(out.get('text'), str):
        raise ValueError('generator must return a dict with a string "text"')
    for key in ('input_tokens', 'output_tokens', 'seconds'):
        if not _is_num(out.get(key)) or out[key] < 0:
            raise ValueError(f'generator field {key!r} must be a finite non-negative number')
    return out


# ---------------------------------------------------------------------------------------------------
# Spans (trace records)
# ---------------------------------------------------------------------------------------------------

def request_spans(trace_key, start_ms, latency_ms, attributes, model_id='unknown', simulated=False):
    """One request trace: a root ``llm.request`` span and a child ``llm.generate`` span."""
    trace_id = hashlib.sha256(trace_key.encode()).hexdigest()[:32]
    root_id = hashlib.sha256((trace_id + '|root').encode()).hexdigest()[:16]
    child_id = hashlib.sha256((trace_id + '|generate').encode()).hexdigest()[:16]
    root = {'trace_id': trace_id, 'span_id': root_id, 'parent_span_id': None, 'name': REQUEST_SPAN,
            'start_ms': float(start_ms), 'duration_ms': float(latency_ms),
            'status': 'error' if attributes.get('invalid_format') else 'ok',
            'attributes': {**attributes, 'simulated': bool(simulated)}}
    child = {'trace_id': trace_id, 'span_id': child_id, 'parent_span_id': root_id, 'name': GENERATE_SPAN,
             'start_ms': float(start_ms), 'duration_ms': float(latency_ms), 'status': 'ok',
             'attributes': {'model': model_id, 'input_tokens': attributes.get('input_tokens'),
                            'output_tokens': attributes.get('output_tokens')}}
    return [root, child]


def validate_spans(spans):
    """Check span shape, unique ids and parent links; raise ValueError naming the first problem."""
    seen = {}
    for i, span in enumerate(spans):
        where = f'span {i}'
        if not isinstance(span, dict):
            raise ValueError(f'{where}: not an object')
        for key in ('trace_id', 'span_id', 'name'):
            if not isinstance(span.get(key), str) or not span[key]:
                raise ValueError(f'{where}: {key} must be a non-empty string')
        parent = span.get('parent_span_id')
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f'{where}: parent_span_id must be a string or null')
        for key in ('start_ms', 'duration_ms'):
            if not _is_num(span.get(key)) or span[key] < 0:
                raise ValueError(f'{where}: {key} must be a finite non-negative number')
        if span.get('status') not in ('ok', 'error'):
            raise ValueError(f'{where}: status must be "ok" or "error"')
        if not isinstance(span.get('attributes'), dict):
            raise ValueError(f'{where}: attributes must be an object')
        key = (span['trace_id'], span['span_id'])
        if key in seen:
            raise ValueError(f'{where}: duplicate span id {span["span_id"]}')
        seen[key] = span
    for i, span in enumerate(spans):
        parent = span.get('parent_span_id')
        if parent is not None and (span['trace_id'], parent) not in seen:
            raise ValueError(f'span {i}: parent {parent} not found in trace {span["trace_id"]}')
    return spans


def export_traces_jsonl(spans, path):
    """Write one JSON span per line. Validates first; refuses to overwrite an existing file."""
    spans = list(spans)
    if not spans:
        raise ValueError('no spans to export')
    validate_spans(spans)
    lines = [json.dumps(s, sort_keys=True, allow_nan=False) for s in spans]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return len(lines)


def load_traces_jsonl(path):
    spans = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            spans.append(json.loads(line))
        except ValueError:
            raise ValueError(f'line {number}: invalid JSON') from None
    if not spans:
        raise ValueError('trace file contains no spans')
    try:
        validate_spans(spans)
    except ValueError as exc:
        raise ValueError(f'invalid trace file: {exc}') from None
    return spans


# ---------------------------------------------------------------------------------------------------
# Request records and windowed statistics
# ---------------------------------------------------------------------------------------------------

_NUMERIC = ('ts_ms', 'latency_ms', 'input_tokens', 'output_tokens')
_FLAGS = ('invalid_citation', 'ungrounded', 'abstained', 'invalid_format')


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError('record must be an object')
    for key in _NUMERIC:
        if not _is_num(record.get(key)) or record[key] < 0:
            raise ValueError(f'record field {key!r} must be a finite non-negative number')
    for key in _FLAGS:
        if not isinstance(record.get(key), bool):
            raise ValueError(f'record field {key!r} must be a boolean')
    score = record.get('retrieval_score')
    if score is not None and not _is_num(score):
        raise ValueError("record field 'retrieval_score' must be a finite number or null")
    return record


def records_from_spans(spans):
    """Build monitor records from the root ``llm.request`` spans, ordered by start time."""
    validate_spans(spans)
    records = []
    for span in spans:
        if span['name'] != REQUEST_SPAN:
            continue
        a = span['attributes']
        records.append(validate_record({
            'ts_ms': span['start_ms'], 'latency_ms': span['duration_ms'],
            'input_tokens': a.get('input_tokens'), 'output_tokens': a.get('output_tokens'),
            'retrieval_score': a.get('retrieval_score'),
            'invalid_citation': a.get('invalid_citation'), 'ungrounded': a.get('ungrounded'),
            'abstained': a.get('abstained'), 'invalid_format': a.get('invalid_format')}))
    if not records:
        raise ValueError(f'no {REQUEST_SPAN} spans found')
    return sorted(records, key=lambda r: r['ts_ms'])


def spans_from_records(records, run_id='replay', prompt_version='synthetic'):
    spans = []
    for i, r in enumerate(records):
        validate_record(r)
        attrs = {'prompt_version': prompt_version, 'input_tokens': r['input_tokens'],
                 'output_tokens': r['output_tokens'], 'retrieval_score': r.get('retrieval_score'),
                 'invalid_citation': r['invalid_citation'], 'ungrounded': r['ungrounded'],
                 'abstained': r['abstained'], 'invalid_format': r['invalid_format']}
        spans += request_spans(f'{run_id}|{i}', r['ts_ms'], r['latency_ms'], attrs,
                               model_id='synthetic-stream', simulated=True)
    return spans


def percentile(values, q):
    """Linear-interpolation percentile (NumPy's default definition), q in [0, 100]."""
    if not _is_num(q) or not 0 <= q <= 100:
        raise ValueError('q must be within [0, 100]')
    ordered = sorted(distribution(values)['raw_ms'])   # validates: nonempty, finite, non-negative
    point = q / 100.0 * (len(ordered) - 1)
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def latency_stats(values):
    """p50/p95 from ``measurement.distribution`` plus p99 from :func:`percentile`."""
    d = distribution(values)
    return {'count': d['count'], 'p50_ms': d['p50_ms'], 'p95_ms': d['p95_ms'],
            'p99_ms': percentile(values, 99)}


def _vector(values, name):
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be a nonempty 1-D sequence of finite numbers')
    return arr


def psi(reference, current, bins=5, eps=1e-4):
    """Population stability index with reference-quantile bins (a constant reference gets below/equal/above)."""
    ref, cur = _vector(reference, 'reference'), _vector(current, 'current')
    if int(bins) != bins or bins < 2:
        raise ValueError('bins must be an integer >= 2')
    if np.ptp(ref) == 0:
        edges = np.array([ref[0], np.nextafter(ref[0], np.inf)])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, int(bins) + 1)[1:-1]))
    k = len(edges) + 1
    p_ref = np.bincount(np.searchsorted(edges, ref, side='right'), minlength=k) / ref.size
    p_cur = np.bincount(np.searchsorted(edges, cur, side='right'), minlength=k) / cur.size
    p_ref, p_cur = np.clip(p_ref, eps, None), np.clip(p_cur, eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov distance (maximum ECDF difference)."""
    a, b = np.sort(_vector(a, 'a')), np.sort(_vector(b, 'b'))
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side='right') / a.size
                               - np.searchsorted(b, grid, side='right') / b.size)))


def ks_critical(n, m, alpha):
    """Asymptotic two-sample critical distance at significance ``alpha``."""
    if n < 1 or m < 1 or not 0 < alpha < 1:
        raise ValueError('need n, m >= 1 and 0 < alpha < 1')
    return math.sqrt(-0.5 * math.log(alpha / 2.0)) * math.sqrt((n + m) / (n * m))


@dataclass(frozen=True)
class MonitorConfig:
    window_ms: float = 20000.0
    min_window_count: int = 30
    min_reference: int = 100
    latency_ratio: float = 1.5
    rate_margin: float = 0.05
    psi_bins: int = 5
    psi_alert: float = 0.25
    ks_alpha: float = 0.0001
    drift_features: tuple = ('input_tokens', 'output_tokens', 'retrieval_score')

    def __post_init__(self):
        if not self.window_ms > 0:
            raise ValueError('window_ms must be positive')
        if self.min_window_count < 1 or self.min_reference < 1:
            raise ValueError('minimum counts must be >= 1')
        if not self.latency_ratio > 1:
            raise ValueError('latency_ratio must exceed 1')
        if not 0 <= self.rate_margin < 1:
            raise ValueError('rate_margin must be within [0, 1)')
        if self.psi_bins < 2 or not self.psi_alert > 0 or not 0 < self.ks_alpha < 1:
            raise ValueError('invalid drift thresholds')


# metric name, record flag, alert kind, alert on a decrease too
_RATES = (('invalid_citation_rate', 'invalid_citation', 'hallucination_proxy', False),
          ('ungrounded_rate', 'ungrounded', 'hallucination_proxy', False),
          ('invalid_format_rate', 'invalid_format', 'format', False),
          ('abstention_rate', 'abstained', 'abstention', True))


def _rate(records, flag):
    return sum(1 for r in records if r[flag]) / len(records)


def monitor(reference, current, config=None):
    """Compare windows of ``current`` records with ``reference`` and return stats and alerts.

    Latency: a window p95/p99 above ``latency_ratio`` times the reference value alerts. Rates: a window
    alerts when the Wilson 95% interval lies entirely beyond the reference rate by ``rate_margin``
    (above only, except abstention which also alerts below). Drift: PSI >= ``psi_alert`` or a KS distance
    above its asymptotic critical value at ``ks_alpha``. Windows with fewer than ``min_window_count``
    records are reported as ``insufficient_data`` and never alert.
    """
    cfg = config or MonitorConfig()
    ref = [validate_record(r) for r in reference]
    cur = sorted((validate_record(r) for r in current), key=lambda r: r['ts_ms'])
    if len(ref) < cfg.min_reference:
        raise ValueError(f'need at least {cfg.min_reference} reference records, got {len(ref)}')
    if not cur:
        raise ValueError('no current records to monitor')
    ref_latency = latency_stats([r['latency_ms'] for r in ref])
    ref_rates = {name: _rate(ref, flag) for name, flag, _k, _b in _RATES}
    ref_features = {f: [r[f] for r in ref if r.get(f) is not None] for f in cfg.drift_features}
    t0 = cur[0]['ts_ms']
    buckets = {}
    for r in cur:
        buckets.setdefault(int((r['ts_ms'] - t0) // cfg.window_ms), []).append(r)
    windows, alerts = [], []
    for index in sorted(buckets):
        recs = buckets[index]
        start = t0 + index * cfg.window_ms
        window = {'index': index, 'start_ms': start, 'count': len(recs), 'status': 'insufficient_data',
                  'latency': None, 'rates': None, 'drift': {}}
        windows.append(window)
        if len(recs) < cfg.min_window_count:
            continue
        window['status'] = 'evaluated'

        def alert(kind, metric, observed, reference_value, threshold, direction='up'):
            alerts.append({'window_index': index, 'window_start_ms': start, 'kind': kind, 'metric': metric,
                           'observed': float(observed), 'reference': float(reference_value),
                           'threshold': float(threshold), 'direction': direction})

        stats = latency_stats([r['latency_ms'] for r in recs])
        window['latency'] = stats
        for metric in ('p95_ms', 'p99_ms'):
            threshold = ref_latency[metric] * cfg.latency_ratio
            if stats[metric] > threshold:
                alert('latency', metric, stats[metric], ref_latency[metric], threshold)
        window['rates'] = {}
        for name, flag, kind, both in _RATES:
            hits = sum(1 for r in recs if r[flag])
            observed = hits / len(recs)
            window['rates'][name] = observed
            low, high = wilson(hits, len(recs))
            if low > ref_rates[name] + cfg.rate_margin:
                alert(kind, name, observed, ref_rates[name], ref_rates[name] + cfg.rate_margin)
            elif both and high < ref_rates[name] - cfg.rate_margin:
                alert(kind, name, observed, ref_rates[name], ref_rates[name] - cfg.rate_margin, 'down')
        for feature in cfg.drift_features:
            ref_values = ref_features[feature]
            values = [r[feature] for r in recs if r.get(feature) is not None]
            if len(ref_values) < cfg.min_reference or len(values) < cfg.min_window_count:
                continue
            value_psi = psi(ref_values, values, cfg.psi_bins)
            distance = ks_statistic(ref_values, values)
            critical = ks_critical(len(ref_values), len(values), cfg.ks_alpha)
            window['drift'][feature] = {'psi': value_psi, 'ks': distance, 'ks_critical': critical}
            if value_psi >= cfg.psi_alert:
                alert('drift', f'psi:{feature}', value_psi, 0.0, cfg.psi_alert)
            if distance > critical:
                alert('drift', f'ks:{feature}', distance, 0.0, critical)
    return {'status': 'alert' if alerts else 'ok', 'reference': {'count': len(ref), 'latency': ref_latency,
            'rates': ref_rates}, 'windows': windows, 'alerts': alerts,
            'alert_kinds': sorted({a['kind'] for a in alerts}), 'config': asdict(cfg), 'scope': SCOPE}


def synthetic_records(seed, n, start_ms=0.0, interval_ms=100.0, latency_median_ms=120.0, latency_sigma=0.25,
                      input_tokens_mean=300.0, output_tokens_mean=40.0, retrieval_mean=0.6,
                      invalid_citation_rate=0.02, ungrounded_rate=0.03, abstain_rate=0.20,
                      invalid_format_rate=0.01):
    """Seeded synthetic request stream; fault injection is done by changing the parameters."""
    if int(n) != n or n < 1:
        raise ValueError('n must be a positive integer')
    for name, rate in (('invalid_citation_rate', invalid_citation_rate), ('ungrounded_rate', ungrounded_rate),
                       ('abstain_rate', abstain_rate), ('invalid_format_rate', invalid_format_rate)):
        if not 0 <= rate <= 1:
            raise ValueError(f'{name} must be within [0, 1]')
    rng = np.random.default_rng(seed)
    n = int(n)
    latency = latency_median_ms * np.exp(latency_sigma * rng.standard_normal(n))
    inputs = np.maximum(1.0, np.rint(rng.normal(input_tokens_mean, 0.15 * input_tokens_mean, n)))
    outputs = np.maximum(1.0, np.rint(rng.normal(output_tokens_mean, 0.15 * output_tokens_mean, n)))
    retrieval = np.clip(rng.normal(retrieval_mean, 0.1, n), 0.0, 1.0)
    flags = {key: rng.random(n) < rate for key, rate in (
        ('invalid_citation', invalid_citation_rate), ('ungrounded', ungrounded_rate),
        ('abstained', abstain_rate), ('invalid_format', invalid_format_rate))}
    return [{'ts_ms': float(start_ms + i * interval_ms), 'latency_ms': float(latency[i]),
             'input_tokens': int(inputs[i]), 'output_tokens': int(outputs[i]),
             'retrieval_score': float(retrieval[i]),
             **{key: bool(flags[key][i]) for key in flags}} for i in range(n)]


# ---------------------------------------------------------------------------------------------------
# Paired prompt-version evaluation
# ---------------------------------------------------------------------------------------------------

def _check_seeds(seeds):
    seeds = list(seeds)
    if not seeds:
        raise ValueError('at least one seed is required')
    return seeds


def evaluate_version(cases, version_name, generator, seeds, run_id='eval', clock_ms=0.0):
    """Run one prompt version over every case and seed; returns rows, the outcome matrix and spans."""
    version = get_version(version_name)
    seeds = _check_seeds(seeds)
    cases = list(cases)
    if not cases:
        raise ValueError('at least one case is required')
    simulated = bool(getattr(generator, 'simulated', False))
    model_id = str(getattr(generator, 'model_id', 'unknown'))
    rows, matrix, spans = [], [], []
    clock = float(clock_ms)
    for case in cases:
        matrix_row = []
        for seed in seeds:
            out = _check_generation(generator.generate(render_messages(version, case), seed=seed))
            parsed, error = parse_output(out['text'])
            flags = score_case(case, parsed, error)
            latency_ms = out['seconds'] * 1000.0
            row = {'case_id': case.case_id, 'seed': seed, 'version': version.name,
                   'answerable': case.answerable, 'latency_ms': latency_ms,
                   'input_tokens': int(out['input_tokens']), 'output_tokens': int(out['output_tokens']),
                   'error': error, **flags}
            rows.append(row)
            matrix_row.append(1.0 if flags['grounded_correct'] else 0.0)
            attrs = {'prompt_version': version.name, 'case_id': case.case_id, 'seed': seed,
                     'input_tokens': row['input_tokens'], 'output_tokens': row['output_tokens'],
                     'retrieval_score': retrieval_score(case), 'invalid_citation': flags['invalid_citation'],
                     'ungrounded': flags['ungrounded'], 'abstained': flags['abstained'],
                     'invalid_format': flags['invalid_format'], 'grounded_correct': flags['grounded_correct'],
                     'parse_error': error}
            spans += request_spans(f'{run_id}|{version.name}|{case.case_id}|{seed}', clock, latency_ms, attrs,
                                   model_id=model_id, simulated=simulated)
            clock += latency_ms + 1.0
        matrix.append(matrix_row)
    return {'version': version.name, 'case_ids': [c.case_id for c in cases], 'seeds': seeds,
            'rows': rows, 'matrix': matrix, 'spans': spans, 'end_clock_ms': clock}


def summarize_rows(rows):
    if not rows:
        raise ValueError('no rows to summarize')
    n = len(rows)
    answerable = [r for r in rows if r['answerable']]
    unanswerable = [r for r in rows if not r['answerable']]

    def share(items, key):
        return sum(1 for r in items if r[key]) / len(items) if items else None

    return {'requests': n, 'grounded_correct_rate': share(rows, 'grounded_correct'),
            'answerable_grounded_correct_rate': share(answerable, 'grounded_correct'),
            'unanswerable_abstention_rate': share(unanswerable, 'abstained'),
            'format_error_rate': share(rows, 'format_error'),
            'invalid_citation_rate': share(rows, 'invalid_citation'),
            'ungrounded_rate': share(rows, 'ungrounded'),
            'latency': latency_stats([r['latency_ms'] for r in rows])}


def compare_evaluations(a, b, seed=73, draws=2000):
    """Paired comparison of two evaluations that used the same cases and seeds."""
    if a['case_ids'] != b['case_ids'] or a['seeds'] != b['seeds']:
        raise ValueError('evaluations must use the same cases and seeds to be paired')
    diff = clustered_difference(a['matrix'], b['matrix'], seed=seed, draws=draws)
    low, high = diff['ci95']
    decision = 'b_better' if low > 0 else 'a_better' if high < 0 else 'inconclusive'
    return {'a': a['version'], 'b': b['version'], 'a_summary': summarize_rows(a['rows']),
            'b_summary': summarize_rows(b['rows']), 'difference_b_minus_a': diff, 'decision': decision}


def compare_versions(cases, name_a, name_b, generator, seeds, run_id='compare', seed=73, draws=2000):
    cases = list(cases)
    ev_a = evaluate_version(cases, name_a, generator, seeds, run_id)
    ev_b = evaluate_version(cases, name_b, generator, seeds, run_id, clock_ms=ev_a['end_clock_ms'])
    return compare_evaluations(ev_a, ev_b, seed=seed, draws=draws)


# ---------------------------------------------------------------------------------------------------
# Fault-injection scenarios and the finite check
# ---------------------------------------------------------------------------------------------------

SCENARIOS = {
    'latency_spike': ({'latency_median_ms': 300.0}, ('latency', 'p95_ms')),
    'invalid_citation_surge': ({'invalid_citation_rate': 0.30}, ('hallucination_proxy', 'invalid_citation_rate')),
    'ungrounded_surge': ({'ungrounded_rate': 0.25}, ('hallucination_proxy', 'ungrounded_rate')),
    'format_failures': ({'invalid_format_rate': 0.20}, ('format', 'invalid_format_rate')),
    'abstention_surge': ({'abstain_rate': 0.50}, ('abstention', 'abstention_rate')),
    'abstention_collapse': ({'abstain_rate': 0.0}, ('abstention', 'abstention_rate')),
    'input_length_drift': ({'input_tokens_mean': 450.0}, ('drift', 'psi:input_tokens')),
    'retrieval_score_drift': ({'retrieval_mean': 0.35}, ('drift', 'psi:retrieval_score')),
}
REFERENCE_N, CURRENT_N = 600, 600


def _scenario(seed, name, params):
    reference = synthetic_records(seed, REFERENCE_N)
    current = synthetic_records(seed + 1, CURRENT_N, start_ms=REFERENCE_N * 100.0, **params)
    report = monitor(reference, current, MonitorConfig())
    return {'scenario': name, 'status': report['status'], 'windows': len(report['windows']),
            'alerts': [{k: a[k] for k in ('window_index', 'kind', 'metric', 'direction')}
                       for a in report['alerts']]}


def _has(result, kind, metric):
    return any(a['kind'] == kind and a['metric'] == metric for a in result['alerts'])


def run_demo(seed=17, generator=None):
    """Finite check. Returns ``(summary, spans)``; ``summary['all_passed']`` is the pass/fail verdict."""
    generator = generator or StubGenerator()
    simulated = bool(getattr(generator, 'simulated', False))
    cases = make_cases(seed)
    seeds = [1, 2, 3]
    run_id = f'demo-{seed}'
    evals, clock = {}, 0.0
    for name in PROMPT_VERSIONS:
        evals[name] = evaluate_version(cases, name, generator, seeds, run_id, clock_ms=clock)
        clock = evals[name]['end_clock_ms']
    pairs = {'v1_json_to_v2_json_abstain': ('v1_json', 'v2_json_abstain'),
             'v2_json_abstain_to_v1_json': ('v2_json_abstain', 'v1_json'),
             'v2_json_abstain_to_itself': ('v2_json_abstain', 'v2_json_abstain'),
             'v0_freeform_to_v1_json': ('v0_freeform', 'v1_json')}
    comparisons = {key: compare_evaluations(evals[x], evals[y]) for key, (x, y) in pairs.items()}
    scenarios = {'control_same_distribution': _scenario(seed, 'control_same_distribution', {})}
    for name, (params, _expect) in SCENARIOS.items():
        scenarios[name] = _scenario(seed, name, params)
    spans = [s for ev in evals.values() for s in ev['spans']]
    with tempfile.TemporaryDirectory(prefix='llm-quality-ops-') as temp:
        path = Path(temp) / 'traces.jsonl'
        exported = export_traces_jsonl(spans, path)
        reloaded = load_traces_jsonl(path)
    checks = {'trace_roundtrip_identical': reloaded == spans and exported == len(spans),
              'control_has_no_alerts': scenarios['control_same_distribution']['status'] == 'ok'}
    for name, (_params, (kind, metric)) in SCENARIOS.items():
        checks[f'{name}_detected'] = _has(scenarios[name], kind, metric)
    if simulated:
        checks['abstain_prompt_beats_json_only'] = comparisons['v1_json_to_v2_json_abstain']['decision'] == 'b_better'
        checks['reversed_order_prefers_a'] = comparisons['v2_json_abstain_to_v1_json']['decision'] == 'a_better'
        checks['identical_versions_inconclusive'] = comparisons['v2_json_abstain_to_itself']['decision'] == 'inconclusive'
        checks['freeform_fails_format'] = comparisons['v0_freeform_to_v1_json']['a_summary']['format_error_rate'] == 1.0
    summary = {'seed': seed, 'cases': len(cases), 'repeats_per_case': len(seeds),
               'generator': {'model_id': str(getattr(generator, 'model_id', 'unknown')), 'simulated': simulated,
                             'note': 'rule-based stub; latency is simulated from token counts' if simulated
                             else 'local model; synthetic cases, small sample; prompt-comparison expectations not asserted'},
               'comparisons': comparisons, 'monitoring_scenarios': scenarios,
               'traces_exported': exported, 'checks': checks, 'all_passed': all(checks.values()),
               'scope': SCOPE}
    return summary, spans


def render_report(summary):
    lines = ['# LLM quality operations run', '', summary['scope'], '',
             f"Generator: `{summary['generator']['model_id']}` ({summary['generator']['note']}).", '',
             '## Prompt-version comparisons (paired, case-clustered bootstrap)', '',
             '| Comparison | Mean difference (b - a) | 95% CI | Decision |', '|---|---:|---|---|']
    for key, c in summary['comparisons'].items():
        d = c['difference_b_minus_a']
        lines.append(f"| {key} | {d['mean_difference']:.3f} | [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] | {c['decision']} |")
    lines += ['', '## Monitoring scenarios', '', '| Scenario | Status | Alerts (window, kind, metric) |', '|---|---|---|']
    for name, s in summary['monitoring_scenarios'].items():
        shown = '; '.join(f"{a['window_index']}, {a['kind']}, {a['metric']}" for a in s['alerts']) or 'none'
        lines.append(f"| {name} | {s['status']} | {shown} |")
    lines += ['', '## Checks', ''] + [f"- {k}: {'pass' if v else 'FAIL'}" for k, v in summary['checks'].items()]
    return '\n'.join(lines) + '\n'


def _replay(path, reference_count, window_ms):
    records = records_from_spans(load_traces_jsonl(path))
    if int(reference_count) != reference_count or not 0 < reference_count < len(records):
        raise ValueError(f'reference-count must be between 1 and {len(records) - 1}, got {reference_count}')
    report = monitor(records[:reference_count], records[reference_count:],
                     MonitorConfig(window_ms=window_ms, min_reference=min(100, reference_count)))
    return {'status': report['status'], 'alert_kinds': report['alert_kinds'],
            'alerts': [{k: a[k] for k in ('window_index', 'kind', 'metric', 'observed', 'threshold', 'direction')}
                       for a in report['alerts']],
            'windows': len(report['windows']), 'scope': SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--out', help='record the run (protocol, summary, REPORT.md, traces.jsonl) in a new empty directory')
    parser.add_argument('--model-dir', help='use a downloaded local model instead of the stub generator')
    parser.add_argument('--replay', help='monitor a recorded traces.jsonl instead of running the demo')
    parser.add_argument('--reference-count', type=int, default=600, help='replay: leading records used as reference')
    parser.add_argument('--window-ms', type=float, default=20000.0, help='replay: window length')
    args = parser.parse_args(argv)
    try:
        if args.replay:
            result = _replay(args.replay, args.reference_count, args.window_ms)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 3 if result['status'] == 'alert' else 0
        generator = None
        if args.model_dir:
            from .local_lm import LocalLM
            generator = LocalLMGenerator(LocalLM(args.model_dir))
        if args.out:
            protocol = {'tool': 'llm_quality_ops', 'seed': args.seed, 'scope': SCOPE,
                        'generator': 'local model' if args.model_dir else 'rule-based stub'}
            out = Path(args.out)
            with RunRecord(out, protocol, [Path(__file__)]):
                summary, spans = run_demo(args.seed, generator)
                write_json(out / 'summary.json', summary)
                export_traces_jsonl(spans, out / 'traces.jsonl')
                (out / 'REPORT.md').write_text(render_report(summary), encoding='utf-8')
        else:
            summary, _spans = run_demo(args.seed, generator)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0 if summary['all_passed'] else 1
    except (ValueError, OSError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
