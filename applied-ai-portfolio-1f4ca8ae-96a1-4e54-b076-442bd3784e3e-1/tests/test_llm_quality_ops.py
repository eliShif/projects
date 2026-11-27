"""Prompt-version comparison, windowed monitoring, drift and trace export, with success and failure paths.

All data are synthetic and the generator is a rule-based stub, so these tests verify the code paths and
their failure handling, not the quality of any prompt or model.
"""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from portfolio import llm_quality_ops as ops
from portfolio.measurement import distribution

SEEDS = [1, 2, 3]
REF_N = 600


def reference():
    return ops.synthetic_records(100, REF_N)


def current(seed=101, n=600, **params):
    return ops.synthetic_records(seed, n, start_ms=REF_N * 100.0, **params)


def metrics(report, kind=None):
    return {a['metric'] for a in report['alerts'] if kind is None or a['kind'] == kind}


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ops.main(argv)
    return code, out.getvalue(), err.getvalue()


class CaseTests(unittest.TestCase):
    def test_cases_are_deterministic_and_well_formed(self):
        a, b = ops.make_cases(5), ops.make_cases(5)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertEqual(sum(1 for c in a if not c.answerable), 6)
        for case in a:
            texts = dict(case.passages)
            if case.answerable:
                self.assertIn(case.gold_answer, texts[case.gold_id])
            else:
                self.assertIsNone(case.gold_answer)
                self.assertIsNone(case.gold_id)

    def test_bad_arguments_are_rejected(self):
        for kwargs in ({'n': 1}, {'n': 4, 'unanswerable': 5}, {'n': 4, 'unanswerable': -1}):
            with self.assertRaises(ValueError):
                ops.make_cases(1, **kwargs)


class ParsingAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.answerable = next(c for c in ops.make_cases(5) if c.answerable)
        self.unanswerable = next(c for c in ops.make_cases(5) if not c.answerable)

    def payload(self, **kw):
        return json.dumps({'answer': None, 'citations': [], 'abstain': True, **kw})

    def test_parse_accepts_plain_and_fenced_json(self):
        text = self.payload(answer='1', abstain=False, citations=['d1'])
        self.assertEqual(ops.parse_output(text)[1], None)
        self.assertEqual(ops.parse_output('```json\n' + text + '\n```')[1], None)

    def test_parse_rejects_malformed_outputs(self):
        cases = {'not json': 'invalid_json', '': 'invalid_json', '[1]': 'schema', '{"answer": "x"}': 'schema',
                 self.payload(abstain='yes'): 'schema', self.payload(citations='d1'): 'schema',
                 self.payload(answer=5, abstain=False): 'schema',
                 self.payload(answer='x', abstain=True): 'schema'}
        for text, expected in cases.items():
            parsed, error = ops.parse_output(text)
            self.assertIsNone(parsed, text)
            self.assertEqual(error, expected, text)
        self.assertEqual(ops.parse_output(None), (None, 'invalid_json'))

    def score(self, case, **kw):
        parsed, error = ops.parse_output(json.dumps({'answer': None, 'citations': [], 'abstain': True, **kw}))
        return ops.score_case(case, parsed, error)

    def test_grounded_correct_answer(self):
        c = self.answerable
        flags = self.score(c, answer=c.gold_answer, citations=[c.gold_id], abstain=False)
        self.assertTrue(flags['grounded_correct'])
        self.assertFalse(flags['invalid_citation'] or flags['ungrounded'])

    def test_invalid_citation_blocks_grounded_correct(self):
        c = self.answerable
        flags = self.score(c, answer=c.gold_answer, citations=['d9'], abstain=False)
        self.assertTrue(flags['correct'] and flags['invalid_citation'] and flags['ungrounded'])
        self.assertFalse(flags['grounded_correct'])

    def test_correct_answer_with_unsupporting_citation_is_ungrounded(self):
        c = self.answerable
        other = next(pid for pid, _t in c.passages if pid != c.gold_id)
        flags = self.score(c, answer=c.gold_answer, citations=[other], abstain=False)
        self.assertTrue(flags['correct'] and flags['ungrounded'])
        self.assertFalse(flags['grounded_correct'])

    def test_wrong_answer_is_not_correct(self):
        c = self.answerable
        flags = self.score(c, answer='no-such-value', citations=[c.gold_id], abstain=False)
        self.assertFalse(flags['correct'] or flags['grounded_correct'])
        self.assertTrue(flags['ungrounded'])

    def test_abstention_is_right_only_when_unanswerable(self):
        self.assertTrue(self.score(self.unanswerable)['grounded_correct'])
        flags = self.score(self.answerable)
        self.assertTrue(flags['abstained'])
        self.assertFalse(flags['grounded_correct'])

    def test_fabricated_answer_on_unanswerable_case_fails(self):
        c = self.unanswerable
        flags = self.score(c, answer='1234', citations=[c.passages[0][0]], abstain=False)
        self.assertFalse(flags['grounded_correct'])
        self.assertTrue(flags['ungrounded'])

    def test_format_error_scores_as_failure(self):
        flags = ops.score_case(self.answerable, None, 'invalid_json')
        self.assertTrue(flags['format_error'] and flags['invalid_format'])
        self.assertFalse(flags['grounded_correct'])


class PromptComparisonTests(unittest.TestCase):
    def setUp(self):
        self.cases = ops.make_cases(7)
        self.gen = ops.StubGenerator()

    def test_abstention_instruction_improves_grounded_correct(self):
        res = ops.compare_versions(self.cases, 'v1_json', 'v2_json_abstain', self.gen, SEEDS)
        diff = res['difference_b_minus_a']
        self.assertAlmostEqual(diff['mean_difference'], 6 / 24)
        self.assertGreater(diff['ci95'][0], 0.0)
        self.assertEqual(res['decision'], 'b_better')
        self.assertEqual(res['b_summary']['unanswerable_abstention_rate'], 1.0)
        self.assertEqual(res['a_summary']['unanswerable_abstention_rate'], 0.0)
        self.assertEqual(diff['unit'], 'question')

    def test_reversed_order_prefers_a(self):
        res = ops.compare_versions(self.cases, 'v2_json_abstain', 'v1_json', self.gen, SEEDS)
        self.assertEqual(res['decision'], 'a_better')
        self.assertLess(res['difference_b_minus_a']['ci95'][1], 0.0)

    def test_identical_versions_are_inconclusive(self):
        res = ops.compare_versions(self.cases, 'v1_json', 'v1_json', self.gen, SEEDS)
        self.assertEqual(res['difference_b_minus_a']['mean_difference'], 0.0)
        self.assertEqual(res['decision'], 'inconclusive')

    def test_freeform_prompt_fails_the_format_contract(self):
        res = ops.compare_versions(self.cases, 'v0_freeform', 'v1_json', self.gen, SEEDS)
        self.assertEqual(res['a_summary']['format_error_rate'], 1.0)
        self.assertEqual(res['a_summary']['grounded_correct_rate'], 0.0)
        self.assertEqual(res['decision'], 'b_better')

    def test_comparison_is_deterministic(self):
        a = ops.compare_versions(self.cases, 'v1_json', 'v2_json_abstain', self.gen, SEEDS)
        b = ops.compare_versions(self.cases, 'v1_json', 'v2_json_abstain', self.gen, SEEDS)
        self.assertEqual(a, b)

    def test_versions_are_paired_on_the_same_cases_and_seeds(self):
        e1 = ops.evaluate_version(self.cases, 'v1_json', self.gen, SEEDS)
        e2 = ops.evaluate_version(self.cases, 'v2_json_abstain', self.gen, SEEDS)
        self.assertEqual([(r['case_id'], r['seed']) for r in e1['rows']],
                         [(r['case_id'], r['seed']) for r in e2['rows']])
        self.assertEqual(len(e1['matrix']), len(self.cases))
        self.assertEqual(len(e1['matrix'][0]), len(SEEDS))

    def test_unpaired_evaluations_are_refused(self):
        a = ops.evaluate_version(self.cases, 'v1_json', self.gen, SEEDS)
        b = ops.evaluate_version(self.cases, 'v2_json_abstain', self.gen, [1, 2])
        with self.assertRaises(ValueError):
            ops.compare_evaluations(a, b)
        c = ops.evaluate_version(self.cases[:-1], 'v2_json_abstain', self.gen, SEEDS)
        with self.assertRaises(ValueError):
            ops.compare_evaluations(a, c)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ops.evaluate_version(self.cases, 'v9_missing', self.gen, SEEDS)
        with self.assertRaises(ValueError):
            ops.evaluate_version(self.cases, 'v1_json', self.gen, [])
        with self.assertRaises(ValueError):
            ops.evaluate_version([], 'v1_json', self.gen, SEEDS)
        with self.assertRaises(ValueError):   # the bootstrap needs at least two paired cases
            ops.compare_versions(self.cases[:1], 'v1_json', 'v2_json_abstain', self.gen, SEEDS)

    def test_malformed_generator_output_is_rejected(self):
        class Bad:
            def __init__(self, out):
                self.out = out

            def generate(self, messages, seed=1):
                return self.out

        good = {'text': '{}', 'input_tokens': 1, 'output_tokens': 1, 'seconds': 0.1}
        for out in ({**good, 'text': 5}, {**good, 'seconds': -1.0}, {**good, 'seconds': float('nan')},
                    {k: v for k, v in good.items() if k != 'input_tokens'}, 'text'):
            with self.assertRaises(ValueError):
                ops.evaluate_version(self.cases, 'v1_json', Bad(out), SEEDS)

    def test_local_lm_adapter_wraps_any_generate_object(self):
        class FakeLM:
            model_path = 'fake-model'

            def generate(self, messages, seed=1, max_tokens=48, temperature=.7):
                return {'text': '{"answer": null, "citations": [], "abstain": true}',
                        'input_tokens': 10, 'output_tokens': 5, 'seconds': 0.2}

        gen = ops.LocalLMGenerator(FakeLM())
        ev = ops.evaluate_version(self.cases, 'v2_json_abstain', gen, [1])
        self.assertTrue(all(r['abstained'] for r in ev['rows']))
        root = next(s for s in ev['spans'] if s['name'] == ops.REQUEST_SPAN)
        self.assertFalse(root['attributes']['simulated'])
        child = next(s for s in ev['spans'] if s['name'] == ops.GENERATE_SPAN)
        self.assertEqual(child['attributes']['model'], 'fake-model')
        self.assertAlmostEqual(root['duration_ms'], 200.0)

    def test_stub_slip_rate_is_validated(self):
        with self.assertRaises(ValueError):
            ops.StubGenerator(slip_rate=1.5)


class StatisticsTests(unittest.TestCase):
    def test_percentiles_match_linear_definition(self):
        values = [float(v) for v in range(1, 101)]
        stats = ops.latency_stats(values)
        self.assertAlmostEqual(stats['p50_ms'], 50.5)
        self.assertAlmostEqual(stats['p95_ms'], 95.05)
        self.assertAlmostEqual(stats['p99_ms'], 99.01)
        self.assertAlmostEqual(ops.percentile(values, 95), distribution(values)['p95_ms'])
        rng = np.random.default_rng(3)
        sample = rng.lognormal(0, 1, 257).tolist()
        for q in (0, 50, 99, 100):
            self.assertAlmostEqual(ops.percentile(sample, q), float(np.percentile(sample, q)))

    def test_percentile_rejects_bad_input(self):
        for values, q in (([], 50), ([1.0, -1.0], 50), ([1.0, float('nan')], 50), ([1.0], 101), ([1.0], -1)):
            with self.assertRaises(ValueError):
                ops.percentile(values, q)

    def test_psi_is_near_zero_for_same_distribution_and_large_for_shift(self):
        rng = np.random.default_rng(5)
        ref, same = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
        self.assertLess(ops.psi(ref, same), 0.05)
        self.assertGreater(ops.psi(ref, rng.normal(2, 1, 2000)), 0.25)
        self.assertEqual(ops.psi(ref, ref), 0.0)

    def test_psi_handles_constant_reference(self):
        self.assertEqual(ops.psi([5.0] * 50, [5.0] * 20), 0.0)
        self.assertGreater(ops.psi([5.0] * 50, [9.0] * 20), 0.25)

    def test_psi_and_ks_reject_bad_input(self):
        for fn in (ops.psi, ops.ks_statistic):
            with self.assertRaises(ValueError):
                fn([], [1.0])
            with self.assertRaises(ValueError):
                fn([1.0, float('nan')], [1.0])
        with self.assertRaises(ValueError):
            ops.psi([1.0, 2.0], [1.0], bins=1)

    def test_ks_matches_scipy(self):
        from scipy import stats
        rng = np.random.default_rng(9)
        a, b = rng.normal(0, 1, 300), rng.normal(0.4, 1.3, 200)
        self.assertAlmostEqual(ops.ks_statistic(a, b), float(stats.ks_2samp(a, b).statistic))
        self.assertEqual(ops.ks_statistic(a, a), 0.0)

    def test_ks_critical_shrinks_with_sample_size_and_validates(self):
        self.assertGreater(ops.ks_critical(50, 50, 0.01), ops.ks_critical(500, 500, 0.01))
        for args in ((0, 5, .01), (5, 5, 0.0), (5, 5, 1.0)):
            with self.assertRaises(ValueError):
                ops.ks_critical(*args)


class MonitorTests(unittest.TestCase):
    def test_same_distribution_raises_no_alert(self):
        report = ops.monitor(reference(), current())
        self.assertEqual(report['status'], 'ok', report['alerts'])
        self.assertEqual(report['alerts'], [])
        self.assertEqual(len(report['windows']), 3)
        self.assertTrue(all(w['status'] == 'evaluated' and w['count'] == 200 for w in report['windows']))

    def test_every_injected_fault_is_detected(self):
        for name, (params, (kind, metric)) in ops.SCENARIOS.items():
            with self.subTest(name):
                report = ops.monitor(reference(), current(**params))
                self.assertEqual(report['status'], 'alert')
                self.assertIn(metric, metrics(report, kind))

    def test_latency_spike_reports_p95_and_p99(self):
        report = ops.monitor(reference(), current(latency_median_ms=300.0))
        self.assertTrue({'p95_ms', 'p99_ms'} <= metrics(report, 'latency'))
        self.assertEqual(report['alert_kinds'], ['latency'])

    def test_citation_surge_does_not_trigger_latency_or_drift(self):
        report = ops.monitor(reference(), current(invalid_citation_rate=0.30))
        self.assertEqual(report['alert_kinds'], ['hallucination_proxy'])

    def test_abstention_directions(self):
        up = ops.monitor(reference(), current(abstain_rate=0.5))
        down = ops.monitor(reference(), current(abstain_rate=0.0))
        self.assertEqual({a['direction'] for a in up['alerts']}, {'up'})
        self.assertEqual({a['direction'] for a in down['alerts']}, {'down'})

    def test_input_drift_does_not_flag_unshifted_features(self):
        report = ops.monitor(reference(), current(input_tokens_mean=450.0))
        self.assertTrue({'psi:input_tokens', 'ks:input_tokens'} <= metrics(report, 'drift'))
        self.assertFalse(any(m.endswith('retrieval_score') or m.endswith('output_tokens')
                             for m in metrics(report, 'drift')))

    def test_alert_is_localised_to_the_degraded_window(self):
        normal = ops.synthetic_records(201, 200, start_ms=REF_N * 100.0)
        bad = ops.synthetic_records(202, 200, start_ms=REF_N * 100.0 + 20000.0, latency_median_ms=300.0)
        report = ops.monitor(reference(), normal + bad)
        self.assertEqual({a['window_index'] for a in report['alerts']}, {1})

    def test_small_windows_never_alert(self):
        records = ops.synthetic_records(5, 10, start_ms=REF_N * 100.0, latency_median_ms=900.0)
        report = ops.monitor(reference(), records)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['windows'][0]['status'], 'insufficient_data')

    def test_thresholds_come_from_config(self):
        mild = current(latency_median_ms=144.0)       # 1.2x the reference median
        self.assertNotIn('latency', ops.monitor(reference(), mild)['alert_kinds'])
        strict = ops.MonitorConfig(latency_ratio=1.05)
        self.assertIn('latency', ops.monitor(reference(), mild, strict)['alert_kinds'])

    def test_invalid_inputs_are_rejected(self):
        good = ops.synthetic_records(1, 5)
        with self.assertRaises(ValueError):
            ops.monitor(reference(), [])
        with self.assertRaises(ValueError):
            ops.monitor(good, current())                      # reference below min_reference
        for field, value in (('latency_ms', -1.0), ('latency_ms', float('nan')), ('input_tokens', None),
                             ('abstained', 1), ('retrieval_score', 'x')):
            bad = dict(good[0], **{field: value})
            with self.assertRaises(ValueError, msg=field):
                ops.validate_record(bad)
        with self.assertRaises(ValueError):
            ops.validate_record({k: v for k, v in good[0].items() if k != 'ts_ms'})
        with self.assertRaises(ValueError):
            ops.validate_record('record')

    def test_invalid_config_is_rejected(self):
        for kwargs in ({'window_ms': 0}, {'latency_ratio': 1.0}, {'rate_margin': 1.0}, {'ks_alpha': 0.0},
                       {'min_window_count': 0}, {'psi_bins': 1}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ops.MonitorConfig(**kwargs)

    def test_synthetic_records_validate_arguments(self):
        with self.assertRaises(ValueError):
            ops.synthetic_records(1, 0)
        with self.assertRaises(ValueError):
            ops.synthetic_records(1, 5, abstain_rate=1.5)
        self.assertEqual(ops.synthetic_records(3, 20), ops.synthetic_records(3, 20))


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'traces.jsonl'

    def test_records_survive_a_span_export_and_reload(self):
        records = ops.synthetic_records(4, 25)
        spans = ops.spans_from_records(records, run_id='t')
        self.assertEqual(ops.export_traces_jsonl(spans, self.path), 50)
        loaded = ops.load_traces_jsonl(self.path)
        self.assertEqual(loaded, spans)
        self.assertEqual(ops.records_from_spans(loaded), sorted(records, key=lambda r: r['ts_ms']))

    def test_request_trace_has_root_and_child_with_tokens(self):
        spans = ops.request_spans('k', 5.0, 12.5, {'input_tokens': 7, 'output_tokens': 3})
        root, child = spans
        self.assertIsNone(root['parent_span_id'])
        self.assertEqual(child['parent_span_id'], root['span_id'])
        self.assertEqual(child['attributes']['input_tokens'], 7)
        self.assertEqual(spans, ops.request_spans('k', 5.0, 12.5, {'input_tokens': 7, 'output_tokens': 3}))
        self.assertNotEqual(root['trace_id'], ops.request_spans('k2', 5.0, 12.5, {})[0]['trace_id'])

    def test_export_refuses_to_overwrite_and_empty_input(self):
        spans = ops.spans_from_records(ops.synthetic_records(4, 3))
        ops.export_traces_jsonl(spans, self.path)
        with self.assertRaises(FileExistsError):
            ops.export_traces_jsonl(spans, self.path)
        with self.assertRaises(ValueError):
            ops.export_traces_jsonl([], Path(self.tmp.name) / 'empty.jsonl')

    def test_export_rejects_invalid_spans_without_writing(self):
        spans = ops.spans_from_records(ops.synthetic_records(4, 2))
        spans[0] = dict(spans[0], duration_ms=-1.0)
        with self.assertRaises(ValueError):
            ops.export_traces_jsonl(spans, self.path)
        self.assertFalse(self.path.exists())

    def test_load_rejects_malformed_files(self):
        good = [json.dumps(s) for s in ops.spans_from_records(ops.synthetic_records(4, 2))]
        cases = {'bad_json': good + ['{not json'], 'empty': [], 'orphan': good[1:2],
                 'bad_status': [json.dumps(dict(json.loads(good[0]), status='weird'))]}
        for name, lines in cases.items():
            path = Path(self.tmp.name) / f'{name}.jsonl'
            path.write_text('\n'.join(lines) + '\n')
            with self.assertRaises(ValueError, msg=name):
                ops.load_traces_jsonl(path)
        with self.assertRaisesRegex(ValueError, 'line 5'):
            path = Path(self.tmp.name) / 'line.jsonl'
            path.write_text('\n'.join(good + ['{not json']) + '\n')
            ops.load_traces_jsonl(path)

    def test_duplicate_span_ids_are_rejected(self):
        spans = ops.spans_from_records(ops.synthetic_records(4, 1))
        with self.assertRaises(ValueError):
            ops.validate_spans(spans + [spans[0]])

    def test_records_from_spans_needs_request_spans(self):
        child_only = [ops.request_spans('k', 0.0, 1.0, {'input_tokens': 1, 'output_tokens': 1})[0]]
        child_only[0] = dict(child_only[0], name='other')
        with self.assertRaises(ValueError):
            ops.records_from_spans(child_only)

    def test_evaluation_spans_carry_scoring_attributes(self):
        ev = ops.evaluate_version(ops.make_cases(7), 'v2_json_abstain', ops.StubGenerator(), [1])
        roots = [s for s in ev['spans'] if s['name'] == ops.REQUEST_SPAN]
        self.assertEqual(len(roots), 24)
        self.assertEqual(len({s['trace_id'] for s in ev['spans']}), 24)
        self.assertTrue(all(s['attributes']['simulated'] for s in roots))
        self.assertEqual(len(ops.records_from_spans(ev['spans'])), 24)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_default_run_passes_every_check(self):
        code, out, _err = run_main(['--seed', '17'])
        self.assertEqual(code, 0, out)
        summary = json.loads(out)
        self.assertTrue(summary['all_passed'], summary['checks'])
        self.assertTrue(summary['generator']['simulated'])
        self.assertEqual(summary['monitoring_scenarios']['control_same_distribution']['status'], 'ok')
        self.assertIn('not live monitoring', summary['scope'])

    def test_run_is_recorded_and_refuses_reuse(self):
        out = self.root / 'run'
        code, _o, _e = run_main(['--out', str(out)])
        self.assertEqual(code, 0)
        for name in ('protocol.json', 'summary.json', 'REPORT.md', 'traces.jsonl', 'manifest.json'):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
        self.assertEqual(len(ops.load_traces_jsonl(out / 'traces.jsonl')), 3 * 72 * 2)
        code, _o, err = run_main(['--out', str(out)])
        self.assertEqual(code, 2)
        self.assertIn('error', err)

    def test_missing_model_directory_is_an_error(self):
        code, _o, err = run_main(['--model-dir', str(self.root / 'absent')])
        self.assertEqual(code, 2)
        self.assertIn('error', err)

    def write_stream(self, **params):
        records = reference() + current(**params)
        path = self.root / 'stream.jsonl'
        ops.export_traces_jsonl(ops.spans_from_records(records, run_id='replay'), path)
        return path

    def test_replay_of_a_healthy_stream_exits_zero(self):
        code, out, _e = run_main(['--replay', str(self.write_stream()), '--reference-count', str(REF_N)])
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)['status'], 'ok')

    def test_replay_of_a_degraded_stream_alerts(self):
        path = self.write_stream(latency_median_ms=300.0, input_tokens_mean=450.0)
        code, out, _e = run_main(['--replay', str(path), '--reference-count', str(REF_N)])
        self.assertEqual(code, 3)
        result = json.loads(out)
        self.assertEqual(result['alert_kinds'], ['drift', 'latency'])

    def test_replay_rejects_bad_arguments_and_files(self):
        path = self.write_stream()
        for argv in (['--replay', str(self.root / 'missing.jsonl')],
                     ['--replay', str(path), '--reference-count', '0'],
                     ['--replay', str(path), '--reference-count', '5000'],
                     ['--replay', str(path), '--reference-count', '600', '--window-ms', '0']):
            code, _o, err = run_main(argv)
            self.assertEqual(code, 2, argv)
            self.assertIn('error', err)
        broken = self.root / 'broken.jsonl'
        broken.write_text('{oops\n')
        self.assertEqual(run_main(['--replay', str(broken)])[0], 2)


if __name__ == '__main__':
    unittest.main()
