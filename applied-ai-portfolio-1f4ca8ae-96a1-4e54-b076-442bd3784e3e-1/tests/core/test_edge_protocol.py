from dataclasses import replace
import unittest

from portfolio.edge_protocol import Config, VARIANTS, choose_variant, schedule


def fixture_rows(config):
    """Synthetic contracts only; these numbers are never experiment evidence."""
    sizes = dict(zip(VARIANTS, [40000, 9000, 9000, 5000]))
    return [{**item, 'artifact_sha256': f"fixture-{item['seed']}-{item['variant']}",
             'validation_accuracy': .98, 'serialized_state_bytes': sizes[item['variant']],
             'latency_batch1': {'p50_ms': .05, 'p95_ms': .1},
             'latency_batch32': {'p50_ms': .1, 'p95_ms': .2},
             'peak_memory': {'status': 'measured', 'bytes': 300 * 2**20}} for item in schedule(config)]


class ProtocolTests(unittest.TestCase):
    def setUp(self): self.config = Config(seeds=(7, 19))

    def test_balanced_positions_within_each_training_seed(self):
        rows = schedule(self.config)
        for seed in self.config.seeds:
            for variant in VARIANTS:
                self.assertEqual(sorted(r['position'] for r in rows if r['seed'] == seed and r['variant'] == variant), [0, 1, 2, 3])

    def test_invalid_configuration_rejected_before_execution(self):
        for change in ({'rounds': 3}, {'epochs': 0}, {'threads': 0}, {'seeds': (7, 7)},
                       {'seeds': ()}, {'max_p95_ms': float('nan')}, {'min_accuracy': 1.1},
                       {'max_peak_mib': -1}, {'power_condition': 'estimated'}, {'device_label': 'a\nb'}):
            with self.subTest(change=change), self.assertRaises(ValueError): replace(self.config, **change).validate()

    def test_smallest_feasible_model_selected_without_test_access(self):
        rows = fixture_rows(self.config)
        result = choose_variant(rows, self.config)
        self.assertEqual(result['selected_variant'], 'student_dynamic_int8')
        for row in rows: row['test_accuracy'] = 0 if row['variant'] == 'student_dynamic_int8' else 1
        self.assertEqual(choose_variant(rows, self.config), result)

    def test_slow_quantization_is_rejected_despite_small_storage(self):
        rows = fixture_rows(self.config)
        for row in rows:
            if row['variant'] == 'student_dynamic_int8': row['latency_batch1']['p95_ms'] = 2
        result = choose_variant(rows, self.config)
        self.assertNotEqual(result['selected_variant'], 'student_dynamic_int8')
        quantized = next(c for c in result['candidates'] if c['variant'] == 'student_dynamic_int8')
        self.assertIn('batch-1 p95 exceeds budget in at least one trial', quantized['reasons'])

    def test_bad_seed_not_hidden_by_mean_quality(self):
        rows = fixture_rows(self.config)
        for row in rows:
            if row['variant'] == 'student_dynamic_int8' and row['seed'] == 19: row['validation_accuracy'] = .94
        self.assertNotEqual(choose_variant(rows, self.config)['selected_variant'], 'student_dynamic_int8')

    def test_exact_budget_boundaries_are_inclusive(self):
        rows = fixture_rows(self.config)
        for row in rows:
            if row['variant'] == 'student_dynamic_int8':
                row['validation_accuracy'] = .96
                row['latency_batch1']['p95_ms'] = 1
        config = replace(self.config, min_accuracy=.96, max_state_kib=5000/1024, max_peak_mib=300)
        self.assertEqual(choose_variant(rows, config)['selected_variant'], 'student_dynamic_int8')

    def test_no_candidate_when_memory_unavailable_or_budgets_fail(self):
        rows = fixture_rows(self.config)
        for row in rows: row['peak_memory']['bytes'] = None
        result = choose_variant(rows, self.config)
        self.assertIsNone(result['selected_variant'])
        self.assertTrue(all(c['reasons'] for c in result['candidates']))
        self.assertIsNone(choose_variant(fixture_rows(self.config), replace(self.config, max_state_kib=1))['selected_variant'])

    def test_missing_duplicate_or_mutated_trials_fail(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
                       lambda rows: rows[0].update(artifact_sha256='changed'),
                       lambda rows: rows[0].update(validation_accuracy=float('nan'))):
            rows = fixture_rows(self.config); mutate(rows)
            with self.assertRaises(ValueError): choose_variant(rows, self.config)
