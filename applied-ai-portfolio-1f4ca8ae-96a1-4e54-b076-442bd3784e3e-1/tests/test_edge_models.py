"""Real PyTorch checks. Missing ML dependencies are errors, not simulated passes."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from portfolio.edge_benchmark import load_variant, run
from portfolio.edge_protocol import Config, VARIANTS
from portfolio.edge_vision import digits_data, fit, measure, model, predict, quantize


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.x, cls.y, cls.splits, cls.digest = digits_data()

    def test_original_stratified_splits_are_disjoint_and_normalized(self):
        sets = [set(value) for value in self.splits.values()]
        self.assertEqual([len(v) for v in sets], [1149, 288, 360])
        self.assertEqual(len(set.union(*sets)), 1797)
        self.assertTrue(all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
        self.assertGreaterEqual(float(self.x.min()), 0)
        self.assertLessEqual(float(self.x.max()), 1)
        for indices in self.splits.values(): self.assertEqual(set(self.y[indices].tolist()), set(range(10)))

    def test_distillation_updates_student_preserves_teacher_and_restores_best_validation(self):
        torch.manual_seed(7); teacher = model(128).eval(); student = model(24)
        teacher_before = copy.deepcopy(teacher.state_dict()); before = copy.deepcopy(student.state_dict())
        train, valid = self.splits['train'], self.splits['validation']
        history = fit(student, self.x[train], self.y[train], self.x[valid], self.y[valid], teacher=teacher, epochs=3)
        self.assertTrue(any(not torch.equal(value, student.state_dict()[key]) for key, value in before.items()))
        for key, value in teacher_before.items(): torch.testing.assert_close(teacher.state_dict()[key], value)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        with torch.no_grad(): actual = float(torch.nn.functional.cross_entropy(student(self.x[valid]), self.y[valid]))
        self.assertAlmostEqual(actual, min(row['validation_loss'] for row in history), places=6)

    def test_actual_quantized_layers_and_raw_measurements(self):
        net = quantize(model(24).eval())
        layers = [m for m in net.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
        self.assertEqual(len(layers), 2)
        self.assertTrue(all(m.weight().dtype == torch.qint8 for m in layers))
        self.assertEqual(len(predict(net, self.x[:32])), 32)
        metrics = measure(net, self.x[:1], repetitions=7)
        self.assertEqual(len(metrics['raw_ms']), 7)
        self.assertEqual(metrics['warmup'], 10)
        self.assertAlmostEqual(metrics['p95_ms'], np.quantile(metrics['raw_ms'], .95))
        with self.assertRaises(ValueError): measure(net, self.x[:1], repetitions=0)
        with self.assertRaises(ValueError): fit(net, self.x[:2], self.y[:2], self.x[2:3], self.y[2:3], epochs=0)

    def test_isolated_pipeline_and_artifact_corruption_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'vision'
            result = run(out, Config(seeds=(7,), epochs=2, rounds=4, repetitions=5))
            self.assertEqual(len(result['rows']), 16)
            self.assertEqual({row['variant'] for row in result['rows']}, set(VARIANTS))
            self.assertEqual(len(result['test_rows']), 4)
            self.assertEqual(len({r['selection_sha256'] for r in result['test_rows']}), 1)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
            self.assertTrue(all(len(r['latency_batch32']['raw_ms']) == 5 for r in result['rows']))
            # State restoration works for real packed INT8 tensors; tampering is rejected before loading.
            loaded, _, _, _ = load_variant(out, 7, 'student_dynamic_int8')
            self.assertEqual(len(predict(loaded, self.x[:1])), 1)
            (out / 'models/7_student_dynamic_int8.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): load_variant(out, 7, 'student_dynamic_int8')
