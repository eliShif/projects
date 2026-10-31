import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from portfolio.edge_benchmark import launch_worker, run
from portfolio.edge_protocol import Config


class WorkerFailureTests(unittest.TestCase):
    def test_child_failure_keeps_command_log_and_no_success_report(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            with patch('portfolio.edge_benchmark.subprocess.run', return_value=subprocess.CompletedProcess(['fixture'], 1)):
                with self.assertRaises(subprocess.CalledProcessError): run(out, Config(seeds=(7,), epochs=1))
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'failed')
            self.assertEqual(json.loads((out / 'logs/train_7_0_all.exit.json').read_text())['returncode'], 1)
            self.assertFalse((out / 'selection.json').exists())
            self.assertFalse((out / 'summary.json').exists())

    def test_timeout_does_not_publish_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            with patch('portfolio.edge_benchmark.subprocess.run', side_effect=subprocess.TimeoutExpired('fixture', 600)):
                with self.assertRaises(subprocess.TimeoutExpired): run(out, Config(seeds=(7,), epochs=1))
            self.assertEqual(json.loads((out / 'status.json').read_text())['error_type'], 'TimeoutExpired')
            self.assertFalse((out / 'summary.json').exists())

    def test_worker_command_uses_fresh_interpreter_and_bounded_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('portfolio.edge_benchmark.subprocess.run', return_value=subprocess.CompletedProcess(['fixture'], 0)) as call:
                launch_worker(Path(temp), Config(threads=2), 'measure', 7, 'teacher_fp32', 1, 2)
            args, kwargs = call.call_args
            self.assertEqual(args[0][1:3], ['-m', 'portfolio.edge_benchmark'])
            self.assertEqual(kwargs['timeout'], 600)
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
            self.assertNotIn('PORTFOLIO_API_TOKEN', kwargs['env'])
