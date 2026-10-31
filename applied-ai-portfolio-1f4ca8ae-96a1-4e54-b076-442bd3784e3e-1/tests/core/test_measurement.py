import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from portfolio.measurement import RunRecord, distribution, environment, file_hash, peak_memory


class MeasurementTests(unittest.TestCase):
    def test_raw_samples_and_linear_percentiles(self):
        result = distribution(range(1, 101))
        self.assertEqual(result['p50_ms'], 50.5)
        self.assertAlmostEqual(result['p95_ms'], 95.05)
        self.assertEqual(result['raw_ms'], list(range(1, 101)))
        self.assertEqual(distribution([3])['p95_ms'], 3)

    def test_bad_samples_rejected(self):
        for data in ([], [-1], [float('nan')], [float('inf')]):
            with self.subTest(data=data), self.assertRaises(ValueError): distribution(data)

    def test_native_peak_memory_is_positive_or_explicitly_unavailable(self):
        record = peak_memory()
        if record['status'] == 'measured': self.assertGreater(record['bytes'], 0)
        else: self.assertIsNone(record['bytes']); self.assertIn('reason', record)

    @unittest.skipIf(__import__('sys').platform == 'win32', 'getrusage unit conversion is a Unix contract')
    def test_linux_and_macos_peak_units(self):
        with patch('resource.getrusage', return_value=SimpleNamespace(ru_maxrss=42)):
            with patch('portfolio.measurement.platform.system', return_value='Linux'):
                self.assertEqual(peak_memory()['bytes'], 42 * 1024)
            with patch('portfolio.measurement.platform.system', return_value='Darwin'):
                self.assertEqual(peak_memory()['bytes'], 42)

    def test_unsupported_memory_and_power_are_not_zero_estimates(self):
        with patch('portfolio.measurement.platform.system', return_value='unsupported'):
            self.assertIsNone(peak_memory()['bytes'])
        record = environment()
        self.assertIsNone(record['power']['watts'])
        self.assertIsNone(record['power']['joules'])
        self.assertFalse(record['target_edge_device_verified'])
        self.assertNotIn('hostname', record)


class RunRecordTests(unittest.TestCase):
    def test_success_manifest_identifies_actual_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            with RunRecord(out, {'case': 'fixture'}, [Path(__file__)]):
                (out / 'summary.json').write_text('{}')
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['summary.json'], file_hash(out / 'summary.json'))
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')

    def test_failure_keeps_partial_evidence_and_removes_success_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            with self.assertRaises(RuntimeError):
                with RunRecord(out, {}, [Path(__file__)]):
                    (out / 'worker.txt').write_text('partial output')
                    (out / 'summary.json').write_text('{}')
                    (out / 'REPORT.md').write_text('partial report')
                    raise RuntimeError('injected')
            self.assertFalse((out / 'summary.json').exists())
            self.assertFalse((out / 'REPORT.md').exists())
            self.assertEqual((out / 'worker.txt').read_text(), 'partial output')
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'failed')

    def test_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'status.json'; marker.write_text('old evidence')
            with self.assertRaises(FileExistsError):
                with RunRecord(temp, {}, []): pass
            self.assertEqual(marker.read_text(), 'old evidence')
