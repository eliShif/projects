import json
from pathlib import Path
import tempfile
import unittest

from portfolio.service_load import run


class HttpLoadTests(unittest.TestCase):
    def test_concurrent_http_retries_conflicts_receipts_and_faults(self):
        # Socket restrictions are reported as errors; this is not a mocked HTTP success.
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'http'
            result = run(out, jobs=6, clients=3, copies=3)
            self.assertEqual(result['submissions']['http_requests'], 18)
            self.assertEqual(result['status_counts']['202'], 6)
            self.assertEqual(result['status_counts']['409'], 6)
            self.assertEqual(result['status_counts']['401'], 1)
            self.assertEqual(result['status_counts']['400'], 1)
            self.assertEqual(result['unexpected_http_outcomes'], 0)
            self.assertTrue(all(result['checks'].values()))
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
