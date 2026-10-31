from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portfolio.reliable_service import Conflict, JobStore, LeaseLost
from portfolio.service_load import ReceiptEngine, fault_probe, run_store


class QueueContracts(unittest.TestCase):
    def test_expired_completion_and_failure_at_exact_boundary_then_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            result = fault_probe(temp)
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual([r['event'] for r in result['retry_job']['trace']],
                         ['enqueued', 'claimed', 'queued', 'claimed', 'completed'])

    def test_old_failure_cannot_requeue_replaced_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            now = [0.0]; store = JobStore(Path(temp) / 'db', clock=lambda: now[0])
            job_id, _ = store.enqueue('key', {'mode': 'research', 'question': 'fixture'})
            old = store.claim(lease_seconds=1); now[0] = 2; current = store.claim()
            with self.assertRaises(LeaseLost): store.fail(old, 'late failure')
            store.complete(current, {'ok': True})
            self.assertEqual(store.get(job_id)['result'], {'ok': True})

    def test_nonfinite_result_is_not_partially_committed(self):
        with tempfile.TemporaryDirectory() as temp:
            store = JobStore(Path(temp) / 'db')
            job_id, _ = store.enqueue('key', {'mode': 'research', 'question': 'fixture'})
            job = store.claim()
            with self.assertRaises(ValueError): store.complete(job, {'score': float('nan')})
            self.assertEqual(store.get(job_id)['status'], 'running')
            self.assertEqual([r['event'] for r in store.get(job_id)['trace']], ['enqueued', 'claimed'])
            store.complete(job, {'score': 1})

    def test_concurrent_content_conflict_has_one_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            store = JobStore(Path(temp) / 'db')
            def enqueue(value):
                try: return store.enqueue('same-key', {'mode': 'research', 'question': value})[1]
                except Conflict: return False
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(enqueue, [f'payload-{i}' for i in range(12)]))
            self.assertEqual(sum(results), 1)
            self.assertEqual(store.metrics()['jobs_by_status'], {'queued': 1})

    def test_real_store_load_artifacts_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'load'
            result = run_store(out, jobs=6, clients=3, copies=3)
            self.assertEqual(result['enqueue']['operations'], 18)
            self.assertEqual(result['expected_conflicts'], 6)
            self.assertEqual(result['metrics']['total_attempts'], 6)
            self.assertTrue(all(result['checks'].values()))
            self.assertEqual(len(json.loads((out / 'operations.json').read_text())), 18)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
            self.assertNotIn('http', result['enqueue'])

    def test_processing_failure_does_not_publish_success(self):
        # A worker that fails keeps the run in a failed state; no synthetic success fallback.
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'load'
            with patch.object(ReceiptEngine, 'answer', side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError): run_store(out, jobs=2, clients=2, copies=2)
            self.assertFalse((out / 'summary.json').exists())
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'failed')

    def test_wrong_completed_receipt_does_not_publish_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'load'
            with patch.object(ReceiptEngine, 'answer', return_value={'receipt': 'wrong', 'workload': 'queue-probe-no-model'}):
                with self.assertRaises(RuntimeError): run_store(out, jobs=2, clients=2, copies=2)
            self.assertFalse(json.loads((out / 'checks.json').read_text())['content_preserved_after_conflicts'])
            self.assertFalse((out / 'summary.json').exists())
            self.assertTrue((out / 'operations.json').exists())

    def test_load_bounds_rejected(self):
        for change in ({'jobs': 0}, {'clients': 17}, {'copies': 1}):
            with self.assertRaises(ValueError): run_store('unused-out', **change)
