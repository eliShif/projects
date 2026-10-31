"""Finite loopback HTTP/SQLite load and fault probes for the existing service."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.request

from .common import write_json
from .measurement import RunRecord, distribution, environment
from .reliable_service import Conflict, JobStore, LeaseLost, make_server, work_once


class ReceiptEngine:
    """Deterministic queue probe. It does not simulate model inference or quality."""
    def answer(self, payload):
        return {'workload': 'queue-probe-no-model',
                'receipt': hashlib.sha256(payload['question'].encode()).hexdigest()}


def fault_probe(directory):
    now = [100.0]
    store = JobStore(Path(directory) / 'faults.db', clock=lambda: now[0])
    job_id, _ = store.enqueue('lease-probe', {'mode': 'research', 'question': 'Lease boundary probe'})
    old = store.claim(lease_seconds=2)
    now[0] += 2  # Exactly the boundary, without sleeping or relying on scheduler timing.
    checks = {}
    for name, operation in [('expired_complete_rejected', lambda: store.complete(old, {'stale': True})),
                            ('expired_fail_rejected', lambda: store.fail(old, 'stale'))]:
        try:
            operation(); checks[name] = False
        except LeaseLost:
            checks[name] = True
    replacement = store.claim(lease_seconds=20)
    try:
        store.complete(old, {'stale': True}); checks['replaced_owner_rejected'] = False
    except LeaseLost:
        checks['replaced_owner_rejected'] = True
    store.complete(replacement, {'recovered': True})
    checks['recovered_once'] = store.get(job_id)['result'] == {'recovered': True} and store.claim() is None
    retry_id, _ = store.enqueue('retry-probe', {'mode': 'research', 'question': 'Retry probe'})

    class FailedEngine:
        def answer(self, payload):
            raise RuntimeError('injected processing failure')

    try:
        work_once(store, FailedEngine())
        checks['processing_failure_propagated'] = False
    except RuntimeError:
        checks['processing_failure_propagated'] = True
    checks['failure_requeued'] = store.get(retry_id)['status'] == 'queued'
    work_once(store, ReceiptEngine())
    recovered = store.get(retry_id)
    checks['retry_completed'] = recovered['status'] == 'succeeded' and recovered['attempts'] == 2
    return {'checks': checks, 'lease_job': store.get(job_id), 'retry_job': recovered,
            'clock': 'deterministic boundary clock; not a wall-clock expiry benchmark'}


def request(base, path, token, *, payload=None, key='', expected=(200,)):
    headers = {'Content-Type': 'application/json', 'Idempotency-Key': key,
               'Authorization': 'Bearer ' + token}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers)
    start = time.perf_counter_ns()
    status = None; body = {}; error = None
    try:
        # Disable proxy lookup even if the host has proxy environment variables.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=5) as response:
            status = response.status; body = json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            status = exc.code; body = json.load(exc)
    except (OSError, ValueError) as exc:
        error = type(exc).__name__
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return {'path': path, 'key': key, 'status': status, 'body': body, 'latency_ms': elapsed,
            'expected_statuses': list(expected), 'ok': status in expected, 'error_type': error}


def run(out, jobs=24, clients=4, copies=3):
    if not 1 <= jobs <= 200 or not 1 <= clients <= 16 or not 2 <= copies <= 5:
        raise ValueError('Bounds: jobs 1..200, clients 1..16, copies 2..5')
    source_dir = Path(__file__).parent
    sources = [source_dir / name for name in ('service_load.py', 'reliable_service.py', 'measurement.py', 'common.py')]
    out = Path(out)
    protocol = {'experiment': 'local-service-load-v1', 'jobs': jobs, 'clients': clients,
                'copies_per_key': copies, 'request_timeout_seconds': 5,
                'environment': environment(), 'workload': 'queue-probe-no-model',
                'arrival_model': 'bounded closed-loop thread clients; no offered-rate or saturation claim'}
    with RunRecord(out, protocol, sources), tempfile.TemporaryDirectory(prefix='queue-', dir=out) as temp:
        store = JobStore(Path(temp) / 'jobs.db')
        token = 'local-load-fixture-not-a-real-secret'
        server = make_server(store, token, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = 'http://127.0.0.1:' + str(server.server_address[1])
        records = []
        try:
            def submit(index):
                return request(base, '/jobs', token, key=f'load-{index}', expected=(200, 202),
                               payload={'mode': 'research', 'question': f'Local queue probe {index}'})
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=clients) as pool:
                submissions = list(pool.map(submit, [i for i in range(jobs) for _ in range(copies)]))
            submission_seconds = time.perf_counter() - started
            records.extend(submissions)
            created = sum(r['status'] == 202 for r in submissions)
            duplicates = sum(r['status'] == 200 for r in submissions)
            ids = {r['body'].get('id') for r in submissions if r['ok']}
            same_keys = all(len({r['body'].get('id') for r in submissions if r['key'] == f'load-{i}'}) == 1 for i in range(jobs))
            # Exercise conflicts over HTTP while preserving the already-enqueued content.
            conflicts = [request(base, '/jobs', token, key=f'load-{i}', expected=(409,),
                                payload={'mode': 'research', 'question': 'Conflicting content'}) for i in range(jobs)]
            records.extend(conflicts)
            records.append(request(base, '/jobs', 'invalid-fixture-token', key='unauthorized',
                                   expected=(401,), payload={'mode': 'research', 'question': 'Rejected probe'}))
            records.append(request(base, '/jobs', token, key='invalid', expected=(400,),
                                   payload={'mode': 'unsupported', 'question': 'Rejected probe'}))
            started = time.perf_counter()

            def drain(_):
                count = 0
                while work_once(store, ReceiptEngine()):
                    count += 1
                return count

            with ThreadPoolExecutor(max_workers=clients) as pool:
                completed = sum(pool.map(drain, range(clients)))
            drain_seconds = time.perf_counter() - started
            reads = [request(base, '/jobs/' + job_id, token) for job_id in sorted(ids) if job_id]
            records.extend(reads)
            checks = {'one_create_per_key': created == jobs, 'duplicate_count': duplicates == jobs * (copies - 1),
                      'one_id_per_key': same_keys and len(ids) == jobs,
                      'all_http_outcomes_expected': all(r['ok'] for r in records),
                      'each_job_completed_once': completed == jobs and all(
                          r['body'].get('status') == 'succeeded' and r['body'].get('attempts') == 1 and
                          [e['event'] for e in r['body'].get('trace', [])] == ['enqueued', 'claimed', 'completed'] for r in reads),
                      'receipt_engine_used': all(r['body'].get('result', {}).get('workload') == 'queue-probe-no-model' for r in reads)}
            expected_receipts = {r['body'].get('id'): hashlib.sha256(
                f"Local queue probe {r['key'].split('-')[-1]}".encode()).hexdigest() for r in submissions if r['ok']}
            checks['content_preserved_after_conflicts'] = all(
                r['body'].get('result', {}).get('receipt') == expected_receipts.get(r['body'].get('id')) for r in reads)
            faults = fault_probe(temp)
            checks.update(faults['checks'])
            write_json(out / 'requests.json', records)
            write_json(out / 'faults.json', faults)
            write_json(out / 'checks.json', checks)
            if not all(checks.values()):
                raise RuntimeError('Local load invariants failed; inspect checks.json and requests.json')
            summary = {**protocol, 'checks': checks, 'status_counts': {str(code): sum(r['status'] == code for r in records)
                       for code in (200, 202, 400, 401, 409)},
                       'unexpected_http_outcomes': sum(not r['ok'] for r in records),
                       'submissions': {'http_requests': len(submissions), 'unique_jobs': created,
                                       'seconds': submission_seconds, 'requests_per_second': len(submissions) / submission_seconds,
                                       'unique_jobs_per_second': created / submission_seconds,
                                       **distribution(r['latency_ms'] for r in submissions)},
                       'drain': {'completed_jobs': completed, 'seconds': drain_seconds,
                                 'jobs_per_second': completed / drain_seconds}, 'metrics': store.metrics(),
                       'limitations': ['Loopback HTTP and SQLite share one host; workload uses a deterministic receipt engine, not AI inference.',
                                       'Closed-loop client timings can omit queueing under an external offered load (coordinated omission).',
                                       'Submission and drain are separate phases; drain throughput excludes enqueue and HTTP result fetching.',
                                       'Short bounded load is not saturation, availability, production scalability or customer impact evidence.',
                                       'Lease faults use a controlled clock; no real process-crash or power-loss durability experiment.']}
            write_json(out / 'summary.json', summary)
            s = summary['submissions']
            lines = ['# Local service load', '',
                     f"{jobs} unique jobs; {clients} client threads; {copies} submissions per key.",
                     'Workload: deterministic receipt processing, with no model inference.', '',
                     f"Submission p50: {s['p50_ms']:.3f} ms; p95: {s['p95_ms']:.3f} ms; {s['requests_per_second']:.2f} HTTP requests/s.",
                     f"Unique enqueue rate: {s['unique_jobs_per_second']:.2f} jobs/s. Separate drain: {summary['drain']['jobs_per_second']:.2f} jobs/s.",
                     f"Unexpected HTTP outcomes: {summary['unexpected_http_outcomes']}. Expected conflicts: {len(conflicts)} (409).",
                     '', 'Concurrent duplicate keys, content conflicts, expired/replaced leases and processing retry checks passed.',
                     '', 'Recommendation: retain transactional idempotency and lease fencing. Test an actual model workload on the target host before setting capacity or availability targets.',
                     '', '## Limits', ''] + ['- ' + s for s in summary['limitations']]
            (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
    return summary


def run_store(out, jobs=24, clients=4, copies=3):
    """Measure the real SQLite queue directly when sockets are unavailable."""
    if not 1 <= jobs <= 200 or not 1 <= clients <= 16 or not 2 <= copies <= 5:
        raise ValueError('Bounds: jobs 1..200, clients 1..16, copies 2..5')
    source_dir = Path(__file__).parent
    sources = [source_dir / name for name in ('service_load.py', 'reliable_service.py', 'measurement.py', 'common.py')]
    out = Path(out)
    protocol = {'experiment': 'local-store-load-v1', 'jobs': jobs, 'clients': clients,
                'copies_per_key': copies, 'environment': environment(),
                'transport': 'direct SQLite; HTTP not exercised', 'workload': 'queue-probe-no-model'}
    with RunRecord(out, protocol, sources), tempfile.TemporaryDirectory(prefix='queue-', dir=out) as temp:
        store = JobStore(Path(temp) / 'jobs.db')

        def enqueue(index):
            start = time.perf_counter_ns()
            job_id, created = store.enqueue(f'load-{index}', {'mode': 'research', 'question': f'Local queue probe {index}'})
            return {'key': f'load-{index}', 'id': job_id, 'created': created,
                    'latency_ms': (time.perf_counter_ns() - start) / 1e6}

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            rows = list(pool.map(enqueue, [i for i in range(jobs) for _ in range(copies)]))
        elapsed = time.perf_counter() - started
        conflicts = 0
        for i in range(jobs):
            try: store.enqueue(f'load-{i}', {'mode': 'research', 'question': 'Conflicting content'})
            except Conflict: conflicts += 1
        started = time.perf_counter()

        def drain(_):
            count = 0
            while work_once(store, ReceiptEngine()): count += 1
            return count

        with ThreadPoolExecutor(max_workers=clients) as pool:
            completed = sum(pool.map(drain, range(clients)))
        drain_seconds = time.perf_counter() - started
        receipts = [store.get(job_id) for job_id in sorted({r['id'] for r in rows})]
        checks = {'one_create_per_key': sum(r['created'] for r in rows) == jobs,
                  'one_id_per_key': len(receipts) == jobs and all(
                      len({r['id'] for r in rows if r['key'] == f'load-{i}'}) == 1 for i in range(jobs)),
                  'conflicts_rejected': conflicts == jobs,
                  'each_job_completed_once': completed == jobs and all(
                      r['status'] == 'succeeded' and r['attempts'] == 1 and
                      [e['event'] for e in r['trace']] == ['enqueued', 'claimed', 'completed'] for r in receipts)}
        expected_receipts = {r['id']: hashlib.sha256(
            f"Local queue probe {r['key'].split('-')[-1]}".encode()).hexdigest() for r in rows}
        checks['content_preserved_after_conflicts'] = all(
            r['result'].get('receipt') == expected_receipts[r['id']] and r['result'].get('workload') == 'queue-probe-no-model'
            for r in receipts)
        faults = fault_probe(temp); checks.update(faults['checks'])
        write_json(out / 'operations.json', rows); write_json(out / 'receipts.json', receipts)
        write_json(out / 'faults.json', faults); write_json(out / 'checks.json', checks)
        if not all(checks.values()):
            raise RuntimeError('Store load invariants failed; inspect checks.json')
        summary = {**protocol, 'checks': checks, 'expected_conflicts': conflicts, 'unexpected_errors': 0,
                   'enqueue': {'operations': len(rows), 'seconds': elapsed,
                               'operations_per_second': len(rows) / elapsed, 'unique_jobs_per_second': jobs / elapsed,
                               **distribution(r['latency_ms'] for r in rows)},
                   'drain': {'completed_jobs': completed, 'seconds': drain_seconds, 'jobs_per_second': completed / drain_seconds},
                   'metrics': store.metrics(),
                   'limitations': ['Direct SQLite calls only; sockets and HTTP are not exercised.',
                                   'Receipt processing is a deterministic queue probe, not model inference or quality evaluation.',
                                   'Closed-loop threads and separate enqueue/drain phases do not measure saturation or end-to-end client latency.',
                                   'Short local measurements do not establish production availability, scalability or customer impact.',
                                   'Lease faults use a controlled clock, not process crashes or physical power loss.']}
        write_json(out / 'summary.json', summary)
        stats = summary['enqueue']
        lines = ['# Local SQLite load', '', f'{jobs} unique jobs, {clients} threads, {copies} submissions per key.',
                 'Transport: direct SQLite. HTTP was not exercised by this run.', '',
                 f"Enqueue p50: {stats['p50_ms']:.3f} ms; p95: {stats['p95_ms']:.3f} ms; {stats['operations_per_second']:.2f} operations/s.",
                 f"Unique enqueue rate: {stats['unique_jobs_per_second']:.2f} jobs/s. Separate receipt-worker drain: {summary['drain']['jobs_per_second']:.2f} jobs/s.",
                 f'Expected content conflicts: {conflicts}. Unexpected errors: 0. All recorded invariants passed.', '',
                 'Recommendation: retain transactional idempotency and fencing. Verify HTTP and actual inference under representative arrival rates before setting capacity targets.',
                 '', '## Limits', ''] + ['- ' + value for value in summary['limitations']]
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/service_load')
    parser.add_argument('--jobs', type=int, default=24)
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--copies', type=int, default=3)
    parser.add_argument('--transport', choices=['http', 'store'], default='http')
    args = parser.parse_args()
    try:
        runner = run if args.transport == 'http' else run_store
        summary = runner(args.out, args.jobs, args.clients, args.copies)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({'status': 'complete', 'transport': args.transport,
                      'unexpected_errors': summary.get('unexpected_http_outcomes', summary.get('unexpected_errors'))}))


if __name__ == '__main__':
    main()
