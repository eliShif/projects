"""Resume an interrupted frozen evaluation without discarding measured outputs.

The original source, protocol, prompts, task order and per-call seeds stay fixed.
Only the scheduling loop is adapted to skip already recorded (task, repeat,
policy) keys. The generated recovery source and its hashes are saved separately.
Run from the repository root; the frozen directory provides the measured code.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from datetime import datetime, timezone


def resume(model, out, threads=4):
    out = Path(out).resolve()
    frozen = out / 'frozen_source'
    protocol = json.loads((out / 'protocol.json').read_text())
    rows = json.loads((out / 'runs.json').read_text())
    if not rows:
        raise ValueError('Use the ordinary evaluator for a new run')
    for name, expected in protocol['source_hashes'].items():
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source mismatch: {name}')
    package = types.ModuleType('_portfolio_frozen_recovery')
    package.__path__ = [str(frozen), str(Path(__file__).resolve().parents[1] / 'portfolio')]
    sys.modules[package.__name__] = package
    evaluator = importlib.import_module(package.__name__ + '.evidence_eval')
    records, tasks = evaluator.build_cases(protocol['split'])
    corpus = [r.as_dict() for r in records]
    if evaluator.sha256_json(tasks) != protocol['dataset_sha256']:
        raise ValueError('Task hash mismatch')
    if evaluator.sha256_json(corpus) != protocol['corpus_sha256']:
        raise ValueError('Corpus hash mismatch')
    if json.loads((out/'cases.json').read_text()) != tasks:
        raise ValueError('Saved cases changed')
    if json.loads((out/'corpus.json').read_text()) != corpus:
        raise ValueError('Saved corpus changed')
    task_by_id = {t['id']: t for t in tasks}
    expected_keys = []
    for ti, task in enumerate(tasks):
        for repeat in range(protocol['repeats']):
            policies = ['rag', 'typed_agent'] if (ti+repeat)%2 == 0 else ['typed_agent', 'rag']
            expected_keys.extend((task['id'], repeat, p) for p in policies)
    keys = [(r['task_id'], r['repeat'], r['policy']) for r in rows]
    if keys != expected_keys[:len(rows)] or len(keys) != len(set(keys)):
        raise ValueError('Checkpoint is not a unique prefix of the frozen schedule')
    hashes = {r['result']['model_response']['model_sha256'] for r in rows
              if 'model_response' in r['result']}
    with Path(model).open('rb') as handle:
        model_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if hashes != {model_hash}:
        raise ValueError('Model weights differ from checkpoint')
    for row in rows:
        scored = evaluator.score(row['result'], task_by_id[row['task_id']])
        if any(row[k] != v for k, v in scored.items()):
            raise ValueError('Checkpoint scores differ from frozen scorer')
    source = (frozen/'evidence_eval.py').read_text()
    changes = {
        "    write_json(out/'protocol.json',protocol)":
            "    protocol=json.loads((out/'protocol.json').read_text())",
        '    rows=[]': "    rows=json.loads((out/'runs.json').read_text())\n"
                        "    completed={(r['task_id'],r['repeat'],r['policy']) for r in rows}",
        '            for policy in policies:\n':
            "            for policy in policies:\n"
            "                if (task['id'],repeat,policy) in completed: continue\n",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError('Unsupported frozen runner; refusing an ambiguous patch')
        source = source.replace(old, new, 1)
    recovery = out / 'recovery'
    recovery.mkdir(exist_ok=True)
    (recovery/'checkpoint.json').write_text(json.dumps(rows, indent=2)+'\n')
    (recovery/'resumed_runner.py').write_text(source)
    metadata = {'resumed_utc': datetime.now(timezone.utc).isoformat(),
                'preserved_rows': len(rows), 'expected_rows': len(expected_keys),
                'model_sha256': model_hash,
                'recovery_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'scope': 'Scheduling recovery only; frozen prompts, tasks, scoring and seeds unchanged'}
    evaluator.write_json(recovery/'metadata.json', metadata)
    exec(compile(source, str(frozen/'evidence_eval.py'), 'exec'), evaluator.__dict__)
    evaluator.run(str(model), str(out), protocol['split'], protocol['repeats'], threads=threads)
    completed_rows = json.loads((out/'runs.json').read_text())
    if completed_rows[:len(rows)] != rows or len(completed_rows) != len(expected_keys):
        raise RuntimeError('Final checkpoint validation failed')
    print('Recovery complete; all saved rows preserved exactly.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    resume(args.model, args.out, args.threads)
