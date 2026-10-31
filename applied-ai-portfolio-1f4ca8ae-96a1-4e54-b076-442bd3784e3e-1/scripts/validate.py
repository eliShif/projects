"""Record real test/build/smoke commands and return nonzero if any check fails."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['core', 'all'], default='all')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): parser.error('Use a fresh output directory')
    source_paths = sorted(p for root in ('portfolio', 'tests', 'scripts') for p in Path(root).rglob('*.py'))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if args.scope == 'all':
        tests = [('tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])]
    else:
        tests = [('core-tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/core', '-v']),
                 ('original-store-tests', [sys.executable, '-m', 'unittest', 'tests.test_service.StoreTests', '-v'])]
    commands = tests + [('build', [sys.executable, '-m', 'compileall', '-q', 'portfolio', 'tests', 'scripts']),
                        ('smoke', [sys.executable, '-m', 'portfolio.smoke'])]
    rows = []
    for name, command in commands:
        started = time.perf_counter()
        with (out / (name + '.txt')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
        rows.append({'check': name, 'argv': command, 'returncode': result.returncode,
                     'seconds': time.perf_counter() - started, 'log': name + '.txt'})
    summary = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': args.scope,
               'python': sys.version, 'checks': rows, 'source_sha256': before,
               'source_unchanged_during_checks': all(hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)] for p in source_paths),
               'all_passed': all(r['returncode'] == 0 for r in rows)}
    summary['all_passed'] = summary['all_passed'] and summary['source_unchanged_during_checks']
    (out / 'commands.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['all_passed'] else 1)


if __name__ == '__main__':
    main()
