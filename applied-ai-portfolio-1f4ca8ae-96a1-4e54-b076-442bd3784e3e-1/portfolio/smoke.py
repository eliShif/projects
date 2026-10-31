"""Finite verification of the documented store path, optionally real vision/HTTP."""
import argparse
import json
from pathlib import Path
import tempfile

from .edge_benchmark import run as vision_run
from .edge_protocol import Config
from .measurement import file_hash
from .service_load import run, run_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision', action='store_true', help='Require PyTorch and run all four real model variants')
    parser.add_argument('--http', action='store_true', help='Require loopback sockets and verify the HTTP path')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='portfolio-smoke-') as temp:
        root = Path(temp)
        summary = (run if args.http else run_store)(root / 'service', jobs=4, clients=2, copies=2)
        if not all(summary['checks'].values()): raise RuntimeError('Service invariants failed')
        manifest = json.loads((root / 'service/manifest.json').read_text())
        if any(file_hash(root / 'service' / name) != digest for name, digest in manifest.items()):
            raise RuntimeError('Recorded artifacts do not match their manifest')
        if args.vision:
            result = vision_run(root / 'vision', Config(seeds=(7,), epochs=2, rounds=4, repetitions=5))
            if len(result['rows']) != 16 or len(result['test_rows']) != 4: raise RuntimeError('Incomplete vision run')
        print(json.dumps({'status': 'passed', 'service_transport': 'http' if args.http else 'store',
                          'vision_executed': args.vision, 'power_measured': False}))


if __name__ == '__main__':
    main()
