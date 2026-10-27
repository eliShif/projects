"""CLI for a finite, reproducible experiment."""

import argparse
import json
import tempfile

from .experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke"))
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    if args.command == "smoke":
        with tempfile.TemporaryDirectory() as directory:
            path = run(directory)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["protocol"]["split_counts"] != {
                "train": 48, "validation": 24, "test": 24
            } or set(report["methods"]) != {"sum", "mean"}:
                raise RuntimeError("Experiment report failed smoke checks")
            print("smoke: generated dataset, read CSV, evaluated, and wrote report")
    else:
        print(run(args.output_dir))


if __name__ == "__main__":
    main()
