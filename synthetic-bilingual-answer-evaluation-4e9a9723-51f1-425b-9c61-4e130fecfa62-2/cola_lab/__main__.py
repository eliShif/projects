"""CLI for a finite, reproducible experiment."""

import argparse
import json
import tempfile

from .experiment import run
from .answer_eval import run as run_answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "answers", "answers-smoke"))
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--predictions", help="Offline JSON outputs from one local model (answers only)")
    args = parser.parse_args()
    if args.command == "answers-smoke":
        with tempfile.TemporaryDirectory() as directory:
            path = run_answers(directory)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["protocol"]["case_counts"] != {"validation": 24, "test": 24}:
                raise RuntimeError("Answer evaluation report failed smoke checks")
            print("answers-smoke: generated dataset, read JSON, evaluated both languages, and wrote report")
    elif args.command == "answers":
        print(run_answers(args.output_dir, args.predictions))
    elif args.command == "smoke":
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
