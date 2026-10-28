"""CLI for a finite, reproducible experiment."""

import argparse
import json
import tempfile

from .experiment import run
from .answer_eval import run as run_answers
from .local_gguf import run as run_gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "answers", "answers-smoke", "gguf"))
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--predictions", help="Offline JSON outputs from one local model (answers only)")
    parser.add_argument("--model", help="Existing local GGUF file (gguf only)")
    args = parser.parse_args()
    if args.command == "gguf":
        if not args.model or args.predictions:
            parser.error("gguf requires --model and does not accept --predictions")
        print(run_gguf(args.model, args.output_dir))
        return
    if args.model:
        parser.error("--model is only valid with gguf")
    if args.command == "answers-smoke":
        with tempfile.TemporaryDirectory() as directory:
            path = run_answers(directory)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["protocol"]["case_counts"] != {"validation": 24, "test": 24}:
                raise RuntimeError("Answer evaluation report failed smoke checks")
            if (report["comparisons"]["lexical_retrieval"]["cluster_resampling"]["draws"] != 256
                    or report["methods"]["lexical_retrieval"]["overall"]["count"] != 24
                    or report["paired_audit"]["pair_count"] != 12
                    or len(report["paired_audit"]["pairs"]) != 12):
                raise RuntimeError("Answer comparison failed smoke checks")
            print("answers-smoke: generated dataset, read JSON, evaluated both languages, and wrote report")
    elif args.command == "answers":
        print(run_answers(args.output_dir, args.predictions))
    elif args.command == "smoke":
        with tempfile.TemporaryDirectory() as directory:
            path = run(directory)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["protocol"]["split_counts"] != {
                "train": 48, "validation": 24, "test": 24
            } or set(report["methods"]) != {"sum", "mean"} or \
                    report["training_sensitivity"]["protocol"]["draws"] != 24 or \
                    not (path.parent / "REPORT.md").is_file():
                raise RuntimeError("Experiment report failed smoke checks")
            print("smoke: generated dataset, read CSV, evaluated, and wrote report")
    else:
        print(run(args.output_dir))


if __name__ == "__main__":
    main()
