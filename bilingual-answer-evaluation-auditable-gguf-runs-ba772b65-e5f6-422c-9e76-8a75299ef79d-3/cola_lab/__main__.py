"""CLI for a finite, reproducible experiment."""

import argparse
import json
import tempfile
from pathlib import Path

from .experiment import run
from .answer_eval import CONTEXT_MODES, input_exposure, method_inputs, run as run_answers
from .local_gguf import prepare, run as run_gguf
from .gguf_audit import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "answers", "answers-smoke", "gguf",
                                            "gguf-plan", "gguf-verify", "study-smoke"))
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--predictions", help="Offline JSON outputs from one local model (answers only)")
    parser.add_argument("--model", help="Existing local GGUF file (gguf or optional gguf-verify hash check)")
    parser.add_argument("--protocol", help="Frozen gguf_protocol.json (gguf only; no protocol overrides)")
    parser.add_argument("--context-mode", choices=CONTEXT_MODES,
                        help="GGUF evidence policy (default: shared-document)")
    parser.add_argument("--seed", type=int, help="GGUF base seed (default: 17)")
    parser.add_argument("--repeats", type=int, help="GGUF complete repeats, 1–100 (default: 1)")
    args = parser.parse_args()
    if args.protocol and args.command != "gguf":
        parser.error("--protocol is only valid with gguf")
    if args.command == "gguf-plan":
        if args.model or args.predictions:
            parser.error("gguf-plan does not accept --model or --predictions")
        try:
            print(prepare(args.output_dir, context_mode=args.context_mode or "shared-document",
                          base_seed=17 if args.seed is None else args.seed,
                          repeats=2 if args.repeats is None else args.repeats))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return
    if args.command == "gguf-verify":
        if args.predictions or any(value is not None for value in (args.context_mode, args.seed, args.repeats)):
            parser.error("gguf-verify accepts only --output-dir and optional --model")
        try:
            result = verify_bundle(args.output_dir, model_path=args.model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return
    if args.command == "gguf":
        if not args.model or args.predictions:
            parser.error("gguf requires --model and does not accept --predictions")
        try:
            print(run_gguf(args.model, args.output_dir,
                          context_mode=args.context_mode, base_seed=args.seed,
                          repeats=args.repeats, protocol_path=args.protocol))
        except (ValueError, RuntimeError, OSError) as exc:
            parser.error(str(exc))
        return
    if args.model or any(value is not None for value in (args.context_mode, args.seed, args.repeats)):
        parser.error("--model, --context-mode, --seed and --repeats are only valid with gguf")
    if args.predictions and args.command != "answers":
        parser.error("--predictions is only valid with answers")
    if args.command == "study-smoke":
        # No fixture backend or weights: exercise the real missing-resource path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = run_answers(root / "baseline")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["protocol"]["information_control"]["status"] != "baseline_only" \
                    or report["paired_audit"]["pair_count"] != 12:
                raise RuntimeError("Baseline report failed study smoke")
            protocol = prepare(root / "plan", repeats=2)
            try:
                run_gguf(root / "unavailable.gguf", root / "attempt", protocol_path=protocol)
            except ValueError as exc:
                if "existing local" not in str(exc):
                    raise
            else:
                raise RuntimeError("Missing weights must fail")
            audit = verify_bundle(root / "attempt")
            if (audit["status"] != "failed" or audit["quality_summary"] is not None
                    or audit["model_quality_evidence"]
                    or any(row["attempted"] != 0 or row["planned"] != 24
                           for row in audit["by_language"].values())):
                raise RuntimeError("Missing-model audit failed study smoke")
            print("study-smoke: 24 baseline cases, 12 pairs, 48 frozen requests; "
                  "missing-model failure recorded and verified; no model quality result")
    elif args.command == "answers-smoke":
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
            dataset = json.loads((path.parent / "answer_dataset.json").read_text(encoding="utf-8"))
            cases = {case["id"]: case for case in dataset["cases"] if case["split"] == "test"}
            control = report["protocol"]["information_control"]
            if control["status"] != "baseline_only" or len(control["case_exposures"]) != len(cases):
                raise RuntimeError("Incorrect baseline exposure provenance")
            for row in control["case_exposures"]:
                inputs = method_inputs(cases[row["case_id"]], dataset["documents"])
                if (row["methods"]["lexical_retrieval"] != input_exposure(inputs["lexical_retrieval"])
                        or inputs["local_model"] != inputs["lexical_retrieval"]):
                    raise RuntimeError("Shared document protocol failed smoke checks")
            print("answers-smoke: evaluated 24 bilingual test cases, checked selected-source exposure, "
                  "paired audit and report; baseline only, no model run")
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
