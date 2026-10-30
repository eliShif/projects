"""Capture real local commands and raw output; never run a network/model service."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/current")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error("Validation output must be a new directory inside this project")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Validation output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    relative = output.relative_to(ROOT).as_posix()
    paths = sorted([path for folder in ("cola_lab", "tests", "scripts")
                    for path in (ROOT / folder).glob("*.py")]
                   + list((ROOT / "tests/fixtures").glob("*.json"))
                   + [ROOT / name for name in ("pyproject.toml", "README.md", "CODE_MAP.md",
                                               "EXERCISES.md", "UPSTREAM.md")])
    source = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    write(output / "source-manifest.json", source)
    py = ".venv/bin/python"
    commands = [
        ("venv", ["python3", "-m", "venv", "--system-site-packages", ".venv"], 0),
        ("install", [py, "-m", "pip", "install", "--no-index", "--no-build-isolation", "--no-deps", "-e", "."], 0),
        ("tests", [py, "-m", "unittest", "discover", "-s", "tests", "-v"], 0),
        ("build", [py, "-m", "pip", "wheel", "--no-index", "--no-build-isolation", "--no-deps", "--wheel-dir", "dist", "."], 0),
        ("baseline", [py, "-m", "cola_lab", "answers", "--output-dir", relative + "/baseline"], 0),
        ("plan", [py, "-m", "cola_lab", "gguf-plan", "--output-dir", relative + "/plan",
                  "--context-mode", "shared-document", "--seed", "17", "--repeats", "2"], 0),
        ("missing_model", [py, "-m", "cola_lab", "gguf", "--model", relative + "/not-provided.gguf",
                           "--protocol", relative + "/plan/gguf_protocol.json",
                           "--output-dir", relative + "/missing-model"], 2),
        ("audit_failure", [py, "-m", "cola_lab", "gguf-verify", "--output-dir", relative + "/missing-model"], 0),
        ("baseline_smoke", [py, "-m", "cola_lab", "answers-smoke"], 0),
        ("smoke", [py, "-m", "cola_lab", "study-smoke"], 0),
        ("retained_smoke", [py, "-m", "cola_lab", "smoke"], 0),
    ]
    records = []
    for name, argv, expected_exit in commands:
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=180)
        stdout, stderr = output / f"{name}.stdout.txt", output / f"{name}.stderr.txt"
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        record = {"name": name, "command": {"executable": argv[0], "args": argv[1:]},
                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "exit_code": result.returncode, "expected_exit_code": expected_exit,
                  "stdout": stdout.name, "stderr": stderr.name,
                  "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr),
                  "source_manifest_sha256": digest(output / "source-manifest.json")}
        records.append(record)
        write(output / "commands.json", records)
        print(f"{name}: exit {result.returncode} (expected {expected_exit})", flush=True)
        if result.returncode != expected_exit:
            print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
            return 1
    if source != {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}:
        raise RuntimeError("Source files changed during validation")
    baseline = json.loads((output / "baseline/answer_report.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit_failure.stdout.txt").read_text(encoding="utf-8"))
    write(output / "observations.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "baseline_case_counts": baseline["protocol"]["case_counts"],
        "validation_selected_threshold": baseline["protocol"]["threshold"],
        "reference_validation": baseline["protocol"]["reference_validation"],
        "metric_policy": baseline["protocol"]["metric_policy"],
        "baseline_methods": {name: {"overall": row["overall"], "by_language": row["by_language"],
                                     "by_answerability": row["by_answerability"]}
                             for name, row in baseline["methods"].items()},
        "missing_model_audit": audit,
        "scope": "Actual baseline and unavailable-file preflight. No weights or real-model outputs.",
        "wheels": {path.name: digest(path) for path in sorted((ROOT / "dist").glob("*.whl"))}})
    print(f"Raw validation evidence: {relative}/commands.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
