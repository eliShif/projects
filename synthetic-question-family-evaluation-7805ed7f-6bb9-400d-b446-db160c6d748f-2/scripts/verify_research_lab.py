"""Run bounded local verification and retain actual command output and status."""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="also record a local offline pip installation")
    args = parser.parse_args()
    out = ROOT / "results" / "research_validation"
    out.mkdir(parents=True, exist_ok=True)
    interpreter = str(Path(sys.executable).relative_to(ROOT)) if Path(sys.executable).is_relative_to(ROOT) else sys.executable
    commands = [
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_research*.py", "-v"]),
        ("shared_generator_caller_tests", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_evidence_agent.py", "-v"]),
        ("build", ["-m", "compileall", "-q", "portfolio", "tests", "scripts"]),
        ("documented_run", ["-m", "portfolio.research_lab", "--out", "results/research_lab"]),
        ("smoke", ["scripts/smoke_research_lab.py", "--out", "results/research_lab", "--installed"]),
    ]
    if args.install:
        commands.insert(0, ("install", ["-m", "pip", "--isolated", "--disable-pip-version-check", "--no-cache-dir",
                                        "install", "--no-index", "--no-build-isolation", "--no-deps", "."]))
    records = []
    for name, args in commands:
        completed = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=60)
        path = out / f"{name}.txt"
        path.write_bytes(completed.stdout)
        record = {"name": name, "command": {"executable": interpreter, "args": args},
                  "exit_code": completed.returncode, "output_path": str(path.relative_to(ROOT)),
                  "output_sha256": hashlib.sha256(completed.stdout).hexdigest()}
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if completed.returncode != 0:
            break
    # A failed step is recorded as such; later steps are not claimed as run.
    versions = {}
    for package in ("setuptools", "synthetic-question-family-evaluation"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed (source execution)"
    manifest = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "package_versions": versions,
                "verification_source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in sorted([*ROOT.glob("tests/test_research*.py"),
                                                                   ROOT / "tests/test_evidence_agent.py",
                                                                   ROOT / "scripts/smoke_research_lab.py",
                                                                   ROOT / "scripts/verify_research_lab.py", ROOT / "pyproject.toml"])},
                "commands": records, "scope": "Question-family lab plus shared evidence-agent caller tests; no remote APIs or models"}
    (out / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if len(records) == len(commands) and all(r["exit_code"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
