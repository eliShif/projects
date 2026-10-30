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
    success = len(records) == len(commands) and all(r["exit_code"] == 0 for r in records)
    # Only attach artifacts after this invocation completed all checks. Old
    # result files are not evidence that a failed invocation produced them.
    artifacts = {}
    if success:
        for name in ("cases.json", "challenge_cases.json", "protocol.json", "selection.json", "summary.json", "REPORT.md"):
            path = ROOT / "results" / "research_lab" / name
            artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "package_versions": versions,
                "verification_source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in sorted([*ROOT.glob("tests/test_research*.py"),
                                                                   ROOT / "tests/test_evidence_agent.py",
                                                                   ROOT / "scripts/smoke_research_lab.py",
                                                                   ROOT / "scripts/verify_research_lab.py", ROOT / "pyproject.toml",
                                                                   ROOT / "README.md", ROOT / "CODE_MAP.md", ROOT / "EXERCISES.md"])},
                "artifact_sha256": artifacts, "status": "passed" if success else "failed",
                "commands": records, "scope": "Question-family lab plus shared evidence-agent caller tests; no remote APIs or models"}
    (out / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Local verification", "",
             f"Recorded from actual subprocess results at {manifest['executed_at_utc']}.", "",
             f"Status: **{manifest['status']}**. Python: {sys.version.split()[0]}. "
             f"Package: {versions['synthetic-question-family-evaluation']}; setuptools: {versions['setuptools']}.", "",
             "The command manifest contains exact arguments, exit codes, output hashes, verification source hashes "
             "and (after success) hashes of the six generated experiment artifacts.", "",
             "[Command manifest](results/research_validation/commands.json) · "
             "[Experiment report](results/research_lab/REPORT.md)", "",
             "| Step | Exit code | Captured output |", "|---|---:|---|"]
    for record in records:
        lines.append(f"| {record['name']} | {record['exit_code']} | [{record['name']}.txt]({record['output_path']}) |")
    lines += ["", "## Commands", ""]
    for record in records:
        lines += [f"### {record['name']}", "", "```json", json.dumps(record["command"]), "```", "",
                  f"Output SHA-256: `{record['output_sha256']}`.", ""]
    lines += ["Scope: " + manifest["scope"] + ". The unrelated full portfolio suite was not run.", "",
              "These automated results verify code behavior in this environment. They do not establish "
              "personal mastery, independent authorship or completion of the learning exercises.", ""]
    (ROOT / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
