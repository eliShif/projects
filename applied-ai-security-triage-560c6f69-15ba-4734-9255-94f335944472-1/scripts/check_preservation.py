"""Compare the supplied workspace inventory without rewriting historical files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


ALLOWED_CHANGES = {
    "portfolio/local_lm.py", "portfolio/security_models.py", "portfolio/security_pipeline.py",
    "README.md", "CODE_MAP.md", "EXERCISES.md", "UPSTREAM.md",
}


def generated_output(name):
    """Exclude project-root packaging/dependency outputs, never saved results.

    Installation rewrites egg-info, and clean builds may remove it entirely.
    Keep these entries in the supplied baseline for provenance while reporting
    them separately from the source/result preservation contract.
    """
    parts = PurePosixPath(name).parts
    if name == ".report-cpu-constraints.txt":
        return True  # Ephemeral host pip constraints, never project source/results.
    return len(parts) > 1 and (parts[0] in {".venv", "build", "dist"}
                              or parts[0].endswith(".egg-info"))


def compare(root, baseline):
    root = Path(root)
    inventory = baseline["files"]
    missing, changed, unchanged = [], [], []
    ignored = []
    for name, original in sorted(inventory.items()):
        if generated_output(name):
            ignored.append(name)
            continue
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != original["sha256"]:
            changed.append(name)
        else:
            unchanged.append(name)
    unexpected = sorted(set(changed) - ALLOWED_CHANGES)
    return {"source_commit": baseline["source_commit"], "baseline_files": len(inventory),
            "protected_files": len(inventory) - len(ignored), "ignored_generated_files": ignored,
            "unchanged_files": len(unchanged), "changed_files": changed,
            "missing_files": missing, "unexpected_changes": unexpected,
            "preserved_tests": sum(n.startswith(("tests/", "tests_ml/", "tests_security/")) for n in unchanged),
            "preserved_result_files": sum(n.startswith("results/") for n in unchanged),
            "ok": not missing and not unexpected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", default="results/model_identity_validation/baseline.json")
    args = parser.parse_args()
    result = compare(args.root, json.loads((args.root / args.baseline).read_text()))
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
