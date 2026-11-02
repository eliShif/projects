"""Check the complete recovery checkpoint against this focused repair."""
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_preservation import compare


BASELINE = "results/model_identity_layout_repair/repair_baseline.json"
REPAIR_CHANGES = {"portfolio/model_identity.py", "portfolio/local_lm.py",
                  "portfolio/security_models.py", "portfolio/security_pipeline.py", "scripts/check_preservation.py", "README.md", "CODE_MAP.md",
                  "EXERCISES.md", "UPSTREAM.md"}


def compare_repair(root, baseline):
    result = compare(root, baseline)
    result["unexpected_changes"] = sorted(set(result["changed_files"]) - REPAIR_CHANGES)
    # The older checker predates the separate real identity-test directory.
    unpreserved = set(result["missing_files"]) | set(result["changed_files"])
    result["preserved_tests"] = sum(
        name.startswith(("tests/", "tests_security/", "tests_ml/", "tests_ml_identity/"))
        and name not in unpreserved for name in baseline["files"])
    result["ok"] = not result["missing_files"] and not result["unexpected_changes"]
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    result = compare_repair(root, json.loads((root / BASELINE).read_text()))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
