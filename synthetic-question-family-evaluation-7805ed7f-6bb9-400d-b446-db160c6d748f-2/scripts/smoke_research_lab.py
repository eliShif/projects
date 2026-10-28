"""Finite offline smoke: run the documented CLI, replay its input, reject bad data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("cases.json", "challenge_cases.json", "protocol.json", "selection.json", "summary.json", "REPORT.md")
SOURCES = ("portfolio/research_lab.py", "portfolio/research_cases.py", "portfolio/evidence_agent.py")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def invoke(out, cases=None, success=True, isolated=False):
    command = [sys.executable, *(["-I"] if isolated else []), "-m", "portfolio.research_lab", "--out", str(out)]
    if cases is not None:
        command += ["--cases", str(cases)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    if success:
        require(result.returncode == 0, f"Documented command failed: {result.stderr}")
        return json.loads(result.stdout)
    require(result.returncode != 0, "Invalid input unexpectedly succeeded")
    require(not out.exists(), "Rejected input created an output directory")
    return result.stderr


def check_artifacts(out, printed):
    for name in ARTIFACTS:
        require((out / name).is_file(), f"Missing artifact: {name}")
    rows, challenge, protocol, selection, summary = (
        json.loads((out / name).read_text(encoding="utf-8")) for name in ARTIFACTS[:-1])
    require(protocol["cases_sha256"] == digest(rows), "Input hash mismatch")
    require(protocol["challenge_cases_sha256"] == digest(challenge), "Challenge hash mismatch")
    require(summary["selection_sha256"] == digest(selection), "Frozen choice hash mismatch")
    require(set(protocol["source_hashes"]) == set(SOURCES), "Unexpected source inventory")
    for name in SOURCES:
        require(protocol["source_hashes"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                f"Source hash mismatch: {name}")
    by_split = {s: [r for r in rows if r["split"] == s] for s in protocol["splits"]}
    require(selection["training_sha256"] == digest(by_split["train"]), "Training hash mismatch")
    require(selection["validation_sha256"] == digest(by_split["validation"]), "Validation hash mismatch")
    require(selection["fit_case_ids"] == [r["id"] for r in by_split["train"]], "Fit IDs mismatch")
    require(selection["validation_case_ids"] == [r["id"] for r in by_split["validation"]], "Selection IDs mismatch")
    for split, split_rows in by_split.items():
        quality = summary["quality"][split]
        require(quality["cases"] == len(split_rows), f"Split size mismatch: {split}")
        for label in protocol["label_order"]:
            count = sum(r["label"] == label for r in split_rows)
            require(count > 0 and quality["label_counts"][label] == count, f"Family support mismatch: {split}/{label}")
        require(quality["missing_labels"] == [], f"Missing family: {split}")
    for row in rows + challenge:
        key = "challenge_mentions" if row["id"].startswith("challenge-") else "case_mentions"
        mentions = summary["entity_audit"][key][row["id"]]
        require(row["entity"] in mentions, "Primary entity not audited")
        require(set(mentions) <= set(protocol["splits"][row["split"]]), "Cross-split entity mention")
    ranked = sorted(selection["candidate_metrics"],
                    key=lambda n: (-selection["candidate_metrics"][n]["macro_f1"],
                                   -selection["candidate_metrics"][n]["accuracy"], n))
    winner = selection["selected"]
    require(winner == ranked[0] == summary["selected"] == printed["selected"], "Selection mismatch")
    for name in protocol["candidates"]:
        for metric, value in selection["candidate_metrics"][name].items():
            require(value == summary["validation"][name][metric], "Validation score mismatch")
        require(sum(map(sum, summary["test"][name]["confusion"])) == len(by_split["test"]), "Lost test cases")
    require(printed["test_accuracy"] == summary["test"][winner]["accuracy"], "CLI metric mismatch")
    require(printed["challenge_accuracy"] == summary["new_template_challenge"][winner]["accuracy"], "CLI challenge mismatch")
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/research_lab")
    parser.add_argument("--installed", action="store_true", help="also run the installed package with Python -I")
    args = parser.parse_args(argv)
    out = (ROOT / args.out).resolve()
    require(out.is_relative_to(ROOT), "Smoke output must stay inside the project workspace")
    printed = invoke(out)
    rows, summary = check_artifacts(out, printed)
    with TemporaryDirectory(prefix=".research-smoke-", dir=out.parent) as temporary:
        temporary = Path(temporary)
        replay = temporary / "replay"
        replay_printed = invoke(replay, out / "cases.json")
        check_artifacts(replay, replay_printed)
        for name in ARTIFACTS:
            require((out / name).read_bytes() == (replay / name).read_bytes(), f"Replay differs: {name}")
        if args.installed:
            installed = temporary / "installed"
            installed_printed = invoke(installed, isolated=True)
            check_artifacts(installed, installed_printed)
            for name in ARTIFACTS:
                require((out / name).read_bytes() == (installed / name).read_bytes(), f"Installed package differs: {name}")
        bad = json.loads(json.dumps(rows))
        target = next(r for r in bad if r["entity"] == "Cobalt" and r["label"] == "sum")
        target["question"] = "What is the combined daily quota for Cobalt and Garnet?"
        path = temporary / "secondary_leak.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        require("leakage through question mentions" in invoke(temporary / "rejected_leak", path, False),
                "Wrong rejection for secondary entity leakage")
        bad = [r for r in rows if not (r["split"] == "validation" and r["label"] == "ratio")]
        path = temporary / "missing_family.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        require("missing: ratio" in invoke(temporary / "rejected_family", path, False),
                "Wrong rejection for missing family")
    print(json.dumps({"status": "ok", "documented_run": printed,
                      "cases": {s: item["cases"] for s, item in summary["quality"].items()},
                      "checks": ["artifact hashes and counts", "validation choice", "identical JSON-input replay",
                                 "secondary-entity rejection", "missing-family rejection"]
                                + (["installed package replay with Python -I"] if args.installed else [])}, sort_keys=True))


if __name__ == "__main__":
    main()
