"""Finite offline smoke: run the documented CLI, replay its input, reject bad data."""
import argparse
import hashlib
import json
import math
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


def invoke(out, cases=None, success=True, isolated=False, train_sizes=None):
    command = [sys.executable, *(["-I"] if isolated else []), "-m", "portfolio.research_lab", "--out", str(out)]
    if cases is not None:
        command += ["--cases", str(cases)]
    if train_sizes is not None:
        command += ["--train-sizes", *map(str, train_sizes)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    if success:
        require(result.returncode == 0, f"Documented command failed: {result.stderr}")
        return json.loads(result.stdout)
    require(result.returncode != 0, "Invalid input unexpectedly succeeded")
    require(not out.exists(), "Rejected input created an output directory")
    return result.stderr


def check_curve_scores(score, rows, labels):
    """Recalculate both curve metrics from stored counts, with known support."""
    matrix = score["confusion"]
    require(isinstance(matrix, list) and len(matrix) == len(labels)
            and all(isinstance(row, list) and len(row) == len(labels)
                    and all(type(n) is int and n >= 0 for n in row) for row in matrix),
            "Learning-curve confusion matrix invalid")
    supports = [sum(r["label"] == label for r in rows) for label in labels]
    require([sum(row) for row in matrix] == supports, "Learning-curve confusion support mismatch")
    accuracy = sum(matrix[i][i] for i in range(len(labels))) / len(rows)
    f1s = []
    for i, support in enumerate(supports):
        predicted = sum(row[i] for row in matrix)
        f1s.append(2 * matrix[i][i] / (support + predicted) if support + predicted else 0.0)
    expected = {"accuracy": accuracy, "macro_f1": sum(f1s) / len(labels)}
    for key, value in expected.items():
        require(type(score[key]) in (int, float) and math.isclose(score[key], value, abs_tol=1e-12),
                f"Learning-curve metric mismatch: {key}")


def check_learning_curve(curve, protocol, by_split, summary):
    settings = protocol["learning_curve"]
    require(curve["protocol"] == settings, "Learning-curve protocol mismatch")
    sizes = settings["training_sizes"]
    labels = protocol["label_order"]
    require(len(sizes) >= 2 and all(type(n) is int and len(labels) <= n <= len(by_split["train"]) for n in sizes)
            and all(a < b for a, b in zip(sizes, sizes[1:])), "Learning-curve sizes invalid")
    require(type(settings["seed"]) is int, "Learning-curve seed invalid")
    require(settings["candidates"] == list(protocol["candidates"]), "Learning-curve candidates mismatch")
    require([p["training_size"] for p in curve["points"]] == sizes, "Learning-curve sizes mismatch")
    for split in ("train", "validation"):
        key = "training_sha256" if split == "train" else "validation_sha256"
        require(curve[key] == digest(sorted(by_split[split], key=lambda r: r["id"])), "Learning-curve input hash mismatch")
    validation_ids = sorted(r["id"] for r in by_split["validation"])
    require(curve["validation_case_ids"] == validation_ids, "Learning-curve validation IDs mismatch")
    mentions = summary["entity_audit"]["case_mentions"]
    validation_mentions = set().union(*(set(mentions[case_id]) for case_id in validation_ids))
    require(curve["validation_mentions"] == sorted(validation_mentions), "Learning-curve validation mentions mismatch")
    training = {r["id"]: r for r in by_split["train"]}
    previous = []
    for point in curve["points"]:
        ids = point["training_case_ids"]
        require(len(ids) == point["training_size"] == len(set(ids)) and set(ids) <= set(training)
                and ids[:len(previous)] == previous, "Learning-curve subset invalid or not nested")
        previous = ids
        sample = [training[case_id] for case_id in ids]
        counts = {label: sum(r["label"] == label for r in sample) for label in labels}
        require(point["training_label_counts"] == counts and all(counts.values()), "Learning-curve family coverage mismatch")
        training_mentions = set().union(*(set(mentions[case_id]) for case_id in ids))
        require(point["training_mentions"] == sorted(training_mentions)
                and training_mentions <= set(protocol["splits"]["train"])
                and not training_mentions & validation_mentions, "Learning-curve entity isolation mismatch")
        require(set(point["metrics"]) == set(protocol["candidates"]), "Learning-curve methods mismatch")
        for name, scores in point["metrics"].items():
            check_curve_scores(scores["train"], sample, labels)
            check_curve_scores(scores["validation"], by_split["validation"], labels)
            for metric in ("accuracy", "macro_f1"):
                gap = scores["train_minus_validation"][metric]
                require(type(gap) in (int, float) and math.isclose(
                    gap, scores["train"][metric] - scores["validation"][metric], abs_tol=1e-12),
                    f"Learning-curve gap mismatch: {name}/{metric}")
                if len(sample) == len(training):
                    require(scores["validation"][metric] == summary["validation"][name][metric],
                            "Learning-curve full-training validation mismatch")


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
    check_learning_curve(summary["learning_curve"], protocol, by_split, summary)
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
        require("strictly increasing integers" in invoke(temporary / "rejected_sizes", success=False, train_sizes=(8, 25)),
                "Wrong rejection for infeasible learning-curve sizes")
    print(json.dumps({"status": "ok", "documented_run": printed,
                      "cases": {s: item["cases"] for s, item in summary["quality"].items()},
                      "checks": ["artifact hashes and counts", "validation choice", "identical JSON-input replay",
                                 "learning-curve metrics, gaps and nested subset isolation",
                                 "secondary-entity rejection", "missing-family rejection", "infeasible training-size rejection"]
                                + (["installed package replay with Python -I"] if args.installed else [])}, sort_keys=True))


if __name__ == "__main__":
    main()
