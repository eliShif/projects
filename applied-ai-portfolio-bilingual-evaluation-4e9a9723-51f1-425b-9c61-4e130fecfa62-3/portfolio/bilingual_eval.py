"""Offline, paired Hebrew/English answer evaluation over synthetic records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import provenance, sha256_json, write_json
from .evidence_eval import score
from .retrieval import Document, Retriever


def dataset():
    # Same facts and question types in each language; no model output is embedded.
    corpus = [
        {"id": "cedar-old", "service": "Cedar", "field": "retention_days", "value": 60, "revision": 1},
        {"id": "cedar-current", "service": "Cedar", "field": "retention_days", "value": 23, "revision": 2},
        {"id": "dune-current", "service": "Dune", "field": "retention_days", "value": 17, "revision": 2},
    ]
    questions = {
        "en": ["What is Cedar retention_days in the current record?",
               "What is Dune retention_days in the current record?",
               "What is Cedar on_call_staff in the current record?"],
        "he": ["מה ערך retention_days העדכני של Cedar?",
               "מה ערך retention_days העדכני של Dune?",
               "מה ערך on_call_staff העדכני של Cedar?"],
    }
    cases = []
    for language, prompts in questions.items():
        for name, question, expected, ids in zip(
            ("cedar", "dune", "missing"), prompts, (23, 17, None),
            (["cedar-current"], ["dune-current"], []), strict=True
        ):
            cases.append({"id": f"{language}-{name}", "pair_id": name,
                          "language": language, "question": question,
                          "expected": expected, "evidence_ids": ids})
    return corpus, cases


def lexical_baseline(corpus, cases):
    """Retrieve one JSON record, then copy its value; deliberately simple."""
    retriever = Retriever([Document(row["id"], json.dumps(row, ensure_ascii=False)) for row in corpus])
    by_id = {row["id"]: row for row in corpus}
    predictions = []
    for case in cases:
        hits = retriever.search(case["question"], 1)
        row = by_id[hits[0]["id"]] if hits else None
        # A baseline may abstain when the requested field does not appear in the record.
        if row and row["field"] in case["question"]:
            predictions.append({"case_id": case["id"], "status": "answered",
                                "value": row["value"], "citations": [row["id"]]})
        else:
            predictions.append({"case_id": case["id"], "status": "abstained",
                                "value": None, "citations": []})
    return predictions


def evaluate(corpus, cases, predictions):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON array")
    ids = {case["id"] for case in cases}
    source_ids = {row["id"] for row in corpus}
    seen = set()
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(prediction) != {"case_id", "status", "value", "citations"}:
            raise ValueError("Prediction fields must be case_id, status, value, citations")
        case_id = prediction["case_id"]
        if case_id not in ids or case_id in seen:
            raise ValueError("Unknown or duplicate case_id")
        seen.add(case_id)
        status, value, cites = prediction["status"], prediction["value"], prediction["citations"]
        if status not in ("answered", "abstained", "failed") or (status == "answered") != (value is not None):
            if not (status == "failed" and value is None):
                raise ValueError("Status and value disagree")
        if (value is not None and (type(value) not in (int, float))) or not isinstance(cites, list) or any(type(c) is not str or c not in source_ids for c in cites):
            raise ValueError("Invalid value or citation")
    if seen != ids:
        raise ValueError("Missing predictions")
    by_case = {p["case_id"]: p for p in predictions}
    rows = []
    for case in cases:
        prediction = by_case[case["id"]]
        result = {"status": prediction["status"], "value": prediction["value"],
                  "citations": prediction["citations"]}
        marks = score(result, case)
        reason = None
        if not marks["correct"]:
            reason = "wrong_value_or_abstention"
        elif not marks["grounded_correct"]:
            reason = "missing_expected_evidence"
        rows.append({"case_id": case["id"], "pair_id": case["pair_id"],
                     "language": case["language"], "prediction": prediction,
                     **marks, "error_type": reason})
    summary = {}
    for language in ("en", "he"):
        selected = [row for row in rows if row["language"] == language]
        summary[language] = {"cases": len(selected),
                             "correct": sum(row["correct"] for row in selected),
                             "grounded_correct": sum(row["grounded_correct"] for row in selected),
                             "errors": [row["case_id"] for row in selected if row["error_type"]]}
    return rows, summary


def run(out, predictions_path=None):
    corpus, cases = dataset()
    predictions = (json.loads(Path(predictions_path).read_text(encoding="utf-8"))
                   if predictions_path else lexical_baseline(corpus, cases))
    rows, summary = evaluate(corpus, cases, predictions)
    out = Path(out)
    source_dir = Path(__file__).parent
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("bilingual_eval.py", "evidence_eval.py", "retrieval.py")}
    write_json(out / "protocol.json", {"version": "bilingual-eval-v1", "mode": "supplied" if predictions_path else "lexical_baseline",
                                        "corpus_sha256": sha256_json(corpus), "cases_sha256": sha256_json(cases),
                                        "predictions_sha256": sha256_json(predictions), "source_hashes": source_hashes,
                                        "criteria": "same numeric answer, abstention and expected citation rules in both languages",
                                        "provenance": provenance()})
    write_json(out / "corpus.json", corpus)
    write_json(out / "cases.json", cases)
    write_json(out / "predictions.json", predictions)
    write_json(out / "runs.json", rows)
    write_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions", help="JSON array of case_id, status, value, citations")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.predictions), ensure_ascii=False, indent=2))
