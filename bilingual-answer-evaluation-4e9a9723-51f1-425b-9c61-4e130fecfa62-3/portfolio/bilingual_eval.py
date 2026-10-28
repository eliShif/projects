"""Offline, paired Hebrew/English answer evaluation over synthetic records."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from .common import provenance, sha256_json, write_json
from .evidence_eval import score
from .retrieval import Document, Retriever


ANSWER_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": {"value": {"type": ["number", "null"]},
                                "citations": {"type": "array", "items": {"type": "string"}}},
                 "required": ["value", "citations"]}
ANSWER_PROMPT = ("Answer using only the supplied records. Use the highest revision for the "
                 "requested service and field. If the field is absent, abstain with null. "
                 "Return JSON with value and cited record IDs. The question may be in Hebrew or English.")


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


def retrieval_trace(corpus, cases):
    """Freeze the record IDs actually available to either answer system."""
    retriever = Retriever([Document(row["id"], json.dumps(row, ensure_ascii=False)) for row in corpus])
    return {case["id"]: [hit["id"] for hit in retriever.search(case["question"], 1)]
            for case in cases}


def lexical_baseline(corpus, cases):
    """Retrieve one JSON record, then copy its value; deliberately simple."""
    trace = retrieval_trace(corpus, cases)
    by_id = {row["id"]: row for row in corpus}
    predictions = []
    for case in cases:
        ids = trace[case["id"]]
        row = by_id[ids[0]] if ids else None
        # A baseline may abstain when the requested field does not appear in the record.
        if row and row["field"] in case["question"]:
            predictions.append({"case_id": case["id"], "status": "answered",
                                "value": row["value"], "citations": [row["id"]]})
        else:
            predictions.append({"case_id": case["id"], "status": "abstained",
                                "value": None, "citations": []})
    return predictions


def model_predictions(corpus, cases, model):
    """Run a local model on the same top-one retrieved record as the lexical baseline."""
    trace = retrieval_trace(corpus, cases)
    by_id = {row["id"]: row for row in corpus}
    predictions, generations = [], []
    for index, case in enumerate(cases):
        records = [by_id[record_id] for record_id in trace[case["id"]]]
        request = json.dumps({"question": case["question"], "records": records}, ensure_ascii=False)
        response = None
        try:
            response = model.generate_json(ANSWER_PROMPT, request, ANSWER_SCHEMA, seed=4200 + index)
            parsed = json.loads(response["text"])
            if not isinstance(parsed, dict) or set(parsed) != {"value", "citations"}:
                raise ValueError("Model answer must contain only value and citations")
            value, citations = parsed["value"], parsed["citations"]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Model value must be numeric or null")
            if not isinstance(citations, list) or any(type(c) is not str or c not in {r["id"] for r in records} for c in citations):
                raise ValueError("Model cited a record it was not shown")
            prediction = {"case_id": case["id"], "status": "abstained" if value is None else "answered",
                          "value": value, "citations": citations}
            generations.append({"case_id": case["id"], "retrieved_ids": [r["id"] for r in records],
                                "response": response})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            prediction = {"case_id": case["id"], "status": "failed", "value": None, "citations": []}
            generations.append({"case_id": case["id"], "retrieved_ids": [r["id"] for r in records],
                                "response": response, "error_type": type(exc).__name__, "error": str(exc)})
        predictions.append(prediction)
    return predictions, generations


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
        if (value is not None and (type(value) not in (int, float) or not math.isfinite(value))) or not isinstance(cites, list) or any(type(c) is not str or c not in source_ids for c in cites):
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
        if prediction["status"] == "failed":
            reason = "generation_failure"
        elif not marks["correct"]:
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


def diagnose_cases(cases, rows, baseline_rows, trace):
    """Expose paired outcomes and whether the expected source reached the answerer."""
    evaluated = {row["case_id"]: row for row in rows}
    baseline = {row["case_id"]: row for row in baseline_rows}
    diagnostics = []
    for case in cases:
        case_id = case["id"]
        retrieved = trace[case_id]
        # Absence cannot be established from a single retrieved record.
        evidence_available = (None if case["expected"] is None else
                              set(case["evidence_ids"]).issubset(retrieved))
        result = evaluated[case_id]
        if result["error_type"] is None:
            cause = None
        elif evidence_available is False:
            cause = "required_record_not_retrieved"
        else:
            cause = result["error_type"]
        diagnostics.append({"case_id": case_id, "pair_id": case["pair_id"],
                            "language": case["language"], "retrieved_ids": retrieved,
                            "expected_evidence_available": evidence_available,
                            "baseline_grounded_correct": baseline[case_id]["grounded_correct"],
                            "evaluated_grounded_correct": result["grounded_correct"],
                            "error_type": result["error_type"], "diagnostic_cause": cause})
    return diagnostics


def run(out, predictions_path=None, model_path=None):
    if predictions_path and model_path:
        raise ValueError("Choose either saved predictions or a local model")
    corpus, cases = dataset()
    generations = None
    model_info = None
    if model_path:
        if not Path(model_path).is_file():
            raise FileNotFoundError("Supply an existing local GGUF model file")
        from .gguf_lm import GGUFModel
        model = GGUFModel(model_path)
        predictions, generations = model_predictions(corpus, cases, model)
        model_info = {"name": model.model_name, "sha256": model.sha256,
                      "runtime": "llama-cpp-python", "seed_start": 4200,
                      "temperature": 0.2, "max_tokens": 180}
    elif predictions_path:
        predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    else:
        predictions = lexical_baseline(corpus, cases)
    rows, summary = evaluate(corpus, cases, predictions)
    baseline_predictions = lexical_baseline(corpus, cases)
    baseline_rows, baseline_summary = evaluate(corpus, cases, baseline_predictions)
    trace = retrieval_trace(corpus, cases)
    diagnostics = diagnose_cases(cases, rows, baseline_rows, trace)
    comparison = {language: {"baseline_grounded_correct": baseline_summary[language]["grounded_correct"],
                             "evaluated_grounded_correct": summary[language]["grounded_correct"],
                             "paired_difference": summary[language]["grounded_correct"] - baseline_summary[language]["grounded_correct"]}
                  for language in ("en", "he")}
    errors = [{"case_id": row["case_id"], "language": row["language"],
               "error_type": row["error_type"], "question": next(c["question"] for c in cases if c["id"] == row["case_id"]),
               "prediction": row["prediction"]} for row in rows if row["error_type"]]
    out = Path(out)
    source_dir = Path(__file__).parent
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("bilingual_eval.py", "evidence_eval.py", "retrieval.py", "gguf_lm.py")}
    write_json(out / "protocol.json", {"version": "bilingual-eval-v2", "mode": "local_gguf" if model_path else ("supplied" if predictions_path else "lexical_baseline"),
                                        "corpus_sha256": sha256_json(corpus), "cases_sha256": sha256_json(cases),
                                        "predictions_sha256": sha256_json(predictions),
                                        "retrieval_trace_sha256": sha256_json(trace),
                                        "source_hashes": source_hashes,
                                        "dependencies": {name: importlib.metadata.version(name)
                                                         for name in ("numpy", "scikit-learn")},
                                        "model": model_info, "retrieval_k": 1,
                                        "prompt_sha256": sha256_json(ANSWER_PROMPT),
                                        "criteria": "same numeric answer, abstention and expected citation rules in both languages",
                                        "provenance": provenance()})
    write_json(out / "corpus.json", corpus)
    write_json(out / "cases.json", cases)
    write_json(out / "predictions.json", predictions)
    write_json(out / "runs.json", rows)
    write_json(out / "summary.json", summary)
    write_json(out / "baseline_predictions.json", baseline_predictions)
    write_json(out / "baseline_runs.json", baseline_rows)
    write_json(out / "comparison.json", comparison)
    write_json(out / "errors.json", errors)
    write_json(out / "retrieval_trace.json", trace)
    write_json(out / "case_diagnostics.json", diagnostics)
    if generations is not None:
        write_json(out / "generations.json", generations)
        generation_by_id = {item["case_id"]: item for item in generations}
        write_json(out / "review_queue.json", [
            {"case_id": case["id"], "language": case["language"],
             "question": case["question"], "expected": case["expected"],
             "retrieved_ids": trace[case["id"]],
             "raw_response": generation_by_id[case["id"]]["response"],
             "automatic_diagnostic": next(d for d in diagnostics if d["case_id"] == case["id"]),
             "human_review": "pending"} for case in cases])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions", help="JSON array of case_id, status, value, citations")
    parser.add_argument("--model", help="Existing local GGUF model file; runs real CPU inference")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.predictions, args.model), ensure_ascii=False, indent=2))
