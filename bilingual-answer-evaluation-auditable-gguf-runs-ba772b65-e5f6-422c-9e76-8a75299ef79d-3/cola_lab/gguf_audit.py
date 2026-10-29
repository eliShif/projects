"""Offline integrity and scoring replay for the existing GGUF experiment.

Hashes detect changed artifacts, not authorship or trusted execution. No weights
are loaded and no inference is performed by this verifier.
"""

import json
from pathlib import Path

from .answer_eval import evaluate, load_model_predictions
from .local_gguf import (SEED_POLICY, load_protocol, parse_generation,
                         sha256_file, validate_records)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file(root, name):
    """Artifact names come from code, never from a manifest-supplied path."""
    path = root / name
    current = path
    while current != root.parent:
        _require(not current.is_symlink(), "Symlinked audit artifacts are not supported")
        current = current.parent
    _require(path.is_file(), f"Missing artifact: {name}")
    return path


def _read(root, name):
    return json.loads(_file(root, name).read_text(encoding="utf-8"))


def _hash(root, name, expected):
    _require(sha256_file(_file(root, name)) == expected, f"Hash mismatch: {name}")


def verify_bundle(output_dir, *, model_path=None):
    """Check complete or failed runs. Incomplete/crashed runs are never certified."""
    try:
        return _verify_bundle(Path(output_dir).absolute(), model_path)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"Invalid GGUF bundle schema: {exc}") from exc


def _verify_bundle(root, model_path):
    state = _read(root, "gguf_status.json")
    _require(state["schema_version"] == 1, "Unsupported status schema")
    _require(state["status"] in ("completed", "failed"), "Run is incomplete; no final result")
    _hash(root, "gguf_protocol.json", state["protocol_sha256"])
    protocol = load_protocol(_file(root, "gguf_protocol.json"))
    _hash(root, "gguf_attempts.jsonl", state["attempts_sha256"])
    attempts = [json.loads(line) for line in _file(root, "gguf_attempts.jsonl")
                .read_text(encoding="utf-8").splitlines()]
    requests, dataset = protocol["requests"], protocol["dataset"]
    languages = {case["id"]: case["language"] for case in dataset["cases"]}
    counts = {language: {"planned": sum(languages[r["case_id"]] == language for r in requests),
                         "attempted": 0, "parsed": 0, "invalid_output": 0, "backend_error": 0}
              for language in ("en", "he")}
    _require(len(attempts) <= len(requests), "Too many generation attempts")
    records = []
    for index, attempt in enumerate(attempts):
        request = requests[index]
        for field, expected in {"request_index": index, "case_id": request["case_id"],
                                "repeat_index": request["repeat_index"],
                                "language": languages[request["case_id"]]}.items():
            _require(attempt[field] == expected, f"Attempt {index} changed {field}")
        count = counts[attempt["language"]]
        count["attempted"] += 1
        allowed = set(request["exposures"]["local_model"]["document_ids"])
        outcome = attempt["outcome"]
        _require(outcome in ("ok", "invalid_output", "backend_error"), "Unknown attempt outcome")
        if outcome == "ok":
            parsed = parse_generation(attempt["raw_generation"], allowed)
            _require(parsed == attempt["prediction"] and attempt["error"] is None,
                     "Attempt prediction disagrees with raw generation")
            records.append({**request, "raw_generation": attempt["raw_generation"], "prediction": parsed})
            count["parsed"] += 1
        else:
            _require(index == len(attempts) - 1 and state["status"] == "failed",
                     "Only a failed run's last attempt can fail")
            _require(attempt["prediction"] is None and isinstance(attempt["error"], dict),
                     "Failed attempt must have an error and no prediction")
            if outcome == "invalid_output":
                try:
                    parse_generation(attempt["raw_generation"], allowed)
                except ValueError:
                    pass
                else:
                    raise ValueError("Invalid-output attempt actually contains a valid response")
            else:
                _require(attempt["raw_generation"] is None, "Backend failure must have no response")
            count[outcome] += 1
    for count in counts.values():
        count["not_attempted"] = count["planned"] - count["attempted"]
    _require(state["planned_requests"] == len(requests)
             and state["attempted_requests"] == len(attempts)
             and state["completed_requests"] == len(records), "Status/attempt counts disagree")
    _require(state["backend"] in ("llama-cpp-python", "injected-test-backend"), "Unknown backend")
    comparison = ("shared_document_verified" if protocol["context_mode"] == "shared-document"
                  else "different_evidence_system_comparison")
    result = {"status": state["status"], "protocol_sha256": state["protocol_sha256"],
              "context_mode": protocol["context_mode"],
              "comparison": comparison if state["status"] == "completed" else None,
              "backend": state["backend"], "model_sha256": state["model_sha256"],
              "model_file_checked": False, "model_quality_evidence": False,
              "by_language": counts, "quality_summary": None, "error": state["error"]}
    if model_path is not None:
        _require(state["model_sha256"] is not None, "Run has no recorded model hash")
        _require(sha256_file(model_path) == state["model_sha256"], "Model file hash mismatch")
        result["model_file_checked"] = True
    if state["status"] == "failed":
        _require(state["model_quality_evidence"] is False and isinstance(state["error"], dict),
                 "Failed run cannot claim quality evidence")
        _require(not (root / "gguf_run.json").exists(), "Failed run contains a final manifest")
        for repeat in range(protocol["repeats"]):
            repeat_dir = root if repeat == 0 else root / f"repeat-{repeat}"
            _require(not (repeat_dir / "answer_report.json").exists()
                     and not (repeat_dir / "gguf_predictions.json").exists(),
                     "Failed run contains published partial results")
        result["failure_phase"] = state["phase"]
        result["failure_case"] = ({key: attempts[-1][key] for key in
                                   ("case_id", "language", "repeat_index", "outcome")}
                                  if attempts and attempts[-1]["outcome"] != "ok" else None)
        return result

    _require(len(records) == len(requests) and state["error"] is None
             and state["phase"] == "completed", "Completed run is missing successful attempts")
    validate_records(dataset, records, protocol["context_mode"], protocol["base_seed"], protocol["repeats"])
    _hash(root, "gguf_run.json", state["manifest_sha256"])
    manifest = _read(root, "gguf_run.json")
    real = state["backend"] == "llama-cpp-python"
    expected_fields = {"schema_version": 3, "generations": records,
                       "context_mode": protocol["context_mode"], "comparison": comparison,
                       "repeats": protocol["repeats"], "settings": protocol["settings"],
                       "source_sha256": protocol["source_sha256"], "seed_policy": SEED_POLICY,
                       "retrieval_rule": protocol["retrieval_rule"], "model_quality_evidence": real,
                       "generation_code_sha256": protocol["source_sha256"]["local_gguf.py"],
                       "evaluator_code_sha256": protocol["source_sha256"]["answer_eval.py"]}
    for key in ("model_file", "model_sha256", "protocol_sha256", "attempts_sha256",
                "backend", "backend_version", "python", "platform"):
        expected_fields[key] = state[key]
    for key, expected in expected_fields.items():
        _require(manifest[key] == expected, f"Manifest changed {key}")
    _require(state["model_quality_evidence"] is real, "Status quality provenance disagrees")
    _require(len(manifest["repeat_artifacts"]) == protocol["repeats"], "Missing repeat artifacts")
    quality = []
    for repeat, artifact in enumerate(manifest["repeat_artifacts"]):
        prefix = "" if repeat == 0 else f"repeat-{repeat}/"
        predictions_name, report_name = prefix + "gguf_predictions.json", prefix + "answer_report.json"
        _require(artifact["repeat_index"] == repeat and artifact["predictions_file"] == predictions_name
                 and artifact["report_file"] == report_name, "Unexpected repeat artifact path/index")
        for name, field in ((predictions_name, "predictions_sha256"), (report_name, "report_sha256"),
                            (prefix + "answer_dataset.json", "dataset_sha256")):
            _hash(root, name, artifact[field])
        _require(artifact["dataset_sha256"] == manifest["dataset_sha256"]
                 and _read(root, prefix + "answer_dataset.json") == dataset, "Repeat dataset drift")
        repeat_records = [record for record in records if record["repeat_index"] == repeat]
        payload = _read(root, predictions_name)
        _require(payload == {"model_id": state["model_file"], "model_version": state["model_sha256"],
                             "predictions": [{"case_id": r["case_id"], **r["prediction"]}
                                             for r in repeat_records]}, "Predictions differ from attempts")
        replay = evaluate(dataset, artifact["dataset_sha256"],
                          load_model_predictions(_file(root, predictions_name), dataset))
        report = _read(root, report_name)
        for key in ("methods", "comparisons", "comparisons_with_lexical", "paired_audit"):
            _require(report[key] == replay[key], f"Report {key} disagrees with scoring replay")
        for key in ("dataset_sha256", "code_sha256", "threshold", "case_counts", "data_quality"):
            _require(report["protocol"][key] == replay["protocol"][key], f"Report protocol changed {key}")
        control = report["protocol"]["information_control"]
        _require(control["status"] == comparison and control["context_mode"] == protocol["context_mode"]
                 and control["repeat_index"] == repeat and control["seed_policy"] == SEED_POLICY,
                 "Report changed information-control protocol")
        by_id = {r["case_id"]: r for r in repeat_records}
        expected_exposures = [{"case_id": row["case_id"], "pair_id": row["pair_id"],
                               "methods": by_id[row["case_id"]]["exposures"],
                               "generation_settings": by_id[row["case_id"]]["settings"]}
                              for row in replay["protocol"]["information_control"]["case_exposures"]]
        _require(control["case_exposures"] == expected_exposures, "Report exposure/settings drift")
        outputs = report["protocol"]["model_outputs"]
        for key, expected in {"backend": state["backend"], "backend_version": state["backend_version"],
                              "model_quality_evidence": real, "model_id": state["model_file"],
                              "model_version": state["model_sha256"],
                              "output_sha256": artifact["predictions_sha256"]}.items():
            _require(outputs[key] == expected, f"Report changed model provenance: {key}")
        model = report["methods"]["local_model"]
        lexical = report["methods"]["lexical_retrieval"]
        _require(artifact["local_model_accuracy"] == model["overall"]["accuracy"], "Repeat accuracy drift")
        quality.append({"repeat_index": repeat, "by_language": {
            language: {"count": model["by_language"][language]["count"],
                       "local_model_accuracy": model["by_language"][language]["accuracy"],
                       "lexical_accuracy": lexical["by_language"][language]["accuracy"],
                       "accuracy_delta": (model["by_language"][language]["accuracy"]
                                          - lexical["by_language"][language]["accuracy"]),
                       "error_counts": model["by_language"][language]["error_counts"]}
            for language in ("en", "he")}, "versus_lexical": report["comparisons_with_lexical"]["local_model"]})
    _require(manifest["predictions_sha256"] == manifest["repeat_artifacts"][0]["predictions_sha256"],
             "Root predictions hash mismatch")
    result.update(model_quality_evidence=real, quality_summary=quality)
    return result
