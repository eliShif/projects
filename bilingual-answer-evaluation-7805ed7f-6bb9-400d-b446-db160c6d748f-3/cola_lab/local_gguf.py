"""Optional offline GGUF generation, sharing the existing evaluator's retrieval."""

import copy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .answer_eval import (CONTEXT_MODES, RETRIEVAL_RULE, input_exposure, make_dataset,
                          method_inputs, pair_id, run as run_answers, validate_dataset)


SETTINGS = {"n_ctx": 2048, "n_gpu_layers": 0, "n_threads": 1, "n_threads_batch": 1, "n_batch": 128,
            "temperature": 0.0, "max_tokens": 96, "top_p": 1.0, "top_k": 0,
            "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "mirostat_mode": 0,
            "stop": [], "seed": 17}
GENERATION_KEYS = ("temperature", "max_tokens", "top_p", "top_k", "min_p",
                   "typical_p", "repeat_penalty", "frequency_penalty", "presence_penalty",
                   "mirostat_mode", "stop", "seed")
SEED_POLICY = ("pair-repeat-v1: SHA256 of compact UTF-8 JSON [base_seed, pair_id]; "
               "first 4 bytes as unsigned big-endian integer, plus zero-based repeat_index, "
               "modulo 2147483647. Language and case order are excluded. Reset before every call.")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_options(context_mode, base_seed, repeats):
    if context_mode not in CONTEXT_MODES:
        raise ValueError("Unknown context mode")
    if type(base_seed) is not int or not 0 <= base_seed < 2147483647:
        raise ValueError("Base seed must be an integer in [0, 2147483647)")
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("Repeats must be an integer from 1 to 100")


def settings_for(case, base_seed, repeat_index):
    if type(repeat_index) is not int or not 0 <= repeat_index < 100:
        raise ValueError("Repeat index must be an integer from 0 to 99")
    validate_options("shared-document", base_seed, repeat_index + 1)
    key = json.dumps([base_seed, pair_id(case)], ensure_ascii=False, separators=(",", ":"))
    seed = (int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")
            + repeat_index) % 2147483647
    return {**copy.deepcopy(SETTINGS), "seed": seed}


def render_prompt(value):
    evidence = "\n".join(f"{doc['id']}: {doc['text']}" for doc in value["documents"])
    return ("Answer the question using only the evidence below. Return one JSON object "
            'with keys "answer" and "citation". Use a short exact answer and one listed '
            'document ID. If the evidence does not answer the question, use '
            '{"answer": null, "citation": null}. Do not add prose.\n'
            f"Evidence:\n{evidence}\nQuestion: {value['question']}\nJSON:")


def prompt_for(case, documents, context_mode="shared-document"):
    """Build a label-free prompt from the same selection used by the baseline."""
    return render_prompt(method_inputs(case, documents, context_mode)["local_model"])


def generation_requests(dataset, context_mode="shared-document", base_seed=17, repeats=1):
    """Freeze the case inputs and per-call settings before generating any outputs."""
    validate_options(context_mode, base_seed, repeats)
    validate_dataset(dataset)
    cases = sorted((case for case in dataset["cases"] if case["split"] == "test"),
                   key=lambda case: case["id"])
    requests = []
    for repeat_index in range(repeats):
        for case in cases:
            inputs = method_inputs(case, dataset["documents"], context_mode)
            prompt = render_prompt(inputs["local_model"])
            requests.append({"case_id": case["id"], "pair_id": pair_id(case),
                             "repeat_index": repeat_index, "inputs": inputs,
                             "exposures": {method: input_exposure(value)
                                           for method, value in inputs.items()},
                             "settings": settings_for(case, base_seed, repeat_index),
                             "prompt": prompt,
                             "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return requests


def parse_generation(raw, allowed_ids):
    """Reject malformed responses and citations outside the actual exposed evidence."""
    try:
        value = json.loads(raw.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError("Model response must be one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"answer", "citation"}:
        raise ValueError("Model response needs answer and citation")
    answer, citation = value["answer"], value["citation"]
    if answer is None and citation is None:
        return {"answer": None, "citations": []}
    if not isinstance(answer, str) or not answer.strip() or not isinstance(citation, str):
        raise ValueError("Answer and citation must both be nonempty strings")
    if citation not in allowed_ids:
        raise ValueError("Model cited a document outside the case evidence pool")
    return {"answer": answer.strip(), "citations": [citation]}


def validate_records(dataset, records, context_mode, base_seed, repeats):
    """Recompute requests to reject missing cases, input drift or unequal paired settings."""
    expected = {(request["repeat_index"], request["case_id"]): request for request in
                generation_requests(dataset, context_mode, base_seed, repeats)}
    seen = set()
    for record in records:
        key = (record.get("repeat_index"), record.get("case_id"))
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate generation record")
        seen.add(key)
        request = expected[key]
        for field, value in request.items():
            if record.get(field) != value:
                raise ValueError(f"Generation record {key} changed {field}")
        allowed = set(request["exposures"]["local_model"]["document_ids"])
        parsed = parse_generation(record.get("raw_generation"), allowed)
        if record.get("prediction") != parsed:
            raise ValueError("Saved prediction disagrees with raw generation")
    if seen != set(expected):
        raise ValueError("Generation records must cover every case and repeat")


def create_backend(model_path, settings):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("Install optional llama-cpp-python to run a GGUF model") from exc
    model = Llama(model_path=str(model_path), n_ctx=settings["n_ctx"],
                  n_gpu_layers=settings["n_gpu_layers"], n_threads=settings["n_threads"],
                  n_threads_batch=settings["n_threads_batch"], n_batch=settings["n_batch"],
                  seed=settings["seed"], verbose=False)

    def generate(prompt, call_settings):
        # Avoid carrying a preceding case's KV history or sampling seed into this call.
        model.reset()
        result = model(prompt, stream=False,
                       **{key: call_settings[key] for key in GENERATION_KEYS})
        return result["choices"][0]["text"]

    return generate


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(model_path, output_dir, backend_factory=create_backend, *,
        context_mode="shared-document", base_seed=17, repeats=1):
    """Generate all repeats, validate their protocol, then reuse the existing scorer.

    Injection is for tests only and is labeled in every saved model report. An empty
    output directory prevents a failed run from appearing to have a stale result.
    """
    validate_options(context_mode, base_seed, repeats)
    model_path = Path(model_path)
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        raise ValueError("An existing local .gguf model file is required")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("GGUF output directory must be empty; choose a new run directory")
    dataset = make_dataset()
    requests = generation_requests(dataset, context_mode, base_seed, repeats)
    model_hash = sha256_file(model_path)
    settings = {**copy.deepcopy(SETTINGS), "seed": base_seed}
    generate = backend_factory(model_path, copy.deepcopy(settings))
    records = []
    for request in requests:
        raw = generate(request["prompt"], copy.deepcopy(request["settings"]))
        allowed = set(request["exposures"]["local_model"]["document_ids"])
        try:
            prediction = parse_generation(raw, allowed)
        except ValueError as exc:
            raise ValueError(f"Invalid generation for {request['case_id']}: {exc}") from exc
        records.append({**request, "raw_generation": raw, "prediction": prediction})
    validate_records(dataset, records, context_mode, base_seed, repeats)
    is_real_backend = backend_factory is create_backend
    backend = "llama-cpp-python" if is_real_backend else "injected-test-backend"
    backend_version = None
    if is_real_backend:
        try:
            backend_version = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            pass
    comparison = ("shared_document_verified" if context_mode == "shared-document"
                  else "different_evidence_system_comparison")
    repeat_artifacts = []
    for repeat_index in range(repeats):
        repeat_output = output if repeat_index == 0 else output / f"repeat-{repeat_index}"
        repeat_output.mkdir(parents=True, exist_ok=True)
        repeat_records = [record for record in records if record["repeat_index"] == repeat_index]
        prediction_path = repeat_output / "gguf_predictions.json"
        _write_json(prediction_path, {"model_id": model_path.name, "model_version": model_hash,
                    "predictions": [{"case_id": record["case_id"], **record["prediction"]}
                                    for record in repeat_records]})
        # Pass this exact dataset, not a second generator call that could silently drift.
        report_path = run_answers(repeat_output, prediction_path, dataset=dataset)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        control = report["protocol"]["information_control"]
        by_id = {record["case_id"]: record for record in repeat_records}
        for exposure in control["case_exposures"]:
            record = by_id[exposure["case_id"]]
            if exposure["methods"]["lexical_retrieval"] != record["exposures"]["lexical_retrieval"]:
                raise ValueError("Evaluator and generator used different baseline inputs")
            exposure["methods"] = record["exposures"]
            exposure["generation_settings"] = record["settings"]
        control.update({"status": comparison, "context_mode": context_mode,
                        "repeat_index": repeat_index, "seed_policy": SEED_POLICY})
        report["protocol"]["model_outputs"].update({"backend": backend,
            "backend_version": backend_version, "model_quality_evidence": is_real_backend,
            "input": "generated locally; exact inputs and settings checked against frozen protocol"})
        report["limitations"].extend([
            "Shared evidence controls document exposure, not pretraining, template knowledge or language effects.",
            "The all-documents mode compares systems with different evidence; it cannot isolate a model effect.",
            "Seeds match within a translation pair; translations need not retrieve equivalent facts.",
            "Repeated greedy runs may be identical; repeats are not independent samples or confidence intervals.",
            "Malformed model output aborts the run; no partial quality score is published."])
        if not is_real_backend:
            report["limitations"].append("Injected test backend: these outputs are not model-quality evidence.")
        _write_json(report_path, report)
        repeat_artifacts.append({"repeat_index": repeat_index,
            "predictions_file": str(prediction_path.relative_to(output)),
            "predictions_sha256": sha256_file(prediction_path),
            "report_file": str(report_path.relative_to(output)), "report_sha256": sha256_file(report_path),
            "dataset_sha256": sha256_file(repeat_output / "answer_dataset.json"),
            "local_model_accuracy": report["methods"]["local_model"]["overall"]["accuracy"]})
    manifest = {"schema_version": 2, "model_file": model_path.name, "model_sha256": model_hash,
                "dataset_sha256": repeat_artifacts[0]["dataset_sha256"],
                "predictions_sha256": repeat_artifacts[0]["predictions_sha256"],
                "generation_code_sha256": sha256_file(__file__),
                "evaluator_code_sha256": sha256_file(Path(__file__).with_name("answer_eval.py")),
                "python": platform.python_version(), "platform": platform.platform(),
                "backend": backend, "backend_version": backend_version,
                "model_quality_evidence": is_real_backend, "context_mode": context_mode,
                "comparison": comparison, "retrieval_rule": RETRIEVAL_RULE,
                "settings": settings, "seed_policy": SEED_POLICY, "repeats": repeats,
                "repeat_artifacts": repeat_artifacts, "generations": records}
    _write_json(output / "gguf_run.json", manifest)
    return output / "answer_report.json"
