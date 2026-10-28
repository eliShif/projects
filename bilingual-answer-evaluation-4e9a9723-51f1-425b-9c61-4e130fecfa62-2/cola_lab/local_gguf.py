"""Optional, offline GGUF generation for the existing bilingual evaluator."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .answer_eval import make_dataset, run as run_answers, validate_dataset


SETTINGS = {"n_ctx": 2048, "n_gpu_layers": 0, "temperature": 0,
            "max_tokens": 96, "seed": 17}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for(case, documents):
    """Use only same-split, same-language source text; never expose labels."""
    candidates = sorted((doc for doc in documents if doc["split"] == case["split"]
                         and doc["language"] == case["language"]), key=lambda doc: doc["id"])
    if not candidates:
        raise ValueError("No documents for case")
    evidence = "\n".join(f"{doc['id']}: {doc['text']}" for doc in candidates)
    return ("Answer the question using only the evidence below. Return one JSON object "
            'with keys "answer" and "citation". Use a short exact answer and one listed '
            'document ID. If the evidence does not answer the question, use '
            '{"answer": null, "citation": null}. Do not add prose.\n'
            f"Evidence:\n{evidence}\nQuestion: {case['question']}\nJSON:")


def parse_generation(raw, allowed_ids):
    """Reject malformed or cross-language evidence instead of scoring it as a model answer."""
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


def create_backend(model_path, settings):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("Install optional llama-cpp-python to run a GGUF model") from exc
    model = Llama(model_path=str(model_path), n_ctx=settings["n_ctx"],
                  n_gpu_layers=settings["n_gpu_layers"], seed=settings["seed"], verbose=False)

    def generate(prompt):
        result = model(prompt, max_tokens=settings["max_tokens"],
                       temperature=settings["temperature"])
        return result["choices"][0]["text"]

    return generate


def run(model_path, output_dir, backend_factory=create_backend):
    """Generate frozen test outputs, then reuse the existing evaluator unchanged."""
    model_path = Path(model_path)
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        raise ValueError("An existing local .gguf model file is required")
    dataset = make_dataset()
    validate_dataset(dataset)
    settings = dict(SETTINGS)
    model_hash = sha256_file(model_path)
    generate = backend_factory(model_path, settings)
    records, predictions = [], []
    for case in (case for case in dataset["cases"] if case["split"] == "test"):
        prompt = prompt_for(case, dataset["documents"])
        raw = generate(prompt)
        allowed = {doc["id"] for doc in dataset["documents"]
                   if doc["split"] == case["split"] and doc["language"] == case["language"]}
        try:
            prediction = parse_generation(raw, allowed)
        except ValueError as exc:
            raise ValueError(f"Invalid generation for {case['id']}: {exc}") from exc
        predictions.append({"case_id": case["id"], **prediction})
        records.append({"case_id": case["id"], "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(), "raw_generation": raw})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "gguf_predictions.json"
    prediction_path.write_text(json.dumps({"model_id": model_path.name,
        "model_version": model_hash, "predictions": predictions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report_path = run_answers(output, prediction_path)
    dataset_hash = sha256_file(output / "answer_dataset.json")
    backend_version = None
    if backend_factory is create_backend:
        try:
            backend_version = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {"model_file": model_path.name, "model_sha256": model_hash,
                "dataset_sha256": dataset_hash, "predictions_sha256": sha256_file(prediction_path),
                "generation_code_sha256": sha256_file(__file__), "python": platform.python_version(),
                "backend": "llama-cpp-python" if backend_factory is create_backend else "injected-test-backend",
                "backend_version": backend_version, "settings": settings, "generations": records}
    (output / "gguf_run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return report_path
