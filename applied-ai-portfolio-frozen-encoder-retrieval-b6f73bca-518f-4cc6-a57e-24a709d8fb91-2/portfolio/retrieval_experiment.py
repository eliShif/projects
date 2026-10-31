"""Reproducible, local-only comparison of TF-IDF and frozen text encoders.

prepare/audit/smoke use the standard library. run imports the existing ML stack
only after writing the protocol. No command downloads or invents model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import provenance, sha256_json, write_json
from .retrieval_study import (
    CANDIDATES, METRICS, check_k, compare, evaluate_queries, load_study,
    paired_differences, select_on_validation, summarize, validate_study,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "paraphrase_retrieval.json"
PAPER = {"title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
         "authors": ["Nils Reimers", "Iryna Gurevych"], "year": 2019,
         "venue": "EMNLP-IJCNLP", "pages": "3982–3992",
         "doi": "10.18653/v1/D19-1410", "url": "https://aclanthology.org/D19-1410/"}
LIMITS = [
    "Authored, small, in-language English/Hebrew cases; no independently reviewed relevance labels.",
    "Topic groups are held out, but shared writing style and authored templates limit generalization.",
    "All arms see the same full corpus, including documents relevant to held-out queries; labels are not indexed.",
    "No training in this experiment; the existing LoRA experiment is a separate protocol.",
    "TF-IDF omits zero-overlap results; dense retrieval returns up to k neighbors, even for negative cosine scores.",
    "Effects are descriptive. Paraphrases in a topic are correlated; no significance or general superiority claim.",
    "Checkpoint hashes identify local bytes, not their pretraining history or multilingual quality.",
    "Timings include Python/scoring overhead and fixed execution order; not a latency benchmark.",
]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _packages():
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _options(study, k, batch_size, max_length, threads, mode):
    check_k(k, len(study.documents))
    for name, value, maximum in (("batch_size", batch_size, 32), ("max_length", max_length, 512),
                                  ("threads", threads, 8)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    if mode not in ("compare", "baseline"):
        raise ValueError("mode must be compare or baseline")
    if max_length < 4:
        raise ValueError("max_length must be at least 4 for text and special tokens")


def prepare(data, out, *, k=3, batch_size=8, max_length=128, threads=2, mode="compare"):
    study = load_study(data)
    _options(study, k, batch_size, max_length, threads, mode)
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Output directory must be new or empty; previous runs are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    source = ["common.py", "retrieval.py", "dense_retrieval.py", "retrieval_study.py", "retrieval_experiment.py"]
    budget_texts = 2 * (len(study.documents) + len(study.queries)) if mode == "compare" else 0
    budget_calls = 2 * (math.ceil(len(study.documents) / batch_size) + len(study.queries)) if mode == "compare" else 0
    protocol = {
        "schema_version": 1, "experiment": "frozen-encoder-paraphrase-retrieval-v1",
        "paper": PAPER,
        "hypothesis": "Frozen masked-mean sentence embeddings improve held-out paraphrase MRR@k over TF-IDF on this corpus.",
        "status": "planned", "mode": mode,
        "candidates": list(CANDIDATES if mode == "compare" else ("tfidf",)),
        "k": k, "shared_corpus": True, "data_sha256": study.fingerprint,
        "counts": {"documents": len(study.documents), "validation": len(study.split("validation")),
                   "test": len(study.split("test"))},
        "selection": "validation MRR@k, then Recall@k, then tfidf/dense_mean/dense_cls order; saved before test",
        "model": {"suggested_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                  "version": None, "status": "not_loaded; local content version recorded by run"},
        "budget": {"device": "cpu", "threads": threads, "batch_size": batch_size,
                   "max_length": max_length, "training_steps": 0,
                   "max_encoder_texts": budget_texts, "max_encoder_forward_calls": budget_calls,
                   "max_padded_token_slots": budget_texts * max_length,
                   "note": "Two pooling arms each index the corpus once and encode each query once; includes special tokens and truncation."},
        "source_sha256": {"portfolio/" + name: hashlib.sha256((ROOT / "portfolio" / name).read_bytes()).hexdigest()
                          for name in source},
        "environment": provenance(), "packages": _packages(), "limits": LIMITS,
    }
    write_json(out / "study.json", study.as_dict())
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"status": "prepared", "model_evaluated": False})
    return study, protocol


def _render_report(summary):
    selected = summary["selection"]["selected"]
    lines = ["# Retrieval experiment", "", f"Mode: {summary['mode']}. Selected on validation: **{selected}**.",
             "", "The selected arm is the primary held-out result. Other test arms are predeclared descriptive comparisons; do not reselect on test.",
             "", "| Arm | Precision@k | Recall@k | MRR@k | Hit@k |", "|---|---:|---:|---:|---:|"]
    for name, metrics in summary["test"].items():
        lines.append("| " + name + " | " + " | ".join(f"{metrics[m]:.6f}" for m in METRICS) + " |")
    lines += ["", "## Interpretation", ""]
    if summary["mode"] == "baseline":
        lines.append("Only TF-IDF ran. The embedding hypothesis remains untested.")
    else:
        delta = summary["paired_test_differences"]["dense_mean"]["mean_query_difference"]
        lines.append(f"Masked-mean minus TF-IDF MRR@k: {delta:+.6f}. "
                     "This is an observed difference on these authored cases, not a general capability estimate.")
    lines += ["", "See rankings.json for individual successes/errors and summary.json for language slices and topic-paired differences.",
              "", "## Limits", ""] + ["- " + limit for limit in LIMITS]
    return "\n".join(lines) + "\n"


def run(data, out, *, model_path=None, model_id=None, **options):
    study, protocol = prepare(data, out, **options)
    out = Path(out)
    write_json(out / "status.json", {"status": "running", "model_evaluated": False})
    started = time.perf_counter()
    try:
        from .retrieval import Document, Retriever

        documents = [Document(d["id"], d["text"]) for d in study.documents]
        factories = {"tfidf": lambda: Retriever(documents)}
        encoder, load_seconds = None, None
        if protocol["mode"] == "compare":
            if model_path is None:
                raise ValueError("compare requires --model-path to local safetensors encoder weights")
            from .dense_retrieval import DenseRetriever, FrozenEncoder

            budget = protocol["budget"]
            start = time.perf_counter()
            encoder = FrozenEncoder(model_path, batch_size=budget["batch_size"],
                                    max_length=budget["max_length"], threads=budget["threads"],
                                    max_texts=budget["max_encoder_texts"],
                                    max_forward_calls=budget["max_encoder_forward_calls"])
            load_seconds = time.perf_counter() - start
            write_json(out / "encoder.json", {**encoder.manifest, "declared_model_id": model_id,
                       "provenance_note": "Identifier is operator-supplied; version is computed from local model/tokenizer bytes.",
                       "model_type": encoder.model.config.model_type,
                       "trainable_parameters": sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)})
            factories.update({"dense_mean": lambda: DenseRetriever(documents, encoder, pooling="mean"),
                              "dense_cls": lambda: DenseRetriever(documents, encoder, pooling="cls")})
        elif model_path is not None or model_id is not None:
            raise ValueError("baseline does not accept a model path or model identifier")
        result = compare(study, factories, protocol["k"],
                         on_selection=lambda selection: write_json(out / "selection.json", selection))
        rankings = {"validation": result.pop("validation_rows"), "test": result.pop("test_rows")}
        write_json(out / "rankings.json", rankings)
        summary = {"status": "complete", "mode": protocol["mode"], "k": protocol["k"],
                   "data_sha256": study.fingerprint, "protocol_sha256": sha256_json(protocol),
                   "rankings_sha256": sha256_json(rankings), "model_evaluated": encoder is not None,
                   "encoder_version": encoder.manifest["version"] if encoder else None,
                   "encoder_statistics": dict(encoder.stats) if encoder else None,
                   "encoder_load_seconds": load_seconds,
                   "run_seconds": time.perf_counter() - started, **result, "limits": LIMITS}
        (out / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
        # Write the completion marker last. Interrupted/failed runs have no summary.
        write_json(out / "summary.pending.json", summary)
        (out / "summary.pending.json").replace(out / "summary.json")
        write_json(out / "status.json", {"status": "complete", "model_evaluated": encoder is not None})
        return summary
    except Exception as exc:
        for name in ("summary.json", "summary.pending.json", "REPORT.md"):
            (out / name).unlink(missing_ok=True)
        write_json(out / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
        write_json(out / "status.json", {"status": "failed", "model_evaluated": False})
        raise


def audit(out):
    """Recompute saved metrics from rankings; not proof of model inference."""
    out = Path(out)
    study = validate_study(_read(out / "study.json"))
    protocol = _read(out / "protocol.json")
    if protocol["data_sha256"] != study.fingerprint:
        raise ValueError("Dataset fingerprint mismatch")
    budget = protocol["budget"]
    _options(study, protocol["k"], budget["batch_size"], budget["max_length"], budget["threads"], protocol["mode"])
    expected_candidates = list(CANDIDATES if protocol["mode"] == "compare" else ("tfidf",))
    expected_texts = 2 * (len(study.documents) + len(study.queries)) if protocol["mode"] == "compare" else 0
    expected_calls = 2 * (math.ceil(len(study.documents) / budget["batch_size"]) + len(study.queries)) if protocol["mode"] == "compare" else 0
    if (protocol["candidates"] != expected_candidates or protocol["shared_corpus"] is not True
            or budget["training_steps"] != 0 or budget["device"] != "cpu"
            or budget["max_encoder_texts"] != expected_texts
            or budget["max_encoder_forward_calls"] != expected_calls
            or budget["max_padded_token_slots"] != expected_texts * budget["max_length"]):
        raise ValueError("Protocol candidates or compute budget mismatch")
    state = _read(out / "status.json")
    status = state["status"]
    if status == "prepared":
        if state["model_evaluated"] or any((out / name).exists() for name in ("summary.json", "rankings.json", "selection.json")):
            raise ValueError("Prepared run cannot contain scored results")
        return {"status": "prepared", "data_sha256": study.fingerprint, "model_evaluated": False}
    if status != "complete":
        raise ValueError(f"Run is {status}, not complete or prepared")
    summary, rankings = _read(out / "summary.json"), _read(out / "rankings.json")
    if (summary["status"] != "complete" or summary["data_sha256"] != study.fingerprint
            or summary["protocol_sha256"] != sha256_json(protocol)
            or summary["rankings_sha256"] != sha256_json(rankings)
            or summary["k"] != protocol["k"] or summary["mode"] != protocol["mode"]
            or summary["model_evaluated"] != state["model_evaluated"]):
        raise ValueError("Run artifact fingerprint or protocol mismatch")
    if set(rankings) != {"validation", "test"}:
        raise ValueError("Missing evaluation split")
    for split in ("validation", "test"):
        if set(rankings[split]) != set(protocol["candidates"]):
            raise ValueError("Missing comparison arm")
        for rows in rankings[split].values():
            if len(rows) != len(study.split(split)):
                raise ValueError("Missing scored query")
            by_text = {row["query"]: row["hits"] for row in rows}
            class SavedRankings:
                def search(self, query, k):
                    return by_text[query]
            recomputed = evaluate_queries(study, study.split(split), SavedRankings(), protocol["k"])
            if rows != recomputed:
                raise ValueError("Saved query labels or metrics disagree with rankings")
    selection = select_on_validation(rankings["validation"])
    if selection != summary["selection"] or selection != _read(out / "selection.json"):
        raise ValueError("Selection was not determined by validation rankings")
    if summary["test"] != {name: summarize(rows) for name, rows in rankings["test"].items()}:
        raise ValueError("Test summaries disagree with rankings")
    expected_pairs = {name: paired_differences(rankings["test"]["tfidf"], rows)
                      for name, rows in rankings["test"].items() if name != "tfidf"}
    if expected_pairs != summary["paired_test_differences"]:
        raise ValueError("Paired effects disagree with rankings")
    if protocol["mode"] == "compare":
        encoder = _read(out / "encoder.json")
        if (encoder["version"] != "sha256:" + sha256_json(encoder["files"])
                or summary["encoder_version"] != encoder["version"]
                or encoder["trainable_parameters"] != 0 or not summary["model_evaluated"]):
            raise ValueError("Encoder identity or freezing metadata mismatch")
        stats, budget = summary["encoder_statistics"], protocol["budget"]
        for name, maximum in (("texts", "max_encoder_texts"), ("forward_calls", "max_encoder_forward_calls"),
                              ("padded_token_slots", "max_padded_token_slots")):
            if type(stats[name]) is not int or not 0 < stats[name] <= budget[maximum]:
                raise ValueError("Encoder statistics exceed the declared budget")
        if (stats["texts"] != expected_texts or stats["forward_calls"] != expected_calls
                or type(stats["attended_tokens"]) is not int
                or not 0 < stats["attended_tokens"] <= stats["padded_token_slots"]):
            raise ValueError("Incomplete encoder execution statistics")
    elif summary["model_evaluated"] or summary["encoder_version"] is not None:
        raise ValueError("Baseline cannot claim encoder evaluation")
    return {"status": "complete", "data_sha256": study.fingerprint,
            "model_evaluated": summary["model_evaluated"],
            "scope": "artifact consistency, not independent proof of inference"}


def smoke(*, baseline=False):
    """Finite CLI smoke: prepare, audit, reject tampering; optionally real TF-IDF."""
    with tempfile.TemporaryDirectory(prefix=".retrieval-smoke-", dir=ROOT) as directory:
        out = Path(directory) / "run"
        command = "run" if baseline else "prepare"
        args = [sys.executable, "-m", "portfolio.retrieval_experiment", command,
                "--out", str(out), "--mode", "baseline" if baseline else "compare"]
        done = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60)
        if done.returncode:
            raise RuntimeError(done.stderr.strip() or "Smoke run failed")
        checked = audit(out)
        raw = _read(out / "study.json")
        raw["documents"][0]["text"] += " Changed after preparation."
        write_json(out / "study.json", raw)
        try:
            audit(out)
        except ValueError:
            pass
        else:
            raise RuntimeError("Smoke failed to detect dataset tampering")
    return {"smoke": "ok", "path": command, "status": checked["status"],
            "scope": "real TF-IDF run and artifact audit" if baseline else "data/protocol preparation and tamper detection; no model scored"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        p = subs.add_parser(command)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--mode", choices=("compare", "baseline"), default="compare")
        p.add_argument("--k", type=int, default=3)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--max-length", type=int, default=128)
        p.add_argument("--threads", type=int, default=2)
        if command == "run":
            p.add_argument("--model-path", type=Path)
            p.add_argument("--model-id", help="Optional provenance label; file hashes identify the actual version")
    subs.add_parser("audit").add_argument("--out", type=Path, required=True)
    subs.add_parser("smoke").add_argument("--baseline", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            _, protocol = prepare(**args)
            result = {"status": "prepared", "data_sha256": protocol["data_sha256"], "model_evaluated": False}
        elif command == "run":
            summary = run(**args)
            result = {"status": summary["status"], "selected": summary["selection"]["selected"],
                      "model_evaluated": summary["model_evaluated"]}
        elif command == "audit":
            result = audit(**args)
        else:
            result = smoke(**args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
