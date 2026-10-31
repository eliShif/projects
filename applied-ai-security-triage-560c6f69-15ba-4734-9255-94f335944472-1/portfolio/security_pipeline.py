"""Finite, resumable prepare -> fit -> evaluate workflow over the existing JobStore."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

from .common import sha256_json
from .reliable_service import JobStore, LeaseLost
from .security_data import build_dataset, validate_dataset, observations
from .security_eval import evaluate, replay, report
from .security_acceptance import run_acceptance, audit_acceptance, WORKERS, MAX_JOBS
from .security_models import ReferencePolicy, PretrainedPolicy, train_reference, train_pretrained, model_files

STAGES = ("prepare", "train", "evaluate")
SOURCE_FILES = ("common.py", "evidence_agent.py", "telemetry.py", "reliable_service.py",
                "adapter_tuning.py", "local_lm.py", "security_data.py", "security_triage.py",
                "security_models.py", "security_eval.py", "security_acceptance.py", "security_pipeline.py")


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def atomic_text(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_stage(payload):
    if not isinstance(payload, dict) or set(payload) != {"mode", "run_id", "stage"}:
        raise ValueError("Invalid stage payload")
    if payload["mode"] != "pipeline_stage" or payload["stage"] not in STAGES:
        raise ValueError("Unsupported stage")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or len(run_id) != 64 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid run identity")
    return dict(payload)


def verified_path(root, receipt, name):
    relative = receipt["artifacts"][name]["path"]
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Artifact outside run or absent")
    if file_hash(path) != receipt["artifacts"][name]["sha256"]:
        raise ValueError("Artifact checksum mismatch")
    return path


def read_artifact(root, receipt, name):
    return json.loads(verified_path(root, receipt, name).read_text())


def protocol(backend, steps, model_path):
    if backend not in ("reference", "pretrained") or type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError("Invalid experiment configuration")
    if (backend == "pretrained") != bool(model_path):
        raise ValueError("Pretrained backend requires an explicit local model path")
    models = {p.name: file_hash(p) for p in model_files(model_path)} if model_path else {}
    versions = {}
    for name in ("torch", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {"format": 2, "experiment": "security-triage-v2", "backend": backend, "steps": steps,
            "seed": 2026, "model_files_sha256": models, "python": sys.version.split()[0],
            "packages": versions, "dataset_sha256": sha256_json(build_dataset()),
            "source_sha256": {name: file_hash(Path(__file__).parent/name) for name in SOURCE_FILES},
            "information": "identical actor/time evidence and versioned policy for every arm; no labels",
            "metrics": ["correct", "grounded_correct", "correct_abstentions", "failed", "paired_changes"],
            "selection": "validation NLL only; test cases excluded from fitting and checkpoint selection",
            "acceptance": {"workers": WORKERS, "max_jobs": MAX_JOBS,
                           "transport": "in-memory HTTP handler, no sockets",
                           "faults": ["expired lease and reopened DB", "duplicate submissions", "invalid requests",
                                      "hostile fixture plans", "planner evidence mutation"]},
            "budgets": {"training_steps_max": 512, "worker_attempts": 3, "lease_seconds": 300,
                        "events_per_observation": 64, "stages": list(STAGES)}}


class Pipeline:
    def __init__(self, out, backend="reference", steps=96, model_path=None, clock=None):
        self.root = Path(out).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol(backend, steps, model_path)
        self.run_id = sha256_json(self.protocol)
        self.model_path = model_path
        previous = self.root / "protocol.json"
        if previous.exists():
            if json.loads(previous.read_text()) != self.protocol:
                raise ValueError("Run configuration/source changed; use a new output directory")
        else:
            if any(self.root.iterdir()):
                raise ValueError("New run requires an empty output directory")
            atomic_json(previous, self.protocol)
        kwargs = {"clock": clock} if clock else {}
        self.store = JobStore(self.root/"jobs.db", validator=validate_stage, **kwargs)

    def enqueue(self, stage):
        return self.store.enqueue(self.run_id + ":" + stage,
                                  {"mode": "pipeline_stage", "run_id": self.run_id, "stage": stage})[0]

    def predecessor(self, stage):
        job_id = self.enqueue(stage)
        job = self.store.get(job_id)
        if job["status"] != "succeeded":
            raise ValueError("Predecessor stage has not completed")
        receipt = job["result"]
        for name in receipt["artifacts"]:
            verified_path(self.root, receipt, name)
        return receipt

    def work_once(self,job_id=None):
        job = self.store.claim(job_id=job_id)
        if job is None:
            return False
        stage = job["payload"]["stage"]
        # Each lease owns a private attempt directory. Only a fenced successful
        # DB receipt publishes these files; stale attempts cannot overwrite them.
        attempt = self.root / "attempts" / (job["id"] + "-" + job["lease_token"])
        attempt.mkdir(parents=True, exist_ok=False)
        heartbeat = lambda: self.store.renew(job)
        try:
            if job["payload"]["run_id"] != self.run_id:
                raise ValueError("Wrong experiment")
            if stage == "prepare":
                data = build_dataset()
                validate_dataset(data)
                atomic_json(attempt/"dataset.json", data)
            else:
                prepared = self.predecessor("prepare")
                data = read_artifact(self.root, prepared, "dataset.json")
                validate_dataset(data)
                if sha256_json(data) != self.protocol["dataset_sha256"]:
                    raise ValueError("Dataset does not match protocol")
                if stage == "train":
                    args = (observations(data["train"]), data["train"], observations(data["validation"]), data["validation"], attempt)
                    options = {"steps": self.protocol["steps"], "heartbeat": heartbeat}
                    if self.protocol["backend"] == "reference":
                        train_reference(*args, **options)
                    else:
                        train_pretrained(self.model_path, *args, **options)
                else:
                    trained = self.predecessor("train")
                    training = read_artifact(self.root, trained, "training.json")
                    if self.protocol["backend"] == "reference":
                        base = ReferencePolicy()
                        adapted = ReferencePolicy(state=read_artifact(self.root, trained, "adapter.json"))
                    else:
                        base = PretrainedPolicy(self.model_path)
                        adapted = PretrainedPolicy(self.model_path, verified_path(self.root, trained, "adapters.safetensors"))
                    summary = evaluate(data["test"], base, adapted, attempt, heartbeat)
                    replay(data["test"], json.loads((attempt/"runs.json").read_text()), summary)
                    acceptance = run_acceptance(data["train"], attempt, heartbeat)
                    (attempt/"REPORT.md").write_text(report(summary, training, acceptance["summary"]))
            for path in attempt.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            directory_fd = os.open(attempt, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            artifacts = {p.name: {"path": str(p.relative_to(self.root)), "sha256": file_hash(p)} for p in sorted(attempt.iterdir())}
            receipt = {"stage": stage, "run_id": self.run_id, "artifacts": artifacts}
            self.store.complete(job, receipt)
        except LeaseLost:
            raise
        except Exception as exc:
            # Save the failure category, never provider error messages or secrets.
            atomic_json(attempt/"failure.json", {"error_type": type(exc).__name__, "stage": stage})
            self.store.fail(job, "stage_error:" + type(exc).__name__)
            raise
        return True

    def run(self):
        receipts = {}
        for stage in STAGES:
            job_id = self.enqueue(stage)
            for _ in range(3):
                job = self.store.get(job_id)
                if job["status"] == "succeeded":
                    break
                if job["status"] == "failed":
                    raise RuntimeError("Stage attempt budget exhausted")
                if not self.work_once(job_id):
                    raise RuntimeError("Stage leased by another worker; retry after completion or expiry")
            receipts[stage] = self.predecessor(stage)
        manifest = {"run_id": self.run_id, "protocol_sha256": file_hash(self.root/"protocol.json"),
                    "stages": receipts, "queue_metrics": self.store.metrics()}
        report_path = verified_path(self.root, receipts["evaluate"], "REPORT.md")
        atomic_text(self.root/"REPORT.md", report_path.read_text())
        manifest["report_sha256"] = file_hash(self.root/"REPORT.md")
        atomic_json(self.root/"manifest.json", manifest)
        audit(self.root)
        return manifest


def audit(out):
    root = Path(out)
    manifest = json.loads((root/"manifest.json").read_text())
    config = json.loads((root/"protocol.json").read_text())
    if file_hash(root/"protocol.json") != manifest["protocol_sha256"] or sha256_json(config) != manifest["run_id"]:
        raise ValueError("Protocol receipt mismatch")
    if file_hash(root/"REPORT.md") != manifest["report_sha256"]:
        raise ValueError("Report checksum mismatch")
    if set(manifest["stages"]) != set(STAGES):
        raise ValueError("Incomplete lifecycle")
    for stage, receipt in manifest["stages"].items():
        if receipt["stage"] != stage or receipt["run_id"] != manifest["run_id"]:
            raise ValueError("Cross-run artifact")
        for name in receipt["artifacts"]:
            verified_path(root, receipt, name)
    data = read_artifact(root, manifest["stages"]["prepare"], "dataset.json")
    validate_dataset(data)
    if sha256_json(data) != config["dataset_sha256"]:
        raise ValueError("Dataset receipt mismatch")
    evaluated = manifest["stages"]["evaluate"]
    summary = read_artifact(root, evaluated, "summary.json")
    result = replay(data["test"], read_artifact(root, evaluated, "runs.json"), summary)
    if config["format"] >= 2:
        acceptance = read_artifact(root, evaluated, "acceptance.json")
        result["acceptance"] = audit_acceptance(data["train"], acceptance)
        training = read_artifact(root, manifest["stages"]["train"], "training.json")
        expected_report = report(summary, training, result["acceptance"])
        if ((root/"REPORT.md").read_text() != expected_report or
                verified_path(root, evaluated, "REPORT.md").read_text() != expected_report):
            raise ValueError("Report differs from recorded outcomes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "audit"))
    parser.add_argument("--out", default="results/local_security_run")
    parser.add_argument("--backend", choices=("reference", "pretrained"), default="reference")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(args.out), sort_keys=True))
    else:
        manifest = Pipeline(args.out, args.backend, args.steps, args.model_path).run()
        print(json.dumps({"run_id": manifest["run_id"], "stages": list(manifest["stages"]),
                          "manifest": str(Path(args.out)/"manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
