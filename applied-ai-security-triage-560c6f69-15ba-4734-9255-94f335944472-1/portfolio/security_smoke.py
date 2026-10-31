"""Finite offline check of the documented run, resume, and audit path."""
import json
from pathlib import Path
import tempfile

from .reliable_service import JobStore, work_once, LeaseLost
from .security_pipeline import Pipeline, audit, read_artifact
from .security_triage import SecurityTriageEngine, RulePlanner


def main():
    # The temporary run is independent of committed results and needs no token,
    # model download, socket, network service or indefinitely running process.
    with tempfile.TemporaryDirectory(prefix="triage-smoke-", dir=Path.cwd()) as out:
        pipeline = Pipeline(out, steps=24)
        manifest = pipeline.run()
        before = pipeline.store.metrics()
        if Pipeline(out, steps=24).run() != manifest or pipeline.store.metrics() != before:
            raise RuntimeError("Resume repeated completed work")
        data = read_artifact(out, manifest["stages"]["prepare"], "dataset.json")
        now = [100.]
        store = JobStore(Path(out)/"triage.db", clock=lambda: now[0])
        case = data["test"][7]
        payload = {"mode": "security_triage", "alert": case["alert"]}
        job_id, created = store.enqueue("smoke", payload)
        if not created or store.enqueue("smoke", payload) != (job_id, False):
            raise RuntimeError("Idempotency failed")
        stale = store.claim(lease_seconds=1)
        now[0] += 2
        work_once(store, SecurityTriageEngine(RulePlanner(), case["events"]))
        try:
            store.complete(stale, {"invalid": True})
        except LeaseLost:
            pass
        else:
            raise RuntimeError("Stale completion accepted")
        result = store.get(job_id)
        if result["result"]["structured_answer"]["disposition"] != "escalate":
            raise RuntimeError("Hostile-note case did not retain the evidence-based finding")
        print(json.dumps({"status": "ok", "backend": "reference_linear_policy_not_llm",
                          "verification": audit(out), "pipeline": before,
                          "triage_attempts": result["attempts"], "transport": "direct queue; no sockets"}, sort_keys=True))


if __name__ == "__main__":
    main()
