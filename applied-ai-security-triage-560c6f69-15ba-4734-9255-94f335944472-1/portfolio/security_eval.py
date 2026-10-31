"""Paired, replayable investigation evaluation with raw decisions retained."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .common import sha256_json, write_json
from .security_data import observations
from .security_triage import investigate, RulePlanner


def score(case, observation, result):
    expected = case["expected"]["disposition"]
    correct = result.get("status") == "completed" and result.get("disposition") == expected
    required = set() if expected == "abstain" else {
        *[e["id"] for e in observation["events"]],
        *[e["source_id"] for e in observation["policy_evidence"]]}
    cites = result.get("citations", [])
    grounded = correct and isinstance(cites, list) and all(isinstance(c, str) for c in cites) and set(cites) == required
    return {"correct": bool(correct), "grounded_correct": bool(grounded),
            "failed": result.get("status") != "completed",
            "correct_abstention": expected == "abstain" and bool(grounded)}


def summarize(rows):
    result = {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm]
        count = len(selected)
        families = sorted({r["family"] for r in selected})
        result[arm] = {"cases": count, "correct": sum(r["score"]["correct"] for r in selected),
                       "grounded_correct": sum(r["score"]["grounded_correct"] for r in selected),
                       "failed": sum(r["score"]["failed"] for r in selected),
                       "correct_abstentions": sum(r["score"]["correct_abstention"] for r in selected),
                       "abstention_cases": sum(r["expected"]["disposition"] == "abstain" for r in selected),
                       "failure_reasons": dict(Counter(r["result"]["reason"] for r in selected if not r["score"]["correct"])),
                       "by_family": {f: {"cases": sum(r["family"] == f for r in selected),
                                          "grounded_correct": sum(r["family"] == f and r["score"]["grounded_correct"] for r in selected)} for f in families}}
    paired = {}
    arms = sorted(result)
    if "base" in arms and "adapted" in arms:
        a = {r["case_id"]: r["score"]["grounded_correct"] for r in rows if r["arm"] == "base"}
        b = {r["case_id"]: r["score"]["grounded_correct"] for r in rows if r["arm"] == "adapted"}
        paired = {"improved": sum(b[k] and not a[k] for k in a),
                  "regressed": sum(a[k] and not b[k] for k in a),
                  "unchanged": sum(a[k] == b[k] for k in a)}
    return {"arms": result, "paired_base_to_adapted": paired}


def evaluate(cases, base, adapted, out, heartbeat=lambda: None):
    obs = observations(cases)
    rows = []
    for case, observation in zip(cases, obs):
        for arm, planner in (("rule", RulePlanner()), ("base", base), ("adapted", adapted)):
            heartbeat()
            try:
                result = investigate(observation, planner)
            except Exception as exc:
                # Record failed calls; never replace them with deterministic answers.
                result = {"status": "failed", "disposition": "abstain", "citations": [],
                          "reason": "backend_error", "error_type": type(exc).__name__, "response_actions": []}
            row = {"case_id": case["id"], "family": case["family"], "arm": arm,
                   "observation_sha256": sha256_json(observation), "expected": case["expected"],
                   "result": result, "score": score(case, observation, result)}
            rows.append(row)
    summary = summarize(rows)
    write_json(Path(out)/"runs.json", rows)
    write_json(Path(out)/"summary.json", summary)
    return summary


def replay(cases, rows, summary):
    """Check row completeness, information parity, scores and executed findings."""
    lookup = {c["id"]: (c, o) for c, o in zip(cases, observations(cases))}
    expected_pairs = {(c, arm) for c in lookup for arm in ("rule", "base", "adapted")}
    pairs = [(r["case_id"], r["arm"]) for r in rows]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError("Missing, duplicate or unexpected evaluation row")
    for row in rows:
        case, observation = lookup[row["case_id"]]
        if row["observation_sha256"] != sha256_json(observation) or row["expected"] != case["expected"] or row["family"] != case["family"]:
            raise ValueError("Evaluation input mismatch")
        result = row["result"]
        if result.get("response_actions") != []:
            raise ValueError("Unexpected response action")
        if row["score"] != score(case, observation, result):
            raise ValueError("Score mismatch")
        if result.get("status") == "completed" or result.get("reason") == "invalid_plan":
            class RecordedPlanner:
                def plan(self, _):
                    return result["model_response"]
            executed = investigate(observation, RecordedPlanner())
            if result != executed:
                raise ValueError("Finding differs from replay")
        elif not (result.get("status") == "failed" and result.get("reason") == "backend_error" and
                  result.get("disposition") == "abstain" and result.get("citations") == [] and
                  isinstance(result.get("error_type"), str) and result["error_type"]):
            raise ValueError("Unrecognized failure record")
    if summarize(rows) != summary:
        raise ValueError("Summary mismatch")
    return {"verified_rows": len(rows), "verification": "artifact replay; does not rerun inference"}


def report(summary, training, acceptance=None):
    lines = ["# Read-only security triage experiment", "",
             "## Question and method", "",
             "Can a low-rank policy update improve procedure selection while an evidence executor restricts findings and actions?",
             "Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (2023),",
             "[arXiv:2302.12173](https://arxiv.org/abs/2302.12173), motivates treating retrieved strings as untrusted data.",
             "This experiment maps that threat to hostile alert notes, fixed actor/time scope, an operation allowlist, and deterministic finding construction.",
             "Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), [arXiv:2106.09685](https://arxiv.org/abs/2106.09685),",
             "motivates freezing the base and fitting a low-rank update. The reference backend applies that algebra to a small classifier.", "",
             f"Backend: `{training['backend']}`. Selected step: {training['selected_step']} of {training['steps']}.",
             "Training and validation use separate actors/events; the selected checkpoint is fixed before held-out evaluation.",
             "All three arms receive the same serialized, label-free observation and the same evidence executor.", "",
             "## Observed held-out results", "",
             "| Arm | Grounded correct / cases | Correct abstentions / expected | Failed plans/calls |",
             "|---|---:|---:|---:|"]
    for arm, values in summary["arms"].items():
        lines.append(f"| {arm} | {values['grounded_correct']} / {values['cases']} | {values['correct_abstentions']} / {values['abstention_cases']} | {values['failed']} |")
    lines += ["", "Paired changes: `" + json.dumps(summary["paired_base_to_adapted"], sort_keys=True) + "`.",
              "Raw decisions, candidate scores, failures, citations and traces are retained in `runs.json`; `summary.json` contains family breakdowns.", "",
              "## Limits and follow-up", "",
              "- Synthetic event templates are shared across splits. Entity separation does not establish generalization to new attacks.",
              "- The standard-library reference is a seeded linear classifier with a rank-2 update, not an LLM or a pretrained model. Its features omit note text by construction.",
              "- The optional pretrained path uses actual local weights and PyTorch LoRA, with candidate answer-token NLL selection. It requires a separate explicit run; reference results do not verify that path.",
              "- Candidate NLL is length-normalized and is not calibrated confidence. Constrained candidate selection does not evaluate free-form generation.",
              "- The rule baseline is supplied the task policy and is a strong control. A learned policy may tie or underperform it; every such outcome is retained.",
              "- Findings are temporal correlations. Maintenance intent, live threat detection, analyst productivity and production capacity were not measured.",
              "- Hostile-note and invalid-plan checks cover these boundaries only; they do not establish general prompt-injection resistance.",
              "- Collection completeness is an explicit caller assertion. Missing or falsified upstream events cannot be detected from a completeness flag.",
              "- Single seed and small held-out set; paired counts are descriptive and no significance or operational improvement is claimed.",
              "- Replay validates artifact consistency, not historical inference authenticity. Hashes are integrity checks, not signatures.", ""]
    if acceptance is not None:
        lines += ["## Bounded service acceptance", "",
                  f"{acceptance['jobs']} jobs were submitted twice through the HTTP handler and processed by {acceptance['workers']} worker threads.",
                  f"The run recovered {acceptance['recovered_jobs']} expired lease after reopening the database ({acceptance['claims']} total claims).",
                  f"It rejected {acceptance['rejected_requests']} invalid/unauthorized/conflicting requests, {acceptance['rejected_fixture_plans']} adversarial fixture plans, and {acceptance['stale_writes_rejected']} stale completion/failure/renewal attempts.",
                  "A mutating planner fixture could not alter the executor's evidence-derived finding.",
                  "Observed HTTP statuses, raw fixture outputs, job traces and stored findings are retained in `acceptance.json` and checked by audit.",
                  "This scenario uses in-memory HTTP byte streams and an injected clock. It measures bounded correctness, not socket behavior, throughput or recovery latency.",
                  "The planner fixtures are contract tests, not model inference or a prompt-injection benchmark. Evidence copies prevent aliasing; they are not a sandbox for arbitrary Python plugins.", ""]
    return "\n".join(lines)
