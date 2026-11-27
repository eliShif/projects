# applied-ai-portfolio: LLM quality operations extension

Changes to an LLM prompt, model or input traffic can silently degrade structured-output validity, grounding, latency or input distribution. Judging them needs paired comparisons on identical cases, a strict output contract, windowed monitoring against a reference period, and traces that can be replayed. This extension builds those pieces as an offline, reproducible toolkit on synthetic data.

## What it does

Offline Python toolkit for LLM quality operations: paired prompt-version comparison under a strict JSON output contract, windowed monitoring of request streams (p50/p95/p99 latency, hallucination-proxy rates, PSI/KS drift), and span-style JSONL trace export with replay. In the seed-17 run on synthetic data with a rule-based stub generator, the comparison picked the expected prompt version, the control stream raised no alert, all 8 injected faults raised alerts, and the exported traces reloaded identically. Tests and the smoke command passed on the worker host. Scope is offline and synthetic. The real-model path is untested, thresholds are uncalibrated, and hallucination figures are proxies.

1) Define three prompt versions with a strict JSON output contract (answer, citations, abstain). Score each output as grounded-correct and compare versions pairwise on identical cases and seeds, with a case-clustered bootstrap and a three-way decision (a_better, b_better, inconclusive). 2) Monitor a request-record stream in windows against a reference period. Track p50/p95/p99 latency, invalid-citation, ungrounded-answer, format-failure and abstention rates as hallucination proxies, and PSI/KS drift on input length, output length and retrieval score. Each alert records its observed value, reference value and threshold. 3) Export span-style JSONL traces with strict loading, and replay a recorded trace file through the monitor from a CLI with defined exit codes. 4) Check all of this with a finite, seeded command and tests that include injected faults, a no-alert control and invalid input.

## Solution and technologies

The code is one new module, portfolio/llm_quality_ops.py, and one new test file, tests/test_llm_quality_ops.py. Existing code is reused: common.clustered_difference and wilson, measurement.distribution, RunRecord and write_json, and local_lm.LocalLM through an optional adapter. Data flow: make_cases + prompt version + generator → parsed and scored rows → outcome matrix → paired comparison. Separately, synthetic_records (or a loaded trace file) → monitor → windows and alerts. Spans from the comparison run are exported and reloaded as a round-trip check. The default generator is a deterministic rule-based stub whose latency is simulated from token counts. The --model-dir option swaps in a local model. The CLI supports --out (recorded run: protocol, summary, REPORT.md, traces.jsonl, manifest, in a new empty directory) and --replay FILE --reference-count N (exit 0 no alert, 3 alert, 2 invalid input). The finite check exits nonzero if any check fails. Technologies: Python, NumPy, unittest, JSONL span-style tracing, PSI/KS drift detection and bootstrap A/B comparison. SciPy is used only to cross-check KS in a test.

Python · NumPy · SciPy · unittest · JSONL span-style tracing · PSI/KS drift detection · bootstrap A/B comparison

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir 'torch>=2.9,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m portfolio.llm_quality_ops --seed 17
```

## Results

Latest worker validation (Python 3.14, CPU torch 2.15.0+cpu environment): the full discovery command passed with exit 0 (181 tests, 158.182 s), and the smoke command `.venv/bin/python -m portfolio.llm_quality_ops --seed 17` exited 0 with all_passed true and all 14 internal checks true. Scope of the smoke run: seed 17, 24 synthetic questions × 3 repeats = 72 requests per prompt version, generator = rule-based stub with simulated latency. Bootstrap CIs resample the 24 questions as clusters.

Prompt-version comparison (grounded-correct rate, stub generator; the abstention effect holds by construction of the stub):
- v0_freeform: 0.0 (format error rate 1.0). v1_json: 0.583. v2_json_abstain: 0.833.
- v1_json → v2_json_abstain: mean difference +0.25, 95% CI [0.083, 0.417], decision b_better. The gain comes from unanswerable-question abstention (0.0 → 1.0). Answerable grounded-correct stayed 0.778 and invalid-citation rate stayed 0.167.
- The reversed order gave a_better with difference -0.25, and a version compared with itself gave inconclusive with CI [0, 0].
- v0_freeform → v1_json: +0.583, CI [0.417, 0.75].
- Latency p50 under the stub was 77.5 / 95.5 / 103.5 ms for v0 / v1 / v2. These values are simulated from token counts and are not measurements.

Monitoring on seeded synthetic streams (3 windows per scenario, uncalibrated default thresholds):
- The same-distribution control gave status ok with no alerts.
- Latency spike: p95 and p99 alerts in all 3 windows.
- Invalid-citation surge, ungrounded surge, format failures, abstention surge: one proxy alert per window in each case.
- Abstention collapse: alerts with direction down in all 3 windows.
- Input-length drift and retrieval-score drift: PSI and KS alerts in all 3 windows.
- 432 spans were exported and the reloaded records were identical.

These are single fixed-size injected shifts, so they show the detectors work on the designed faults and say nothing about real-world detection rates or delay. The earlier results/store_load* reports in the repository concern the SQLite queue, not this module, and remain unchanged in their own reports.

## Scope and limitations

All data are synthetic. The default generator is a deterministic rule-based stub (not a language model) whose behavior is keyed to phrases in the system prompt, and its latency is simulated from token counts. The comparison results verify the framework only and say nothing about the quality of any prompt or model. The abstain-prompt advantage holds by construction.

The real-model path (--model-dir via portfolio.local_lm.LocalLM) is implemented but not exercised. The tests use a fake object with the same interface.

Monitoring is offline replay on synthetic or recorded traces, not live monitoring and not production scale. Alert thresholds (ratio 1.5, margin 0.05, PSI 0.25, KS alpha 1e-4) are uncalibrated defaults. Fault scenarios are single fixed-size shifts, with no gradual drift and no multiple-testing control, so they do not estimate real detection rates or delay.

Hallucination figures are proxies (citation ids absent from the passages, answers absent from the cited text, abstention rate, format failures). They do not detect fluent wrong claims that cite valid passages.

Trace export is self-built and span-style, not an OpenTelemetry exporter, and no observability product is used. SciPy appears only in a test cross-check of KS.

Image-generation prompting and voice/video are not covered.

Test and smoke results come from a workspace that is not pinned to a published commit. Fixed seeds and statistical thresholds passed in the tested runs, but only those runs are evidence.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
