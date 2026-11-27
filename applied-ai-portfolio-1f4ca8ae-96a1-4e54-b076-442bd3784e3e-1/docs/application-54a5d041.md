# LLM quality operations extension

This extension adds an offline toolkit for prompt-version comparison, windowed monitoring of LLM request streams, drift detection and span-style trace export. It is purely additive: `portfolio/llm_quality_ops.py` and `tests/test_llm_quality_ops.py` are new, and no existing source, test, result or documentation line was changed. The earlier test suite remains part of the documented discovery command.

## What was added

| Path | Purpose |
|---|---|
| `portfolio/llm_quality_ops.py` | Synthetic evidence cases, prompt versions, strict output parsing and grounded-correct scoring, a deterministic stub generator and an adapter for the existing local model, paired prompt comparison, windowed monitor (p50/p95/p99, proxy rates, PSI/KS drift), JSONL span export/load, synthetic request streams with injectable faults, a finite check and a CLI |
| `tests/test_llm_quality_ops.py` | Success and failure tests for each part (see below) |

Reused from the existing code: `portfolio.common.clustered_difference` (paired case-clustered bootstrap), `portfolio.common.wilson`, `portfolio.common.write_json`, `portfolio.measurement.distribution` (p50/p95), `RunRecord` (empty-output requirement, failure cleanup, manifest hashes) and `portfolio.local_lm.LocalLM` (only through the optional `--model-dir` path).

## Requirement coverage

| IDs | Behaviour | Implementation | Tests |
|---|---|---|---|
| R01 (LLM part) | Versioned prompts and strict JSON output contract (answer, citations, abstain); prompt versions compared on identical cases and seeds | `portfolio/llm_quality_ops.py` (`PROMPT_VERSIONS`, `render_messages`, `parse_output`, `score_case`, `evaluate_version`) | `tests/test_llm_quality_ops.py` (`ParsingAndScoringTests`, `PromptComparisonTests`) |
| R04 | Experiment side: paired comparisons with recorded protocol, manifest and traces. Monitoring side: replay of a recorded trace file through the monitor, with alerts | `portfolio/llm_quality_ops.py` (`run_demo`, `main`, `monitor`), `portfolio/measurement.py` (`RunRecord`) | `tests/test_llm_quality_ops.py` (`CommandLineTests`, `MonitorTests`) |
| R06 | Paired prompt-version A/B comparison on shared synthetic data with a clustered bootstrap interval and a three-way decision; identical versions and reversed order are tested | `portfolio/llm_quality_ops.py` (`compare_evaluations`, `compare_versions`), `portfolio/common.py` (`clustered_difference`) | `tests/test_llm_quality_ops.py` (`PromptComparisonTests`) |
| R07 | Windowed latency p50/p95 (existing `distribution`) and p99 (new `percentile`); alert when p95/p99 exceed the reference by a ratio; injected latency spike detected, same-distribution control quiet, localisation to the degraded window | `portfolio/llm_quality_ops.py` (`percentile`, `latency_stats`, `monitor`), `portfolio/measurement.py` (`distribution`) | `tests/test_llm_quality_ops.py` (`StatisticsTests`, `MonitorTests`) |
| R08 | Hallucination proxies per window: invalid-citation rate, ungrounded-answer rate, format-failure rate, abstention rate, using a Wilson-interval rule against the reference rate | `portfolio/llm_quality_ops.py` (`score_case`, `monitor`), `portfolio/common.py` (`wilson`) | `tests/test_llm_quality_ops.py` (`ParsingAndScoringTests`, `MonitorTests`) |
| R09 | PSI and KS drift on input length, output length and retrieval score against a reference window, with thresholds, a no-alert control and unshifted-feature check; windows with too little data never alert | `portfolio/llm_quality_ops.py` (`psi`, `ks_statistic`, `ks_critical`, `monitor`) | `tests/test_llm_quality_ops.py` (`StatisticsTests` incl. SciPy cross-check of KS, `MonitorTests`) |
| R19 | Span-style JSONL trace export (`trace_id`, `span_id`, parent, start, duration, token counts, scoring attributes), strict loading with line-numbered errors, replay through the monitor | `portfolio/llm_quality_ops.py` (`request_spans`, `export_traces_jsonl`, `load_traces_jsonl`, `validate_spans`) | `tests/test_llm_quality_ops.py` (`TraceTests`, `CommandLineTests`) |

## Behaviour that is tested to fail

Unknown prompt versions, empty seeds or cases, fewer than two cases for the bootstrap, unpaired evaluations, malformed generator output (non-string text, negative or NaN seconds, missing fields), malformed model output (not JSON, wrong schema, abstain/answer mismatch), invalid monitor records, too-small reference, empty current stream, invalid configuration, trace files with bad JSON, orphan spans, duplicate ids or bad status, refusal to overwrite an existing trace file or run directory, and CLI exit codes (0 healthy, 3 alert on replay, 2 invalid input, 1 failed check).

## Run

See the README section "LLM quality operations extension" for the exact setup and commands. The finite check is `.venv/bin/python -m portfolio.llm_quality_ops --seed 17`; it prints its own computed values and no result files are shipped.

## Limitations

- All data are synthetic. The default generator is a rule-based stub, not a language model. With it the comparison results verify the framework (pairing, scoring, bootstrap, decisions), not the quality of any prompt; its latency is simulated from token counts.
- Prompt-version differences found on 24 synthetic cases and 3 repeats are not evidence about real prompts. No image-generation prompting is involved; the extension covers LLM text prompts only.
- Monitoring is offline replay over synthetic or recorded traces. It is not live monitoring, has no connection to any serving system and makes no production-scale claim. The windows use a logical clock in the evaluation traces.
- Hallucination figures are proxies: ids not present in the supplied passages, answers absent from the cited text, and abstention rate. They do not detect fluent but wrong claims that cite valid passages.
- Alert thresholds (ratio 1.5, margin 0.05, PSI 0.25, KS alpha 1e-4) are defaults chosen for the synthetic streams; they have not been calibrated against real traffic. Fault scenarios are single injected shifts of fixed size, so detection of them does not estimate real-world detection rates or delay.
- The `--model-dir` path (real local model) was not exercised by this extension's tests, which use a fake object with the same interface.
- Trace export follows a span-style layout but is not an OpenTelemetry exporter and no observability product is used.
- Existing portfolio limitations (single-seed model integrations, SQLite prototype scope, no production deployment evidence) are unchanged.
