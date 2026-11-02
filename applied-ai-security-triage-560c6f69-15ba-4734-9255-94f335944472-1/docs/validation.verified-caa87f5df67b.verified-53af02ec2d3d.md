# Validation record

The restricted builder validation was recorded on 31 October 2026 with
Python 3.14.4. [checks.json](../results/security_validation_v2/checks.json) records
its exact commands, exit codes, UTC timestamps and retained-log hashes. The
[recovery checkpoint record](validation-checkpoint.md) and original experiment
artifacts remain available separately. No hosted inference or deployment was run.

The earlier worker installation record confirms CPU PyTorch `2.15.0+cpu`
and local package installation. The checks below belong to the historical worker
record marked verified at `2026-10-31T15:44:35.381Z` in
[VALIDATION.md](../VALIDATION.md); they precede the model-identity recovery:

| Worker check | Observed outcome | Output SHA-256 |
|---|---|---|
| Security suite | 44 tests passed in 6.307 seconds; exit 0 | `e0aa8105ea20d30b9baf8bb3face029d7729da598d44512c7e01a9fc007fbec8` |
| Wheel build | Exit 0; `applied_ai_security_triage-0.1.0-py3-none-any.whl`, 65,893 bytes | `97c17ba8e5ad89923e4a23afd09162b876cb341229d4b1eef25895a6b8ab3d4e` |
| Finite reference smoke | Exit 0; three successful stages, 72 replayed rows and bounded acceptance for 32 jobs/four workers | `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92` |

The [model-identity recovery validation](../results/model_identity_recovery_validation/REPORT.md)
dated 1 November 2026 records Python 3.14.4, CPU PyTorch `2.15.0+cpu`,
Transformers `4.57.6` and safetensors `0.8.0`. Its offline dependency check found
the existing requirements satisfied in the recovered environment. The latest
supplied worker outputs establish:

| Worker check | Observed outcome | Output SHA-256 |
|---|---|---|
| Combined security, adapter and model-identity suite | 75 tests passed in 33.928 seconds; exit 0 | `7061c6ded9cf97801d3c8ec4d0b047b278a0154e78e80f4a2d28d2b2c05da8a4` |
| Wheel build | Exit 0; `applied_ai_security_triage-0.1.0-py3-none-any.whl`, 70,215 bytes | `a4f6cb777590aaf5224cf968080a23c1599063d5998c961073d301f96819b430` |
| Finite reference smoke | Exit 0; three successful stages, 72 replayed rows and bounded acceptance for 32 jobs/four workers; direct queue, no sockets | `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92` |

The 75 tests comprise 65 security tests, two inherited tensor tests and eight
real model-identity tests; these counts overlap with the separately reported
suite runs. The tensor checks execute real loading, one-step LoRA fitting and
both evaluation arms with tiny randomly initialized local checkpoints. They
establish integration behavior, not pretrained-model quality. The retained
96-step research experiment below is separate evidence; no full pretrained-security
research experiment was added.

The recovery report also records passing preservation checks, isolated
installed-package smoke and both historical artifact audits. Those audits replay
saved artifacts without rerunning inference. The full inherited suite returned
exit 1: 39 tests passed and four tests errored with `PermissionError` during
socket creation. Live HTTP transport remains unverified.

The restricted-builder dependency failures from 31 October 2026 remain in the
historical table below. The earlier identity-validation environment on
1 November 2026, also using Python 3.14.4, retained missing-dependency/import
failures in [its original record](../results/model_identity_validation/REPORT.md).
The successful recovery tensor executions supersede those import failures;
they do not remove the separately observed socket errors.

The following table preserves the historical restricted-builder outcomes from
31 October 2026; its dependency and socket failures are not current worker results.

| Check | Observed outcome | Log |
|---|---|---|
| Security suite | 44 tests passed | [tests.txt](../results/security_validation_v2/tests.txt) |
| Inherited evidence regression | 10 tests passed | [evidence_regression.txt](../results/security_validation_v2/evidence_regression.txt) |
| Local package installation | Passed without index access or runtime dependencies | [install.txt](../results/security_validation_v2/install.txt) |
| Wheel build | Passed; wheel saved in `dist/` | [build.txt](../results/security_validation_v2/build.txt) |
| Documented finite smoke | Passed; 72 evaluation rows, three completed stages, combined service acceptance and idempotent resume | [smoke.txt](../results/security_validation_v2/smoke.txt) |
| Installed-package smoke | Passed using Python `-I`, excluding the source working directory from imports | [installed_smoke.txt](../results/security_validation_v2/installed_smoke.txt) |
| Persistent experiment | Passed; 96 reference-policy updates and held-out evaluation | [experiment.txt](../results/security_validation_v2/experiment.txt) |
| Artifact audit | Passed; evaluation, service acceptance and report consistency verified | [audit.txt](../results/security_validation_v2/audit.txt) |
| ML dependency installation | Failed: no installable `numpy==2.3.5` distribution was supplied by the package index | [ml_dependency_install.txt](../results/security_validation_v2/ml_dependency_install.txt) |
| Full inherited suite | Failed: 19 passing entries and 13 errors | [inherited_suite.txt](../results/security_validation_v2/inherited_suite.txt) |
| Optional tensor suite | Failed importing `torch`; tensor test bodies did not execute | [optional_tensor_suite.txt](../results/security_validation_v2/optional_tensor_suite.txt) |

The full inherited suite's 13 errors comprise four socket-creation failures and
nine dependency/import errors (three test-module imports and six telemetry tests).
Its 19 successes are the same 10 evidence tests plus nine queue tests; these are
not an additional independent experiment. The optional tensor suite's one failed
loader entry does not count as an executed tensor test. Earlier checkpoint
installation attempts also recorded DNS failures; those logs are retained.

## Regressions observed before repair

Four newly added regression tests failed against the recovered checkpoint:
planner evidence mutation, contamination of another evaluation arm, duplicate
JSON operation keys, and duplicate request fields. The actual
[failing log](../results/security_validation_v2/regressions_before_fix.txt) and
[command record](../results/security_validation_v2/regressions_before_fix.json)
are retained. The final suite includes all four and passes after the repairs.
Additional checks reject nonfinite/deep JSON, short bodies, unsupported labels,
modified traces, changed rejected-plan outputs and reports inconsistent with
recorded counts, even when the report checksum is recomputed.

## Observed research result

The [format-2 report](../results/security_triage_v2/REPORT.md) and
[manifest](../results/security_triage_v2/manifest.json) describe the executed run.
The unadapted seeded linear policy achieved 6/24 grounded-correct outcomes with
18 rejected plans. Its trained rank-2 update achieved 24/24, tying the deterministic
rule. The adapted and rule arms each correctly abstained on 9/9 expected cases.
The paired comparison contains 18 improvements over the random base, zero
regressions and six unchanged cases.

Validation NLL decreased from 1.0978938206762408 at step zero to
0.01980738085773406 at the selected step 96. The 22 trainable parameters update a
frozen random linear base. These are reference-policy measurements, not LLM
results or calibrated confidence. The learned policy did not outperform the
rule baseline. Training/validation/test actors and events are disjoint, but
synthetic task templates are shared and the experiment uses one seed.

## Combined service result

The evaluation receipt contains `acceptance.json`: 64 byte-stream HTTP submissions
for 32 jobs, four workers, 33 claims and one expired-lease recovery after reopening
the database. Seven unauthorized, malformed or conflicting requests were rejected.
Five hostile planner fixtures were rejected; completion, failure and renewal from
the stale worker were all rejected. A mutating planner fixture did not change the
executor's evidence-derived finding. Saved request statuses, job results, traces,
raw fixture outputs and derived counts are verified during audit.

This is a bounded correctness scenario using in-memory HTTP streams and an
injected clock. It does not measure socket transport, latency, throughput,
production capacity, physical process termination or a recovery service-level
objective. Planner fixtures are contract tests, separate from model inference.
The reference policy omits note text by construction. Neither establishes broad
prompt-injection robustness; evidence copies do not sandbox Python plugins.

## Reproduction limits

- The separate pretrained-security research experiment remains unrun. On
  1 November 2026, the recovered ML environment passed both inherited tensor
  tests and all eight real model-identity tests with tiny randomly initialized
  local checkpoints. These checks establish loading, fitting and evaluation
  integration, not pretrained-model quality. Availability of pretrained weights
  for a full security research run is not established by these fixtures.
  Historical pretrained reports retain their original measurements and were
  not rerun during recovery.
- Collection completeness is a caller assertion. Findings are temporal
  correlations; malicious intent, live-incident accuracy and analyst productivity
  were not measured. The workflow has no response actions.
- Receipt hashes and replay check artifact consistency, not historical inference
  authenticity. An actor able to replace all inputs and receipts can forge a
  consistent record; the hashes are not signatures.
- Original negative findings remain: the small agent failed its usefulness gate,
  LoRA lowered NLL without improving exact match, INT8 worsened some latency,
  perplexity and memory results, and telemetry missed slow attacks and flagged
  some maintenance. These earlier experiments were not rerun here.
- The original structured-agent/RAG comparison uses different evidence access.
  Its small synthetic suite and six held-out entity clusters limit generalization;
  early baseline timing also included concurrent CPU work.
- KQL, cloud ingestion, production deployment, edge-device power and remote CI for
  this extension remain unexecuted. Source/license attribution is retained in
  [UPSTREAM.md](../UPSTREAM.md).
