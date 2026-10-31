# Recovered checkpoint validation

These records were supplied with the recovery checkpoint. See [current validation](validation.md) for the resumed implementation and its new execution logs.

The recorded commands, exit codes and logs are in
[checks.json](../results/security_validation/checks.json). Runs used Python
3.14.4 in the restricted workspace. No hosted inference, credentials or paid APIs
were used. The upstream source and remote were not changed outside this workspace.

| Check | Observed result | Record |
|---|---|---|
| Local package installation | Succeeded with `--no-build-isolation --no-deps`; no third-party runtime dependencies needed for the reference path | [install.log](../results/security_validation/install.log) |
| New offline suite | 33 tests passed | [tests.log](../results/security_validation/tests.log) |
| Inherited typed-evidence regression | 10 tests passed | [evidence_regression.log](../results/security_validation/evidence_regression.log) |
| Wheel build | Succeeded; wheel written to `dist/` | [build.log](../results/security_validation/build.log) |
| Finite smoke | Three completed stages, 72 replayed rows, idempotent resume and rejected stale completion | [smoke.log](../results/security_validation/smoke.log) |
| Persistent reference experiment | Completed 96 updates and held-out evaluation | [experiment.log](../results/security_validation/experiment.log) |
| Artifact audit | Verified all 72 arm/case rows and recomputed findings and metrics | [audit.log](../results/security_validation/audit.log) |
| ML dependency installation | Failed; required distributions unavailable through the restricted package index. Initial attempts also reported DNS resolution failures | [ml_dependency_install.log](../results/security_validation/ml_dependency_install.log) |
| Full inherited suite | Failed: 19 passing entries, 13 errors; four socket-creation errors and nine dependency/import errors | [inherited_suite.log](../results/security_validation/inherited_suite.log) |
| Optional tensor integration suite | Failed importing `torch`; tensor test bodies did not execute | [optional_tensor_suite.log](../results/security_validation/optional_tensor_suite.log) |

The 10 evidence regressions also occur in the inherited suite and are not an
additional independent test population. The inherited suite's 19 successes are
those 10 evidence tests plus nine queue tests. Its dependency failures include
three test-module import failures and six telemetry tests requiring NumPy.
The four socket tests failed with `PermissionError` before starting a server.
Byte-stream HTTP checks exercise parsing, authorization, request validation and
worker results, but do not establish live network behavior.

## Observed reference experiment

The [report](../results/security_triage/REPORT.md) is generated from the retained
raw rows. The unadapted seeded linear policy achieved 6/24 grounded-correct
outcomes and produced 18 rejected plans. The trained rank-2 update achieved
24/24, tying the deterministic rule's 24/24. Both the adapted and rule arms
correctly abstained in 9/9 expected abstention cases. The paired comparison has
18 improvements over the random base, zero regressions and six unchanged cases.

Validation NLL decreased from 1.0978938206762408 at step zero to
0.01980738085773406 at the selected step 96. There are 22 trainable reference
parameters, and the frozen base digest is retained with the training history.
These are measured outcomes for a small linear policy on synthetic templates;
they are not LLM results, calibrated confidence or evidence of operational value.
The learned policy did not outperform the explicit rule baseline.

## Reproduction boundaries

- The new pretrained security experiment remains unrun: no ML dependencies or
  local pretrained weights were available. The provided optional tensor tests
  are not counted as passed checks.
- The reference path needs no model download and was also smoke-tested from the
  installed package outside the source-package directory.
- The bounded concurrency test submits 32 structured jobs twice and processes
  them with four worker threads. It checks one result per idempotency key and
  correct evidence-derived outcomes; it does not measure throughput or latency.
- Lease expiry, stale completion/failure/renewal, retry exhaustion and recovery
  use an injectable clock. They test state transitions without claiming a
  production crash/recovery service-level objective.
- Malformed inputs, cross-scope plans, forbidden tools, hostile notes, missing
  evidence, modified labels and corrupted reports are negative tests. Broader
  adversarial-model behavior and real telemetry need separate experiments.
- Source/data/model digests make runs inspectable. They are not signatures, and
  artifact replay does not rerun inference or prove artifact authenticity.
- Historical experiment reports retain their original measurements and limits.
  They were not rerun here. The extension's CI definition has not run remotely.
