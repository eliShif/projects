# Local model identity validation

Recorded on 1 November 2026 with Python 3.14.4. This directory contains new
validation evidence for the execution-time identity extension. It does not
replace the research results in `results/security_triage` or
`results/security_triage_v2`. The [check index](checks.json) and adjacent JSON
records contain exact argument arrays, exit codes, timestamps, elapsed times
and SHA-256 hashes of the retained output logs.

## Observed checks

| Check | Observed result | Retained output |
|---|---|---|
| Original security suite before edits | 44 tests passed, exit 0 | [baseline_security.txt](baseline_security.txt) |
| First focused identity suite | 17 tests passed, exit 0 | [identity_tests_attempt1.txt](identity_tests_attempt1.txt) |
| Final security suite | 63 tests passed, exit 0: original 44 plus 17 identity and two preservation tests | [security_tests.txt](security_tests.txt) |
| Source compilation | Exit 0, including the new tensor test module; compilation does not execute its tests | [syntax.txt](syntax.txt) |
| Local package installation | Exit 0, installed the existing project with its new identity module | [install.txt](install.txt) |
| Wheel build | Exit 0, produced `applied_ai_security_triage-0.1.0-py3-none-any.whl`, 69,666 bytes | [build.txt](build.txt) |
| Documented finite smoke | Exit 0: three successful stages, 72 replayed rows, 32-job/four-worker bounded acceptance and idempotent resume | [smoke.txt](smoke.txt) |
| Installed-package smoke (`-I`) | Exit 0, same finite lifecycle through the installed package | [installed_smoke.txt](installed_smoke.txt) |
| Preservation inventory | Exit 0: 148 baseline files; 141 unchanged, seven permitted edits, no missing/unexpected changes | [preservation.txt](preservation.txt) |
| Historical format-1 artifact audit | Exit 0, 72 replayed rows | [historical_v1_audit.txt](historical_v1_audit.txt) |
| Historical format-2 artifact audit | Exit 0, 72 replayed rows and retained service acceptance | [historical_v2_audit.txt](historical_v2_audit.txt) |
| Full inherited suite | Exit 1: 19 passing entries and 13 errors | [inherited_tests.txt](inherited_tests.txt) |
| Existing tensor suite | Exit 1, missing `torch` at module import; no tensor test bodies executed | [inherited_tensor_tests.txt](inherited_tensor_tests.txt) |
| New real tensor identity suite | Exit 1, missing `torch` at module import; no tensor test bodies executed | [real_tensor_identity.txt](real_tensor_identity.txt) |
| Existing ML dependency installation | Exit 1, no matching `numpy==2.3.5` distribution | [dependency_install.txt](dependency_install.txt) |
| Focused ML dependency installation | Exit 1, no matching `torch>=2.9,<3` distribution | [focused_ml_install.txt](focused_ml_install.txt) |

The 13 inherited errors are four prohibited socket creations and nine missing
dependency/import errors (three failed test-module imports and six NumPy-dependent
telemetry tests). Their 19 successes comprise 10 evidence tests and nine queue
tests. Baseline and final security counts overlap; they are not independent
research experiments. Neither a failed loader entry nor successful compilation
counts as an executed tensor test.

## What the executed regressions establish

The shared verifier rejects changed, missing and added relevant model inputs.
The tests independently change weights and configuration while preserving size
and modification time. Queued tests freeze the protocol and enqueue before
mutation, then exercise training, the base evaluation boundary, the interval
between evaluation arms, retry, reopening and expired-lease recovery. They assert
that the affected loader is not called and that the attempt contains only its
failure record, with no successful receipt or evaluation metrics. Original
protocol bytes remain unchanged. Separate corruption tests preserve the adapter
receipt check.

The actual `LocalLM` code is exercised with explicit Transformers loader doubles:
unchanged inputs reach both library calls, mutations stop them, and a change
during tokenizer loading stops the model call. Inventory tests bind shard-index
bytes and safe nested shard paths, and reject traversal, missing shards,
symlinks and implicit base-model redirects.

These offline tests establish Python guard/control-flow behavior. Their scalar
safetensors and prerequisite adapter receipts are labeled test fixtures. They
do not establish tensor loading, fitting or pretrained quality. The unexecuted
[real tensor suite](../../tests_ml_identity/test_pretrained_identity.py) specifies
positive complete-lifecycle controls, real tokenizer/model spies, mutation cases
for both evaluation arms, successful retries after restoration, genuine sharded
checkpoint loading and adapter-receipt rejection. It reuses the inherited tiny
randomly initialized Llama builder. Actual execution remains required.

## Preservation

The source revision supplied for this extension is
`1158b22985963249917a1f66caee1feac6f785b2`; the local application branch has no
commits. [baseline.json](baseline.json) records the supplied file inventory
before edits. The only changed existing files are `README.md`, `CODE_MAP.md`,
`EXERCISES.md`, `UPSTREAM.md`, `portfolio/local_lm.py`,
`portfolio/security_models.py` and `portfolio/security_pipeline.py`.
All 13 original test files and 81 original result files are byte-for-byte
unchanged. No source file was deleted. Original remote references and source/license
attribution remain in [UPSTREAM.md](../../UPSTREAM.md). Private Markdown archives
were not read or copied into this project.

Both historical security result directories pass artifact audit with the
extended code. That verifies retained artifact consistency, not fresh inference
or historical authenticity. The format-2 adapted reference still records 24/24
grounded-correct outcomes, tying its explicit rule control; no new model-quality
improvement is claimed. Earlier negative agent, LoRA, efficiency and telemetry
findings remain in their original reports.

## Reproduction

From the project root, use the setup in [README.md](../../README.md), then:

```sh
.venv/bin/python -m unittest discover -s tests_security -v
.venv/bin/python -m unittest discover -s tests_ml_identity -v
.venv/bin/python -m unittest discover -s tests_ml -v
.venv/bin/python -m pip wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m portfolio.security_smoke
.venv/bin/python -I -m portfolio.security_smoke
.venv/bin/python scripts/check_preservation.py
```

The tensor commands intentionally report missing dependencies as errors. Install
the existing `requirements.txt` in an environment that supplies the required
packages before rerunning them. No model download, hosted inference, credentials
or paid service is needed for the tiny test fixtures. The finite reference smoke
uses temporary output and does not replace saved experiments.

## Limits

- Real tensor identity tests, inherited adapter tests and a full pretrained-security
  experiment remain unverified in this environment. The dependency failures above
  are retained, not replaced with skipped-test success.
- Checks occur before loading; they do not lock files across library reads. A
  concurrent writer can still race after verification. Model directories and
  expected protocol identities must be trusted and stable. Hashes are not signatures.
- Custom tokenizer formats/external references are outside the supported
  self-contained checkpoint contract. Symlinks and implicit PEFT redirects fail
  explicitly. Repeated hash I/O and large-checkpoint performance were not measured.
- Reference smoke and in-memory HTTP acceptance establish bounded behavior, not
  network transport, process-crash durability, production capacity, live-incident
  accuracy or operational improvement. The original synthetic-template, one-seed
  and evidence-access limitations still apply.
