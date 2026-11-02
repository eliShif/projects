# Model identity recovery validation

Validated on 1 November 2026. The existing execution-time SHA-256 extension is
retained, and the packaging-related preservation failure is corrected. The
combined security, adapter and model-identity command passes **75 tests**,
including real loading and LoRA fitting with tiny local safetensors fixtures.
The separate inherited suite passes 39 tests and reports four socket-creation
errors. No full pretrained-security research experiment was run.

## Focused correction

Package installation regenerates `applied_ai_security_triage.egg-info/PKG-INFO`
and `SOURCES.txt`. The preservation checker previously treated these outputs as
original source and failed after a successful installation. The
[reproduced failure](packaging_regressions_before_fix.txt) includes that existing
assertion and a new regression for regenerated or missing build metadata.

[check_preservation.py](../../scripts/check_preservation.py) now reports generated
project-root `.egg-info`, `.venv`, `build` and `dist` entries separately from the
protected inventory. The supplied baseline hashes are unchanged. Exclusions do
not apply to similarly named directories inside source, documentation or results.
[Packaging regressions](../../tests_security/test_preservation_packaging.py)
verify both the permitted metadata changes and rejection of changed or missing
source, test and saved-result files. All four preservation tests pass after the
correction; the complete security suite also passes after installation and build.

The model-identity implementation, existing passing tests and historical outputs
were retained from the recovery checkpoint. The added
[combined test command](../../scripts/test_model_identity.py) runs security,
inherited adapter and real model-identity tests together. Missing ML dependencies
remain errors, with no skip or substitute backend.

## Observed checks

Every command has a JSON record containing its arguments, UTC start, measured
elapsed time, exit code and log SHA-256. [checks.json](checks.json) indexes these
records; [environment.json](environment.json) records the installed versions.
Python was 3.14.4, CPU PyTorch 2.15.0+cpu, Transformers 4.57.6 and safetensors 0.8.0.

| Check | Observed outcome | Log |
|---|---|---|
| Preservation before correction | Exit 1: two failing assertions, including the reported installed-metadata failure | [Before correction](packaging_regressions_before_fix.txt) |
| Preservation regressions after correction | Exit 0: four tests passed | [Preservation tests](packaging_regressions.txt) |
| Existing security command after installation/build | Exit 0: 65 tests passed | [Security suite](security_tests_after_build.txt) |
| Real tensor identity suite | Exit 0: eight tests passed | [Identity integration](real_tensor_identity.txt) |
| Inherited tensor suite | Exit 0: two tests passed | [Tensor tests](inherited_tensor_tests.txt) |
| Combined documented test command | Exit 0: 75 tests passed, comprising the 65 + 2 + 8 above | [Combined suite](combined_tests.txt) |
| Full inherited suite | Exit 1: 39 tests passed, four live-socket setup errors | [Inherited suite](inherited_tests.txt) |
| Virtual-environment setup | Exit 0: existing environment retained | [Setup](venv_setup.txt) |
| Existing ML dependency installation, offline | Exit 0: all requirements already satisfied in the recovered environment | [Dependencies](dependency_install.txt) |
| Local package installation | Exit 0: built and installed the current package | [Install](install.txt) |
| Wheel build | Exit 0: `applied_ai_security_triage-0.1.0-py3-none-any.whl`, 70,076 bytes | [Build](build.txt) |
| Source and isolated installed smoke | Both exit 0: three successful pipeline stages, 72 replayed rows and bounded service acceptance | [Source](smoke.txt), [installed](installed_smoke.txt) |
| Original inventory after build | Exit 0: 143 protected files, 136 unchanged and seven permitted edits; five generated entries reported separately | [Preservation](preservation_after_build.txt) |
| Historical format-1 and format-2 artifact audits | Both exit 0, each replaying 72 saved rows; format-2 service acceptance remains consistent | [Format 1](historical_v1_audit.txt), [format 2](historical_v2_audit.txt) |

The suite counts overlap as stated; they are not independent research trials.
The four inherited errors are `PermissionError` while constructing a socket in
`tests/test_service.py`, before those HTTP tests can run. The earlier missing-ML
dependency and socket failures remain unchanged in
[the previous validation directory](../model_identity_validation/REPORT.md).
The current real tensor executions supersede the earlier import-only failure for
these fixture tests, while live network transport remains unverified. The tensor
logs also retain a Transformers `torch_dtype` deprecation warning.

## Identity behavior exercised

The protocol captures expected hashes before enqueue. A shared verifier re-reads
the weight/configuration bytes and relevant file set before training and each
evaluation arm. `LocalLM` also checks immediately before each tokenizer/model
`from_pretrained` call. The pipeline supplies the original map for delayed work,
reopened runs, retries and reclaimed leases; current hashes never replace it.

[Offline regressions](../../tests_security/test_model_identity.py) independently
change safetensors weights and configuration while preserving size and mtime.
They cover queued training, both evaluation boundaries, retry, reopening and
lease expiry. They assert no affected loader call and no successful receipt or
evaluation metrics for the failed attempt. Missing inputs, relevant additions,
shard indexes, referenced shards, traversal and symlinks have separate checks.

The eight [real tensor tests](../../tests_ml_identity/test_pretrained_identity.py)
reuse the inherited tiny randomly initialized Llama builder and real Transformers
loaders. They exercise one-step LoRA fitting, both evaluation loads, a complete
72-row lifecycle/replay, unchanged-input resume, exact-byte restoration and
successful retry. Mutation tests stop the actual affected loader, including
between evaluation arms and between tokenizer/model loading. A genuine 16-shard
checkpoint loads successfully; changed shard or index bytes are rejected.
The adapter output receipt remains independently verified before loading.

These checkpoints are test fixtures. Their successful training/loading checks
are integration evidence; they do not measure pretrained model quality or an
improvement over the retained reference-policy experiment. Implementation/test
paths for every covered requirement are listed in
[the identity contract](../../docs/model-identity.md#executable-evidence).

## Preservation

The supplied source revision is `1158b22985963249917a1f66caee1feac6f785b2`.
The workspace has no locally resolvable Git commit; that identifier is pinned
source provenance. No remote was fetched or altered. Source/license links remain
in [UPSTREAM.md](../../UPSTREAM.md); private archives were not read or copied.

The authoritative [148-entry baseline](../model_identity_validation/baseline.json)
is retained byte-for-byte. Excluding its five generated metadata entries leaves
143 protected files. The seven permitted earlier-source changes are the four
root guides and `portfolio/local_lm.py`, `portfolio/security_models.py` and
`portfolio/security_pipeline.py`. All 13 original test files and all 81 original
result files are unchanged. No original source file is missing.

A separate [recovery inventory](recovery_baseline.json) captured 191 supplied
non-generated files before this correction. Its
[comparison](recovery_preservation.json) finds 185 unchanged files and six edits:
the four root guides, `docs/model-identity.md` and `scripts/check_preservation.py`.
All previously added identity code, tests and validation records are unchanged.
New tests, the combined runner and this directory are additions.

Both historical security experiments pass artifact audit, which checks saved
consistency rather than rerunning inference. The adapted reference's retained
24/24 result still ties the rule control; the random base remains 6/24. Earlier
LoRA exact-match stagnation, poor small-agent accuracy, efficiency tradeoffs,
telemetry misses and false positives remain in their original reports.

## Reproduce

Follow [README setup](../../README.md#setup), then run from the project root:

```sh
.venv/bin/python scripts/test_model_identity.py
.venv/bin/python -m pip wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m unittest discover -s tests_security -v
.venv/bin/python scripts/check_preservation.py
.venv/bin/python -m portfolio.security_smoke
.venv/bin/python -I -m portfolio.security_smoke
```

The full inherited suite remains available through
`.venv/bin/python -m unittest discover -s tests -v`; it needs socket permission
for its four live HTTP tests. No credentials, model download or paid service is
needed for the local tensor fixtures. The installed smoke requires the local
package installation step; both smoke commands remove their temporary runs.

## Remaining limitations

- Hashes are integrity checks, not signatures or locks. A concurrent writer can
  race after verification and before/during a library read. The protocol and
  self-contained model directory must be trusted and stable.
- The supported inventory covers standard local Llama-style checkpoints, safe
  indexed shards and documented tokenizer/configuration inputs. Arbitrary
  external tokenizer references, symlinks and implicit PEFT redirects are not
  supported. Large-checkpoint hash I/O has not been benchmarked.
- No full pretrained-security experiment, generalization gain or operational
  security improvement was measured. The retained synthetic reference experiment
  uses one seed and shared templates and ties the explicit rule baseline.
- In-memory HTTP acceptance and same-process SQLite recovery do not establish
  live socket behavior, process-crash durability, throughput or production
  capacity. The four socket tests remain blocked by this environment.
- Prior research experiments were not rerun. Their model-access confounds,
  negative results, unrun deployment/power experiments and measurement limits
  remain in the original documentation.
