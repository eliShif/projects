# Checkpoint layout repair validation

Executed locally on 1 November 2026. The existing Security Triage implementation
now accepts safe local checkpoint symlinks and binds versioned tokenizer files.
The affected combined suite passed **95 tests**. Package installation, wheel
build, finite source/installed smoke, both historical audits and preservation
checks also passed. The separate full inherited suite retained four environment
errors at socket creation; it is not reported as a successful full-suite run.

## Change and evidence

Only `portfolio/model_identity.py` changed among supplied implementation modules.
It preserves logical artifact names, confines links to the resolved checkpoint
and hashes all present tokenizer versions advertised in `fast_tokenizer_files`.
The existing per-load checks in `LocalLM`, `PretrainedPolicy`, `train_pretrained`
and `Pipeline` remain in place. The queue, reference/LoRA training, evaluation,
adapter verification and artifact replay code were retained without edits in
this repair.

The new tests reuse the original tiny randomly initialized Llama fixture. They
execute real Transformers/safetensors loading, one-step LoRA training, both
evaluation arms and artifact replay. No pretrained weights were downloaded and
no pretrained research or model-quality experiment was added.

- [Inventory tests](../../tests_security/test_model_layouts.py) exercise directory
  aliases, internal file/directory links, indexed shards, chat templates, changed
  targets, escapes, dangling links, loops, versioned-tokenizer mutations and
  file-set changes. Escape checks assert that outside fixture files are not opened.
- [Real loading tests](../../tests_ml_identity/test_checkpoint_layouts.py) exercise
  standalone APIs without an expected map, queued work after reopening, actual
  linked-shard loading, retarget rejection and successful restoration. A selected
  versioned tokenizer has different token IDs from the default file, proving
  selection through the real library. Valid same-size/same-mtime mutations fail
  before training, base evaluation, adapted evaluation and retry/resume loads.
  A mutation inside tokenizer loading also prevents the causal-LM load.
- The unchanged [original real identity tests](../../tests_ml_identity/test_pretrained_identity.py)
  still verify weight/configuration mutations at all corresponding boundaries,
  unchanged-input success, sharding and adapter-receipt corruption.
- Failed attempts are checked for no successful receipt and no training/evaluation
  metrics. Retry exhaustion and exact-byte restoration use the existing queue.

Implementation/test mapping for R02–R06, P01, P02 and D01 is in the
[current identity contract](../../docs/model-input-layouts.md#executable-coverage).

## Observed executions

All commands ran from the project root. Each row links the raw output; its
matching JSON record contains the exact argument array, timestamp, exit code,
elapsed time and output SHA-256. [checks.json](checks.json) indexes these records.
Test counts below overlap; the 76 security cases and 18 layout cases are included
in the combined 95, and the final two preservation checks repeat existing cases.

| Check | Observed outcome | Raw output |
|---|---|---|
| Initial layout regressions before repair | Exit 1; 18 tests, 17 failure entries and 14 error entries, including subtests | [regressions_before_fix.txt](regressions_before_fix.txt) |
| Layout regressions after repair | Exit 0; 18 tests passed | [layout_regressions.txt](layout_regressions.txt) |
| Original security discovery command, with added tests | Exit 0; 76 tests passed | [security_tests.txt](security_tests.txt) |
| Combined security, inherited tensor and real identity suites | Exit 0; 95 tests passed | [combined_tests.txt](combined_tests.txt) |
| Full inherited suite | Exit 1; 43 tests, 39 passed and four socket-creation errors | [inherited_tests.txt](inherited_tests.txt) |
| Virtual-environment setup | Exit 0 | [venv_setup.txt](venv_setup.txt) |
| Packaging dependency and existing ML requirements | Exit 0; requirements already installed, checked offline | [packaging_dependency.txt](packaging_dependency.txt), [dependency_install.txt](dependency_install.txt) |
| Local package installation | Exit 0 | [install.txt](install.txt) |
| Wheel build | Exit 0 | [build.txt](build.txt) |
| Finite source and isolated installed-package smoke | Both exit 0 | [smoke.txt](smoke.txt), [installed_smoke.txt](installed_smoke.txt) |
| Historical format-1 and format-2 audits | Both exit 0; 72 rows replayed each, without rerunning inference | [historical_v1_audit.txt](historical_v1_audit.txt), [historical_v2_audit.txt](historical_v2_audit.txt) |
| Original authoritative inventory check | Exit 0; no missing files or unexpected changes | [original_preservation.txt](original_preservation.txt) |
| Final repair inventory check | Exit 0; 230 of 235 files unchanged, five permitted edits | [repair_preservation_final.txt](repair_preservation_final.txt) |
| Final preservation regressions | Exit 0; two tests passed | [preservation_regressions_final.txt](preservation_regressions_final.txt) |

The initial regression log reproduces rejection of safe symlink layouts and the
versioned-tokenizer identity bypass. Its failure counts include subtests and
subsequent checks affected by an earlier failed mutation assertion; they are not
counts of independent defects. The tokenizer subtests were made independent by
resetting their fixture for each boundary before the passing run. The original
test files were never edited.

The repair preservation wrapper now includes `tests_ml_identity` in its test-file
count, unlike the older checker's narrower counter. The initial
[repair check](repair_preservation.txt) reported 16 tests under that older counting
rule; the final check reports all 17 supplied test files. Both checks compared
all 235 file hashes. The two preservation tests were rerun after this reporting
correction; no runtime model code changed after the successful combined run.

The four inherited errors are `PermissionError: [Errno 1] Operation not permitted`
in `HttpTests.setUp` while creating sockets. All other inherited cases passed,
including adapter, evidence, provider-contract, retrieval/statistics, telemetry
and direct queue tests. Provider contracts use an in-memory mock transport.
The socket failures remain in the log; in-memory HTTP checks do not establish
live transport behavior.

The smoke runs each completed three stages in three attempts and replayed 72
arm/case rows. Their bounded acceptance scenario records 32 jobs, four workers,
32 duplicate requests and one recovered lease. These are observed integration
counts, not model-quality or capacity measurements.

## Reproduction

The recovered environment contains the required packages. These are the exact
offline installation, combined test, build and finite run commands used here:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir "setuptools>=77"
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir -r requirements.txt
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir --no-build-isolation --no-deps .
.venv/bin/python scripts/test_model_identity.py
.venv/bin/python -m pip --isolated --disable-pip-version-check wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m portfolio.security_smoke
.venv/bin/python -I -m portfolio.security_smoke
.venv/bin/python scripts/check_preservation.py
.venv/bin/python scripts/check_layout_preservation.py
```

The [README](../../README.md#setup) also documents installation from a package
index for clean environments. Offline installation here verified existing
dependencies; it does not demonstrate downloading them on a clean host.
[environment.json](environment.json) records Python 3.14.4, CPU PyTorch 2.15.0+cpu,
Transformers 4.57.6, tokenizers 0.22.2 and safetensors 0.8.0. Validation used offline
Hub/Transformers flags and disabled implicit Hub tokens and telemetry. No
credentials, hosted inference or model downloads were needed.

## Preservation

The supplied pinned revision is `1158b22985963249917a1f66caee1feac6f785b2`.
The local repository has no HEAD, so this is supplied source provenance rather
than a locally resolved Git object. No remote was read or changed.

[repair_baseline.json](repair_baseline.json) was captured before editing. The
final comparison preserves all **17 supplied test files** and **156 earlier
result files** byte-for-byte. Of 235 supplied files, 230 are unchanged. The only
edits are `portfolio/model_identity.py`, `README.md`, `CODE_MAP.md`, `EXERCISES.md`
and `UPSTREAM.md`. New tests, the supplemental contract, this validation directory
and the repair preservation wrapper are additions. The old source inventory
also passes its original change policy.

The preceding identity and packaging validation records remain in
`results/model_identity_validation` and `results/model_identity_recovery_validation`.
Historical research artifacts, failed installation records, socket failures,
negative findings and source/license attribution remain accessible separately.
Original Markdown is archived privately outside this workspace and was not
accessed or copied. [implementation_sha256.json](implementation_sha256.json)
identifies the implementation and test files for this validation.

## Remaining limitations

- Hashes check content against a trusted frozen protocol. They do not sign
  provenance or lock files. Concurrent writes or link retargets can still race
  between verification and library reads. Hashing overhead is not benchmarked.
- Safe local links are supported. Artifact links escaping the resolved checkpoint,
  implicit PEFT redirects and arbitrary external/custom tokenizer formats remain
  unsupported. Identical bytes at a different safe local target have the same
  content identity; inode or symlink-text identity is not part of the protocol.
- Real tensor execution uses tiny random fixtures. A full pretrained-security
  research experiment remains unrun, and this repair establishes no model-quality
  improvement or operational security benefit.
- Live sockets remain blocked in this environment. Same-process queue recovery
  and in-memory HTTP tests do not establish process-crash durability, transport
  behavior, throughput or production capacity.
- Earlier one-seed synthetic results, the adapted reference policy's tie with
  its rule control, architecture/access confounds, missed attacks, maintenance
  false positives and negative efficiency findings are unchanged. Historical
  research experiments were audited where indicated, not rerun.
