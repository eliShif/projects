# Source provenance

Original source: https://github.com/eliShif/applied-ai-portfolio/tree/f06d214375d45572c92dae6e7229d5f1d652f05b
This directory preserves the upstream implementation and adds the changes described in README.md.

The pinned source was supplied in this workspace. No upstream remote or branch
was modified. Original experiment artifacts remain under `results/`; their
reported measurements are distinct from the new security experiment.

The extension reuses the versioned evidence store, telemetry normalization,
SQLite job queue, local model loader, answer-token loss and PyTorch LoRA layer.
Changes add structured alert investigation, queue-backed experiment stages,
lease renewal, artifact receipts, replay and focused tests. Optional imports
allow the standard-library path to run without installing the ML stack.

The supplied workspace also contained a recovery checkpoint for this extension.
Its `results/security_triage` artifacts and `results/security_validation` records
are retained. The resumed implementation adds separate planner/evidence copies,
strict JSON decoding, rejected-plan replay and combined HTTP/queue recovery
acceptance. The fresh format-2 experiment is in `results/security_triage_v2`;
its verification logs are in `results/security_validation_v2`. Four regression
failures observed before the boundary fixes are preserved with those logs.
The upstream `results/source_manifest.json` and `results/current_source_manifest.json`
describe the supplied historical experiment source. The format-2 protocol records
the hashes of the current implementation used for the new run.

The supplied snapshot contains no top-level license file. No new license is
assigned to the upstream source. Existing source links and model/dataset license
references in the documentation are retained. Model weights are not included.


# Extension source provenance

Original source: https://github.com/eliShif/projects/tree/1158b22985963249917a1f66caee1feac6f785b2/applied-ai-security-triage-560c6f69-15ba-4734-9255-94f335944472-1
This directory preserves the upstream implementation and adds the changes described in README.md.

## Local model identity extension

The supplied Security Triage revision is
`1158b22985963249917a1f66caee1feac6f785b2`. This workspace's application branch
has no local commits; the revision is supplied provenance, not a locally resolved
Git object. The initial pre-publication inventory contained 148 entries,
including five generated `.egg-info` entries. Those totals belong to the
1 November 2026 validation record. The [normalized authoritative inventory](results/model_identity_validation/baseline.json)
now contains 141 entries, excluding private worker output and generated packaging
files while retaining every authoritative source entry. No remote was changed or fetched.
The original Markdown archive is managed privately outside this workspace; it
was not read or copied into the public project.

The extension changes only three existing Python modules (`security_pipeline`,
`security_models`, `local_lm`) and four root guides. The shared identity module,
regression tests, preservation checker and current validation files are additions.
Original tests, fixtures, data, research results, source manifests and historical
validation reports and logs are retained byte-for-byte. The [preservation check](scripts/check_preservation.py)
enforces this boundary while excluding generated root packaging/dependency output.
The published baseline excludes private worker output and generated packaging
entries while retaining every authoritative source entry. The original
implementation-attempt observations live in `results/model_identity_validation`;
recovery verification is separate in `results/model_identity_recovery_validation`.
Neither replaces prior research results.

The current capability checks SHA-256 before queued local pretrained loads.
The [earlier validation record](results/model_identity_validation/REPORT.md) retains
the real tensor tests blocked by unavailable dependencies. The
[current validation record](results/model_identity_recovery_validation/REPORT.md)
records successful real tensor execution in the recovered environment and the
targeted correction for generated metadata being mistaken for protected source.
The prior one-seed synthetic results, negative findings, architecture
confounds, unrun pretrained-security experiment and deployment limits still apply.

## Checkpoint layout repair

The next focused repair changes `portfolio/model_identity.py` to accept safe
local directory/artifact links and include versioned fast-tokenizer inputs.
Existing loading boundaries, training, evaluation, queue/recovery logic and
adapter-receipt verification remain unchanged. New tests use the existing tiny
Llama fixture and real Transformers semantics; no model weights are downloaded.

The [repair baseline](results/model_identity_layout_repair/repair_baseline.json)
records 235 supplied files before the 1 November 2026 layout repair, including
17 test files and 156 previous result files. At that stage, the
[repair preservation checker](scripts/check_layout_preservation.py) permitted
only the verifier and four root guides to change. The current operator
compatibility follow-up uses a nine-file allowlist and excludes the ephemeral
project-root `.report-cpu-constraints.txt` from source preservation. Historical
tests and results remain protected. Follow-up verification is recorded separately
in `results/operator_identity_collision/verification.json`.
New observations, including the failing regressions before correction, are in
`results/model_identity_layout_repair`. The original pinned source, privately
managed Markdown archive, source/license attribution and earlier validation
directories remain separate and unchanged. No remote was accessed or modified.
