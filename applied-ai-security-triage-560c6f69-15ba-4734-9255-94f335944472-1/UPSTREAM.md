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
