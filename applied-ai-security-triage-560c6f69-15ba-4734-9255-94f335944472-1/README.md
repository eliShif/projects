# Applied AI Security Triage

Queued ML experiments can load changed weights, tokenizers or configurations after their protocol is frozen, undermining reproducibility. Security-triage workflows also need findings tied to scoped evidence, explicit abstention and recoverable execution.

## What it does

Python/PyTorch research infrastructure for evidence-based security triage, combining cited findings and abstention with recoverable experiment queues, model-input integrity checks and artifact replay. Supported local Transformers/PEFT inputs are checked against frozen protocol hashes before training and evaluation.

Verify supported model inputs against frozen protocol hashes before training and evaluation, including delayed jobs, retries and lease recovery. Combine evidence-based investigation with validation-selected checkpoints, auditable artifacts and reproducible experiment execution.

## Solution and technologies

A Python pipeline uses SQLite to coordinate preparation, training and evaluation through idempotent jobs, renewable leases, bounded retries and fenced artifact publication. A bounded planner selects an investigation; an executor validates actor/time scope, retrieves versioned evidence and constructs cited findings or abstentions without response actions.

Protocol-bound SHA-256 identities reach training and both evaluation arms, with separate checks before tokenizer and model loading. The inventory supports shared-cache links, indexed and nested safetensors shards, advertised configuration/tokenizer versions, effective tokenizer overrides and tested optional PEFT bases and adapters. Compatible loading tolerates unused non-file template entries while retaining regular-file integrity checks. The closed-bundle helper applies a separate, stricter policy.

The reference backend trains a rank-2 update of a seeded linear classifier. The optional PyTorch/Transformers backend uses frozen base parameters, rank-4 query/value LoRA, masked answer-token loss and validation-selected safetensors checkpoints. Saved receipts, decisions, citations and manifests support artifact replay.

Python · PyTorch · Transformers · safetensors · SQLite · unittest · PEFT

## Quick start

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir --no-build-isolation --no-deps .
```

```sh
.venv/bin/python -m portfolio.security_smoke
```

## Results

Verified mutation cases stopped affected model loads without publishing successful receipts or evaluation metrics; unchanged and restored inputs completed successfully. Actual-loader regressions also confirmed compatibility with unused non-file chat-template children while preserving regular-file mutation rejection.

The latest recorded reference smoke completed three pipeline stages in three attempts and verified 72 evaluation rows through artifact replay. Its bounded service scenario exercised 32 jobs, four workers and 32 duplicate requests, recovered one expired lease, and rejected seven invalid requests, five adversarial fixture plans and three stale operations. The smoke used direct queue transport without sockets.

The completed, inherited 96-step synthetic security experiment remains separate: adapted and rule policies each achieved 24/24 grounded-correct outcomes and 9/9 correct abstentions; the random base achieved 6/24 with 18 rejected plans. Adaptation tied the rule control.

Other inherited experiments retain their mixed findings. LoRA reduced answer-token NLL from 6.153 to 0.521 while exact match remained 33.3%. Small-model retrieval achieved 19.44% exact match despite recovering the expected sources; direct and tool-agent policies achieved 0%. The typed Qwen system recorded 88/96 grounded-correct answers versus 42/96 for text RAG, with different evidence-access architectures; remaining errors included reversed operands and unnecessary abstentions. Host measurements retained slower INT8 vision latency, worse language-model perplexity and higher process RSS, alongside telemetry misses and maintenance false positives. These research measurements were not refreshed by the latest worker validation.

## Scope and limitations

Hashes require a trusted protocol and process. They provide neither signatures nor immutable snapshots or locks; concurrent changes can race with library reads. Large-checkpoint hashing overhead remains unmeasured.

Delayed-work protection is verified for the guarded Security Triage workflow. Direct LocalLM calls without expected identities remain unchecked; standalone training/policy calls capture identities on entry unless supplied an earlier map.

PEFT compatibility is bounded to the tested optional peft==0.21.2 configuration and supported local/cached bases and adapter formats. Arbitrary remote/custom-code tokenizer formats remain unverified. Strict-helper rejection rules do not describe the compatible inventory policy.

A full pretrained-security research experiment remains unrun. Tiny random-model tests establish integration only, and hosted-provider coverage remains limited to contract fixtures.

Research uses small synthetic datasets, shared task templates and limited seeds or entity clusters. Different evidence-access architectures constrain causal comparisons; initial agent timings were confounded by concurrent CPU work.

Artifact replay verifies consistency of saved inputs, decisions and scores; it does not authenticate historical inference or rerun it.

Bounded service checks do not establish process-crash durability, production throughput, live-incident accuracy or analyst-productivity improvement.

Collection completeness is caller asserted, temporal correlations do not establish malicious intent, and hostile-note fixtures do not establish general prompt-injection resistance or isolation of arbitrary Python plugins.

Verified CPU-host results do not establish native edge-device, power, Windows/NPU or cloud/TLS deployment performance. KQL and native cloud ingestion remain unverified.

Dependency installation reused an existing CPU environment and system packages. An independent clean-machine or complete offline dependency installation has not been established.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
