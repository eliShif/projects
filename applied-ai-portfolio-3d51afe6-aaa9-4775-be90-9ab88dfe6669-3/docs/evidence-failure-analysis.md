# Held-out evidence-agent findings

The completed frozen test contains 48 authored questions, two generations per
system and six held-out service entities. The same Qwen3-4B GGUF model is used
for both systems; the architecture and evidence access differ.

| Measure | Text RAG | Typed agent |
|---|---:|---:|
| Correct with expected evidence | 42/96 (43.75%) | 88/96 (91.67%) |
| Correct abstentions | 18/36 (50%) | 36/36 (100%) |
| Invalid/failed outputs | 0 | 0 |
| Median generation latency | 14.18 s | 6.65 s |

The entity-bootstrap improvement interval is 37.5 to 58.3 percentage points.
Only six entity clusters are available, so uncertainty is not broadly calibrated.
The serialized bootstrap labels `questions=6` and `repeats_per_question=1`
represent six entity-level means, not the original question or repeat counts.

## Remaining errors

Six of the typed agent's eight incorrect outputs reverse the operands in
"Subtract B from A". Two include the verb "Express" in the service name, causing
an unnecessary safe abstention. These are four distinct questions repeated twice.
Dimension checks and exact evidence provenance cannot detect a valid but
semantically incorrect plan. They preserve facts, not interpretation correctness.

Do not tune on these held-out questions and rebrand the resulting score as an
unseen test. A next iteration should use development paraphrases and a newly
frozen test with independently authored wording and more domains.

## Recovery and provenance

The run was interrupted after 130 of 192 model outputs. Recovery preserved those
130 records exactly and generated only the remaining 62. The frozen source,
model checksum, corpus, task identities, seeds, prompts and scoring were checked
before continuation. The scheduling-only recovery and original checkpoint are
recorded in `results/evidence_agent_test/recovery/`. No samples were replaced.
Latency pools both execution segments; interruption and cache state limit direct
hardware-performance interpretation.

## Scope

This is a bounded synthetic numeric-research task. Typed tools access structured
records, whereas RAG receives six retrieved text records. The result supports
this system design within that task, not a model ranking, universal prompt-
injection protection, a frontier-model claim or production readiness.
