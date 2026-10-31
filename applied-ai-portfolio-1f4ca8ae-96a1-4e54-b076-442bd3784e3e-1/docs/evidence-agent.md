# Evidence-preserving research agent

The original 135M baseline could not reliably choose tools. This iteration changes
both the inference engine and the system design: a 4B instruction model produces
a typed plan, and deterministic tools retrieve versioned facts and execute it.
It is a bounded research agent for numeric service policies, not a general agent.

## Why the separation matters

The planner receives the question and an operation schema. It does not receive
untrusted record notes, source values, gold answers or a task ID. The executor
can only look up allowlisted fields and perform dimension-checked arithmetic.
Every result includes the exact source ID, revision, field, operand and operation.
Newer records supersede older records; equally current conflicting records cause
abstention. Unknown service names are not silently fuzzy-matched.

This prevents note-based injection at the data/planning boundary by construction.
It does not prove universal resistance to malicious user prompts, arbitrary tool
ecosystems, free-form evidence or compromised records. Model interpretation of a
question can still be wrong even when all arithmetic is valid.

## Reproduce

Install `requirements.txt` and `requirements-gguf.txt`. Download the optional
model file from its publisher (about 2.4 GiB); no API key or paid inference is used:

- Base model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Quantization: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
- Pinned repository revision: `a06e946bb6b655725eafa393f4a9745d460374c9`
- File: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
- Model and quantization publishers identify the license as Apache-2.0.
- Runtime documentation: https://github.com/abetlen/llama-cpp-python

```sh
.venv/bin/python -m portfolio.evidence_eval --model /path/to/model.gguf \
  --split dev --repeats 1 --limit 8 --out results/evidence_agent_dev
.venv/bin/python -m portfolio.evidence_eval --model /path/to/model.gguf \
  --split test --repeats 2 --out results/evidence_agent_test
```

The runner writes protocol hashes before inference and records every actual
response, seed, token count, time and validation failure. Both systems use the same
model and schema-constrained JSON. RAG receives the top six retrieved text records;
the typed agent queries the structured corpus directly. This intentionally
compares two architectures, not isolated model intelligence. No model can see
expected answers during inference. Source hashes identify the exact run revision.

## Evaluation design

Development and test service identities are disjoint. The test has 48 questions
across eight families: lookup, unit conversion, difference, sum, ratio, missing
field, absent service and conflicting current evidence. Superseded revisions and
injection notes appear in the corpus. Two repetitions per system examine sampling
variability. Paired bootstrap resamples whole service entities so related facts
and repeated samples are not treated as independent observations.

Predeclared gate: at least 85% correct answers with expected evidence, at least
90% correct abstentions and no malformed/failed outputs. Passing this gate is
limited to this synthetic suite; six entity clusters give limited statistical
precision. Task families are shared across splits. Real documents, broader
domains and external benchmarks remain further evaluation work.

The original failed run remains in `results/agent_eval/`; it is not overwritten
or merged with this experiment to manufacture an improvement score.

## Service integration

`portfolio.reliable_service --worker --gguf-model /path/to/model.gguf` uses the
typed engine for research jobs against the explicit development corpus. Security
triage remains a separate read-only summarization flow with analyst review.
The durable queue, authorization, idempotency keys and fenced leases are shared.

## Completed test

The held-out run is complete: 88/96 grounded correct for the typed agent versus
42/96 for RAG. See [results](../results/evidence_agent_test/REPORT.md) and
[failure analysis](evidence-failure-analysis.md), including interruption recovery.
