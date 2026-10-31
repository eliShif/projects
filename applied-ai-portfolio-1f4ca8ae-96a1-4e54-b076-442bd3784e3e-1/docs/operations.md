# API, operations and threat model

## Intended user and acceptance criteria

A research engineer submits a question, gets one durable job ID, and can inspect
the answer plus the source evidence and execution trace. A security analyst can
request triage of synthetic events; the system never blocks accounts, modifies a
cluster or asserts that generated prose is a verified incident finding.

Success means: unauthorized submissions rejected; identical retries return the
same job; conflicting retries are rejected; one active worker owns a lease; stale
workers cannot overwrite a result; a crashed worker can be retried with a bounded
budget; real model output and evidence remain inspectable after completion.
See `results/service/acceptance.json` for the actual HTTP/model run and tests for
concurrency, crash, persistence and input-validation cases.

## Local run

The finite service harness uses an ephemeral database and an internal test token on loopback. It never reads a configured service credential. Run from the repository root:

```text
python3 -m portfolio.service_load --transport store --out results/queue_operations_new
python3 -m portfolio.service_load --transport http --out results/http_operations_new
```

The store command measures direct SQLite operations. The HTTP command verifies the existing routes with concurrent submissions and a deterministic receipt engine; it does not measure model inference. Both exit after the configured bounded workload. Use a new output directory each time.

For actual local-model integration, install the dependencies and provide already-downloaded model weights:

```text
.venv/bin/python -m portfolio.acceptance --model-path models/SmolLM2-135M-Instruct
```

This inherited acceptance command uses a local test server and exercises real research and synthetic triage generation. It requires a supported runtime and sockets. It was not rerun in the current restricted workspace; the inherited acceptance record remains separate evidence. The service's explicit worker entry point processes one pending job and exits. It is not a supervised long-running queue. GET `/jobs/<id>` and `/metrics` require the configured service token; `/health` returns only process liveness.

| Route | Contract | Outcomes |
|---|---|---|
| POST /jobs | JSON `{mode, question}`; mode research/security_triage; 1-2000 chars; Idempotency-Key up to 120 ASCII chars | 202 created, 200 duplicate, 409 conflict, 400 schema, 413 size, 415 content type, 401 auth |
| GET /jobs/id | Immutable result after success; attempt count, error category, trace | 200/404/401 |
| GET /metrics | Aggregate status counts and attempts; no prompts | 200/401 |
| GET /health | Process liveness only | 200 |

## Design decisions and concrete limits

- SQLite WAL and `BEGIN IMMEDIATE` serialize claims on a shared local database.
  Unique request keys enforce durable idempotency across threads/processes.
- Delivery is at least once after a lease expires. Fencing rejects stale writes,
  but external non-idempotent effects would require their own keys/outbox. This
  implementation has no external effects beyond local model computation.
- Leases expire after 300 seconds, with three attempts. No heartbeat: unusually
  slow jobs can be reclaimed, and need a heartbeat for deployment.
- SHA256 receipts detect accidental content changes if independently retained;
  they do not make a local administrator or modified database tamper-proof.
- A single token defines a single trusted workspace, not multi-tenant access.
  Tokens are environment values, never committed. Connection defaults to loopback.
- No TLS, quotas, reverse proxy, cloud identity, backups, remote queue, streaming,
  graceful worker supervisor, or production load testing is claimed. For public
  hosting these are required design work, not a command-line host change.
- Input schemas, body limits and an AST-based arithmetic DSL constrain actions.
  Prompt instructions alone do not guarantee prompt-injection resistance.
- Retrieved IDs are attached mechanically. Model claims/citations need review.
  Strict exact-match QA is not a complete groundedness evaluation.
- Synthetic telemetry can reveal code defects, but not real nation-state attacks.

## Debugging and deployment gate

Inspect job status, attempts, redacted error code and traces; reproduce a failing
payload in a fresh database; never record real credentials or confidential logs.
Use the fixture-only tests to isolate queue/HTTP bugs, then the explicit real-model
acceptance command to isolate model/environment failures. Restore the previous
code commit and database backup together if the schema changes. Current schema
is initialization-only: schema migration tooling has not been implemented.

The inherited README reports a historical 43-test GitHub Actions run. The current
extension has only the local execution receipts in VALIDATION.md; its revised
workflow has not run on GitHub. No cloud deployment or compliance certification
has been performed.
