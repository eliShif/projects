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

The finite offline workflow uses no server or token and exercises the same queue
and structured triage engine as the service:

```sh
python3 -m portfolio.security_smoke
python3 -m portfolio.security_pipeline run --out results/operations_run
python3 -m portfolio.security_pipeline audit --out results/operations_run
python3 -m portfolio.security_acceptance --out results/operations_acceptance
```

For a live service, `make_server(store, token)` and `SecurityTriageEngine` provide
the transport and structured worker interfaces. The older service CLI supports
local model workers and reads a deployment-supplied token. A live socket run of
this extension was blocked in the restricted builder environment on
31 October 2026 (Python 3.14.4). No successful worker live-socket rerun is
supplied; the worker smoke reports `direct queue; no sockets`. Byte-stream HTTP
contract results are recorded separately in [validation](validation.md).

The worker processes **one** pending job and exits; rerun it for another job.
This makes failures and model startup costs visible. It is not a continuously
running scalable production queue. GET `/jobs/<returned-id>` and `/metrics` need
the bearer token. GET `/health` is unauthenticated and returns no user data.

| Route | Contract | Outcomes |
|---|---|---|
| POST /jobs | JSON `{mode, question}` or structured `{mode: security_triage, alert}`; Idempotency-Key up to 120 ASCII chars; internal pipeline stages excluded | 202 created, 200 duplicate, 409 conflict, 400 schema/ambiguous length, 413 size, 415 content type, 401 auth |
| GET /jobs/id | Immutable result after success; attempt count, error category, trace | 200/404/401 |
| GET /metrics | Aggregate status counts and attempts; no prompts | 200/401 |
| GET /health | Process liveness only | 200 |

## Design decisions and concrete limits

- SQLite WAL and `BEGIN IMMEDIATE` serialize claims on a shared local database.
  Unique request keys enforce durable idempotency across threads/processes.
- Delivery is at least once after a lease expires. Fencing rejects stale writes,
  but external non-idempotent effects would require their own keys/outbox. This
  implementation has no external effects beyond local model computation.
- Leases expire after 300 seconds, with three attempts. Experiment workers renew
  leases at training/evaluation boundaries. The legacy one-shot prose worker
  does not renew; a single model operation exceeding its lease can be reclaimed.
- SHA256 receipts detect accidental content changes if independently retained;
  they do not make a local administrator or modified database tamper-proof.
- A single token defines a single trusted workspace, not multi-tenant access.
  Tokens are environment values, never committed. Connection defaults to loopback.
- No TLS, quotas, reverse proxy, cloud identity, backups, remote queue, streaming,
  graceful worker supervisor, or production load testing is claimed. For public
  hosting these are required design work, not a command-line host change.
- Input schemas, body limits and an AST-based arithmetic DSL constrain actions.
  Prompt instructions alone do not guarantee prompt-injection resistance.
- JSON rejects duplicate keys, nonfinite numbers and nesting beyond 32 levels;
  incomplete request bodies are rejected. Planner copies are separate from
  executor evidence and comparison arms. These controls prevent ambiguous input
  interpretation and data aliasing; they do not sandbox in-process plugins.
- The legacy prose route attaches retrieved IDs mechanically and needs claim
  review. Structured alerts bind procedures to an actor/time scope and derive
  findings/citations from events; this is still a synthetic temporal policy,
  not a determination of malicious intent or a complete groundedness evaluation.
- Synthetic telemetry can reveal code defects, but not real nation-state attacks.

The format-2 experiment's `acceptance.json` combines request rejection, 32 jobs
submitted twice, four worker threads, one expired lease, database reopen and
stale-write rejection. Raw handler outcomes and stored job traces are audited.
The scenario uses byte streams and an injected clock; it supplies a bounded
correctness check, without a throughput or recovery-time claim.

## Debugging and deployment gate

Inspect job status, attempts, redacted error code and traces; reproduce a failing
payload in a fresh database; never record real credentials or confidential logs.
Use the fixture-only tests to isolate queue/HTTP bugs, then the explicit real-model
acceptance command to isolate model/environment failures. Restore the previous
code commit and database backup together if the schema changes. Current schema
is initialization-only: schema migration tooling has not been implemented.

The extension's CI definition has not been run on GitHub. The upstream README
links a historical 43-test CI run; that record does not validate these changes.
No cloud deployment or compliance certification has been performed.
