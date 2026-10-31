# Local SQLite load

24 unique jobs, 4 threads, 3 submissions per key.
Transport: direct SQLite. HTTP was not exercised by this run.

Enqueue p50: 1.482 ms; p95: 18.820 ms; 1065.03 operations/s.
Unique enqueue rate: 355.01 jobs/s. Separate receipt-worker drain: 253.06 jobs/s.
Expected content conflicts: 24. Unexpected errors: 0. All recorded invariants passed.

Recommendation: retain transactional idempotency and fencing. Verify HTTP and actual inference under representative arrival rates before setting capacity targets.

## Limits

- Direct SQLite calls only; sockets and HTTP are not exercised.
- Receipt processing is a deterministic queue probe, not model inference or quality evaluation.
- Closed-loop threads and separate enqueue/drain phases do not measure saturation or end-to-end client latency.
- Short local measurements do not establish production availability, scalability or customer impact.
- Lease faults use a controlled clock, not process crashes or physical power loss.
