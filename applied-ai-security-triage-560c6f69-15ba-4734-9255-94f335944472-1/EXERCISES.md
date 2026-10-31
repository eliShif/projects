# Investigation and experiment exercises

These exercises build familiarity through execution, inspection and changes.
Use a new output directory for each protocol or source change. The runner rejects
resuming an experiment against changed source or configuration.

## Walk through a complete run

1. Follow README setup and run `.venv/bin/python -m portfolio.security_smoke`.
2. Run `.venv/bin/python -m portfolio.security_pipeline run --out results/practice`.
3. Open its `manifest.json`. Follow the three stage receipts to the prepared
   data, checkpoint history, raw evaluation rows and generated report.
4. Trace one `burst` and one `missing` case through `prepare_observation`,
   `policy_values`, `validate_plan` and `execute_investigation`.
5. Run `.venv/bin/python -m portfolio.security_pipeline audit --out results/practice`.
   Repeat `run` with the same settings and verify that attempt counts do not grow.
6. Inspect the evaluation receipt's `acceptance.json`. Match the two submissions
   per alert to one job ID. Find the job with two `claimed` events and identify
   the expired lease. Separate fixture outputs from the reference-policy rows.

Questions: Why is a `no_finding` conclusion invalid if the time window is
incomplete? Why does an escalated case still say that malicious intent is
unknown? Which fields may a model choose, and which are supplied by the executor?

## Explore failure boundaries

- Add an event for another actor at the same time. Verify that it cannot enter
  the alert's evidence or change the finding.
- Change one event ID while retaining its content, then compare that behavior
  with an exact duplicate and a conflicting duplicate. Explain the deduplication
  contract and why distinct IDs are not automatically the same observation.
- Add a planner fixture that returns an unseen citation, a different alert ID,
  or a response action. Extend the existing failure tests before changing the
  validator. Fixtures must retain their `test_fixture` label.
- Corrupt an artifact in a disposable run and observe audit failure. Explain
  what an attacker who can replace both an artifact and its receipt could do.
- Inspect `test_planner_cannot_rewrite_evidence_used_by_the_executor` and the
  [recorded failing regressions](results/security_validation_v2/regressions_before_fix.txt).
  Trace how sharing an evidence dictionary allowed a backend to alter findings
  and another evaluation arm. Explain why separate copies fix aliasing but do
  not sandbox a Python plugin.
- Send a duplicate `complete` key through the byte-stream HTTP harness and a
  duplicate `operation` key through a planner fixture. Predict the status, then
  run the boundary tests. Why can default JSON decoding hide ambiguity?

## Understand adaptation and evaluation

1. Read the finite-difference test for the reference policy. Derive the gradients
   for `B(Ax)` and explain why the base matrix must remain unchanged.
2. Run a new experiment with `--steps 1`, then with `--steps 96` in another
   directory. Compare validation histories, selected step and held-out failures.
   Do not choose a training budget using the held-out scores.
3. Add a second fixed training seed and report all seeds with the same protocol.
   Separate seed variability from shared synthetic-template limitations.
4. With the ML dependencies available, run `tests_ml` using the locally
   initialized tiny tensor model. Then use explicit local pretrained weights for
   a separate experiment. Explain why neither tensor-test results nor linear
   policy scores establish pretrained LLM quality.

Questions: Does lower validation NLL guarantee better investigation outcomes?
What does token-normalized candidate NLL favor? Why can a frozen correct rule
baseline outperform a learned planner? Which assumptions are shared across all
three arms and therefore cannot be isolated by this comparison?

## Extend the research question

Read the Greshake et al. indirect prompt-injection paper and Hu et al. LoRA paper
linked in the generated report. Map one threat assumption and one adaptation
assumption to specific functions and tests. Propose an additional experiment,
including a control, failure criterion and limitation, before implementing it.

Suggested change: add authorized-maintenance context with provenance. Distinguish
unknown intent from an independently verified approval record. Retain existing
false-positive observations and avoid converting a note string into permission.

## Exercise recovery

Use the injectable clock in `tests_security/test_pipeline.py` to expire a lease,
reclaim a job and attempt completion and renewal from the old worker. Explain
why an attempt directory can remain on disk without becoming an authoritative
artifact. Add a test for a crash immediately before publication.
Run `.venv/bin/python -m portfolio.security_acceptance --out results/practice_acceptance`
to exercise request validation, concurrency and recovery together. Change one
saved HTTP status and one saved trace independently, and verify that
`audit_acceptance` rejects each change. Which behavior still needs a live socket
test, and which performance questions cannot be answered by this scenario?

The current workflow has no response actions. Any future simulated response
exercise needs an enforced approval state transition bound to the exact action
and evidence digest, plus tests rejecting missing, stale and mismatched approvals.
A boolean review label is insufficient to authorize an action.
