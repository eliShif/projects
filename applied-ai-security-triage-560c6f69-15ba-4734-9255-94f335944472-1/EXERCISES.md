# Investigation and experiment exercises

These exercises build familiarity through execution, inspection and changes.
Use a new output directory for each protocol or source change. The runner rejects
resuming an experiment against changed source or configuration.

## Trace model identity through delayed work

1. Run `.venv/bin/python -m unittest tests_security.test_model_identity -v`.
   Locate the test that enqueues training before replacing weight bytes without
   changing file size or modification time. Follow the failure from
   `train_pretrained` back to the queue's `failure.json` and empty result receipt.
2. Trace `protocol → Pipeline.work_once → PretrainedPolicy → LocalLM`. Identify
   where the expected map originates and where current file bytes are read.
   Explain why recapturing expected hashes inside a retry would accept the defect.
3. Follow the test that changes configuration between the base and adapted
   evaluation loads. Predict the loader call count and the files left in the
   failed attempt before running it.
4. Inspect `LoaderBoundaryTests`: its loaders are doubles. Contrast its evidence
   with `tests_ml_identity/test_pretrained_identity.py`, which uses the inherited
   randomly initialized tiny Llama fixture and real Transformers calls. After
   installing the ML dependencies, run
   `.venv/bin/python -m unittest discover -s tests_ml_identity -v`.
   A missing-dependency error is an unexecuted check, not a passed tensor test.
5. Read the saved protocol and compare it before and after a rejected retry.
   Restore the exact bytes in a disposable run and explain the three-attempt
   budget. Compare resuming an unchanged run with starting a new experiment.
6. Run `.venv/bin/python scripts/check_preservation.py`. Match each changed
   source file to this code map; inspect the separately retained historical
   reports and their limitations.
7. Read [the packaging failure before correction](results/model_identity_recovery_validation/packaging_regressions_before_fix.txt).
   Explain why installing the package changed `PKG-INFO` and `SOURCES.txt`.
   Run `.venv/bin/python -m unittest tests_security.test_preservation_packaging -v`,
   then build the wheel and repeat the preservation check. Identify why an
   altered `results/build/report.json` must still fail even though root `build/`
   is disposable. Add a regression for an additional saved-result path before
   expanding any exclusion rule.

For the combined security, adapter and real checkpoint checks, run
`.venv/bin/python scripts/test_model_identity.py`. Compare the current execution
logs with the retained missing-dependency failures; distinguish a fixture-based
integration check from a new pretrained-model research experiment.

Questions: Why do size/mtime checks miss some replacements? Why does the second
`from_pretrained` call need a new check after tokenizer loading? How do input
identities differ from the adapter's output receipt? Which concurrent-writer
race remains between hashing and reading? What would a signature establish that
a SHA-256 checksum alone cannot?

## Follow checkpoint aliases and tokenizer selection

1. Run `.venv/bin/python -m unittest tests_security.test_model_layouts tests_ml_identity.test_checkpoint_layouts -v`.
   Trace a model-directory alias through `_local_path`. Compare the logical
   protocol key with the resolved file whose bytes are hashed. Identify why the
   closed-bundle helper accepts an internal link but rejects an external link
   even with identical bytes. Compare that policy with the guarded workflow's
   trusted shared-cache loading in the standard-checkpoint compatibility tests.
2. Inspect `versioned_tokenizer` in `tests_ml_identity/test_checkpoint_layouts.py`.
   The default and selected tokenizer have different token IDs. Follow the
   actual `AutoTokenizer.from_pretrained` call that proves which file is consumed.
   Explain why the absent future version can remain absent without preventing
   a valid load, and why adding it after capture changes the frozen file set.
3. Follow the same-size tokenizer mutation through training and each evaluation
   arm. Count calls to the real loader spies and inspect failed attempts before
   restoration. Explain why neither `tokenizer.json` alone nor file size/mtime
   is sufficient to identify the input consumed by Transformers.
4. Run `.venv/bin/python scripts/check_layout_preservation.py`. Compare the
   repair's inventory with the original-source preservation check. Find the
   unchanged earlier test logs and distinguish their observations from the new
   [repair validation](results/model_identity_layout_repair/REPORT.md).

Questions: Does retargeting an internal link to identical bytes change content
identity? What does the directory-alias boundary permit? What happens if a link
is replaced after verification? Why are present unselected tokenizer versions
also hashed? What happens to a job after its third failed attempt?

Change exercises (use disposable fixture directories):

- Add a real loading regression for a selected versioned tokenizer behind an
  internal link, then retarget it between the two evaluation arms. Assert the
  affected loader is never called and the failed attempt has no metrics.
- Add a nested shard-directory escape regression. Verify rejection before the
  target file is opened, while preserving the existing successful shard case.
- Prototype a stable snapshot for loading and compare its disk/lifetime costs
  with repeated hashing. Retain the current race limitation until a test shows
  that writes during loading cannot affect the consumed bytes.

Change exercises (use separate disposable model/run directories):

- Add a regression for removing an indexed shard after enqueue. Assert rejection
  before any affected loader and no successful receipt or evaluation metrics.
- Add a supported tokenizer input format: identify every consumed file, update
  the inventory and add positive, mutation, deletion and addition cases. Do not
  broaden scanning to unrelated directories.
- Design a verified-snapshot loading variant to close the documented race.
  State its disk-space/lifetime tradeoffs before implementing it, then test a
  replacement during loading. Keep existing results and tests unchanged.
- Extend evaluation retry coverage with a lease expiry between the two arms.
  Preserve the original expected map and assert stale-worker publication fails.

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
