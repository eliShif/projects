# Learning walkthrough and change exercises

These prompts are for a future reader to run, inspect and explain. Completion or
personal mastery is not asserted by this repository.

## Walkthrough

1. Run `python3 -m portfolio.research_lab --out results/research_lab` and read
   `protocol.json` before `summary.json`. Identify the question, synthetic data
   source, six entities, split rule and model selection metric.
2. Open `cases.json`. Count labels and entities in each split. Compare your count
   with the quality section of `summary.json`. Explain why questions for one
   service should not cross splits.
3. Trace `TextBaseline.fit` and `predict`. Explain the add-one term, why log
   probabilities are summed, and what the majority baseline predicts.
4. Read the validation and test confusion matrices. Locate one error in the
   `failures` list if present; if the selected model has none, inspect a baseline
   error. Explain what the case says and why the baseline failed.
5. Run the same command into another output directory and compare `summary.json`
   and `protocol.json`. Explain what the hashes and Python version do and do not
   establish about reproduction.
6. Pair each original test question with its row in `challenge_cases.json` by
   entity and label. Read the six new-template failures in `REPORT.md`. Explain
   why model selection remains based on validation, despite the diagnostic drop.
7. Inspect `grouped_stability` and its six rows. Explain why perfect scores
   across entities can coexist with six alternate-wording mistakes.
8. Inspect `training_imbalance_stress` and one row in
   `challenge_error_review.failures`. Separate the observed class counts and
   unseen features from hypotheses about annotation, composition and model fit.
9. Read `paired_wording_review.examples` and group its transitions by family.
   Find one correct-to-wrong pair, inspect both questions, and explain what the
   transition establishes and what it cannot establish about the cause.

## Questions to discuss

- Why is macro F1 useful when a future dataset may have unequal family counts?
- What does the population variance across two held-out entities describe? Why
  should it not be reported as a confidence interval?
- How could the current repeated templates make test accuracy optimistic even
  with strict entity separation?
- Why does the paired 16/16 versus 10/16 comparison reveal wording sensitivity
  but not prove that one changed word caused every error?
- Why must the paired review verify entity and label equality before comparing
  predictions? What would a missing alternate do to the transition counts?
- Which facts here come from constructed labels, and why does feature-label
  correlation fail to prove causality?
- Which claims about Hebrew answers, model-generated responses or production
  quality are unsupported by this experiment?
- What would an independent label audit need to check before calling a
  challenge mistake a model error? Why cannot a single downsampling result
  estimate a general effect of imbalance?

## Change exercises

1. Copy `cases.json`, delete a label value, and pass the copy with `--cases`.
   Predict the failure and verify that no output directory is created.
2. Change one entity's split in the copied file. Verify that the leakage check
   rejects it. Restore the split and rerun.
3. Change one challenge template without changing the original cases or model
   selection rule. Predict which labels may be confused, rerun, and record
   accuracy, macro F1 and failure examples. Treat this as exploration of the
   diagnostic set, not a fresh held-out score.
4. Add a fourth candidate with a documented smoothing value. Select it using
   validation only, add a failure-path unit test, and explain whether test
   inspection influenced your design.
5. As a separate future task, write a comparable Hebrew synthetic template set
   with the same labels and criteria. Report each language separately; do not
   merge scores until tokenization and task difficulty are examined.
6. Try a copied input with one question repeated under another label, then
   repeat it in a second split with the same label. Confirm both validation
   failures occur before any result directory is written.
7. Change the family chosen for training downsampling. Predict the affected
   recall before running the diagnostic; explain any difference without
   claiming a causal generalization from this tiny synthetic dataset.
8. In a copied script, remove one alternate question or change its label.
   Run the paired review and explain why each input must be rejected. Then
   restore the pair and examine its transition in the JSON output.
9. In a copied case file, change one generated case's label while keeping its
   ID. Confirm the source-ledger audit rejects it before creating output.
   Then give the case a custom ID and explain why the audit lists it as
   unmatched rather than treating its label as independently verified.
10. Recompute the mean and population variance from the six grouped stability
    fold accuracies. Change one entity's question wording in a copied input,
    rerun the diagnostic, and discuss why fold variance changes without
    becoming an uncertainty interval for natural language performance.

If using this work in an interview or review, demonstrate the run, describe one
change you made yourself, and distinguish your work from AI-assisted code.
