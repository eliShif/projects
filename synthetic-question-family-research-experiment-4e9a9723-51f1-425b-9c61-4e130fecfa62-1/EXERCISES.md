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

## Questions to discuss

- Why is macro F1 useful when a future dataset may have unequal family counts?
- What does the population variance across two held-out entities describe? Why
  should it not be reported as a confidence interval?
- How could the current repeated templates make test accuracy optimistic even
  with strict entity separation?
- Which facts here come from constructed labels, and why does feature-label
  correlation fail to prove causality?
- Which claims about Hebrew answers, model-generated responses or production
  quality are unsupported by this experiment?

## Change exercises

1. Copy `cases.json`, delete a label value, and pass the copy with `--cases`.
   Predict the failure and verify that no output directory is created.
2. Change one entity's split in the copied file. Verify that the leakage check
   rejects it. Restore the split and rerun.
3. Add a genuinely new question template for one family to the generator. Keep
   that template exclusive to held-out entities, then rerun. Record how accuracy,
   macro F1 and failure examples change. Discuss whether the original result was
   mainly template recognition.
4. Add a fourth candidate with a documented smoothing value. Select it using
   validation only, add a failure-path unit test, and explain whether test
   inspection influenced your design.
5. As a separate future task, write a comparable Hebrew synthetic template set
   with the same labels and criteria. Report each language separately; do not
   merge scores until tokenization and task difficulty are examined.

If using this work in an interview or review, demonstrate the run, describe one
change you made yourself, and distinguish your work from AI-assisted code.
