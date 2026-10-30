# Walkthrough and future learning exercises

These prompts support a future run, explanation and review. No exercise is
marked complete and no personal mastery, independent authorship or mentorship
is asserted by this AI-assisted repository.

## Walkthrough

1. Follow README setup and run the finite smoke. Open `protocol.json` before the
   metrics. Identify the research question, experiment version, Python version,
   seven service roots and the eight family labels.
2. Trace `entity_assignments` and `build_cases`. Locate Ember's comparison
   partner and explain why the old global-name ring put a validation entity in
   training text. Explain why a one-entity validation pool cannot supply two
   distinct operands and why Ibis was added before generation.
3. Inspect `summary.entity_audit.case_mentions` for a sum question and a
   Mirror question. Compare the text with its primary `entity` field. Account
   for every reference, including the secondary operand and suffix aliases.
4. Count all families by split. Check 24/16/16 total cases and the all-eight-key
   count maps. Explain what `missing_labels` and a null imbalance ratio mean,
   and why null differs from both zero and a balanced ratio of one.
5. Trace `TextBaseline.fit`, `features`, `class_scores` and `predict`. Explain
   priors, add-one smoothing, log sums, unseen features and deterministic ties.
   Read the reversed-word-order test and predict both models' outputs first.
6. Open `selection.json`. Recalculate the chosen method from validation macro
   F1, accuracy and alphabetical name. Verify the fitting IDs are training IDs.
   Locate the line that saves this file before test evaluation and explain why
   the primary models are not refit on validation.
7. Recalculate the majority baseline's conflict precision (2/16), recall (2/2),
   F1 (2/9) and eight-family macro F1 (1/36) from the test confusion matrix.
   Explain why reporting only accuracy hides which families are never predicted.
8. Pair an original test question with its alternate by ID. Inspect one of the
   six alternate-wording failures and its unseen-feature fraction. Separate
   what was observed from a hypothesis about why the classifier failed.
9. Inspect one `grouped_stability` fold. Reconstruct its held-out mentions,
   excluded case IDs and remaining training references. Explain why holding
   out only primary-entity rows would leave secondary-entity leakage.
10. Pick one training-resample seed and reconstruct its sampled IDs. Explain
    why duplicated samples and reused evaluation cases make the reported
    population variance a sensitivity description, not a confidence interval.
11. Run with `--cases` into a second directory and compare outputs. Explain what
    input/source hashes certify and what they do not establish about scientific
    validity or who operated the experiment.
12. Read the two English technical references linked in CODE_MAP. Write your own
    short summary of smoothing and macro averaging, then identify one code
    decision each source helps explain. Leave the summary unclaimed until you
    actually do this reading and writing.
13. Trace the new path from `evaluate(model, grouped["train"])` to
    `train_validation_review` to the report table. Recompute training accuracy
    and macro F1 from one saved confusion matrix. Subtract validation scores
    without rounding and explain the difference between 0.10 and 10 percent
    relative improvement. Locate the JSON direction for each metric.
14. Read `tests/test_research_generalization.py::known_scores`. Before running
    it, derive the two swapped-label errors: accuracy and macro F1 fall from
    1 to 6/8, so both gaps are +1/4. Reverse the score inputs and derive -1/4.
    Explain why this constructed arithmetic check is not a measured finding of
    overfitting. Run only that test module with:

    ```text
    .venv/bin/python -m unittest discover -s tests -p test_research_generalization.py -v
    ```

## Questions for discussion

- How can all entity pools be disjoint while every original test template still
  matches training? What additional split would test unfamiliar templates?
- Which names can the closed-catalogue scanner recognize? Why would a new
  corpus require an entity inventory or independently checked annotations?
- Why is it useful for an audit to report a missing evaluation family even
  though the full experiment rejects that dataset for model selection?
- Why does a missing class contribute zero to this fixed-label macro F1 instead
  of disappearing from its denominator? Which alternative metric definition
  would answer a different question?
- Does a validation tie establish that bigrams are better than unigrams? What
  does the alphabetical tie-break actually establish?
- Why can zero gaps describe both the weak majority baseline and the perfect
  template-based classifiers? Which result would you expect to survive a change
  in wording, and what additional data would be needed to test that expectation?
- When is high train performance and lower validation performance consistent
  with overfitting? List two alternatives that this experiment cannot separate.
  Why does a negative gap also fail to certify generalization?
- Why are training scores optimistic and validation scores affected by model
  selection? Why would tuning on the displayed test scores invalidate a fresh
  held-out claim? How can accuracy and macro F1 gap directions disagree?
- Why may all candidates be compared on test while the chosen method remains
  fixed? What would selecting the best test result invalidate?
- Which errors could come from labels, wording or model assumptions? What would
  independent human annotation review add beyond a generator ledger match?
- Why is a positive log-score margin compatible with a wrong prediction?
- How do shared templates and reused entities limit conclusions from the
  stability results? Why does the protocol avoid a population significance claim?
- What evidence would still be needed for a claim about someone's own coding,
  research reasoning, work under guidance or progress in learning?

## Change exercises

Use copies of inputs and a new output directory so the saved run stays available
for comparison. Record your prediction before running each change.

1. In a Cobalt sum question, change only the secondary entity to Garnet. Keep
   the ID, primary entity and split unchanged. Run the copied file with
   `--cases`; expect a nonzero exit and no new output directory. Repeat with
   mixed-case `gArNeTMirror`. Explain how the regression differs from changing
   a primary entity's split.
2. Remove every validation `ratio` row. Call `validate_cases` in a small script:
   inspect the zero count, missing-label list and null imbalance ratio. Then
   pass the same file through the CLI and explain why model fitting is rejected.
   Repeat for training and test, identifying each rejection point.
3. Evaluate a majority model on one conflict example. Derive accuracy 1 and
   macro F1 1/8 by hand. Explain why absence of the other seven families prevents
   interpreting this as successful coverage of all tasks.
4. In a copy of the pool mapping, leave only Fjord in validation. Verify the
   generator rejects it before making questions. Also try sharing a name across
   pools and adding a base name that collides with another name's Mirror alias.
5. Add a synthetic case to the word-order representation test. Predict whether
   its unigrams or bigrams distinguish the labels. Keep the example small enough
   to explain the scores without relying solely on a test pass.
6. Propose a fourth candidate with a different documented smoothing value.
   State the selection rule and expected effect before running; keep test and
   challenge out of tuning. Add a test for both valid and invalid settings.
7. Inspect the test that forces a majority validation winner. Explain why its
   altered validation scores are a control-flow fixture rather than measured
   performance. Modify the test to force unigram instead, preserving the
   guarantee that no test metric can replace the validation choice.
8. Change a challenge template and inspect matched correct/wrong transitions.
   Record the baseline and selected-method results, a failure example and a
   limitation. Treat this as exploratory analysis of a viewed diagnostic set.
9. Disable purging in a copied `grouped_stability` implementation and show one
   training question containing a held-out reference. Restore purging and
   explain the excluded IDs, reduced training size and retained label support.
10. Tamper with a copied `cases.json` after evaluation. Run `check_artifacts`
    against those artifacts and explain why the hash failure is useful but
    cannot prove that the original labels were correct.
11. In copied input, paraphrase only validation questions while keeping their
    meaning, labels, IDs and entities. Use a new output directory and record
    all candidate train/validation scores and gaps. Does template overlap
    change? Distinguish wording shift from a claim of overfitting; a nonzero gap
    is not guaranteed. Do not use test or challenge performance to choose the
    paraphrases or a model.
12. Inspect `test_training_scores_cannot_replace_validation_selection` and
    `test_changed_test_wording_cannot_change_choice_or_train_validation_review`.
    Explain why one uses controlled scores and the other executes real models.
    Predict which regression each would catch if training or test results
    accidentally entered the ranking rule.
13. Set a copied summary's bigram gap to 0.5 without changing its saved scores.
    Run `check_artifacts` and explain the failure. Repeat by changing training
    accuracy alone: the confusion-matrix check should now fail first. Explain
    why this consistency check still does not independently verify labeling.
14. Propose a small follow-up experiment with template-disjoint training and
    validation, the same label coverage and an untouched final test set. Write
    the split and selection rules before implementing it. Discuss what can be
    learned about fitting templates and what still cannot be established from
    a small authored dataset. Keep this proposal separate from measured results.

For a future review, prepare a brief account of the question, data, validation
choice, one failure and one limitation. Ask a reviewer to challenge the split
and missing-class policies. Record real feedback and a change only after that
review happens; distinguish your own contribution from the prepared code.
