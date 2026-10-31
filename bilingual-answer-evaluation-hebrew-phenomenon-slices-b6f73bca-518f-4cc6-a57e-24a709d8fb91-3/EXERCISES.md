# Walkthrough and future exercises

These are learning prompts to perform later. The code, tests and generated explanations do not establish that anyone has mastered, independently implemented, explained, read or presented this material. Keep your own observations and questions separate from the recorded automated checks.

## Run and explain one example

1. Follow the setup commands in `README.md`, run `phenomena-smoke`, then save an `answers --dataset hebrew-phenomena` run in a new directory. Open its `answer_dataset.json` and `answer_report.json`. Identify the validation/test entities, the threshold candidates and selected threshold. Why is there no training split here? What operation actually uses validation labels?
2. Open the packaged JSON before reading the parser. For the two Hebrew Fjord questions, write down the answer and source ID yourself. Compare `raw_question` with `question`. Which character distinguishes `התיאור` and `התאור`? What does cleaning change, and what must it preserve?
3. Trace the spelling variant through `phenomena.question_target` and `validate_dataset`. Explain why a word's presence in some source is insufficient: the entity, attribute, answer and citation must agree. Then run `python3 -m unittest discover -s tests -p test_phenomena_data.py -v` and explain one incorrect label and one false-abstention rejection.
4. Count the word overlap for the Fjord description and shape documents by hand. Apply the documented document-ID tie rule, then the validation-selected threshold. Find the same selected source and error stage in `methods.lexical_retrieval.by_phenomenon.spelling.failures`. Why is this a retrieval miss even though a correct answer exists in the corpus?
5. Compare the control and variant for Ember and Grove. Explain the narrow inflection and mixed-language definitions. Why does a Latin entity ID alone not define mixed-language content in this dataset? What additional examples would be needed before generalizing to Hebrew morphology or code-switching?

## Interpret results without hiding limits

6. Recompute the exact answer-and-citation accuracy from the saved test cases and errors. Explain the distinction between overall accuracy, grounded-answer rate and abstention accuracy. Inspect the small support counts for each phenomenon. Do not substitute expected test constants for values from your actual run.
7. Run the original `answers` command without `--dataset`. Explain why its phenomenon slices have `count: 0` and null metrics. Compare that with the challenge's nonempty spelling slice with an accuracy of zero. Why is `abstention_accuracy` null for both an all-correct and an all-wrong answerable-only slice?
8. Inspect `test_missing_source_and_answer_error_with_source_are_distinct`. Explain how the same false abstention can occur when a source was or was not supplied. Why is `error_stage` not inferred for imported model predictions? Which information would be required to verify model exposure?
9. Write a short future research note: narrow question, data source, budget, methods, actual per-slice observations, one negative finding, and at least three limitations. Explain why four entities, constructed templates, English paraphrases and no independent linguistic review prevent broad causal claims. A useful negative result does not require an improved score.

## Reproducibility and failure walkthrough

10. Run `answers-verify` on your saved baseline. In a disposable copy of that run, change a slice count or error stage and run it again. Explain why it rejects the report even though the JSON is valid. Compare this to `test_injected_success_replays_new_slices_and_rejects_rehashed_edits`, which updates file hashes and still fails scoring replay.
11. Create a challenge `gguf-plan` in a new directory without providing weights. Inspect the two translations of one form: same pair ID and seed, same generation settings, and no labels or annotations in the answerer input. Inspect a different form's pair ID. Does equal seed imply equal tokens, equivalent retrieved evidence or identical outputs?
12. Read the missing-model test and saved failed status. Explain why a failed run has no quality summary or final manifest, and why auditing that failure can exit successfully. Distinguish the injected test backend from actual local-model inference. Never report fixture scores as model quality.
13. Build the wheel and run `scripts/wheel_smoke.py` with it. Locate the resource in the wheel. Explain how the import-origin assertion prevents an editable installation from masking omitted package data.

## Bounded change exercises

14. In a practice copy, add one new **explicit** inflection form. First write a literal source/question/label fixture and failures for wrong entity, wrong annotation and missing paired control. Update the grammar, group schema if necessary, corpus and data card. Keep all related forms in one partition. Rebuild the wheel and create fresh GGUF plans because code/data hashes changed. Do not silently broaden the validator to accept arbitrary text.
15. Propose a query normalization alternative for the single deficient-spelling pair. State a hypothesis and budget before inspecting test results. Keep the original raw question and baseline results; pass the same original question/source information to every compared answerer. Select any tunable rule only on validation. Include an unrelated word that must not be changed. Explain whether this is a narrowly handwritten rule or learned semantic retrieval.
16. Add a new phenomenon with **no test examples** first. Specify the empty-slice behavior and tests before adding data. Then introduce one complete control/variant translation group in one split, preserving entity isolation and source references. Show that missing translations, duplicated text and a wrong gold value are rejected.
17. Add an answerable case whose source is retrieved but whose response has the wrong citation. Write the expected error category and diagnostic stage before running code. Explain why answer errors and retrieval misses should not be merged, and why neither diagnoses a linguistic cause by itself.
18. Prepare a future review note describing one proposed change, its success/failure tests and limits. Ask a reviewer to assess the Hebrew examples and experimental controls if review becomes available; record actual feedback separately. This exercise is not evidence that review, supervision or a presentation has already happened.

The earlier experiments remain available for further practice: trace the original hue paraphrase in `answer_eval.py`, or the bigram scores and training/validation/test flow in `experiment.py`. Treat their synthetic data and earlier validation logs within their documented scope.
