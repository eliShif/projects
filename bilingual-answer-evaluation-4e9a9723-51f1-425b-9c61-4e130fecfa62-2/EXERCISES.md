# Walkthrough and practice

These are prompts for future learning. Completing them requires running and explaining the code; this repository does not claim that anyone has done so.

1. Run the commands in `README.md`. Open the generated CSV and report. Identify the research question, the label rule, the train/validation/test counts, and why group IDs matter. Which file is read by the evaluator?
2. Trace one sentence through `BigramModel.token_log_probabilities`, `scores`, `choose_threshold`, and `metrics`. Why are there four scored tokens for a three-word sentence? What does add-one smoothing do for an unseen transition?
3. Compare `methods.sum` and `methods.mean` in `report.json`. Look at both length buckets, their mean scores, and individual errors. Did the methods differ in this run? What would a difference in score range alone establish? What would it fail to establish?
4. Read the definitions of balanced accuracy and ROC AUC in `cola_lab/experiment.py`. Explain why a majority-class baseline can have 50% balanced accuracy here. Which measure is independent of the chosen threshold?
5. Change the seed in `generate_rows` and the recorded protocol seed together. Rerun the tests and experiment. Compare the test metrics and failures. What does variation across splits imply about claims of stability?
6. Add a third scoring rule, such as a length-penalized sum, using the same training data and validation-only threshold selection. Add a test with known token log probabilities and document any changed results. Avoid choosing a rule after examining test labels.
7. Deliberately move one `group_id` across splits in a copy of the CSV. Confirm that `read_rows` rejects it. Then add a test for a new data-quality failure, such as a wrong length bucket.
8. Draft a short research note in your own words: question, data provenance, protocol, observed values from your run, two concrete error examples, and limitations. Explain why these generated examples cannot validate claims about GPT-2 or CoLA.

## Bilingual answer evaluation walkthrough

9. Run `answers` and open `answer_dataset.json` and `answer_report.json`. Trace one Hebrew and one English question through `retrieve`, `predict`, and `score_case`. Which fields are used as retrieval input, and which fields are used only for scoring?
10. Explain why a correct string with no evidence ID fails. Find an unanswerable case and explain the difference between correct abstention and a false answer. Compare the always-abstain baseline with lexical retrieval under the same criteria.
11. Inspect the two false abstentions. What wording differs from the supporting document? Explain why the observed equal scores across languages do not prove equal performance on natural questions.
12. Change a test expected answer in a copy of the dataset while keeping its evidence consistent. Show that the selected threshold stays fixed while test metrics can change. Why must threshold selection exclude test labels?
13. Add a second retrieval rule, such as character n-gram overlap, and compare it on the same validation and test cases. Document the rule before inspecting test outcomes; add a test for a paraphrase and a missing fact. Write a short note covering question, data source, result, one failure, and limits in your own words.
14. Trace `--predictions` from the CLI to `load_model_predictions`, `evaluate`, and `score_case`. Why must every test case have exactly one output? Why is a citation from validation rejected even if its answer string matches?
15. If an authorized local model is available, freeze its outputs before inspecting test labels and record its exact model ID, version, generation settings, and input construction in your own research note. Compare `local_model` with both baselines by language, then inspect one correct answer and one error. Explain what the synthetic template design prevents you from concluding. This exercise is optional; the repository does not contain a model run.
16. Inspect `protocol.data_quality`, `by_entity`, and `paired_languages`. Which entity contains the lexical method's paired paraphrase failures? Explain why a `both_wrong` pair differs from a one-language-only failure. Remove one Hebrew test case in a copy of the dataset and confirm validation rejects the missing pair. Add a test for a second pairing failure of your choice.
17. Read `comparisons.lexical_retrieval` in the answer report. Recompute `accuracy_delta` from the two overall accuracies, then calculate the four entity differences from `by_entity`. Why are the six questions for one entity resampled together? Explain why the reported interval cannot establish performance on natural questions or a wider population.
18. Change a document's `answer` and its linked case's `expected_answer` to the same incorrect value in a copied dataset, leaving the document sentence intact. Confirm `validate_dataset` rejects it. Then explain one labeling mistake this source-text check would miss; propose a human review step without claiming it was performed.
19. Add a frozen prediction fixture with one English-only failure and compare its `paired_languages` result and its `comparisons.local_model` entity differences. Record the model provenance honestly: fixture outputs are useful for testing the evaluator, not evidence of an actual language-model run.

## Combined experiment report walkthrough

20. Run `python3 -m cola_lab run --output-dir artifacts` and compare `artifacts/REPORT.md` with the same values in `artifacts/report.json`. Trace the label counts from the CSV through `evaluate` to the report. Which columns show whether the test split is balanced?
21. Recompute one score slice's mean and population variance from the test CSV and `scores`. Read the linked Python `statistics.pvariance` documentation in the generated report. Why is the denominator the number of observed scores rather than one less? What would the value say, and fail to say, about future sentences?
22. Trace one `training_sensitivity.methods.sum.draws` entry. Explain which groups were sampled, why all rows in a group move together, how validation selects the threshold, and when test labels are used. Compare its errors with the fixed-fit errors. Does any zero variance imply the model is generally stable?
23. Choose one error that appears in multiple refits. Write a short finding in your own words with its observed frequency, a possible mechanism, and a limitation. Explain why score-length association in this generated data does not demonstrate a causal effect of length. Identify a controlled change that would test the mechanism more directly.
24. Change the resampling seed or draw count in a copy of the code, then rerun tests and the report. Predict which data-quality and fixed-fit values should stay constant. Add one assertion for a failure case, such as zero draws or an empty statistic input, and explain why the error is useful.

## Optional local GGUF walkthrough

25. Read `cola_lab/local_gguf.py` and trace one test case from `prompt_for` through `parse_generation`, `gguf_predictions.json`, and the existing scorer. Identify the exact point where a wrong citation is rejected and where a supported but wrong answer becomes an evaluation error. Which case fields never enter the prompt?
26. If you have an authorized local GGUF file, run the documented `gguf` command. Check the model and prediction hashes, settings, raw output and one per-language error in `gguf_run.json` and `answer_report.json`. Write your own short finding with one observed comparison and its limits. Without weights, use the tests to study the path; do not report their fixture score as model quality.
27. Change `parse_generation` in a copy to accept a clearly defined alternate response schema. Add one success and one rejection test. Explain how permissive parsing could change measured errors, and why parsing rules should be fixed before inspecting held-out results.

## Paired error audit walkthrough

28. Run `answers` and inspect `paired_audit.pairs` in the report. Find the Ember color pair and trace its two false abstentions back through `retrieve` and `score_case`. Compare those concrete rows with the aggregate `paired_languages` count. What does the pair-level view reveal that the count alone hides?
29. Trace `paired_case_audit` from `evaluate` into the saved report. Recompute `versus_lexical.always_abstain` by comparing the two methods' correctness on every test case. Explain why an observed improvement or regression is a scoring outcome, not a causal explanation.
30. Build a frozen output fixture with one English-only wrong answer and one Hebrew-only unsupported citation. Run `answers --predictions` and inspect their paired rows, `outcome_counts.local_model`, and `versus_lexical.local_model`. Add a failure test that removes a case or duplicates a language; explain why incomplete pairing invalidates the comparison. Fixture outputs test the analysis only and are not model-quality evidence.
