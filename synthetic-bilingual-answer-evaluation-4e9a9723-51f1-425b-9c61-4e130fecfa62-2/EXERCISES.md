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
