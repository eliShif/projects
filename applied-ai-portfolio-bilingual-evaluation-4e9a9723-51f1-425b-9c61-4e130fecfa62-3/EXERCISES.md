# Guided exercises

These are prompts for future hands-on learning. Completing the automated tests
alone does not show that a person can explain the work.

1. Run the README smoke command. Open `cases.json`, `predictions.json`,
   `runs.json` and `summary.json`. Trace one Hebrew case from question to
   retrieval, prediction, score and per-language error list. Explain what the
   citation proves and what it cannot prove.
2. Read the English `TfidfVectorizer` documentation for tokenization and
   `ngram_range`. In your own words, write how those settings affect Hebrew
   queries here. Identify a failure that token overlap cannot solve. Record
   the exact documentation version and your observation after running code.
3. Change a current corpus value without changing the expected answer, run
   the tests, and explain the failure. Then update the case deliberately and
   verify that both languages use the same fact. Discuss how a test split
   should be frozen before model selection.
4. Add a conflicting current record for Dune. Update the expected answer to
   abstention in both languages and add a test proving that an answer with a
   citation still fails. Explain why checking only numeric accuracy is weak.
5. Provide actual model predictions with `--predictions`, including one wrong
   number, one missing citation and one bad abstention. Compare the two
   language summaries. State the small sample size and synthetic-data limits
   when presenting findings; do not infer a causal language effect from a
   difference in these six cases.
