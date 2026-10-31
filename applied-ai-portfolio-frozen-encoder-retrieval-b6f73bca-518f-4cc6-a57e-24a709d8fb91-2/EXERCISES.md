# Walkthrough and learning exercises

These are future activities, not statements that anyone has completed them or
mastered the code. Keep observations separate from predictions and fixture
expectations. Use a fresh output directory for each run.

## Walk through one query

1. Read `data/paraphrase_retrieval.json`. Pick English and Hebrew queries; find
   their reference paragraphs and plausible distractors. Explain any ambiguity
   in the authored relevance labels before changing them.
2. Run preparation and audit from `README.md`. Locate the corpus hash, split
   groups, planned model version, pooling arms, k and budget in `protocol.json`.
3. Trace `load_study` into `validate_study`. In a copy of the data, introduce a
   duplicate query and predict the error. Try moving a test query to a validation
   topic with a matching reference. Why does different wording not prevent leakage?
4. Trace `Document` through TF-IDF fitting and sparse matrix multiplication.
   Explain why indexing public test-relevant paragraphs differs from fitting on
   test questions or selecting a method using test labels.
5. Trace `FrozenEncoder.encode` and `masked_pool`: `[B,T]` IDs → `[B,T,D]`
   contextual states → `[B,D]` vectors → `[N,D]` index. Work through the test's
   hand-calculated `[0.6,0.8]` vector before running any model.
6. Find where `compare` persists selection. Run the test that changes held-out
   labels. Explain why selection stays fixed while test scores change.
7. After installing dependencies, run the real TF-IDF baseline. Choose an error
   from `rankings.json`, calculate its precision, recall and reciprocal rank by
   hand, and compare with the audit.
8. If vetted pretrained local weights are available, run the encoder comparison.
   Record its exact content version. Explain positive, zero or negative effects
   without tuning against test or treating fixture values as model results.

## Explanation questions

- Why are token IDs not learned sentence embeddings? What does contextual
  attention add, and what can mean pooling lose?
- Why must padding be masked during attention and again during pooling? What
  happens to a short sentence batched with a long one if pooling ignores the mask?
- Why are `eval()`, frozen parameters and `inference_mode()` different controls?
  Which disables dropout, and which prevents parameter updates?
- Why does the dot product of unit vectors equal cosine similarity? What should
  happen with a zero norm or NaN hidden state?
- Does similarity 0.8 imply an 80% relevance probability? Why are TF-IDF and
  encoder score scales not directly comparable?
- Why does a short ranking's precision still divide by k? When do Recall@k and
  Hit@k differ?
- Why would choosing CLS after seeing a higher test score invalidate the held-out
  interpretation? Which method was selected before that score existed?
- Why can a training/validation gap suggest overfitting without proving its
  cause? Why does a small gap on authored examples fail to establish generalization?
- Why does passing a random-BERT contract test say nothing about semantic quality?
  What evidence would a pretrained-model result require?
- Why is artifact consistency not proof that inference happened? What does an
  independent rerun with identical weights and inputs add?

## Change exercises

1. **Padding regression:** remove the mask from the pooling sum temporarily.
   Predict the failing test, run it, restore correctness, and verify. Do not
   change the numerical oracle to match the bug.
2. **Multiple references:** author an unambiguous query with two relevant
   paragraphs in one topic. Test partial recall, fixed-k precision and first
   relevant rank. Keep the topic in one split.
3. **Lexical alternative:** add character n-gram TF-IDF for Hebrew spelling
   variation. Predeclare settings and the selection rule before a new held-out
   run. Extend shared-query and tie-break tests. Do not call it a morphological
   analyzer.
4. **Long documents:** place an answer-bearing sentence beyond 128 tokens.
   Design a validation-only truncation experiment with updated token/forward
   budgets. Explain how this changes information available to each arm.
5. **Pooling alternative:** implement max pooling with padded positions excluded
   by an appropriate sentinel. Test padded states larger than real states and
   all-padding input. Predeclare this arm before a new held-out comparison.
6. **LoRA tracing:** derive A/B dimensions and `x @ A.T @ B.T` in the preserved
   adapter code. Explain why only B initially receives a useful gradient when
   B starts at zero. Trace `-100` answer masks and validation checkpoint choice.
   Use the supervised-update test before a real local causal-model experiment.
7. **Failure handling:** simulate a retrieval exception in a test. Verify that
   no completion summary is accepted. Explain why replacing a failed arm with
   fixture rankings would invalidate the experiment.

## Research discussion worksheet

When carrying out this exercise, read the paper identified in
`docs/retrieval-study.md`. Explain its objective, sentence-vector construction,
and the difference between training a sentence encoder and using one frozen.
Separate the paper's claims from this repository's hypothesis.

For an actual run, record the command, data hash, model version, budget,
validation decision, held-out effect, two errors and limitations. Propose a next
experiment without revising the current test decision. A peer can review labels,
masking math and selection ordering; no peer review or presentation is claimed
by these materials.
