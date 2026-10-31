# Retrieval experiment

Mode: compare. Selected on validation: **dense_mean**.

The selected arm is the primary held-out result. Other test arms are predeclared descriptive comparisons; do not reselect on test.

| Arm | Precision@k | Recall@k | MRR@k | Hit@k |
|---|---:|---:|---:|---:|
| tfidf | 0.222222 | 0.666667 | 0.388889 | 0.666667 |
| dense_mean | 0.333333 | 1.000000 | 1.000000 | 1.000000 |
| dense_cls | 0.333333 | 1.000000 | 0.958333 | 1.000000 |

## Interpretation

Masked-mean minus TF-IDF MRR@k: +0.611111. This is an observed difference on these authored cases, not a general capability estimate.

See rankings.json for individual successes/errors and summary.json for language slices and topic-paired differences.

## Limits

- Authored, small, in-language English/Hebrew cases; no independently reviewed relevance labels.
- Topic groups are held out, but shared writing style and authored templates limit generalization.
- All arms see the same full corpus, including documents relevant to held-out queries; labels are not indexed.
- No training in this experiment; the existing LoRA experiment is a separate protocol.
- TF-IDF omits zero-overlap results; dense retrieval returns up to k neighbors, even for negative cosine scores.
- Effects are descriptive. Paraphrases in a topic are correlated; no significance or general superiority claim.
- Checkpoint hashes identify local bytes, not their pretraining history or multilingual quality.
- Timings include Python/scoring overhead and fixed execution order; not a latency benchmark.
