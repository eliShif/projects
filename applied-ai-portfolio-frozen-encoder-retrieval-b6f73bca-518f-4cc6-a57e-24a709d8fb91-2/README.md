# Applied AI Portfolio — Frozen Encoder Retrieval

English and Hebrew paraphrases can have little word overlap with relevant documents. The project evaluates whether frozen sentence embeddings improve retrieval over TF-IDF under shared data, retrieval cutoffs and selection rules.

## What it does

Reproducible English/Hebrew retrieval evaluation comparing TF-IDF with frozen multilingual MiniLM embeddings, using validation-only selection, bounded CPU inference and auditable rankings. The completed study recorded masked-mean MRR@3 of 1.000000 versus 0.388889 for TF-IDF on 12 authored held-out queries.

Compare TF-IDF, masked-mean pooling and CLS pooling at k=3; select the primary method using validation data before test search; enforce inference budgets; and retain model identities, rankings and recomputable evaluation artifacts.

## Solution and technologies

A Python CLI validates bilingual JSON and separates validation/test queries by topic group. All methods index the same text-only corpus. A scikit-learn TF-IDF baseline is compared with local PyTorch/Hugging Face Transformers inference from safetensors checkpoints. The frozen encoder produces padding-aware mean or CLS vectors, followed by L2 normalization and cosine ranking.

Validation MRR@3 selects the primary method; Recall@3 and a fixed method order break ties. Selection is saved before test search. Dataset snapshots, source and checkpoint hashes, package versions, inference counters and saved rankings support reproducibility. Auditing recomputes metrics and selection. A separate LoRA module trains rank-4 query/value adapters with a frozen base, answer-token loss and validation checkpoint selection.

Python · NumPy · PyTorch · Hugging Face Transformers · safetensors · unittest

## Quick start

```text
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.9,<3"
.venv/bin/python -m pip install -r requirements.txt
```

```text
.venv/bin/python -m portfolio.retrieval_experiment prepare --out artifacts/retrieval-plan
.venv/bin/python -m portfolio.retrieval_experiment audit --out artifacts/retrieval-plan
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q portfolio tests scripts
.venv/bin/python -m portfolio.retrieval_experiment smoke
```

## Results

The worker-verified pretrained CPU comparison completed on 2026-10-31 using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, pinned to upstream revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42. The authored dataset contains 12 documents and 24 queries: 12 validation and 12 test queries, with six English and six Hebrew queries in each split. Validation selected masked mean before test search.

| Method | Validation MRR@3 | Test MRR@3 | Test Recall@3 | Test Precision@3 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.361111 | 0.388889 | 0.666667 | 0.222222 |
| Masked mean | 1.000000 | 1.000000 | 1.000000 | 0.333333 |
| CLS | 0.777778 | 0.958333 | 1.000000 | 0.333333 |

Masked mean exceeded TF-IDF test MRR@3 by 0.611111. English/Hebrew test MRR@3 was 1.000000/1.000000 for masked mean, 1.000000/0.916667 for CLS and 0.416667/0.361111 for TF-IDF. Recall and Hit coincide for these single-reference queries. These are descriptive findings on this corpus, without a significance or general superiority claim.

The completed run used one CPU thread and batches of four, recording 72 text encodings, 54 forward calls, 1,448 attended tokens and 1,526 padded token slots, with zero training updates. Saved rankings support metric and selection recomputation; the recorded audit passed. See the [completed report](https://github.com/eliShif/projects/blob/eb24780e9b1c5e3872b4297e0cfde5b2aec7c906/applied-ai-portfolio-frozen-encoder-retrieval-b6f73bca-518f-4cc6-a57e-24a709d8fb91-2/results/pretrained-minilm-20261031/REPORT.md).

The separate LoRA experiment is inherited from 2026-10-22: 15,360 trainable parameters and 48 updates reduced held-out answer-token NLL from 6.153 to 0.521, while exact match remained 33.3%. Other inherited portfolio findings include slower INT8 vision latency and worsened small-LM perplexity and resident memory despite smaller saved states. These remain historical measurements; this documentation review performed no reruns.

## Scope and limitations

The small authored dataset lacks independently reviewed relevance labels. Its English and Hebrew queries are in-language cases, not a cross-language benchmark.

Topic-group holdout does not eliminate shared templates, writing style or possible pretraining contamination. Related paraphrases are correlated, so results support descriptive comparisons rather than significance or general superiority claims.

Every method indexes the full corpus, including documents relevant to held-out queries. Relevance labels are used for scoring and are not indexed.

CLS is a pooling ablation on the same checkpoint, which may favor mean pooling. TF-IDF excludes zero-overlap results, while dense retrieval can return negative-cosine neighbors. Truncation can affect longer-document comparisons.

The retrieval experiment performs no training, and no new full pretrained LoRA experiment was completed. The inherited LoRA result uses one seed and a narrow authored distribution; its gold-document, greedy-decoding protocol is not causally comparable with stochastic top-three-document retrieval.

Checkpoint hashes identify local bytes without certifying pretraining history or multilingual quality. Artifact auditing verifies consistency and is not independent proof of inference.

Fixed execution order and Python/scoring overhead prevent treating the retrieval run as a latency benchmark. No physical edge-device, power or Windows measurements are supplied; verified host CPU results do not establish those outcomes.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
