# Frozen representations for paraphrase retrieval

## Research source and hypothesis

Nils Reimers and Iryna Gurevych (2019), **Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks**, EMNLP-IJCNLP, pages 3982–3992.
DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410).
Paper record: <https://aclanthology.org/D19-1410/>.

The paper describes adapting BERT with siamese/triplet training so sentences
can be encoded separately and compared using vector similarity. Corpus vectors
can then be reused across queries. Pooling turns token-level representations
into sentence vectors. This motivates the small experiment here: compare a
pretrained sentence encoder's masked mean with lexical TF-IDF, with CLS pooling
as an ablation using the same checkpoint.

**Hypothesis:** at fixed k, masked-mean embeddings improve held-out paraphrase
MRR relative to TF-IDF on this corpus. Zero or negative differences are valid
results. This does not reproduce SBERT training, its published benchmarks or
performance, or establish the paper's findings for Hebrew. This note is learning
material, not a record of a candidate reading or presenting the paper.

## Data and controls

`data/paraphrase_retrieval.json` has 12 authored paragraphs and two paraphrases
per paragraph. Six topic groups are validation and six are test; each split
includes English and Hebrew. These are in-language cases, not a cross-language
benchmark. Relevance is authored, not independently judged. No personal records
or imported job descriptions are included.

Every arm indexes the entire corpus, including publicly available paragraphs
relevant to test queries. This is the retrieval task, not hidden training labels.
No query text is used to fit TF-IDF's vocabulary. No relevance label is passed
to `Document`, the tokenizer, the encoder, or `search`. There is no task-specific
training. Topic groups prevent sibling paraphrases from crossing splits but do
not rule out shared style, related topics, or pretraining contamination.

The loader validates schema, IDs, text sizes, UTF-8 input, language tags, unique
known relevance IDs and matching topic groups. It rejects normalized duplicates
(NFKC, case-folding, collapsed whitespace), including verbatim query/document
copies. Duplicate checking does not change the text sent to retrievers. No
Hebrew morphological analyzer or stemming has been added.

## Representation math

The TF-IDF arm reuses `retrieval.Retriever` unchanged: unigram/bigram features,
sublinear term frequency and positive-score filtering. Both dense arms use the
same local encoder, tokenizer, truncation and frozen parameters. `dense_mean`
averages attended states; `dense_cls` uses the first token. A mean-trained
checkpoint may disadvantage CLS; this is a pooling ablation, not a separately
trained CLS sentence model.

For states `H[B,T,D]` and mask `M[B,T]`, the mean is
`s[b] = sum_t(H[b,t] * M[b,t]) / sum_t(M[b,t])`. Special tokens are attended;
padding is excluded. Normalize each row with `u = s / ||s||₂`. For a corpus
matrix `[documents,D]`, `U @ u_query` equals cosine similarity. Stable sorting
breaks ties by corpus order. Empty input, all-padding masks, zero norms, shape
mismatches and non-finite values raise errors rather than producing rankings.

Contextual states come from Transformer attention and feed-forward layers, not
token IDs themselves. Attention forms queries, keys and values, and weights
values by normalized query/key similarities. The attention mask excludes padded
positions during attention; pooling still needs its own mask. `eval()` disables
dropout, freezing prevents parameter updates, and `inference_mode()` prevents
graph construction. These controls have distinct purposes.

## Version, budget and selection

Executed model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
The pretrained CPU comparison completed on 2026-10-31 using upstream revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42` and local encoder version
`sha256:58e014cbf4b3559379c8b9da32c807fa3d57604116af6ebb86416f4e2487831d`.
See [the executed report](../results/pretrained-minilm-20261031/REPORT.md).
Real runs hash local config, tokenizer and safetensors files into
a content-addressed version in `encoder.json`; installed library versions go
into `protocol.json`. Retain those exact local assets for reproduction. A name
alone is not an immutable version; hashes do not certify pretraining history.
The loader uses local files only, no remote code, and no pickle fallback.

Default caps: CPU/two threads, batch eight, 128 tokens, zero updates, at most
128 documents and 128 queries. With N documents, Q queries and batch size B,
the two dense arms permit `2*(N+Q)` text encodings and
`2*(ceil(N/B)+Q)` forward calls. They share one loaded model but compute their
own indexes and query vectors. Tokenization truncates long inputs; losing
information can confound a representation comparison on longer documents.

Completed runs record actual text counts, forward calls, attended tokens,
padded slots and wall times. Model loading is separate from indexing/validation
and test search/scoring. Fixed execution order, Python overhead and one run do
not support a latency benchmark claim.

Selection maximizes validation MRR@k, then Recall@k, then the fixed order
TF-IDF, masked mean, CLS. `selection.json` is written before any test query is
searched. Test labels never enter the selector. All predeclared test arms are
reported, but the validation-selected arm remains primary. Tuning new variants
after viewing test outputs would invalidate the held-out interpretation.

## Metrics and interpretation

Precision@k divides relevant retrieved items by requested k even if fewer items
are returned. Recall@k divides by the number of relevant documents. Reciprocal
rank is `1/rank` for the first relevant item within k, otherwise zero; MRR is
its query average. Hit@k indicates any relevant result. Recall and Hit coincide
for the current single-reference cases; the scorer also supports multiple
references.

TF-IDF omits zero-overlap results. Dense retrieval returns the nearest k even
when similarities are negative. Their scores are not calibrated probabilities
or comparable across spaces; the comparison uses ranked relevance. Reports
include language slices and alternative-minus-TF-IDF MRR effects paired by query
and averaged by topic. Group averaging matters if topics contain unequal
numbers of paraphrases. There is no significance or independence claim.

**Current conclusion:** the pretrained CPU comparison completed on 2026-10-31.
Masked mean was selected on validation MRR@3: 1.000000 versus 0.777778 for
CLS and 0.361111 for TF-IDF. On the 12 held-out queries, MRR@3 was 1.000000
for masked mean, 0.958333 for CLS and 0.388889 for TF-IDF; Recall@3 was
1.000000, 1.000000 and 0.666667, respectively. Masked-mean minus TF-IDF
MRR@3 was +0.611111. These are descriptive results on this authored corpus,
not a significance result or general superiority claim.

Earlier builder-sandbox dependency and weight failures are historical; see
[VALIDATION.md](../VALIDATION.md) for their source and timing limits. Default
smoke checks preparation and integrity only. The later worker baseline smoke
completed a real TF-IDF run and artifact audit. Random BERT checkpoints and
hand-built rankings remain test fixtures and cannot answer the research hypothesis.

For this completed run, inspect case-level errors before explaining any
difference. Report its exact model/data versions, budget, selected arm, positive
or negative effects and language slices. Do not equate performance on these
authored cases with general semantic understanding or personal language ability.
The existing LoRA protocol is separate; this study does not fine-tune an encoder.
