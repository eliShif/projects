# Code map

## Main flow

`data/paraphrase_retrieval.json` → `retrieval_study.load_study` →
`retrieval_experiment.prepare` → shared text-only `retrieval.Document` corpus →
existing `Retriever` / new `DenseRetriever` → validation rankings → persisted
selection → test rankings → metrics/report → `retrieval_experiment.audit`.

| Path | Responsibility |
|---|---|
| `portfolio/retrieval.py` | Unchanged source: `Document`, TF-IDF index/search and bounded arithmetic. |
| `portfolio/dense_retrieval.py` | Local safetensors identity/loading, frozen tokenizer/model inference, masked pooling, normalization, cosine retrieval and inference caps. |
| `portfolio/retrieval_study.py` | Data integrity, grouped splits, scoring, language summaries, validation selection and paired effects. No ML import needed. |
| `portfolio/retrieval_experiment.py` | CLI, provenance, compute protocol, shared arm construction, failure/complete artifacts, audit and finite smoke. |
| `portfolio/common.py` | Reused JSON hashing/writing and provenance. NumPy import moved into the existing bootstrap function; its math is unchanged. |
| `portfolio/adapter_tuning.py` | Unchanged source: LoRA, frozen base, answer-token labels/NLL, bounded updates and validation checkpoint selection. Separate from retrieval. |
| `data/paraphrase_retrieval.json` | Authored bilingual corpus and grouped queries; no personal information. |
| `tests/test_retrieval_study.py` | Dependency-free data, metric, selection, failure, audit and CLI tests. Hand-built rankings are fixtures only. |
| `tests/test_dense_retrieval.py` | PyTorch math and local Transformers contracts using random BERT test weights. Requires ML dependencies. |
| `tests/test_retrieval_integration.py` | Real TF-IDF run, recomputation, tampering and failure artifacts. Requires ML dependencies. |
| `tests/test_adapters.py` | Preserved freeze tests plus matrix arithmetic, supervised updates, token masks and weighted NLL. Requires PyTorch. |
| `tests/test_statistics_retrieval.py` | Preserved statistics, TF-IDF and bounded-tool tests. |
| `docs/retrieval-study.md` | Research source, hypothesis, math, budget, completed pretrained comparison and research limits. |
| `UPSTREAM.md`, `UPSTREAM_FILES.json` | Supplied pinned source and pre-adaptation hashes. No independent remote-fetch claim. |
| `VALIDATION.md`, `artifacts/validation/` | Actual local command outcomes, including failures; not model-performance evidence. |

## Assigned requirement evidence

These mappings describe code and tests, not personal competence. Execution status
is in `VALIDATION.md`; dependency-based tests and pretrained experiments are not
marked passed merely because they exist.

| ID | Implementation | Tests | Scope |
|---|---|---|---|
| R02 | `portfolio/retrieval_study.py:load_study`; `portfolio/retrieval_experiment.py:run` | `tests/test_retrieval_study.py`; `tests/test_retrieval_integration.py` | Python files, validation, failure handling and library run path. |
| R04 | `portfolio/adapter_tuning.py:encode`, `answer_loss`, `run`; `portfolio/retrieval_study.py:select_on_validation` | `tests/test_adapters.py`; `tests/test_retrieval_study.py:EvaluationTests` | Preserved supervised optimization and new validation isolation; no claim of absent overfitting. |
| R05 | `portfolio/dense_retrieval.py:masked_pool`; `portfolio/retrieval_study.py:paired_differences`; `portfolio/common.py:wilson` | `tests/test_dense_retrieval.py:PoolingTests`; `tests/test_retrieval_study.py:EvaluationTests`; `tests/test_statistics_retrieval.py` | Vector math, descriptive statistics and existing probability interval tests. |
| R06 | `portfolio/dense_retrieval.py:FrozenEncoder.encode`, `masked_pool` | `tests/test_dense_retrieval.py:LocalEncoderContractTests`, `PoolingTests` | Tokenizer-to-learned-vector contracts passed; a separate pretrained MiniLM comparison completed on the authored bilingual set (see `results/pretrained-minilm-20261031/REPORT.md`). |
| R11 | `portfolio/retrieval_experiment.py:prepare`, `run`; `portfolio/retrieval_study.py:compare` | `tests/test_retrieval_study.py`; `tests/test_retrieval_integration.py` | Existing baseline plus two pooling arms under explicit caps. |
| R12 | `portfolio/dense_retrieval.py:masked_pool`; `portfolio/retrieval_study.py:compare` | `tests/test_dense_retrieval.py:PoolingTests`; `tests/test_retrieval_study.py:EvaluationTests` | Focused representation ablation with common cases and k. |
| R13 | `portfolio/adapter_tuning.py:LoRALinear`, `run` (preserved) | `tests/test_adapters.py` | Adapter optimization and contracts; no new pretrained fine-tuning run. |
| R16 | `docs/retrieval-study.md`; `portfolio/retrieval_experiment.py:PAPER`, `prepare`, `run` | `tests/test_retrieval_study.py`; `tests/test_retrieval_integration.py` | Identified paper, limited hypothesis and executable comparison; no personal reading/presentation claim. |
| R19 | `portfolio/retrieval.py:Retriever`; `portfolio/dense_retrieval.py:DenseRetriever` | `tests/test_dense_retrieval.py:DenseRankingTests`; `tests/test_retrieval_integration.py` | Adds learned-representation search; no new general extraction/classification claim. |
| P02 | `portfolio/dense_retrieval.py:FrozenEncoder`; `portfolio/adapter_tuning.py:LoRALinear` | `tests/test_dense_retrieval.py:LocalEncoderContractTests`; `tests/test_adapters.py` | Actual PyTorch/Transformers operations; attention/pooling explanation in research note. |
| I01 | `portfolio/retrieval_experiment.py:run`; `portfolio/retrieval_study.py:compare`, `select_on_validation` | `tests/test_retrieval_study.py:EvaluationTests` | Common query/k inputs, selection before test, test-label mutation leaves selection unchanged. |
| I02 | `portfolio/retrieval_study.py:validate_study`, `score_ranking`; `portfolio/retrieval_experiment.py:audit`; `portfolio/dense_retrieval.py:masked_pool` | `tests/test_retrieval_study.py`; `tests/test_dense_retrieval.py`; `tests/test_retrieval_integration.py` | Numerical oracles, bad input, tensor edges, missing weights and artifact tampering. |

Other original modules and historical results remain available. They were not
expanded to create unrelated coverage.
