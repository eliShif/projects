# Code map

| Path | Role |
|---|---|
| `portfolio/bilingual_eval.py` | Synthetic paired cases, lexical baseline, prediction validation, per-language scoring, frozen inputs and outputs |
| `portfolio/evidence_eval.py` | Reused numeric answer, abstention and expected evidence scorer; also holds the existing local-model experiment |
| `portfolio/retrieval.py` | Reused NumPy/scikit-learn TF-IDF retrieval; lexical token matching only |
| `portfolio/common.py` | JSON writing, content hashes and runtime provenance |
| `tests/test_bilingual_eval.py` | Success, error classification and malformed-input tests |
| `requirements-bilingual.txt` | Small dependency set for the offline adaptation |

Data flow: `dataset()` creates synthetic corpus and paired questions →
`lexical_baseline()` or supplied predictions → `evaluate()` validates coverage
and calls `score()` → `run()` saves protocol, inputs, runs and summaries.
The baseline copies a value from one retrieved record, so stale or irrelevant
retrieval can yield a visible failure. It is not an LLM or a semantic retriever.
