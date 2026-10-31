# Applied AI Portfolio: frozen-encoder retrieval study

This local extension compares the existing TF-IDF retriever with a frozen
Transformers encoder using masked-mean and CLS pooling. It asks one question:
do sentence embeddings retrieve these English and Hebrew paraphrases better
than lexical overlap, at the same `k` and over the same corpus?

The existing `portfolio/retrieval.py` and `portfolio/adapter_tuning.py` are
preserved. The only change to shared production code is making NumPy a local
import inside `common.clustered_difference`, so data preparation can run without
ML dependencies. [UPSTREAM.md](UPSTREAM.md) and
[UPSTREAM_FILES.json](UPSTREAM_FILES.json) record the supplied source lineage.
No remote repository was changed.

**Validation status:** the dependency-free evaluation/protocol tests run here;
ML dependencies could not be installed from the unavailable package indexes.
There are no new TF-IDF or pretrained-encoder scores in this adaptation. The
local-weight comparison, measurements, and conclusions remain unrun.
See [VALIDATION.md](VALIDATION.md) for exact outcomes. Historical files under
`results/` belong to the supplied source and are not new validation.

## Setup

Use Python 3.11+ on Linux; CPU inference is supported. Preparation was verified
with Python 3.14.4. Run commands from this repository root. No shell activation
or credentials are needed. Installation requires access to public package
indexes and compatible wheels.

```text
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.9,<3"
.venv/bin/python -m pip install -r requirements.txt
```

These are the existing Python/NumPy/scikit-learn/PyTorch/Transformers dependencies.
The existing hosted-model dependency is not used by this study. The new run
path performs no API calls, downloads, or network inference. Each protocol
records installed package versions; upstream version ranges are retained rather
than represented as a fully locked environment.

## Prepare, test, build, and smoke

Preparation validates the authored data, rejects duplicate texts and topic-group
leakage, and writes a protocol and snapshot without importing ML packages.
It records `prepared`, never a completed model result.

```text
.venv/bin/python -m portfolio.retrieval_experiment prepare --out artifacts/retrieval-plan
.venv/bin/python -m portfolio.retrieval_experiment audit --out artifacts/retrieval-plan
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q portfolio tests scripts
.venv/bin/python -m portfolio.retrieval_experiment smoke
```

The build compiles Python modules and tests to bytecode; no native build or
bundler is required. The finite smoke invokes the documented preparation CLI
in a temporary workspace directory, audits the snapshot, checks that a changed
dataset is rejected, and cleans up. Its output explicitly says **no model
scored**. It works without third-party packages.

For dependency-free checks when package installation is unavailable:

```text
python3 -m unittest discover -s tests -p test_retrieval_study.py -v
python3 -m portfolio.retrieval_experiment smoke
```

The full suite requires the installed dependencies. Dense tests create a small
**random BERT checkpoint as a contract fixture**, exercising real local loading,
tokenization, padding, normalization and freezing. These tests are not evidence
of pretrained quality and do not substitute for the comparison below.

## Run real retrieval

TF-IDF needs no model weights, but requires NumPy and scikit-learn:

```text
.venv/bin/python -m portfolio.retrieval_experiment run --mode baseline --out artifacts/tfidf
.venv/bin/python -m portfolio.retrieval_experiment audit --out artifacts/tfidf
.venv/bin/python -m portfolio.retrieval_experiment smoke --baseline
```

For the comparison, supply a locally obtained pretrained AutoModel-compatible
encoder/tokenizer in `models/encoder`. The intended model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; no revision or
weights were available here. The directory must contain `config.json`, tokenizer
assets, and `model.safetensors` or valid safetensors shards. Custom remote code
and pickle weights are not loaded. The exact local version is computed from
SHA-256 hashes of model/tokenizer files and saved in `encoder.json`; the optional
model ID is only an operator-supplied provenance label.

```text
.venv/bin/python -m portfolio.retrieval_experiment run --model-path models/encoder --model-id sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --out artifacts/retrieval-encoder
.venv/bin/python -m portfolio.retrieval_experiment audit --out artifacts/retrieval-encoder
```

Defaults: `k=3`, CPU, two threads, batches of eight, 128 tokens including special
tokens, zero training updates. Both dense arms share the same frozen checkpoint.
For the bundled 12 documents and 24 queries, the budget permits 72 text encodings
and 52 forward calls across both pooling arms. These are calculated limits,
**not measurements of an executed model**.

Every arm uses the same corpus text, queries and `k`. Validation MRR@k selects
the primary method before any test search; Recall@k and a fixed arm order break
ties. The report also shows predeclared test comparisons and language slices,
without reselecting on test. See [the research protocol](docs/retrieval-study.md).

## Outputs and failures

Use a new or empty output directory per run. Existing runs are never overwritten.
Preparation writes `protocol.json`, `study.json`, and `status.json`. Successful
retrieval adds `selection.json` before test search, `rankings.json` with case-level
evidence, `REPORT.md`, and a final `summary.json`. Dense runs also write
`encoder.json` and measured inference counters. Audit recomputes metrics and
selection from saved rankings and verifies hashes; it does not independently
prove that inference took place.

Missing dependencies or weights, non-finite vectors, budget exhaustion, and
invalid evidence abort a run. After preparation, `failure.json` records the
error and no success summary is written. Failed audits exit with code 2. If
installation fails, use the offline checks; never replace missing model results
with fixture scores.

## Preserved LoRA path and learning material

The original LoRA experiment freezes its base, trains low-rank query/value
adapters on answer-token loss, and selects a checkpoint on validation. It needs
its own local **causal** model, not the sentence encoder:

```text
.venv/bin/python -m portfolio.adapter_tuning --model-path models/SmolLM2-135M-Instruct --out artifacts/lora --steps 48
```

This is a separate experiment, not a retrieval arm. No new LoRA run is claimed.
[CODE_MAP.md](CODE_MAP.md) links implementations and tests for the assigned
requirements. [EXERCISES.md](EXERCISES.md) provides a walkthrough, explanation
questions, and change exercises. AI-assisted code does not establish anyone's
mastery, language ability, personal paper reading, or team review experience.
