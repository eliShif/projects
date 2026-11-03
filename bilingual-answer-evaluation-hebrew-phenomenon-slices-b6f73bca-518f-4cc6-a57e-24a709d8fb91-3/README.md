# Bilingual Answer Evaluation — Hebrew Phenomenon Slices

Aggregate bilingual scores can conceal whether Hebrew wording changes cause failures in source retrieval or answer extraction. This project examines possessive inflection, full/deficient spelling and Hebrew–English content-word substitutions using matched controls.

## What it does

A Python toolkit for Hebrew–English answer evaluation with validated variant groups, matched controls and diagnostics that distinguish lexical retrieval misses from answer errors. It documents failures across three narrowly defined Hebrew phenomena and preserves inputs for reproducible baseline replay.

Preserve raw inputs, validate answers and citations against source text, reject exact duplicates and keep related variants and translations within one partition. Select retrieval thresholds using validation data only, then report test outcomes with support counts, explicit empty slices, concrete failures and reproducible replay.

## Solution and technologies

A Python 3.10+ CLI processes a packaged JSON corpus using standard-library components. Text preparation applies NFC and whitespace normalization while retaining raw strings; validation checks explicit grammar, source references and complete, isolated variant groups. Word-overlap retrieval and template extraction are compared with an always-abstain baseline. Reports contain exact answer-and-citation accuracy, grounded-answer and abstention rates, matched Hebrew slices and lexical retrieval diagnostics. Saved datasets support baseline recomputation. The optional llama-cpp-python/GGUF adapter freezes inputs, seeds and hashes and records generation attempts; successful validation used fixture backends. A separate retained bigram experiment includes training and validation-selected thresholds.

Python · unittest · JSON

## Quick start

```text
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip wheel --no-index --no-build-isolation --no-deps --wheel-dir dist .
```

```text
.venv/bin/python -m cola_lab answers --dataset hebrew-phenomena --output-dir artifacts/hebrew-study
.venv/bin/python -m cola_lab answers-verify --output-dir artifacts/hebrew-study
.venv/bin/python -m cola_lab phenomena-smoke
```

## Results

Reviewed findings document lexical-baseline failures across all three Hebrew phenomena. The documented Fjord deficient-spelling example illustrates a retrieval miss despite an available supporting fact. The challenge contains 48 questions and 32 documents; its test partition has 24 questions, including 16 answerable and 8 unanswerable cases. Hebrew variant support is two inflection cases, one spelling case and one mixed-language case. The complete challenge result report is not supplied, so aggregate accuracy and numerical control/variant differences cannot be confirmed here.

The separate, completed historical bigram experiment in artifacts/REPORT.md reports accuracy and balanced accuracy of 0.667 for both summed and mean scores, versus 0.500 for the majority baseline, on 24 held-out cases. Both scoring methods report AUC 0.694, zero false positives and eight false negatives, with different error patterns. Across 24 seeded training-group refits with fixed validation/test sets, mean accuracy was 0.606 for summed scores and 0.602 for mean scores; both ranged from 0.500 to 0.750. These are inherited experiment measurements, not Hebrew-challenge results or fresh reruns.

## Scope and limitations

The constructed corpus has four held-out entities and narrow linguistic patterns, with no independent linguistic review; it does not represent natural Hebrew usage.

English variants are paraphrases, so aggregate language differences cannot isolate a causal Hebrew effect. Four-entity resampling and the inherited bigram refits provide descriptive results, not population confidence claims.

Source validation uses a closed grammar, retrieval is lexical, and exact matching can reject reasonable answers. Learned semantic search is not implemented.

Real GGUF model inference is not demonstrated. Fixture success, frozen requests and verified failure archives do not establish model quality; imported predictions do not verify generation provenance or model exposure.

GGUF replay verifies selected report and protocol fields; reviewed evidence identifies policy fields that replay does not verify.

The complete challenge answer report and raw worker logs are not supplied. answer_eval.py and the inherited missing-model protocol are partial, limiting independent verification.

The supplied evidence contains host software checks but no edge-device, power-use or Windows measurements.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md) · [Source provenance](UPSTREAM.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.
