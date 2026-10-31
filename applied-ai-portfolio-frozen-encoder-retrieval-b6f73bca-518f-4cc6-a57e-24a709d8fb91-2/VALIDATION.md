# Reproducible validation

Verified at 2026-10-31T05:14:14.058Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `006eebbde8e9b42d45805547baebafb1400e59972b6b95e8d2d8c2621067ddbd`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio","tests","scripts"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.retrieval_experiment","smoke"]}`

Exit code: 0

Output SHA-256: `b0a7f18b5de53dd1aad783088bc6bd279dee77c5493ddfb85f221cf74cc070e3`

Raw execution output remains in private application storage.


## Completed worker validation and pretrained comparison — 2026-10-31

The pinned pretrained MiniLM comparison, result audit, full test suite,
bytecode build and documented baseline smoke all completed with exit code zero.
The worker log records **90 tests passed in 6.962 seconds**, including real
TF-IDF integration, random-BERT loading/freezing contracts, adapter supervised
updates and HTTP/service checks. Adapter updates occurred within tests; they
are not a new full pretrained LoRA experiment.

The baseline smoke reported `real TF-IDF run and artifact audit`. The default
smoke command recorded above remains a preparation/integrity check. The
completed-comparison audit reported `model_evaluated: true` and explicitly
limited its scope to artifact consistency, not independent proof of inference.
The separate worker execution and review verified real pretrained CPU inference.
The `torch_dtype` deprecation and HTTPError cleanup ResourceWarnings did not
fail the suite.

See `results/pretrained-minilm-20261031/` for the protocol, environment lock,
model hashes, validation selection, full rankings and descriptive test metrics.
No new full LoRA experiment or generalization beyond the authored corpus is asserted.

## Historical builder-sandbox record

The original report dated `2026-10-31T05:15:07.506Z` described 36 passing
executable tests and seven dependency import errors in the initial builder
sandbox. Package installation and pretrained weights were unavailable there;
PyTorch/Transformers contracts, real TF-IDF integration and new model
measurements had not executed in that environment. Its smoke covered
preparation and artifact integrity only. This is the timestamp of the original
report, not a supplied timestamp for the failed commands. The later worker
outputs above supersede those execution limitations.
