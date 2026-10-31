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
