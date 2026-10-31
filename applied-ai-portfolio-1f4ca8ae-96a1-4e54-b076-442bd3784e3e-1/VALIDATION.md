# Reproducible validation

Verified at 2026-10-31T14:15:10.319Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `10c1d57e48df13cacc86de9c82d466dfb4772740a128ffd0a67d3273d7513a01`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio","tests","scripts"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.smoke"]}`

Exit code: 0

Output SHA-256: `39e9bb208bb3953b56533fdbafc6fd365356312c24cfd782c7d4be1f75283ac7`

Raw execution output remains in private application storage.
