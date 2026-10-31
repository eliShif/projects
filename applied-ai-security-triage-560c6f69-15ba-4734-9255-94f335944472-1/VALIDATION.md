# Reproducible validation

Verified at 2026-10-31T15:44:35.381Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests_security","-v"]}`

Exit code: 0

Output SHA-256: `e0aa8105ea20d30b9baf8bb3face029d7729da598d44512c7e01a9fc007fbec8`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `97c17ba8e5ad89923e4a23afd09162b876cb341229d4b1eef25895a6b8ab3d4e`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
