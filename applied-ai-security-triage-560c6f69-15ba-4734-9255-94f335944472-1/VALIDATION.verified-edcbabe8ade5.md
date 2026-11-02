# Reproducible validation

Verified at 2026-11-02T17:19:21.501Z

## tests

Command: `{"executable":".venv/bin/python","args":["scripts/test_model_identity.py"]}`

Exit code: 0

Output SHA-256: `b4f80dc374d8c844db48be2d0048981f356a3a26b615e6b9c4b228e45d6f8ab1`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","--isolated","--disable-pip-version-check","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `0c00945d1a5bde362bab6802ac570e6e4cab923c036225a09d12115b34a0fa39`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
