# Reproducible validation

Verified at 2026-11-02T03:32:21.187Z

## tests

Command: `{"executable":".venv/bin/python","args":["scripts/test_model_identity.py"]}`

Exit code: 0

Output SHA-256: `420d49fff5fa772475a556e6b1fa00ba5488660c105946280ac771021c60a098`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","--isolated","--disable-pip-version-check","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `da619022bfe99a620d1ab1074dc6193b17ad5cc25944273e8eba470f45beff12`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
