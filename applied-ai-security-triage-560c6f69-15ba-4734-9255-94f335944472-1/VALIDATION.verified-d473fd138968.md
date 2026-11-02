# Reproducible validation

Verified at 2026-11-02T04:58:12.401Z

## tests

Command: `{"executable":".venv/bin/python","args":["scripts/test_model_identity.py"]}`

Exit code: 0

Output SHA-256: `d4d0da55545550881868443bf72d9bed56c2007a00560a42b5ae909068f03dec`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","--isolated","--disable-pip-version-check","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `0585bbd3f260abfa003358c908a2db95cff467732adac00a5194f13e23f9c183`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
