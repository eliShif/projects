# Reproducible validation

Verified at 2026-11-01T23:32:28.673Z

## tests

Command: `{"executable":".venv/bin/python","args":["scripts/test_model_identity.py"]}`

Exit code: 0

Output SHA-256: `d220f8dc10804641f8342bbfcebb9bbea6c3f18b1291d83c4b082acd93b54e16`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","--isolated","--disable-pip-version-check","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `7cf9213ae8ae6632a2d4c533604c18482554fda84bd13c89f5890bfb42d27196`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
