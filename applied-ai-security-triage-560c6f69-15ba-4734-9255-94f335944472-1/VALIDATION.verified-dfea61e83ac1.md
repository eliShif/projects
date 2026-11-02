# Reproducible validation

Verified at 2026-11-02T06:03:52.951Z

## tests

Command: `{"executable":".venv/bin/python","args":["scripts/test_model_identity.py"]}`

Exit code: 0

Output SHA-256: `742abb4ed7f1ed6a0e1ccc74807a2bd5370e93419351727e6d345ae9c63f1685`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","--isolated","--disable-pip-version-check","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `476383a8ff543494cae1939b599bbbb3f4c4992d10ac9b699700d04924db6948`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
