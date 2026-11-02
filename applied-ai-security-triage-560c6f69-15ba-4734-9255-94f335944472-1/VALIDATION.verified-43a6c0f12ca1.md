# Reproducible validation

Verified at 2026-11-01T22:12:33.238Z

## tests

Command: `{"executable":".venv/bin/python","args":["scripts/test_model_identity.py"]}`

Exit code: 0

Output SHA-256: `59ab947b3083bc985884878b9e3a24fa5e31f9cdc82450ae8c2a63cec9e48388`

## build

Command: `{"executable":".venv/bin/python","args":["-m","pip","wheel","--no-index","--no-cache-dir","--no-build-isolation","--no-deps","--wheel-dir","dist","."]}`

Exit code: 0

Output SHA-256: `f32356208d91524a71eca3f2c272979af69b978bdf4ab6f936f02386f264b522`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.security_smoke"]}`

Exit code: 0

Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`

Raw execution output remains in private application storage.
