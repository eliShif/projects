# Reproducible validation

Verified at 2026-10-30T21:20:29.361Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-p","test_research*.py","-v"]}`

Exit code: 0

Output SHA-256: `3444a4738a2c6d6767607643ca15edc20b12670fdfae049b35254341c1a7b81c`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio","tests","scripts"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["scripts/smoke_research_lab.py","--out","results/research_lab","--installed"]}`

Exit code: 0

Output SHA-256: `ebb52f5006b0bf653afee6d26d51e6bb887964246942ddedbf723f3aecc90036`

Raw execution output remains in private application storage.
