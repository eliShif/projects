# Reproducible validation

Verified at 2026-10-28T18:12:28.660Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-p","test_research*.py","-v"]}`

Exit code: 0

Output SHA-256: `4d5a2f898fb406da910305dd23e71955966ed7f285325197e9749aa77d0b23d5`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio","tests","scripts"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["scripts/smoke_research_lab.py","--out","results/research_lab","--installed"]}`

Exit code: 0

Output SHA-256: `95e536ca944fc1ea346e57866de594249bdb6d4bb7b16cf3ae7fdeff82a3cc92`

Raw execution output remains in private application storage.
