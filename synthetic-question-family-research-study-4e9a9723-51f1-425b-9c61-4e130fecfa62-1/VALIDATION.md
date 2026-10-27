# Reproducible validation

Verified at 2026-10-27T20:03:17.436Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-p","test_research_lab.py","-v"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio/research_lab.py","tests/test_research_lab.py"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.research_lab","--out","results/research_lab"]}`

Exit code: 0

Output SHA-256: `f7a68ffe84295784f93c3ec5d474810b9f182f2175e95020572603882d393bea`

Raw execution output remains in private application storage.
