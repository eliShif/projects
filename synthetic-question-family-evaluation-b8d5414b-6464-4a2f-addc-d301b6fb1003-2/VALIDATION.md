# Reproducible validation

Verified at 2026-10-30T20:07:23.301Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-p","test_research*.py","-v"]}`

Exit code: 0

Output SHA-256: `d6fdfd8c00bcd774f419a852ff0ef357b2d7669cccfe5671349d624d568a1342`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio","tests","scripts"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["scripts/smoke_research_lab.py","--out","results/research_lab","--installed"]}`

Exit code: 0

Output SHA-256: `fac5b94fa0e8f36a537ec72f5eb16721b48f1e4c8a3012bbdcf81234f255fed8`

Raw execution output remains in private application storage.
