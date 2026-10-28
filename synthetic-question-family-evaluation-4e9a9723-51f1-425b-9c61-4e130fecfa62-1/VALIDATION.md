# Reproducible validation

Verified at 2026-10-28T00:45:21.978Z

## tests

Command: `{"executable":"python3","args":["-m","unittest","discover","-s","tests","-p","test_research_lab.py","-v"]}`

Exit code: 0

Output SHA-256: `8ceccbd4421e46a1cca6a7b1ec4332fe6ec6cc571bf2da34afbcc27bd99023de`

## build

Command: `{"executable":"python3","args":["-m","compileall","-q","portfolio","tests"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":"python3","args":["-m","portfolio.research_lab","--out","results/research_lab"]}`

Exit code: 0

Output SHA-256: `f7a68ffe84295784f93c3ec5d474810b9f182f2175e95020572603882d393bea`

Raw execution output remains in private application storage.
