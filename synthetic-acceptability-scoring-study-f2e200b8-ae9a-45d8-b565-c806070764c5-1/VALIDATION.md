# Reproducible validation

Verified at 2026-10-28T12:56:29.780Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `3b037854d0297c7624a0bbd57b01a2f476e72c8382df8475d31ffcb3f79ebbbb`

## build

Command: `{"executable":"python3","args":["-m","compileall","-q","cola_lab","tests"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":"python3","args":["-m","cola_lab","smoke"]}`

Exit code: 0

Output SHA-256: `3480048b83b7d64d0aa6b21491ccf56845a641f488500793410620260fcdd7be`

Raw execution output remains in private application storage.
