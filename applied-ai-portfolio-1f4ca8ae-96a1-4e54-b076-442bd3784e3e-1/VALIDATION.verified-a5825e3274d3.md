# Reproducible validation

Verified at 2026-11-27T13:30:39.158Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `4b7a896727191c7f77c5f962450c3be01ad03b497eb05c02e4e87cca5d2a057b`

## build

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.classical_native"]}`

Exit code: 0

Output SHA-256: `0d1052fb2da6a7be425653646989c5ad28839a8bc2356bca5fbc748e2649e0b2`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.classical_benchmark","--require-native","--repeats","1"]}`

Exit code: 0

Output SHA-256: `cdb2197727a9ff719297e013835bb0e5500ea6f48175e6a07fcb4e1ca0e97ebc`

Raw execution output remains in private application storage.
