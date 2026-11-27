# Reproducible validation

Verified at 2026-11-25T16:52:43.956Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `e18342015acb5bb515bc80c1e7d8d7c6141e3cecea8cd6a4bf542d4622e65918`

## build

Command: `{"executable":"g++","args":["-O2","-std=c++17","-Wall","-Wextra","-o","cpp/classical_filters.bin","cpp/filters.cpp"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.classical_benchmark","--require-native","--repeats","1"]}`

Exit code: 0

Output SHA-256: `73be8f8919c1b007e957c8264dbd09292e3e69dfae2a4d2c5c43bd0ffc1b585e`

Raw execution output remains in private application storage.
