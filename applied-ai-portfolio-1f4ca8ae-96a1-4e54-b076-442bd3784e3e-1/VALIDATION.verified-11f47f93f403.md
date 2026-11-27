# Reproducible validation

Verified at 2026-10-31T14:15:10.319Z

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `10c1d57e48df13cacc86de9c82d466dfb4772740a128ffd0a67d3273d7513a01`

## build

Command: `{"executable":".venv/bin/python","args":["-m","compileall","-q","portfolio","tests","scripts"]}`

Exit code: 0

Output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","portfolio.smoke"]}`

Exit code: 0

Output SHA-256: `39e9bb208bb3953b56533fdbafc6fd365356312c24cfd782c7d4be1f75283ac7`

Raw execution output remains in private application storage.

## Classical vision and signal-processing extension (later worker validation)

The sections above predate the classical extension and are retained as an earlier record (their timestamp is not repeated here). The commands below were run later by the worker with the extension files present, on Python 3.14 with torch 2.15.0+cpu.

- tests: `.venv/bin/python -m unittest discover -s tests -v`. Exit code 0; Ran 121 tests in 167.110 s, OK. Output SHA-256: `e18342015acb5bb515bc80c1e7d8d7c6141e3cecea8cd6a4bf542d4622e65918`. The native tests ran with the temporary-build fallback, because the build step ran after test discovery.
- build: `g++ -O2 -std=c++17 -Wall -Wextra -o cpp/classical_filters.bin cpp/filters.cpp`. Exit code 0; empty output (SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
- documented_run: `.venv/bin/python -m portfolio.classical_benchmark --require-native --repeats 1`. Exit code 0; all_passed true; native_executed true; seed 11; max absolute C++/NumPy differences from 0.0 to 3.4e-14 against a 1e-9 tolerance. Output SHA-256: `73be8f8919c1b007e957c8264dbd09292e3e69dfae2a4d2c5c43bd0ffc1b585e`.

Earlier builder-session record: a previous validation of these files reported 8 errors and 1 failure, from `cv.NATIVE` references and flat-image edge detection. These were fixed and are not current failures. These results are software checks and a single-repeat smoke run on synthetic data. They do not establish performance, real-data accuracy or production readiness.
