# Reproducible validation

Implementation and tests are unchanged from source commit `f06d214375d45572c92dae6e7229d5f1d652f05b`. These commands ran on that source; documentation publication is not a new run or experiment.

## install

Command: `{"args":["-m","venv",".venv"],"executable":"python3"}`

Started: 2026-11-03T21:19:15.555Z; exit code: 0; output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## install

Command: `{"args":["-m","pip","install","--no-cache-dir","--index-url","https://download.pytorch.org/whl/cpu","torch>=2.9,<3"],"executable":".venv/bin/python"}`

Started: 2026-11-03T21:19:19.721Z; exit code: 0; output SHA-256: `0f174aecf6637cb3db9c694d2903bf3ed6cc05d4cfc65f0bbd2c707f7b1df2fd`.

## install

Command: `{"args":["-I","-c","import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in [\"torch\"]}))"],"executable":".venv/bin/python"}`

Started: 2026-11-03T21:20:06.755Z; exit code: 0; output SHA-256: `51343ec64029252cbcd9e79711538d1c25db8ed2057af14f9f46b124a6b162ec`.

## install

Command: `{"args":["-m","pip","install","-r","requirements.txt","--no-cache-dir","--constraint",".report-cpu-constraints.txt"],"executable":".venv/bin/python"}`

Started: 2026-11-03T21:20:06.917Z; exit code: 0; output SHA-256: `688891e8508cbd9ef5ee10e5c380542540c91819cad9a7a5c70c9ca30f843af2`.

## tests

Command: `{"args":["-B","-m","unittest","-v","tests/test_adapters.py","tests/test_statistics_retrieval.py","tests/test_evidence_agent.py","tests/test_telemetry.py"],"executable":".venv/bin/python"}`

Started: 2026-11-03T21:20:52.200Z; exit code: 0; output SHA-256: `7e574944fd501c387acdce8493c2fc092d66d65ae9e377e06f18eae52d9701f9`.

## build

Command: `{"args":["-m","compileall","-q","portfolio","tests","scripts"],"executable":".venv/bin/python"}`

Started: 2026-11-03T21:21:01.197Z; exit code: 0; output SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## smoke

Command: `{"args":["-B","-m","portfolio.telemetry","--out",".venv/validation/telemetry"],"executable":".venv/bin/python"}`

Started: 2026-11-03T21:21:01.415Z; exit code: 0; output SHA-256: `0eeb52dde58a9f2e21d11be8aba97506a27cf34d2cf592bab5a3a27f25264bf4`.
