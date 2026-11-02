# Local checkpoint compatibility verification

The existing loading APIs retain shared Hugging Face cache links and dot-relative shard paths. The strict closed-bundle inventory remains an explicit helper policy. Advertised versioned configuration inputs are included in the saved byte-content identity. A monolithic checkpoint ignores unused index parsing while retaining its file digest.

- `.venv/bin/python scripts/test_model_identity.py`: exit 0, 101 tests. Output SHA-256: `e4247b5b99751ebf0df3108bb6406a691c04c625023e0df3593f954d426dec08`.
- `.venv/bin/python -m pip --isolated --disable-pip-version-check wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .`: exit 0. Output SHA-256: `07e8a3539a6307831ca6413d79fafe8d61eb3fc060136bfcc67b7263355732ac`.
- `.venv/bin/python -m portfolio.security_smoke`: exit 0. Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`.
- `.venv/bin/python -m unittest discover -s tests -v`: exit 0, 43 tests. Output SHA-256: `e0aa34d0185b703982a057753712755d4ecde719ab90936247afefa054c788ea`.

All 94 original published test/result files retain their exact bytes. Earlier failed and successful recorded runs are preserved as historical evidence. The new tests use small scalar files and real tiny random causal-language models with one-step adapter training; they establish loader compatibility and integrity enforcement, not general security-model quality. Hashes do not authorize filesystem access or lock files between verification and loading. All raw validation output remains in private acceptance storage.
