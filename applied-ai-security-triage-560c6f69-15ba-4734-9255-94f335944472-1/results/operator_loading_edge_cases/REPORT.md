# Local checkpoint compatibility verification

The newest loading-reference run includes actual CWD/absolute tokenizer overrides, the unused-sidecar non-PEFT branch, actual PEFT adapters with local and offline cached bases, base mutation rejection and restoration. No quality experiment is inferred. The identity inventory preserves distinct symlink/parent shard references instead of collapsing different loaded files. The existing loading APIs retain shared Hugging Face cache links and dot-relative shard paths. The strict closed-bundle inventory remains an explicit helper policy. Advertised versioned configuration inputs are included in the saved byte-content identity. A monolithic checkpoint ignores unused index parsing while retaining its file digest.

- `.venv/bin/python scripts/test_model_identity.py`: exit 0, 118 tests. Output SHA-256: `5a3d1eaf4f9060bcf828e61e1c34490e176ada19d4990fce4c988d5ef8bb86b0`.
- `.venv/bin/python -m pip --isolated --disable-pip-version-check wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .`: exit 0. Output SHA-256: `e88b5c800ce6a54e345c5a0ef22dcb16034d87edadfb3baa75135f5b561d7b29`.
- `.venv/bin/python -m portfolio.security_smoke`: exit 0. Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`.
- `.venv/bin/python -m unittest discover -s tests -v`: exit 0, 43 tests. Output SHA-256: `107cf87784f4f97ba4063be9eac3e1ebab17b14e86ed5315175e6691a72f9be5`.

All 94 original published test/result files retain their exact bytes. Earlier failed and successful recorded runs are preserved as historical evidence. The new tests use small scalar files and real tiny random causal-language models with one-step adapter training; they establish loader compatibility and integrity enforcement, not general security-model quality. Hashes do not authorize filesystem access or lock files between verification and loading. All raw validation output remains in private acceptance storage.
