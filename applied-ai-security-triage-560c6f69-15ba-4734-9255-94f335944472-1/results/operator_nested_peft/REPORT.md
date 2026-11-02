# Local checkpoint compatibility verification

The newest loading-reference run includes actual CWD/absolute tokenizer overrides, the unused-sidecar non-PEFT branch, actual PEFT adapters with local and offline cached bases, base mutation rejection and restoration. No quality experiment is inferred. The identity inventory preserves distinct symlink/parent shard references instead of collapsing different loaded files. The existing loading APIs retain shared Hugging Face cache links and dot-relative shard paths. The strict closed-bundle inventory remains an explicit helper policy. Advertised versioned configuration inputs are included in the saved byte-content identity. A monolithic checkpoint ignores unused index parsing while retaining its file digest.

- `.venv/bin/python scripts/test_model_identity.py`: exit 0, 114 tests. Output SHA-256: `a9e3f1757bcd020112e4819d121878f91858d53ac698e680fb4e835d5dce4318`.
- `.venv/bin/python -m pip --isolated --disable-pip-version-check wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .`: exit 0. Output SHA-256: `f1726ec2966ce9be3d6af1f594d7c77ed6241fde55f836d35f9504cd6b3e82ff`.
- `.venv/bin/python -m portfolio.security_smoke`: exit 0. Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`.
- `.venv/bin/python -m unittest discover -s tests -v`: exit 0, 43 tests. Output SHA-256: `39b86c5bc65ee7762b017ba36c24200e6a05b8b94614229556fa439362697158`.

All 94 original published test/result files retain their exact bytes. Earlier failed and successful recorded runs are preserved as historical evidence. The new tests use small scalar files and real tiny random causal-language models with one-step adapter training; they establish loader compatibility and integrity enforcement, not general security-model quality. Hashes do not authorize filesystem access or lock files between verification and loading. All raw validation output remains in private acceptance storage.
