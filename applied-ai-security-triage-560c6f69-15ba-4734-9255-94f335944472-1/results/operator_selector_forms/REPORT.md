# Local checkpoint compatibility verification

The newest loading-reference run includes actual CWD/absolute tokenizer overrides, the unused-sidecar non-PEFT branch, actual PEFT adapters with local and offline cached bases, base mutation rejection and restoration. No quality experiment is inferred. The identity inventory preserves distinct symlink/parent shard references instead of collapsing different loaded files. The existing loading APIs retain shared Hugging Face cache links and dot-relative shard paths. The strict closed-bundle inventory remains an explicit helper policy. Advertised versioned configuration inputs are included in the saved byte-content identity. A monolithic checkpoint ignores unused index parsing while retaining its file digest.

- `.venv/bin/python scripts/test_model_identity.py`: exit 0, 122 tests. Output SHA-256: `ed21db08e7de85ed0cc1599765f9353945bf382e60b6592c8d0297d208fc07ce`.
- `.venv/bin/python -m pip --isolated --disable-pip-version-check wheel --no-index --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .`: exit 0. Output SHA-256: `7632549b24cd5dbc2afaa2e854103f543590a703a2f1530b48519a52e827abd8`.
- `.venv/bin/python -m portfolio.security_smoke`: exit 0. Output SHA-256: `cc344e588d2b238cc392d281ae35e0982197a3075cff130ea4a8b1d2ac795a92`.
- `.venv/bin/python -m unittest discover -s tests -v`: exit 0, 43 tests. Output SHA-256: `63603af1e8ce40f8c2a208d4d3406f91078923c9f5d178ad6afc4412b219d4b3`.

All 94 original published test/result files retain their exact bytes. Earlier failed and successful recorded runs are preserved as historical evidence. The new tests use small scalar files and real tiny random causal-language models with one-step adapter training; they establish loader compatibility and integrity enforcement, not general security-model quality. Hashes do not authorize filesystem access or lock files between verification and loading. All raw validation output remains in private acceptance storage.
