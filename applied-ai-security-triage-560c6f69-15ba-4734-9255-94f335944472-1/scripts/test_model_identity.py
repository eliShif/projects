"""Run security, adapter and real model-identity tests without model downloads.

Missing ML dependencies fail at import; no tensor tests are silently skipped.
The broader inherited suite (including live sockets) remains separately runnable.
"""
from pathlib import Path
import sys
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    suites = [unittest.TestLoader().discover(str(root / directory))
              for directory in ("tests_security", "tests_ml", "tests_ml_identity")]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
