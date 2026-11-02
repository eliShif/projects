"""Packaging output may change; original source, tests and results may not."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.check_preservation import compare


class PackagingPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def inventory(self, names):
        files = {}
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = b"original bytes"
            path.write_bytes(raw)
            files[name] = {"sha256": hashlib.sha256(raw).hexdigest()}
        return {"source_commit": "fixture", "files": files}

    def test_regenerated_or_missing_packaging_outputs_do_not_fail_preservation(self):
        generated = ["applied_ai_security_triage.egg-info/PKG-INFO",
                     "applied_ai_security_triage.egg-info/SOURCES.txt",
                     ".venv/lib/installed.py", "build/lib/portfolio/copy.py",
                     "dist/package.whl"]
        baseline = self.inventory([*generated, "portfolio/original.py",
                                   "tests/test_original.py", "results/prior.json"])
        for name in generated:
            (self.root / name).write_bytes(b"regenerated metadata or dependencies")
        changed = compare(self.root, baseline)
        self.assertTrue(changed["ok"], changed)
        self.assertEqual(changed["ignored_generated_files"], sorted(generated))
        self.assertEqual(changed["protected_files"], 3)
        self.assertEqual(changed["preserved_tests"], 1)
        self.assertEqual(changed["preserved_result_files"], 1)
        for name in generated:
            (self.root / name).unlink()
        missing = compare(self.root, baseline)
        self.assertTrue(missing["ok"], missing)
        self.assertEqual(missing["missing_files"], [])

    def test_generated_exclusions_do_not_hide_protected_changes(self):
        # Only project-root build/dependency output is disposable. A result or
        # source subtree named 'dist'/'build' remains protected.
        protected = ["portfolio/original.py", "tests/test_original.py",
                     "results/prior.json", "results/build/report.json",
                     "results/dist/scores.json", "portfolio/build/logic.py",
                     "docs/package.egg-info/notes.md"]
        baseline = self.inventory([*protected, "applied_ai_security_triage.egg-info/PKG-INFO"])
        for name in protected:
            path = self.root / name
            with self.subTest(name=name, mutation="changed"):
                path.write_bytes(b"unexpected replacement")
                result = compare(self.root, baseline)
                self.assertFalse(result["ok"])
                self.assertEqual(result["unexpected_changes"], [name])
            with self.subTest(name=name, mutation="missing"):
                path.unlink()
                result = compare(self.root, baseline)
                self.assertFalse(result["ok"])
                self.assertEqual(result["missing_files"], [name])
            path.write_bytes(b"original bytes")


if __name__ == "__main__":
    unittest.main()
