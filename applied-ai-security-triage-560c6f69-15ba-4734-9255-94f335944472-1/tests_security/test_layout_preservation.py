"""Preserve previous identity tests, source and all earlier validation results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_layout_preservation import BASELINE, compare_repair


class LayoutPreservationTests(unittest.TestCase):
    def test_recovery_checkpoint_is_preserved_outside_the_focused_repair(self):
        root = Path(__file__).resolve().parents[1]
        result = compare_repair(root, json.loads((root / BASELINE).read_text()))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["baseline_files"], 235)
        self.assertEqual(result["preserved_result_files"], 156)
        self.assertEqual(result["preserved_tests"], 17)
        baseline = json.loads((root / BASELINE).read_text())
        for name, record in baseline["files"].items():
            if name.startswith(("tests/", "tests_security/", "tests_ml/", "tests_ml_identity/")):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), record["sha256"], name)

    def test_changed_old_identity_test_and_result_or_deleted_source_fail(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            names = ("portfolio/common.py", "tests_ml_identity/test_pretrained_identity.py",
                     "results/model_identity_recovery_validation/real_tensor_identity.txt")
            files = {}
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
                files[name] = {"sha256": hashlib.sha256(b"original").hexdigest()}
            baseline = {"source_commit": "fixture", "files": files}
            self.assertTrue(compare_repair(root, baseline)["ok"])
            for name in names:
                with self.subTest(name=name):
                    path = root / name
                    path.write_bytes(b"changed")
                    result = compare_repair(root, baseline)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["unexpected_changes"], [name])
                    path.unlink()
                    result = compare_repair(root, baseline)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["missing_files"], [name])
                    path.write_bytes(b"original")
