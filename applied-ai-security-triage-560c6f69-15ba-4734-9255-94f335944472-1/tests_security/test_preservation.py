"""Keep new validation separate from inherited tests, datasets and results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_preservation import compare


class PreservationTests(unittest.TestCase):
    def test_supplied_tests_results_and_other_unaffected_files_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / "results/model_identity_validation/baseline.json").read_text())
        result = compare(root, baseline)
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["preserved_tests"], 0)
        self.assertGreater(result["preserved_result_files"], 0)

    def test_deleted_source_and_replaced_results_are_detected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            files = {"portfolio/original.py": b"original source", "results/prior.json": b"original result"}
            for name, raw in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            baseline = {"source_commit": "fixture", "files": {
                name: {"sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()}}
            self.assertTrue(compare(root, baseline)["ok"])
            (root / "portfolio/original.py").unlink()
            (root / "results/prior.json").write_bytes(b"replacement")
            result = compare(root, baseline)
            self.assertFalse(result["ok"])
            self.assertEqual(result["missing_files"], ["portfolio/original.py"])
            self.assertEqual(result["unexpected_changes"], ["results/prior.json"])
