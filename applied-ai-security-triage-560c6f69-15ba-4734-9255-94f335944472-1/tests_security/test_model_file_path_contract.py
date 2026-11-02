"""Preserve the existing enumeration API while identity capture normalizes keys."""
import json
from pathlib import Path
import tempfile
import unittest
from portfolio.security_models import model_files, capture_model_identity, verify_model_identity


class EnumerationPathContract(unittest.TestCase):
    def test_relative_absolute_and_directory_alias_return_paths_keep_caller_root(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            root = Path(name)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({"model_type": "llama"}))
            (checkpoint / "model.safetensors").write_bytes(b"CONTROLLED path-enumeration fixture; not tensor loading")
            alias = root / "alias"
            alias.symlink_to("checkpoint", target_is_directory=True)
            expected = capture_model_identity(checkpoint)
            for value in (checkpoint, alias, checkpoint.relative_to(Path.cwd()), alias.relative_to(Path.cwd())):
                with self.subTest(root=value):
                    paths = model_files(value)
                    self.assertEqual({p.relative_to(value).as_posix() for p in paths}, {"config.json", "model.safetensors"})
                    self.assertEqual(capture_model_identity(value), expected)
                    verify_model_identity(value, expected)


if __name__ == "__main__":
    unittest.main()
