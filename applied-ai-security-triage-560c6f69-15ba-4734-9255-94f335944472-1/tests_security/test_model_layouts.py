"""Local checkpoint layout compatibility and identity regressions.

Small scalar files exercise inventory behavior; real Transformers compatibility
and queue controls are tested separately in tests_ml_identity.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portfolio.model_identity import (
    ModelIdentityError, capture_model_identity, model_files, verify_model_identity,
)
from tests_security.test_model_identity import identity_fixture, change_same_size


class ModelLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "checkpoint"
        identity_fixture(self.model)

    def link_artifact(self, name):
        target = self.model / "payloads" / name
        target.parent.mkdir(exist_ok=True)
        (self.model / name).rename(target)
        (self.model / name).symlink_to("payloads/" + name)
        return target

    def test_directory_alias_and_internal_file_links_keep_logical_identities(self):
        expected = capture_model_identity(self.model)
        for name in expected:
            self.link_artifact(name)
        alias = self.root / "alias"
        alias.symlink_to("checkpoint", target_is_directory=True)
        self.assertEqual(capture_model_identity(alias), expected)
        self.assertEqual({p.relative_to(alias).as_posix() for p in model_files(alias)}, set(expected))
        verify_model_identity(alias, expected)
        # Absolute links and a chain entirely inside the checkpoint also work.
        link = self.model / "config.json"
        link.unlink()
        (self.model / "config-link").symlink_to(self.model / "payloads/config.json")
        link.symlink_to("config-link")
        verify_model_identity(alias, expected)

    def test_internal_directory_links_cover_shards_and_chat_templates(self):
        (self.model / "payloads").mkdir()
        (self.model / "model.safetensors").rename(self.model / "payloads/part.safetensors")
        (self.model / "shards").symlink_to("payloads", target_is_directory=True)
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"fixture": "shards/part.safetensors"}}))
        (self.model / "templates").mkdir()
        (self.model / "templates/default.jinja").write_text("{{ messages }}")
        (self.model / "chat_templates").symlink_to("templates", target_is_directory=True)
        expected = capture_model_identity(self.model)
        self.assertIn("shards/part.safetensors", expected)
        self.assertIn("chat_templates/default.jinja", expected)
        verify_model_identity(self.model, expected)
        (self.model / "templates/default.jinja").write_text("{{ changed }}")
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.model, expected)

    def test_same_size_mutations_follow_internal_link_targets(self):
        for name in ("config.json", "model.safetensors"):
            with self.subTest(name=name):
                target = self.link_artifact(name)
                expected = capture_model_identity(self.model)
                original = change_same_size(self.model, name)
                with self.assertRaisesRegex(ModelIdentityError, "SHA-256"):
                    verify_model_identity(self.model, expected)
                target.write_bytes(original)
                verify_model_identity(self.model, expected)

    def test_escape_even_to_identical_bytes_rejects_before_read(self):
        for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
            with self.subTest(name=name):
                expected = capture_model_identity(self.model)
                original = (self.model / name).read_bytes()
                outside = self.root / name
                outside.write_bytes(original)
                link = self.model / name
                link.unlink()
                link.symlink_to(outside)
                real_open = Path.open

                def guarded_open(path, *args, **kwargs):
                    self.assertNotEqual(path.resolve(), outside, "Escaping artifact was opened")
                    return real_open(path, *args, **kwargs)

                with patch.object(Path, "open", guarded_open):
                    with self.assertRaises(ModelIdentityError):
                        verify_model_identity(self.model, expected)
                link.unlink()
                link.write_bytes(original)

    def test_retargeted_internal_file_and_model_directory_reject_changed_bytes(self):
        target = self.link_artifact("config.json")
        expected = capture_model_identity(self.model)
        replacement = self.model / "replacement.json"
        replacement.write_bytes(target.read_bytes().replace(b"16", b"17"))
        link = self.model / "config.json"
        link.unlink()
        link.symlink_to("replacement.json")
        with self.assertRaisesRegex(ModelIdentityError, "SHA-256"):
            verify_model_identity(self.model, expected)
        link.unlink()
        link.symlink_to("payloads/config.json")
        alias = self.root / "alias"
        alias.symlink_to("checkpoint", target_is_directory=True)
        verify_model_identity(alias, expected)
        other = self.root / "other"
        identity_fixture(other)
        change_same_size(other, "model.safetensors")
        alias.unlink()
        alias.symlink_to("other", target_is_directory=True)
        with self.assertRaisesRegex(ModelIdentityError, "SHA-256"):
            verify_model_identity(alias, expected)

    def test_broken_links_loops_and_escaping_directories_reject(self):
        for target in ("missing.json", "config.json"):
            with self.subTest(target=target):
                config = self.model / "config.json"
                config.unlink()
                config.symlink_to(target)
                with self.assertRaises(ModelIdentityError):
                    capture_model_identity(self.model)
        (self.model / "config.json").unlink()
        identity_fixture(self.model)
        outside = self.root / "templates"
        outside.mkdir()
        (outside / "default.jinja").write_text("outside fixture")
        (self.model / "chat_templates").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ModelIdentityError):
            capture_model_identity(self.model)


class VersionedTokenizerInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / "checkpoint"
        identity_fixture(self.model)

    def configure(self, names):
        (self.model / "tokenizer_config.json").write_text(json.dumps({"fast_tokenizer_files": names}))

    def test_all_present_advertised_versions_are_bound_with_add_remove_and_mutation(self):
        names = ["tokenizer.4.0.0.json", "tokenizer.4.50.0.json", "tokenizer.99.0.0.json"]
        self.configure(names)
        for name in names[:2]:
            (self.model / name).write_text('{"fixture":"a"}')
        expected = capture_model_identity(self.model)
        self.assertTrue(set(names[:2]).issubset(expected))
        # An absent, unselected future version need not prevent local loading.
        verify_model_identity(self.model, expected)
        for name in names[:2]:
            with self.subTest(name=name):
                path = self.model / name
                original = path.read_bytes()
                path.write_bytes(original.replace(b'"a"', b'"b"'))
                with self.assertRaisesRegex(ModelIdentityError, "SHA-256"):
                    verify_model_identity(self.model, expected)
                path.unlink()
                with self.assertRaises(ModelIdentityError):
                    verify_model_identity(self.model, expected)
                path.write_bytes(original)
        (self.model / names[-1]).write_text("{}")
        with self.assertRaisesRegex(ModelIdentityError, "file set"):
            verify_model_identity(self.model, expected)

    def test_nested_versioned_tokenizer_link_uses_logical_reference(self):
        name = "tokenizers/tokenizer.4.0.0.json"
        self.configure([name])
        (self.model / "tokenizers").mkdir()
        (self.model / "payload.json").write_text("{}")
        (self.model / name).symlink_to("../payload.json")
        expected = capture_model_identity(self.model)
        self.assertIn(name, expected)
        verify_model_identity(self.model, expected)

    def test_invalid_and_escaping_references_are_rejected_without_opening_targets(self):
        outside = self.model.parent / "tokenizer.4.0.0.json"
        outside.write_text("outside fixture")
        for names in (None, "tokenizer.4.0.0.json", [1], ["../tokenizer.4.0.0.json"],
                      [str(outside)], ["dir/../tokenizer.4.0.0.json"], ["dir\\tokenizer.4.0.0.json"]):
            with self.subTest(names=names):
                self.configure(names)
                real_open = Path.open

                def guarded_open(path, *args, **kwargs):
                    self.assertNotEqual(path.resolve(), outside, "External tokenizer was opened")
                    return real_open(path, *args, **kwargs)

                with patch.object(Path, "open", guarded_open):
                    with self.assertRaises(ModelIdentityError):
                        capture_model_identity(self.model)
        self.configure(["tokenizer.4.0.0.json"])
        (self.model / "tokenizer.4.0.0.json").symlink_to(outside)
        with self.assertRaises(ModelIdentityError):
            capture_model_identity(self.model)


if __name__ == "__main__":
    unittest.main()
