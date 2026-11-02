"""Real loaders ignore unused non-file children of the optional template folder."""
from portfolio.local_lm import LocalLM
from portfolio.model_identity import ModelIdentityError
from portfolio.security_models import capture_model_identity, verify_model_identity, PretrainedPolicy
from tests_ml_identity.test_checkpoint_layouts import CheckpointCase


class OptionalTemplateChildren(CheckpointCase):
    def test_non_file_template_children_preserve_real_loading_and_regular_files_remain_bound(self):
        templates = self.model / "chat_templates"
        templates.mkdir()
        regular = templates / "extra.jinja"
        regular.write_text("{{ messages }}")
        expected_root = capture_model_identity(self.model)
        for kind in ("directory", "dangling_link", "directory_link"):
            path = templates / "unused.jinja"
            with self.subTest(kind=kind):
                if kind == "directory":
                    path.mkdir()
                elif kind == "dangling_link":
                    path.symlink_to("missing-template")
                else:
                    path.symlink_to(".", target_is_directory=True)
                try:
                    self.assertEqual(LocalLM(self.model).model.config.hidden_size, 16)
                    expected = capture_model_identity(self.model)
                    self.assertEqual(expected, expected_root)
                    self.assertIn("chat_templates/extra.jinja", expected)
                    self.assertEqual(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.config.hidden_size, 16)
                    original = regular.read_bytes()
                    regular.write_bytes(original + b" ")
                    with self.assertRaises(ModelIdentityError):
                        verify_model_identity(self.model, expected)
                    regular.write_bytes(original)
                    verify_model_identity(self.model, expected)
                finally:
                    if kind == "directory":
                        path.rmdir()
                    else:
                        path.unlink()

    def test_closed_bundle_helper_still_rejects_non_file_template_children(self):
        from portfolio.model_identity import capture_model_identity as strict_capture
        templates = self.model / "chat_templates"
        templates.mkdir()
        (templates / "unused.jinja").mkdir()
        with self.assertRaises(ModelIdentityError):
            strict_capture(self.model)
