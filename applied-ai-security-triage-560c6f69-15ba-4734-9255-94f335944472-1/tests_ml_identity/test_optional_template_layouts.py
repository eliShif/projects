"""Optional unused template entries must not narrow real local model loading."""
from portfolio.local_lm import LocalLM
from portfolio.security_models import capture_model_identity, PretrainedPolicy
from tests_ml_identity.test_checkpoint_layouts import CheckpointCase


class OptionalTemplateLayouts(CheckpointCase):
    def test_unused_non_directory_templates_preserve_actual_loading(self):
        path = self.model / "chat_templates"
        for kind in ("regular_file", "dangling_link", "self_link"):
            with self.subTest(kind=kind):
                if kind == "regular_file":
                    path.write_text("Unused optional entry; actual root chat template stays valid.")
                else:
                    path.symlink_to("absent-optional-templates" if kind == "dangling_link" else path.name)
                try:
                    self.assertEqual(LocalLM(self.model).model.config.hidden_size, 16)
                    expected = capture_model_identity(self.model)
                    self.assertTrue(any("chat_template" in key for key in expected))
                    self.assertEqual(PretrainedPolicy(self.model, expected_model_files=expected).lm.model.config.hidden_size, 16)
                finally:
                    path.unlink()
