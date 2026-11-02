"""Legacy loading layouts retain content integrity; strict bundle tools remain strict."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from portfolio.model_identity import ModelIdentityError
from portfolio.security_models import capture_model_identity, verify_model_identity, model_files
from portfolio.security_pipeline import Pipeline
from tests_security.test_model_identity import identity_fixture, transformer_boundaries
from scripts.check_layout_preservation import compare_repair


class StandardCheckpointCompatibility(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "snapshot"
        identity_fixture(self.model)

    def test_external_shared_blobs_are_bound_by_logical_name_and_changed_bytes_fail(self):
        blobs = self.root / "blobs"
        blobs.mkdir()
        for name in ("config.json", "model.safetensors"):
            target = blobs / name
            (self.model / name).rename(target)
            (self.model / name).symlink_to("../blobs/" + name)
        expected = capture_model_identity(self.model)
        self.assertIn("config.json", expected)
        self.assertIn("model.safetensors", expected)
        verify_model_identity(self.model, expected)
        original = (blobs / "config.json").read_bytes()
        (blobs / "config.json").write_bytes(original + b" ")
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.model, expected)
        (blobs / "config.json").write_bytes(original)
        verify_model_identity(self.model, expected)

    def test_dot_relative_shards_are_normalized_and_hashed(self):
        (self.model / "shards").mkdir()
        (self.model / "model.safetensors").rename(self.model / "shards/part.safetensors")
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "./shards/part.safetensors"}}))
        expected = capture_model_identity(self.model)
        self.assertIn("shards/part.safetensors", expected)
        verify_model_identity(self.model, expected)
        (self.model / "shards/part.safetensors").write_bytes(b"changed")
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.model, expected)

    def test_monolithic_checkpoint_does_not_parse_an_unused_malformed_index(self):
        index = self.model / "model.safetensors.index.json"
        for value in ("{", json.dumps({"weight_map": {"x": "absent.safetensors"}})):
            index.write_text(value)
            self.assertTrue(model_files(self.model))
            expected = capture_model_identity(self.model)
            self.assertIn(index.name, expected)
            verify_model_identity(self.model, expected)

    def test_advertised_versioned_configuration_mutation_and_appearance_are_bound(self):
        config = json.loads((self.model / "config.json").read_text())
        config["configuration_files"] = ["config.4.0.0.json", "config.99.0.0.json"]
        (self.model / "config.json").write_text(json.dumps(config))
        variant = self.model / "config.4.0.0.json"
        variant.write_text('{"model_type":"llama","hidden_size":16}')
        expected = capture_model_identity(self.model)
        self.assertIn(variant.name, expected)
        variant.write_text('{"model_type":"llama","hidden_size":17}')
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.model, expected)
        variant.write_text('{"model_type":"llama","hidden_size":16}')
        verify_model_identity(self.model, expected)
        (self.model / "config.99.0.0.json").write_text("{}")
        with self.assertRaises(ModelIdentityError):
            verify_model_identity(self.model, expected)

    def aliased_shards(self):
        (self.model / "model.safetensors").rename(self.model / "a.safetensors")
        (self.model / "payloads/subdir").mkdir(parents=True)
        other = self.model / "payloads/a.safetensors"
        other.write_bytes((self.model / "a.safetensors").read_bytes() + b" ")
        (self.model / "link").symlink_to("payloads/subdir", target_is_directory=True)
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"first": "a.safetensors", "second": "link/../a.safetensors"}}))
        return self.model / "a.safetensors", other

    def test_symlink_parent_references_keep_distinct_byte_identities(self):
        files = self.aliased_shards()
        expected = capture_model_identity(self.model)
        self.assertNotEqual(expected["a.safetensors"], expected["link/../a.safetensors"])
        for file in files:
            original = file.read_bytes()
            file.write_bytes(original + b"changed")
            with self.assertRaises(ModelIdentityError):
                verify_model_identity(self.model, expected)
            file.write_bytes(original)
            verify_model_identity(self.model, expected)

    def test_queued_training_rejects_either_aliased_shard_before_tokenizer_loading(self):
        for index, file in enumerate(self.aliased_shards()):
            with self.subTest(file=file.name, index=index):
                pipeline = Pipeline(self.root / ("queued-" + str(index)), backend="pretrained",
                                    steps=1, model_path=str(self.model))
                pipeline.work_once(pipeline.enqueue("prepare"))
                job = pipeline.enqueue("train")
                original = file.read_bytes()
                file.write_bytes(original + b"changed")
                with transformer_boundaries() as (tokenizer, model):
                    with self.assertRaises(ModelIdentityError):
                        pipeline.work_once(job)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                state = pipeline.store.get(job)
                self.assertEqual(state["status"], "queued")  # Bounded retry after the first failure.
                self.assertEqual(state["error"], "stage_error:ModelIdentityError")
                self.assertIsNone(state["result"])
                self.assertFalse((pipeline.root / "manifest.json").exists())
                file.write_bytes(original)
                verify_model_identity(self.model, pipeline.protocol["model_files_sha256"])

    def test_ephemeral_host_constraint_is_not_source_but_result_files_remain_protected(self):
        result = self.root / "results/.report-cpu-constraints.txt"
        result.parent.mkdir()
        result.write_bytes(b"historical result")
        baseline = {"source_commit": "fixture", "files": {
            ".report-cpu-constraints.txt": {"sha256": hashlib.sha256(b"host constraint").hexdigest()},
            "results/.report-cpu-constraints.txt": {"sha256": hashlib.sha256(result.read_bytes()).hexdigest()}}}
        check = compare_repair(self.root, baseline)
        self.assertTrue(check["ok"], check)
        self.assertEqual(check["ignored_generated_files"], [".report-cpu-constraints.txt"])
        self.assertEqual(check["preserved_result_files"], 1)
        result.unlink()
        self.assertFalse(compare_repair(self.root, baseline)["ok"])
