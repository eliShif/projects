"""PyTorch contracts, including an explicitly random local BERT test checkpoint.

These tests require the real dependencies. No skip or simulated library fallback
turns a missing dependency into a passed encoder test. No quality score is claimed.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizerFast

from portfolio.dense_retrieval import DenseRetriever, FrozenEncoder, local_model_manifest, masked_pool, normalize_embeddings
from portfolio.retrieval import Document

ROOT = Path(__file__).resolve().parent.parent


class PoolingTests(unittest.TestCase):
    def test_masked_mean_matches_hand_calculation(self):
        hidden = torch.tensor([[[3., 0.], [0., 4.], [999., -999.]],
                               [[0., 5.], [88., 99.], [-77., 66.]]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        pooled = masked_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[0.6, 0.8], [0., 1.]]))
        self.assertEqual(pooled.shape, (2, 2))
        extra = torch.cat([hidden, torch.full((2, 4, 2), 1234.)], dim=1)
        extra_mask = torch.cat([mask, torch.zeros((2, 4), dtype=torch.long)], dim=1)
        torch.testing.assert_close(masked_pool(extra, extra_mask), pooled)

    def test_cls_is_a_separate_pooling_ablation(self):
        hidden = torch.tensor([[[3., 4.], [9., 0.]]])
        mask = torch.ones((1, 2), dtype=torch.long)
        torch.testing.assert_close(masked_pool(hidden, mask, "cls"), torch.tensor([[0.6, 0.8]]))
        self.assertFalse(torch.allclose(masked_pool(hidden, mask, "mean"), masked_pool(hidden, mask, "cls")))

    def test_zero_empty_nonfinite_and_mismatched_tensors_are_rejected(self):
        good = torch.ones((1, 2, 3))
        cases = [(good, torch.zeros((1, 2))), (good, torch.ones((2, 1))),
                 (good, torch.tensor([[1., 0.5]])), (good, torch.tensor([[1., float("nan")]])),
                 (torch.zeros((1, 2, 3)), torch.ones((1, 2))),
                 (torch.empty((0, 2, 3)), torch.empty((0, 2))),
                 (torch.ones((1, 2, 3), dtype=torch.long), torch.ones((1, 2))),
                 (torch.full((1, 2, 3), float("inf")), torch.ones((1, 2))),
                 (torch.full((1, 2, 3), float("nan")), torch.tensor([[1, 0]]))]
        for hidden, mask in cases:
            with self.subTest(shape=tuple(hidden.shape)), self.assertRaises(ValueError):
                masked_pool(hidden, mask)
        with self.assertRaises(ValueError):
            masked_pool(good, torch.tensor([[0, 1]]), "cls")
        with self.assertRaises(ValueError):
            masked_pool(good, torch.ones((1, 2)), "unknown")

    def test_norm_checks_dimension_and_nonfinite_values(self):
        for vectors in (torch.ones(3), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.long),
                        torch.empty(1, 0), torch.tensor([[float("nan"), 1.]])):
            with self.assertRaises(ValueError):
                normalize_embeddings(vectors)
        torch.testing.assert_close(normalize_embeddings(torch.tensor([[3., 4.]])), torch.tensor([[0.6, 0.8]]))


class DenseRankingTests(unittest.TestCase):
    def setUp(self):
        self.docs = [Document("a", "first"), Document("b", "second"), Document("c", "third")]

    def test_cosine_ranking_ties_and_evidence_contract(self):
        class VectorFixture:
            def encode(self, texts, *, pooling):
                table = {"first": [1., 0.], "second": [1., 0.], "third": [-1., 0.], "query": [2., 0.]}
                return torch.tensor([table[t] for t in texts])
        retriever = DenseRetriever(self.docs, VectorFixture())
        hits = retriever.search("query", 3)
        self.assertEqual([h["id"] for h in hits], ["a", "b", "c"])
        self.assertEqual([h["score"] for h in hits], [1., 1., -1.])
        self.assertEqual([h["text"] for h in hits], [d.text for d in self.docs])
        self.assertEqual(len(retriever.search("query", 1)), 1)

    def test_invalid_queries_and_dimensions_fail(self):
        class VectorFixture:
            def encode(self, texts, *, pooling):
                return torch.ones(len(texts), 3 if texts == ["bad dimension"] else 2)
        retriever = DenseRetriever(self.docs, VectorFixture())
        for query, k in (("", 1), ("  ", 1), ("x" * 4001, 1), (None, 1), ("first", True), ("first", 11)):
            with self.assertRaises(ValueError):
                retriever.search(query, k)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            retriever.search("bad dimension")
        with self.assertRaises(ValueError):
            DenseRetriever([], VectorFixture())
        with self.assertRaises(ValueError):
            DenseRetriever([self.docs[0], self.docs[0]], VectorFixture())

    def test_nonfinite_or_wrong_document_count_fails(self):
        class VectorFixture:
            def __init__(self, output):
                self.output = output
            def encode(self, texts, *, pooling):
                return self.output
        for output in (torch.tensor([[float("nan"), 1.]] * 3), torch.ones((2, 2))):
            with self.assertRaises(ValueError):
                DenseRetriever(self.docs, VectorFixture(output))


class LocalEncoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        cls.path = Path(cls.tmp.name) / "random-test-checkpoint"
        cls.path.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "short", "long", "שלום", "ספר"]
        (cls.path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(cls.path / "vocab.txt"), do_lower_case=False)
        tokenizer.save_pretrained(cls.path)
        torch.manual_seed(27)
        config = BertConfig(vocab_size=len(vocab), hidden_size=12, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=3, max_position_embeddings=64,
                            hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5)
        BertModel(config).save_pretrained(cls.path, safe_serialization=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def encoder(self, **options):
        return FrozenEncoder(self.path, max_length=16, threads=1, **options)

    def test_actual_local_loading_has_no_remote_or_pickle_fallback(self):
        with patch("transformers.AutoModel.from_pretrained", wraps=AutoModel.from_pretrained) as model_loader, \
                patch("transformers.AutoTokenizer.from_pretrained", wraps=AutoTokenizer.from_pretrained) as tokenizer_loader:
            encoder = self.encoder()
        self.assertTrue(model_loader.call_args.kwargs["local_files_only"])
        self.assertTrue(model_loader.call_args.kwargs["use_safetensors"])
        self.assertFalse(model_loader.call_args.kwargs["trust_remote_code"])
        self.assertTrue(tokenizer_loader.call_args.kwargs["local_files_only"])
        self.assertFalse(tokenizer_loader.call_args.kwargs["trust_remote_code"])
        self.assertTrue(encoder.manifest["version"].startswith("sha256:"))

    def test_padding_batch_invariance_shapes_norms_and_freezing(self):
        encoder = self.encoder(batch_size=2)
        single = encoder.encode(["hello"])
        encoder.model.train()  # encode must restore eval mode even after caller mutation
        batched = encoder.encode(["hello", "hello world long short שלום ספר"])
        torch.testing.assert_close(single[0], batched[0], atol=1e-6, rtol=1e-5)
        self.assertEqual(batched.shape, (2, 12))
        torch.testing.assert_close(torch.linalg.vector_norm(batched, dim=1), torch.ones(2))
        self.assertFalse(batched.requires_grad)
        self.assertFalse(encoder.model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.model.parameters()))
        repeated = encoder.encode(["hello"])
        torch.testing.assert_close(single, repeated)

    def test_truncation_and_inference_budget_are_enforced(self):
        encoder = self.encoder(batch_size=1, max_texts=1, max_forward_calls=1)
        vectors = encoder.encode(["hello " * 100])
        self.assertEqual(vectors.shape, (1, 12))
        self.assertEqual(encoder.stats["padded_token_slots"], 16)
        self.assertLessEqual(encoder.stats["attended_tokens"], 16)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            encoder.encode(["world"])
        self.assertEqual(encoder.stats["forward_calls"], 1)

    def test_empty_inputs_fail_before_forward_pass(self):
        encoder = self.encoder()
        for texts in ([], [""], ["  "], "hello", [None], ["x" * 4001]):
            with self.assertRaises(ValueError):
                encoder.encode(texts)
        self.assertEqual(encoder.stats["forward_calls"], 0)

    def test_random_checkpoint_comparison_is_only_a_pipeline_contract(self):
        from portfolio.retrieval_experiment import DEFAULT_DATA, audit, run
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            out = Path(directory) / "contract-only"
            summary = run(DEFAULT_DATA, out, model_path=self.path,
                          model_id="random-BERT-test-fixture-not-pretrained",
                          max_length=16, threads=1)
            self.assertEqual(audit(out)["status"], "complete")
            self.assertEqual(set(summary["test"]), {"tfidf", "dense_mean", "dense_cls"})
            self.assertEqual(summary["encoder_version"], local_model_manifest(self.path)["version"])
            self.assertEqual(summary["encoder_statistics"]["texts"], 72)
            self.assertEqual(summary["encoder_statistics"]["forward_calls"], 52)
            # No accuracy assertion: random weights cannot test the hypothesis.

    def test_missing_weights_invalid_bounds_and_unsafe_shard_reference(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory)
            with self.assertRaises(FileNotFoundError):
                FrozenEncoder(path)
            (path / "config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "safetensors"):
                FrozenEncoder(path)
            (path / "model-00001-of-00001.safetensors").write_bytes(b"not loaded in this manifest-only test")
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "shard"):
                local_model_manifest(path)
        for options in ({"threads": 0}, {"max_length": 513}, {"batch_size": True}, {"max_texts": 0}):
            with self.assertRaises(ValueError):
                FrozenEncoder(self.path, **options)


if __name__ == "__main__":
    unittest.main()
