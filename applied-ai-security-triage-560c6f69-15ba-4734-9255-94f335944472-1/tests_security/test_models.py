import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from portfolio.security_data import build_dataset, observations
from portfolio.security_models import (ReferencePolicy, features, softmax, train_reference,
                                       model_files, PretrainedPolicy)


class ReferenceModelTests(unittest.TestCase):
    def test_low_rank_gradients_match_finite_differences(self):
        model = ReferencePolicy()
        model.state["b"][0][0] = .15
        x = [1., 1., 0., 0., 0., .2, .1, 1.]
        ga, gb = model.gradients(x, 1)
        for key, grad in (("a", ga), ("b", gb)):
            for row, column in ((0, 0), (1, 1)):
                value = model.state[key][row][column]
                model.state[key][row][column] = value + 1e-6
                plus = -math.log(softmax(model.logits(x))[1])
                model.state[key][row][column] = value - 1e-6
                minus = -math.log(softmax(model.logits(x))[1])
                model.state[key][row][column] = value
                self.assertAlmostEqual(grad[row][column], (plus-minus)/2e-6, places=6)

    def test_updates_change_adapter_and_preserve_base(self):
        model = ReferencePolicy()
        before = copy.deepcopy(model.state)
        x = features(observations(build_dataset()["train"])[0])
        for _ in range(4):
            model.update(x, 0)
        self.assertEqual(model.state["base"], before["base"])
        self.assertNotEqual(model.state["a"], before["a"])
        self.assertNotEqual(model.state["b"], before["b"])

    def test_checkpoint_is_validation_selected_and_reproducible(self):
        data = build_dataset()
        args = (observations(data["train"]), data["train"], observations(data["validation"]), data["validation"])
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = train_reference(*args, first, steps=96)
            b = train_reference(*args, second, steps=96)
            self.assertEqual(a, b)
            self.assertEqual((Path(first)/"adapter.json").read_bytes(), (Path(second)/"adapter.json").read_bytes())
            best = min(a["validation_history"], key=lambda row: row["validation_nll"])
            self.assertEqual(a["selected_step"], best["step"])
            self.assertLess(best["validation_nll"], a["validation_history"][0]["validation_nll"])
            restored = ReferencePolicy(state=json.loads((Path(first)/"adapter.json").read_text()))
            self.assertEqual(restored.backend, "reference_linear_policy_not_llm")
            # No test cases or test labels are accepted by the training API.

    def test_invalid_checkpoint_and_training_budget_rejected(self):
        state = ReferencePolicy().state
        for change in ({"base": [[0.]]}, {"b": [[float("nan")]*2]*3}, {"format": 99}):
            with self.assertRaises(ValueError):
                ReferencePolicy(state={**state, **change})
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(ValueError):
                train_reference([], [], [], [], out, steps=0)

    def test_pretrained_backend_fails_without_weights_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(FileNotFoundError):
                PretrainedPolicy(out)
            (Path(out)/"config.json").write_text("{}")
            with self.assertRaises(FileNotFoundError):
                model_files(out)
