import unittest
import torch
from portfolio.adapter_tuning import LoRALinear,examples

class AdapterTests(unittest.TestCase):
    def test_initial_adapter_preserves_function_and_base_stays_frozen(self):
        torch.manual_seed(9);base=torch.nn.Linear(5,3);x=torch.randn(4,5)
        expected=base(x).detach();old=base.weight.detach().clone();layer=LoRALinear(base,rank=2)
        torch.testing.assert_close(layer(x),expected)
        opt=torch.optim.SGD([layer.a,layer.b],lr=.1);layer(x).square().sum().backward();opt.step()
        self.assertIsNone(base.weight.grad);torch.testing.assert_close(base.weight,old)
        self.assertFalse(torch.equal(layer(x),expected))
    def test_split_entities_do_not_overlap(self):
        a={r['entity'] for r in examples('Training',24)};b={r['entity'] for r in examples('Validation',4)}
        self.assertFalse(a&b)
