import unittest
import torch
from portfolio.adapter_tuning import LoRALinear,examples,answer_loss,encode

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

    def test_low_rank_update_matches_explicit_matrix_product(self):
        base=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():base.weight.copy_(torch.eye(2))
        layer=LoRALinear(base,rank=1,alpha=2)
        with torch.no_grad():
            layer.a.copy_(torch.tensor([[1.,2.]]));layer.b.copy_(torch.tensor([[3.],[4.]]))
        # x A^T = 5, (x A^T) B^T = [15,20], scale = 2.
        torch.testing.assert_close(layer(torch.tensor([[1.,2.]])),torch.tensor([[31.,42.]]))
        self.assertFalse(base.weight.requires_grad)

    def test_supervised_updates_reduce_loss_without_changing_base(self):
        torch.manual_seed(21)
        base=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():base.weight.zero_()
        layer=LoRALinear(base,rank=1,alpha=1)
        with torch.no_grad():layer.a.fill_(1.)
        x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        target=torch.tensor([[1.],[1.],[2.]])
        optimizer=torch.optim.SGD([layer.a,layer.b],lr=.05)
        initial=torch.nn.functional.mse_loss(layer(x),target).item()
        for _ in range(8):
            optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(layer(x),target)
            loss.backward();optimizer.step()
        self.assertLess(torch.nn.functional.mse_loss(layer(x),target).item(),initial)
        torch.testing.assert_close(base.weight,torch.zeros_like(base.weight))
        self.assertIsNone(base.weight.grad)

    def test_answer_nll_is_weighted_by_unmasked_shifted_tokens(self):
        class LossFixture:
            def eval(self):pass
            def __call__(self,**row):
                from types import SimpleNamespace
                return SimpleNamespace(loss=torch.tensor(row['fixture_loss']))
        rows=[{'labels':torch.tensor([[-100,7,-100]]),'fixture_loss':2.},
              {'labels':torch.tensor([[-100,7,8,9]]),'fixture_loss':4.}]
        self.assertEqual(answer_loss(LossFixture(),rows),3.5)

    def test_training_targets_exclude_prompt_tokens(self):
        class TokenizerFixture:
            eos_token='!'
            def apply_chat_template(self,*args,**kwargs):return 'prefix:'
            def __call__(self,text,**kwargs):
                ids=list(range(len(text)))
                return {'input_ids':torch.tensor([ids]) if kwargs.get('return_tensors') else ids}
        encoded=encode(TokenizerFixture(),{'context':'example','entity':'Training1','days':17})
        self.assertTrue((encoded['labels'][0,:7]==-100).all())
        torch.testing.assert_close(encoded['labels'][0,7:],encoded['input_ids'][0,7:])
        self.assertEqual(int((encoded['labels']!=-100).sum()),3)
