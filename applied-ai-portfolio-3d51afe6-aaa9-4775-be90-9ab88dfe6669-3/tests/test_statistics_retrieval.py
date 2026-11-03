import unittest
import numpy as np
from portfolio.common import wilson,clustered_difference
from portfolio.retrieval import calculate,Document,Retriever
from portfolio.agent_eval import parse_action,numeric_answer,run_case


class StatisticsTests(unittest.TestCase):
    def test_wilson_retains_uncertainty_after_zero_errors(self):
        self.assertGreater(wilson(0,20)[1],.1)
        self.assertLess(wilson(20,20)[0],.9)

    def test_bootstrap_is_paired_and_clustered(self):
        a=np.zeros((20,3));b=np.ones((20,3))
        r=clustered_difference(a,b)
        self.assertEqual(r['mean_difference'],1)
        self.assertEqual(r['ci95'],[1,1]);self.assertEqual(r['unit'],'question')

    def test_mismatched_trials_rejected(self):
        with self.assertRaises(ValueError):clustered_difference(np.zeros((2,2)),np.zeros((2,3)))


class ToolTests(unittest.TestCase):
    def test_numeric_dsl(self):self.assertEqual(calculate('(7 + 5) / 3'),4)

    def test_tool_cannot_execute_python(self):
        for text in ["__import__('os').system('id')",'(1).__class__','2**100000','[1,2]','1/0']:
            with self.subTest(text=text),self.assertRaises((ValueError,SyntaxError)):calculate(text)

    def test_only_allowlisted_actions(self):
        with self.assertRaises(ValueError):parse_action('{"tool":"shell","argument":"ls"}')
        with self.assertRaises(ValueError):parse_action('{"tool":"search","argument":"x","url":"https://evil"}')

    def test_ambiguous_answer_is_not_cherry_picked(self):
        self.assertIsNone(numeric_answer('17 or 19 days'))
        self.assertEqual(numeric_answer('17 days'),17)

    def test_retrieval_is_evidence_backed(self):
        r=Retriever([Document('a','Cedar retains audit logs for 23 days.'),Document('b','Dune has a timeout of 13 minutes.')])
        self.assertEqual(r.search('Cedar audit logs',1)[0]['id'],'a')

    def test_invalid_model_action_is_a_failure(self):
        class FixtureModel:
            def generate(self,*args,**kwargs):return {'text':'malformed','output_tokens':1,'seconds':.001}
        r=Retriever([Document('a','Cedar 23 days')])
        row=run_case(FixtureModel(),r,{'id':'x','question':'Cedar?','answer':23,'evidence_id':'a'},'tool_agent',1)
        self.assertFalse(row['correct']);self.assertEqual(row['trace'][0]['error'],'invalid_action')

    def test_repeated_tool_calls_hit_budget(self):
        class FixtureModel:
            def generate(self,*args,**kwargs):return {'text':'{"tool":"search","argument":"Cedar"}','output_tokens':1,'seconds':.001}
        row=run_case(FixtureModel(),Retriever([Document('a','Cedar 23 days')]),{'id':'x','question':'Cedar?','answer':23,'evidence_id':'a'},'tool_agent',1,max_steps=2)
        self.assertFalse(row['correct']);self.assertEqual(row['trace'][-1]['error'],'step_budget_exhausted')
