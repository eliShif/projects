import json,unittest,httpx
from openai import OpenAI
from portfolio.providers import HostedLM

class ProviderContractTests(unittest.TestCase):
    def test_sdk_request_usage_and_no_silent_budget_overrun(self):
        seen=[]
        def handle(request):
            payload=json.loads(request.content);seen.append(payload)
            return httpx.Response(200,json={'id':'fixture-response','object':'response','created_at':1,
                'status':'completed','model':'fixture-model','output':[{'id':'m','type':'message',
                'role':'assistant','status':'completed','content':[{'type':'output_text','text':'17','annotations':[]}]}],
                'usage':{'input_tokens':5,'output_tokens':2,'total_tokens':7}})
        client=OpenAI(api_key='fixture-not-a-real-key',http_client=httpx.Client(transport=httpx.MockTransport(handle)),max_retries=0)
        lm=HostedLM('fixture-model',max_calls=1,client=client)
        r=lm.generate([{'role':'user','content':'hello'}]);self.assertEqual(r['text'],'17')
        self.assertFalse(seen[0]['store']);self.assertNotIn('seed',seen[0]);self.assertEqual(r['output_tokens'],2)
        with self.assertRaisesRegex(RuntimeError,'budget'):lm.generate([])
        self.assertEqual(len(seen),1);client.close()

    def test_incomplete_response_is_not_scored_as_success(self):
        def handle(request):return httpx.Response(200,json={'id':'fixture','object':'response','created_at':1,
            'status':'incomplete','model':'fixture','output':[],'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}})
        client=OpenAI(api_key='fixture-not-a-real-key',http_client=httpx.Client(transport=httpx.MockTransport(handle)),max_retries=0)
        with self.assertRaisesRegex(RuntimeError,'complete'):HostedLM('fixture',client=client).generate([])
        client.close()
