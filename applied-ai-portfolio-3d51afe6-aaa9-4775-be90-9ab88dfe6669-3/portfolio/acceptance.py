"""End-to-end HTTP -> durable queue -> actual local model -> stored evidence."""
from __future__ import annotations
import argparse,json,secrets,tempfile,threading,urllib.request,urllib.error
from pathlib import Path
from .reliable_service import JobStore,make_server,ResearchEngine,work_once
from .local_lm import LocalLM
from .common import write_json,provenance,sha256_json


def run(model_path,events_path,out):
    events=json.loads(Path(events_path).read_text());token=secrets.token_urlsafe(24)
    engine=ResearchEngine(LocalLM(model_path),events);records=[]
    with tempfile.TemporaryDirectory() as tmp:
        store=JobStore(Path(tmp)/'jobs.db');server=make_server(store,token,port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_address[1])
        def request(path,body=None,key='',authenticated=True):
            headers={'Content-Type':'application/json','Idempotency-Key':key}
            if authenticated:headers['Authorization']='Bearer '+token
            req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
            try:
                with urllib.request.urlopen(req,timeout=10) as r:return r.status,json.load(r)
            except urllib.error.HTTPError as e:return e.code,json.load(e)
        try:
            code,_=request('/jobs',{'mode':'research','question':'x'},authenticated=False);assert code==401
            for mode,question in [('research','How many days are audit logs retained by the Aster service?'),
                ('security_triage','Review secret access for test-user-0004. What evidence supports investigation?')]:
                payload={'mode':mode,'question':question};status,created=request('/jobs',payload,key=mode);assert status==202
                code,duplicate=request('/jobs',payload,key=mode);assert code==200 and duplicate['id']==created['id']
                assert work_once(store,engine)
                code,result=request('/jobs/'+created['id']);assert code==200 and result['status']=='succeeded'
                assert result['result']['evidence'] and result['result']['model_metrics']['output_tokens']>0
                assert [t['event'] for t in result['trace']]==['enqueued','claimed','completed']
                assert not work_once(store,engine)
                records.append({'mode':mode,'create_status':status,'duplicate_status':200,'result':result})
        finally:server.shutdown();server.server_close();thread.join()
    result={'provenance':provenance(),'model':'SmolLM2-135M-Instruct, actual local generation',
      'transport':'loopback HTTP','unauthorized_status':401,'events_sha256':sha256_json(events),
      'checks_passed':True,'records':records,'limits':'Workflow and evidence attachment verified; natural-language factual correctness is not certified.'}
    write_json(out,result);print('Real-model HTTP acceptance passed for research and security_triage',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--events',default='results/telemetry/test_events.json');p.add_argument('--out',default='results/service/acceptance.json')
    a=p.parse_args();run(a.model_path,a.events,a.out)
