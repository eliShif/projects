import concurrent.futures,json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from portfolio.reliable_service import JobStore,Conflict,LeaseLost,make_server,work_once


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.now=[100.];self.path=Path(self.tmp.name)/'jobs.db'
        self.store=JobStore(self.path,clock=lambda:self.now[0]);self.payload={'mode':'research','question':'Aster retention?'}
    def tearDown(self):self.tmp.cleanup()

    def test_concurrent_idempotency(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            rows=list(ex.map(lambda _:self.store.enqueue('same',self.payload),range(20)))
        self.assertEqual(len({r[0] for r in rows}),1);self.assertEqual(sum(r[1] for r in rows),1)

    def test_key_conflict(self):
        self.store.enqueue('same',self.payload)
        with self.assertRaises(Conflict):self.store.enqueue('same',{**self.payload,'question':'different'})

    def test_persistence_after_reopen(self):
        job_id,_=self.store.enqueue('x',self.payload)
        self.assertEqual(JobStore(self.path).get(job_id)['status'],'queued')

    def test_concurrent_claim_has_one_owner(self):
        self.store.enqueue('x',self.payload)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            rows=list(ex.map(lambda _:self.store.claim(),range(16)))
        self.assertEqual(sum(r is not None for r in rows),1)

    def test_expired_worker_cannot_overwrite_reclaimed_job(self):
        job_id,_=self.store.enqueue('x',self.payload);old=self.store.claim(lease_seconds=2)
        self.now[0]+=3;new=self.store.claim(lease_seconds=20)
        with self.assertRaises(LeaseLost):self.store.complete(old,{'bad':'stale'})
        self.store.complete(new,{'good':True})
        self.assertEqual(self.store.get(job_id)['result'],{'good':True})

    def test_expired_lease_rejected_even_without_reclaim(self):
        self.store.enqueue('x',self.payload);job=self.store.claim(lease_seconds=2);self.now[0]+=3
        with self.assertRaises(LeaseLost):self.store.complete(job,{'late':True})

    def test_crashes_eventually_exhaust_attempts(self):
        job_id,_=self.store.enqueue('x',self.payload)
        for _ in range(3):self.assertIsNotNone(self.store.claim(lease_seconds=1));self.now[0]+=2
        self.assertIsNone(self.store.claim());self.assertEqual(self.store.get(job_id)['status'],'failed')

    def test_completed_job_is_not_repeated(self):
        self.store.enqueue('x',self.payload);job=self.store.claim();self.store.complete(job,{'ok':True})
        self.assertIsNone(self.store.claim())
        with self.assertRaises(LeaseLost):self.store.complete(job,{'again':True})

    def test_invalid_input_rejected(self):
        for payload in [{}, {'mode':'research','question':'x','secret':'y'},{'mode':'shell','question':'x'},{'mode':'research','question':'x'*2001}]:
            with self.assertRaises(ValueError):self.store.enqueue('x',payload)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=JobStore(Path(self.tmp.name)/'db')
        self.token='test-only-token-not-a-real-secret';self.server=make_server(self.store,self.token,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def request(self,path,body=None,key='k',authorized=True):
        headers={'Content-Type':'application/json','Idempotency-Key':key}
        if authorized:headers['Authorization']='Bearer '+self.token
        req=urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=3) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)

    def test_auth_is_required(self):
        code,_=self.request('/jobs',{'mode':'research','question':'x'},authorized=False);self.assertEqual(code,401)

    def test_http_job_worker_result_and_receipt(self):
        payload={'mode':'research','question':'Aster?'}
        status,body=self.request('/jobs',payload);self.assertEqual(status,202)
        status,duplicate=self.request('/jobs',payload);self.assertEqual(status,200);self.assertEqual(body['id'],duplicate['id'])
        class FixtureEngine:
            def answer(self,payload):return {'answer':'fixture only','test_fixture':True}
        work_once(self.store,FixtureEngine())
        status,row=self.request('/jobs/'+body['id']);self.assertEqual(row['status'],'succeeded')
        self.assertEqual([r['event'] for r in row['trace']],['enqueued','claimed','completed'])

    def test_conflict_returns_409(self):
        self.request('/jobs',{'mode':'research','question':'a'})
        self.assertEqual(self.request('/jobs',{'mode':'research','question':'b'})[0],409)

    def test_oversized_body_rejected(self):
        self.assertEqual(self.request('/jobs',{'mode':'research','question':'a'*9000})[0],413)
