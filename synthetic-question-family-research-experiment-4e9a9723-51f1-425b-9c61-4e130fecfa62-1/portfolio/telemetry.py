"""Defensive detection lab on explicitly synthetic authentication/workload logs."""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from .common import write_json,provenance,sha256_json,wilson


KINDS=('login_success','login_failure','role_change','pod_exec','secret_read')


def make_dataset(seed,episodes=200,split='train'):
    rng=np.random.default_rng(seed);events=[];labels=[]
    for episode in range(episodes):
        attack=(episode%4==0)
        # Disjoint subject identifiers across splits; label is never an event field.
        actor=f'{split}-user-{episode:04d}'
        start={'train':0,'validation':1000000,'test':2000000}.get(split,3000000)+episode*3600
        n=int(rng.integers(12,35));rows=[]
        for i in range(n):
            kind=str(rng.choice(['login_success','login_failure'],p=[.83,.17]))
            rows.append({'actor':actor,'ts':start+int(rng.integers(0,600)),
                'kind':kind,'country':str(rng.choice(['IL','US'],p=[.96,.04])),
                'source_ip':'192.0.2.'+str(int(rng.integers(1,200))),
                'resource':'identity','id':f'{split}-{episode}-{i}'})
        subtype='benign'
        if attack:
            subtype=['burst_then_success','privilege_and_secrets','workload_exec','slow_password_guessing'][episode//4%4]
            if subtype=='burst_then_success':
                count=int(rng.integers(7,15))
                for j in range(count):rows.append({'actor':actor,'ts':start+650+j*3,'kind':'login_failure','country':'DE','source_ip':'198.51.100.22','resource':'identity','id':f'{split}-{episode}-a{j}'})
                rows.append({'actor':actor,'ts':start+700,'kind':'login_success','country':'DE','source_ip':'198.51.100.22','resource':'identity','id':f'{split}-{episode}-success'})
            elif subtype=='privilege_and_secrets':
                for j,kind in enumerate(['role_change','secret_read','secret_read','secret_read']):
                    rows.append({'actor':actor,'ts':start+650+j*20,'kind':kind,'country':'IL','source_ip':'203.0.113.14','resource':'k8s/prod','id':f'{split}-{episode}-a{j}'})
            elif subtype=='workload_exec':
                for j,kind in enumerate(['pod_exec','secret_read','secret_read']):
                    rows.append({'actor':actor,'ts':start+650+j*20,'kind':kind,'country':'US','source_ip':'203.0.113.15','resource':'k8s/prod','id':f'{split}-{episode}-a{j}'})
            else:
                for j in range(8):
                    rows.append({'actor':actor,'ts':start+650+j*180,'kind':'login_failure','country':'IL','source_ip':'198.51.100.44','resource':'identity','id':f'{split}-{episode}-slow{j}'})
                rows.append({'actor':actor,'ts':start+2150,'kind':'login_success','country':'IL','source_ip':'198.51.100.44','resource':'identity','id':f'{split}-{episode}-slow-success'})
        elif episode%9==0:
            # Hard negatives: authorized administrators and harmless password retries.
            for j,kind in enumerate(['role_change','pod_exec','login_failure','login_failure','login_failure']):
                rows.append({'actor':actor,'ts':start+650+j*80,'kind':kind,'country':'IL','source_ip':'192.0.2.7','resource':'k8s/dev','id':f'{split}-{episode}-b{j}'})
        elif episode%11==0:
            # Authorized production maintenance is deliberately indistinguishable
            # from escalation without an external change-ticket/identity context.
            for j,kind in enumerate(['role_change','secret_read']):
                rows.append({'actor':actor,'ts':start+650+j*20,'kind':kind,'country':'IL','source_ip':'192.0.2.9','resource':'k8s/prod','id':f'{split}-{episode}-maintenance{j}'})
        events.extend(sorted(rows,key=lambda r:(r['ts'],r['id'])))
        labels.append({'actor':actor,'label':int(attack),'scenario':subtype})
    return events,labels


def validate_event(event):
    required={'actor','ts','kind','country','source_ip','resource','id'}
    if set(event)!=required or event.get('kind') not in KINDS:
        raise ValueError('Invalid event schema or event kind')
    if not isinstance(event['ts'],int) or event['ts']<0:
        raise ValueError('Invalid timestamp')
    if any(not isinstance(event[k],str) or not 0<len(event[k])<=200 for k in required-{'ts'}):
        raise ValueError('Invalid string field')
    return event


def normalize(events):
    seen={};result=[]
    for event in events:
        validate_event(event)
        if event['id'] in seen:
            if event!=seen[event['id']]:raise ValueError('Conflicting duplicate event')
            continue
        seen[event['id']]=event;result.append(event)
    return sorted(result,key=lambda r:(r['ts'],r['id']))


def analyze(events,threshold=8):
    """Bounded correlation windows, with exact evidence ids for every finding."""
    grouped=collections.defaultdict(list)
    for event in normalize(events):grouped[event['actor']].append(event)
    alerts=[]
    for actor,rows in grouped.items():
        for event in rows:
            previous=[r for r in rows if 0 < event['ts']-r['ts'] <= 120]
            evidence=[];rule=None
            if event['kind']=='login_success':
                failures=[r for r in previous if r['kind']=='login_failure' and r['source_ip']==event['source_ip']]
                if len(failures)>=threshold:rule='burst_then_success';evidence=failures
            elif event['kind']=='secret_read' and event['resource']=='k8s/prod':
                escalation=[r for r in previous if r['kind'] in ('role_change','pod_exec') and r['resource']==event['resource']]
                if escalation:rule='privilege_then_secret';evidence=escalation
            if rule:
                alerts.append({'actor':actor,'rule':rule,'ts':event['ts'],
                    'evidence_ids':[r['id'] for r in evidence]+[event['id']],
                    'explanation':'Time-correlated synthetic events; analyst review required.'})
    return alerts


def features(events,actors):
    grouped=collections.defaultdict(list)
    for event in normalize(events):grouped[event['actor']].append(event)
    output=[]
    for actor in actors:
        rows=grouped[actor];counts=collections.Counter(r['kind'] for r in rows)
        output.append([counts[k] for k in KINDS]+[len({r['country'] for r in rows}),len(rows)])
    return np.asarray(output,float)


def scores(labels,positive_actors):
    y=np.array([r['label'] for r in labels]);pred=np.array([int(r['actor'] in positive_actors) for r in labels])
    pr,rc,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'precision':float(pr),'recall':float(rc),'f1':float(f1),
        'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
        'false_positive_rate':float(fp/(fp+tn)),
        'recall_ci95':wilson(int(tp),int(tp+fn)),
        'false_positive_rate_ci95':wilson(int(fp),int(fp+tn))}


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train,lt=make_dataset(17,240,'train');valid,lv=make_dataset(29,120,'validation');test,le=make_dataset(43,240,'test')
    # Tune rule threshold on validation only; choose the most conservative tied optimum.
    trials=[{'threshold':t,**scores(lv,{r['actor'] for r in analyze(valid,t)})} for t in (6,8,10,12)]
    selected=max(trials,key=lambda r:(r['f1'],r['threshold']))['threshold']
    alerts=analyze(test,selected);rule_scores=scores(le,{r['actor'] for r in alerts})
    # Fit an unsupervised baseline to benign training episodes only.
    xt=features(train,[r['actor'] for r in lt]);xv=features(valid,[r['actor'] for r in lv]);xe=features(test,[r['actor'] for r in le])
    iso=IsolationForest(n_estimators=100,random_state=7,contamination='auto',n_jobs=1)
    iso.fit(xt[np.array([r['label'] for r in lt])==0])
    sv=-iso.score_samples(xv);se=-iso.score_samples(xe)
    cutoff_candidates=np.unique(np.quantile(sv,np.linspace(0,1,51)))
    best=max(cutoff_candidates,key=lambda t:(scores(lv,{r['actor'] for r,s in zip(lv,sv) if s>=t})['f1'],t))
    baseline=scores(le,{r['actor'] for r,s in zip(le,se) if s>=best})
    baseline['roc_auc']=float(roc_auc_score([r['label'] for r in le],se))
    baseline['validation_cutoff']=float(best)
    write_json(out/'test_events.json',test);write_json(out/'test_labels.json',le);write_json(out/'alerts.json',alerts)
    result={'experiment':'cloud-telemetry-v1','data_kind':'synthetic','provenance':provenance(),
        'counts':{'train_episodes':len(lt),'validation_episodes':len(lv),'test_episodes':len(le),'test_events':len(test)},
        'selected_rule_threshold':selected,'validation_trials':trials,
        'rule_detector':rule_scores,'isolation_forest_baseline':baseline,
        'test_data_sha256':sha256_json(test),
        'limitations':['Synthetic scenario generator; these scores do not measure real attack detection.',
            'Disjoint actors, seeds and time ranges by split; scenario families are shared.',
            'No live cloud, Kubernetes cluster, nation-state investigation or KQL engine was used.',
            'KQL queries are reference translations and have not been executed on Azure.']}
    write_json(out/'summary.json',result)
    text=['# Cloud Telemetry Lab','','Dataset: **synthetic**. Test set held out from threshold selection.','',
      '| Detector | Precision | Recall | F1 | False-positive rate |','|---|---:|---:|---:|---:|']
    for name,key in [('Temporal rules','rule_detector'),('Isolation Forest','isolation_forest_baseline')]:
        r=result[key];text.append(f"| {name} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.3f} | {r['false_positive_rate']:.2%} |")
    text+=['','## Limits']+['- '+s for s in result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps(result['rule_detector']),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/telemetry');a=p.parse_args();run(a.out)
