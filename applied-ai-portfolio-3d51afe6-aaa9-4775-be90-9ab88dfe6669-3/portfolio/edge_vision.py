"""Real digit-image training, distillation and dynamic INT8 CPU inference."""
from __future__ import annotations
import argparse, copy, io, time, warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from .common import provenance, write_json, sha256_json, wilson


def model(width):
    return nn.Sequential(nn.Linear(64,width),nn.ReLU(),nn.Linear(width,10))


def fit(net,x,y,xv,yv,teacher=None,epochs=100,seed=7):
    torch.manual_seed(seed)
    optim=torch.optim.Adam(net.parameters(),lr=.008)
    best=None;best_loss=float('inf');history=[]
    if teacher is not None:teacher.eval()
    for epoch in range(epochs):
        net.train()
        order=torch.randperm(len(x))
        for ids in order.split(128):
            logits=net(x[ids]);hard=nn.functional.cross_entropy(logits,y[ids])
            if teacher is None:loss=hard
            else:
                with torch.no_grad():soft=nn.functional.softmax(teacher(x[ids])/2.,dim=1)
                kd=nn.functional.kl_div(nn.functional.log_softmax(logits/2.,dim=1),soft,reduction='batchmean')*4.
                loss=.5*hard+.5*kd
            optim.zero_grad();loss.backward();optim.step()
        net.eval()
        with torch.no_grad():vl=float(nn.functional.cross_entropy(net(xv),yv))
        history.append({'epoch':epoch+1,'validation_loss':vl})
        if vl < best_loss:
            best_loss=vl;best=copy.deepcopy(net.state_dict())
    net.load_state_dict(best);net.eval()
    return history


def predict(net,x):
    with torch.inference_mode():return net(x).argmax(1).numpy()


def measure(net,x,repetitions=100):
    with torch.inference_mode():
        for _ in range(10):net(x)
        times=[]
        for _ in range(repetitions):
            t=time.perf_counter_ns();net(x);times.append((time.perf_counter_ns()-t)/1e6)
    return {'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
            'batch_size':len(x),'repetitions':repetitions,'warmup':10}


def evaluate(net,x,y):
    predictions=predict(net,x)
    blob=io.BytesIO();torch.save(net.state_dict(),blob)
    correct=int((predictions==y.numpy()).sum())
    return {'test_accuracy':correct/len(y),'accuracy_ci95':wilson(correct,len(y)),
            'serialized_state_bytes':blob.tell(),'latency_batch1':measure(net,x[:1]),
            'latency_batch32':measure(net,x[:32]),
            'confusion_matrix':confusion_matrix(y,predictions,labels=list(range(10))).tolist()}


def run(out,seeds=(7,19,43),epochs=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    raw=load_digits();x=torch.tensor(raw.data/16.,dtype=torch.float32);y=torch.tensor(raw.target)
    indices=np.arange(len(y))
    trainval,test=train_test_split(indices,test_size=.2,random_state=2026,stratify=raw.target)
    train,val=train_test_split(trainval,test_size=.2,random_state=2027,stratify=raw.target[trainval])
    write_json(out/'splits.json',{'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()})
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed);teacher=model(128)
        ht=fit(teacher,x[train],y[train],x[val],y[val],epochs=epochs,seed=seed)
        torch.manual_seed(seed);student=model(24)
        hs=fit(student,x[train],y[train],x[val],y[val],teacher=teacher,epochs=epochs,seed=seed)
        torch.manual_seed(seed);control=model(24)
        hc=fit(control,x[train],y[train],x[val],y[val],epochs=epochs,seed=seed)
        # Actual PyTorch INT8 linear kernels; not just rounding floating-point weights.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            quant=torch.ao.quantization.quantize_dynamic(copy.deepcopy(student),{nn.Linear},dtype=torch.qint8)
        models={'teacher_fp32':teacher,'small_supervised_fp32':control,'student_distilled_fp32':student,'student_dynamic_int8':quant}
        for name,net in models.items():
            metrics=evaluate(net,x[test],y[test])
            rows.append({'seed':seed,'variant':name,**metrics})
        write_json(out/f'training_seed_{seed}.json',{'teacher':ht,'student_distilled':hs,'student_supervised':hc})
        print('vision seed',seed,[(r['variant'],round(r['test_accuracy'],4)) for r in rows if r['seed']==seed],flush=True)
    result={'experiment':'edge-vision-v1','provenance':provenance(),'torch':torch.__version__,
        'dataset':'scikit-learn load_digits / UCI handwritten digits, 1797 8x8 images',
        'dataset_hash':sha256_json({'x':raw.data.tolist(),'y':raw.target.tolist()}),
        'split_sizes':{'train':len(train),'validation':len(val),'test':len(test)},
        'seeds':list(seeds),'epochs':epochs,'cpu_threads':1,'rows':rows,
        'limitations':['Small educational image dataset, not a production vision benchmark.',
        'Model/checkpoint selection used validation only; test split fixed before training.',
        'Three training seeds share one test split; they are not independent datasets.',
        'No hardware power, phone or accelerator measurements; latency is local CPU only.',
        'Serialized state size is storage, not total process memory.']}
    write_json(out/'summary.json',result)
    lines=['# Edge Vision: measured results','','Fixed held-out test set; validation-only checkpoint selection.','',
      '| Variant | Mean accuracy | Mean state KiB | Median batch-1 latency (ms) |','|---|---:|---:|---:|']
    for name in models:
        rr=[r for r in rows if r['variant']==name]
        lines.append(f"| {name} | {np.mean([r['test_accuracy'] for r in rr]):.2%} | {np.mean([r['serialized_state_bytes'] for r in rr])/1024:.1f} | {np.median([r['latency_batch1']['p50_ms'] for r in rr]):.4f} |")
    lines+=['','## Limits']+['- '+s for s in result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/edge_vision');p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args();run(a.out,epochs=a.epochs)
