"""Isolated-process FP32 vs dynamic-INT8 small-LM measurements."""
from __future__ import annotations
import argparse, io, json, math, os, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
from .common import write_json,provenance,sha256_json


TEXTS=[
 'The experiment compares two models on the same held-out examples. A separate validation set is used to choose the checkpoint.',
 'An engineer reads an audit log and investigates a failed request. The trace links the request to the retrieved evidence.',
 'A smaller neural network can use less storage. Compression may change accuracy and does not always reduce latency.',
 'The research report includes both positive and negative results, along with the assumptions and limitations of the experiment.',
 'A data pipeline checks input types, rejects malformed records and stores the version of the dataset used in each run.',
 'When a worker stops unexpectedly, its unfinished job can be reclaimed after the lease expires. An old worker must not overwrite a new result.',
]


def resident_bytes():
    try:return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError,IndexError):return None


def worker(model_path,variant,out):
    import torch
    from .local_lm import LocalLM
    before=resident_bytes();t=time.perf_counter();lm=LocalLM(model_path,quantized=variant=='dynamic_int8',threads=1)
    load_seconds=time.perf_counter()-t;after=resident_bytes()
    tokens=0;total_nll=0.;examples=[];latencies=[]
    with torch.inference_mode():
        for text in TEXTS:
            inp=lm.tokenizer(text,return_tensors='pt')
            output=lm.model(**inp,labels=inp['input_ids'])
            n=int(inp['input_ids'].numel()-1)
            total_nll+=float(output.loss)*n;tokens+=n
        # Warmup is separate from timed prefill measurements.
        sample=lm.tokenizer(TEXTS[0],return_tensors='pt')
        for _ in range(3):lm.model(**sample,use_cache=False)
        for _ in range(20):
            t=time.perf_counter();lm.model(**sample,use_cache=False);latencies.append((time.perf_counter()-t)*1000)
    for text in ['Explain why a held-out test set matters.','Explain a trade-off between latency and model size.','What evidence should an incident report contain?']:
        for repeat in range(3):
            r=lm.generate([{'role':'user','content':text}],seed=100+repeat,max_tokens=48,temperature=0)
            examples.append({'prompt':text,'repeat':repeat,**r})
    # Stream state serialization to disk, avoiding an extra in-memory copy.
    state_path=Path(out).with_suffix('.pt');torch.save(lm.model.state_dict(),state_path);state_bytes=state_path.stat().st_size;state_path.unlink()
    result={'variant':variant,'provenance':provenance(),'torch':torch.__version__,
        'load_seconds':load_seconds,'resident_before_load_bytes':before,'resident_after_load_bytes':after,
        'serialized_state_bytes':state_bytes,'text_tokens_scored':tokens,
        'negative_log_likelihood':total_nll/tokens,'perplexity_on_authored_texts':math.exp(total_nll/tokens),
        'prefill_p50_ms':float(np.median(latencies)),'prefill_p95_ms':float(np.quantile(latencies,.95)),
        'prefill_tokens':int(sample['input_ids'].numel()),'prefill_repeats':20,'cpu_threads':1,
        'generation_tokens_per_second_median':float(np.median([r['output_tokens']/r['seconds'] for r in examples])),
        'generations':examples,'weights_revision':model_revision(model_path)}
    write_json(out,result)


def model_revision(path):
    root=Path(path)/'.cache/huggingface/download'
    meta=root/'model.safetensors.metadata'
    if meta.exists():return meta.read_text().splitlines()[0]
    return 'unknown-local-copy'


def run(model_path,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    records=[]
    for variant in ('fp32','dynamic_int8'):
        target=out/f'{variant}.json'
        subprocess.run([sys.executable,'-m','portfolio.small_lm','--model-path',str(Path(model_path).resolve()),'--worker',variant,'--out',str(target)],check=True)
        records.append(json.loads(target.read_text()));print('small LM completed',variant,flush=True)
    result={'experiment':'small-lm-efficiency-v1','model':'HuggingFaceTB/SmolLM2-135M-Instruct',
        'texts_sha256':sha256_json(TEXTS),'provenance':provenance(),'rows':records,
        'limitations':['Six authored prose passages are a smoke benchmark, not standard language-model perplexity.',
            'Same weights and tokenizer in both variants; no task-specific fine-tuning.',
            'Variants use separate processes. RSS includes Python, libraries and model; it is not weights-only memory.',
            'Dynamic INT8 targets Linear modules; embeddings remain floating point.',
            'No power meter or physical edge device was available. No energy-efficiency claim.',
            'This small model is a reproducible baseline, not a frontier-model capability claim.']}
    write_json(out/'summary.json',result)
    lines=['# Small Language Model Efficiency','','Real CPU inference, isolated process per variant.','',
      '| Variant | State MiB | Resident MiB after load | Authored-text perplexity | Prefill p50 ms | Generated tokens/s |','|---|---:|---:|---:|---:|---:|']
    for r in records:
        rss=r['resident_after_load_bytes'];rss_text=f'{rss/2**20:.1f}' if rss is not None else 'unavailable'
        lines.append(f"| {r['variant']} | {r['serialized_state_bytes']/2**20:.1f} | {rss_text} | {r['perplexity_on_authored_texts']:.2f} | {r['prefill_p50_ms']:.2f} | {r['generation_tokens_per_second_median']:.2f} |")
    lines+=['','## Limits']+['- '+s for s in result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--out',default='results/small_lm');p.add_argument('--worker',choices=['fp32','dynamic_int8'])
    a=p.parse_args()
    if a.worker:worker(a.model_path,a.worker,a.out)
    else:run(a.model_path,a.out)
