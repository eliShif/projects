"""Small, real LoRA experiment inside the agent-evaluation project.

Frozen base model; only rank-4 adapters in the last two layers are optimized.
Training, validation and test entities are disjoint. This is a format-learning
smoke experiment, not a benchmark of generalization beyond authored examples.
"""
from __future__ import annotations
import argparse, copy, json, random, time
from pathlib import Path
import torch
from torch import nn
from .local_lm import LocalLM
from .agent_eval import dataset, SYSTEM, numeric_answer
from .common import write_json, sha256_json, provenance


class LoRALinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.a=nn.Parameter(torch.empty(rank,base.in_features))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=5**.5)
        for p in self.base.parameters():p.requires_grad_(False)
    def forward(self,x):return self.base(x)+(x@self.a.T@self.b.T)*self.scale


def examples(prefix,count):
    return [{'entity':f'{prefix}{i}', 'days':11+(i*7)%61,
             'context':f'{prefix}{i} service runbook. The audit-log retention period is {11+(i*7)%61} days. The session timeout is {5+i} minutes.'}
            for i in range(count)]


def messages(example):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':
        f"Documents:\n{example['context']}\nQuestion: How many days are audit logs retained by the {example['entity']} service?"}]


def encode(tokenizer,example):
    prefix=tokenizer.apply_chat_template(messages(example),tokenize=False,add_generation_prompt=True)
    full=prefix+str(example['days'])+tokenizer.eos_token
    ids=tokenizer(full,return_tensors='pt',add_special_tokens=False)['input_ids']
    prefix_length=len(tokenizer(prefix,add_special_tokens=False)['input_ids'])
    labels=ids.clone();labels[:,:prefix_length]=-100
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}


def answer_loss(model,encoded):
    model.eval();total=0.;n=0
    with torch.no_grad():
        for row in encoded:
            count=int((row['labels'][:,1:]!=-100).sum());total+=float(model(**row).loss)*count;n+=count
    return total/n


def run(model_path,out,steps=48):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    random.seed(2026);torch.manual_seed(2026)
    lm=LocalLM(model_path,threads=2);model=lm.model
    train=examples('TrainingService',24);valid=examples('ValidationService',4)
    docs,tasks=dataset();test=[{'entity':d.text.split()[0],'days':t['answer'],'context':d.text} for d,t in zip(docs,tasks)]
    splits={'train':train,'validation':valid,'test':test}
    assert not ({r['entity'] for r in train}&{r['entity'] for r in valid+test})
    write_json(out/'splits.json',splits)
    encoded_train=[encode(lm.tokenizer,e) for e in train];encoded_valid=[encode(lm.tokenizer,e) for e in valid]
    encoded_test=[encode(lm.tokenizer,e) for e in test]
    base_loss=answer_loss(model,encoded_test)
    before=[lm.generate(messages(e),temperature=0,max_tokens=8)['text'] for e in test]
    for p in model.parameters():p.requires_grad_(False)
    targets=[]
    for i in range(len(model.model.layers)-2,len(model.model.layers)):
        for name in ('q_proj','v_proj'):
            attention=model.model.layers[i].self_attn
            setattr(attention,name,LoRALinear(getattr(attention,name)))
            targets.append(f'model.layers.{i}.self_attn.{name}')
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.005)
    best_loss=float('inf');best=None;history=[];t=time.perf_counter()
    # Fixed budget; validation chooses the checkpoint, never the held-out set.
    for step in range(steps):
        model.train();row=encoded_train[(step*7)%len(encoded_train)]
        optimizer.zero_grad();loss=model(**row).loss;loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if (step+1)%12==0 or step==steps-1:
            vl=answer_loss(model,encoded_valid)
            history.append({'step':step+1,'train_loss':float(loss.detach()),'validation_answer_nll':vl})
            if vl<best_loss:
                best_loss=vl;best={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
            print('adapter step',step+1,'validation',round(vl,4),flush=True)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in best:p.copy_(best[name])
    model.eval();after_loss=answer_loss(model,encoded_test)
    after=[lm.generate(messages(e),temperature=0,max_tokens=8)['text'] for e in test]
    # Safetensors contains only trained adapters, no base weights or executable pickle.
    from safetensors.torch import save_file
    save_file(best,str(out/'adapters.safetensors'))
    rows=[{'entity':e['entity'],'expected':e['days'],'base_answer':b,'adapter_answer':a,
           'base_correct':numeric_answer(b)==e['days'],'adapter_correct':numeric_answer(a)==e['days']}
          for e,b,a in zip(test,before,after)]
    result={'experiment':'lora-format-adaptation-v1','provenance':provenance(),
      'trainable_parameters':sum(p.numel() for p in params),'rank':4,'alpha':8,'targets':targets,
      'steps':steps,'lr':.005,'seed':2026,'training_seconds':time.perf_counter()-t,
      'dataset_sha256':sha256_json(splits),'selection':'lowest validation answer-token NLL',
      'validation_history':history,'base_test_answer_nll':base_loss,'adapter_test_answer_nll':after_loss,
      'base_test_exact_match':sum(r['base_correct'] for r in rows)/len(rows),
      'adapter_test_exact_match':sum(r['adapter_correct'] for r in rows)/len(rows),
      'test_rows':rows,'limits':['Single seed and 12 authored test examples; not a general capability estimate.',
      'Fine-tuning learns an answer format on a narrow synthetic distribution.',
      'This arm uses one gold document and greedy decoding, unlike the three-document stochastic retrieval experiment; do not compare their scores causally.']}
    write_json(out/'summary.json',result)
    (out/'REPORT.md').write_text(f"# Real low-rank adaptation\n\n{result['trainable_parameters']:,} trainable parameters; {steps} updates; validation-selected checkpoint.\n\n| Metric | Frozen base | Adapter |\n|---|---:|---:|\n| Held-out answer-token NLL | {base_loss:.3f} | {after_loss:.3f} |\n| Held-out exact match (12 cases) | {result['base_test_exact_match']:.1%} | {result['adapter_test_exact_match']:.1%} |\n\n"+'\n'.join('- '+x for x in result['limits'])+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--out',default='results/adapter_tuning');p.add_argument('--steps',type=int,default=48)
    a=p.parse_args();run(a.model_path,a.out,a.steps)
