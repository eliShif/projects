"""Explicit hosted-model adapter; no implicit billable calls or fallback."""
from __future__ import annotations
import os,time


class HostedLM:
    def __init__(self,model,max_calls=120,max_output_tokens=1024,client=None):
        if not model:raise ValueError('Choose an explicit model available in your account')
        if max_calls<1 or max_output_tokens<64:raise ValueError('Invalid call/token budget')
        if client is None:
            if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('OPENAI_API_KEY is required; no local-model fallback')
            from openai import OpenAI
            client=OpenAI(timeout=90,max_retries=0)
        self.client=client;self.model_name=model;self.max_calls=max_calls
        self.max_output_tokens=max_output_tokens;self.calls=0;self.usage=[]
    def generate(self,messages,seed=1,max_tokens=48,temperature=.7):
        if self.calls>=self.max_calls:raise RuntimeError('Configured API-call budget exhausted')
        self.calls+=1;t=time.perf_counter()
        # Responses does not promise seeded sampling. Do not invent repeatability
        # or send unsupported sampling fields to reasoning model families.
        response=self.client.responses.create(model=self.model_name,input=messages,
            max_output_tokens=self.max_output_tokens,store=False)
        if response.status!='completed' or not response.output_text:
            raise RuntimeError('Hosted model did not return a complete text answer')
        metrics={'input_tokens':response.usage.input_tokens,'output_tokens':response.usage.output_tokens,
            'seconds':time.perf_counter()-t,'provider':'openai','model':response.model,
            'response_id':response.id,'seed_controlled':False,'sampling':'provider default'}
        self.usage.append(metrics)
        return {'text':response.output_text,**metrics}
