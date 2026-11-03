"""Local, real Transformers inference. No credentials and no fixture fallback."""
from __future__ import annotations
import time
from pathlib import Path


class LocalLM:
    def __init__(self,path,quantized=False,threads=2):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        path=Path(path)
        if not (path/'config.json').is_file():raise FileNotFoundError('Supply a downloaded local model directory')
        torch.set_num_threads(threads)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32).eval()
        if quantized:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',DeprecationWarning)
                self.model=torch.ao.quantization.quantize_dynamic(self.model,{torch.nn.Linear},dtype=torch.qint8,inplace=False)
        self.model_path=str(path);self.quantized=quantized

    def generate(self,messages,seed=1,max_tokens=48,temperature=.7):
        torch=self.torch;torch.manual_seed(seed)
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(prompt,return_tensors='pt',truncation=True,max_length=1536)
        t=time.perf_counter()
        options={'max_new_tokens':max_tokens,'do_sample':temperature>0,'pad_token_id':self.tokenizer.eos_token_id}
        if temperature>0:options.update(temperature=temperature,top_p=.95)
        with torch.inference_mode():output=self.model.generate(**inputs,**options)
        elapsed=time.perf_counter()-t
        generated=output[0,inputs['input_ids'].shape[1]:]
        return {'text':self.tokenizer.decode(generated,skip_special_tokens=True),
            'input_tokens':int(inputs['input_ids'].numel()),'output_tokens':len(generated),'seconds':elapsed}
