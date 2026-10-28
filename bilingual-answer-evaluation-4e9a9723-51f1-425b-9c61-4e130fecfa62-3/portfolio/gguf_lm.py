"""Optional CPU GGUF inference; explicit real weights, constrained JSON output."""
from pathlib import Path
import hashlib
import time


class GGUFModel:
    def __init__(self, path, threads=4, context=2048):
        from llama_cpp import Llama
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError("Supply the real GGUF model file")
        self.model_name = self.path.name
        with self.path.open("rb") as f:
            self.sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        self.model = Llama(model_path=str(self.path), n_ctx=context, n_threads=threads,
                           n_batch=256, verbose=False, seed=0)

    def generate_json(self, system, question, schema, seed=0):
        started = time.perf_counter()
        output = self.model.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": question}],
            response_format={"type": "json_object", "schema": schema},
            max_tokens=180, temperature=0.2, top_p=0.9, seed=seed)
        choice = output["choices"][0]
        if choice["finish_reason"] != "stop":
            raise RuntimeError("Model response was truncated")
        return {"text": choice["message"]["content"], "seconds": time.perf_counter()-started,
                "usage": output["usage"], "model": self.model_name,
                "model_sha256": self.sha256, "seed": seed}

    def generate(self, messages, seed=1, max_tokens=128, temperature=0):
        started = time.perf_counter()
        output = self.model.create_chat_completion(messages=messages, seed=seed,
                    max_tokens=max_tokens, temperature=temperature)
        choice = output['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise RuntimeError('Model response was truncated')
        return {'text':choice['message']['content'], 'seconds':time.perf_counter()-started,
                'input_tokens':output['usage']['prompt_tokens'],
                'output_tokens':output['usage']['completion_tokens'],
                'model':self.model_name,'model_sha256':self.sha256}
