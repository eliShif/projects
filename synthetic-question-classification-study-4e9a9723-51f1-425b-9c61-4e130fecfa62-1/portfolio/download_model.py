"""Explicit, revision-pinned download. Never called on module import."""
import argparse
from huggingface_hub import snapshot_download

MODEL_ID='HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',default='models/SmolLM2-135M-Instruct');a=p.parse_args()
    snapshot_download(MODEL_ID,revision=REVISION,local_dir=a.destination,
        allow_patterns=['*.json','*.safetensors','tokenizer.model'])
