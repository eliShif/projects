"""Download a pinned public safetensors encoder; inference itself stays offline."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
FILES = ("config.json", "model.safetensors", "sentencepiece.bpe.model",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("models/encoder"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": MODEL, "revision": REVISION, "files": {}}
    for name in FILES:
        target = args.destination / name
        if target.exists():
            raise FileExistsError("Use an empty destination; checkpoints are not overwritten")
        digest = hashlib.sha256()
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response, target.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
        manifest["files"][name] = {"sha256": digest.hexdigest(), "bytes": target.stat().st_size}
    (args.destination / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_id": MODEL, "revision": REVISION, "files": len(FILES)}))


if __name__ == "__main__":
    main()
