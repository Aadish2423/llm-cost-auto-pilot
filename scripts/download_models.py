"""Download the on-device privacy model (≈110 MB) into models/bert-base-ner.

Model: dslim/bert-base-NER (MIT licence), INT8 ONNX export published as
Xenova/bert-base-NER on Hugging Face. models/ is gitignored.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://huggingface.co/Xenova/bert-base-NER/resolve/main"
FILES = {
    "config.json": "config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "onnx/model_quantized.onnx": "model_quantized.onnx",
}


def main() -> None:
    target = ROOT / "models" / "bert-base-ner"
    target.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        path = target / local
        if path.is_file() and path.stat().st_size > 0:
            print(f"  exists  {path.relative_to(ROOT)}")
            continue
        print(f"  fetch   {remote} ...", flush=True)
        urlretrieve(f"{REPO}/{remote}", path)
    print(f"[+] Privacy model ready in {target.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
