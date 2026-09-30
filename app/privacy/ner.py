"""On-device named-entity recognition: finds people, organisations and
places that regexes can't.

Model: dslim/bert-base-NER (MIT), ONNX export from Xenova/bert-base-NER,
~109 MB INT8. Runs through app.edge.accelerators, so it lands on the
Snapdragon NPU when onnxruntime-qnn is installed. Download it with
`python scripts/download_models.py`.

The prompt is scanned in 510-token windows, so long prompts are covered
end to end rather than truncated.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..edge.accelerators import PROVIDER_LABELS, create_session
from .detectors import Finding

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bert-base-ner"
MODEL_FILE = "model_quantized.onnx"
WINDOW = 510  # + [CLS]/[SEP] = BERT's 512 limit

# Identifier / finance keywords the English NER model mistakes for entities.
KEYWORD_STOPLIST = {
    "aadhaar", "aadhar", "pan", "ifsc", "upi", "gstin", "gst", "itr", "otp", "kyc",
    "emi", "neft", "rtgs", "imps", "uan", "epf", "tds", "cvv", "pin",
    # formats / tech terms tagged as places or organisations
    "json", "xml", "csv", "yaml", "sql", "html", "pdf", "api", "sms", "url",
}

# CoNLL entity type → (finding kind, severity)
ENTITY_MAP = {
    "PER": ("person", "medium"),
    "ORG": ("organization", "low"),
    "LOC": ("location", "low"),
}


class NerModel:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str = "auto"):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.session, self.provider = create_session(model_dir / MODEL_FILE, device)
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.last_latency_ms = 0.0

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)

    def _run(self, ids: list[int]) -> np.ndarray:
        input_ids = np.array([[self.cls_id, *ids, self.sep_id]], dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
            "token_type_ids": np.zeros_like(input_ids),
        }
        logits = self.session.run(None, feed)[0][0, 1:-1]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def extract(self, text: str, min_confidence: float = 0.6) -> list[Finding]:
        t0 = time.perf_counter()
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        probs = np.concatenate(
            [self._run(enc.ids[i:i + WINDOW]) for i in range(0, len(enc.ids), WINDOW)]
        ) if enc.ids else np.empty((0, len(self.id2label)))

        findings: list[Finding] = []
        current = None  # [entity_type, start, end, [confidences]]
        for (start, end), p in zip(enc.offsets, probs):
            label = self.id2label[int(p.argmax())]
            if label == "O":
                current = self._close(current, findings, min_confidence, text)
                continue
            prefix, ent = label.split("-", 1)
            # Continue the entity for I- tags, word pieces, or same-type tokens touching it.
            if current and current[0] == ent and (prefix == "I" or start <= current[2] + 1):
                current[2] = end
                current[3].append(float(p.max()))
            else:
                self._close(current, findings, min_confidence, text)
                current = [ent, start, end, [float(p.max())]]
        self._close(current, findings, min_confidence, text)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return findings

    @staticmethod
    def _close(current, findings: list[Finding], min_confidence: float, text: str):
        if not current or current[0] not in ENTITY_MAP:
            return None
        # Snap to whole words: word pieces can tag "IF" inside "IFSC".
        start, end = current[1], current[2]
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        while end < len(text) and text[end].isalnum():
            end += 1
        confidence = float(np.mean(current[3]))
        if confidence >= min_confidence and text[start:end].lower() not in KEYWORD_STOPLIST:
            kind, severity = ENTITY_MAP[current[0]]
            if findings and findings[-1].method == "ner" and findings[-1].end >= start:
                return None  # already covered by the previous (snapped) entity
            findings.append(Finding(kind, severity, start, end, "ner", confidence))
        return None


@lru_cache(maxsize=1)
def get_ner_model(model_dir: str = str(DEFAULT_MODEL_DIR), device: str = "auto") -> NerModel | None:
    """Shared instance, or None when the model hasn't been downloaded."""
    path = Path(model_dir)
    if not (path / MODEL_FILE).is_file():
        return None
    return NerModel(path, device)
