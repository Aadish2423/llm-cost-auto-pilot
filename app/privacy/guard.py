"""The privacy guard: scans a prompt on-device before any routing decision.

    from app.privacy.guard import scan
    report = scan("My PAN is ABCPE1234F, file my return")
    report.sensitive     # True → the prompt must stay on this machine
    report.redacted      # "My PAN is [PAN], file my return"

Two layers, both local:
  1. Rules with checksums (detectors.py) — IDs, financial numbers, secrets.
  2. On-device BERT NER (ner.py) — people, organisations, places.
"""

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .detectors import SEVERITY_RANK, Finding, detect

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "hybrid_config.yaml"


@lru_cache(maxsize=1)
def load_hybrid_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class PrivacyReport:
    findings: list[Finding]
    sensitive: bool
    redacted: str
    lock_on_severity: str
    scan_ms: float
    ner_used: bool
    ner_accelerator: str | None = None
    categories: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.sensitive:
            return "no personal data found"
        locking = sorted({f.label for f in self.findings
                          if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[self.lock_on_severity]})
        return "contains " + ", ".join(locking)


def redact(text: str, findings: list[Finding]) -> str:
    out, cursor = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        if f.start < cursor:  # overlaps a span already redacted
            continue
        out.append(text[cursor:f.start])
        out.append(f"[{f.label}]")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)


def scan(text: str, config: dict | None = None) -> PrivacyReport:
    cfg = (config or load_hybrid_config())["privacy"]
    t0 = time.perf_counter()

    findings = detect(text)
    ner_model = None
    if cfg.get("use_ner_model", True):
        from .ner import get_ner_model

        ner_model = get_ner_model()
        if ner_model is not None:
            taken = [(f.start, f.end) for f in findings]
            findings += [
                f for f in ner_model.extract(text, cfg.get("ner_min_confidence", 0.6))
                if not any(f.start < e and f.end > s for s, e in taken)
            ]
    findings.sort(key=lambda f: f.start)

    threshold = cfg.get("lock_on_severity", "medium")
    sensitive = any(SEVERITY_RANK[f.severity] >= SEVERITY_RANK[threshold] for f in findings)

    return PrivacyReport(
        findings=findings,
        sensitive=sensitive,
        redacted=redact(text, findings),
        lock_on_severity=threshold,
        scan_ms=(time.perf_counter() - t0) * 1000,
        ner_used=ner_model is not None,
        ner_accelerator=ner_model.provider_label if ner_model else None,
        categories=sorted({f.kind for f in findings}),
    )
