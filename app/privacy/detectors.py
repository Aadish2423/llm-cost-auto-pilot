"""Rule-based detectors for structured personal / secret data.

Every pattern that has a checksum is validated with it (Aadhaar → Verhoeff,
cards → Luhn), so a random 12- or 16-digit number doesn't lock a prompt to
the device. Patterns that are too generic on their own (bank account,
passport) only fire when a context word sits right before them.

Severity drives routing (see guard.py):
    high   — government IDs, financial identifiers, credentials
    medium — direct contact details, personal names (from the NER model)
    low    — organisations, places (context, not identity on their own)
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    kind: str  # "aadhaar", "pan", "phone", ..., "person" (NER)
    severity: str  # "high" | "medium" | "low"
    start: int
    end: int
    method: str  # "rule" | "ner"
    confidence: float = 1.0

    @property
    def label(self) -> str:
        return self.kind.upper()


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


# ── checksums ────────────────────────────────────────────────────────────────

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    """Aadhaar's check digit algorithm."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(digits: str) -> bool:
    """Payment card check digit algorithm."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# ── patterns ─────────────────────────────────────────────────────────────────
# (kind, severity, regex, validator or None, required context regex or None)
# Order matters: earlier detectors win when spans overlap.

_CONTEXT_WINDOW = 40  # chars before a match searched for a context word

DETECTORS = [
    ("api_key", "high", re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
        r"|AIza[0-9A-Za-z_-]{35}|gh[pousr]_[A-Za-z0-9]{36}|xox[abprs]-[A-Za-z0-9-]{10,})"), None, None),
    ("password", "high", re.compile(
        r"(?i)\b(?:password|passwd|pwd|passcode|pin|otp)\s*(?:is|:|=)\s*\S+"), None, None),
    ("card_number", "high", re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"),
     lambda m: luhn_valid(_digits(m)), None),
    ("aadhaar", "high", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
     lambda m: verhoeff_valid(_digits(m)), None),
    ("gstin", "high", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"), None, None),
    ("pan", "high", re.compile(r"\b[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]\b"), None, None),
    ("ifsc", "high", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), None, None),
    ("upi_id", "high", re.compile(
        r"\b[\w.-]{2,}@(?:ok(?:axis|sbi|hdfcbank|icici)|ybl|paytm|upi|apl|ibl|axl|sbi|icici"
        r"|hdfcbank|axisbank|kotak|yesbank|freecharge)\b", re.I), None, None),
    ("bank_account", "high", re.compile(r"(?<!\d)\d{9,18}(?!\d)"), None,
     re.compile(r"(?i)\b(?:account|a/c|acct|acc)\b(?:\s*(?:no|number|num)\b)?\.?\s*[:#-]?\s*$")),
    ("passport", "high", re.compile(r"\b[A-PR-WY][1-9]\d{6}\b"), None,
     re.compile(r"(?i)passport[^.\n]*$")),
    ("email", "medium", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"), None, None),
    ("phone", "medium", re.compile(r"(?<![\w+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), None, None),
]


def detect(text: str) -> list[Finding]:
    """Run every rule; earlier detectors claim overlapping spans first."""
    findings: list[Finding] = []
    taken: list[tuple[int, int]] = []

    for kind, severity, pattern, validator, context in DETECTORS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            if validator and not validator(m.group()):
                continue
            if context and not context.search(text[max(0, start - _CONTEXT_WINDOW):start]):
                continue
            findings.append(Finding(kind, severity, start, end, "rule"))
            taken.append((start, end))

    return sorted(findings, key=lambda f: f.start)
