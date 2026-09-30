"""Privacy guard: checksums, detectors, redaction, and the labelled prompt set
from scripts/test_privacy_guard.py. Cases that need the BERT NER model
(names) are skipped when models/bert-base-ner hasn't been downloaded."""

import pytest

from app.privacy.detectors import detect, luhn_valid, verhoeff_valid
from app.privacy.guard import load_hybrid_config, redact, scan
from app.privacy.ner import get_ner_model
from scripts.test_privacy_guard import CASES

NER_AVAILABLE = get_ner_model() is not None
NER_KINDS = {"person", "organization", "location"}


def test_verhoeff_checksum():
    assert verhoeff_valid("499512345670")
    assert not verhoeff_valid("499512345671")


def test_luhn_checksum():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


def test_random_12_digit_number_is_not_aadhaar():
    # Right shape, wrong check digit → must not lock the prompt.
    assert not any(f.kind == "aadhaar" for f in detect("Reference 4995 1234 5671"))


def test_bank_account_needs_context_word():
    assert any(f.kind == "bank_account" for f in detect("account number 50100234567890"))
    assert not any(f.kind == "bank_account" for f in detect("population 50100234567890"))


def test_redact_masks_values_and_keeps_text():
    text = "PAN ABCPE1234F, phone 9876543210"
    assert redact(text, detect(text)) == "PAN [PAN], phone [PHONE]"


def test_rules_only_mode_still_locks_ids():
    cfg = {**load_hybrid_config(), "privacy": {**load_hybrid_config()["privacy"], "use_ner_model": False}}
    report = scan("My Aadhaar is 4995 1234 5670", cfg)
    assert report.sensitive and not report.ner_used and "[AADHAAR]" in report.redacted


@pytest.mark.parametrize("prompt,should_lock,kinds", CASES, ids=[c[0][:40] for c in CASES])
def test_labelled_prompts(prompt, should_lock, kinds):
    if kinds & NER_KINDS and not NER_AVAILABLE:
        pytest.skip("needs models/bert-base-ner (python scripts/download_models.py)")
    report = scan(prompt)
    assert report.sensitive == should_lock, report.redacted
    assert kinds <= set(report.categories)
