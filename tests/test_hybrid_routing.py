"""Hybrid router guarantees, with model calls faked (offline, free, deterministic).

The property that matters most: a prompt containing personal data is never
sent to a cloud model — not on the happy path, not when the on-device model
fails, and not when no on-device model exists (fail closed).
"""

import sqlite3

import pytest

from app.logging.db import log_request
from app.models.registry import get_model
from app.models.response import LLMResponse
from app.privacy.guard import load_hybrid_config
from app.router import hybrid

LOCAL = get_model("ollama", "llama3.2")
SENSITIVE = "Fill my KYC form: Aadhaar 4995 1234 5670, PAN ABCPE1234F"
SIMPLE = "What is the capital of Japan?"
HARD = ("Design a scalable event-driven architecture for a payments platform, compare Kafka and "
        "RabbitMQ, and justify the trade-offs of event sourcing versus CRUD step by step.")


@pytest.fixture
def cfg():
    base = load_hybrid_config()
    return {**base, "privacy": {**base["privacy"], "use_ner_model": False}}


class CallLog(list):
    """Models the router called, in order; providers in `failing` return errors."""

    def __init__(self):
        super().__init__()
        self.failing: set[str] = set()


@pytest.fixture
def calls(monkeypatch):
    log = CallLog()

    def fake_send(prompt, model):
        log.append(model)
        if model.provider in log.failing:
            return LLMResponse("", 0, 0, 5.0, 0.0, model.model_id, model.provider, error="simulated failure")
        return LLMResponse(f"answer from {model.key}", 20, 30, 5.0,
                           model.estimate_cost(20, 30), model.model_id, model.provider)

    monkeypatch.setattr(hybrid, "send_request", fake_send)
    return log


@pytest.fixture
def local_available(monkeypatch):
    monkeypatch.setattr(hybrid, "pick_on_device_model", lambda cfg=None: (LOCAL, "CPU/GPU (Ollama)"))


@pytest.fixture
def no_local(monkeypatch):
    monkeypatch.setattr(hybrid, "pick_on_device_model", lambda cfg=None: None)


def only_local(calls):
    return all(m.local for m in calls)


def test_sensitive_prompt_stays_on_device(cfg, calls, local_available):
    r = hybrid.route_hybrid(SENSITIVE, cfg)
    assert r.placement == "on_device" and r.privacy_locked
    assert "AADHAAR" in r.reason and "PAN" in r.reason
    assert only_local(calls)


def test_sensitive_prompt_is_blocked_without_on_device_model(cfg, calls, no_local):
    r = hybrid.route_hybrid(SENSITIVE, cfg)
    assert r.placement == "blocked" and not r.response.ok
    assert calls == []  # nothing was sent anywhere


def test_sensitive_prompt_never_falls_back_to_cloud(cfg, calls, local_available):
    calls.failing.add("ollama")
    r = hybrid.route_hybrid(SENSITIVE, cfg)
    assert not r.response.ok and r.fallback is None
    assert only_local(calls)


def test_simple_prompt_runs_on_device(cfg, calls, local_available):
    r = hybrid.route_hybrid(SIMPLE, cfg)
    assert r.placement == "on_device" and r.response.cost_usd == 0.0
    assert only_local(calls)


def test_hard_prompt_goes_to_cloud(cfg, calls, local_available):
    r = hybrid.route_hybrid(HARD, cfg)
    assert r.placement == "cloud" and not r.model_config.local
    assert "quality" in r.reason


def test_cloud_failure_is_answered_on_device(cfg, calls, local_available):
    calls.failing.add(hybrid.load_routing_map()[hybrid.classify(HARD).tier][0])
    r = hybrid.route_hybrid(HARD, cfg)
    assert r.response.ok and r.placement == "on_device"
    assert r.fallback and "cloud failed" in r.fallback


def test_simple_prompt_falls_back_to_cloud_when_on_device_fails(cfg, calls, local_available):
    calls.failing.add("ollama")
    r = hybrid.route_hybrid(SIMPLE, cfg)
    assert r.placement == "cloud" and r.fallback and "on-device failed" in r.fallback


def test_sensitive_prompts_are_logged_redacted(cfg, calls, local_available, tmp_path):
    db = tmp_path / "requests.db"
    row_id = log_request(hybrid.route_hybrid(SENSITIVE, cfg), path=db)
    with sqlite3.connect(db) as conn:
        prompt, locked, categories, redacted = conn.execute(
            "SELECT prompt_text, privacy_locked, privacy_categories, redacted FROM requests WHERE id = ?",
            (row_id,),
        ).fetchone()
    assert "4995" not in prompt and "ABCPE1234F" not in prompt
    assert "[AADHAAR]" in prompt and "[PAN]" in prompt
    assert locked == 1 and redacted == 1 and "aadhaar" in categories
