"""Pydantic request/response models for the Phase 10 API. Kept separate
from main.py so the shape of the API is easy to scan without wading
through endpoint logic.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: Literal["hybrid", "static"] = "hybrid"  # hybrid = on-device/cloud with privacy guard
    verify: bool = False  # run Phase 6's verify_and_escalate() after routing
    log: bool = True  # write the result to Phase 7's database


class CompletionResponse(BaseModel):
    text: str
    ok: bool
    error: str | None
    tier: str
    tier_score: int
    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    verification_status: str | None = None
    escalated: bool = False
    logged_row_id: int | None = None
    placement: str | None = None  # on_device | cloud | blocked
    routing_reason: str | None = None
    privacy_locked: bool = False
    privacy_categories: list[str] = []
    on_device_runtime: str | None = None
    fallback: str | None = None


class ModelInfo(BaseModel):
    provider: str
    model_id: str
    quality_tier: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: float
    local: bool
    available: bool  # whether a required API key is actually present right now


class StatsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    total_cost_usd: float
    total_baseline_cost_usd: float
    savings_usd: float
    savings_pct: float
    verified_count: int
    escalation_count: int
    escalation_rate_pct: float
    on_device_count: int = 0
    cloud_count: int = 0
    blocked_count: int = 0
    on_device_pct: float = 0.0
    privacy_locked_count: int = 0


class PrivacyScanRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PrivacyFindingOut(BaseModel):
    kind: str
    severity: str
    method: str  # rule | ner
    confidence: float
    start: int
    end: int


class PrivacyScanResponse(BaseModel):
    sensitive: bool
    reason: str
    redacted: str
    findings: list[PrivacyFindingOut]
    scan_ms: float
    ner_accelerator: str | None


class RoutingConfigUpdate(BaseModel):
    tier: str  # "tier_1" | "tier_2" | "tier_3"
    provider: str
    model_id: str
