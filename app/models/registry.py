"""Loads config/model_registry.yaml into ModelConfig objects the rest of the
app can query, and does the cost arithmetic for a given token count.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "model_registry.yaml"

# Single source of truth for "which env var does this provider's key live
# in." Used by is_model_available() below - originally duplicated as a
# script-local dict in scripts/test_providers.py before being consolidated
# here so Phase 9's live benchmarking can reuse the exact same check.
ENV_VAR_BY_PROVIDER: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
}


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Single source of truth for "which quality tier beats which." Declaration
# order above is HIGH/MEDIUM/LOW (alphabetical-ish, not rank order), so
# anything that needs to compare tiers must use this mapping rather than
# enum declaration order or a locally-redefined list — three separate
# modules independently redefined this before it was consolidated here,
# which is exactly the kind of thing that silently drifts out of sync.
QUALITY_RANK: dict[QualityTier, int] = {
    QualityTier.LOW: 0,
    QualityTier.MEDIUM: 1,
    QualityTier.HIGH: 2,
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: float
    quality_tier: QualityTier
    requires_api_key: bool = True
    local: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_id}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


def load_registry(path: Path = REGISTRY_PATH) -> list[ModelConfig]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return [
        ModelConfig(
            provider=entry["provider"],
            model_id=entry["model_id"],
            cost_per_input_token=entry["cost_per_input_token"],
            cost_per_output_token=entry["cost_per_output_token"],
            avg_latency_ms=entry["avg_latency_ms"],
            quality_tier=QualityTier(entry["quality_tier"]),
            requires_api_key=entry.get("requires_api_key", True),
            local=entry.get("local", False),
        )
        for entry in raw["models"]
    ]


def get_model(provider: str, model_id: str, path: Path = REGISTRY_PATH) -> ModelConfig:
    for model in load_registry(path):
        if model.provider == provider and model.model_id == model_id:
            return model
    raise ValueError(f"No model '{model_id}' registered for provider '{provider}'")


def models_by_provider(path: Path = REGISTRY_PATH) -> dict[str, list[ModelConfig]]:
    grouped: dict[str, list[ModelConfig]] = {}
    for model in load_registry(path):
        grouped.setdefault(model.provider, []).append(model)
    return grouped


def is_model_available(model: ModelConfig) -> tuple[bool, str]:
    """Whether this model can actually be called right now: local models
    need no key; everything else needs its provider's env var set. Only
    checks presence, not validity - a stray/wrong key still counts as
    "available" here and will fail with a clean ProviderError at call time.
    """
    if model.local:
        return True, ""
    env_var = ENV_VAR_BY_PROVIDER.get(model.provider)
    if env_var and not os.environ.get(env_var):
        return False, f"missing {env_var}"
    return True, ""
