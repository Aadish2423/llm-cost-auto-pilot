"""Loads config/model_registry.yaml into ModelConfig objects the rest of the
app can query, and does the cost arithmetic for a given token count.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "model_registry.yaml"


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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
