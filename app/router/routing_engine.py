"""Ties the classifier, the routing map, and Phase 1's dispatcher together
into the one function the rest of the app will call:

    from app.router.routing_engine import route_request
    result = route_request("Summarize this...")
    print(result.tier, result.model_config.key, result.response.cost_usd)
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..classifier.heuristic import classify
from ..classifier.tiers import ComplexityTier
from ..models.dispatcher import send_request
from ..models.registry import ModelConfig, get_model
from ..models.response import LLMResponse

ROUTING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing_config.yaml"


@dataclass
class RoutingResult:
    prompt: str
    response: LLMResponse
    tier: ComplexityTier
    score: int
    matched_signals: list[str]
    model_config: ModelConfig


def load_routing_map(path: Path = ROUTING_CONFIG_PATH) -> dict[ComplexityTier, tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "routing" not in raw:
        raise ValueError(f"{path} is missing its top-level 'routing' key")

    routing_map = {
        ComplexityTier(tier_key): (entry["provider"], entry["model_id"])
        for tier_key, entry in raw["routing"].items()
    }

    missing = [tier.value for tier in ComplexityTier if tier not in routing_map]
    if missing:
        raise ValueError(f"{path} is missing routing entries for: {missing}")

    return routing_map


def route_request(prompt: str) -> RoutingResult:
    classification = classify(prompt)
    provider, model_id = load_routing_map()[classification.tier]
    model_config = get_model(provider, model_id)
    response = send_request(prompt, model_config)

    return RoutingResult(
        prompt=prompt,
        response=response,
        tier=classification.tier,
        score=classification.score,
        matched_signals=classification.matched_signals,
        model_config=model_config,
    )
