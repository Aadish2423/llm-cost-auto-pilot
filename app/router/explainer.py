"""Phase 4: turn a routing decision into a human-readable explanation —
why this tier, how confident, what got picked, and what the cheapest
qualifying alternative would have cost instead. Reuses Phase 2's
classifier signals and Phase 3's cost predictor; adds no new estimation
logic of its own.
"""

from dataclasses import dataclass

from ..classifier.heuristic import TIER_2_MIN_SCORE, TIER_3_MIN_SCORE
from ..classifier.tiers import ComplexityTier
from .cost_predictor import predict_cost
from .routing_engine import RoutingResult

# How far the raw classifier score sits from the nearest tier boundary,
# mapped onto 0.5-1.0. NOT a calibrated probability — a score sitting
# right on a boundary means "could easily have landed in the neighboring
# tier" (confidence ~0.5); a score deep inside a tier's range means
# several signals agreed (confidence approaches 1.0). Documented as a
# heuristic-of-a-heuristic on purpose, same honesty rule as Phase 3's
# refusal to show a fabricated quality percentage.
_CONFIDENCE_SLOPE = 0.12

_QUALITY_ORDER = ["low", "medium", "high"]


def _confidence_from_score(tier: ComplexityTier, score: int) -> float:
    if tier == ComplexityTier.TIER_1:
        distance = TIER_2_MIN_SCORE - score
    elif tier == ComplexityTier.TIER_2:
        distance = min(score - TIER_2_MIN_SCORE, TIER_3_MIN_SCORE - score)
    else:
        distance = score - TIER_3_MIN_SCORE

    distance = max(distance, 0)
    return round(min(1.0, 0.5 + _CONFIDENCE_SLOPE * distance), 2)


@dataclass
class RoutingExplanation:
    tier: ComplexityTier
    tier_score: int
    confidence: float
    matched_signals: list[str]

    chosen_model: str
    chosen_quality_tier: str
    chosen_cost_usd: float
    chosen_avg_latency_ms: float
    chosen_meets_quality_bar: bool

    alternative_model: str | None
    alternative_quality_tier: str | None
    alternative_cost_usd: float | None
    alternative_avg_latency_ms: float | None

    cost_delta_usd: float | None  # chosen minus alternative; positive = chosen costs more
    quality_delta: str | None


def explain(routing_result: RoutingResult) -> RoutingExplanation:
    confidence = _confidence_from_score(routing_result.tier, routing_result.score)
    chosen = routing_result.model_config
    chosen_key = chosen.key

    prediction = predict_cost(routing_result.prompt)
    chosen_estimate = next(
        (e for e in prediction.estimates if e.model_config.key == chosen_key), None
    )
    alternative = next(
        (
            e
            for e in prediction.estimates
            if e.model_config.key != chosen_key and e.meets_quality_bar
        ),
        None,
    )

    chosen_cost = chosen_estimate.estimated_cost_usd if chosen_estimate else routing_result.response.cost_usd
    # Conservative default (flag as a concern) for the practically-unreachable
    # case where the chosen model isn't found in the registry scan at all.
    chosen_meets_bar = chosen_estimate.meets_quality_bar if chosen_estimate else False

    cost_delta = None
    quality_delta = None
    if alternative is not None:
        cost_delta = chosen_cost - alternative.estimated_cost_usd
        chosen_rank = _QUALITY_ORDER.index(chosen.quality_tier.value)
        alt_rank = _QUALITY_ORDER.index(alternative.model_config.quality_tier.value)
        if chosen_rank == alt_rank:
            quality_delta = "same quality tier"
        elif chosen_rank > alt_rank:
            quality_delta = f"chosen is {chosen_rank - alt_rank} quality tier(s) higher"
        else:
            quality_delta = f"chosen is {alt_rank - chosen_rank} quality tier(s) lower"

    return RoutingExplanation(
        tier=routing_result.tier,
        tier_score=routing_result.score,
        confidence=confidence,
        matched_signals=routing_result.matched_signals,
        chosen_model=chosen_key,
        chosen_quality_tier=chosen.quality_tier.value,
        chosen_cost_usd=chosen_cost,
        chosen_avg_latency_ms=chosen.avg_latency_ms,
        chosen_meets_quality_bar=chosen_meets_bar,
        alternative_model=alternative.model_config.key if alternative else None,
        alternative_quality_tier=alternative.model_config.quality_tier.value if alternative else None,
        alternative_cost_usd=alternative.estimated_cost_usd if alternative else None,
        alternative_avg_latency_ms=alternative.model_config.avg_latency_ms if alternative else None,
        cost_delta_usd=cost_delta,
        quality_delta=quality_delta,
    )
