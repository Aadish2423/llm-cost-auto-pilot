"""Phase 5: multi-objective routing. Instead of "cheapest model that meets
a quality bar" (Phase 3), score every eligible candidate across cost,
latency, and quality using configurable weights, and pick the best
combined score.

Data-locality is deliberately NOT a weight — a request that must stay
on-prem ("Hospital -> must stay on-prem -> choose Ollama" from the
original design notes) doesn't get a spectrum, it gets a hard filter down
to local models only, same as a hard latency ceiling. Weights only apply
to candidates that already survive the hard constraints.
"""

from dataclasses import dataclass

from ..classifier.heuristic import classify
from ..classifier.tiers import ComplexityTier
from ..models.registry import ModelConfig, QualityTier, load_registry
from .cost_predictor import ESTIMATED_OUTPUT_TOKENS, MIN_QUALITY_FOR_TIER, estimate_input_tokens

_QUALITY_ORDER = ["low", "medium", "high"]


def _quality_rank(tier: QualityTier) -> int:
    return _QUALITY_ORDER.index(tier.value)


@dataclass
class RoutingObjectives:
    """Weights are relative, not required to sum to 1 - they're normalized
    internally. Defaults bias toward cost since that's this project's core
    mission.

    require_local and max_latency_ms are hard constraints: candidates that
    fail them are dropped before scoring, not penalized within it.
    min_quality overrides the tier's own default quality bar (from
    cost_predictor.MIN_QUALITY_FOR_TIER) if you want to demand more or
    allow less than the tier's usual minimum.
    """

    cost_weight: float = 0.5
    latency_weight: float = 0.2
    quality_weight: float = 0.3
    require_local: bool = False
    max_latency_ms: float | None = None
    min_quality: QualityTier | None = None


# Named profiles for common cases, matching the original design notes'
# examples ("need response <500ms -> choose Groq", "must stay on-prem").
BALANCED = RoutingObjectives(cost_weight=0.34, latency_weight=0.33, quality_weight=0.33)
COST_FOCUSED = RoutingObjectives(cost_weight=0.7, latency_weight=0.1, quality_weight=0.2)
LATENCY_FOCUSED = RoutingObjectives(cost_weight=0.2, latency_weight=0.6, quality_weight=0.2)
QUALITY_FOCUSED = RoutingObjectives(cost_weight=0.2, latency_weight=0.1, quality_weight=0.7)
ON_PREM_ONLY = RoutingObjectives(cost_weight=0.5, latency_weight=0.2, quality_weight=0.3, require_local=True)


@dataclass
class ScoredCandidate:
    model_config: ModelConfig
    estimated_cost_usd: float
    score: float
    cost_score: float
    latency_score: float
    quality_score: float


@dataclass
class MultiObjectiveResult:
    tier: ComplexityTier
    objectives: RoutingObjectives
    candidates: list[ScoredCandidate]  # sorted best first
    chosen: ScoredCandidate | None
    excluded_by_constraints: int  # registry models dropped before scoring


def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [
        (v - lo) / (hi - lo) if higher_is_better else 1 - (v - lo) / (hi - lo)
        for v in values
    ]


def select_model(
    prompt: str,
    objectives: RoutingObjectives = BALANCED,
    registry: list[ModelConfig] | None = None,
) -> MultiObjectiveResult:
    if objectives.cost_weight + objectives.latency_weight + objectives.quality_weight <= 0:
        raise ValueError("RoutingObjectives weights must sum to a positive number")

    classification = classify(prompt)
    tier = classification.tier
    models = registry if registry is not None else load_registry()

    input_tokens = estimate_input_tokens(prompt)
    output_tokens = ESTIMATED_OUTPUT_TOKENS[tier]
    min_rank = _quality_rank(objectives.min_quality or MIN_QUALITY_FOR_TIER[tier])

    eligible = [
        model
        for model in models
        if not (objectives.require_local and not model.local)
        and _quality_rank(model.quality_tier) >= min_rank
        and not (objectives.max_latency_ms is not None and model.avg_latency_ms > objectives.max_latency_ms)
    ]
    excluded = len(models) - len(eligible)

    if not eligible:
        return MultiObjectiveResult(
            tier=tier, objectives=objectives, candidates=[], chosen=None, excluded_by_constraints=excluded
        )

    costs = [m.estimate_cost(input_tokens, output_tokens) for m in eligible]
    latencies = [m.avg_latency_ms for m in eligible]
    qualities = [float(_quality_rank(m.quality_tier)) for m in eligible]

    cost_scores = _normalize(costs, higher_is_better=False)
    latency_scores = _normalize(latencies, higher_is_better=False)
    quality_scores = _normalize(qualities, higher_is_better=True)

    total_weight = objectives.cost_weight + objectives.latency_weight + objectives.quality_weight

    candidates = [
        ScoredCandidate(
            model_config=model,
            estimated_cost_usd=cost,
            score=(
                objectives.cost_weight * cost_s
                + objectives.latency_weight * lat_s
                + objectives.quality_weight * qual_s
            )
            / total_weight,
            cost_score=cost_s,
            latency_score=lat_s,
            quality_score=qual_s,
        )
        for model, cost, cost_s, lat_s, qual_s in zip(eligible, costs, cost_scores, latency_scores, quality_scores)
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)

    return MultiObjectiveResult(
        tier=tier,
        objectives=objectives,
        candidates=candidates,
        chosen=candidates[0],
        excluded_by_constraints=excluded,
    )
