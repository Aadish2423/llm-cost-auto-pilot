"""Phase 3: estimate what a prompt would cost across every candidate model
BEFORE sending it anywhere. This never calls a provider — it's pure token
math against the registry, using the Phase 2 classifier's tier to guess at
output length.
"""

from dataclasses import dataclass

from ..classifier.heuristic import classify
from ..classifier.tiers import ComplexityTier
from ..models.registry import QUALITY_RANK, ModelConfig, QualityTier, load_registry

# Rough ~4-chars-per-token rule of thumb for English text. Real tokenizers
# differ per provider/model; this is a prediction, not a guarantee.
CHARS_PER_TOKEN_ESTIMATE = 4

# Output length is unknown until generation actually happens. Tied to the
# same complexity tier the classifier already produces, on the assumption
# simple tasks get short answers and complex tasks get long ones. Starting
# assumptions, not measured data — Phase 7's logging will let these be
# replaced with real historical averages per model/tier.
ESTIMATED_OUTPUT_TOKENS = {
    ComplexityTier.TIER_1: 80,
    ComplexityTier.TIER_2: 250,
    ComplexityTier.TIER_3: 500,
}

# Minimum acceptable quality tier per complexity tier — used only to filter
# the "recommended" pick, matching the project's original tier -> quality
# mapping (Tier 1: cheapest is fine, Tier 3: should be a strong model).
MIN_QUALITY_FOR_TIER = {
    ComplexityTier.TIER_1: QualityTier.LOW,
    ComplexityTier.TIER_2: QualityTier.MEDIUM,
    ComplexityTier.TIER_3: QualityTier.HIGH,
}


@dataclass
class CostEstimate:
    model_config: ModelConfig
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    meets_quality_bar: bool


@dataclass
class CostPrediction:
    prompt_tier: ComplexityTier
    tier_score: int
    estimates: list[CostEstimate]  # sorted cheapest first
    recommended: CostEstimate | None


def estimate_input_tokens(prompt: str) -> int:
    return max(1, round(len(prompt) / CHARS_PER_TOKEN_ESTIMATE))


def predict_cost(prompt: str, registry: list[ModelConfig] | None = None) -> CostPrediction:
    classification = classify(prompt)
    tier = classification.tier
    models = registry if registry is not None else load_registry()

    input_tokens = estimate_input_tokens(prompt)
    output_tokens = ESTIMATED_OUTPUT_TOKENS[tier]
    min_quality = MIN_QUALITY_FOR_TIER[tier]

    estimates = [
        CostEstimate(
            model_config=model,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=model.estimate_cost(input_tokens, output_tokens),
            meets_quality_bar=QUALITY_RANK[model.quality_tier] >= QUALITY_RANK[min_quality],
        )
        for model in models
    ]
    estimates.sort(key=lambda e: e.estimated_cost_usd)

    qualifying = [e for e in estimates if e.meets_quality_bar]
    recommended = qualifying[0] if qualifying else (estimates[0] if estimates else None)

    return CostPrediction(
        prompt_tier=tier,
        tier_score=classification.score,
        estimates=estimates,
        recommended=recommended,
    )
