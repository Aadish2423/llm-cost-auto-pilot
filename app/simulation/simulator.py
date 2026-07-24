"""Phase 9a: simulation mode. Replay N prompts against several routing
strategies (always-provider-X, the static Phase 2 router, Phase 5's
multi-objective balanced scorer) and compare projected cost/quality.

Deliberately uses registry math (Phase 3's token estimates, Phase 5's
scoring), not live calls, for every strategy — running N prompts against
every strategy live would need a working key for every provider AND cost
real money per simulation run. This mirrors how Phase 3/5 already work:
predictions from the registry, not measurements from the network. Phase
9b (live_benchmark.py) is the live-call counterpart, run separately and
only against whichever providers actually have keys.
"""

from dataclasses import dataclass, field

from ..classifier.heuristic import classify
from ..models.registry import QUALITY_RANK, ModelConfig, get_model, load_registry
from ..router.cost_predictor import ESTIMATED_OUTPUT_TOKENS, estimate_input_tokens
from ..router.multi_objective import BALANCED, select_model
from ..router.routing_engine import load_routing_map


@dataclass
class StrategyResult:
    name: str
    total_cost_usd: float
    avg_latency_ms: float
    quality_tier_counts: dict[str, int] = field(default_factory=dict)
    unroutable_count: int = 0


@dataclass
class SimulationReport:
    prompt_count: int
    strategies: list[StrategyResult]
    cheapest: StrategyResult
    most_expensive: StrategyResult
    router_savings_vs_most_expensive_pct: float


def _best_model_per_provider(registry: list[ModelConfig]) -> dict[str, ModelConfig]:
    """The highest-quality model each provider offers, ties broken toward
    the cheaper output cost. Used as the "always send everything to
    provider X's best model" baseline strategy.
    """
    best: dict[str, ModelConfig] = {}
    for model in registry:
        current = best.get(model.provider)
        if current is None or (
            QUALITY_RANK[model.quality_tier],
            -model.cost_per_output_token,
        ) > (QUALITY_RANK[current.quality_tier], -current.cost_per_output_token):
            best[model.provider] = model
    return best


def _simulate_fixed_model(prompts: list[str], model: ModelConfig) -> StrategyResult:
    total_cost = 0.0
    quality_counts: dict[str, int] = {}
    for prompt in prompts:
        tier = classify(prompt).tier
        input_tokens = estimate_input_tokens(prompt)
        output_tokens = ESTIMATED_OUTPUT_TOKENS[tier]
        total_cost += model.estimate_cost(input_tokens, output_tokens)
        quality_counts[model.quality_tier.value] = quality_counts.get(model.quality_tier.value, 0) + 1
    return StrategyResult(
        name=f"{model.provider}-only ({model.model_id})",
        total_cost_usd=total_cost,
        avg_latency_ms=model.avg_latency_ms,
        quality_tier_counts=quality_counts,
    )


def _simulate_static_router(prompts: list[str]) -> StrategyResult:
    routing_map = load_routing_map()
    total_cost = 0.0
    total_latency = 0.0
    quality_counts: dict[str, int] = {}
    for prompt in prompts:
        tier = classify(prompt).tier
        provider, model_id = routing_map[tier]
        model = get_model(provider, model_id)
        input_tokens = estimate_input_tokens(prompt)
        output_tokens = ESTIMATED_OUTPUT_TOKENS[tier]
        total_cost += model.estimate_cost(input_tokens, output_tokens)
        total_latency += model.avg_latency_ms
        quality_counts[model.quality_tier.value] = quality_counts.get(model.quality_tier.value, 0) + 1
    n = len(prompts) or 1
    return StrategyResult(
        name="router (static tier map, Phase 2)",
        total_cost_usd=total_cost,
        avg_latency_ms=total_latency / n,
        quality_tier_counts=quality_counts,
    )


def _simulate_multi_objective(prompts: list[str], registry: list[ModelConfig]) -> StrategyResult:
    total_cost = 0.0
    total_latency = 0.0
    quality_counts: dict[str, int] = {}
    unroutable = 0
    routed = 0
    for prompt in prompts:
        result = select_model(prompt, objectives=BALANCED, registry=registry)
        if result.chosen is None:
            unroutable += 1
            continue
        routed += 1
        total_cost += result.chosen.estimated_cost_usd
        total_latency += result.chosen.model_config.avg_latency_ms
        q = result.chosen.model_config.quality_tier.value
        quality_counts[q] = quality_counts.get(q, 0) + 1
    return StrategyResult(
        name="router (multi-objective balanced, Phase 5)",
        total_cost_usd=total_cost,
        avg_latency_ms=(total_latency / routed) if routed else 0.0,
        quality_tier_counts=quality_counts,
        unroutable_count=unroutable,
    )


def simulate(prompts: list[str]) -> SimulationReport:
    if not prompts:
        raise ValueError("simulate() needs at least one prompt")

    registry = load_registry()
    static_router_result = _simulate_static_router(prompts)
    strategies = [_simulate_fixed_model(prompts, model) for model in _best_model_per_provider(registry).values()]
    strategies.append(static_router_result)
    strategies.append(_simulate_multi_objective(prompts, registry))

    cheapest = min(strategies, key=lambda s: s.total_cost_usd)
    most_expensive = max(strategies, key=lambda s: s.total_cost_usd)

    savings_pct = (
        (most_expensive.total_cost_usd - static_router_result.total_cost_usd) / most_expensive.total_cost_usd * 100
        if most_expensive.total_cost_usd > 0
        else 0.0
    )

    return SimulationReport(
        prompt_count=len(prompts),
        strategies=strategies,
        cheapest=cheapest,
        most_expensive=most_expensive,
        router_savings_vs_most_expensive_pct=savings_pct,
    )
