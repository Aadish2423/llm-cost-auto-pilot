"""Phase 3 demo: show the pre-call cost/quality prediction for a prompt
across every model in the registry, without calling any provider.

Usage:
    python scripts/predict_cost.py "Your prompt here"
    python scripts/predict_cost.py                     (uses a demo prompt)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.router.cost_predictor import predict_cost  # noqa: E402
from app.router.routing_engine import load_routing_map  # noqa: E402

DEMO_PROMPT = (
    "Analyze the sentiment of this review and explain your reasoning: "
    "'The food was decent but the service was painfully slow.'"
)


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or DEMO_PROMPT
    prediction = predict_cost(prompt)
    routed_provider, routed_model_id = load_routing_map()[prediction.prompt_tier]

    shown_prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
    print(f'Prompt: "{shown_prompt}"')
    print(f"Classified as: {prediction.prompt_tier.value} (score={prediction.tier_score})")
    if not prediction.estimates:
        print("No models in the registry to estimate against.")
        return
    print(f"Estimated input tokens: {prediction.estimates[0].estimated_input_tokens}")
    print(f"Estimated output tokens: {prediction.estimates[0].estimated_output_tokens}")
    print("(both are rough ~4-chars-per-token / per-tier estimates, not exact — see README)")
    print()

    widths = [max(50, max(len(e.model_config.key) for e in prediction.estimates)), 9, 12, 10, 24]
    headers = ["model", "quality", "est. cost", "meets bar", ""]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for est in prediction.estimates:
        is_current_route = (
            est.model_config.provider == routed_provider
            and est.model_config.model_id == routed_model_id
        )
        is_recommended = est is prediction.recommended
        flags = [f for f, on in [("RECOMMENDED", is_recommended), ("current route", is_current_route)] if on]
        row = [
            est.model_config.key,
            est.model_config.quality_tier.value,
            f"${est.estimated_cost_usd:.6f}",
            "yes" if est.meets_quality_bar else "no",
            ", ".join(flags),
        ]
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
