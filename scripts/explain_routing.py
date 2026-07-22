"""Phase 4 demo: route a real prompt and print a full explanation of the
decision — tier, confidence, matched signals, the model chosen, and the
cheapest qualifying alternative. This DOES make a real provider call
through route_request() (unlike the Phase 2/3 demo scripts), so it will
hit a live API or fail cleanly depending on which keys are in .env.

Usage:
    python scripts/explain_routing.py "Your prompt here"
    python scripts/explain_routing.py                     (uses a demo prompt)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.router.explainer import explain  # noqa: E402
from app.router.routing_engine import route_request  # noqa: E402

DEMO_PROMPT = (
    "Summarize in 2 sentences: The company reported record revenue this "
    "quarter driven by strong cloud subscription growth, though margins "
    "narrowed slightly due to increased infrastructure spending."
)


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or DEMO_PROMPT
    result = route_request(prompt)
    exp = explain(result)

    shown_prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
    print(f'Prompt: "{shown_prompt}"')
    print()
    print("=== Classification ===")
    print(f"Tier: {exp.tier.value}  (score={exp.tier_score}, confidence={exp.confidence:.0%})")
    print("Signals:")
    for signal in exp.matched_signals or ["(none matched — defaulted to tier_1)"]:
        print(f"  - {signal}")
    print()
    print("=== Routing decision ===")
    print(f"Chosen model: {exp.chosen_model}  (quality: {exp.chosen_quality_tier})")
    print(f"Estimated cost: ${exp.chosen_cost_usd:.6f}   avg latency: {exp.chosen_avg_latency_ms:.0f} ms")
    print()
    if exp.alternative_model:
        print("=== Cheapest qualifying alternative ===")
        print(f"Model: {exp.alternative_model}  (quality: {exp.alternative_quality_tier})")
        print(f"Estimated cost: ${exp.alternative_cost_usd:.6f}   avg latency: {exp.alternative_avg_latency_ms:.0f} ms")
        direction = "MORE" if exp.cost_delta_usd > 0 else "LESS"
        print(f"Chosen model costs ${abs(exp.cost_delta_usd):.6f} {direction} than this alternative")
        print(f"Quality: {exp.quality_delta}")
    else:
        print("No qualifying alternative found (chosen model is the only one that meets the bar).")
    print()
    print("=== Actual response ===")
    print(f"ok: {result.response.ok}")
    if result.response.ok:
        print(f"text: {result.response.text[:200]}")
        print(f"actual cost: ${result.response.cost_usd:.6f}   actual latency: {result.response.latency_ms:.0f} ms")
    else:
        print(f"error: {result.response.error}")


if __name__ == "__main__":
    main()
