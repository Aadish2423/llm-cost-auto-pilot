"""Phase 5 demo: show how the SAME prompt gets routed differently under
different objective profiles (balanced / cost-focused / latency-focused /
quality-focused / on-prem-only). Never calls a provider - pure scoring
against the registry, like Phase 3's cost predictor.

Usage:
    python scripts/select_model_demo.py "Your prompt here"
    python scripts/select_model_demo.py                     (demo prompt)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.router.multi_objective import (  # noqa: E402
    BALANCED,
    COST_FOCUSED,
    LATENCY_FOCUSED,
    ON_PREM_ONLY,
    QUALITY_FOCUSED,
    RoutingObjectives,
    select_model,
)

STRICT_LATENCY = RoutingObjectives(cost_weight=0.34, latency_weight=0.33, quality_weight=0.33, max_latency_ms=300)

DEMO_PROMPT = (
    "Given the constraints: budget under $500, must be waterproof, needs GPS - "
    "recommend a category of device and explain your reasoning."
)

PROFILES = [
    ("balanced", BALANCED),
    ("cost-focused", COST_FOCUSED),
    ("latency-focused", LATENCY_FOCUSED),
    ("quality-focused", QUALITY_FOCUSED),
    ("on-prem-only (hard constraint)", ON_PREM_ONLY),
    ("latency <300ms (hard constraint)", STRICT_LATENCY),
]


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or DEMO_PROMPT
    shown_prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
    print(f'Prompt: "{shown_prompt}"')
    print()

    headers = ["profile", "chosen model", "score", "est. cost", "excluded"]
    rows = []
    for name, objectives in PROFILES:
        result = select_model(prompt, objectives=objectives)
        if result.chosen:
            rows.append(
                [
                    name,
                    result.chosen.model_config.key,
                    f"{result.chosen.score:.3f}",
                    f"${result.chosen.estimated_cost_usd:.6f}",
                    str(result.excluded_by_constraints),
                ]
            )
        else:
            rows.append([name, "(no eligible model)", "-", "-", str(result.excluded_by_constraints)])

    # Width per column = longest cell actually present (header or data),
    # not a guessed constant - a fixed guess breaks alignment the moment a
    # profile name or model key longer than expected shows up (this
    # happened once already with together's long model keys in Phase 3).
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print()
    print("'excluded' = registry models dropped by hard constraints before scoring")
    print("(quality bar for this tier, require_local, max_latency_ms) - not by weights.")


if __name__ == "__main__":
    main()
