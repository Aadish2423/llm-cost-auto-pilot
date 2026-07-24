"""Phase 9a demo: replay a batch of prompts against several routing
strategies (always-provider-X, the static router, multi-objective
balanced) and compare projected cost. Never calls a provider - pure
registry math, like Phase 3/5's demos.

Usage:
    python scripts/simulate_demo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.simulation.simulator import simulate  # noqa: E402

# Reuses the same 15-prompt spread as seed_demo_data.py for consistency
# across the project's demo scripts.
SIMULATION_PROMPTS = [
    "What is the capital of France?",
    "Extract the date and amount from: 'Invoice #4471, due $1,250.00 on 2026-08-01.'",
    "Reformat this list into a comma-separated string: apple, banana, cherry",
    "Translate 'good morning' into Spanish.",
    "What is the boiling point of water in Celsius?",
    "Summarize in 2 sentences: The company reported record revenue this quarter "
    "driven by strong cloud subscription growth, though margins narrowed slightly.",
    "Classify this support ticket as billing, technical, or account: "
    "'I was charged twice for my subscription this month.'",
    "Compare the tradeoffs between REST and GraphQL APIs in a short paragraph.",
    "Categorize this document type: a one-page letter with a signature block and date.",
    "Write a short, creative product description for a reusable coffee cup.",
    "Given the constraints: budget under $500, must be waterproof, needs GPS - "
    "recommend a category of device and explain your reasoning.",
    "Analyze the sentiment of this review and explain your reasoning: "
    "'The food was decent but the service was painfully slow.'",
    "Walk through, step by step, how you would debug a Python function that "
    "returns None instead of the expected list.",
    "Design a simple onboarding flow for a mobile banking app and justify each step.",
    "Argue for and against remote work in a balanced short essay.",
]


def main() -> None:
    report = simulate(SIMULATION_PROMPTS)

    print(f"Simulated {report.prompt_count} prompts across {len(report.strategies)} strategies\n")

    headers = ["strategy", "total cost", "avg latency", "quality mix", "unroutable"]
    rows = []
    for s in sorted(report.strategies, key=lambda s: s.total_cost_usd):
        quality_mix = ", ".join(f"{k}={v}" for k, v in sorted(s.quality_tier_counts.items()))
        rows.append(
            [s.name, f"${s.total_cost_usd:.6f}", f"{s.avg_latency_ms:.0f} ms", quality_mix, str(s.unroutable_count)]
        )
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print()
    print(f"Cheapest strategy: {report.cheapest.name} (${report.cheapest.total_cost_usd:.6f})")
    print(f"Most expensive strategy: {report.most_expensive.name} (${report.most_expensive.total_cost_usd:.6f})")
    print(
        f"Static router saves {report.router_savings_vs_most_expensive_pct:.1f}% "
        f"vs. always using the most expensive strategy"
    )


if __name__ == "__main__":
    main()
