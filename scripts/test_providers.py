"""Phase 1 validation: send the same 10 prompts to every model in the
registry, print a summary table, and save raw results to data/.

Models whose required API key is missing (or, for Ollama, that aren't
reachable) are skipped with a clear message rather than crashing the run.

Usage:
    python scripts/test_providers.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# override=True: this project's .env always wins over stray global env vars
# (e.g. an OPENAI_API_KEY left set in your shell from another project).
load_dotenv(ROOT / ".env", override=True)

from app.models.dispatcher import send_request  # noqa: E402
from app.models.registry import is_model_available, load_registry  # noqa: E402

TEST_PROMPTS = [
    "What is the capital of France?",
    "Extract the date and amount from: 'Invoice #4471, due $1,250.00 on 2026-08-01.'",
    "Reformat this list into a comma-separated string: apple, banana, cherry",
    "Summarize in 2 sentences: The company reported record revenue this quarter "
    "driven by strong cloud subscription growth, though margins narrowed slightly "
    "due to increased infrastructure spending.",
    "Classify this support ticket as billing, technical, or account: "
    "'I was charged twice for my subscription this month.'",
    "Compare the tradeoffs between REST and GraphQL APIs in a short paragraph.",
    "Write a short, creative product description for a reusable coffee cup.",
    "Given the constraints: budget under $500, must be waterproof, needs GPS — "
    "recommend a category of device and explain your reasoning.",
    "Analyze the sentiment of this review and explain your reasoning: "
    "'The food was decent but the service was painfully slow.'",
    "Walk through, step by step, how you would debug a Python function that "
    "returns None instead of the expected list.",
]


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    table = [headers, *rows]
    widths = [max(len(str(row[index])) for row in table) for index in range(len(headers))]

    def format_row(row: list[str]) -> str:
        return " | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)



def main() -> None:
    registry = load_registry()
    all_results = []
    summary_rows = []

    for model in registry:
        available, reason = is_model_available(model)
        if not available:
            print(f"SKIPPED {model.key} ({reason})")
            summary_rows.append([model.key, "skipped", reason, "-", "-", "-"])
            continue

        print(f"Testing {model.key} ...")
        successes, total_cost, total_latency = 0, 0.0, 0.0

        for prompt in TEST_PROMPTS:
            response = send_request(prompt, model)
            all_results.append(
                {
                    "model": model.key,
                    "prompt": prompt,
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    "error": response.error,
                }
            )
            if response.ok:
                successes += 1
                total_cost += response.cost_usd
                total_latency += response.latency_ms
            else:
                print(f"  error: {response.error}")

        n = len(TEST_PROMPTS)
        avg_cost = total_cost / successes if successes else 0.0
        avg_latency = total_latency / successes if successes else 0.0
        summary_rows.append(
            [
                model.key,
                f"{successes}/{n} ok",
                "-",
                f"${avg_cost:.6f}",
                f"{avg_latency:.0f} ms",
                f"${total_cost:.6f}",
            ]
        )

    print()
    print(format_table(summary_rows, ["model", "success", "skip reason", "avg cost/req", "avg latency", "total cost"]))

    out_path = ROOT / "data" / "phase1_test_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {out_path.relative_to(ROOT)}")
    print(f"Run at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
