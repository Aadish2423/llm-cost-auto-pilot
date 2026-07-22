"""Phase 2 sanity check: run the heuristic classifier against a spread of
example prompts and eyeball whether the tier assignments look reasonable.
No API keys or network calls needed — this only exercises app/classifier/.

Usage:
    python scripts/test_classifier.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.classifier.heuristic import classify  # noqa: E402

EXAMPLE_PROMPTS = [
    ("What is the capital of France?", "tier_1"),
    ("Extract the date and amount from: 'Invoice #4471, due $1,250.00 on 2026-08-01.'", "tier_1"),
    ("Reformat this list into a comma-separated string: apple, banana, cherry", "tier_1"),
    (
        "Summarize in 2 sentences: The company reported record revenue this quarter "
        "driven by strong cloud subscription growth, though margins narrowed slightly "
        "due to increased infrastructure spending.",
        "tier_2",
    ),
    (
        "Classify this support ticket as billing, technical, or account: "
        "'I was charged twice for my subscription this month.'",
        "tier_2",
    ),
    ("Compare the tradeoffs between REST and GraphQL APIs in a short paragraph.", "tier_2"),
    ("Write a short, creative product description for a reusable coffee cup.", "tier_2"),
    (
        "Given the constraints: budget under $500, must be waterproof, needs GPS — "
        "recommend a category of device and explain your reasoning.",
        "tier_3",
    ),
    (
        "Analyze the sentiment of this review and explain your reasoning: "
        "'The food was decent but the service was painfully slow.'",
        "tier_2",
    ),
    (
        "Walk through, step by step, how you would debug a Python function that "
        "returns None instead of the expected list.",
        "tier_3",
    ),
]


def main() -> None:
    correct = 0
    rows = []
    results = []

    for prompt, expected in EXAMPLE_PROMPTS:
        result = classify(prompt)
        results.append(result)
        got = result.tier.value
        match = "OK" if got == expected else "MISMATCH"
        correct += got == expected
        rows.append((prompt[:60] + ("..." if len(prompt) > 60 else ""), expected, got, result.score, match))

    widths = [60, 8, 8, 6, 8]
    headers = ["prompt", "expected", "got", "score", ""]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for prompt, expected, got, score, match in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip([prompt, expected, got, score, match], widths)))

    print(f"\n{correct}/{len(EXAMPLE_PROMPTS)} matched my own expected tier (a hand guess, not ground truth).")
    print("Mismatches aren't necessarily bugs — the heuristic is v1 and expected to be imperfect.")
    print("Run with -v to see the matched signals behind each score:")
    if "-v" in sys.argv:
        print()
        for (prompt, _), result in zip(EXAMPLE_PROMPTS, results):
            print(f"[{result.tier.value} score={result.score}] {prompt[:70]}")
            for signal in result.matched_signals:
                print(f"    - {signal}")


if __name__ == "__main__":
    main()
