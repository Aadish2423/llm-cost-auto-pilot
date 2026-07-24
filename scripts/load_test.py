"""Phase 11: load test. Push a batch of diverse, programmatically-varied
prompts through the full route + log pipeline and report the final
aggregate numbers — this is the actual portfolio deliverable the whole
project builds up to.

Scope note: the original design calls for 500-1000 prompts. This
session's live run used a smaller batch (see README for the actual count
and why) — Ollama is the only consistently free live provider available
right now, and each call takes several seconds to tens of seconds on
this hardware, so 500-1000 sequential calls would take multiple hours.
The script itself scales to any --count; run it larger (ideally with a
second live key so calls could run in parallel) for a bigger real run.

Usage:
    python scripts/load_test.py --count 50
    python scripts/load_test.py --count 50 --verify
"""

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.logging.db import get_summary, log_request  # noqa: E402
from app.router.routing_engine import route_request  # noqa: E402
from app.router.verifier import verify_and_escalate  # noqa: E402

TOPICS = [
    "France", "Japan", "Brazil", "Canada", "Egypt", "Kenya", "Norway", "Vietnam",
    "electric cars", "solar panels", "quantum computing", "coral reefs",
    "the printing press", "the stock market", "machine learning", "jazz music",
]

PROMPT_TEMPLATES = [
    # Tier 1: simple
    lambda t: f"What is the capital of {t}?" if t[0].isupper() else f"What is {t}?",
    lambda t: f"Extract the key entity from this sentence: 'A report on {t} was published today.'",
    lambda t: f"Reformat this into a comma-separated list: {t}, research, data, summary",
    lambda t: f"Convert this to uppercase: {t}",
    # Tier 2: moderate
    lambda t: f"Summarize in 2 sentences why {t} matters for the next decade.",
    lambda t: "Classify this statement about "
    f"{t} as optimistic, neutral, or pessimistic: 'progress has been steady but uneven.'",
    lambda t: f"Compare the tradeoffs of investing in {t} versus a more established alternative.",
    lambda t: f"Categorize {t} as a technology, a place, or a concept.",
    # Tier 3: complex
    lambda t: "Given the constraints: limited budget, six-month timeline, and public skepticism, "
    f"recommend a strategy for advancing {t} and explain your reasoning.",
    lambda t: f"Analyze the long-term risks of {t} and explain your reasoning in detail.",
    lambda t: f"Walk through, step by step, how you would evaluate whether {t} is a sound long-term bet.",
    lambda t: f"Argue for and against prioritizing {t} in a balanced short essay.",
]


def generate_prompts(count: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    prompts = set()
    attempts = 0
    while len(prompts) < count and attempts < count * 20:
        topic = rng.choice(TOPICS)
        template = rng.choice(PROMPT_TEMPLATES)
        prompts.add(template(topic))
        attempts += 1
    return list(prompts)[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--verify", action="store_true", help="also run Phase 6 verification per request")
    args = parser.parse_args()

    prompts = generate_prompts(args.count)
    verb = "route+verify+log" if args.verify else "route+log"
    print(f"Load testing {len(prompts)} prompts through the full {verb} pipeline...\n")

    start = time.perf_counter()
    successes = 0
    for i, prompt in enumerate(prompts, start=1):
        result = route_request(prompt)
        verification = verify_and_escalate(result) if args.verify else None
        log_request(result, verification)
        successes += int(result.response.ok)
        print(f"[{i}/{len(prompts)}] tier={result.tier.value:7} model={result.model_config.key:25} ok={result.response.ok}")

    elapsed = time.perf_counter() - start
    summary = get_summary()

    print()
    print(
        f"Load test complete: {successes}/{len(prompts)} succeeded in {elapsed:.1f}s "
        f"({elapsed / len(prompts):.1f}s/request avg)"
    )
    print()
    print("=== Cumulative stats (all logged history in data/requests.db, not just this run) ===")
    print(f"Total requests logged: {summary['total_requests']}")
    print(f"Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"Baseline (Tier-3-for-everything) cost: ${summary['total_baseline_cost_usd']:.6f}")
    print(f"Savings: {summary['savings_pct']:.1f}% (${summary['savings_usd']:.6f})")
    print()
    print("Full breakdown available at: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
