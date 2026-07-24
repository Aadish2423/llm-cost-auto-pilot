"""Phase 8b demo: retrieve the k most similar past logged requests to a
new prompt, along with their verified outcomes, as a second signal
alongside the Phase 2 classifier. Reads data/requests.db - run
scripts/seed_demo_data.py or scripts/run_and_log.py first if it's empty.

Usage:
    python scripts/routing_memory_demo.py "your prompt"
    python scripts/routing_memory_demo.py                (demo prompt)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.routing_memory import get_routing_memory_signal  # noqa: E402

DEMO_PROMPT = "Classify this ticket: 'My payment failed twice this week.'"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or DEMO_PROMPT
    signal = get_routing_memory_signal(prompt)

    print(f'Query: "{prompt}"')
    print(f"History available: {signal.history_size} logged requests")
    print()

    if not signal.similar:
        print("No similar past requests found (empty history, or a degenerate corpus).")
        print("Run scripts/seed_demo_data.py first if data/requests.db is empty.")
        return

    print(f"Most common tier among similar past requests: {signal.most_common_tier_among_similar}")
    if signal.escalation_rate_among_similar is not None:
        print(f"Escalation rate among similar VERIFIED past requests: {signal.escalation_rate_among_similar:.0%}")
    else:
        print("Escalation rate among similar past requests: no verified history to compute from")
    print()

    print("=== Most similar past requests ===")
    for s in signal.similar:
        row = s.logged_request
        print(f"[similarity={s.similarity:.2f}] tier={row.tier} model={row.provider}:{row.model_id}")
        print(f"    prompt: {row.prompt_text[:80]}")


if __name__ == "__main__":
    main()
