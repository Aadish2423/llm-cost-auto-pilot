"""Dev/demo utility: populate data/requests.db with real logged requests
so the Streamlit dashboard (and Phase 8's routing memory) have something
meaningful to show, without needing paid API keys.

Routes each prompt directly through Ollama (bypassing routing_config.yaml
entirely, since it doesn't route anything there yet). Deliberately does
NOT run verification: Tier 3's routing target is also Ollama's
llama3.2 right now, so "verifying" would compare the routed model
against itself and always short-circuit to already_top_tier - not a
real comparison. The verification mechanism itself is already proven
correct by Phase 6's deterministic tests; faking a second model here
just to manufacture demo variety would be less honest than just not
running it. Add a second real provider key and re-seed for genuine
escalation data in the dashboard.

Usage:
    python scripts/seed_demo_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.classifier.heuristic import classify  # noqa: E402
from app.logging.db import log_request  # noqa: E402
from app.models.dispatcher import send_request  # noqa: E402
from app.models.registry import get_model  # noqa: E402
from app.router.routing_engine import RoutingResult  # noqa: E402

SEED_PROMPTS = [
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

OLLAMA_MODEL = get_model("ollama", "llama3.2")


def main() -> None:
    print(f"Seeding {len(SEED_PROMPTS)} requests through Ollama (this will take a while - "
          f"each call can take several seconds)...")

    for i, prompt in enumerate(SEED_PROMPTS, start=1):
        classification = classify(prompt)
        response = send_request(prompt, OLLAMA_MODEL)
        result = RoutingResult(
            prompt=prompt,
            response=response,
            tier=classification.tier,
            score=classification.score,
            matched_signals=classification.matched_signals,
            model_config=OLLAMA_MODEL,
        )
        row_id = log_request(result, verification_result=None)
        print(f"[{i}/{len(SEED_PROMPTS)}] row #{row_id}  tier={result.tier.value}  ok={response.ok}")

    print("\nDone. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
