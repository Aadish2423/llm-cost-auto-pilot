"""Route a realistic mix of prompts through the hybrid on-device / cloud
router, log every result, and print the decisions — populates the
dashboard with real, live-generated data.

Makes real calls: on-device prompts go to the local runtime (Ollama here,
Foundry Local on a Snapdragon PC), cloud prompts to whatever
routing_config.yaml maps their tier to. All personal data below is
synthetic.

Usage:
    python scripts/hybrid_demo.py
    python scripts/hybrid_demo.py --no-log
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from app.logging.db import get_summary, log_request  # noqa: E402
from app.router.hybrid import route_hybrid  # noqa: E402

DEMO_PROMPTS = [
    # simple → on-device
    "What is the capital of Japan? Answer in one sentence.",
    "Convert 2026-09-30 to DD/MM/YYYY format.",
    "Translate 'thank you very much' into Hindi.",
    "Reformat as a comma-separated list: mango, guava, papaya, banana",
    "Extract the amount and due date: 'Invoice #4471, Rs 12,500 due on 15 October 2026.'",
    # personal data → locked on-device
    "Fill this KYC update form as JSON with keys name, aadhaar, pan, new_address: Name Priya Sharma, "
    "Aadhaar 4995 1234 5670, PAN ABCPE1234F, new address 12 MG Road, Bengaluru 560001.",
    "Write a short email to HDFC Bank asking them to update my registered mobile number to "
    "+91 98765 43210 for my account number 50100234567890. Sign it as Priya Sharma.",
    "Write a two-line SMS reminding Rahul Verma (+91 98765 43210) about our meeting on Friday at 11am.",
    "Why does my code return 401 Unauthorized? I'm calling the API with key "
    "sk-proj-4f9aQ2mZx81LwT7vB0cR5yN3 in the Authorization header.",
    # medium / hard, no personal data → cloud
    "Summarize the following in 2 sentences and classify its sentiment: The quarterly results beat "
    "expectations as cloud subscriptions grew 32%, although hardware margins narrowed because of higher "
    "memory prices and the company lowered its full-year guidance.",
    "Classify this support ticket as billing, technical, or account and explain why: 'I was charged "
    "twice for my annual plan and the second charge has not been refunded.'",
    "Design a scalable event-driven architecture for a payments platform. Compare Kafka with RabbitMQ, "
    "explain the trade-offs of event sourcing versus CRUD, and justify each choice step by step.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        result = route_hybrid(prompt)
        status = "ok " if result.response.ok else "ERR"
        where = result.on_device_runtime or f"{result.model_config.provider} cloud"
        print(f"{i:2}. [{result.placement:9}] {status} {result.tier.value} → {result.model_config.key} "
              f"on {where}  ({result.response.latency_ms:,.0f} ms, ${result.response.cost_usd:.6f})")
        print(f"      why: {result.reason}" + (f" | fallback: {result.fallback}" if result.fallback else ""))
        if result.privacy_locked:
            print(f"      logged as: {result.privacy.redacted[:100]}")
        if not args.no_log:
            log_request(result)

    if not args.no_log:
        s = get_summary()
        print(f"\n{s['total_requests']} requests | {s['on_device_pct']:.0f}% on-device | "
              f"{s['privacy_locked_count']} privacy-locked | cost ${s['total_cost_usd']:.6f} vs "
              f"baseline ${s['total_baseline_cost_usd']:.6f} ({s['savings_pct']:.1f}% saved)")


if __name__ == "__main__":
    main()
