"""Phase 6 demo: route a prompt, then verify it against the strongest
configured model and auto-escalate if agreement is too low. Makes real
provider calls (one for the route, one for the reference check unless
already routed to the top tier), so it hits live APIs or fails cleanly
depending on which keys are in .env.

Usage:
    python scripts/verify_routing.py "Your prompt here"
    python scripts/verify_routing.py                     (demo prompt)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.router.routing_engine import route_request  # noqa: E402
from app.router.verifier import VerificationStatus, verify_and_escalate  # noqa: E402

DEMO_PROMPT = "What is the capital of France?"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or DEMO_PROMPT
    shown_prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
    print(f'Prompt: "{shown_prompt}"')
    print()

    result = route_request(prompt)
    print(f"Routed to: {result.model_config.key}  (tier: {result.tier.value})")
    if result.response.ok:
        print(f"Response: {result.response.text[:150]}")
    else:
        print(f"Response error: {result.response.error}")
    print()

    verification = verify_and_escalate(result)
    print(f"=== Verification: {verification.status.value} ===")

    if verification.status == VerificationStatus.ALREADY_TOP_TIER:
        print("Routed model IS the reference model - nothing stronger to check against.")
    elif verification.status == VerificationStatus.COMPARISON_FAILED:
        print(f"Reference model: {verification.reference_model}")
        print("Could not compare - the original and/or reference call failed outright.")
    else:
        print(f"Reference model: {verification.reference_model}")
        print(f"Agreement score: {verification.agreement_score:.2f}  (threshold: 0.50)")
        print(f"Reference response: {verification.reference_text[:150]}")
        print(f"Extra cost from verification call: ${verification.cost_delta_usd:.6f}")

    print()
    if verification.escalated:
        print(f"ESCALATED -> final response now comes from {verification.final_model}")
    else:
        print(f"No escalation -> final response stays from {verification.final_model}")
    print(f"Final text: {verification.final_response_text[:150]}")


if __name__ == "__main__":
    main()
