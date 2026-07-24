"""Phase 7 pipeline: route a prompt, verify it, and log the full result to
data/requests.db. This is what actually populates the database the
Streamlit dashboard reads from — routing and verification alone (Phase
2-6) never persist anything.

Usage:
    python scripts/run_and_log.py "Your prompt here"
    python scripts/run_and_log.py                     (demo prompt)
    python scripts/run_and_log.py --no-verify "..."    (skip the extra
                                                         verification call)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.logging.db import log_request  # noqa: E402
from app.router.routing_engine import route_request  # noqa: E402
from app.router.verifier import verify_and_escalate  # noqa: E402

DEMO_PROMPT = "What is the capital of France?"


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-verify"]
    skip_verify = "--no-verify" in sys.argv[1:]
    prompt = " ".join(args) or DEMO_PROMPT

    result = route_request(prompt)
    verification = None if skip_verify else verify_and_escalate(result)

    row_id = log_request(result, verification)

    print(f"Logged row #{row_id} to data/requests.db")
    print(f"  tier={result.tier.value}  model={result.model_config.key}  ok={result.response.ok}")
    if verification:
        print(f"  verification={verification.status.value}  escalated={verification.escalated}")


if __name__ == "__main__":
    main()
