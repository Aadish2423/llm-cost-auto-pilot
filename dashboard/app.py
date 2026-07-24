"""Phase 7 dashboard: cost savings, routing distribution, and escalation
trends read straight from data/requests.db. Nothing here computes
anything new — it's a read-only view over what Phase 7's logger already
wrote.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.logging.db import (  # noqa: E402
    get_recent_requests,
    get_routing_distribution,
    get_summary,
    get_tier_distribution,
)

st.set_page_config(page_title="LLM Cost Auto Pilot", layout="wide")
st.title("LLM Cost Auto Pilot — Cost Dashboard")

summary = get_summary()

if summary["total_requests"] == 0:
    st.info(
        "No logged requests yet. Run `python scripts/run_and_log.py \"your prompt\"` "
        "or `python scripts/seed_demo_data.py` first, then reload this page."
    )
    st.stop()

# --- The money shot ---
st.header(f"{summary['savings_pct']:.1f}% cost reduction vs. always using the top tier")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", summary["total_requests"])
col2.metric("Successful", f"{summary['successful_requests']} / {summary['total_requests']}")
col3.metric("Actual cost", f"${summary['total_cost_usd']:.6f}")
col4.metric(
    "Tier-3-for-everything baseline",
    f"${summary['total_baseline_cost_usd']:.6f}",
    delta=f"-${summary['savings_usd']:.6f} saved",
)
st.caption(
    "Baseline = what every successful request would have cost on whatever "
    "config/routing_config.yaml's Tier 3 currently points to, computed from "
    "the ACTUAL token counts of each real response — not an estimate."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Routing distribution")
    dist = get_routing_distribution()
    df_dist = pd.DataFrame(dist).set_index("model")
    st.bar_chart(df_dist)

with col2:
    st.subheader("Complexity tier distribution")
    tiers = get_tier_distribution()
    df_tiers = pd.DataFrame(tiers).set_index("tier")
    st.bar_chart(df_tiers)

st.divider()

st.subheader("Verification / escalation")
col1, col2, col3 = st.columns(3)
col1.metric("Verified requests", summary["verified_count"])
col2.metric("Escalations", summary["escalation_count"])
col3.metric("Escalation rate", f"{summary['escalation_rate_pct']:.1f}%")
if summary["verified_count"] == 0:
    st.caption(
        "No verification data yet — rows logged via seed_demo_data.py skip "
        "verification (see that script's docstring for why). Use "
        "run_and_log.py without --no-verify to populate this."
    )

st.divider()

st.subheader("Cost over time")
recent_all = get_recent_requests(limit=10_000)
df_all = pd.DataFrame([r.__dict__ for r in recent_all])
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
daily_cost = df_all.set_index("timestamp")["cost_usd"].resample("D").sum()
st.line_chart(daily_cost)

st.divider()

st.subheader("Recent requests")
recent = get_recent_requests(limit=50)
options = {
    f"#{r.id} [{r.tier}] {r.provider}:{r.model_id} — {r.prompt_text[:50]}": r for r in recent
}
selected_label = st.selectbox("Drill into a request", list(options.keys()))
selected = options[selected_label]

col1, col2 = st.columns(2)
with col1:
    st.write("**Prompt**")
    st.code(selected.prompt_text, language=None)
    st.write("**Response**" if selected.ok else "**Error**")
    st.code(selected.response_text if selected.ok else (selected.error or ""))
with col2:
    st.json(
        {
            "tier": selected.tier,
            "tier_score": selected.tier_score,
            "model": f"{selected.provider}:{selected.model_id}",
            "tokens": {"input": selected.input_tokens, "output": selected.output_tokens},
            "cost_usd": selected.cost_usd,
            "baseline_cost_usd": selected.baseline_cost_usd,
            "latency_ms": selected.latency_ms,
            "ok": selected.ok,
            "verification_status": selected.verification_status,
            "agreement_score": selected.agreement_score,
            "escalated": selected.escalated,
        }
    )

st.dataframe(
    pd.DataFrame([r.__dict__ for r in recent])[
        ["id", "timestamp", "tier", "provider", "model_id", "cost_usd", "latency_ms", "ok", "escalated"]
    ],
    width="stretch",
    hide_index=True,
)
