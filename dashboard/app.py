"""Autopilot Edge dashboard: hybrid on-device / cloud routing with an
on-device privacy guard, plus the Phase 7 cost analytics.

Run with:
    streamlit run dashboard/app.py

Tabs:
    Try it live       — type a prompt, watch the privacy scan, routing decision and answer
    Savings & privacy — what the logger recorded: cost saved, % on-device, privacy locks
    Snapdragon        — on-device runtimes, accelerators, privacy-guard latency
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from app.classifier.tiers import ComplexityTier  # noqa: E402
from app.edge.accelerators import system_info  # noqa: E402
from app.logging.db import (  # noqa: E402
    get_recent_requests,
    get_routing_distribution,
    get_summary,
    get_tier_distribution,
    log_request,
)
from app.models.registry import get_model  # noqa: E402
from app.privacy.guard import load_hybrid_config, scan  # noqa: E402
from app.router.hybrid import on_device_status, route_hybrid  # noqa: E402
from app.router.routing_engine import load_routing_map  # noqa: E402

st.set_page_config(page_title="Autopilot Edge", page_icon="🛡️", layout="wide")

EXAMPLES = {
    "✏️ Write your own": "",
    "Simple question → on-device": "What is the capital of Japan? Answer in one sentence.",
    "Aadhaar + PAN → locked on-device": (
        "Fill this KYC update form as JSON with keys name, aadhaar, pan, new_address: Name Priya Sharma, "
        "Aadhaar 4995 1234 5670, PAN ABCPE1234F, new address 12 MG Road, Bengaluru 560001."
    ),
    "Bank account + phone → locked on-device": (
        "Write a short email to HDFC Bank asking them to update my registered mobile number to "
        "+91 98765 43210 for my account number 50100234567890. Sign it as Priya Sharma."
    ),
    "Leaked API key → locked on-device": (
        "Why does my code return 401 Unauthorized? I'm calling the API with key "
        "sk-proj-4f9aQ2mZx81LwT7vB0cR5yN3 in the Authorization header."
    ),
    "Summarisation → cloud (medium)": (
        "Summarize the following in 2 sentences and classify its sentiment: The quarterly results "
        "beat expectations as cloud subscriptions grew 32%, although hardware margins narrowed "
        "because of higher memory prices and the company lowered its full-year guidance."
    ),
    "Hard reasoning → cloud (high)": (
        "Design a scalable event-driven architecture for a payments platform. Compare Kafka with "
        "RabbitMQ, explain the trade-offs of event sourcing versus CRUD, and justify each choice step by step."
    ),
}

PLACEMENT_BADGE = {
    "on_device": "💻 **ON-DEVICE**",
    "cloud": "☁️ **CLOUD**",
    "blocked": "⛔ **BLOCKED**",
}


@st.cache_resource(show_spinner="Loading the on-device privacy model…")
def warm_up() -> bool:
    # Load the NER model and run it once at a realistic length so the first demo scan is warm.
    scan("Warm-up: Priya Sharma from Infosys in Bengaluru asked about the invoice. " * 3)
    return True


def baseline_cost(result) -> float | None:
    """What this answer would have cost on the Tier-3 cloud model (actual token counts)."""
    if not result.response.ok:
        return None
    provider, model_id = load_routing_map()[ComplexityTier.TIER_3]
    return get_model(provider, model_id).estimate_cost(result.response.input_tokens, result.response.output_tokens)


def highlight(text: str, findings) -> str:
    """Markdown with each finding shown as a coloured tag."""
    colours = {"high": "red", "medium": "orange", "low": "blue"}
    out, cursor = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        if f.start < cursor:
            continue
        out.append(text[cursor:f.start].replace("$", "\\$"))
        out.append(f":{colours[f.severity]}-background[{f.label}]")
        cursor = f.end
    out.append(text[cursor:].replace("$", "\\$"))
    return "".join(out)


warm_up()
cfg = load_hybrid_config()

st.title("🛡️ Autopilot Edge")
st.caption("Hybrid AI router for Snapdragon PCs. Personal data stays on the laptop, simple prompts run "
           "free on-device, and only hard, non-sensitive prompts go to the cheapest capable cloud model.")

tab_live, tab_stats, tab_device = st.tabs(["💬 Try it live", "📊 Savings & privacy", "⚡ Snapdragon on-device"])


# ── Try it live ──────────────────────────────────────────────────────────────

with tab_live:
    choice = st.selectbox("Load an example", list(EXAMPLES), index=1)
    prompt = st.text_area("Prompt", EXAMPLES[choice], height=110, key=f"prompt::{choice}")
    col_a, col_b = st.columns([1, 4])
    run = col_a.button("Route it", type="primary", disabled=not prompt.strip())
    log_it = col_b.toggle("Log to dashboard", True)

    if run:
        report = scan(prompt)
        st.markdown("#### 1 · On-device privacy scan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", "SENSITIVE" if report.sensitive else "Clean")
        c2.metric("Scan time", f"{report.scan_ms:.0f} ms")
        c3.metric("Name detector runs on", report.ner_accelerator or "rules only")
        st.markdown(highlight(prompt, report.findings) if report.findings else prompt)
        if report.findings:
            st.caption("🔴 high: IDs, financial data, secrets · 🟠 medium: names, contact details · "
                       "🔵 low: organisations, places (context only). Prompts with 🔴 or 🟠 never leave the device.")

        with st.spinner("Routing and generating…"):
            t0 = time.perf_counter()
            result = route_hybrid(prompt)
            total_ms = (time.perf_counter() - t0) * 1000

        st.markdown("#### 2 · Routing decision")
        d1, d2, d3, d4 = st.columns(4)
        d1.markdown(f"Placement  \n{PLACEMENT_BADGE[result.placement]}")
        d2.markdown(f"Complexity  \n**{result.tier.value.replace('_', ' ').title()}** (score {result.score})")
        d3.markdown(f"Model  \n`{result.model_config.key}`")
        d4.markdown(f"Runs on  \n**{result.on_device_runtime or result.model_config.provider + ' cloud'}**")
        st.info(f"**Why:** {result.reason}")
        if result.fallback:
            st.warning(f"**Fallback:** {result.fallback}")

        st.markdown("#### 3 · Answer")
        if result.response.ok:
            st.write(result.response.text)
        else:
            st.error(result.response.error)

        base = baseline_cost(result)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost", f"${result.response.cost_usd:.6f}")
        m2.metric("Tier-3 cloud baseline", f"${base:.6f}" if base is not None else "–",
                  help="Same answer's token counts priced on the Tier-3 cloud model")
        m3.metric("Model latency", f"{result.response.latency_ms:,.0f} ms")
        m4.metric("Routing overhead", f"{result.decision_ms:,.0f} ms", help="Privacy scan + classification + runtime probe")

        if log_it:
            row = log_request(result)
            st.caption(f"Logged as request #{row}" + (" (stored redacted)" if result.privacy_locked else ""))


# ── Savings & privacy ────────────────────────────────────────────────────────

with tab_stats:
    summary = get_summary()
    if summary["total_requests"] == 0:
        st.info("No logged requests yet. Route a prompt in **Try it live**, or run "
                "`python scripts/hybrid_demo.py`.")
    else:
        st.header(f"{summary['savings_pct']:.1f}% cost reduction vs. sending everything to the top cloud tier")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", summary["total_requests"])
        c2.metric("Answered on-device", f"{summary['on_device_pct']:.0f}%",
                  f"{summary['on_device_count']} of {summary['total_requests']}", delta_color="off")
        c3.metric("Privacy-locked prompts", summary["privacy_locked_count"],
                  "never sent to the cloud", delta_color="off")
        c4.metric("Cost / baseline", f"${summary['total_cost_usd']:.5f}",
                  f"-${summary['savings_usd']:.5f} vs ${summary['total_baseline_cost_usd']:.5f}",
                  delta_color="inverse")  # lower cost = good = green
        st.caption("Baseline = every successful request priced on the Tier-3 cloud model using its actual "
                   "token counts. Rows logged before the hybrid router have their placement inferred "
                   "from the provider.")

        recent_all = get_recent_requests(limit=10_000)
        df_all = pd.DataFrame([r.__dict__ for r in recent_all])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Placement")
            st.bar_chart(df_all["placement"].value_counts())
        with col2:
            st.subheader("Model")
            st.bar_chart(pd.DataFrame(get_routing_distribution()).set_index("model"))
        with col3:
            st.subheader("Complexity tier")
            st.bar_chart(pd.DataFrame(get_tier_distribution()).set_index("tier"))

        st.subheader("Verification / escalation")
        v1, v2, v3 = st.columns(3)
        v1.metric("Verified requests", summary["verified_count"])
        v2.metric("Escalations", summary["escalation_count"])
        v3.metric("Escalation rate", f"{summary['escalation_rate_pct']:.1f}%")

        st.subheader("Cost over time")
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        st.line_chart(df_all.set_index("timestamp")["cost_usd"].resample("D").sum())

        st.subheader("Recent requests")
        recent = get_recent_requests(limit=50)
        options = {f"#{r.id} [{r.placement}] {r.provider}:{r.model_id} — {r.prompt_text[:50]}": r for r in recent}
        selected = options[st.selectbox("Drill into a request", list(options))]
        left, right = st.columns(2)
        with left:
            st.write("**Prompt**" + (" (redacted)" if selected.redacted else ""))
            st.code(selected.prompt_text, language=None)
            st.write("**Response**" if selected.ok else "**Error**")
            st.code(selected.response_text if selected.ok else (selected.error or ""), language=None)
        with right:
            st.json({
                "placement": selected.placement,
                "routing_reason": selected.routing_reason,
                "privacy_locked": selected.privacy_locked,
                "privacy_categories": selected.privacy_categories,
                "on_device_runtime": selected.on_device_runtime,
                "fallback": selected.fallback,
                "tier": selected.tier,
                "model": f"{selected.provider}:{selected.model_id}",
                "tokens": {"input": selected.input_tokens, "output": selected.output_tokens},
                "cost_usd": selected.cost_usd,
                "baseline_cost_usd": selected.baseline_cost_usd,
                "latency_ms": selected.latency_ms,
                "verification_status": selected.verification_status,
            })
        st.dataframe(
            pd.DataFrame([r.__dict__ for r in recent])[
                ["id", "timestamp", "placement", "privacy_locked", "tier", "provider", "model_id",
                 "cost_usd", "latency_ms", "ok"]
            ],
            width="stretch", hide_index=True,
        )


# ── Snapdragon on-device ─────────────────────────────────────────────────────

with tab_device:
    info = system_info()
    st.subheader("This machine")
    s1, s2, s3 = st.columns(3)
    s1.metric("Processor", info["machine"])
    s2.metric("Hexagon NPU (QNN)", "available" if info["npu_available"] else "not detected")
    s3.metric("ONNX Runtime", info["onnxruntime"] or "not installed")
    st.caption(f"{info['processor']} · execution providers: {', '.join(info['available_providers']) or '–'}")

    st.subheader("On-device language models (first reachable wins)")
    st.dataframe(pd.DataFrame(on_device_status()), width="stretch", hide_index=True)

    st.subheader("Privacy guard latency")
    sample = EXAMPLES["Aadhaar + PAN → locked on-device"]
    if st.button("Benchmark the privacy guard (50 scans)"):
        times = [scan(sample).scan_ms for _ in range(50)]
        b1, b2, b3 = st.columns(3)
        b1.metric("Median", f"{pd.Series(times).median():.1f} ms")
        b2.metric("p95", f"{pd.Series(times).quantile(0.95):.1f} ms")
        b3.metric("Runs on", scan(sample).ner_accelerator or "rules only")

    with st.expander("Moving everything onto the Snapdragon NPU"):
        st.markdown(
            "- **Privacy model (BERT NER):** `pip install onnxruntime-qnn` on a Snapdragon X PC. "
            "`app/edge/accelerators.py` then selects the QNN execution provider (Hexagon NPU) automatically.\n"
            "- **Language model:** `winget install Microsoft.FoundryLocal`, `pip install foundry-local-sdk`, "
            "`foundry model run phi-3.5-mini`. Foundry Local serves the NPU build and the router prefers it "
            "over Ollama (`config/hybrid_config.yaml`).\n"
            "- Nothing else changes: same code, same config, same dashboard."
        )
    st.caption(f"Privacy lock threshold: **{cfg['privacy']['lock_on_severity']}** · logs redacted: "
               f"**{cfg['privacy']['redact_logs']}** · cloud fallback: **{cfg['on_device']['fallback_to_cloud']}**")
