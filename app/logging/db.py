"""Phase 7: the audit trail. Every routed (and optionally verified)
request gets one row in SQLite — this is what the Streamlit dashboard
reads from, and what Phase 8's routing memory queries for history.

Deliberate deviation from the original spec: it lists "prompt hash" as
the logged field (likely for privacy at production scale). This is a
personal portfolio project with no real user data at stake, and the
dashboard's planned "explainability drill-down" needs to show actual
past requests — so both prompt_hash (kept, matches the spec) and
prompt_text (added, for practical debugging/demo value) are stored.
"""

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..classifier.tiers import ComplexityTier
from ..models.registry import get_model
from ..router.routing_engine import RoutingResult, load_routing_map
from ..router.verifier import VerificationResult

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "requests.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    tier TEXT NOT NULL,
    tier_score INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    baseline_provider TEXT,
    baseline_model_id TEXT,
    baseline_cost_usd REAL,
    verification_status TEXT,
    agreement_score REAL,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalation_cost_usd REAL NOT NULL DEFAULT 0.0
);
"""


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def _baseline_cost(routing_result: RoutingResult) -> tuple[str | None, str | None, float | None]:
    """What this request would have cost on Tier 3's model, using the
    ACTUAL token counts from the real response — this is the number the
    dashboard's headline "money saved" metric is built from. None if the
    routing config can't be read (shouldn't happen in practice; Phase 2
    already validates it at startup) or the call itself failed (nothing
    was actually generated to compare against).
    """
    response = routing_result.response
    if not response.ok:
        return None, None, None
    try:
        provider, model_id = load_routing_map()[ComplexityTier.TIER_3]
        baseline_model = get_model(provider, model_id)
    except Exception:
        # Belt-and-suspenders, same pattern as dispatcher.py's catch-all:
        # route_request() already read this same config successfully to
        # get this far, so a failure here should be near-impossible in
        # practice — but logging a request should never itself crash
        # because of a config-reading hiccup, so degrade to "no baseline"
        # instead of raising.
        return None, None, None
    cost = baseline_model.estimate_cost(response.input_tokens, response.output_tokens)
    return provider, model_id, cost


@dataclass
class LoggedRequest:
    id: int
    timestamp: str
    prompt_hash: str
    prompt_text: str
    response_text: str
    tier: str
    tier_score: int
    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    ok: bool
    error: str | None
    baseline_cost_usd: float | None
    verification_status: str | None
    agreement_score: float | None
    escalated: bool
    escalation_cost_usd: float


def log_request(
    routing_result: RoutingResult,
    verification_result: VerificationResult | None = None,
    path: Path = DB_PATH,
) -> int:
    init_db(path)
    response = routing_result.response
    baseline_provider, baseline_model_id, baseline_cost = _baseline_cost(routing_result)

    verification_status = verification_result.status.value if verification_result else None
    agreement_score = verification_result.agreement_score if verification_result else None
    escalated = bool(verification_result and verification_result.escalated)
    escalation_cost = verification_result.cost_delta_usd if verification_result else 0.0

    with closing(sqlite3.connect(path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt_hash, prompt_text, response_text, tier, tier_score,
                provider, model_id, input_tokens, output_tokens, cost_usd,
                latency_ms, ok, error, baseline_provider, baseline_model_id,
                baseline_cost_usd, verification_status, agreement_score,
                escalated, escalation_cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                hashlib.sha256(routing_result.prompt.encode("utf-8")).hexdigest(),
                routing_result.prompt,
                response.text,
                routing_result.tier.value,
                routing_result.score,
                routing_result.model_config.provider,
                routing_result.model_config.model_id,
                response.input_tokens,
                response.output_tokens,
                response.cost_usd,
                response.latency_ms,
                int(response.ok),
                response.error,
                baseline_provider,
                baseline_model_id,
                baseline_cost,
                verification_status,
                agreement_score,
                int(escalated),
                escalation_cost,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_summary(path: Path = DB_PATH) -> dict:
    init_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(ok), 0) AS successful_requests,
                COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                COALESCE(SUM(baseline_cost_usd), 0.0) AS total_baseline_cost_usd,
                COALESCE(SUM(escalation_cost_usd), 0.0) AS total_verification_cost_usd,
                COALESCE(SUM(escalated), 0) AS escalation_count,
                COALESCE(SUM(CASE WHEN verification_status IS NOT NULL THEN 1 ELSE 0 END), 0) AS verified_count
            FROM requests
            """
        ).fetchone()

    total_requests = row["total_requests"]
    total_cost = row["total_cost_usd"]
    total_baseline = row["total_baseline_cost_usd"]
    savings_usd = total_baseline - total_cost
    savings_pct = (savings_usd / total_baseline * 100) if total_baseline > 0 else 0.0
    escalation_rate = (row["escalation_count"] / row["verified_count"] * 100) if row["verified_count"] else 0.0

    return {
        "total_requests": total_requests,
        "successful_requests": row["successful_requests"],
        "total_cost_usd": total_cost,
        "total_baseline_cost_usd": total_baseline,
        "savings_usd": savings_usd,
        "savings_pct": savings_pct,
        "total_verification_cost_usd": row["total_verification_cost_usd"],
        "verified_count": row["verified_count"],
        "escalation_count": row["escalation_count"],
        "escalation_rate_pct": escalation_rate,
    }


def get_routing_distribution(path: Path = DB_PATH) -> list[dict]:
    init_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT provider || ':' || model_id AS model, COUNT(*) AS count
            FROM requests
            GROUP BY model
            ORDER BY count DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_tier_distribution(path: Path = DB_PATH) -> list[dict]:
    init_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT tier, COUNT(*) AS count FROM requests GROUP BY tier ORDER BY tier"
        ).fetchall()
    return [dict(row) for row in rows]


def get_recent_requests(limit: int = 50, path: Path = DB_PATH) -> list[LoggedRequest]:
    init_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    return [
        LoggedRequest(
            id=row["id"],
            timestamp=row["timestamp"],
            prompt_hash=row["prompt_hash"],
            prompt_text=row["prompt_text"],
            response_text=row["response_text"],
            tier=row["tier"],
            tier_score=row["tier_score"],
            provider=row["provider"],
            model_id=row["model_id"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=row["cost_usd"],
            latency_ms=row["latency_ms"],
            ok=bool(row["ok"]),
            error=row["error"],
            baseline_cost_usd=row["baseline_cost_usd"],
            verification_status=row["verification_status"],
            agreement_score=row["agreement_score"],
            escalated=bool(row["escalated"]),
            escalation_cost_usd=row["escalation_cost_usd"],
        )
        for row in rows
    ]
