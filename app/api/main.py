"""Phase 10: FastAPI service wrapping the router as a real HTTP API. You
don't choose the model — POST /v1/completions routes it for you and
tells you which model got picked, why, and what it cost.

Run with (from the project root, so the app.* package resolves):
    uvicorn app.api.main:app --reload

Must be run as a module (uvicorn app.api.main:app), not as a script
(python app/api/main.py) — it uses package-relative imports like the
rest of app/, which only work when Python knows it's part of the `app`
package.
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from ..classifier.tiers import ComplexityTier  # noqa: E402
from ..logging.db import get_summary, log_request  # noqa: E402
from ..models.registry import get_model, is_model_available, load_registry  # noqa: E402
from ..router.routing_engine import ROUTING_CONFIG_PATH, route_request  # noqa: E402
from ..router.verifier import verify_and_escalate  # noqa: E402
from .schemas import (  # noqa: E402
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    RoutingConfigUpdate,
    StatsResponse,
)

app = FastAPI(
    title="LLM Cost Auto Pilot",
    description="Routes requests to the cheapest model that can handle them, and explains why.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict:
    return {"service": "LLM Cost Auto Pilot", "docs": "/docs"}


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(body: CompletionRequest) -> CompletionResponse:
    result = route_request(body.prompt)
    verification = verify_and_escalate(result) if body.verify else None

    row_id = log_request(result, verification) if body.log else None

    return CompletionResponse(
        text=verification.final_response_text if verification else result.response.text,
        ok=result.response.ok,
        error=result.response.error,
        tier=result.tier.value,
        tier_score=result.score,
        provider=result.model_config.provider,
        model_id=result.model_config.model_id,
        input_tokens=result.response.input_tokens,
        output_tokens=result.response.output_tokens,
        cost_usd=result.response.cost_usd,
        latency_ms=result.response.latency_ms,
        verification_status=verification.status.value if verification else None,
        escalated=bool(verification and verification.escalated),
        logged_row_id=row_id,
    )


@app.get("/v1/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    infos = []
    for model in load_registry():
        available, _ = is_model_available(model)
        infos.append(
            ModelInfo(
                provider=model.provider,
                model_id=model.model_id,
                quality_tier=model.quality_tier.value,
                cost_per_input_token=model.cost_per_input_token,
                cost_per_output_token=model.cost_per_output_token,
                avg_latency_ms=model.avg_latency_ms,
                local=model.local,
                available=available,
            )
        )
    return infos


@app.get("/v1/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    return StatsResponse(**get_summary())


@app.put("/v1/routing-config")
def update_routing_config(body: RoutingConfigUpdate) -> dict:
    try:
        tier = ComplexityTier(body.tier)
    except ValueError as e:
        valid = [t.value for t in ComplexityTier]
        raise HTTPException(status_code=400, detail=f"Invalid tier '{body.tier}'. Must be one of: {valid}") from e

    try:
        get_model(body.provider, body.model_id)  # raises if the model isn't in the registry
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # NOTE: this overwrites routing_config.yaml via yaml.dump, which does
    # NOT preserve the file's explanatory comments (PyYAML limitation) -
    # re-add them by hand if you want them back after calling this.
    with open(ROUTING_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["routing"][tier.value] = {"provider": body.provider, "model_id": body.model_id}
    with open(ROUTING_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    return {"tier": tier.value, "provider": body.provider, "model_id": body.model_id, "status": "updated"}
