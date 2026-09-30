"""Hybrid on-device / cloud routing — the Snapdragon edition of route_request().

    from app.router.hybrid import route_hybrid
    result = route_hybrid("My PAN is ABCPE1234F, draft a tax query")
    result.placement   # "on_device"
    result.reason      # "privacy lock: contains PAN"

Decision order:
  1. Privacy guard (on-device rules + BERT NER). Sensitive → on-device only.
     If no on-device model is reachable the request is blocked rather than
     sent to the cloud (fail closed).
  2. Tier fit. If the on-device model meets the tier's quality bar
     (cost_predictor.MIN_QUALITY_FOR_TIER) → on-device: $0, private, offline.
  3. Otherwise → the cloud model routing_config.yaml maps the tier to.

Fallbacks (config/hybrid_config.yaml):
  * on-device call fails, prompt not sensitive → cloud
  * cloud call fails (outage, 429, offline)   → answer on-device
"""

import os
import time
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import urlopen

from ..classifier.heuristic import classify
from ..models.dispatcher import get_provider, send_request
from ..models.registry import QUALITY_RANK, ModelConfig, get_model
from ..models.response import LLMResponse
from ..privacy.guard import PrivacyReport, load_hybrid_config, scan
from .cost_predictor import MIN_QUALITY_FOR_TIER
from .routing_engine import RoutingResult, load_routing_map

PROBE_TTL_S = 30.0
_probe_cache: dict[str, tuple[float, bool, str]] = {}


@dataclass
class HybridRoutingResult(RoutingResult):
    placement: str = "cloud"  # "on_device" | "cloud" | "blocked"
    reason: str = ""
    privacy: PrivacyReport | None = None
    privacy_locked: bool = False
    on_device_runtime: str | None = None  # e.g. "Snapdragon NPU (Foundry Local)"
    fallback: str | None = None
    decision_ms: float = 0.0
    attempts: list[str] = field(default_factory=list)


# ── on-device runtime discovery ──────────────────────────────────────────────

def _http_ok(url: str, timeout: float = 3.0) -> bool:
    # Windows resolves "localhost" to ::1 first; a refused IPv6 connect can eat
    # the whole timeout before IPv4 is tried, so probe 127.0.0.1 directly.
    url = url.replace("://localhost", "://127.0.0.1")
    try:
        with urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (URLError, OSError, ValueError):
        return False


def _probe(model: ModelConfig) -> tuple[bool, str]:
    """Is this local runtime reachable right now? Returns (ok, runtime label)."""
    if model.provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        return _http_ok(f"{host}/api/tags"), "CPU/GPU (Ollama)"

    if model.provider == "foundry_local":
        endpoint = os.environ.get("FOUNDRY_LOCAL_ENDPOINT")
        if endpoint:
            return _http_ok(f"{endpoint.rstrip('/')}/models"), "Snapdragon NPU (Foundry Local)"
        try:
            from foundry_local import FoundryLocalManager

            manager = FoundryLocalManager(bootstrap=False)
            cached = {m.alias for m in manager.list_cached_models()}
            ok = manager.is_service_running() and model.model_id in cached
            device = get_provider("foundry_local").resolved_device(model.model_id) or "Snapdragon NPU"
            return ok, f"{device} (Foundry Local)"
        except Exception:
            return False, "Foundry Local (not installed)"

    return model.local, model.provider


def probe(model: ModelConfig) -> tuple[bool, str]:
    now = time.monotonic()
    cached = _probe_cache.get(model.key)
    if cached and now - cached[0] < PROBE_TTL_S:
        return cached[1], cached[2]
    ok, label = _probe(model)
    _probe_cache[model.key] = (now, ok, label)
    return ok, label


def pick_on_device_model(config: dict | None = None) -> tuple[ModelConfig, str] | None:
    """First reachable runtime from hybrid_config.yaml's on_device.preference."""
    cfg = config or load_hybrid_config()
    for entry in cfg["on_device"]["preference"]:
        try:
            model = get_model(entry["provider"], entry["model_id"])
        except ValueError:
            continue
        ok, label = probe(model)
        if ok:
            return model, label
    return None


def on_device_status(config: dict | None = None) -> list[dict]:
    """Every configured on-device runtime and whether it's reachable (for the dashboard/API)."""
    cfg = config or load_hybrid_config()
    rows = []
    for entry in cfg["on_device"]["preference"]:
        try:
            model = get_model(entry["provider"], entry["model_id"])
        except ValueError:
            rows.append({"model": f"{entry['provider']}:{entry['model_id']}", "available": False,
                         "runtime": "not in model_registry.yaml"})
            continue
        ok, label = probe(model)
        rows.append({"model": model.key, "available": ok, "runtime": label})
    return rows


# ── routing ──────────────────────────────────────────────────────────────────

def _blocked_response(model: ModelConfig, reason: str) -> LLMResponse:
    return LLMResponse(
        text="", input_tokens=0, output_tokens=0, latency_ms=0.0, cost_usd=0.0,
        model_id=model.model_id, provider=model.provider, error=reason,
    )


def route_hybrid(prompt: str, config: dict | None = None) -> HybridRoutingResult:
    cfg = config or load_hybrid_config()
    t0 = time.perf_counter()

    privacy = scan(prompt, cfg)
    classification = classify(prompt)
    tier = classification.tier
    local = pick_on_device_model(cfg)
    cloud_provider, cloud_model_id = load_routing_map()[tier]
    cloud = get_model(cloud_provider, cloud_model_id)
    decision_ms = (time.perf_counter() - t0) * 1000

    local_fits = local is not None and QUALITY_RANK[local[0].quality_tier] >= QUALITY_RANK[MIN_QUALITY_FOR_TIER[tier]]

    if privacy.sensitive:
        placement, reason = "on_device", f"privacy lock: {privacy.reason}"
    elif local_fits:
        placement = "on_device"
        reason = f"{tier.value}: on-device {local[0].model_id} meets the {MIN_QUALITY_FOR_TIER[tier].value} quality bar"
    elif cloud.local:
        placement, reason = "cloud", f"{tier.value}: routing_config maps this tier to {cloud.key}"
    else:
        need = MIN_QUALITY_FOR_TIER[tier].value
        reason = (f"{tier.value}: needs {need} quality, sent to cloud {cloud.key}" if local
                  else f"{tier.value}: no on-device runtime reachable, sent to cloud {cloud.key}")
        placement = "cloud"

    result_kwargs = dict(
        prompt=prompt, tier=tier, score=classification.score,
        matched_signals=classification.matched_signals, privacy=privacy,
        privacy_locked=privacy.sensitive, decision_ms=decision_ms,
    )

    # Sensitive but nowhere safe to run it: fail closed.
    if privacy.sensitive and local is None:
        first = cfg["on_device"]["preference"][0]
        model = get_model(first["provider"], first["model_id"])
        message = (f"Blocked: prompt {privacy.reason} and no on-device model is reachable, "
                   f"so it was not sent to the cloud.")
        return HybridRoutingResult(response=_blocked_response(model, message), model_config=model,
                                   placement="blocked", reason=message, **result_kwargs)

    if placement == "on_device":
        model, runtime = local
        attempts = [f"on-device {model.key}"]
        response = send_request(prompt, model)
        fallback = None
        if not response.ok and not privacy.sensitive and cfg["on_device"].get("fallback_to_cloud", True):
            attempts.append(f"cloud {cloud.key}")
            fallback = f"on-device failed ({response.error[:80]}), answered by cloud"
            model, runtime, placement, response = cloud, None, "cloud", send_request(prompt, cloud)
        return HybridRoutingResult(response=response, model_config=model, placement=placement,
                                   reason=reason, on_device_runtime=runtime, fallback=fallback,
                                   attempts=attempts, **result_kwargs)

    attempts = [f"cloud {cloud.key}"]
    response = send_request(prompt, cloud)
    if not response.ok and local and cfg["on_device"].get("answer_locally_on_cloud_failure", True):
        attempts.append(f"on-device {local[0].key}")
        local_response = send_request(prompt, local[0])
        if local_response.ok:
            return HybridRoutingResult(
                response=local_response, model_config=local[0], placement="on_device", reason=reason,
                on_device_runtime=local[1], attempts=attempts,
                fallback=f"cloud failed ({response.error[:80]}), answered on-device", **result_kwargs)
    return HybridRoutingResult(response=response, model_config=cloud, placement="cloud", reason=reason,
                               attempts=attempts, **result_kwargs)
