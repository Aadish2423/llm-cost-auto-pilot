"""Phase 9b: on-demand live benchmarking. Tests every model that
actually has a working key with a small set of prompts and records REAL
latency and success rate.

Deliberately a snapshot, not a scheduled/automatic refresh — there's no
scheduler/cron infrastructure in this project, so "live" here means "run
this command whenever you want fresh numbers," not "runs every few
hours" like the original design describes. Doesn't rewrite
config/model_registry.yaml automatically either; that's the same
"predicts and surfaces, doesn't auto-rewrite" pattern as Phase 3/5 — you
decide whether measured drift is worth a manual config update.
"""

from dataclasses import dataclass, field

from ..models.dispatcher import send_request
from ..models.registry import ModelConfig, is_model_available, load_registry

BENCHMARK_PROMPTS = [
    "What is the capital of Japan?",
    "Summarize in one sentence: the sky is blue due to Rayleigh scattering of sunlight.",
]


@dataclass
class BenchmarkResult:
    model_config: ModelConfig
    skipped_reason: str | None
    attempted: int
    succeeded: int
    avg_latency_ms: float | None
    registry_avg_latency_ms: float
    latency_drift_ms: float | None  # measured minus the registry's static assumption
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float | None:
        return (self.succeeded / self.attempted) if self.attempted else None


def benchmark_model(model: ModelConfig, prompts: list[str] = BENCHMARK_PROMPTS) -> BenchmarkResult:
    available, reason = is_model_available(model)
    if not available:
        return BenchmarkResult(
            model_config=model,
            skipped_reason=reason,
            attempted=0,
            succeeded=0,
            avg_latency_ms=None,
            registry_avg_latency_ms=model.avg_latency_ms,
            latency_drift_ms=None,
        )

    latencies = []
    errors = []
    for prompt in prompts:
        response = send_request(prompt, model)
        if response.ok:
            latencies.append(response.latency_ms)
        else:
            errors.append(response.error or "unknown error")

    avg_latency = sum(latencies) / len(latencies) if latencies else None
    drift = (avg_latency - model.avg_latency_ms) if avg_latency is not None else None

    return BenchmarkResult(
        model_config=model,
        skipped_reason=None,
        attempted=len(prompts),
        succeeded=len(latencies),
        avg_latency_ms=avg_latency,
        registry_avg_latency_ms=model.avg_latency_ms,
        latency_drift_ms=drift,
        errors=errors,
    )


def benchmark_available_models(
    registry: list[ModelConfig] | None = None, prompts: list[str] = BENCHMARK_PROMPTS
) -> list[BenchmarkResult]:
    models = registry if registry is not None else load_registry()
    return [benchmark_model(model, prompts) for model in models]
