"""The standardized shape every provider call returns, regardless of vendor."""

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str
    provider: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
