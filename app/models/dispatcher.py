"""The single entry point the rest of the app uses to talk to any model:

    from app.models.dispatcher import send_request
    response = send_request("Summarize this...", model_config)

Callers never import a provider SDK directly. This is what makes it
possible to swap/add providers later without touching routing logic.
"""

import time

from .providers.anthropic_provider import AnthropicProvider
from .providers.base import ProviderError
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from .registry import ModelConfig
from .response import LLMResponse

_PROVIDERS = {
    "gemini": GeminiProvider(),
    "ollama": OllamaProvider(),
    "openai": OpenAICompatibleProvider("OPENAI_API_KEY", None, "OpenAI"),
    "anthropic": AnthropicProvider(),
    "groq": OpenAICompatibleProvider(
        "GROQ_API_KEY", "https://api.groq.com/openai/v1", "Groq"
    ),
    "together": OpenAICompatibleProvider(
        "TOGETHER_API_KEY", "https://api.together.xyz/v1", "Together AI"
    ),
}


def send_request(prompt: str, model_config: ModelConfig) -> LLMResponse:
    provider = _PROVIDERS.get(model_config.provider)
    if provider is None:
        return LLMResponse(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
            model_id=model_config.model_id,
            provider=model_config.provider,
            error=f"Unknown provider '{model_config.provider}'",
        )

    start = time.perf_counter()
    try:
        text, input_tokens, output_tokens = provider.call(prompt, model_config.model_id)
        error = None
    except ProviderError as e:
        text, input_tokens, output_tokens = "", 0, 0
        error = str(e)
    latency_ms = (time.perf_counter() - start) * 1000

    cost_usd = model_config.estimate_cost(input_tokens, output_tokens)

    return LLMResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_id=model_config.model_id,
        provider=model_config.provider,
        error=error,
    )
