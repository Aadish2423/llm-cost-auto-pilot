"""Every provider adapter implements this interface. The dispatcher never
talks to an SDK directly — it only ever calls `.call(prompt, model_id)` on
whichever BaseProvider subclass matches the request's ModelConfig.provider.
"""

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Raised for anything that stops a provider call from producing a
    usable response: missing API key, network failure, unexpected SDK
    response shape, etc. The dispatcher catches this and turns it into an
    LLMResponse with `error` set, instead of letting the process crash.
    """


class BaseProvider(ABC):
    @abstractmethod
    def call(self, prompt: str, model_id: str) -> tuple[str, int, int]:
        """Send `prompt` to `model_id` and return (text, input_tokens, output_tokens).

        Token counts must come from the provider's own usage metadata, not
        an estimate — cost tracking downstream depends on them being exact.
        """
        raise NotImplementedError
