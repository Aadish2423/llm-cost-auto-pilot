import os

from .base import BaseProvider, ProviderError

MAX_OUTPUT_TOKENS = 1024


class AnthropicProvider(BaseProvider):
    """Coded and ready to go, but dormant until ANTHROPIC_API_KEY is set —
    you don't have this key yet, so this path is untested against the
    real API. The shape matches the other providers so it should just work
    once you add the key.
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                "(see .env.example) to enable this provider."
            )

        try:
            import anthropic
        except ImportError as e:
            raise ProviderError(
                "anthropic package is not installed. Run: pip install -r requirements.txt"
            ) from e

        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def call(self, prompt: str, model_id: str) -> tuple[str, int, int]:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise ProviderError(f"Anthropic call failed for model '{model_id}': {e}") from e

        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return text, input_tokens, output_tokens
