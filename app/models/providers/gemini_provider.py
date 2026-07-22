import os

from .base import BaseProvider, ProviderError

# Kept in sync with the cap the other providers use (see MAX_OUTPUT_TOKENS
# in anthropic_provider.py) so the Phase 1 "same 10 prompts to every model"
# benchmark is comparing apples to apples instead of whatever each
# provider's own default output length happens to be.
MAX_OUTPUT_TOKENS = 1024


class GeminiProvider(BaseProvider):
    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )

        try:
            from google import genai
        except ImportError as e:
            raise ProviderError(
                "google-genai is not installed. Run: pip install -r requirements.txt"
            ) from e

        self._client = genai.Client(api_key=api_key)
        return self._client

    def call(self, prompt: str, model_id: str) -> tuple[str, int, int]:
        client = self._get_client()

        # Response parsing lives inside this same try/except, not after it —
        # a safety-filtered or otherwise malformed response can raise from
        # `.text`/`.usage_metadata` just as easily as the network call can
        # fail, and both need to turn into a ProviderError instead of an
        # uncaught crash that would take down the whole batch run.
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
            )
            usage = response.usage_metadata
            input_tokens = (usage.prompt_token_count or 0) if usage else 0
            output_tokens = (usage.candidates_token_count or 0) if usage else 0
            text = response.text or ""
        except Exception as e:
            raise ProviderError(f"Gemini call failed for model '{model_id}': {e}") from e

        return text, input_tokens, output_tokens
