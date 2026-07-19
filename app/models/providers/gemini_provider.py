import os

from .base import BaseProvider, ProviderError


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
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
        except Exception as e:
            raise ProviderError(f"Gemini call failed for model '{model_id}': {e}") from e

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        return response.text or "", input_tokens, output_tokens
