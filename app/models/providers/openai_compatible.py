"""A lot of providers (OpenAI itself, Groq, Together AI, and others) expose
the exact same request/response shape as the OpenAI Chat Completions API —
only the base URL and the API key differ. Rather than duplicating near-
identical adapter code per vendor, one parameterized class covers all of
them.

If you add another OpenAI-compatible provider later, this is usually the
only file you need — just instantiate it with a new env var + base_url in
dispatcher.py, no new class required.
"""

import os

from .base import BaseProvider, ProviderError

# Kept in sync with the other providers' output cap (see MAX_OUTPUT_TOKENS
# in anthropic_provider.py) so the Phase 1 benchmark is comparable across
# providers instead of each one running to its own default length.
MAX_OUTPUT_TOKENS = 1024


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, env_var: str, base_url: str | None, display_name: str) -> None:
        self._env_var = env_var
        self._base_url = base_url
        self._display_name = display_name
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._env_var)
        if not api_key:
            raise ProviderError(
                f"{self._env_var} is not set. Add it to your .env file "
                f"(see .env.example) to enable {self._display_name}."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError(
                "openai package is not installed. Run: pip install -r requirements.txt"
            ) from e

        self._client = OpenAI(api_key=api_key, base_url=self._base_url)
        return self._client

    def call(self, prompt: str, model_id: str) -> tuple[str, int, int]:
        client = self._get_client()
        # Response parsing stays inside this try/except (not after it) — an
        # empty `choices` list (e.g. a content-filter block) or a missing
        # `usage` block should become a ProviderError, not an uncaught
        # IndexError/AttributeError that crashes the whole batch run.
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            if not response.choices:
                raise ValueError("response contained no choices (likely content-filtered)")
            text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            raise ProviderError(
                f"{self._display_name} call failed for model '{model_id}': {e}"
            ) from e

        return text, input_tokens, output_tokens
