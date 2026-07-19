import os

from .base import BaseProvider, ProviderError


class OllamaProvider(BaseProvider):
    """Talks to a local Ollama daemon. Requires no API key, but requires
    Ollama to actually be installed and running (`ollama serve`) with the
    requested model pulled (`ollama pull <model_id>`).
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import ollama
        except ImportError as e:
            raise ProviderError(
                "ollama package is not installed. Run: pip install -r requirements.txt"
            ) from e

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)
        return self._client

    def call(self, prompt: str, model_id: str) -> tuple[str, int, int]:
        client = self._get_client()
        try:
            response = client.chat(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise ProviderError(
                f"Ollama call failed for model '{model_id}'. Is `ollama serve` "
                f"running and has `ollama pull {model_id}` been run? Original error: {e}"
            ) from e

        text = response["message"]["content"]
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        return text, input_tokens, output_tokens
