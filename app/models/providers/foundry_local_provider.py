"""On-device LLMs on Snapdragon X PCs via Microsoft Foundry Local.

Foundry Local downloads the build of a model that matches the hardware —
on Snapdragon X that's the Hexagon NPU (QNN) variant — and serves it on an
OpenAI-compatible localhost endpoint. Nothing leaves the machine.

Setup on a Snapdragon PC:
    winget install Microsoft.FoundryLocal
    pip install foundry-local-sdk
    foundry model run phi-3.5-mini          # first run downloads the NPU build

Alternatively set FOUNDRY_LOCAL_ENDPOINT (e.g. http://localhost:5273/v1) and
optionally FOUNDRY_LOCAL_MODEL (the full model id) to skip the SDK.

`model_id` in the registry is the Foundry Local alias (e.g. phi-3.5-mini).
"""

import os

from .base import BaseProvider, ProviderError
from .openai_compatible import MAX_OUTPUT_TOKENS


def device_from_model_id(model_id: str) -> str:
    """Foundry Local model ids name their execution target, e.g. '...-qnn-npu'."""
    lowered = model_id.lower()
    if "npu" in lowered or "qnn" in lowered:
        return "Snapdragon NPU"
    if "gpu" in lowered or "cuda" in lowered:
        return "GPU"
    return "CPU"


class FoundryLocalProvider(BaseProvider):
    def __init__(self) -> None:
        self._clients: dict[str, tuple[object, str]] = {}  # alias -> (OpenAI client, resolved model id)

    def connect(self, alias: str) -> tuple[object, str]:
        if alias in self._clients:
            return self._clients[alias]
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("openai package is not installed. Run: pip install -r requirements.txt") from e

        endpoint = os.environ.get("FOUNDRY_LOCAL_ENDPOINT")
        try:
            if endpoint:
                client = OpenAI(base_url=endpoint, api_key="foundry-local")
                model_id = os.environ.get("FOUNDRY_LOCAL_MODEL", alias)
            else:
                from foundry_local import FoundryLocalManager

                manager = FoundryLocalManager(alias)  # starts the service, loads the model
                client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
                model_id = manager.get_model_info(alias).id
        except ImportError as e:
            raise ProviderError(
                "Foundry Local is not set up. On a Snapdragon PC: `winget install "
                "Microsoft.FoundryLocal` and `pip install foundry-local-sdk`, or set "
                "FOUNDRY_LOCAL_ENDPOINT."
            ) from e
        except Exception as e:
            raise ProviderError(f"Could not start Foundry Local model '{alias}': {e}") from e

        self._clients[alias] = (client, model_id)
        return client, model_id

    def resolved_device(self, alias: str) -> str | None:
        """Hardware the model runs on, once connected (None before the first call)."""
        entry = self._clients.get(alias)
        return device_from_model_id(entry[1]) if entry else None

    def call(self, prompt: str, model_id: str) -> tuple[str, int, int]:
        client, resolved_id = self.connect(model_id)
        try:
            response = client.chat.completions.create(
                model=resolved_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            if not response.choices:
                raise ValueError("response contained no choices")
            text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            raise ProviderError(f"Foundry Local call failed for model '{resolved_id}': {e}") from e
        return text, input_tokens, output_tokens
