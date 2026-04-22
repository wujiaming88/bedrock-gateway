"""
Model registry for Bedrock Gateway.

Provides model resolution (alias → Bedrock model ID) and metadata
(context length, max output tokens) for all registered models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GatewayConfig, ModelEntry


@dataclass
class ModelInfo:
    """Resolved model information."""
    alias: str
    bedrock_id: str
    context_length: int
    max_output: int


class ModelRegistry:
    """
    Thread-safe registry mapping user-facing model aliases to Bedrock model IDs.

    Models are loaded from :class:`GatewayConfig` at startup.  Unknown aliases
    are passed through as-is (the Bedrock ID *is* the alias).
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._models: dict[str, ModelEntry] = dict(config.models)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, alias: str) -> str:
        """Return the Bedrock model ID for *alias*, or *alias* itself."""
        entry = self._models.get(alias)
        return entry.bedrock_id if entry else alias

    def get_info(self, alias: str) -> ModelInfo | None:
        """Return full metadata for *alias*, or ``None`` if unknown."""
        entry = self._models.get(alias)
        if entry is None:
            return None
        return ModelInfo(
            alias=alias,
            bedrock_id=entry.bedrock_id,
            context_length=entry.context_length,
            max_output=entry.max_output,
        )

    def get_max_output(self, alias: str, default: int = 128_000) -> int:
        """Return max output tokens for *alias*, with a fallback."""
        entry = self._models.get(alias)
        return entry.max_output if entry else default

    def list_models(self) -> list[dict]:
        """Return an OpenAI-compatible model list."""
        return [
            {
                "id": alias,
                "object": "model",
                "owned_by": "bedrock",
                "context_length": entry.context_length,
                "max_output_tokens": entry.max_output,
            }
            for alias, entry in self._models.items()
        ]
