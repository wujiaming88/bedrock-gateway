"""
Model registry for Bedrock Gateway.

Provides model resolution (alias → Bedrock model ID) and metadata
(context length, max output tokens) for all registered models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GatewayConfig, ModelEntry, _MODEL_ALIASES

# ---------------------------------------------------------------------------
# Valid Bedrock model ID prefixes — anything starting with one of these is
# considered a plausible Bedrock model ID and allowed to pass through.
# ---------------------------------------------------------------------------

_BEDROCK_ID_PREFIXES: tuple[str, ...] = (
    # Cross-region inference prefixes
    "us.",
    "eu.",
    "ap.",
    "me.",
    "sa.",
    "af.",
    # Vendor prefixes
    "anthropic.",
    "amazon.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
    "stability.",
)


def _looks_like_bedrock_id(model: str) -> bool:
    """Return True if *model* starts with a known Bedrock model ID prefix."""
    return model.startswith(_BEDROCK_ID_PREFIXES)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownModelError(Exception):
    """Raised when a model alias cannot be resolved to a valid Bedrock ID."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Unknown model: {model}. Use /v1/models to list available models."
        )


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
    are first checked against a built-in alias table, then passed through as-is
    (the Bedrock ID *is* the alias).
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._models: dict[str, ModelEntry] = dict(config.models)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, alias: str) -> str:
        """Return the Bedrock model ID for *alias*, or *alias* itself.

        Resolution order:
        1. Exact match in registered models
        2. Lookup in common alias table → re-resolve canonical name
        3. Validate format — if it looks like a Bedrock model ID, pass
           through; otherwise raise :exc:`UnknownModelError`.

        Raises
        ------
        UnknownModelError
            If *alias* is not registered, not a known alias, and does
            not match a valid Bedrock model ID format.
        """
        entry = self._models.get(alias)
        if entry:
            return entry.bedrock_id

        # Try common alias table
        canonical = _MODEL_ALIASES.get(alias)
        if canonical:
            entry = self._models.get(canonical)
            if entry:
                return entry.bedrock_id

        # Validate: only pass through if it looks like a real Bedrock ID
        if not _looks_like_bedrock_id(alias):
            raise UnknownModelError(alias)

        return alias

    def get_info(self, alias: str) -> ModelInfo | None:
        """Return full metadata for *alias*, or ``None`` if unknown."""
        entry = self._models.get(alias)
        if entry is None:
            canonical = _MODEL_ALIASES.get(alias)
            if canonical:
                entry = self._models.get(canonical)
                alias = canonical
        if entry is None:
            return None
        return ModelInfo(
            alias=alias,
            bedrock_id=entry.bedrock_id,
            context_length=entry.context_length,
            max_output=entry.max_output,
        )

    def get_max_output(self, alias: str, default: int = 64_000) -> int:
        """Return max output tokens for *alias*, with a fallback.

        Default lowered to 64K (safe for most Bedrock models) to avoid
        validation errors when an unknown model is passed through.
        """
        entry = self._models.get(alias)
        if entry:
            return entry.max_output
        canonical = _MODEL_ALIASES.get(alias)
        if canonical:
            entry = self._models.get(canonical)
            if entry:
                return entry.max_output
        return default

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
