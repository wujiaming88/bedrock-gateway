"""
Model registry for Bedrock Gateway.

Provides model resolution (alias → Bedrock model ID) and metadata
(context length, max output tokens) for all registered models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import AzureResource, GatewayConfig, ModelEntry, _MODEL_ALIASES

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
    "openai.",
    "xai.",
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
        # Azure resources that opt into passthrough-by-model-prefix, indexed by
        # their ``prefix`` (e.g. "azure" → resource). Enables ``azure/<dep>``
        # model routing with no per-model config.
        self._prefix_resources: dict[str, AzureResource] = {
            res.prefix: res
            for res in config.azure_resources.values()
            if res.prefix
        }

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

    def get_entry(self, alias: str) -> ModelEntry | None:
        """Return the registered :class:`ModelEntry` for *alias*.

        Resolves through the common alias table, mirroring :meth:`resolve`.
        Returns ``None`` when *alias* is not a registered model (e.g. a raw
        Bedrock ID passed straight through) — callers then fall back to the
        default (runtime / anthropic) upstream dialect.
        """
        entry = self._models.get(alias)
        if entry is not None:
            return entry
        canonical = _MODEL_ALIASES.get(alias)
        if canonical:
            return self._models.get(canonical)
        return None

    def resolve_prefixed(self, alias: str, dialect: str) -> ModelEntry | None:
        """Resolve a ``<prefix>/<deployment>`` model into a passthrough entry.

        If *alias* begins with a configured Azure resource ``prefix`` followed
        by ``/``, build an Azure passthrough :class:`ModelEntry` on the fly:
        the deployment is the remainder after the prefix, the endpoint + key
        come from the matched resource, and *dialect* is supplied by the caller
        (the server picks it from the endpoint the request hit). Returns
        ``None`` when no prefix matches, so callers fall through to normal
        resolution.

        This is what lets ``azure/<any-deployment>`` route with **no per-model
        config** — the resource (endpoint + key + prefix) is registered once.
        """
        if "/" not in alias or not self._prefix_resources:
            return None
        prefix, _, deployment = alias.partition("/")
        res = self._prefix_resources.get(prefix)
        if res is None or not deployment:
            return None
        return ModelEntry(
            bedrock_id=deployment,
            transport="azure",
            dialect=dialect,
            deployment=deployment,
            azure_endpoint=res.base_url,
            azure_api_key=res.api_key,
        )

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
