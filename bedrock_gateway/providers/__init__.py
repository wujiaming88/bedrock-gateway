"""
Provider registry.

Maps a :class:`~bedrock_gateway.config.ModelEntry`'s ``protocol`` to the
stateless :class:`Provider` singleton that knows how to talk to that upstream.
Adding a new upstream dialect = add a Provider subclass + register it here.
"""

from __future__ import annotations

from ..config import ModelEntry
from .anthropic_bedrock import AnthropicBedrockProvider
from .base import Provider
from .openai_responses import OpenAIResponsesProvider


class UnsupportedProtocolError(Exception):
    """Raised when a ModelEntry names a protocol with no registered provider."""

    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        super().__init__(f"Unsupported model protocol: {protocol!r}")


# Providers are stateless (pure transforms) → module-level singletons are safe.
_REGISTRY: dict[str, Provider] = {
    AnthropicBedrockProvider.name: AnthropicBedrockProvider(),
    OpenAIResponsesProvider.name: OpenAIResponsesProvider(),
}


def get_provider(entry: ModelEntry) -> Provider:
    """Return the provider singleton for *entry*'s protocol.

    Raises :class:`UnsupportedProtocolError` for an unknown protocol.
    """
    try:
        return _REGISTRY[entry.protocol]
    except KeyError as exc:
        raise UnsupportedProtocolError(entry.protocol) from exc


__all__ = [
    "Provider",
    "AnthropicBedrockProvider",
    "OpenAIResponsesProvider",
    "UnsupportedProtocolError",
    "get_provider",
]
