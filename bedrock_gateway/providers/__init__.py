"""
Provider registries — Transport × Dialect (see design doc).

A model resolves to a ``(Transport, Dialect)`` pair:
  * transport ← ``entry.transport`` (``bedrock`` | ``azure``)
  * dialect   ← ``entry.dialect``   (``anthropic`` | ``openai-responses`` | …)

Adding a cloud = register one Transport; adding a wire format = register one
Dialect. The server orchestration consumes both, staying format-/cloud-agnostic.
"""

from __future__ import annotations

from ..config import ModelEntry
from .base import Dialect, Transport
from .dialect_anthropic import AnthropicMessagesDialect
from .dialect_chat import ChatPassthroughDialect
from .dialect_images import ImagesPassthroughDialect
from .dialect_responses import ResponsesPassthroughDialect
from .transports import AzureTransport, BedrockTransport


class UnsupportedProtocolError(Exception):
    """Raised when a ModelEntry names a transport/dialect with no provider."""

    def __init__(self, what: str) -> None:
        super().__init__(f"Unsupported model provider component: {what!r}")


# Stateless singletons (pure transforms) — safe to share across requests.
_TRANSPORTS: dict[str, Transport] = {
    BedrockTransport.name: BedrockTransport(),
    AzureTransport.name: AzureTransport(),
}

_DIALECTS: dict[str, Dialect] = {
    AnthropicMessagesDialect.name: AnthropicMessagesDialect(),
    ResponsesPassthroughDialect.name: ResponsesPassthroughDialect(),
    ChatPassthroughDialect.name: ChatPassthroughDialect(),
    ImagesPassthroughDialect.name: ImagesPassthroughDialect(),
}


def get_transport(entry: ModelEntry) -> Transport:
    """Return the transport singleton for *entry*."""
    try:
        return _TRANSPORTS[entry.transport]
    except KeyError as exc:
        raise UnsupportedProtocolError(entry.transport) from exc


def get_dialect(entry: ModelEntry) -> Dialect:
    """Return the dialect singleton for *entry*."""
    try:
        return _DIALECTS[entry.dialect]
    except KeyError as exc:
        raise UnsupportedProtocolError(entry.dialect) from exc


__all__ = [
    "Transport",
    "Dialect",
    "BedrockTransport",
    "AzureTransport",
    "AnthropicMessagesDialect",
    "ResponsesPassthroughDialect",
    "ChatPassthroughDialect",
    "ImagesPassthroughDialect",
    "UnsupportedProtocolError",
    "get_transport",
    "get_dialect",
]
