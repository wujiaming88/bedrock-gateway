"""
Provider abstraction — the *upstream dialect* layer.

A :class:`Provider` encapsulates everything that differs between the ways the
gateway can talk to an upstream model service:

  * which host / URL path to hit (``sync_url`` / ``stream_url``)
  * how to render a successful upstream response into the client-facing shape
    (``render_sync``)
  * how to transform the upstream byte stream into client-facing SSE
    (``transform_stream``), plus how to format a mid-stream error for that
    protocol (``stream_error``)

Everything *protocol-agnostic* — retries, backoff, timeouts, metrics, the
pre-stream error preflight (``_open_upstream_stream``), error-severity logging —
stays in ``server.py`` and is shared by every provider. Adding a new upstream
(model family, endpoint, wire format) means adding one Provider subclass and one
``ModelEntry``; ``server.py`` does not change.

Two concrete providers exist today:

  * :class:`~bedrock_gateway.providers.anthropic_bedrock.AnthropicBedrockProvider`
    — ``bedrock-runtime`` + Anthropic Messages wire format (all Claude models).
  * :class:`~bedrock_gateway.providers.openai_responses.OpenAIResponsesProvider`
    — ``bedrock-mantle`` + OpenAI Responses API (GPT-5.5), passthrough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class Provider(ABC):
    """Upstream-dialect strategy consumed by the shared server orchestration."""

    #: Stable identifier, also used as the registry key's mirror for guards.
    name: str = "base"

    # -- URL construction ------------------------------------------------

    @abstractmethod
    def sync_url(self, region: str, bedrock_id: str) -> str:
        """Full URL for a non-streaming call to *bedrock_id* in *region*."""

    @abstractmethod
    def stream_url(self, region: str, bedrock_id: str) -> str:
        """Full URL for a streaming call to *bedrock_id* in *region*."""

    # -- Sync response rendering ----------------------------------------

    @abstractmethod
    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        """Render a 200 upstream JSON body into the client-facing response.

        Returns ``(client_body, log_info)`` where *client_body* is the dict
        serialised back to the caller and *log_info* is a small dict with
        ``input_tokens`` / ``output_tokens`` / ``finish`` for the access log.
        """

    # -- Streaming ------------------------------------------------------

    @abstractmethod
    def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Transform the upstream byte stream into client-facing SSE strings.

        Must yield fully-formed SSE payloads (``"data: ...\\n\\n"`` etc.),
        handle any mid-stream upstream fault frames, and emit the protocol's
        stream terminator. Implemented as an async generator.
        """

    @abstractmethod
    def stream_error(self, message: str, status: int) -> str:
        """Format a mid-stream/timeout error as one client-facing SSE payload.

        Called by the server's stream wrapper when the connection faults
        *after* a 200 was already committed, so the client sees a clean error
        instead of a truncated stream.
        """
