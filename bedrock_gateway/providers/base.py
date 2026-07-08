"""
Provider abstraction — two orthogonal axes: **Transport** × **Dialect**.

A model's upstream behaviour decomposes into two independent concerns:

  * **Transport** — *where* to send bytes and *how to authenticate*: the host,
    the URL assembly, and the auth headers. One per cloud (Bedrock, Azure, …).
  * **Dialect** — *what shape* the request / response / stream take: request
    building, sync rendering, stream transformation. One per wire format
    (Anthropic Messages, OpenAI Responses passthrough, OpenAI Chat passthrough,
    Embeddings, …).

A model = one ``(Transport, Dialect)`` pair. This keeps the matrix additive:
adding a cloud = one Transport; adding a modality/format = one Dialect —
instead of N clouds × M formats concrete classes.

Everything *cross-cutting* — retries, backoff, timeouts, metrics, the
pre-stream error preflight, error-severity logging — stays in ``server.py`` and
is shared by every (Transport, Dialect) combination.

See ``docs/multi-cloud-multimodal-design.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ..config import ModelEntry


# ---------------------------------------------------------------------------
# Transport — where + how to authenticate
# ---------------------------------------------------------------------------

class Transport(ABC):
    """Locates the upstream and authenticates. Format-agnostic."""

    #: Stable identifier (config ``transport`` value).
    name: str = "base-transport"

    @abstractmethod
    def build_url(
        self, operation_path: str, region: str, entry: "ModelEntry"
    ) -> str:
        """Assemble the full upstream URL.

        *operation_path* is the dialect-supplied path fragment (e.g.
        ``/model/{id}/invoke`` for Bedrock, ``/responses`` for Azure). The
        transport prepends its host and, for Azure, preserves any query string
        already carried by ``entry.azure_endpoint``.
        """

    def auth_headers(self, entry: "ModelEntry") -> dict[str, str] | None:
        """Auth headers for this upstream, or ``None`` to use the gateway's
        global :class:`AuthProvider` (SigV4 / Bearer — the Bedrock case).

        Azure overrides this to inject the resource's ``api-key`` header.
        """
        return None


# ---------------------------------------------------------------------------
# Dialect — request / response / stream shape
# ---------------------------------------------------------------------------

class Dialect(ABC):
    """Shapes the request, response, and stream. Transport-agnostic."""

    #: Stable identifier (config ``dialect`` value).
    name: str = "base-dialect"

    #: Whether this dialect supports streaming (embeddings: False).
    supports_stream: bool = True

    # -- URL operation path (transport prepends the host) ----------------

    @abstractmethod
    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        """Path fragment for this operation, e.g. ``/responses`` or
        ``/model/{bedrock_id}/invoke-with-response-stream``.
        """

    # -- Request shaping -------------------------------------------------

    @abstractmethod
    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        """Shape the client request into the upstream request body.

        Passthrough dialects only swap the model/deployment id; the Anthropic
        dialect performs OpenAI→Anthropic conversion (done upstream in server
        today — see migration notes)."""

    # -- Sync response rendering ----------------------------------------

    @abstractmethod
    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        """Render a 200 upstream JSON body into the client-facing response.

        Returns ``(client_body, log_info)`` where *log_info* carries
        ``input_tokens`` / ``output_tokens`` / ``finish`` for the access log.
        """

    # -- Streaming ------------------------------------------------------

    def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Transform the upstream byte stream into client-facing SSE strings.

        Default raises — non-streaming dialects (embeddings) leave it unset.
        """
        raise NotImplementedError(
            f"dialect {self.name!r} does not support streaming"
        )

    def stream_error(self, message: str, status: int) -> str:
        """Format a mid-stream/timeout error as one client-facing SSE payload."""
        raise NotImplementedError(
            f"dialect {self.name!r} does not support streaming"
        )
