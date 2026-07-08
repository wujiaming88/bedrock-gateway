"""
OpenAI Responses (on Bedrock ``mantle``) provider — passthrough.

GPT-5.5 on Bedrock is only reachable via the ``bedrock-mantle`` endpoint's
OpenAI Responses API (``/openai/v1/responses``). The client already speaks
the native Responses dialect, so this provider is almost an identity map:

  * ``render_sync`` returns the upstream JSON verbatim (only ensuring ``model``
    reflects the client-facing alias), and reads ``usage`` for the access log.
  * ``transform_stream`` forwards the upstream SSE bytes unchanged, and sniffs
    the terminal ``response.completed`` event for usage — without altering or
    reordering any frame.

No field whitelisting / cleaning is done: the body (including ``input_image``
blocks, ``reasoning`` config, tools) flows straight through, preserving fidelity.
"""

from __future__ import annotations

import codecs
import json
from typing import TYPE_CHECKING, AsyncIterator

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry

class ResponsesPassthroughDialect(Dialect):
    """OpenAI Responses API, verbatim passthrough (GPT-5.5 / Grok / Azure).

    Request body flows through untouched except the model id (swapped by the
    server before dispatch); response and SSE are re-emitted verbatim.
    """

    name = "openai-responses"

    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        # Bare operation, relative to the OpenAI-compat API root. The transport
        # owns the root prefix (Bedrock mantle adds ``/openai/v1``; Azure's base
        # already ends in it) — the dialect stays cloud-agnostic.
        return "/responses"

    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        # Server already swapped model→upstream id; pure passthrough here.
        return client_body

    # ------------------------------------------------------------------
    # Sync — verbatim passthrough
    # ------------------------------------------------------------------

    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        usage = upstream_json.get("usage") or {}
        log_info = {
            "input_tokens": usage.get("input_tokens", "?"),
            "output_tokens": usage.get("output_tokens", "?"),
            "finish": upstream_json.get("status", "?"),
        }
        # Passthrough: return the Responses body unchanged.
        return upstream_json, log_info

    # ------------------------------------------------------------------
    # Streaming — SSE passthrough
    # ------------------------------------------------------------------

    async def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Forward upstream Responses SSE bytes unchanged.

        The upstream already emits well-formed ``event:``/``data:`` SSE frames,
        so the gateway simply re-yields them as text. An incremental UTF-8
        decoder is used so multi-byte characters split across upstream chunk
        boundaries are not corrupted (e.g. CJK text in ``delta`` fields).
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        async for raw in byte_iter:
            if raw:
                text = decoder.decode(raw)
                if text:
                    yield text
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail

    def stream_error(self, message: str, status: int) -> str:
        """Emit a Responses-style ``error`` SSE event as a clean terminator."""
        payload = {
            "type": "error",
            "code": status,
            "message": message,
        }
        return (
            f"event: error\ndata: {json.dumps(payload)}\n\n"
        )
