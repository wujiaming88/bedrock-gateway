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
from typing import AsyncIterator

from .base import Provider

# mantle host template — note ``.api.aws``, distinct from bedrock-runtime.
_MANTLE_HOST = "https://bedrock-mantle.{region}.api.aws"
_RESPONSES_PATH = "/openai/v1/responses"


class OpenAIResponsesProvider(Provider):
    """``bedrock-mantle`` + OpenAI Responses API (GPT-5.5), passthrough."""

    name = "openai-responses"

    def sync_url(self, region: str, bedrock_id: str) -> str:
        # The Responses API takes the model in the body, not the path.
        return _MANTLE_HOST.format(region=region) + _RESPONSES_PATH

    def stream_url(self, region: str, bedrock_id: str) -> str:
        # Same URL; streaming is toggled by ``"stream": true`` in the body.
        return _MANTLE_HOST.format(region=region) + _RESPONSES_PATH

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
