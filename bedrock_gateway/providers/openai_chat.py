"""
OpenAI Chat Completions dialect — verbatim passthrough.

For upstreams that already speak the OpenAI Chat Completions wire format
(Azure OpenAI, and any Bedrock mantle model exposed via ``/v1/chat/completions``).
The client body is forwarded untouched except the model id (swapped by the
server before dispatch); the response and its SSE stream are re-emitted verbatim.

Contrast with :class:`AnthropicMessagesDialect`, which *converts* OpenAI Chat
into Anthropic Messages. This dialect does no conversion — upstream already
returns OpenAI-shaped ``choices``.
"""

from __future__ import annotations

import codecs
import json
from typing import TYPE_CHECKING, AsyncIterator

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class ChatPassthroughDialect(Dialect):
    """OpenAI Chat Completions, verbatim passthrough (Azure / mantle chat)."""

    name = "openai-chat"

    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        # Bare operation, relative to the OpenAI-compat API root. The transport
        # owns the root prefix (Bedrock mantle adds ``/openai/v1``; Azure's base
        # already ends in it) — the dialect stays cloud-agnostic.
        return "/chat/completions"

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
        # OpenAI chat usage uses prompt_tokens / completion_tokens.
        log_info = {
            "input_tokens": usage.get("prompt_tokens", "?"),
            "output_tokens": usage.get("completion_tokens", "?"),
            "finish": (
                (upstream_json.get("choices") or [{}])[0].get("finish_reason", "?")
            ),
        }
        return upstream_json, log_info

    # ------------------------------------------------------------------
    # Streaming — SSE passthrough
    # ------------------------------------------------------------------

    async def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Forward upstream Chat Completions SSE bytes unchanged.

        Upstream already emits ``data: {...}\\n\\n`` chunks ending with
        ``data: [DONE]``. An incremental UTF-8 decoder guards against multi-byte
        characters split across chunk boundaries (e.g. CJK in ``delta.content``).
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
        """OpenAI-style error chunk + ``[DONE]`` terminator."""
        return (
            "data: "
            + json.dumps(
                {"error": {"message": message, "type": "api_error", "code": status}}
            )
            + "\n\n"
            + "data: [DONE]\n\n"
        )
