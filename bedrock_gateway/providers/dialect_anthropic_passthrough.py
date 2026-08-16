"""Anthropic Messages dialect — verbatim HTTP passthrough."""

from __future__ import annotations

import codecs
import json
from typing import TYPE_CHECKING, AsyncIterator

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class AnthropicPassthroughDialect(Dialect):
    """Native Anthropic Messages JSON and SSE without protocol conversion."""

    name = "anthropic-passthrough"

    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        return "/v1/messages"

    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        return client_body

    def render_sync(self, upstream_json: dict, model: str) -> tuple[dict, dict]:
        usage = upstream_json.get("usage") or {}
        return upstream_json, {
            "input_tokens": usage.get("input_tokens", "?"),
            "output_tokens": usage.get("output_tokens", "?"),
            "finish": upstream_json.get("stop_reason", "?"),
        }

    async def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
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
        payload = {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
        return f"event: error\ndata: {json.dumps(payload)}\n\n"
