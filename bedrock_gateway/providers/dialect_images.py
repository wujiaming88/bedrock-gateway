"""
OpenAI Images Generations provider — passthrough.

Azure OpenAI image deployments (for example ``gpt-image-2``) are exposed via
``/openai/v1/images/generations`` rather than the Responses API. The client
already speaks the native OpenAI Images dialect, so this dialect is an identity
map: request body, response body, and image payloads (``b64_json`` / URLs) flow
through unchanged except for the server-side model/deployment swap.
"""

from __future__ import annotations

import codecs
import json
from typing import TYPE_CHECKING, AsyncIterator

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class ImagesPassthroughDialect(Dialect):
    """OpenAI Images Generations API, verbatim passthrough."""

    name = "openai-images"
    supports_stream = True

    def operation_path(
        self, entry: "ModelEntry", stream: bool, *, operation: str | None = None
    ) -> str:
        # Bare operation, relative to the OpenAI-compat API root. The transport
        # owns the root prefix (Azure's base already ends in /openai/v1).
        paths = {
            None: "/images/generations",
            "generations": "/images/generations",
            "edits": "/images/edits",
        }
        try:
            return paths[operation]
        except KeyError as exc:
            raise ValueError(f"Unsupported Images operation: {operation}") from exc

    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        # Server already swapped model→upstream id; pure passthrough here.
        return client_body

    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        log_info = {
            "input_tokens": "?",
            "output_tokens": "?",
            "finish": upstream_json.get("status", "completed"),
        }
        return upstream_json, log_info

    async def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Forward upstream Images SSE without corrupting split UTF-8."""
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
        payload = {"type": "error", "code": status, "message": message}
        return f"event: error\ndata: {json.dumps(payload)}\n\n"
