"""
OpenAI Audio Transcriptions provider — passthrough.

OpenRouter exposes OpenAI's speech-to-text wire format at
``/api/v1/audio/transcriptions``. The client already speaks native OpenAI Audio,
so this dialect is an identity map: the multipart request (audio file + text
fields) and the JSON response flow through unchanged except for the server-side
model swap. The server encodes the multipart body directly (see
``/v1/audio/transcriptions`` in ``server.py``), mirroring ``images/edits``.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class AudioPassthroughDialect(Dialect):
    """OpenAI Audio Transcriptions API, verbatim passthrough."""

    name = "openai-audio"
    supports_stream = True

    def operation_path(
        self, entry: ModelEntry, stream: bool, *, operation: str | None = None
    ) -> str:
        # Bare operation, relative to the OpenAI-compat API root. The transport
        # owns the root prefix (Azure's base already ends in /openai/v1; the
        # generic http transport uses the route's configured path instead).
        paths = {
            None: "/audio/transcriptions",
            "transcriptions": "/audio/transcriptions",
            "translations": "/audio/translations",
        }
        try:
            return paths[operation]
        except KeyError as exc:
            raise ValueError(f"Unsupported Audio operation: {operation}") from exc

    def build_request(self, client_body: dict, entry: ModelEntry) -> dict:
        # Multipart is encoded by the server into a PreparedRequestBody; the
        # model swap happens there. Pure passthrough here.
        return client_body

    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        # Privacy: the access log must never carry the transcript text. Report
        # only a completion marker; the transcript lives solely in the response
        # body, which is returned verbatim to the client.
        log_info = {
            "input_tokens": "?",
            "output_tokens": "?",
            "finish": "completed",
        }
        return upstream_json, log_info

    async def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Forward upstream Audio SSE without corrupting split UTF-8."""
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
