"""
Anthropic-on-Bedrock provider.

Wraps the gateway's original behaviour: talk to ``bedrock-runtime`` using the
Anthropic Messages wire format (all Claude models). The rendering and
stream-transformation logic here is a faithful port of what used to live inline
in ``server.py``'s ``_handle_sync`` / ``_handle_stream`` — no behaviour change.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

import httpx

from ..converter import (
    convert_usage,
    decode_event_stream_chunk,
    make_stream_chunk,
    parse_bedrock_error,
    parse_bedrock_response,
)
from typing import TYPE_CHECKING

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class AnthropicMessagesDialect(Dialect):
    """Anthropic Messages wire format (Claude family, on Bedrock runtime).

    Request bodies are built by the server today (OpenAI→Anthropic conversion
    happens there), so ``build_request`` is identity in this dialect; response
    and stream shaping are the substantive part.
    """

    name = "anthropic"

    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        op = "invoke-with-response-stream" if stream else "invoke"
        return f"/model/{entry.bedrock_id}/{op}"

    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        # Server already assembled the Anthropic body before dispatch.
        return client_body

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        message, finish = parse_bedrock_response(upstream_json)
        usage = upstream_json.get("usage", {})
        client_body = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish}
            ],
            "usage": convert_usage(usage),
        }
        log_info = {
            "input_tokens": usage.get("input_tokens", "?"),
            "output_tokens": usage.get("output_tokens", "?"),
            "finish": finish,
        }
        return client_body, log_info

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def transform_stream(
        self, byte_iter: AsyncIterator[bytes], model: str, msg_id: str
    ) -> AsyncIterator[str]:
        """Port of the original chat streaming ``generate()`` body.

        Decodes the AWS binary event-stream and re-emits OpenAI SSE chunks.
        Mid-stream fault frames (``_exception``) are surfaced as a visible
        error chunk + ``[DONE]`` so the client never hangs.
        """
        buf = b""
        stream_input_tokens = 0
        stream_output_tokens = 0
        current_tool_id: str | None = None
        current_tool_name: str | None = None

        async for raw in byte_iter:
            buf += raw
            events, consumed = decode_event_stream_chunk(buf)
            if consumed > 0:
                buf = buf[consumed:]
            for event in events:
                etype = event.get("type", "")

                if etype == "_exception":
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "error": {
                                    "message": event.get(
                                        "message", "upstream stream error"
                                    ),
                                    "type": parse_bedrock_error(
                                        event.get("status", 500), ""
                                    )["type"],
                                    "code": event.get("status", 500),
                                }
                            }
                        )
                        + "\n\n"
                    )
                    yield "data: [DONE]\n\n"
                    return

                if etype == "message_start":
                    _mu = event.get("message", {}).get("usage", {})
                    stream_input_tokens = _mu.get("input_tokens", 0)
                    yield make_stream_chunk(msg_id, model, {"role": "assistant"})

                elif etype == "content_block_start":
                    cb = event.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        current_tool_id = cb.get("id", "")
                        current_tool_name = cb.get("name", "")
                        yield make_stream_chunk(
                            msg_id,
                            model,
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": current_tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": current_tool_name,
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                        )
                    elif cb.get("type") == "thinking":
                        pass

                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    dtype = delta.get("type", "")
                    if dtype == "text_delta":
                        yield make_stream_chunk(
                            msg_id, model, {"content": delta.get("text", "")}
                        )
                    elif dtype == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        yield make_stream_chunk(
                            msg_id,
                            model,
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": partial},
                                    }
                                ]
                            },
                        )
                    elif dtype == "thinking_delta":
                        yield make_stream_chunk(
                            msg_id,
                            model,
                            {"reasoning_content": delta.get("thinking", "")},
                        )
                    elif dtype == "signature_delta":
                        pass

                elif etype == "content_block_stop":
                    current_tool_id = None
                    current_tool_name = None

                elif etype == "message_delta":
                    sr = event.get("delta", {}).get("stop_reason", "end_turn")
                    fr = "tool_calls" if sr == "tool_use" else "stop"
                    _du = event.get("usage", {})
                    if _du.get("output_tokens"):
                        stream_output_tokens = _du["output_tokens"]
                    if _du.get("input_tokens"):
                        stream_input_tokens = _du["input_tokens"]
                    yield make_stream_chunk(msg_id, model, {}, fr)
                    _usage = {
                        "prompt_tokens": stream_input_tokens,
                        "completion_tokens": stream_output_tokens,
                        "total_tokens": stream_input_tokens
                        + stream_output_tokens,
                    }
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [],
                                "usage": _usage,
                            }
                        )
                        + "\n\n"
                    )

        yield "data: [DONE]\n\n"

    def stream_error(self, message: str, status: int) -> str:
        return (
            "data: "
            + json.dumps(
                {
                    "error": {
                        "message": message,
                        "type": "api_error",
                        "code": status,
                    }
                }
            )
            + "\n\n"
            + "data: [DONE]\n\n"
        )
