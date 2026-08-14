"""
Inbound translation: Anthropic Messages ⇄ OpenAI Responses.

This is the **mirror** of ``converter.py`` (which translates OpenAI Chat →
Anthropic for Claude on ``/v1/chat/completions``). Here the *client* speaks the
Anthropic Messages protocol (e.g. Claude Code via ``ANTHROPIC_BASE_URL``) while
the *upstream* model speaks OpenAI Responses (GPT-5.5 / Grok on Bedrock mantle,
or ``azure/<deployment>`` on Azure). It lets an Anthropic-only agent drive any
Responses-dialect model with no client changes.

Three pure concerns, no I/O — the server owns transport/auth/retries/metrics:

  * :func:`to_responses_request`   — Anthropic Messages body → Responses body.
  * :func:`to_anthropic_response`  — Responses 200 JSON → Anthropic Messages JSON.
  * :class:`AnthropicStreamAdapter` — Responses SSE → Anthropic Messages SSE, a
    stateful event translator (the bulk of the work: content-block index
    bookkeeping + tool-call ``input_json_delta`` assembly).

Scope (v1): direction is Anthropic-in → Responses-out only. ``thinking`` /
reasoning blocks are dropped (no signature round-trip); ``cache_control`` is
stripped (prompt caching simply doesn't engage). These are lossy-but-safe
degradations, documented in the README.
"""

from __future__ import annotations

import codecs
import json
import logging
import uuid
from typing import Any, AsyncIterator

from .converter import make_anthropic_sse

logger = logging.getLogger("bedrock_gateway")

# ---------------------------------------------------------------------------
# stop_reason mapping
# ---------------------------------------------------------------------------

# Responses terminal status / incomplete-reason → Anthropic stop_reason.
# A function call in the output always wins (→ "tool_use"), handled separately.
_INCOMPLETE_REASON_TO_STOP: dict[str, str] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "end_turn",
}


def _system_to_instructions(system: Any) -> str | None:
    """Flatten an Anthropic ``system`` (string or text-block array) to a string.

    ``cache_control`` and any non-text blocks are ignored — Responses takes a
    plain ``instructions`` string.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else None
    return None


def _image_block_to_url(source: dict) -> str | None:
    """Convert an Anthropic image ``source`` to a Responses ``image_url`` string.

    base64 → a ``data:`` URL; url → the URL verbatim. Mirror of
    ``converter._convert_image_url`` in the opposite direction.
    """
    stype = source.get("type")
    if stype == "base64":
        media = source.get("media_type", "image/png")
        data = source.get("data", "")
        return f"data:{media};base64,{data}"
    if stype == "url":
        return source.get("url") or None
    return None


def _text_image_part(block: Any, text_type: str) -> dict | None:
    """Convert one Anthropic content block to a Responses text/image part.

    Returns ``None`` for blocks that are not text/image (``tool_use`` /
    ``tool_result`` / ``thinking`` — handled by the caller or dropped) and for
    empty/unusable blocks. Single source of truth so string and list content
    take the same path.
    """
    if isinstance(block, str):
        return {"type": text_type, "text": block} if block else None
    if not isinstance(block, dict):
        return {"type": text_type, "text": str(block)} if block else None
    btype = block.get("type")
    if btype == "text":
        return {"type": text_type, "text": block.get("text", "")}
    if btype == "image":
        url = _image_block_to_url(block.get("source", {}))
        return {"type": "input_image", "image_url": url} if url else None
    return None


def _message_to_items(msg: dict) -> list[dict]:
    """Convert a single Anthropic message into an ordered list of Responses
    input items.

    Anthropic nests ``tool_use`` (assistant) and ``tool_result`` (user) inside
    message ``content``; Responses expects ``function_call`` /
    ``function_call_output`` as *top-level* input items. So a single message may
    fan out into: an optional role message (its text/image parts) interleaved
    with standalone call items, in original order.
    """
    role = msg.get("role", "user")
    content = msg.get("content")
    text_type = "output_text" if role == "assistant" else "input_text"
    items: list[dict] = []

    # Normalise to a block list so string and list content share one path.
    blocks = content if isinstance(content, list) else [content]

    pending_parts: list[dict] = []

    def flush() -> None:
        if pending_parts:
            items.append({"role": role, "content": list(pending_parts)})
            pending_parts.clear()

    for block in blocks:
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "tool_use":
            # Assistant tool call → standalone function_call item.
            flush()
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                }
            )
        elif btype == "tool_result":
            # User-supplied tool result → standalone function_call_output.
            flush()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id", ""),
                    "output": _tool_result_to_text(block.get("content")),
                }
            )
        else:
            # text / image / plain string; thinking → None → dropped.
            part = _text_image_part(block, text_type)
            if part is not None:
                pending_parts.append(part)

    flush()
    return items


def _tool_result_to_text(content: Any) -> str:
    """Flatten an Anthropic ``tool_result`` content into a plain string.

    Responses ``function_call_output.output`` is a string; Anthropic allows a
    string or a list of blocks. Concatenate text blocks; JSON-encode the rest.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                else:
                    out.append(json.dumps(block))
            else:
                out.append(str(block))
        return "".join(out)
    return str(content)


def _tools_to_responses(tools: list[dict]) -> list[dict]:
    """Anthropic tools → Responses function tools (flat, not nested)."""
    result: list[dict] = []
    for tool in tools:
        result.append(
            {
                "type": "function",
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            }
        )
    return result


def _tool_choice_to_responses(tc: Any) -> Any:
    """Anthropic ``tool_choice`` → Responses ``tool_choice``."""
    if not isinstance(tc, dict):
        return None
    ttype = tc.get("type")
    if ttype == "auto":
        return "auto"
    if ttype == "any":
        return "required"
    if ttype == "none":
        return "none"
    if ttype == "tool" and tc.get("name"):
        return {"type": "function", "name": tc["name"]}
    return None


def to_responses_request(anthropic_body: dict, upstream_model: str) -> dict:
    """Translate an Anthropic Messages request body into a Responses body.

    *upstream_model* is the already-resolved upstream id (Bedrock model id or
    Azure deployment name) the server will send. Streaming is decided by the
    caller and set on the returned body separately.
    """
    out: dict[str, Any] = {"model": upstream_model}

    instructions = _system_to_instructions(anthropic_body.get("system"))
    if instructions is not None:
        out["instructions"] = instructions

    input_items: list[dict] = []
    for msg in anthropic_body.get("messages", []) or []:
        if isinstance(msg, dict):
            input_items.extend(_message_to_items(msg))
    out["input"] = input_items

    max_tokens = anthropic_body.get("max_tokens")
    if max_tokens is not None:
        out["max_output_tokens"] = max_tokens

    if "temperature" in anthropic_body:
        out["temperature"] = anthropic_body["temperature"]
    if "top_p" in anthropic_body:
        out["top_p"] = anthropic_body["top_p"]

    tools = anthropic_body.get("tools")
    if tools:
        out["tools"] = _tools_to_responses(tools)
        choice = _tool_choice_to_responses(anthropic_body.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice

    return out


# ---------------------------------------------------------------------------
# Sync response translation
# ---------------------------------------------------------------------------


def _stop_reason_from_response(resp: dict, saw_tool_use: bool) -> str:
    """Derive an Anthropic ``stop_reason`` from a Responses body."""
    if saw_tool_use:
        return "tool_use"
    status = resp.get("status")
    if status == "incomplete":
        reason = (resp.get("incomplete_details") or {}).get("reason", "")
        return _INCOMPLETE_REASON_TO_STOP.get(reason, "end_turn")
    return "end_turn"


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def responses_error_to_anthropic(error: Any) -> tuple[int, str, str]:
    """Return ``(status, Anthropic type, message)`` for a Responses error."""
    error = error if isinstance(error, dict) else {"message": str(error or "upstream error")}
    message = str(error.get("message") or "upstream error")
    code = str(error.get("code") or error.get("type") or "").lower()
    lowered = message.lower()
    overflow = (
        code in {"context_length_exceeded", "context_window_exceeded"}
        or ("context" in lowered and ("too long" in lowered or "exceed" in lowered))
        or ("input" in lowered and "too long" in lowered)
    )
    if overflow:
        return 400, "invalid_request_error", message
    if "rate_limit" in code:
        return 429, "rate_limit_error", message
    if "authentication" in code:
        return 401, "authentication_error", message
    if "permission" in code:
        return 403, "permission_error", message
    return 502, "api_error", message


def responses_usage_to_anthropic(usage: Any) -> dict[str, int]:
    """Convert Responses usage into Anthropic's mutually-exclusive buckets."""
    usage = usage if isinstance(usage, dict) else {}
    total_input = _nonnegative_int(usage.get("input_tokens"))
    details = usage.get("input_tokens_details")
    cached = _nonnegative_int(
        details.get("cached_tokens") if isinstance(details, dict) else 0
    )
    cached = min(cached, total_input)
    return {
        "input_tokens": total_input - cached,
        "output_tokens": _nonnegative_int(usage.get("output_tokens")),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def to_anthropic_response(responses_json: dict, model: str) -> dict:
    """Translate a Responses 200 body into an Anthropic Messages response."""
    content: list[dict] = []
    saw_tool_use = False

    for item in responses_json.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    content.append({"type": "text", "text": part.get("text", "")})
        elif itype == "function_call":
            saw_tool_use = True
            args = item.get("arguments", "{}")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, ValueError):
                parsed = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id", ""),
                    "name": item.get("name", ""),
                    "input": parsed if isinstance(parsed, dict) else {},
                }
            )
        # reasoning items dropped.

    usage = responses_usage_to_anthropic(responses_json.get("usage"))
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": _stop_reason_from_response(responses_json, saw_tool_use),
        "stop_sequence": None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Streaming translation
# ---------------------------------------------------------------------------


class AnthropicStreamAdapter:
    """Stateful translator: Responses SSE → Anthropic Messages SSE.

    Anthropic's event contract is strict and ordered:

        message_start
          (content_block_start → content_block_delta* → content_block_stop)*
        message_delta
        message_stop

    Responses streams are organised by *output item* (``output_index``) and, for
    messages, *content part* (``content_index``). This adapter maps each
    Responses text part and each function_call item onto a contiguous Anthropic
    content-block index, opening/closing blocks in order and assembling
    tool-call arguments as ``input_json_delta`` frames.

    The event-translation core is :meth:`on_event` — a *pure* method (Responses
    event type + data → list of Anthropic SSE strings) so it can be unit-tested
    exhaustively without any I/O. :meth:`translate` wraps it with incremental
    SSE parsing over the upstream byte stream.
    """

    def __init__(self, model: str, msg_id: str) -> None:
        self.model = model
        self.msg_id = msg_id
        self._message_started = False
        self._terminal: str | None = None
        self._next_index = 0
        # anthropic block indices currently open, in open order.
        self._open_indices: list[int] = []
        # (output_index, content_index) → anthropic index, for text parts.
        self._text_index: dict[tuple[int, int], int] = {}
        # output_index → {"index": int, "delta_sent": bool}, for tool calls.
        self._tool_index: dict[int, dict[str, Any]] = {}
        # anthropic indices grouped by responses output_index (for item.done).
        self._by_output: dict[int, list[int]] = {}
        self._saw_tool_use = False
        self._usage = responses_usage_to_anthropic(None)

    # -- helpers --------------------------------------------------------

    def _start_message(self) -> list[str]:
        if self._message_started:
            return []
        self._message_started = True
        message = {
            "id": self.msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": dict(self._usage),
        }
        return [
            make_anthropic_sse(
                "message_start", {"type": "message_start", "message": message}
            )
        ]

    def _open_text_block(self, output_index: int, content_index: int) -> list[str]:
        key = (output_index, content_index)
        if key in self._text_index:
            return []
        out = self._start_message()
        idx = self._next_index
        self._next_index += 1
        self._text_index[key] = idx
        self._open_indices.append(idx)
        self._by_output.setdefault(output_index, []).append(idx)
        out.append(
            make_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        return out

    def _open_tool_block(
        self, output_index: int, call_id: str, name: str
    ) -> list[str]:
        if output_index in self._tool_index:
            return []
        out = self._start_message()
        idx = self._next_index
        self._next_index += 1
        self._tool_index[output_index] = {"index": idx, "delta_sent": False}
        self._open_indices.append(idx)
        self._by_output.setdefault(output_index, []).append(idx)
        self._saw_tool_use = True
        out.append(
            make_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": {},
                    },
                },
            )
        )
        return out

    def _close_index(self, idx: int) -> list[str]:
        if idx not in self._open_indices:
            return []
        self._open_indices.remove(idx)
        return [
            make_anthropic_sse(
                "content_block_stop", {"type": "content_block_stop", "index": idx}
            )
        ]

    def _close_all_open(self) -> list[str]:
        out: list[str] = []
        for idx in list(self._open_indices):
            out.extend(self._close_index(idx))
        return out

    # -- pure event translation ----------------------------------------

    def on_event(self, event_type: str, data: dict) -> list[str]:
        """Translate one Responses SSE event into zero or more Anthropic SSE
        strings. Pure and deterministic — the unit-test surface."""
        if self._terminal is not None:
            logger.debug(
                "RESPONSES-STREAM ignored event=%s after terminal=%s model=%s",
                event_type, self._terminal, self.model,
            )
            return []
        if event_type == "response.created":
            resp = data.get("response") or {}
            if resp.get("usage") is not None:
                self._usage = responses_usage_to_anthropic(resp.get("usage"))
            logger.debug(
                "RESPONSES-STREAM created model=%s usage_present=%s input=%d "
                "cache_read=%d output=%d",
                self.model, resp.get("usage") is not None,
                self._usage["input_tokens"], self._usage["cache_read_input_tokens"],
                self._usage["output_tokens"],
            )
            return self._start_message()

        if event_type == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                oi = data.get("output_index", 0)
                return self._open_tool_block(
                    oi, item.get("call_id") or item.get("id", ""), item.get("name", "")
                )
            return []

        if event_type == "response.content_part.added":
            part = data.get("part") or {}
            if part.get("type") == "output_text":
                return self._open_text_block(
                    data.get("output_index", 0), data.get("content_index", 0)
                )
            return []

        if event_type == "response.output_text.delta":
            oi = data.get("output_index", 0)
            ci = data.get("content_index", 0)
            out = self._open_text_block(oi, ci)  # lazy-open if part.added skipped
            idx = self._text_index[(oi, ci)]
            out.append(
                make_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "text_delta",
                            "text": data.get("delta", ""),
                        },
                    },
                )
            )
            return out

        if event_type == "response.function_call_arguments.delta":
            oi = data.get("output_index", 0)
            tool = self._tool_index.get(oi)
            if tool is None:
                return []
            tool["delta_sent"] = True
            return [
                make_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool["index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": data.get("delta", ""),
                        },
                    },
                )
            ]

        if event_type == "response.function_call_arguments.done":
            oi = data.get("output_index", 0)
            tool = self._tool_index.get(oi)
            if tool is None:
                return []
            out: list[str] = []
            # If arguments arrived whole (no deltas), emit them as one frame so
            # the client can still assemble the input.
            if not tool["delta_sent"] and data.get("arguments"):
                out.append(
                    make_anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool["index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": data["arguments"],
                            },
                        },
                    )
                )
            out.extend(self._close_index(tool["index"]))
            return out

        if event_type == "response.content_part.done":
            key = (data.get("output_index", 0), data.get("content_index", 0))
            idx = self._text_index.get(key)
            return self._close_index(idx) if idx is not None else []

        if event_type == "response.output_item.done":
            oi = data.get("output_index", 0)
            out = []
            for idx in list(self._by_output.get(oi, [])):
                out.extend(self._close_index(idx))
            return out

        if event_type in ("response.completed", "response.incomplete"):
            resp = data.get("response") or {}
            return self._finalize(resp)

        if event_type in ("response.failed", "error"):
            resp = data.get("response") or {}
            err = resp.get("error") or data.get("error") or data
            _, error_type, message = responses_error_to_anthropic(err)
            return self.terminate_error(message, error_type)

        # response.in_progress, *.output_text.done, ping, reasoning, unknown →
        # nothing to emit.
        return []

    def _finalize(self, resp: dict) -> list[str]:
        """Close any open blocks, then emit message_delta + message_stop."""
        if self._terminal is not None:
            return []
        out = self._start_message()  # guard: empty response still needs a start
        out.extend(self._close_all_open())
        if resp.get("usage") is not None:
            self._usage = responses_usage_to_anthropic(resp.get("usage"))
        stop_reason = _stop_reason_from_response(resp, self._saw_tool_use)
        out.append(
            make_anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": dict(self._usage),
                },
            )
        )
        out.append(make_anthropic_sse("message_stop", {"type": "message_stop"}))
        self._terminal = "success"
        logger.info(
            "RESPONSES-STREAM terminal=success model=%s status=%s input=%d "
            "cache_read=%d cache_creation=%d output=%d",
            self.model, resp.get("status"), self._usage["input_tokens"],
            self._usage["cache_read_input_tokens"],
            self._usage["cache_creation_input_tokens"], self._usage["output_tokens"],
        )
        return out

    @property
    def terminal(self) -> bool:
        return self._terminal is not None

    def error_event(self, message: str, error_type: str = "api_error") -> str:
        """A single Anthropic ``error`` SSE frame — a legal stream terminator."""
        return make_anthropic_sse(
            "error",
            {"type": "error", "error": {"type": error_type, "message": message}},
        )

    def terminate_error(
        self, message: str, error_type: str = "api_error"
    ) -> list[str]:
        """Terminate once with an Anthropic error; never synthesize success."""
        if self._terminal is not None:
            logger.debug(
                "RESPONSES-STREAM ignored error after terminal=%s model=%s",
                self._terminal, self.model,
            )
            return []
        self._terminal = "error"
        logger.warning(
            "RESPONSES-STREAM terminal=error model=%s error_type=%s",
            self.model, error_type,
        )
        return [self.error_event(message, error_type)]

    def finalize_on_disconnect(self) -> list[str]:
        """Treat EOF without a Responses terminal event as an upstream error."""
        return self.terminate_error(
            "Upstream stream ended without a terminal event"
        )

    # -- byte-stream driver --------------------------------------------

    async def translate(
        self, byte_iter: AsyncIterator[bytes]
    ) -> AsyncIterator[str]:
        """Consume the upstream Responses SSE byte stream and yield Anthropic
        SSE strings. Incremental UTF-8 decoding guards multi-byte (CJK) chars
        split across chunk boundaries; SSE blocks are split on the blank line.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        async for raw in byte_iter:
            if not raw:
                continue
            buf += decoder.decode(raw)
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                event_type, data = _parse_sse_block(block)
                if event_type is None:
                    continue
                for frame in self.on_event(event_type, data):
                    yield frame
        # Flush any buffered tail block (no trailing blank line).
        tail = buf + decoder.decode(b"", final=True)
        if tail.strip():
            event_type, data = _parse_sse_block(tail)
            if event_type is not None:
                for frame in self.on_event(event_type, data):
                    yield frame
        # Ensure a clean terminator if the upstream cut off early.
        for frame in self.finalize_on_disconnect():
            yield frame


def _parse_sse_block(block: str) -> tuple[str | None, dict]:
    """Parse one raw SSE block into ``(event_type, data_dict)``.

    Recognises ``event:`` and ``data:`` lines. When the event type is carried
    only inside the JSON payload (``{"type": "..."}``) rather than an ``event:``
    line, fall back to that. Malformed / non-JSON data yields ``({}``-data).
    """
    event_type: str | None = None
    data_lines: list[str] = []
    for line in block.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    data: dict = {}
    if data_lines:
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
        except (json.JSONDecodeError, ValueError):
            data = {}
    if event_type is None:
        event_type = data.get("type")
    return event_type, data
