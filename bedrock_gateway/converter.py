"""
Protocol converter: OpenAI ↔ Anthropic (Bedrock) format translation.

Handles:
  - Messages: system extraction, user/assistant/tool conversion
  - Content: text, images (base64 & URL), multimodal arrays
  - Tools: function definitions, tool_calls, tool_results
  - Tool choice: auto / none / required / specific function
  - Response parsing: Bedrock response → OpenAI response
  - Streaming events: AWS event-stream base64 → OpenAI SSE chunks
  - Field sanitization: strips reasoning_content, logprobs, refusal etc.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from typing import Any

# Fields that some clients send but Anthropic doesn't support
STRIP_FIELDS: set[str] = {"reasoning_content", "logprobs", "refusal"}


# ---------------------------------------------------------------------------
# Content conversion
# ---------------------------------------------------------------------------

def convert_content_to_anthropic(content: Any) -> str | list[dict]:
    """
    Convert OpenAI content (string or multimodal array) to Anthropic content.

    Returns either a plain string (for simple text) or a list of content blocks.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content) if content else " "

    blocks: list[dict] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            ptype = part.get("type", "")
            if ptype == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                block = _convert_image_url(part)
                if block:
                    blocks.append(block)
            elif ptype in ("thinking", "reasoning", "redacted_thinking"):
                # Skip thinking blocks — Anthropic doesn't accept them as input
                continue
            else:
                # Unknown type → serialize as text
                blocks.append({"type": "text", "text": json.dumps(part)})

    return blocks if blocks else [{"type": "text", "text": " "}]


def _convert_image_url(part: dict) -> dict | None:
    """Convert an OpenAI image_url content part to an Anthropic image block."""
    url_data = part.get("image_url", {})
    url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)

    if not url:
        return None

    if url.startswith("data:"):
        match = re.match(r"data:(image/[^;]+);base64,(.+)", url, re.DOTALL)
        if match:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": match.group(1),
                    "data": match.group(2),
                },
            }
        return None

    return {"type": "image", "source": {"type": "url", "url": url}}


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def convert_message(msg: dict) -> dict:
    """Convert a single OpenAI message to Anthropic format."""
    role = msg.get("role", "")

    # Assistant with tool_calls
    if role == "assistant" and "tool_calls" in msg:
        return _convert_assistant_tool_calls(msg)

    # Tool result
    if role == "tool":
        return _convert_tool_result(msg)

    # Regular user/assistant message
    content = msg.get("content")
    converted_content = convert_content_to_anthropic(content) if content is not None else " "
    return {"role": role, "content": converted_content}


def _convert_assistant_tool_calls(msg: dict) -> dict:
    """Convert an assistant message with tool_calls to Anthropic format."""
    blocks: list[dict] = []

    # Preserve any text content
    text = msg.get("content")
    if text:
        blocks.append({
            "type": "text",
            "text": text if isinstance(text, str) else str(text),
        })

    # Convert each tool_call to a tool_use block
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"tool_{fn.get('name', 'x')}"),
            "name": fn.get("name", ""),
            "input": args,
        })

    return {
        "role": "assistant",
        "content": blocks or [{"type": "text", "text": " "}],
    }


def _convert_tool_result(msg: dict) -> dict:
    """Convert an OpenAI tool-result message to Anthropic format."""
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id", ""),
            "content": content,
        }],
    }


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

def convert_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    result: list[dict] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
        else:
            fn = tool
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get(
                "parameters",
                fn.get("input_schema", {"type": "object", "properties": {}}),
            ),
        })
    return result


def convert_tool_choice(
    tc: Any, has_tools: bool
) -> dict[str, str] | None:
    """Convert OpenAI tool_choice to Anthropic format."""
    if not tc or not has_tools:
        return None
    if tc == "auto":
        return {"type": "auto"}
    if tc == "none":
        return {"type": "none"}
    if tc == "required":
        return {"type": "any"}
    if isinstance(tc, dict) and tc.get("function"):
        return {"type": "tool", "name": tc["function"]["name"]}
    return None


# ---------------------------------------------------------------------------
# System message extraction
# ---------------------------------------------------------------------------

def extract_system_and_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """
    Split OpenAI messages into (system_prompt, chat_messages).

    System messages are concatenated into a single string.  All other
    messages are converted to Anthropic format.
    """
    system_parts: list[str] = []
    chat_messages: list[dict] = []

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
        else:
            chat_messages.append(convert_message(msg))

    system = "\n".join(system_parts) if system_parts else None
    return system, chat_messages


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def convert_usage(bedrock_usage: dict) -> dict:
    """Convert Bedrock usage format to OpenAI format."""
    input_t = bedrock_usage.get("input_tokens", 0)
    output_t = bedrock_usage.get("output_tokens", 0)
    return {
        "prompt_tokens": input_t,
        "completion_tokens": output_t,
        "total_tokens": input_t + output_t,
    }


def parse_bedrock_response(result: dict) -> tuple[dict, str]:
    """
    Parse a Bedrock (Anthropic) response into an OpenAI-compatible message.

    Returns ``(message_dict, finish_reason)``.
    """
    text = ""
    tool_calls: list[dict] = []
    thinking_blocks: list[dict] = []

    for block in result.get("content", []):
        btype = block.get("type", "")
        if btype == "text":
            text += block.get("text", "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif btype == "thinking":
            thinking_blocks.append(block)
        elif btype == "redacted_thinking":
            thinking_blocks.append(block)

    # Build reasoning_content from thinking blocks
    reasoning_content = ""
    for tb in thinking_blocks:
        reasoning_content += tb.get("thinking", "")

    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else "stop"
    return message, finish_reason


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reasoning effort mapping
# ---------------------------------------------------------------------------

# Maps OpenAI reasoning_effort levels to Anthropic thinking parameters
REASONING_EFFORT_MAP: dict[str, dict[str, Any]] = {
    "minimal": {"type": "enabled", "budget_tokens": 128},
    "low": {"type": "enabled", "budget_tokens": 1024},
    "medium": {"type": "enabled", "budget_tokens": 2048},
    "high": {"type": "enabled", "budget_tokens": 4096},
}

# Bedrock model IDs that support adaptive thinking (Claude 4.6/4.7)
_ADAPTIVE_THINKING_PATTERNS: list[str] = [
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
]


def _model_supports_adaptive(model: str) -> bool:
    """Check if a Bedrock model ID supports adaptive thinking."""
    return any(pattern in model for pattern in _ADAPTIVE_THINKING_PATTERNS)


def map_reasoning_effort(effort: str, model: str) -> dict[str, Any] | None:
    """
    Map an OpenAI ``reasoning_effort`` value to an Anthropic ``thinking`` parameter.

    For Claude 4.6/4.7 models, returns ``{"type": "adaptive"}`` regardless of
    effort level.  For other models, returns a budget_tokens-based config.

    Returns ``None`` if the effort level is not recognized.
    """
    if effort not in REASONING_EFFORT_MAP:
        return None
    if _model_supports_adaptive(model):
        return {"type": "adaptive"}
    return dict(REASONING_EFFORT_MAP[effort])  # shallow copy


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def make_stream_chunk(
    msg_id: str,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    """Build an OpenAI SSE chunk line."""
    payload = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def decode_event_stream_chunk(buf: bytes) -> list[dict]:
    """
    Extract base64-encoded JSON events from an AWS event-stream binary chunk.

    The Bedrock streaming response wraps each event as a JSON payload inside
    binary event-stream frames with ``"bytes":"<base64>"`` encoding.
    """
    events: list[dict] = []
    for match in re.finditer(rb'"bytes":"([A-Za-z0-9+/=]+)"', buf):
        try:
            decoded = json.loads(base64.b64decode(match.group(1)))
            events.append(decoded)
        except (json.JSONDecodeError, ValueError):
            pass
    return events
