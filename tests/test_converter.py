"""
Tests for bedrock_gateway.converter — protocol translation between
OpenAI and Anthropic (Bedrock) formats.
"""

import json

import pytest

from bedrock_gateway.converter import (
    convert_content_to_anthropic,
    convert_message,
    convert_tool_choice,
    convert_tools,
    decode_event_stream_chunk,
    extract_system_and_messages,
    make_stream_chunk,
    map_reasoning_effort,
    parse_bedrock_error,
    parse_bedrock_response,
    stream_exception_status,
)


# ─── Content Conversion ──────────────────────────────────────────────


class TestConvertContentToAnthropic:
    """Tests for convert_content_to_anthropic."""

    def test_string_passthrough(self):
        assert convert_content_to_anthropic("hello") == "hello"

    def test_empty_string(self):
        assert convert_content_to_anthropic("") == ""

    def test_none_returns_space(self):
        # None content should be treated as non-list, non-str
        result = convert_content_to_anthropic(None)
        assert result == " "

    def test_integer_to_string(self):
        result = convert_content_to_anthropic(42)
        assert result == "42"

    def test_text_parts(self):
        parts = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " World"},
        ]
        result = convert_content_to_anthropic(parts)
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " World"},
        ]

    def test_string_in_list(self):
        result = convert_content_to_anthropic(["plain text"])
        assert result == [{"type": "text", "text": "plain text"}]

    def test_image_url_base64(self):
        parts = [{
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgo=",
            },
        }]
        result = convert_content_to_anthropic(parts)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[0]["source"]["data"] == "iVBORw0KGgo="

    def test_image_url_http(self):
        parts = [{
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.jpg"},
        }]
        result = convert_content_to_anthropic(parts)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/cat.jpg"

    def test_image_url_string_value(self):
        """image_url value can be a plain string instead of dict."""
        parts = [{
            "type": "image_url",
            "image_url": "https://example.com/dog.jpg",
        }]
        result = convert_content_to_anthropic(parts)
        assert result[0]["source"]["url"] == "https://example.com/dog.jpg"

    def test_thinking_blocks_skipped(self):
        parts = [
            {"type": "thinking", "thinking": "internal"},
            {"type": "text", "text": "visible"},
            {"type": "reasoning", "content": "also internal"},
            {"type": "redacted_thinking", "data": "secret"},
        ]
        result = convert_content_to_anthropic(parts)
        assert len(result) == 1
        assert result[0]["text"] == "visible"

    def test_unknown_type_serialized(self):
        parts = [{"type": "custom", "data": 123}]
        result = convert_content_to_anthropic(parts)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        parsed = json.loads(result[0]["text"])
        assert parsed["data"] == 123

    def test_empty_list_returns_space(self):
        result = convert_content_to_anthropic([])
        assert result == [{"type": "text", "text": " "}]

    def test_only_thinking_returns_space(self):
        parts = [{"type": "thinking", "content": "hmm"}]
        result = convert_content_to_anthropic(parts)
        assert result == [{"type": "text", "text": " "}]


# ─── Message Conversion ──────────────────────────────────────────────


class TestConvertMessage:
    """Tests for convert_message."""

    def test_user_message(self):
        msg = {"role": "user", "content": "hi"}
        result = convert_message(msg)
        assert result == {"role": "user", "content": "hi"}

    def test_assistant_message(self):
        msg = {"role": "assistant", "content": "hello"}
        result = convert_message(msg)
        assert result == {"role": "assistant", "content": "hello"}

    def test_none_content(self):
        msg = {"role": "assistant", "content": None}
        result = convert_message(msg)
        assert result["content"] == " "

    def test_missing_content(self):
        msg = {"role": "assistant"}
        result = convert_message(msg)
        assert result["content"] == " "

    def test_assistant_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            }],
        }
        result = convert_message(msg)
        assert result["role"] == "assistant"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "Let me check"}
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["id"] == "call_123"
        assert result["content"][1]["name"] == "get_weather"
        assert result["content"][1]["input"] == {"city": "London"}

    def test_assistant_tool_calls_invalid_json_args(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_456",
                "function": {
                    "name": "broken",
                    "arguments": "not-json",
                },
            }],
        }
        result = convert_message(msg)
        # Should fall back to empty dict
        assert result["content"][0]["input"] == {}

    def test_tool_result(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Temperature is 20°C",
        }
        result = convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "call_123"
        assert result["content"][0]["content"] == "Temperature is 20°C"

    def test_tool_result_non_string_content(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_789",
            "content": {"temp": 20},
        }
        result = convert_message(msg)
        assert isinstance(result["content"][0]["content"], str)


# ─── Tools Conversion ────────────────────────────────────────────────


class TestConvertTools:
    """Tests for convert_tools."""

    def test_openai_function_tools(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }]
        result = convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a city"
        assert result[0]["input_schema"]["properties"]["city"]["type"] == "string"

    def test_bare_function_tools(self):
        """Tools without wrapping {"type": "function", "function": ...}."""
        tools = [{
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }]
        result = convert_tools(tools)
        assert result[0]["name"] == "search"

    def test_empty_tools(self):
        assert convert_tools([]) == []


class TestConvertToolChoice:
    """Tests for convert_tool_choice."""

    def test_auto(self):
        assert convert_tool_choice("auto", True) == {"type": "auto"}

    def test_none(self):
        assert convert_tool_choice("none", True) == {"type": "none"}

    def test_required(self):
        assert convert_tool_choice("required", True) == {"type": "any"}

    def test_specific_function(self):
        tc = {"function": {"name": "get_weather"}}
        result = convert_tool_choice(tc, True)
        assert result == {"type": "tool", "name": "get_weather"}

    def test_no_tools_returns_none(self):
        assert convert_tool_choice("auto", False) is None

    def test_none_value(self):
        assert convert_tool_choice(None, True) is None

    def test_empty_string(self):
        assert convert_tool_choice("", True) is None


# ─── System Extraction ────────────────────────────────────────────────


class TestExtractSystemAndMessages:
    """Tests for extract_system_and_messages."""

    def test_basic(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, chat = extract_system_and_messages(messages)
        assert system == "You are helpful."
        assert len(chat) == 1
        assert chat[0]["role"] == "user"

    def test_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        system, chat = extract_system_and_messages(messages)
        assert system == "Rule 1\nRule 2"
        assert len(chat) == 1

    def test_no_system(self):
        messages = [{"role": "user", "content": "Hi"}]
        system, chat = extract_system_and_messages(messages)
        assert system is None
        assert len(chat) == 1

    def test_system_with_content_blocks(self):
        messages = [{
            "role": "system",
            "content": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
            ],
        }]
        system, chat = extract_system_and_messages(messages)
        assert system == "Block 1\nBlock 2"


# ─── Response Parsing ─────────────────────────────────────────────────


class TestParsBedrockResponse:
    """Tests for parse_bedrock_response."""

    def test_text_response(self):
        result = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        message, finish = parse_bedrock_response(result)
        assert message["role"] == "assistant"
        assert message["content"] == "Hello!"
        assert finish == "stop"
        assert "tool_calls" not in message

    def test_tool_use_response(self):
        result = {
            "content": [
                {"type": "text", "text": "Checking weather..."},
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        }
        message, finish = parse_bedrock_response(result)
        assert message["content"] == "Checking weather..."
        assert finish == "tool_calls"
        assert len(message["tool_calls"]) == 1
        tc = message["tool_calls"][0]
        assert tc["id"] == "tu_123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}

    def test_empty_content(self):
        message, finish = parse_bedrock_response({"content": []})
        assert message["content"] is None
        assert finish == "stop"

    def test_parse_thinking_blocks(self):
        """Thinking blocks are extracted into reasoning_content."""
        result = {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "thinking", "thinking": " Step 2."},
                {"type": "text", "text": "Here is my answer."},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        message, finish = parse_bedrock_response(result)
        assert message["role"] == "assistant"
        assert message["content"] == "Here is my answer."
        assert message["reasoning_content"] == "Let me think... Step 2."
        assert finish == "stop"

    def test_parse_redacted_thinking(self):
        """Redacted thinking blocks are collected; they have no 'thinking' text."""
        result = {
            "content": [
                {"type": "thinking", "thinking": "Visible thought."},
                {"type": "redacted_thinking", "data": "encrypted-blob"},
                {"type": "text", "text": "Final answer."},
            ],
        }
        message, finish = parse_bedrock_response(result)
        assert message["content"] == "Final answer."
        # redacted_thinking has no 'thinking' key, so only the first block contributes
        assert message["reasoning_content"] == "Visible thought."
        assert finish == "stop"

    def test_no_thinking_no_reasoning_content(self):
        """When there are no thinking blocks, reasoning_content is absent."""
        result = {"content": [{"type": "text", "text": "Just text."}]}
        message, _ = parse_bedrock_response(result)
        assert "reasoning_content" not in message


# ─── Reasoning Effort Mapping ───────────────────────────────────────


class TestReasoningEffortMapping:
    """Tests for map_reasoning_effort."""

    def test_reasoning_effort_mapping_levels(self):
        """Each effort level maps to the correct budget_tokens."""
        model = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # non-4.6/4.7
        result = map_reasoning_effort("minimal", model)
        assert result == {"type": "enabled", "budget_tokens": 128}

        result = map_reasoning_effort("low", model)
        assert result == {"type": "enabled", "budget_tokens": 1024}

        result = map_reasoning_effort("medium", model)
        assert result == {"type": "enabled", "budget_tokens": 2048}

        result = map_reasoning_effort("high", model)
        assert result == {"type": "enabled", "budget_tokens": 4096}

    def test_reasoning_effort_unknown_returns_none(self):
        result = map_reasoning_effort("extreme", "some-model")
        assert result is None

    def test_reasoning_effort_claude_4_6_adaptive(self):
        """Claude 4.6/4.7 models use adaptive thinking."""
        for model_id in [
            "us.anthropic.claude-opus-4-6-v1",
            "us.anthropic.claude-opus-4-7",
            "us.anthropic.claude-sonnet-4-6",
        ]:
            result = map_reasoning_effort("high", model_id)
            assert result == {"type": "adaptive"}, f"Failed for {model_id}"

    def test_reasoning_effort_returns_copy(self):
        """Returned dict should be a copy, not the original."""
        model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        r1 = map_reasoning_effort("low", model)
        r2 = map_reasoning_effort("low", model)
        assert r1 is not r2

    def test_budget_tokens_clamp(self):
        """Budget tokens below 1024 should be clamped by the caller."""
        model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        result = map_reasoning_effort("minimal", model)
        assert result is not None
        # The mapper returns the raw value; clamping is done in server.py
        assert result["budget_tokens"] == 128


# ─── Streaming Helpers ─────────────────────────────────────────────────


class TestMakeStreamChunk:
    """Tests for make_stream_chunk."""

    def test_text_delta(self):
        line = make_stream_chunk("id-1", "model-1", {"content": "hi"})
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[6:])
        assert payload["id"] == "id-1"
        assert payload["choices"][0]["delta"]["content"] == "hi"
        assert payload["choices"][0]["finish_reason"] is None

    def test_finish_reason(self):
        line = make_stream_chunk("id-2", "m", {}, "stop")
        payload = json.loads(line[6:])
        assert payload["choices"][0]["finish_reason"] == "stop"


class TestDecodeEventStreamChunk:
    """Tests for decode_event_stream_chunk."""

    def test_extracts_events(self):
        import base64 as b64

        event = {"type": "content_block_delta", "delta": {"text": "hi"}}
        encoded = b64.b64encode(json.dumps(event).encode()).decode()
        raw = f'{{"bytes":"{encoded}"}}'.encode()
        events, consumed = decode_event_stream_chunk(raw)
        assert len(events) == 1
        assert events[0]["type"] == "content_block_delta"
        assert consumed > 0

    def test_invalid_base64_skipped(self):
        raw = b'{"bytes":"not-valid-json-after-decode!!!"}'
        events, consumed = decode_event_stream_chunk(raw)
        # Should silently skip invalid entries
        assert isinstance(events, list)

    def test_empty_input(self):
        events, consumed = decode_event_stream_chunk(b"")
        assert events == []
        assert consumed == 0

    def test_multiple_events(self):
        import base64 as b64

        e1 = {"type": "message_start"}
        e2 = {"type": "content_block_delta", "delta": {"text": "x"}}
        parts = []
        for e in [e1, e2]:
            enc = b64.b64encode(json.dumps(e).encode()).decode()
            parts.append(f'{{"bytes":"{enc}"}}'.encode())
        raw = b" ".join(parts)
        events, consumed = decode_event_stream_chunk(raw)
        assert len(events) == 2
        assert consumed > 0
        assert consumed <= len(raw)


def _exc_frame(exc_type: str, message: str, msg_key: str = "message") -> bytes:
    """Build a realistic AWS event-stream exception frame: an ``:exception-type``
    header (a few binary header bytes), the modeled ``*Exception`` name, then a
    JSON ``message`` payload — NOT a ``"bytes"``-wrapped event."""
    return (
        b"\x00\x00\x0d:exception-type\x07\x00"
        + bytes([len(exc_type)])
        + exc_type.encode()
        + json.dumps({msg_key: message}).encode()
    )


class TestDecodeExceptionFrames:
    """The decoder must surface mid-stream Bedrock exception frames instead of
    silently dropping them (which used to leave clients hanging forever)."""

    def test_throttling_exception_surfaced(self):
        events, consumed = decode_event_stream_chunk(
            _exc_frame("throttlingException", "Rate exceeded")
        )
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "_exception"
        assert ev["exception_type"] == "throttlingException"
        assert ev["message"] == "Rate exceeded"
        assert ev["status"] == 429
        assert consumed > 0

    def test_internal_server_exception_capital_message(self):
        # Bedrock varies capitalization: "Message" vs "message".
        events, _ = decode_event_stream_chunk(
            _exc_frame("internalServerException", "boom", msg_key="Message")
        )
        assert events[0]["exception_type"] == "internalServerException"
        assert events[0]["message"] == "boom"
        assert events[0]["status"] == 500

    def test_model_stream_error_maps_to_502(self):
        events, _ = decode_event_stream_chunk(
            _exc_frame("modelStreamErrorException", "stream broke")
        )
        assert events[0]["status"] == 502

    def test_unknown_exception_defaults_to_500(self):
        events, _ = decode_event_stream_chunk(
            _exc_frame("someNovelException", "weird")
        )
        assert events[0]["type"] == "_exception"
        assert events[0]["status"] == 500

    def test_normal_event_then_exception_frame(self):
        import base64 as b64

        good = {"type": "content_block_delta", "delta": {"text": "hi"}}
        enc = b64.b64encode(json.dumps(good).encode()).decode()
        raw = (
            f'{{"bytes":"{enc}"}}'.encode()
            + b"  "
            + _exc_frame("throttlingException", "slow down")
        )
        events, consumed = decode_event_stream_chunk(raw)
        types = [e["type"] for e in events]
        assert types == ["content_block_delta", "_exception"]
        assert events[1]["status"] == 429
        assert consumed <= len(raw)

    def test_exception_without_message_uses_fallback(self):
        # No JSON message payload at all — must still surface, not drop.
        raw = b"\x00\x00\x0d:exception-type\x07\x00\x13throttlingException"
        events, _ = decode_event_stream_chunk(raw)
        assert len(events) == 1
        assert events[0]["type"] == "_exception"
        assert "throttlingException" in events[0]["message"]


class TestStreamExceptionStatus:
    def test_known_types(self):
        assert stream_exception_status("throttlingException") == 429
        assert stream_exception_status("serviceUnavailableException") == 503
        assert stream_exception_status("validationException") == 400
        assert stream_exception_status("modelTimeoutException") == 504

    def test_unknown_defaults_500(self):
        assert stream_exception_status("nopeException") == 500


# ─── Error Parsing ──────────────────────────────────────────────────


class TestParsBedrockError:
    """Tests for parse_bedrock_error."""

    def test_json_with_message_field(self):
        body = json.dumps({"message": "Model not found"})
        err = parse_bedrock_error(404, body)
        assert err["message"] == "Model not found"
        assert err["type"] == "not_found_error"
        assert err["code"] == 404

    def test_json_with_capital_message(self):
        body = json.dumps({"Message": "Access denied"})
        err = parse_bedrock_error(403, body)
        assert err["message"] == "Access denied"
        assert err["type"] == "permission_error"

    def test_plain_text_body(self):
        err = parse_bedrock_error(500, "Internal Server Error")
        assert err["message"] == "Internal Server Error"
        assert err["type"] == "api_error"

    def test_rate_limit(self):
        body = json.dumps({"message": "Too many requests"})
        err = parse_bedrock_error(429, body)
        assert err["type"] == "rate_limit_error"

    def test_invalid_request(self):
        body = json.dumps({"message": "max_tokens too large"})
        err = parse_bedrock_error(400, body)
        assert err["type"] == "invalid_request_error"
        assert err["message"] == "max_tokens too large"

    def test_overloaded(self):
        body = json.dumps({"message": "Overloaded"})
        err = parse_bedrock_error(529, body)
        assert err["type"] == "overloaded_error"

    def test_empty_body(self):
        err = parse_bedrock_error(500, "")
        assert err["message"] == ""
