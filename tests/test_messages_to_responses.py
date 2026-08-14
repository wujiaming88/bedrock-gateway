"""
Unit tests for the inbound Anthropic Messages ⇄ OpenAI Responses translation
(``bedrock_gateway.messages_to_responses``).

Two layers:

  * **Translation correctness** — request/response field mapping across text,
    images, tools, tool results, tool choice, stop reasons, usage.
  * **Anthropic Messages protocol conformance** — every streamed sequence is
    fed through :func:`assert_valid_anthropic_stream`, a strict validator that
    enforces the wire contract Claude Code depends on: correct event ordering
    (message_start → per-block start/delta/stop → message_delta → message_stop),
    monotonic block lifecycle, valid ``stop_reason`` values, and matching
    ``input_json_delta`` framing for tool calls. A stream that merely "runs" but
    violates ordering fails here.
"""

from __future__ import annotations

import json

import pytest

from bedrock_gateway.messages_to_responses import (
    AnthropicStreamAdapter,
    _image_block_to_url,
    _parse_sse_block,
    _system_to_instructions,
    responses_error_to_anthropic,
    responses_usage_to_anthropic,
    _text_image_part,
    _tool_choice_to_responses,
    _tool_result_to_text,
    to_anthropic_response,
    to_responses_request,
)

VALID_STOP_REASONS = {
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
}


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------


def parse_frames(frames: list[str]) -> list[tuple[str, dict]]:
    """Parse a list of Anthropic SSE strings into ``(event, data)`` tuples."""
    out: list[tuple[str, dict]] = []
    for frame in frames:
        assert frame.endswith("\n\n"), f"SSE frame not blank-line terminated: {frame!r}"
        lines = frame.strip("\n").split("\n")
        assert lines[0].startswith("event: "), f"missing event line: {frame!r}"
        event = lines[0][len("event: "):]
        data_line = lines[1]
        assert data_line.startswith("data: "), f"missing data line: {frame!r}"
        data = json.loads(data_line[len("data: "):])
        # The data payload's own "type" must agree with the SSE event name.
        assert data.get("type") == event, (
            f"event/type mismatch: event={event} type={data.get('type')}"
        )
        out.append((event, data))
    return out


def assert_valid_anthropic_stream(frames: list[str]) -> list[tuple[str, dict]]:
    """Assert a streamed frame list obeys the Anthropic Messages event contract.

    Returns the parsed events so callers can make further specific assertions.
    """
    events = parse_frames(frames)
    assert events, "empty stream"

    # An error frame is a legal terminator on its own.
    if any(e == "error" for e, _ in events):
        # error must be last and well-formed.
        assert events[-1][0] == "error"
        err = events[-1][1]["error"]
        assert "type" in err and "message" in err
        return events

    types = [e for e, _ in events]
    assert types[0] == "message_start", f"first event must be message_start, got {types[0]}"
    assert types[-1] == "message_stop", f"last event must be message_stop, got {types[-1]}"

    # Exactly one message_start, one message_delta, one message_stop.
    assert types.count("message_start") == 1
    assert types.count("message_delta") == 1, "exactly one message_delta expected"
    assert types.count("message_stop") == 1

    # message_delta must precede message_stop and follow all block events.
    md = types.index("message_delta")
    ms = types.index("message_stop")
    assert md < ms

    # message_start payload shape.
    start_msg = events[0][1]["message"]
    for field in ("id", "type", "role", "content", "model", "usage"):
        assert field in start_msg, f"message_start.message missing {field}"
    assert start_msg["role"] == "assistant"
    assert start_msg["type"] == "message"

    # Block lifecycle: track open/closed per index; deltas only while open.
    open_blocks: set[int] = set()
    started_blocks: set[int] = set()
    for i, (event, data) in enumerate(events):
        if event == "content_block_start":
            idx = data["index"]
            assert idx not in started_blocks, f"index {idx} started twice"
            assert i < md, "content_block_start after message_delta"
            started_blocks.add(idx)
            open_blocks.add(idx)
            cb = data["content_block"]
            assert cb["type"] in ("text", "tool_use")
            if cb["type"] == "tool_use":
                assert "id" in cb and "name" in cb
        elif event == "content_block_delta":
            idx = data["index"]
            assert idx in open_blocks, f"delta for non-open block {idx}"
            d = data["delta"]
            assert d["type"] in ("text_delta", "input_json_delta")
            if d["type"] == "text_delta":
                assert "text" in d
            else:
                assert "partial_json" in d
        elif event == "content_block_stop":
            idx = data["index"]
            assert idx in open_blocks, f"stop for non-open block {idx}"
            open_blocks.discard(idx)

    assert not open_blocks, f"blocks left open: {open_blocks}"

    # message_delta shape: valid stop_reason + usage.
    delta_evt = events[md][1]
    stop_reason = delta_evt["delta"]["stop_reason"]
    assert stop_reason in VALID_STOP_REASONS, f"invalid stop_reason {stop_reason}"
    assert "usage" in delta_evt and "output_tokens" in delta_evt["usage"]

    return events


def run_events(adapter: AnthropicStreamAdapter, seq: list[tuple[str, dict]]) -> list[str]:
    """Drive the adapter's pure ``on_event`` over a sequence, collecting frames."""
    frames: list[str] = []
    for event_type, data in seq:
        frames.extend(adapter.on_event(event_type, data))
    return frames


# ===========================================================================
# Request translation
# ===========================================================================


class TestRequestTranslation:
    def test_minimal(self):
        out = to_responses_request(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50},
            "openai.gpt-5.5",
        )
        assert out["model"] == "openai.gpt-5.5"
        assert out["max_output_tokens"] == 50
        assert out["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
        assert "instructions" not in out

    def test_system_string(self):
        out = to_responses_request(
            {"system": "be brief", "messages": [], "max_tokens": 10}, "m"
        )
        assert out["instructions"] == "be brief"

    def test_system_block_array(self):
        out = to_responses_request(
            {
                "system": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2", "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [],
                "max_tokens": 10,
            },
            "m",
        )
        assert out["instructions"] == "line1\nline2"

    def test_system_empty_omitted(self):
        out = to_responses_request({"system": "", "messages": [], "max_tokens": 1}, "m")
        assert "instructions" not in out

    def test_assistant_text_uses_output_text(self):
        out = to_responses_request(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "max_tokens": 10,
            },
            "m",
        )
        assert out["input"][0]["content"][0]["type"] == "input_text"
        assert out["input"][1]["content"][0]["type"] == "output_text"

    def test_image_base64(self):
        out = to_responses_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            "m",
        )
        parts = out["input"][0]["content"]
        assert parts[0] == {"type": "input_text", "text": "what is this"}
        assert parts[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,AAAA",
        }

    def test_image_url(self):
        out = to_responses_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://x/y.png"},
                            }
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            "m",
        )
        assert out["input"][0]["content"][0] == {
            "type": "input_image",
            "image_url": "https://x/y.png",
        }

    def test_tool_use_becomes_function_call_item(self):
        out = to_responses_request(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "let me check"},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"city": "NYC"},
                            },
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            "m",
        )
        items = out["input"]
        # text message first, then a standalone function_call item, in order
        assert items[0]["role"] == "assistant"
        assert items[0]["content"][0]["text"] == "let me check"
        assert items[1] == {
            "type": "function_call",
            "call_id": "toolu_1",
            "name": "get_weather",
            "arguments": json.dumps({"city": "NYC"}),
        }

    def test_tool_result_becomes_function_call_output(self):
        out = to_responses_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "72F sunny",
                            }
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            "m",
        )
        assert out["input"][0] == {
            "type": "function_call_output",
            "call_id": "toolu_1",
            "output": "72F sunny",
        }

    def test_tool_result_block_list_content(self):
        out = to_responses_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [
                                    {"type": "text", "text": "part1"},
                                    {"type": "text", "text": "part2"},
                                ],
                            }
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            "m",
        )
        assert out["input"][0]["output"] == "part1part2"

    def test_tools_translated(self):
        out = to_responses_request(
            {
                "messages": [],
                "max_tokens": 10,
                "tools": [
                    {
                        "name": "search",
                        "description": "find things",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    }
                ],
            },
            "m",
        )
        assert out["tools"] == [
            {
                "type": "function",
                "name": "search",
                "description": "find things",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]

    @pytest.mark.parametrize(
        "anthropic_tc,expected",
        [
            ({"type": "auto"}, "auto"),
            ({"type": "any"}, "required"),
            ({"type": "none"}, "none"),
            ({"type": "tool", "name": "search"}, {"type": "function", "name": "search"}),
        ],
    )
    def test_tool_choice(self, anthropic_tc, expected):
        out = to_responses_request(
            {
                "messages": [],
                "max_tokens": 10,
                "tools": [{"name": "search", "input_schema": {}}],
                "tool_choice": anthropic_tc,
            },
            "m",
        )
        assert out["tool_choice"] == expected

    def test_tool_choice_without_tools_absent(self):
        out = to_responses_request(
            {"messages": [], "max_tokens": 10, "tool_choice": {"type": "auto"}}, "m"
        )
        assert "tool_choice" not in out
        assert "tools" not in out

    def test_temperature_top_p_passthrough(self):
        out = to_responses_request(
            {"messages": [], "max_tokens": 10, "temperature": 0.3, "top_p": 0.9}, "m"
        )
        assert out["temperature"] == 0.3
        assert out["top_p"] == 0.9

    def test_thinking_dropped(self):
        # v1 scope: thinking is not forwarded (no signature round-trip).
        out = to_responses_request(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm", "signature": "x"},
                            {"type": "text", "text": "answer"},
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            "m",
        )
        parts = out["input"][0]["content"]
        assert len(parts) == 1
        assert parts[0]["text"] == "answer"

    def test_string_content_roundtrips(self):
        out = to_responses_request(
            {"messages": [{"role": "user", "content": "plain"}], "max_tokens": 5}, "m"
        )
        assert out["input"][0]["content"][0]["text"] == "plain"


# ===========================================================================
# Sync response translation
# ===========================================================================


class TestUsageAndErrorMapping:
    @pytest.mark.parametrize(
        "usage,expected",
        [
            (None, {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
            ({"input_tokens": 100, "output_tokens": 7}, {"input_tokens": 100, "output_tokens": 7, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
            ({"input_tokens": 100, "input_tokens_details": {"cached_tokens": 30}, "output_tokens": 7}, {"input_tokens": 70, "output_tokens": 7, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 30}),
            ({"input_tokens": 10, "input_tokens_details": {"cached_tokens": 30}}, {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 10}),
            ({"input_tokens": "bad", "input_tokens_details": {"cached_tokens": -3}, "output_tokens": True}, {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
        ],
    )
    def test_usage_mapping(self, usage, expected):
        assert responses_usage_to_anthropic(usage) == expected

    @pytest.mark.parametrize(
        "error,status,etype",
        [
            ({"code": "context_length_exceeded", "message": "maximum context"}, 400, "invalid_request_error"),
            ({"message": "Input is too long for requested model"}, 400, "invalid_request_error"),
            ({"type": "rate_limit_error", "message": "busy"}, 429, "rate_limit_error"),
            ({"type": "authentication_error", "message": "bad key"}, 401, "authentication_error"),
            ({"type": "permission_error", "message": "denied"}, 403, "permission_error"),
            ("broken", 502, "api_error"),
        ],
    )
    def test_error_mapping(self, error, status, etype):
        actual_status, actual_type, _ = responses_error_to_anthropic(error)
        assert (actual_status, actual_type) == (status, etype)


class TestSyncResponseTranslation:
    def test_text_response(self):
        result = to_anthropic_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            "gpt-5.5",
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "gpt-5.5"
        assert result["content"] == [{"type": "text", "text": "hello"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["id"].startswith("msg_")

    def test_tool_use_response(self):
        result = to_anthropic_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"NYC"}',
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 12},
            },
            "gpt-5.5",
        )
        assert result["stop_reason"] == "tool_use"
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_1"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "NYC"}

    def test_mixed_text_and_tool(self):
        result = to_anthropic_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "checking"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "f",
                        "arguments": "{}",
                    },
                ],
                "usage": {},
            },
            "m",
        )
        assert [b["type"] for b in result["content"]] == ["text", "tool_use"]
        assert result["stop_reason"] == "tool_use"

    def test_stop_reason_max_tokens(self):
        result = to_anthropic_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "x"}]}
                ],
                "usage": {},
            },
            "m",
        )
        assert result["stop_reason"] == "max_tokens"

    def test_stop_reason_content_filter_maps_end_turn(self):
        result = to_anthropic_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [],
                "usage": {},
            },
            "m",
        )
        assert result["stop_reason"] == "end_turn"

    def test_bad_tool_arguments_default_empty(self):
        result = to_anthropic_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "c",
                        "name": "f",
                        "arguments": "not-json",
                    }
                ],
                "usage": {},
            },
            "m",
        )
        assert result["content"][0]["input"] == {}

    def test_usage_defaults_zero(self):
        result = to_anthropic_response(
            {"status": "completed", "output": []}, "m"
        )
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0
        assert result["stop_reason"] == "end_turn"

    def test_non_dict_output_item_skipped(self):
        result = to_anthropic_response(
            {
                "status": "completed",
                "output": [
                    "junk",
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
                ],
                "usage": {},
            },
            "m",
        )
        assert result["content"] == [{"type": "text", "text": "ok"}]


# ===========================================================================
# Streaming protocol conformance
# ===========================================================================


class TestStreamProtocol:
    def _created(self, input_tokens=3):
        return (
            "response.created",
            {"type": "response.created", "response": {"usage": {"input_tokens": input_tokens}}},
        )

    def test_text_stream_conformance(self):
        adapter = AnthropicStreamAdapter("gpt-5.5", "msg_abc")
        seq = [
            self._created(),
            (
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": 0,
                 "item": {"type": "message"}},
            ),
            (
                "response.content_part.added",
                {"type": "response.content_part.added", "output_index": 0,
                 "content_index": 0, "part": {"type": "output_text"}},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "output_index": 0,
                 "content_index": 0, "delta": "Hel"},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "output_index": 0,
                 "content_index": 0, "delta": "lo"},
            ),
            (
                "response.content_part.done",
                {"type": "response.content_part.done", "output_index": 0,
                 "content_index": 0},
            ),
            (
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": 0},
            ),
            (
                "response.completed",
                {"type": "response.completed",
                 "response": {"status": "completed",
                              "usage": {"input_tokens": 3, "output_tokens": 2}}},
            ),
        ]
        frames = run_events(adapter, seq)
        events = assert_valid_anthropic_stream(frames)
        # Text deltas assembled correctly
        text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"]["type"] == "text_delta"
        )
        assert text == "Hello"
        # usage propagated to message_delta
        md = next(d for e, d in events if e == "message_delta")
        assert md["usage"]["input_tokens"] == 3
        assert md["usage"]["output_tokens"] == 2
        assert md["delta"]["stop_reason"] == "end_turn"

    def test_terminal_usage_overrides_zero_start_and_maps_cache(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = run_events(adapter, [
            ("response.created", {"type": "response.created", "response": {}}),
            ("response.completed", {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {"cached_tokens": 30},
                        "output_tokens": 9,
                    },
                },
            }),
        ])
        events = assert_valid_anthropic_stream(frames)
        start = next(data for event, data in events if event == "message_start")
        assert start["message"]["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        delta = next(data for event, data in events if event == "message_delta")
        assert delta["usage"] == {
            "input_tokens": 70,
            "output_tokens": 9,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 30,
        }

    def test_tool_call_stream_conformance(self):
        adapter = AnthropicStreamAdapter("gpt-5.5", "msg_1")
        seq = [
            self._created(),
            (
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": 0,
                 "item": {"type": "function_call", "call_id": "call_9",
                          "name": "get_weather"}},
            ),
            (
                "response.function_call_arguments.delta",
                {"type": "response.function_call_arguments.delta",
                 "output_index": 0, "delta": '{"ci'},
            ),
            (
                "response.function_call_arguments.delta",
                {"type": "response.function_call_arguments.delta",
                 "output_index": 0, "delta": 'ty":"NYC"}'},
            ),
            (
                "response.function_call_arguments.done",
                {"type": "response.function_call_arguments.done",
                 "output_index": 0, "arguments": '{"city":"NYC"}'},
            ),
            (
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": 0},
            ),
            (
                "response.completed",
                {"type": "response.completed",
                 "response": {"status": "completed",
                              "usage": {"input_tokens": 5, "output_tokens": 7}}},
            ),
        ]
        frames = run_events(adapter, seq)
        events = assert_valid_anthropic_stream(frames)
        # tool_use block opened with id+name
        start = next(
            d for e, d in events
            if e == "content_block_start" and d["content_block"]["type"] == "tool_use"
        )
        assert start["content_block"]["id"] == "call_9"
        assert start["content_block"]["name"] == "get_weather"
        # input_json_delta reassembles to the full arguments
        partial = "".join(
            d["delta"]["partial_json"]
            for e, d in events
            if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(partial) == {"city": "NYC"}
        # stop_reason is tool_use
        md = next(d for e, d in events if e == "message_delta")
        assert md["delta"]["stop_reason"] == "tool_use"

    def test_tool_args_whole_no_deltas(self):
        # If arguments arrive only in the .done event (no deltas), emit one frame.
        adapter = AnthropicStreamAdapter("m", "msg_1")
        seq = [
            self._created(),
            (
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": 0,
                 "item": {"type": "function_call", "call_id": "c", "name": "f"}},
            ),
            (
                "response.function_call_arguments.done",
                {"type": "response.function_call_arguments.done",
                 "output_index": 0, "arguments": '{"a":1}'},
            ),
            (
                "response.completed",
                {"type": "response.completed", "response": {"status": "completed"}},
            ),
        ]
        frames = run_events(adapter, seq)
        events = assert_valid_anthropic_stream(frames)
        partial = "".join(
            d["delta"]["partial_json"]
            for e, d in events
            if e == "content_block_delta"
        )
        assert json.loads(partial) == {"a": 1}

    def test_lazy_open_on_delta_without_part_added(self):
        # Some upstreams skip content_part.added; a text delta must still open
        # a block first (conformance must hold).
        adapter = AnthropicStreamAdapter("m", "msg_1")
        seq = [
            self._created(),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "output_index": 0,
                 "content_index": 0, "delta": "hi"},
            ),
            (
                "response.completed",
                {"type": "response.completed", "response": {"status": "completed"}},
            ),
        ]
        frames = run_events(adapter, seq)
        assert_valid_anthropic_stream(frames)

    def test_multiple_text_blocks(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        seq = [
            self._created(),
            ("response.content_part.added",
             {"type": "response.content_part.added", "output_index": 0,
              "content_index": 0, "part": {"type": "output_text"}}),
            ("response.output_text.delta",
             {"type": "response.output_text.delta", "output_index": 0,
              "content_index": 0, "delta": "a"}),
            ("response.content_part.done",
             {"type": "response.content_part.done", "output_index": 0,
              "content_index": 0}),
            ("response.content_part.added",
             {"type": "response.content_part.added", "output_index": 1,
              "content_index": 0, "part": {"type": "output_text"}}),
            ("response.output_text.delta",
             {"type": "response.output_text.delta", "output_index": 1,
              "content_index": 0, "delta": "b"}),
            ("response.content_part.done",
             {"type": "response.content_part.done", "output_index": 1,
              "content_index": 0}),
            ("response.completed",
             {"type": "response.completed", "response": {"status": "completed"}}),
        ]
        frames = run_events(adapter, seq)
        events = assert_valid_anthropic_stream(frames)
        starts = [d["index"] for e, d in events if e == "content_block_start"]
        assert starts == [0, 1]

    def test_incomplete_max_tokens_stop_reason(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        seq = [
            self._created(),
            ("response.output_text.delta",
             {"type": "response.output_text.delta", "output_index": 0,
              "content_index": 0, "delta": "x"}),
            ("response.incomplete",
             {"type": "response.incomplete",
              "response": {"status": "incomplete",
                           "incomplete_details": {"reason": "max_output_tokens"}}}),
        ]
        frames = run_events(adapter, seq)
        events = assert_valid_anthropic_stream(frames)
        md = next(d for e, d in events if e == "message_delta")
        assert md["delta"]["stop_reason"] == "max_tokens"

    def test_error_event_terminates(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = run_events(
            adapter,
            [
                self._created(),
                ("response.failed",
                 {"type": "response.failed",
                  "response": {"error": {"message": "boom"}}}),
            ],
        )
        events = assert_valid_anthropic_stream(frames)
        assert events[-1][0] == "error"
        assert events[-1][1]["error"]["message"] == "boom"

    def test_failed_is_idempotent_and_ignores_late_completed(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        first = adapter.on_event("response.failed", {
            "response": {"error": {"code": "context_length_exceeded", "message": "context exceeded"}}
        })
        assert parse_frames(first)[0][1]["error"]["type"] == "invalid_request_error"
        assert adapter.terminal is True
        assert adapter.on_event("response.failed", {"error": "again"}) == []
        assert adapter.on_event("response.completed", {"response": {"status": "completed"}}) == []
        assert adapter.finalize_on_disconnect() == []
        assert adapter.terminate_error("late") == []

    def test_top_level_error_message(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = adapter.on_event("error", {"message": "top-level failure"})
        assert parse_frames(frames)[0][1]["error"]["message"] == "top-level failure"

    def test_ping_and_unknown_events_ignored(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        seq = [
            self._created(),
            ("response.in_progress", {"type": "response.in_progress"}),
            ("ping", {"type": "ping"}),
            ("response.output_text.done",
             {"type": "response.output_text.done", "output_index": 0,
              "content_index": 0}),
            ("response.completed",
             {"type": "response.completed", "response": {"status": "completed"}}),
        ]
        frames = run_events(adapter, seq)
        # Only message_start + message_delta + message_stop (no content blocks)
        events = assert_valid_anthropic_stream(frames)
        types = [e for e, _ in events]
        assert types == ["message_start", "message_delta", "message_stop"]

    def test_double_finalize_is_idempotent(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        run_events(adapter, [self._created()])
        first = adapter.on_event(
            "response.completed",
            {"type": "response.completed", "response": {"status": "completed"}},
        )
        second = adapter.on_event(
            "response.completed",
            {"type": "response.completed", "response": {"status": "completed"}},
        )
        assert first  # emitted terminator
        assert second == []  # no duplicate terminator

    def test_finalize_on_disconnect(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = run_events(adapter, [
            self._created(),
            ("response.output_text.delta",
             {"type": "response.output_text.delta", "output_index": 0,
              "content_index": 0, "delta": "partial"}),
        ])
        # Upstream cut off — no terminal event. It must end as an error, not a
        # fabricated successful end_turn/message_stop.
        frames += adapter.finalize_on_disconnect()
        events = assert_valid_anthropic_stream(frames)
        assert events[-1][0] == "error"
        assert not any(event == "message_stop" for event, _ in events)

    def test_disconnect_before_start_is_error(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = adapter.finalize_on_disconnect()
        assert parse_frames(frames)[0][0] == "error"


# ===========================================================================
# Async byte-stream driver + SSE parsing
# ===========================================================================


class TestStreamDriver:
    @pytest.mark.asyncio
    async def test_translate_over_bytes(self):
        raw = [
            b'event: response.created\ndata: {"type":"response.created","response":{"usage":{"input_tokens":2}}}\n\n',
            b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi"}\n\n',
            b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"output_tokens":1}}}\n\n',
        ]

        async def byte_iter():
            for chunk in raw:
                yield chunk

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        assert_valid_anthropic_stream(frames)

    @pytest.mark.asyncio
    async def test_translate_split_across_chunks(self):
        # An SSE frame split mid-way across two byte chunks must still parse.
        full = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"OK"}\n\n'
        created = b'event: response.created\ndata: {"type":"response.created"}\n\n'
        completed = b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        split = len(full) // 2

        async def byte_iter():
            yield created
            yield full[:split]
            yield full[split:]
            yield completed

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        events = assert_valid_anthropic_stream(frames)
        text = "".join(
            d["delta"]["text"] for e, d in events if e == "content_block_delta"
        )
        assert text == "OK"

    @pytest.mark.asyncio
    async def test_translate_cjk_split_bytes(self):
        # Multi-byte UTF-8 (CJK) split across chunk boundary must not corrupt.
        payload = '你好'
        frame = (
            'event: response.output_text.delta\ndata: '
            + json.dumps({"type": "response.output_text.delta", "output_index": 0,
                          "content_index": 0, "delta": payload}, ensure_ascii=False)
            + '\n\n'
        ).encode("utf-8")
        created = b'event: response.created\ndata: {"type":"response.created"}\n\n'
        completed = b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        # split inside the multi-byte sequence
        cut = frame.index(payload.encode("utf-8")) + 1

        async def byte_iter():
            yield created
            yield frame[:cut]
            yield frame[cut:]
            yield completed

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        events = assert_valid_anthropic_stream(frames)
        text = "".join(
            d["delta"]["text"] for e, d in events if e == "content_block_delta"
        )
        assert text == "你好"

    def test_parse_sse_block_event_and_data(self):
        et, data = _parse_sse_block('event: foo\ndata: {"type":"foo","x":1}')
        assert et == "foo"
        assert data == {"type": "foo", "x": 1}

    def test_parse_sse_block_type_from_payload(self):
        # No event: line → fall back to the JSON payload's type.
        et, data = _parse_sse_block('data: {"type":"response.created"}')
        assert et == "response.created"

    def test_parse_sse_block_bad_json(self):
        et, data = _parse_sse_block('event: x\ndata: not-json')
        assert et == "x"
        assert data == {}

    def test_parse_sse_block_empty(self):
        et, data = _parse_sse_block('')
        assert et is None
        assert data == {}

    @pytest.mark.asyncio
    async def test_translate_ignores_empty_chunks(self):
        async def byte_iter():
            yield b""  # empty chunk skipped
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        assert_valid_anthropic_stream(frames)

    @pytest.mark.asyncio
    async def test_translate_tail_block_without_blank_line(self):
        # Final SSE block arrives without a trailing blank line — must still flush.
        async def byte_iter():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}'

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        events = assert_valid_anthropic_stream(frames)
        assert events[-1][0] == "message_stop"

    @pytest.mark.asyncio
    async def test_translate_disconnect_emits_terminator(self):
        # Stream ends mid-content with no terminal event → driver must emit the
        # disconnect closer (finalize_on_disconnect path).
        async def byte_iter():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hi"}\n\n'
            # no completed event

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        events = assert_valid_anthropic_stream(frames)
        assert events[-1][0] == "error"
        assert not any(event == "message_stop" for event, _ in events)

    @pytest.mark.asyncio
    async def test_translate_untyped_midstream_block_skipped(self):
        # A keepalive/comment block (no event: or data: lines) mid-stream must
        # be skipped by the main loop, not crash it.
        async def byte_iter():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield b': keepalive comment\n\n'  # untyped block WITH terminator
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        assert_valid_anthropic_stream(frames)

    @pytest.mark.asyncio
    async def test_translate_untyped_tail_block_skipped(self):
        # Tail block that parses to no event type must be skipped (not crash).
        async def byte_iter():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            yield b'data: 12345'  # non-dict payload → event_type None

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        assert_valid_anthropic_stream(frames)

    @pytest.mark.asyncio
    async def test_translate_blank_tail_ignored(self):
        # Whitespace-only tail after the terminal event must not re-trigger.
        async def byte_iter():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            yield b'   \n'

        adapter = AnthropicStreamAdapter("m", "msg_1")
        frames = [f async for f in adapter.translate(byte_iter())]
        events = assert_valid_anthropic_stream(frames)
        assert events.count(("message_stop", {"type": "message_stop"})) == 1


class TestEdgeCases:
    """Precise coverage of defensive branches."""

    def test_system_none(self):
        assert _system_to_instructions(None) is None

    def test_system_unknown_type(self):
        assert _system_to_instructions(123) is None

    def test_system_list_with_plain_strings(self):
        assert _system_to_instructions(["a", {"type": "text", "text": "b"}]) == "a\nb"

    def test_system_list_empty(self):
        assert _system_to_instructions([]) is None

    def test_image_block_unknown_source_type(self):
        assert _image_block_to_url({"type": "file"}) is None

    def test_image_block_url_missing(self):
        assert _image_block_to_url({"type": "url"}) is None

    def test_text_image_part_empty_string(self):
        assert _text_image_part("", "input_text") is None

    def test_text_image_part_non_dict_non_str(self):
        assert _text_image_part(42, "input_text") == {"type": "input_text", "text": "42"}

    def test_text_image_part_falsy_non_dict(self):
        assert _text_image_part(0, "input_text") is None

    def test_text_image_part_image_no_url(self):
        assert _text_image_part({"type": "image", "source": {}}, "input_text") is None

    def test_text_image_part_unknown_block(self):
        assert _text_image_part({"type": "thinking"}, "input_text") is None

    def test_tool_result_none(self):
        assert _tool_result_to_text(None) == ""

    def test_tool_result_non_text_block_json_encoded(self):
        out = _tool_result_to_text([{"type": "image", "x": 1}])
        assert json.loads(out) == {"type": "image", "x": 1}

    def test_tool_result_plain_items(self):
        assert _tool_result_to_text([1, 2]) == "12"

    def test_tool_result_scalar(self):
        assert _tool_result_to_text(42) == "42"

    def test_tool_choice_not_dict(self):
        assert _tool_choice_to_responses("auto") is None

    def test_tool_choice_unknown(self):
        assert _tool_choice_to_responses({"type": "weird"}) is None

    def test_tool_choice_tool_without_name(self):
        assert _tool_choice_to_responses({"type": "tool"}) is None

    def test_string_message_content(self):
        # top-level dict content that is not a list nor str
        out = to_responses_request(
            {"messages": [{"role": "user", "content": None}], "max_tokens": 5}, "m"
        )
        # None content → no usable parts → no message item
        assert out["input"] == []

    def test_content_part_added_non_text_ignored(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event(
            "response.content_part.added",
            {"output_index": 0, "content_index": 0, "part": {"type": "refusal"}},
        )
        assert out == []

    def test_output_item_added_non_function_ignored(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event(
            "response.output_item.added",
            {"output_index": 0, "item": {"type": "reasoning"}},
        )
        assert out == []

    def test_function_args_delta_unknown_output_ignored(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event(
            "response.function_call_arguments.delta",
            {"output_index": 9, "delta": "x"},
        )
        assert out == []

    def test_function_args_done_unknown_output_ignored(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event(
            "response.function_call_arguments.done",
            {"output_index": 9, "arguments": "{}"},
        )
        assert out == []

    def test_content_part_done_unknown_ignored(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event(
            "response.content_part.done", {"output_index": 5, "content_index": 5}
        )
        assert out == []

    def test_open_text_block_idempotent(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        first = adapter._open_text_block(0, 0)
        second = adapter._open_text_block(0, 0)
        assert first  # opened
        assert second == []  # already open

    def test_open_tool_block_idempotent(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        first = adapter._open_tool_block(0, "c", "f")
        second = adapter._open_tool_block(0, "c", "f")
        assert first
        assert second == []

    def test_close_index_not_open_noop(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        assert adapter._close_index(99) == []

    def test_error_event_via_top_level_error(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event("error", {"error": {"message": "top-level"}})
        assert len(out) == 1
        assert "top-level" in out[0]

    def test_error_event_no_message_default(self):
        adapter = AnthropicStreamAdapter("m", "msg_1")
        out = adapter.on_event("response.failed", {"response": {}})
        assert "upstream error" in out[0]
