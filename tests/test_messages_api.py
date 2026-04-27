"""
Tests for the Anthropic Messages API endpoint (POST /v1/messages).

Covers sync and streaming responses, thinking/redacted_thinking blocks,
tool use, error handling, retries, and edge cases.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
)
from bedrock_gateway.server import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> GatewayConfig:
    """Test configuration with multiple model types."""
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=2, base_delay=0.01),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-model-v1",
                context_length=200000,
                max_output=4096,
            ),
            "claude-opus-4": ModelEntry(
                bedrock_id="us.anthropic.claude-opus-4-6-v1",
                context_length=1000000,
                max_output=128000,
            ),
        },
    )


@pytest.fixture
def client(config: GatewayConfig) -> TestClient:
    app = create_app(config)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_sync_client(response_data: dict):
    """Create a mocked httpx.AsyncClient for sync (non-streaming) calls."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data

    mock_instance = AsyncMock()
    mock_instance.post = AsyncMock(return_value=mock_response)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    return mock_instance


def _bedrock_text_response(text: str = "Hello from Bedrock!") -> dict:
    """A typical Bedrock text response."""
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }


def _bedrock_thinking_response(
    thinking: str = "Let me reason...",
    answer: str = "The answer is 42.",
    include_redacted: bool = False,
) -> dict:
    """A Bedrock response with thinking blocks."""
    content = [{"type": "thinking", "thinking": thinking}]
    if include_redacted:
        content.append({"type": "redacted_thinking", "data": "encrypted-data"})
    content.append({"type": "text", "text": answer})
    return {
        "content": content,
        "usage": {"input_tokens": 50, "output_tokens": 100},
        "stop_reason": "end_turn",
    }


def _bedrock_tool_use_response() -> dict:
    """A Bedrock response with tool_use."""
    return {
        "content": [
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "toolu_01A",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            },
        ],
        "usage": {"input_tokens": 30, "output_tokens": 20},
        "stop_reason": "tool_use",
    }


def _encode_event(event: dict) -> bytes:
    """Encode a single event as an AWS event-stream binary frame."""
    encoded = base64.b64encode(json.dumps(event).encode()).decode()
    return f'{{"bytes":"{encoded}"}}'.encode()


def _setup_stream_mock(mock_client_cls, events: list[dict]):
    """Set up mock httpx.AsyncClient for streaming responses."""
    raw_chunks = [_encode_event(e) for e in events]

    async def aiter_bytes():
        for chunk in raw_chunks:
            yield chunk

    async def aiter_text():
        yield ""

    mock_stream_response = MagicMock()
    mock_stream_response.status_code = 200
    mock_stream_response.aiter_bytes = aiter_bytes
    mock_stream_response.aiter_text = aiter_text

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_instance = AsyncMock()
    mock_instance.stream = MagicMock(return_value=mock_stream_ctx)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_instance
    return mock_instance


# ---------------------------------------------------------------------------
# POST /v1/messages — Sync
# ---------------------------------------------------------------------------


class TestMessagesSyncBasic:
    """Basic sync (non-streaming) Messages API tests."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_basic_text_response(self, mock_client_cls, client: TestClient):
        """Simple text response returns proper Anthropic format."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "us.anthropic.test-model-v1"
        assert data["stop_reason"] == "end_turn"
        assert data["id"].startswith("msg_")
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hello from Bedrock!"
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 5

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_with_system_prompt(self, mock_client_cls, client: TestClient):
        """System prompt is passed through to Bedrock."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        # Verify system was passed to Bedrock
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["system"] == "You are a helpful assistant."

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_optional_params_passthrough(self, mock_client_cls, client: TestClient):
        """Temperature, top_p, top_k, stop_sequences pass through."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "stop_sequences": ["\n\n"],
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.9
        assert body["top_k"] == 50
        assert body["stop_sequences"] == ["\n\n"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_default_max_tokens(self, mock_client_cls, client: TestClient):
        """When max_tokens is omitted, a default is used."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["max_tokens"] == 4096  # test-model's max_output

    def test_invalid_json(self, client: TestClient):
        """Invalid JSON body returns Anthropic error format."""
        resp = client.post(
            "/v1/messages",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"


class TestMessagesSyncThinking:
    """Thinking / extended thinking in sync Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_thinking_response(self, mock_client_cls, client: TestClient):
        """Thinking blocks are returned in Anthropic native format."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_thinking_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Think about this"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200
        data = resp.json()
        # Anthropic native format returns thinking blocks in content array
        assert len(data["content"]) == 2
        assert data["content"][0]["type"] == "thinking"
        assert data["content"][0]["thinking"] == "Let me reason..."
        assert data["content"][1]["type"] == "text"
        assert data["content"][1]["text"] == "The answer is 42."

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_thinking_with_redacted(self, mock_client_cls, client: TestClient):
        """Redacted thinking blocks are returned in native format."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_thinking_response(include_redacted=True)
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200
        data = resp.json()
        # All blocks are returned as-is from Bedrock
        assert len(data["content"]) == 3
        assert data["content"][0]["type"] == "thinking"
        assert data["content"][1]["type"] == "redacted_thinking"
        assert data["content"][2]["type"] == "text"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_thinking_removes_temperature(self, mock_client_cls, client: TestClient):
        """Temperature is removed when thinking is enabled."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_thinking_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Think"}],
            "temperature": 0.7,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert "temperature" not in body
        assert body["thinking"]["budget_tokens"] == 2048

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_thinking_budget_clamp(self, mock_client_cls, client: TestClient):
        """Budget tokens below 1024 are clamped to 1024."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_thinking_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "enabled", "budget_tokens": 128},
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["thinking"]["budget_tokens"] == 1024


class TestMessagesSyncToolUse:
    """Tool use in sync Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_tool_use_response(self, mock_client_cls, client: TestClient):
        """Tool use response is returned in native Anthropic format."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_tool_use_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "tool_use"
        assert len(data["content"]) == 2
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Let me check."
        assert data["content"][1]["type"] == "tool_use"
        assert data["content"][1]["name"] == "get_weather"
        assert data["content"][1]["input"] == {"city": "Tokyo"}

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_tools_passthrough(self, mock_client_cls, client: TestClient):
        """Tools and tool_choice pass through to Bedrock as-is."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        tools = [{
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {}},
        }]
        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Search"}],
            "tools": tools,
            "tool_choice": {"type": "auto"},
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["tools"] == tools
        assert body["tool_choice"] == {"type": "auto"}


class TestMessagesSyncErrors:
    """Error handling in sync Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bedrock_error(self, mock_client_cls, client: TestClient):
        """Bedrock errors return Anthropic error format."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = json.dumps({"message": "max_tokens too large"})

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["message"] == "max_tokens too large"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_retry_on_429(self, mock_sleep, mock_client_cls, client: TestClient):
        """429 triggers retry, then succeeds."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = _bedrock_text_response()

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=[mock_429, mock_200])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 200
        assert mock_instance.post.call_count == 2

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_all_retries_exhausted(self, mock_sleep, mock_client_cls, client: TestClient):
        """All retries exhausted returns 502 in Anthropic format."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_429)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 502
        data = resp.json()
        assert data["type"] == "error"
        assert "retries failed" in data["error"]["message"]


# ---------------------------------------------------------------------------
# POST /v1/messages — Streaming
# ---------------------------------------------------------------------------


class TestMessagesStreamBasic:
    """Streaming Messages API tests."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_text(self, mock_client_cls, client: TestClient):
        """Streaming text produces proper Anthropic SSE events."""
        events = [
            {"type": "message_start", "message": {
                "usage": {"input_tokens": 10},
            }},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": " World"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta",
             "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        _setup_stream_mock(mock_client_cls, events)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events
        lines = resp.text.strip().split("\n")
        sse_events = []
        current_event = {}
        for line in lines:
            if line.startswith("event: "):
                current_event["event"] = line[7:]
            elif line.startswith("data: "):
                current_event["data"] = json.loads(line[6:])
                sse_events.append(current_event)
                current_event = {}

        # Verify event types
        event_types = [e["event"] for e in sse_events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        # Verify message_start enrichment
        msg_start = next(e for e in sse_events if e["event"] == "message_start")
        msg = msg_start["data"]["message"]
        assert msg["id"].startswith("msg_")
        assert msg["model"] == "us.anthropic.test-model-v1"
        assert msg["role"] == "assistant"
        assert msg["type"] == "message"

        # Verify text deltas
        text_deltas = [
            e["data"]["delta"]["text"]
            for e in sse_events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["Hello", " World"]


class TestMessagesStreamThinking:
    """Streaming thinking in Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_thinking_blocks(self, mock_client_cls, client: TestClient):
        """Thinking events are streamed through in native Anthropic format."""
        events = [
            {"type": "message_start", "message": {
                "usage": {"input_tokens": 20},
            }},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "Step 1: "}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "analyze."}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "signature_delta", "signature": "sig123"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "text_delta", "text": "The answer."}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta",
             "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 50}},
            {"type": "message_stop"},
        ]
        _setup_stream_mock(mock_client_cls, events)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Think step by step"}],
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200

        # Parse SSE events
        lines = resp.text.strip().split("\n")
        sse_events = []
        current_event = {}
        for line in lines:
            if line.startswith("event: "):
                current_event["event"] = line[7:]
            elif line.startswith("data: "):
                current_event["data"] = json.loads(line[6:])
                sse_events.append(current_event)
                current_event = {}

        # Verify thinking block events pass through
        thinking_deltas = [
            e["data"]["delta"]
            for e in sse_events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "thinking_delta"
        ]
        assert len(thinking_deltas) == 2
        assert thinking_deltas[0]["thinking"] == "Step 1: "
        assert thinking_deltas[1]["thinking"] == "analyze."

        # Verify signature delta passes through
        sig_deltas = [
            e for e in sse_events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "signature_delta"
        ]
        assert len(sig_deltas) == 1

        # Verify text delta
        text_deltas = [
            e["data"]["delta"]["text"]
            for e in sse_events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["The answer."]


class TestMessagesStreamToolUse:
    """Streaming tool use in Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_tool_use(self, mock_client_cls, client: TestClient):
        """Tool use events are streamed through in native format."""
        events = [
            {"type": "message_start", "message": {
                "usage": {"input_tokens": 15},
            }},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Let me check."}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {
                 "type": "tool_use",
                 "id": "toolu_01A",
                 "name": "get_weather",
             }},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta",
                       "partial_json": '{"city":'}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta",
                       "partial_json": '"Tokyo"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta",
             "delta": {"stop_reason": "tool_use"},
             "usage": {"output_tokens": 25}},
            {"type": "message_stop"},
        ]
        _setup_stream_mock(mock_client_cls, events)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Weather?"}],
            "stream": True,
        })

        assert resp.status_code == 200

        # Parse SSE events
        lines = resp.text.strip().split("\n")
        sse_events = []
        current_event = {}
        for line in lines:
            if line.startswith("event: "):
                current_event["event"] = line[7:]
            elif line.startswith("data: "):
                current_event["data"] = json.loads(line[6:])
                sse_events.append(current_event)
                current_event = {}

        # Verify tool_use block start
        tool_starts = [
            e for e in sse_events
            if e["event"] == "content_block_start"
            and e["data"].get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["data"]["content_block"]["name"] == "get_weather"

        # Verify input_json_delta
        json_deltas = [
            e["data"]["delta"]["partial_json"]
            for e in sse_events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) == 2

        # Verify message_delta with tool_use stop_reason
        msg_delta = [
            e for e in sse_events if e["event"] == "message_delta"
        ]
        assert len(msg_delta) == 1
        assert msg_delta[0]["data"]["delta"]["stop_reason"] == "tool_use"


class TestMessagesStreamErrors:
    """Error handling in streaming Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_bedrock_error(self, mock_client_cls, client: TestClient):
        """Bedrock errors in streaming return Anthropic error SSE."""
        async def aiter_text():
            yield json.dumps({"message": "Model not found"})

        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 404
        mock_stream_response.aiter_text = aiter_text

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_instance = AsyncMock()
        mock_instance.stream = MagicMock(return_value=mock_stream_ctx)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        assert resp.status_code == 200  # SSE stream always returns 200

        # Parse error event
        lines = resp.text.strip().split("\n")
        error_events = []
        current_event = {}
        for line in lines:
            if line.startswith("event: "):
                current_event["event"] = line[7:]
            elif line.startswith("data: "):
                current_event["data"] = json.loads(line[6:])
                error_events.append(current_event)
                current_event = {}

        error_evts = [e for e in error_events if e["event"] == "error"]
        assert len(error_evts) == 1
        assert error_evts[0]["data"]["error"]["type"] == "not_found_error"


class TestMessagesStreamPing:
    """Ping events in streaming."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_ping_event(self, mock_client_cls, client: TestClient):
        """Ping events are forwarded."""
        events = [
            {"type": "ping"},
            {"type": "message_start", "message": {
                "usage": {"input_tokens": 5},
            }},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta",
             "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        _setup_stream_mock(mock_client_cls, events)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        sse_events = []
        current_event = {}
        for line in lines:
            if line.startswith("event: "):
                current_event["event"] = line[7:]
            elif line.startswith("data: "):
                current_event["data"] = json.loads(line[6:])
                sse_events.append(current_event)
                current_event = {}

        ping_events = [e for e in sse_events if e["event"] == "ping"]
        assert len(ping_events) == 1


# ---------------------------------------------------------------------------
# Converter function tests
# ---------------------------------------------------------------------------


class TestFormatAnthropicResponse:
    """Tests for format_anthropic_response."""

    def test_basic_format(self):
        from bedrock_gateway.converter import format_anthropic_response
        result = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        formatted = format_anthropic_response(result, "us.anthropic.test-v1")
        assert formatted["type"] == "message"
        assert formatted["role"] == "assistant"
        assert formatted["model"] == "us.anthropic.test-v1"
        assert formatted["id"].startswith("msg_")
        assert formatted["content"] == result["content"]
        assert formatted["usage"]["input_tokens"] == 10
        assert formatted["usage"]["output_tokens"] == 5
        assert formatted["usage"]["cache_creation_input_tokens"] == 0
        assert formatted["usage"]["cache_read_input_tokens"] == 0

    def test_with_cache_tokens(self):
        from bedrock_gateway.converter import format_anthropic_response
        result = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 30,
            },
            "stop_reason": "end_turn",
        }
        formatted = format_anthropic_response(result, "model-x")
        assert formatted["usage"]["cache_creation_input_tokens"] == 50
        assert formatted["usage"]["cache_read_input_tokens"] == 30

    def test_stop_sequence(self):
        from bedrock_gateway.converter import format_anthropic_response
        result = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
            "stop_reason": "stop_sequence",
            "stop_sequence": "\n\n",
        }
        formatted = format_anthropic_response(result, "model-x")
        assert formatted["stop_reason"] == "stop_sequence"
        assert formatted["stop_sequence"] == "\n\n"


class TestFormatAnthropicError:
    """Tests for format_anthropic_error."""

    def test_invalid_request(self):
        from bedrock_gateway.converter import format_anthropic_error
        err = format_anthropic_error(400, "bad request")
        assert err["type"] == "error"
        assert err["error"]["type"] == "invalid_request_error"
        assert err["error"]["message"] == "bad request"

    def test_rate_limit(self):
        from bedrock_gateway.converter import format_anthropic_error
        err = format_anthropic_error(429, "too many requests")
        assert err["error"]["type"] == "rate_limit_error"

    def test_unknown_status(self):
        from bedrock_gateway.converter import format_anthropic_error
        err = format_anthropic_error(500, "oops")
        assert err["error"]["type"] == "api_error"


class TestMakeAnthropicSSE:
    """Tests for make_anthropic_sse."""

    def test_format(self):
        from bedrock_gateway.converter import make_anthropic_sse
        line = make_anthropic_sse("message_start", {"type": "message_start"})
        assert line.startswith("event: message_start\n")
        assert "data: " in line
        assert line.endswith("\n\n")
        data = json.loads(line.split("data: ")[1].strip())
        assert data["type"] == "message_start"


# ---------------------------------------------------------------------------
# Model resolution for Messages API
# ---------------------------------------------------------------------------


class TestMessagesModelResolution:
    """Model alias resolution works for Messages API."""

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_model_resolved(self, mock_client_cls, client: TestClient):
        """Model alias is resolved to Bedrock ID."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        # Verify the resolved model is in the response
        assert resp.json()["model"] == "us.anthropic.test-model-v1"

        # Verify the resolved model was used in the Bedrock request URL
        mock_instance = mock_client_cls.return_value
        call_args = mock_instance.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
        assert "us.anthropic.test-model-v1" in url

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_passthrough_model(self, mock_client_cls, client: TestClient):
        """Unknown model names pass through as-is."""
        mock_client_cls.return_value = _mock_sync_client(
            _bedrock_text_response()
        )

        resp = client.post("/v1/messages", json={
            "model": "us.anthropic.custom-model-v1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        assert resp.json()["model"] == "us.anthropic.custom-model-v1"
