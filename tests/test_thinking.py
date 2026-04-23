"""
End-to-end tests for thinking / extended thinking support.

Tests the full request → response flow with mocked Bedrock API responses,
covering sync and streaming modes, reasoning_content output, and
reasoning_effort automatic mapping.
"""

import base64  # noqa: I001
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
    """Test configuration with models that cover adaptive + budget_tokens paths."""
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=2, base_delay=0.01),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
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

def _make_bedrock_thinking_response(
    thinking_text: str = "Let me reason...",
    answer_text: str = "The answer is 42.",
    include_redacted: bool = False,
) -> dict:
    """Build a Bedrock response that contains thinking blocks."""
    content = [
        {"type": "thinking", "thinking": thinking_text},
    ]
    if include_redacted:
        content.append({"type": "redacted_thinking", "data": "encrypted-data"})
    content.append({"type": "text", "text": answer_text})
    return {
        "content": content,
        "usage": {"input_tokens": 50, "output_tokens": 100},
        "stop_reason": "end_turn",
    }


def _encode_event(event: dict) -> bytes:
    """Encode a single event dict as an AWS event-stream binary frame."""
    encoded = base64.b64encode(json.dumps(event).encode()).decode()
    return f'{{"bytes":"{encoded}"}}'.encode()


def _mock_sync_client(response_data: dict):
    """Patch httpx.AsyncClient for a sync (non-streaming) Bedrock call."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data

    mock_instance = AsyncMock()
    mock_instance.post = AsyncMock(return_value=mock_response)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    return mock_instance


# ---------------------------------------------------------------------------
# Sync: thinking in response
# ---------------------------------------------------------------------------

class TestSyncThinking:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_sync_thinking_response(self, mock_client_cls, client: TestClient):
        """Sync response with thinking blocks returns reasoning_content."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response()
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think about this"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert msg["content"] == "The answer is 42."
        assert msg["reasoning_content"] == "Let me reason..."
        assert data["choices"][0]["finish_reason"] == "stop"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_sync_thinking_with_redacted(self, mock_client_cls, client: TestClient):
        """Redacted thinking blocks don't add text to reasoning_content."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response(
                thinking_text="Real thought.",
                include_redacted=True,
            )
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] == "Real thought."

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_sync_no_thinking_no_reasoning_content(
        self, mock_client_cls, client: TestClient
    ):
        """When response has no thinking blocks, reasoning_content is absent."""
        mock_client_cls.return_value = _mock_sync_client({
            "content": [{"type": "text", "text": "Plain answer."}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        msg = resp.json()["choices"][0]["message"]
        assert msg["content"] == "Plain answer."
        assert "reasoning_content" not in msg


# ---------------------------------------------------------------------------
# Streaming: thinking_delta in chunks
# ---------------------------------------------------------------------------

class TestStreamThinking:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_thinking_delta(self, mock_client_cls, client: TestClient):
        """Streaming thinking_delta events produce reasoning_content chunks."""
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 20}}},
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
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 50}},
            {"type": "message_stop"},
        ]

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

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think step by step"}],
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })

        assert resp.status_code == 200

        # Parse all data lines
        lines = resp.text.strip().split("\n")
        data_lines = [
            line for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

        # Collect reasoning_content and content chunks
        reasoning_chunks = []
        content_chunks = []
        for line in data_lines:
            payload = json.loads(line[6:])
            choices = payload.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if "reasoning_content" in delta:
                reasoning_chunks.append(delta["reasoning_content"])
            if "content" in delta:
                content_chunks.append(delta["content"])

        assert reasoning_chunks == ["Step 1: ", "analyze."]
        assert content_chunks == ["The answer."]


# ---------------------------------------------------------------------------
# Reasoning effort mapping (end-to-end)
# ---------------------------------------------------------------------------

class TestReasoningEffortE2E:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_reasoning_effort_mapped_to_thinking(
        self, mock_client_cls, client: TestClient
    ):
        """reasoning_effort is mapped to thinking parameter in the Bedrock request."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response()
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "medium",
        })

        assert resp.status_code == 200

        # Verify the Bedrock request body has thinking
        mock_instance = mock_client_cls.return_value
        call_kwargs = mock_instance.post.call_args
        body = json.loads(
            call_kwargs.kwargs.get("content", call_kwargs[1].get("content", b"{}"))
        )
        assert "thinking" in body
        # Budget should be clamped to 1024 minimum (medium=2048, so no clamp)
        assert body["thinking"]["budget_tokens"] == 2048
        # Temperature should be removed when thinking is enabled
        assert "temperature" not in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_reasoning_effort_minimal_clamped(
        self, mock_client_cls, client: TestClient
    ):
        """Minimal reasoning_effort (128 tokens) is clamped to 1024."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response()
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "minimal",
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["thinking"]["budget_tokens"] == 1024

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_thinking_takes_precedence_over_reasoning_effort(
        self, mock_client_cls, client: TestClient
    ):
        """Explicit thinking parameter overrides reasoning_effort."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response()
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "reasoning_effort": "low",
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["thinking"]["budget_tokens"] == 8192

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_reasoning_effort_adaptive_for_opus_4_6(
        self, mock_client_cls, client: TestClient
    ):
        """Claude opus 4.6 should use adaptive thinking."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response()
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "claude-opus-4",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "high",
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        assert body["thinking"] == {"type": "adaptive"}

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_max_tokens_auto_filled(self, mock_client_cls, client: TestClient):
        """When thinking is enabled without explicit max_tokens, it is auto-filled."""
        mock_client_cls.return_value = _mock_sync_client(
            _make_bedrock_thinking_response()
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "high",
            # No max_tokens specified
        })

        assert resp.status_code == 200
        mock_instance = mock_client_cls.return_value
        body = json.loads(
            mock_instance.post.call_args.kwargs.get(
                "content", mock_instance.post.call_args[1].get("content", b"{}")
            )
        )
        # budget_tokens=4096 + default_max=4096 = 8192
        assert body["max_tokens"] == 4096 + 4096
