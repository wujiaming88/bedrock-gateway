"""
Tests for bedrock_gateway.server — API endpoint integration tests.

Uses FastAPI TestClient with mocked Bedrock responses.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    load_config,
)
from bedrock_gateway.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> GatewayConfig:
    """Minimal test configuration."""
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
        },
    )


@pytest.fixture
def client(config: GatewayConfig) -> TestClient:
    """TestClient with mocked configuration."""
    app = create_app(config)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["auth_mode"] == "bearer_token"
        assert data["region"] == "us-east-1"
        assert data["models"] == 1

    def test_health_has_version(self, client: TestClient):
        resp = client.get("/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_lists_models(self, client: TestClient):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        model = data["data"][0]
        assert model["id"] == "test-model"
        assert model["owned_by"] == "bedrock"
        assert model["context_length"] == 200000
        assert model["max_output_tokens"] == 4096


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — Sync
# ---------------------------------------------------------------------------

def _mock_bedrock_response() -> dict:
    """A typical Bedrock (Anthropic) response."""
    return {
        "content": [{"type": "text", "text": "Hello from Bedrock!"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }


class TestChatCompletionsSync:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_basic_completion(self, mock_client_cls, client: TestClient):
        # Set up mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_bedrock_response()

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from Bedrock!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5
        assert data["usage"]["total_tokens"] == 15

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_with_system_message(self, mock_client_cls, client: TestClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_bedrock_response()

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        })

        assert resp.status_code == 200
        # Verify system was extracted and sent
        call_kwargs = mock_instance.post.call_args
        body = json.loads(call_kwargs.kwargs.get("content", call_kwargs[1].get("content", b"{}")))
        assert body.get("system") == "You are helpful."

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_tool_use_response(self, mock_client_cls, client: TestClient):
        bedrock_resp = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_001",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = bedrock_resp

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc = data["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_invalid_json(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bedrock_error(self, mock_client_cls, client: TestClient):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = json.dumps({"message": "max_tokens: 200000 is too large"})

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert data["error"]["message"] == "max_tokens: 200000 is too large"
        assert data["error"]["type"] == "invalid_request_error"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bedrock_error_plain_text(self, mock_client_cls, client: TestClient):
        """Non-JSON error body is still passed through."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 500
        assert resp.json()["error"]["message"] == "Internal Server Error"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_retry_on_429(self, mock_sleep, mock_client_cls, client: TestClient):
        """429 should trigger retry, then succeed."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = _mock_bedrock_response()

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=[mock_429, mock_200])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 200
        assert mock_instance.post.call_count == 2


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — Streaming
# ---------------------------------------------------------------------------

class TestChatCompletionsStream:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_text(self, mock_client_cls, client: TestClient):
        """Streaming text deltas are converted to SSE chunks."""
        # Build fake AWS event stream with base64-encoded events
        events = [
            {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}},
            {"type": "content_block_stop"},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]

        raw_chunks = []
        for event in events:
            encoded = base64.b64encode(json.dumps(event).encode()).decode()
            raw_chunks.append(f'{{"bytes":"{encoded}"}}'.encode())

        # Create an async iterator for the stream
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
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE lines
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        assert len(data_lines) >= 2  # At least text deltas + finish

        # Verify content deltas
        first_data = json.loads(data_lines[0][6:])
        assert first_data["object"] == "chat.completion.chunk"
        assert "content" in first_data["choices"][0]["delta"]


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_app_state(self, client: TestClient, config: GatewayConfig):
        """App state carries the config and registry."""
        assert client.app.state.config is config  # type: ignore
        assert client.app.state.registry is not None
        assert client.app.state.auth is not None


# ---------------------------------------------------------------------------
# Model resolution integration
# ---------------------------------------------------------------------------

class TestModelResolution:
    """Model alias resolution via the registry."""

    def test_exact_match(self, client: TestClient):
        registry = client.app.state.registry  # type: ignore
        assert registry.resolve("test-model") == "us.anthropic.test-model-v1"

    def test_unknown_passthrough(self, client: TestClient):
        registry = client.app.state.registry  # type: ignore
        raw_id = "us.anthropic.some-random-model"
        assert registry.resolve(raw_id) == raw_id

    def test_default_max_output_lowered(self, client: TestClient):
        """Unknown models get 64K default, not 128K (avoids Bedrock rejection)."""
        registry = client.app.state.registry  # type: ignore
        assert registry.get_max_output("unknown-model") == 64_000

    def test_alias_resolution(self):
        """Common aliases resolve through the alias table."""
        from bedrock_gateway.models import ModelRegistry
        cfg = load_config("/nonexistent/config.yaml")  # loads default models
        registry = ModelRegistry(cfg)
        # claude-3.5-sonnet -> claude-sonnet-3.5 -> bedrock id
        resolved = registry.resolve("claude-3.5-sonnet")
        assert "sonnet" in resolved and "anthropic" in resolved
        # claude-opus -> claude-opus-4 -> bedrock id
        resolved = registry.resolve("claude-opus")
        assert "opus" in resolved and "anthropic" in resolved
        # claude-3-5-haiku -> claude-haiku -> bedrock id
        resolved = registry.resolve("claude-3-5-haiku")
        assert "haiku" in resolved and "anthropic" in resolved
