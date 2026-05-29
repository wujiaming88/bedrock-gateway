"""Coverage tests for bedrock_gateway.server — paths not exercised by the
existing sync/stream happy-path tests: error branches, timeouts, retries,
count_tokens, reasoning_effort, optional params, run()."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    DashboardConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    StorageConfig,
)
from bedrock_gateway.server import (
    _note_retry,
    _note_timeout,
    _track_upstream,
    create_app,
    run,
)


@pytest.fixture
def config(tmp_path) -> GatewayConfig:
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=2, base_delay=0.001),
        dashboard=DashboardConfig(
            enabled=True, require_auth=False, api_key=None, localhost_only=False,
            rate_limit=60, max_request_log=20,
            storage=StorageConfig(enabled=False, path=str(tmp_path / "x.db"), retain_days=7),
        ),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-model-v1",
                context_length=200000,
                max_output=4096,
            ),
            "claude-sonnet-4.6": ModelEntry(
                bedrock_id="us.anthropic.claude-sonnet-4-6",
                context_length=1_000_000,
                max_output=64_000,
            ),
        },
    )


@pytest.fixture
def client(config: GatewayConfig) -> TestClient:
    return TestClient(create_app(config))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_mock_client_post(resp):
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _mk_mock_client_post_side_effect(*side_effect):
    inst = AsyncMock()
    inst.post = AsyncMock(side_effect=list(side_effect))
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _encode_event(event: dict) -> bytes:
    return f'{{"bytes":"{base64.b64encode(json.dumps(event).encode()).decode()}"}}'.encode()


def _make_stream_ctx(events: list[dict] | None = None, status: int = 200,
                     err_text: str = ""):
    chunks = [_encode_event(e) for e in (events or [])]

    async def aiter_bytes():
        for c in chunks:
            yield c

    async def aiter_text():
        yield err_text

    resp = MagicMock()
    resp.status_code = status
    resp.aiter_bytes = aiter_bytes
    resp.aiter_text = aiter_text

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)

    inst = AsyncMock()
    inst.stream = MagicMock(return_value=ctx)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


# ---------------------------------------------------------------------------
# Root + helpers
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_get(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root_head(self, client: TestClient):
        resp = client.head("/")
        assert resp.status_code == 200


class TestMetricsHelpers:
    """Internal helper functions — they swallow exceptions and handle None."""

    def test_note_retry_with_none(self):
        # Should not raise.
        _note_retry(None)

    def test_note_timeout_with_none(self):
        _note_timeout(None)

    def test_note_retry_with_empty_state(self):
        # A fake request whose state.metrics_info is None — triggers init path.
        class _State:
            def __init__(self):
                self.metrics_info = None

        class _Req:
            state = _State()

        _note_retry(_Req)  # class-level state works too

    async def test_track_upstream_no_health(self):
        async with _track_upstream(None):
            pass  # no-op path

    def test_note_retry_swallows_exception(self):
        """If the request object blows up when we touch state, don't crash."""

        class BadState:
            @property
            def metrics_info(self):
                raise RuntimeError("no state")

            @metrics_info.setter
            def metrics_info(self, _v):
                raise RuntimeError("cannot set")

        class _Req:
            state = BadState()

        _note_retry(_Req)  # must not raise

    def test_note_timeout_swallows_exception(self):
        class BadState:
            @property
            def metrics_info(self):
                raise RuntimeError("no state")

            @metrics_info.setter
            def metrics_info(self, _v):
                raise RuntimeError("cannot set")

        class _Req:
            state = BadState()

        _note_timeout(_Req)

    def test_note_timeout_initializes_metrics_info(self):
        """If metrics_info is None, _note_timeout must create it."""

        class _State:
            metrics_info = None

        class _Req:
            state = _State()

        _note_timeout(_Req)
        assert _Req.state.metrics_info == {"timeout": True}


# ---------------------------------------------------------------------------
# /v1/chat/completions — branches not covered by test_api.py
# ---------------------------------------------------------------------------


class TestChatCompletionsOptionalFields:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_top_p_and_stop_list(self, mock_cls, client: TestClient):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(resp_mock)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.8,
            "stop": ["END", "STOP"],
        })
        assert resp.status_code == 200
        # Confirm top_p and stop_sequences passed through.
        call = mock_cls.return_value.post.call_args
        body = json.loads(call.kwargs["content"])
        assert body["top_p"] == 0.8
        assert body["stop_sequences"] == ["END", "STOP"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stop_scalar_wrapped(self, mock_cls, client: TestClient):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(resp_mock)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "XYZ",
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        assert body["stop_sequences"] == ["XYZ"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_tool_choice_specific(self, mock_cls, client: TestClient):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(resp_mock)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        assert body["tool_choice"]["type"] == "tool"
        assert body["tool_choice"]["name"] == "f"


class TestChatCompletionsReasoningEffort:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_reasoning_effort_maps_to_thinking(self, mock_cls, client: TestClient):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(resp_mock)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        # thinking is set; temperature dropped.
        assert "thinking" in body
        assert "temperature" not in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_reasoning_effort_minimal_clamps_budget(self, mock_cls, client: TestClient):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(resp_mock)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "minimal",  # budget=128 → clamp up to 1024
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        assert body["thinking"]["budget_tokens"] == 1024

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_thinking_auto_fills_max_tokens(self, mock_cls, client: TestClient):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(resp_mock)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        # default_max=4096 + budget=2048 = 6144
        assert body["max_tokens"] == 2048 + 4096


# ---------------------------------------------------------------------------
# Sync handler — timeout + unknown exception branches
# ---------------------------------------------------------------------------


class TestChatCompletionsTimeouts:
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_timeout_all_retries_exhausted_returns_502(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        inst = AsyncMock()
        inst.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 502
        assert "retries failed" in resp.json()["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_generic_exception_returns_500(self, mock_cls, client: TestClient):
        inst = AsyncMock()
        inst.post = AsyncMock(side_effect=RuntimeError("boom"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 500
        assert "boom" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Streaming handler — retry / error / timeout paths
# ---------------------------------------------------------------------------


class TestStreamRetryAndErrors:
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_429_then_200(self, mock_cls, mock_sleep, client: TestClient):
        # First attempt: 429 (retry). Second: text event + message_delta.
        ctx_429 = _make_stream_ctx([], status=429)
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "message_delta",
             "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 1, "input_tokens": 3}},
        ]
        ctx_200 = _make_stream_ctx(events, status=200)

        # Two client instances needed (each AsyncClient context does one request).
        mock_cls.side_effect = [ctx_429, ctx_200]

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        body = resp.text
        assert "[DONE]" in body
        assert "hi" in body

    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_400_error_body(self, mock_cls, mock_sleep, client: TestClient):
        # Pre-stream 400: must surface as a REAL HTTP 400 (not a fake-200 SSE
        # body) so the client's HTTP layer sees the failure immediately.
        ctx = _make_stream_ctx([], status=400,
                               err_text='{"message":"bad request"}')
        mock_cls.return_value = ctx
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["message"] == "bad request"
        assert data["error"]["type"] == "invalid_request_error"

    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_timeout_retries_then_errors(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        # Always time out opening the stream → exhaust retries → real HTTP 504.
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("t"))
        ctx.__aexit__ = AsyncMock(return_value=False)

        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)

        mock_cls.return_value = inst
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 504
        assert "timeout" in resp.json()["error"]["message"].lower()

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_generic_exception(self, mock_cls, client: TestClient):
        # Connection-level failure before any bytes → real HTTP 500, not a
        # silently-dropped frame that hangs the client.
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("unexpected"))
        ctx.__aexit__ = AsyncMock(return_value=False)

        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)

        mock_cls.return_value = inst
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 500
        assert "unexpected" in resp.json()["error"]["message"]


class TestStreamEventTypes:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_tool_use_and_thinking_events(self, mock_cls, client: TestClient):
        """Exercise content_block_start tool_use + thinking + input_json_delta +
        thinking_delta + signature_delta + content_block_stop branches."""
        events = [
            {"type": "message_start",
             "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_start",
             "content_block": {"type": "thinking"}},
            {"type": "content_block_delta",
             "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_delta",
             "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop"},
            {"type": "content_block_start",
             "content_block": {
                 "type": "tool_use", "id": "tu_1", "name": "do_thing"}},
            {"type": "content_block_delta",
             "delta": {"type": "input_json_delta", "partial_json": '{"x":'}},
            {"type": "content_block_delta",
             "delta": {"type": "input_json_delta", "partial_json": '1}'}},
            {"type": "content_block_stop"},
            {"type": "message_delta",
             "delta": {"stop_reason": "tool_use"},
             "usage": {"output_tokens": 4, "input_tokens": 5}},
        ]
        mock_cls.return_value = _make_stream_ctx(events)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        body = resp.text
        assert "tool_calls" in body
        assert "reasoning_content" in body
        assert "do_thing" in body
        # finish_reason tool_calls
        assert '"finish_reason":"tool_calls"' in body or '"finish_reason": "tool_calls"' in body


# ---------------------------------------------------------------------------
# /v1/messages — branches
# ---------------------------------------------------------------------------


class TestMessagesMetadataAndOthers:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_metadata_passthrough(self, mock_cls, client: TestClient):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(r)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "u_1"},
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        assert body["metadata"] == {"user_id": "u_1"}

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_thinking_auto_fills_max_tokens(self, mock_cls, client: TestClient):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_cls.return_value = _mk_mock_client_post(r)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            # No max_tokens given — auto-fill path.
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        })
        assert resp.status_code == 200
        body = json.loads(mock_cls.return_value.post.call_args.kwargs["content"])
        assert body["max_tokens"] == 2048 + 4096  # test-model max_output=4096

    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_sync_timeout_then_502(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        inst = AsyncMock()
        inst.post = AsyncMock(side_effect=httpx.TimeoutException("t"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 502
        data = resp.json()
        assert data["type"] == "error"
        assert "retries failed" in data["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_sync_generic_exception_returns_500(
        self, mock_cls, client: TestClient
    ):
        inst = AsyncMock()
        inst.post = AsyncMock(side_effect=RuntimeError("nope"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 500
        data = resp.json()
        assert data["type"] == "error"
        assert "nope" in data["error"]["message"]


class TestMessagesStreamErrorPaths:
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_stream_429_then_success(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        ctx_429 = _make_stream_ctx([], status=429)
        events = [
            {"type": "message_start",
             "message": {"usage": {"input_tokens": 3}}},
            {"type": "content_block_start",
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_stop"},
            {"type": "message_delta",
             "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
            {"type": "ping"},
        ]
        ctx_ok = _make_stream_ctx(events, status=200)
        mock_cls.side_effect = [ctx_429, ctx_ok]

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        body = resp.text
        assert "message_start" in body
        assert "message_stop" in body
        assert "ping" in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_stream_400_error_body(self, mock_cls, client: TestClient):
        # Pre-stream 400 → real HTTP 400 with a complete Anthropic error
        # envelope (matches the upstream Anthropic API for stream-open errors).
        ctx = _make_stream_ctx(
            [], status=400, err_text='{"message":"bad"}'
        )
        mock_cls.return_value = ctx
        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["message"] == "bad"

    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_stream_timeout_exhausted(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("t"))
        ctx.__aexit__ = AsyncMock(return_value=False)

        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)

        mock_cls.return_value = inst
        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 504
        assert resp.json()["type"] == "error"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_stream_generic_exception(self, mock_cls, client: TestClient):
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("fail"))
        ctx.__aexit__ = AsyncMock(return_value=False)

        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)

        mock_cls.return_value = inst
        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# /v1/messages/count_tokens — pre-flight estimator
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_invalid_json(self, client: TestClient):
        resp = client.post(
            "/v1/messages/count_tokens",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_simple_text_messages(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] >= 1

    def test_multipart_messages(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "source": {"type": "base64"}},  # no text
                    {"type": "tool_result",
                     "tool_use_id": "t",
                     "content": "result text"},
                ],
            }],
        })
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] >= 1

    def test_non_dict_content_block(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "messages": [{"role": "user", "content": ["plain string"]}],
        })
        assert resp.status_code == 200

    def test_content_none(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "messages": [{"role": "user", "content": None}],
        })
        assert resp.status_code == 200

    def test_non_string_non_list_content(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "messages": [{"role": "user", "content": 42}],
        })
        assert resp.status_code == 200

    def test_system_and_tools_counted(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "t", "description": "tool", "input_schema": {}}],
        })
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] >= 1

    def test_skips_non_dict_messages(self, client: TestClient):
        resp = client.post("/v1/messages/count_tokens", json={
            "messages": ["not a dict"],
        })
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 1  # floor


# ---------------------------------------------------------------------------
# Unknown model rejection (both /v1 endpoints)
# ---------------------------------------------------------------------------


class TestUnknownModelRejection:
    def test_chat_completions_unknown_model(self, client: TestClient):
        resp = client.post("/v1/chat/completions", json={
            "model": "totally-unknown-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_messages_unknown_model(self, client: TestClient):
        resp = client.post("/v1/messages", json={
            "model": "totally-unknown-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# run() and create_app(None) paths
# ---------------------------------------------------------------------------


class TestRunAndCreateAppDefaults:
    def test_create_app_with_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No config.yaml in cwd → pure defaults.
        app = create_app(None)
        assert app is not None

    def test_run_invokes_uvicorn(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        called = {}

        def fake_uvicorn_run(app, host, port, log_level):
            called["host"] = host
            called["port"] = port

        monkeypatch.setattr("bedrock_gateway.server.uvicorn.run", fake_uvicorn_run)
        run()
        assert called["port"] == 4000

    def test_run_with_explicit_config(self, monkeypatch, config):
        calls = {}

        def fake_uvicorn_run(app, host, port, log_level):
            calls["port"] = port

        monkeypatch.setattr("bedrock_gateway.server.uvicorn.run", fake_uvicorn_run)
        run(config)
        assert calls["port"] == 4000


# ---------------------------------------------------------------------------
# Storage-error path in create_app
# ---------------------------------------------------------------------------


class TestAppLifecycle:
    def test_startup_and_shutdown_events(self, config: GatewayConfig):
        """Using TestClient as a context manager triggers startup/shutdown."""
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200


class TestMessagesStreamAllRetriesExhausted:
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_all_retries_exhausted_sse_error(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        """Every attempt returns 429 → retries exhausted → real HTTP 429."""
        # Build N fresh stream contexts, each 429.
        ctxs = [_make_stream_ctx([], status=429) for _ in range(5)]
        mock_cls.side_effect = ctxs
        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 429
        data = resp.json()
        assert data["type"] == "error"
        assert "attempts failed" in data["error"]["message"]


class TestChatStreamAllRetriesExhausted:
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_all_retries_exhausted_sse_error(
        self, mock_cls, mock_sleep, client: TestClient
    ):
        ctxs = [_make_stream_ctx([], status=429) for _ in range(5)]
        mock_cls.side_effect = ctxs
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 429
        assert "attempts failed" in resp.json()["error"]["message"]


class TestCreateAppStorageFailure:
    def test_storage_init_failure_falls_back(self, monkeypatch):
        # Force MetricsStorage to raise — create_app should log and continue.
        def _bad(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr("bedrock_gateway.server.MetricsStorage", _bad)

        cfg = GatewayConfig(
            auth=AuthConfig(mode="bearer_token", bearer_token="t"),
            region="us-east-1",
            server=ServerConfig(),
            retry=RetryConfig(max_retries=1),
            dashboard=DashboardConfig(
                enabled=True, require_auth=False, api_key=None, localhost_only=False,
                rate_limit=60, max_request_log=50,
                storage=StorageConfig(enabled=True, path="/tmp/x", retain_days=7),
            ),
            models={"t": ModelEntry(bedrock_id="us.anthropic.t",
                                    context_length=100, max_output=100)},
        )
        app = create_app(cfg)
        # App comes up, just without storage.
        client = TestClient(app)
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Mid-stream exception frames — the silent-drop bug that hung clients.
# Upstream returns 200, starts a stream, then injects an event-stream
# exception frame (throttling / internal / model-stream error). The gateway
# must surface a VALID error terminator on both protocols, not drop it.
# ---------------------------------------------------------------------------


def _exc_frame(exc_type: str, message: str) -> bytes:
    """A realistic AWS event-stream exception frame (see decoder tests)."""
    return (
        b"\x00\x00\x0d:exception-type\x07\x00"
        + bytes([len(exc_type)])
        + exc_type.encode()
        + json.dumps({"message": message}).encode()
    )


def _make_raw_stream_ctx(raw_frames: list[bytes], status: int = 200):
    """Stream context whose body is arbitrary raw bytes (so we can inject
    exception frames that are NOT `"bytes"`-wrapped normal events)."""
    async def aiter_bytes():
        for f in raw_frames:
            yield f

    async def aiter_text():
        yield ""

    resp = MagicMock()
    resp.status_code = status
    resp.aiter_bytes = aiter_bytes
    resp.aiter_text = aiter_text

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)

    inst = AsyncMock()
    inst.stream = MagicMock(return_value=ctx)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _parse_sse(text: str) -> list[dict]:
    events = []
    current = {}
    for line in text.strip().split("\n"):
        if line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            payload = line[6:]
            current["data"] = None if payload == "[DONE]" else json.loads(payload)
            current["raw"] = payload
            events.append(current)
            current = {}
    return events


class TestMessagesMidStreamException:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_throttling_midstream_emits_error_event(
        self, mock_cls, client: TestClient
    ):
        # 200 open, message_start flows, THEN a throttling exception frame.
        frames = [
            _encode_event({"type": "message_start",
                           "message": {"usage": {"input_tokens": 3}}}),
            _encode_event({"type": "content_block_start",
                           "content_block": {"type": "text", "text": ""}}),
            _encode_event({"type": "content_block_delta",
                           "delta": {"type": "text_delta", "text": "partial"}}),
            _exc_frame("throttlingException", "Rate exceeded mid-stream"),
        ]
        mock_cls.return_value = _make_raw_stream_ctx(frames)

        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        # Stream already opened 200 — error must arrive as a valid SSE event.
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        types = [e["event"] for e in events]
        assert "message_start" in types
        # The partial content was delivered before the fault.
        assert any(e["event"] == "content_block_delta" for e in events)
        # And the stream is terminated by a VALID error event (not silence).
        err = [e for e in events if e["event"] == "error"]
        assert len(err) == 1
        assert err[0]["data"]["type"] == "error"
        assert err[0]["data"]["error"]["type"] == "rate_limit_error"
        assert "Rate exceeded mid-stream" in err[0]["data"]["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_internal_error_midstream(self, mock_cls, client: TestClient):
        frames = [
            _encode_event({"type": "message_start",
                           "message": {"usage": {"input_tokens": 1}}}),
            _exc_frame("internalServerException", "kaboom"),
        ]
        mock_cls.return_value = _make_raw_stream_ctx(frames)
        resp = client.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        events = _parse_sse(resp.text)
        err = [e for e in events if e["event"] == "error"]
        assert len(err) == 1
        assert err[0]["data"]["error"]["type"] == "api_error"


class TestChatMidStreamException:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_throttling_midstream_emits_error_chunk_and_done(
        self, mock_cls, client: TestClient
    ):
        frames = [
            _encode_event({"type": "message_start",
                           "message": {"usage": {"input_tokens": 3}}}),
            _encode_event({"type": "content_block_delta",
                           "delta": {"type": "text_delta", "text": "partial"}}),
            _exc_frame("throttlingException", "slow down"),
        ]
        mock_cls.return_value = _make_raw_stream_ctx(frames)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        body = resp.text
        # Partial content delivered, then a visible error, then [DONE] so the
        # OpenAI-style client terminates cleanly instead of hanging.
        assert "partial" in body
        assert "slow down" in body
        assert "[DONE]" in body
        events = _parse_sse(body)
        err_chunks = [e for e in events
                      if isinstance(e.get("data"), dict) and "error" in e["data"]]
        assert len(err_chunks) == 1
        assert err_chunks[0]["data"]["error"]["code"] == 429
        # [DONE] is the final line.
        assert events[-1]["raw"] == "[DONE]"
