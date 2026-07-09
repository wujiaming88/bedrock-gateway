"""
Integration tests for the inbound Anthropic Messages ⇄ OpenAI Responses feature
on the ``POST /v1/messages`` endpoint.

These exercise the full server path through ``TestClient`` with a mocked
upstream, covering the endpoint fork:

  * Bedrock mantle (``gpt-5.5``) — Anthropic-in → Responses-out → Anthropic-back.
  * Azure prefix passthrough (``azure/gpt-5.5``) — same, different transport
    (api-key header, per-resource URL), proving the two-axis reuse: the inbound
    translation is transport-agnostic.
  * Streaming (SSE) end-to-end.
  * Error mapping (upstream error → Anthropic error envelope).
  * **Claude regression** — an ``anthropic``-dialect model on ``/v1/messages``
    still takes the untouched original path.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    AzureResource,
    GatewayConfig,
    RetryConfig,
    ServerConfig,
    _DEFAULT_MODELS,
    _parse_models,
)
from bedrock_gateway.server import create_app

AZ_BASE = "https://my-res.cognitiveservices.azure.com/openai/v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Gateway with default models (Claude + GPT-5.5 + Grok) and an Azure
    resource that opts into ``azure/<deployment>`` prefix passthrough."""
    resources = {
        "r1": AzureResource(base_url=AZ_BASE, api_key="az-secret", prefix="azure"),
    }
    config = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=_parse_models(_DEFAULT_MODELS, resources),
        azure_resources=resources,
    )
    return TestClient(create_app(config))


def _mock_sync_client(response_data: dict, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_data
    resp.text = json.dumps(response_data)
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _mock_stream_client(frames: list[bytes], status: int = 200):
    async def aiter_bytes():
        for f in frames:
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


def _sent(mock_cls) -> dict:
    call = mock_cls.return_value.post.call_args
    return json.loads(call.kwargs.get("content", b"{}"))


def _sent_url_headers(mock_cls):
    call = mock_cls.return_value.post.call_args
    url = call[0][0] if call[0] else call.kwargs.get("url")
    return url, call.kwargs.get("headers", {})


def _responses_body(text="hi"):
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "upstream-id",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _responses_tool_body():
    return {
        "id": "resp_2",
        "object": "response",
        "status": "completed",
        "model": "upstream-id",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "Read",
            "arguments": '{"path":"/tmp/x"}',
        }],
        "usage": {"input_tokens": 5, "output_tokens": 8},
    }


# ---------------------------------------------------------------------------
# Bedrock mantle (gpt-5.5) — sync
# ---------------------------------------------------------------------------


class TestMantleSync:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_gpt55_text(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body("hello"))
        resp = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 100,
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        # Anthropic-shaped response
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"] == [{"type": "text", "text": "hello"}]
        assert data["stop_reason"] == "end_turn"
        assert data["usage"] == {
            "input_tokens": 3, "output_tokens": 2,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        # Upstream got a translated Responses body over the mantle URL
        url, _ = _sent_url_headers(mock_cls)
        assert "bedrock-mantle.us-east-1.api.aws/openai/v1/responses" in url
        sent = _sent(mock_cls)
        assert sent["model"] == "openai.gpt-5.5"
        assert sent["instructions"] == "be terse"
        assert sent["max_output_tokens"] == 100
        assert sent["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_gpt55_tool_use(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_tool_body())
        resp = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [{
                "name": "Read",
                "description": "read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "tool_use"
        block = data["content"][0]
        assert block["type"] == "tool_use"
        assert block["name"] == "Read"
        assert block["input"] == {"path": "/tmp/x"}
        # Tools translated into Responses function tools
        sent = _sent(mock_cls)
        assert sent["tools"][0]["type"] == "function"
        assert sent["tools"][0]["name"] == "Read"
        assert sent["tools"][0]["parameters"]["properties"]["path"]["type"] == "string"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_default_max_tokens_when_omitted(self, mock_cls, client):
        # Anthropic requires max_tokens; the gateway still forwards a Responses
        # body even if the client omitted it (registry default used upstream).
        mock_cls.return_value = _mock_sync_client(_responses_body())
        resp = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_upstream_error_becomes_anthropic_error(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(
            {"message": "model unavailable"}, status=400)
        resp = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        data = resp.json()
        # Anthropic error envelope, NOT OpenAI's
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "model unavailable" in data["error"]["message"]


# ---------------------------------------------------------------------------
# Bedrock mantle (gpt-5.5) — streaming
# ---------------------------------------------------------------------------


class TestMantleStream:
    def _collect(self, client, mock_cls, frames):
        mock_cls.return_value = _mock_stream_client(frames)
        with client.stream("POST", "/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())
        return body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_text_translated_to_anthropic_events(self, mock_cls, client):
        frames = [
            b'event: response.created\ndata: {"type":"response.created","response":{"usage":{"input_tokens":3}}}\n\n',
            b'event: response.content_part.added\ndata: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text"}}\n\n',
            b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hel"}\n\n',
            b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"lo"}\n\n',
            b'event: response.content_part.done\ndata: {"type":"response.content_part.done","output_index":0,"content_index":0}\n\n',
            b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":2}}}\n\n',
        ]
        body = self._collect(client, mock_cls, frames)
        # Anthropic event names present, Responses names absent
        assert "event: message_start" in body
        assert "event: content_block_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_delta" in body
        assert "event: message_stop" in body
        assert "response.output_text.delta" not in body
        # text_delta assembled
        assert '"text":"Hel"' in body or '"text": "Hel"' in body
        # Upstream stream opened on the mantle responses path (not native invoke)
        # with a translated Responses body carrying stream:true.
        stream_call = mock_cls.return_value.stream.call_args
        url = stream_call[0][1]  # ("POST", url, ...)
        assert "responses" in url
        assert "invoke" not in url
        sent = json.loads(stream_call.kwargs.get("content", b"{}"))
        assert sent["stream"] is True
        assert sent["model"] == "openai.gpt-5.5"
        assert sent["input"][0]["content"][0]["text"] == "hi"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_stream_upstream_preflight_error(self, mock_cls, client):
        # A non-200 on stream open must surface as a real HTTP error (Anthropic
        # envelope), not a 200 SSE body.
        mock_cls.return_value = _mock_stream_client(
            [b'{"message":"nope"}'], status=403)
        r = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert r.status_code == 403
        data = r.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "permission_error"


# ---------------------------------------------------------------------------
# Azure prefix passthrough (azure/gpt-5.5) — proves transport reuse
# ---------------------------------------------------------------------------


class TestAzureInbound:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_azure_gpt55_sync(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body("azured"))
        resp = client.post("/v1/messages", json={
            "model": "azure/gpt-5.5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == [{"type": "text", "text": "azured"}]
        # Routed to the Azure resource URL with api-key auth (NOT bedrock)
        url, headers = _sent_url_headers(mock_cls)
        assert url == AZ_BASE + "/responses"
        assert headers.get("api-key") == "az-secret"
        # Deployment name (prefix stripped) is the upstream model id
        sent = _sent(mock_cls)
        assert sent["model"] == "gpt-5.5"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_azure_gpt55_stream(self, mock_cls, client):
        frames = [
            b'event: response.created\ndata: {"type":"response.created"}\n\n',
            b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hi"}\n\n',
            b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        ]
        mock_cls.return_value = _mock_stream_client(frames)
        with client.stream("POST", "/v1/messages", json={
            "model": "azure/gpt-5.5",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        assert "event: message_start" in body
        assert "event: message_stop" in body


# ---------------------------------------------------------------------------
# Claude regression — the anthropic-dialect path is unchanged
# ---------------------------------------------------------------------------


class TestClaudeRegression:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_claude_still_native_path(self, mock_cls, client):
        # Claude on /v1/messages must hit the native bedrock-runtime invoke path
        # and pass through unchanged (NOT the responses translation).
        claude_upstream = {
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "claude here"}],
            "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 3},
        }
        mock_cls.return_value = _mock_sync_client(claude_upstream)
        resp = client.post("/v1/messages", json={
            "model": "claude-haiku",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == [{"type": "text", "text": "claude here"}]
        # Native Bedrock invoke path — the request body is Anthropic, not Responses
        url, _ = _sent_url_headers(mock_cls)
        assert "/invoke" in url
        assert "bedrock-runtime" in url
        sent = _sent(mock_cls)
        assert sent["anthropic_version"] == "bedrock-2023-05-31"
        assert "messages" in sent          # Anthropic shape
        assert "input" not in sent          # NOT translated to Responses

    def test_openai_chat_model_rejected_on_messages(self):
        # An openai-chat dialect model on /v1/messages is still a client error —
        # only the responses dialect is translated here. Build a gateway with an
        # explicit Azure chat model and assert the guard returns a 400 Anthropic
        # error pointing at the right endpoints.
        from bedrock_gateway.config import ModelEntry

        resources = {"r1": AzureResource(base_url=AZ_BASE, api_key="k")}
        models = {
            "az-chat": ModelEntry(
                bedrock_id="gpt-4o",
                transport="azure",
                dialect="openai-chat",
                deployment="gpt-4o",
                azure_endpoint=AZ_BASE,
                azure_api_key="k",
            ),
        }
        config = GatewayConfig(
            auth=AuthConfig(mode="bearer_token", bearer_token="t"),
            region="us-east-1",
            server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
            retry=RetryConfig(max_retries=1, base_delay=0.01),
            models=models,
            azure_resources=resources,
        )
        chat_client = TestClient(create_app(config))
        resp = chat_client.post("/v1/messages", json={
            "model": "az-chat",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert "not available on /v1/messages" in data["error"]["message"]


# ---------------------------------------------------------------------------
# Unknown model on /v1/messages
# ---------------------------------------------------------------------------


class TestUnknownModel:
    def test_unknown_model_400(self, client):
        resp = client.post("/v1/messages", json={
            "model": "totally-unknown-xyz",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert resp.json()["type"] == "error"


# ---------------------------------------------------------------------------
# Error-path edge cases (mid-stream failure, malformed error body)
# ---------------------------------------------------------------------------


class TestErrorPaths:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_midstream_exception_emits_error_event(self, mock_cls, client):
        # An exception raised while consuming the upstream stream must be caught
        # and surfaced as a valid Anthropic error SSE frame (clean terminator),
        # not leak as an unhandled 500 / hang.
        async def boom_bytes():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            raise RuntimeError("connection reset mid-stream")

        resp = MagicMock()
        resp.status_code = 200
        resp.aiter_bytes = boom_bytes

        async def aiter_text():
            yield ""
        resp.aiter_text = aiter_text
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        with client.stream("POST", "/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        # A legal Anthropic error frame terminates the stream.
        assert "event: error" in body
        assert "connection reset mid-stream" in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_midstream_timeout_emits_error_event(self, mock_cls, client):
        # A read timeout mid-stream must terminate with a valid Anthropic error
        # frame, distinct from the generic-exception branch.
        import httpx as _httpx

        async def timeout_bytes():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            raise _httpx.ReadTimeout("read timed out")

        resp = MagicMock()
        resp.status_code = 200
        resp.aiter_bytes = timeout_bytes

        async def aiter_text():
            yield ""
        resp.aiter_text = aiter_text
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        with client.stream("POST", "/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        assert "event: error" in body
        assert "timeout" in body.lower()

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_malformed_error_body_still_anthropic_shaped(self, mock_cls, client):
        # Upstream 200 with non-JSON body → _handle_sync returns a 502
        # JSONResponse; the OpenAI→Anthropic error rewrap must still produce a
        # valid Anthropic envelope even when the error body can't be parsed.
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        resp.content = b"garbage"
        resp.text = "garbage"
        inst = AsyncMock()
        inst.post = AsyncMock(return_value=resp)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        r = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 502
        data = r.json()
        assert data["type"] == "error"
        assert "error" in data and "message" in data["error"]

    def test_oai_error_rewrap_unparseable_body(self):
        # Defensive branch: if the OpenAI error JSONResponse body can't be
        # parsed, the rewrap still yields a valid Anthropic error envelope with
        # the original status code and a generic message.
        from fastapi.responses import JSONResponse

        from bedrock_gateway.server import _oai_error_to_anthropic

        bad = JSONResponse(status_code=503, content={"error": {}})
        bad.body = b"\xff\xfe not json"  # force json.loads to fail
        out = _oai_error_to_anthropic(bad)
        assert out.status_code == 503
        payload = json.loads(bytes(out.body))
        assert payload["type"] == "error"
        assert payload["error"]["message"] == "upstream error"
