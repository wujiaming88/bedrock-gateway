"""Native Anthropic HTTP passthrough tests."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from bedrock_gateway.auth import AuthConfig
from bedrock_gateway.config import GatewayConfig, RetryConfig, ServerConfig, load_config
from bedrock_gateway.server import create_app


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""
use_default_models: false
upstream_resources:
  vendor:
    prefix: vendor
    secret_env: VENDOR_SECRET
    routes:
      anthropic-passthrough:
        base_url: https://api.example.test/anthropic
        path: /v1/messages
        auth: x-api-key
        default_headers:
          anthropic-version: '2023-06-01'
""")
    cfg = load_config(path)
    cfg.auth = AuthConfig(mode="bearer_token", bearer_token="gateway-global")
    cfg.server = ServerConfig(host="127.0.0.1", port=4000, log_level="warning")
    cfg.retry = RetryConfig(max_retries=1, base_delay=0)
    cfg.dashboard.enabled = False
    return cfg


def _response(status=200, body=None, content_type="application/json"):
    body = body or {"id": "msg_1", "type": "message", "usage": {"input_tokens": 1}}
    response = MagicMock()
    response.status_code = status
    response.content = json.dumps(body).encode()
    response.headers = {"content-type": content_type, "x-request-id": "upstream-id", "set-cookie": "bad"}
    return response


@patch.dict("os.environ", {"VENDOR_SECRET": "upstream-secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_sync_body_headers_status_are_passthrough(mock_cls, tmp_path):
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=_response())
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    mock_cls.return_value = inst
    client = TestClient(create_app(_config(tmp_path)))
    body = {"model": "vendor/model-x", "max_tokens": 8, "future": {"kept": True}}
    response = client.post(
        "/v1/messages", json=body,
        headers={"anthropic-version": "2024-01-01", "anthropic-beta": "future", "Authorization": "Bearer client"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "msg_1"
    assert response.headers["x-request-id"] == "upstream-id"
    assert "set-cookie" not in response.headers
    call = inst.post.call_args
    assert call.args[0] == "https://api.example.test/anthropic/v1/messages"
    sent = json.loads(call.kwargs["content"])
    assert sent == {**body, "model": "model-x"}
    headers = call.kwargs["headers"]
    assert headers["x-api-key"] == "upstream-secret"
    assert headers["anthropic-version"] == "2024-01-01"
    assert headers["anthropic-beta"] == "future"
    assert "client" not in repr(headers)


@patch.dict("os.environ", {"VENDOR_SECRET": "upstream-secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_sync_upstream_error_is_preserved(mock_cls, tmp_path):
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=_response(422, error))
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    mock_cls.return_value = inst
    response = TestClient(create_app(_config(tmp_path))).post(
        "/v1/messages", json={"model": "vendor/model-x", "max_tokens": 8}
    )
    assert response.status_code == 422
    assert response.json() == error


@patch.dict("os.environ", {"VENDOR_SECRET": "upstream-secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_stream_bytes_and_comments_are_passthrough(mock_cls, tmp_path):
    frames = [
        b": ping\n\n",
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"你\"}}\n\n".encode(),
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def aiter_bytes():
        for frame in frames:
            yield frame

    upstream = MagicMock()
    upstream.status_code = 200
    upstream.headers = {"content-type": "text/event-stream", "x-request-id": "stream-id"}
    upstream.aiter_bytes = aiter_bytes
    response_ctx = AsyncMock()
    response_ctx.__aenter__ = AsyncMock(return_value=upstream)
    response_ctx.__aexit__ = AsyncMock(return_value=False)
    inst = AsyncMock()
    inst.stream = MagicMock(return_value=response_ctx)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    mock_cls.return_value = inst
    client = TestClient(create_app(_config(tmp_path)))
    with client.stream("POST", "/v1/messages", json={
        "model": "vendor/model-x", "stream": True, "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }) as response:
        text = response.read().decode()
    assert response.status_code == 200
    assert text == b"".join(frames).decode()
    assert ": ping" in text and "你" in text and "message_stop" in text


@patch.dict("os.environ", {"VENDOR_SECRET": "upstream-secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_stream_preflight_error_is_preserved(mock_cls, tmp_path):
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    upstream = MagicMock()
    upstream.status_code = 503
    upstream.headers = {"content-type": "application/json", "retry-after": "2"}
    upstream.aread = AsyncMock(return_value=json.dumps(error).encode())
    response_ctx = AsyncMock()
    response_ctx.__aenter__ = AsyncMock(return_value=upstream)
    response_ctx.__aexit__ = AsyncMock(return_value=False)
    inst = AsyncMock()
    inst.stream = MagicMock(return_value=response_ctx)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    mock_cls.return_value = inst
    response = TestClient(create_app(_config(tmp_path))).post(
        "/v1/messages", json={"model": "vendor/model-x", "stream": True, "max_tokens": 8}
    )
    assert response.status_code == 503
    assert response.json() == error
    assert response.headers["retry-after"] == "2"
