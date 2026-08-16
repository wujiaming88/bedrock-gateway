"""DeepSeek three-protocol routing through generic upstream resources."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from bedrock_gateway.config import load_config
from bedrock_gateway.server import create_app


def _client(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
use_default_models: false
dashboard:
  enabled: false
upstream_resources:
  deepseek:
    prefix: deepseek
    secret_env: DEEPSEEK_API_KEY
    routes:
      openai-chat:
        base_url: https://api.deepseek.com
        path: /chat/completions
        auth: bearer
      openai-responses:
        base_url: https://api.deepseek.com
        path: /responses
        auth: bearer
      anthropic-passthrough:
        base_url: https://api.deepseek.com/anthropic
        path: /v1/messages
        auth: x-api-key
        default_headers:
          anthropic-version: '2023-06-01'
models:
  deepseek-v4-pro:
    upstream_resource: deepseek
    upstream_id: deepseek-v4-pro
    dialect: openai-chat
    context_length: 1000000
    max_output: 393216
""")
    return TestClient(create_app(load_config(cfg_file)))


def _sync(response):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = response
    resp.content = json.dumps(response).encode()
    resp.headers = {"content-type": "application/json"}
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


@patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_explicit_model_routes_chat(mock_cls, tmp_path):
    upstream = {"object": "chat.completion", "choices": [{"finish_reason": "stop"}], "usage": {}}
    mock_cls.return_value = _sync(upstream)
    response = _client(tmp_path).post("/v1/chat/completions", json={
        "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled"}, "future": True,
    })
    assert response.status_code == 200 and response.json() == upstream
    call = mock_cls.return_value.post.call_args
    assert call.args[0] == "https://api.deepseek.com/chat/completions"
    sent = json.loads(call.kwargs["content"])
    assert sent["model"] == "deepseek-v4-pro" and sent["future"] is True
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret"


@patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_explicit_model_routes_responses_without_bedrock_normalizer(mock_cls, tmp_path):
    upstream = {"object": "response", "status": "completed", "output": [], "usage": {}}
    mock_cls.return_value = _sync(upstream)
    item = {"type": "reasoning", "id": "keep", "encrypted_content": "foreign", "summary": None}
    response = _client(tmp_path).post("/openai/v1/responses", json={
        "model": "deepseek-v4-pro", "input": [item], "store": True,
    })
    assert response.status_code == 200
    call = mock_cls.return_value.post.call_args
    assert call.args[0] == "https://api.deepseek.com/responses"
    sent = json.loads(call.kwargs["content"])
    assert sent["input"] == [item] and sent["store"] is True


@patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"})
@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_explicit_model_routes_native_anthropic(mock_cls, tmp_path):
    upstream = {"id": "msg", "type": "message", "content": [], "usage": {}}
    mock_cls.return_value = _sync(upstream)
    response = _client(tmp_path).post("/v1/messages", json={
        "model": "deepseek-v4-pro", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1}, "future": True,
    })
    assert response.status_code == 200 and response.json() == upstream
    call = mock_cls.return_value.post.call_args
    assert call.args[0] == "https://api.deepseek.com/anthropic/v1/messages"
    sent = json.loads(call.kwargs["content"])
    assert sent["thinking"]["budget_tokens"] == 1 and sent["future"] is True
    assert call.kwargs["headers"]["x-api-key"] == "secret"
