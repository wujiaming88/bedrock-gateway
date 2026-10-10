"""Z.AI GLM 5.3 — Bedrock *cross-region* inference via the runtime host.

Same class as Grok 4.7 / Kimi K3: a Bedrock cross-region inference profile
(Geo-US / Global), in-region unsupported, so it routes through ``endpoint:
runtime`` with the OpenAI-compatible ``/openai/v1`` root and the mandatory
``us.`` inference-profile id. Unlike Grok/Kimi, the GLM 5.3 card exposes both
the Responses and Chat Completions APIs, so a ``-chat`` variant is registered
alongside the Responses primary.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    _DEFAULT_MODELS,
    _MODEL_ALIASES,
    AuthConfig,
    GatewayConfig,
    RetryConfig,
    ServerConfig,
    _parse_models,
)
from bedrock_gateway.models import ModelRegistry
from bedrock_gateway.server import create_app

ALIAS = "glm-5.3"
BEDROCK_ID = "us.zai.glm-5.3"

ALIAS_VARIANTS = [
    "glm5.3",
    "glm-5-3",
    "zai.glm-5.3",
    "zai-glm-5.3",
    "us.zai.glm-5.3",
    "global.zai.glm-5.3",
]


@pytest.fixture
def client() -> TestClient:
    cfg = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=_parse_models(_DEFAULT_MODELS),
    )
    return TestClient(create_app(cfg))


def _mock_sync_client(response_data: dict, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_data
    resp.text = json.dumps(response_data)
    resp.content = json.dumps(response_data).encode()
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _responses_body(model: str) -> dict:
    return {
        "id": "resp_x",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _chat_body(model: str) -> dict:
    return {
        "id": "chatcmpl_x",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _sent(mock_cls) -> dict:
    return json.loads(mock_cls.return_value.post.call_args.kwargs["content"])


class TestGLM53Config:
    def test_default_model_entry(self):
        e = _DEFAULT_MODELS[ALIAS]
        assert e["bedrock_id"] == BEDROCK_ID
        assert e["endpoint"] == "runtime"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 1_000_000
        assert e["max_output"] == 131_072

    def test_chat_variant_entry(self):
        e = _DEFAULT_MODELS[ALIAS + "-chat"]
        assert e["bedrock_id"] == BEDROCK_ID
        assert e["endpoint"] == "runtime"
        assert e["dialect"] == "openai-chat"
        assert e["context_length"] == 1_000_000
        assert e["max_output"] == 131_072

    def test_registry_resolves_canonical(self):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve(ALIAS) == BEDROCK_ID
        assert reg.resolve(ALIAS + "-chat") == BEDROCK_ID
        entry = reg.get_entry(ALIAS)
        assert entry is not None
        assert entry.endpoint == "runtime"
        assert entry.dialect == "openai-responses"

    @pytest.mark.parametrize("variant", ALIAS_VARIANTS)
    def test_alias_variants_point_to_valid_default(self, variant):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert _MODEL_ALIASES[variant] == ALIAS
        assert reg.resolve(variant) == BEDROCK_ID


class TestGLM53ResponsesEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_runtime_url(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": ALIAS, "input": "ping", "max_output_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
        )
        assert _sent(mock_cls)["model"] == BEDROCK_ID

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_full_inference_id_resolves(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": "us.zai.glm-5.3", "input": "ping",
        })
        assert resp.status_code == 200
        assert _sent(mock_cls)["model"] == BEDROCK_ID

    def test_rejected_on_chat_completions(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]


class TestGLM53ChatEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_chat_routes_to_runtime_url(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_chat_body(BEDROCK_ID))
        resp = client.post("/v1/chat/completions", json={
            "model": ALIAS + "-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"
        )
        sent = _sent(mock_cls)
        assert sent["model"] == BEDROCK_ID
        assert sent["max_completion_tokens"] == 16

    def test_rejected_on_responses(self, client):
        resp = client.post("/openai/v1/responses", json={
            "model": ALIAS + "-chat",
            "input": "ping",
        })
        assert resp.status_code == 400
        assert "/v1/chat/completions" in resp.json()["error"]["message"]
