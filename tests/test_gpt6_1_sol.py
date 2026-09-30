"""Bedrock mantle GPT-6.1 Sol registration and routing."""

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

ALIAS = "gpt-6.1-sol"
BEDROCK_ID = "openai.gpt-6.1-sol"

# The dot in "6.1" breaks the `-6-` split used by the GPT-6 test's alias
# template, so list the variants explicitly here.
ALIAS_VARIANTS = [
    "gpt-6.1-sol",
    "gpt-6-1-sol",
    "gpt6.1-sol",
    "openai.gpt-6.1-sol",
    "openai-gpt-6.1-sol",
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


class TestGPT61SolConfig:
    def test_default_model_entry(self):
        e = _DEFAULT_MODELS[ALIAS]
        assert e["bedrock_id"] == BEDROCK_ID
        assert e["endpoint"] == "mantle"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 1_000_000
        assert e["max_output"] == 131_072
        assert e["region"] == "us-east-1"

    def test_registry_resolves_canonical(self):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve(ALIAS) == BEDROCK_ID
        entry = reg.get_entry(ALIAS)
        assert entry is not None
        assert entry.endpoint == "mantle"
        assert entry.dialect == "openai-responses"
        assert entry.region == "us-east-1"

    @pytest.mark.parametrize("variant", ALIAS_VARIANTS)
    def test_alias_variants_point_to_valid_default(self, variant):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert _MODEL_ALIASES[variant] == ALIAS
        assert reg.resolve(variant) == BEDROCK_ID

    def test_chat_variant_entry(self):
        chat = ALIAS + "-chat"
        e = _DEFAULT_MODELS[chat]
        assert e["bedrock_id"] == BEDROCK_ID
        assert e["endpoint"] == "mantle"
        assert e["dialect"] == "openai-chat"
        assert e["context_length"] == 1_000_000
        assert e["max_output"] == 131_072
        assert e["region"] == "us-east-1"

    def test_chat_variant_resolves(self):
        chat = ALIAS + "-chat"
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve(chat) == BEDROCK_ID
        entry = reg.get_entry(chat)
        assert entry is not None
        assert entry.dialect == "openai-chat"
        assert entry.endpoint == "mantle"
        assert entry.region == "us-east-1"


class TestGPT61SolResponsesEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_bedrock_mantle_us_east_1(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": ALIAS,
            "input": "ping",
            "max_output_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
        assert _sent(mock_cls)["model"] == BEDROCK_ID

    def test_rejected_on_chat_completions(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]


class TestGPT61SolChatEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_chat_routes_to_bedrock_mantle_us_east_1(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_chat_body(BEDROCK_ID))
        resp = client.post("/v1/chat/completions", json={
            "model": ALIAS + "-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/chat/completions"
        sent = _sent(mock_cls)
        assert sent["model"] == BEDROCK_ID
        assert sent["max_completion_tokens"] == 16
        assert "max_tokens" not in sent
