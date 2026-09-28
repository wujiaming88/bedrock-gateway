"""Bedrock mantle GPT-6 Sol and Luna registration and routing."""

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

MODELS = [
    ("gpt-6-sol", "openai.gpt-6-sol"),
    ("gpt-6-luna", "openai.gpt-6-luna"),
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


class TestGPT6SolLunaConfig:
    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    def test_default_model_entry(self, alias, bedrock_id):
        e = _DEFAULT_MODELS[alias]
        assert e["bedrock_id"] == bedrock_id
        assert e["endpoint"] == "mantle"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 1_050_000
        assert e["max_output"] == 128_000
        assert e["region"] == "us-east-1"

    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    def test_registry_resolves_canonical(self, alias, bedrock_id):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve(alias) == bedrock_id
        entry = reg.get_entry(alias)
        assert entry is not None
        assert entry.endpoint == "mantle"
        assert entry.dialect == "openai-responses"
        assert entry.region == "us-east-1"

    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    @pytest.mark.parametrize("variant", [
        "gpt-6-{name}",
        "gpt-6{name}",
        "gpt6-{name}",
        "openai.gpt-6-{name}",
        "openai-gpt-6-{name}",
    ])
    def test_alias_variants_point_to_valid_default(self, alias, bedrock_id, variant):
        name = alias.split("-6-", 1)[1]
        variant = variant.format(name=name)
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        expected = reg.resolve(alias)
        assert _MODEL_ALIASES[variant] == alias
        assert reg.resolve(variant) == expected

    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    def test_chat_variant_entry(self, alias, bedrock_id):
        chat = alias + "-chat"
        e = _DEFAULT_MODELS[chat]
        assert e["bedrock_id"] == bedrock_id
        assert e["endpoint"] == "mantle"
        assert e["dialect"] == "openai-chat"
        assert e["context_length"] == 1_050_000
        assert e["max_output"] == 128_000
        assert e["region"] == "us-east-1"

    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    def test_chat_variant_resolves(self, alias, bedrock_id):
        chat = alias + "-chat"
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve(chat) == bedrock_id
        entry = reg.get_entry(chat)
        assert entry is not None
        assert entry.dialect == "openai-chat"
        assert entry.endpoint == "mantle"
        assert entry.region == "us-east-1"


class TestGPT6SolLunaResponsesEndpoint:
    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_bedrock_mantle_us_east_1(self, mock_cls, alias, bedrock_id, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(bedrock_id))
        resp = client.post("/openai/v1/responses", json={
            "model": alias,
            "input": "ping",
            "max_output_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
        assert _sent(mock_cls)["model"] == bedrock_id

    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    def test_rejected_on_chat_completions(self, alias, bedrock_id, client):
        resp = client.post("/v1/chat/completions", json={
            "model": alias,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]


class TestGPT6SolLunaChatEndpoint:
    @pytest.mark.parametrize("alias,bedrock_id", MODELS)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_chat_routes_to_bedrock_mantle_us_east_1(self, mock_cls, alias, bedrock_id, client):
        mock_cls.return_value = _mock_sync_client(_chat_body(bedrock_id))
        resp = client.post("/v1/chat/completions", json={
            "model": alias + "-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/chat/completions"
        sent = _sent(mock_cls)
        assert sent["model"] == bedrock_id
        assert sent["max_completion_tokens"] == 16
        assert "max_tokens" not in sent
