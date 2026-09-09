"""Bedrock mantle GPT-6 Astra registration and routing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    GatewayConfig,
    RetryConfig,
    ServerConfig,
    _DEFAULT_MODELS,
    _MODEL_ALIASES,
    _parse_models,
)
from bedrock_gateway.models import ModelRegistry
from bedrock_gateway.server import create_app


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


def _sent(mock_cls) -> dict:
    return json.loads(mock_cls.return_value.post.call_args.kwargs["content"])


class TestGPT6AstraConfig:
    def test_default_model_entry(self):
        e = _DEFAULT_MODELS["gpt-6-astra"]
        assert e["bedrock_id"] == "openai.gpt-6-astra"
        assert e["endpoint"] == "mantle"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 1_050_000
        assert e["max_output"] == 128_000
        assert e["region"] == "us-west-2"

    def test_registry_resolves_canonical(self):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve("gpt-6-astra") == "openai.gpt-6-astra"
        entry = reg.get_entry("gpt-6-astra")
        assert entry is not None
        assert entry.endpoint == "mantle"
        assert entry.dialect == "openai-responses"
        assert entry.region == "us-west-2"

    @pytest.mark.parametrize("variant", [
        "gpt-6-astra",
        "gpt-6astra",
        "gpt6-astra",
        "openai.gpt-6-astra",
        "openai-gpt-6-astra",
    ])
    def test_alias_variants_point_to_valid_default(self, variant):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        expected = reg.resolve("gpt-6-astra")
        assert _MODEL_ALIASES[variant] == "gpt-6-astra"
        assert reg.resolve(variant) == expected


class TestGPT6AstraResponsesEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_bedrock_mantle_us_west_2(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body("openai.gpt-6-astra"))
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-6-astra",
            "input": "ping",
            "max_output_tokens": 16,
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"
        assert _sent(mock_cls)["model"] == "openai.gpt-6-astra"

    def test_rejected_on_chat_completions(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]
