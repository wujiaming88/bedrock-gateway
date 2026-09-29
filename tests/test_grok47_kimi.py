"""Grok 4.7 + Kimi K3 — Bedrock *cross-region* inference via the runtime host.

Unlike the mantle models (GPT-5.x, grok-4.3/4.6), these two are Bedrock
cross-region inference profiles (Geo-US / Global); in-region routing is
unsupported, so they route through ``endpoint: runtime`` while still speaking
the OpenAI-compatible API under ``/openai/v1``. The model id must carry the
``us.`` inference-profile prefix. This is the first ``runtime`` + OpenAI-dialect
combination, so the URL-construction tests here also guard the transport
refactor that keyed the ``/openai/v1`` root off the *dialect* rather than the
endpoint hint.
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
    ModelEntry,
    RetryConfig,
    ServerConfig,
    _parse_models,
)
from bedrock_gateway.models import ModelRegistry
from bedrock_gateway.providers import (
    BedrockTransport,
    ChatPassthroughDialect,
    ResponsesPassthroughDialect,
    get_dialect,
    get_transport,
)
from bedrock_gateway.server import create_app

GROK47_ALIAS = "grok-4.7"
GROK47_BEDROCK_ID = "us.xai.grok-4.7"
KIMI_K3_ALIAS = "kimi-k3"
KIMI_K3_BEDROCK_ID = "us.moonshotai.kimi-k3"


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


# ---------------------------------------------------------------------------
# Registration & resolution
# ---------------------------------------------------------------------------

class TestGrok47KimiRegistration:
    def test_grok47_registered(self):
        e = _DEFAULT_MODELS[GROK47_ALIAS]
        assert e["bedrock_id"] == GROK47_BEDROCK_ID
        assert e["endpoint"] == "runtime"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 500_000
        assert e["max_output"] == 131_072

    def test_kimi_k3_registered(self):
        e = _DEFAULT_MODELS[KIMI_K3_ALIAS]
        assert e["bedrock_id"] == KIMI_K3_BEDROCK_ID
        assert e["endpoint"] == "runtime"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 1_000_000
        assert e["max_output"] == 131_072

    @pytest.mark.parametrize(
        "alias", ["grok4.7", "grok-4-7", "xai.grok-4.7", "xai-grok-4.7"]
    )
    def test_grok47_aliases_resolve(self, alias):
        assert _MODEL_ALIASES[alias] == GROK47_ALIAS

    @pytest.mark.parametrize(
        "alias", ["moonshotai.kimi-k3", "moonshotai-kimi-k3"]
    )
    def test_kimi_k3_aliases_resolve(self, alias):
        assert _MODEL_ALIASES[alias] == KIMI_K3_ALIAS

    def test_registry_resolves_canonical_bedrock_ids(self):
        reg = ModelRegistry(GatewayConfig(models=_parse_models(_DEFAULT_MODELS)))
        assert reg.resolve(GROK47_ALIAS) == GROK47_BEDROCK_ID
        assert reg.resolve(KIMI_K3_ALIAS) == KIMI_K3_BEDROCK_ID

    def test_grok47_selects_responses_dialect(self):
        e = _parse_models(_DEFAULT_MODELS)[GROK47_ALIAS]
        assert get_dialect(e).name == "openai-responses"
        assert get_transport(e).name == "bedrock"

    def test_kimi_k3_selects_responses_dialect(self):
        e = _parse_models(_DEFAULT_MODELS)[KIMI_K3_ALIAS]
        assert get_dialect(e).name == "openai-responses"
        assert get_transport(e).name == "bedrock"


# ---------------------------------------------------------------------------
# Transport URL — runtime host + OpenAI dialect (the new combination)
# ---------------------------------------------------------------------------

class TestRuntimeOpenaiUrl:
    transport = BedrockTransport()

    def _entry(self, dialect: str) -> ModelEntry:
        return ModelEntry(
            bedrock_id="us.example.model",
            endpoint="runtime",
            transport="bedrock",
            dialect=dialect,
        )

    def test_responses_dialect_urls_runtime_with_openai_root(self):
        e = self._entry("openai-responses")
        op = ResponsesPassthroughDialect().operation_path(e, False)
        url = self.transport.build_url(op, "us-east-1", e)
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
        )

    def test_chat_dialect_urls_runtime_with_openai_root(self):
        e = self._entry("openai-chat")
        op = ChatPassthroughDialect().operation_path(e, False)
        url = self.transport.build_url(op, "us-east-1", e)
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"
        )

    def test_images_dialect_urls_runtime_with_openai_root(self):
        e = self._entry("openai-images")
        url = self.transport.build_url("/images/generations", "us-east-1", e)
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/images/generations"
        )

    def test_region_inherits_global_when_model_region_empty(self):
        e = self._entry("openai-responses")
        op = ResponsesPassthroughDialect().operation_path(e, False)
        url = self.transport.build_url(op, "us-west-2", e)
        assert "bedrock-runtime.us-west-2.amazonaws.com" in url

    def test_native_dialect_on_runtime_has_no_openai_root(self):
        """Anthropic dialect keeps its full native path — the refactor must not
        prepend /openai/v1 to non-OpenAI dialects."""
        e = ModelEntry(bedrock_id="us.anthropic.claude-x")
        url = self.transport.build_url("/model/us.anthropic.claude-x/invoke", "us-east-1", e)
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/us.anthropic.claude-x/invoke"
        )

    def test_embeddings_dialect_on_runtime_has_no_openai_root(self):
        e = ModelEntry(
            bedrock_id="cohere.embed-v4:0",
            endpoint="runtime",
            dialect="openai-embeddings",
        )
        url = self.transport.build_url("/model/cohere.embed-v4:0/invoke", "us-east-1", e)
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/cohere.embed-v4:0/invoke"
        )


# ---------------------------------------------------------------------------
# End-to-end routing
# ---------------------------------------------------------------------------

class TestGrok47Endpoints:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_runtime_url(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(GROK47_BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": GROK47_ALIAS, "input": "ping",
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
        )
        assert _sent(mock_cls)["model"] == GROK47_BEDROCK_ID

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_alias_resolves_to_runtime(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(GROK47_BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": "grok4.7", "input": "ping",
        })
        assert resp.status_code == 200
        assert _sent(mock_cls)["model"] == GROK47_BEDROCK_ID

    def test_rejected_on_chat_completions(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": GROK47_ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]

    def test_models_lists_grok47_metadata(self, client):
        resp = client.get("/v1/models")
        model = next(item for item in resp.json()["data"] if item["id"] == GROK47_ALIAS)
        assert model["context_length"] == 500_000
        assert model["max_output_tokens"] == 131_072


class TestKimiK3Endpoints:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_runtime_url(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(KIMI_K3_BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": KIMI_K3_ALIAS, "input": "ping",
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"
        )
        assert _sent(mock_cls)["model"] == KIMI_K3_BEDROCK_ID

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_alias_resolves_to_runtime(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body(KIMI_K3_BEDROCK_ID))
        resp = client.post("/openai/v1/responses", json={
            "model": "moonshotai.kimi-k3", "input": "ping",
        })
        assert resp.status_code == 200
        assert _sent(mock_cls)["model"] == KIMI_K3_BEDROCK_ID

    def test_rejected_on_chat_completions(self, client):
        """A Responses-dialect model is not served by /v1/chat/completions."""
        resp = client.post("/v1/chat/completions", json={
            "model": KIMI_K3_ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]

    def test_models_lists_kimi_k3_metadata(self, client):
        resp = client.get("/v1/models")
        model = next(item for item in resp.json()["data"] if item["id"] == KIMI_K3_ALIAS)
        assert model["context_length"] == 1_000_000
        assert model["max_output_tokens"] == 131_072
