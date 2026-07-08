"""
Tests for Azure OpenAI support (Transport × Dialect two-axis architecture).

Covers:
  1. Config — two-layer azure_resources + model refs, resolution, error paths.
  2. AzureTransport — URL assembly (query-string preservation), api-key auth.
  3. ChatPassthroughDialect — operation path, verbatim render, SSE passthrough,
     UTF-8 boundary safety, stream_error.
  4. Transport/dialect selection for Azure entries.
  5. Integration — /openai/v1/responses and /v1/chat/completions routing an
     Azure model: mocked upstream, api-key header sent, model→deployment swap,
     content_filter passthrough, guards.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    AzureResource,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    _parse_azure_resources,
    _parse_models,
)
from bedrock_gateway.providers import (
    AzureTransport,
    ChatPassthroughDialect,
    ResponsesPassthroughDialect,
    get_dialect,
    get_transport,
)
from bedrock_gateway.server import create_app

AZ_BASE = "https://my-res.cognitiveservices.azure.com/openai"
AZ_BASE_VER = AZ_BASE + "?api-version=2025-04-01-preview"


# ---------------------------------------------------------------------------
# 1. Config: two-layer resource + model reference
# ---------------------------------------------------------------------------

class TestAzureConfig:
    def test_parse_resources(self):
        res = _parse_azure_resources({
            "r1": {"base_url": AZ_BASE + "/", "api_key": "k1"},
        })
        assert isinstance(res["r1"], AzureResource)
        assert res["r1"].base_url == AZ_BASE  # trailing slash stripped
        assert res["r1"].api_key == "k1"

    def test_model_resolves_resource_ref(self):
        res = _parse_azure_resources({"r1": {"base_url": AZ_BASE, "api_key": "k1"}})
        models = _parse_models(
            {"az": {"transport": "azure", "dialect": "openai-responses",
                    "azure_resource": "r1", "deployment": "gpt-5"}},
            res,
        )
        e = models["az"]
        assert e.transport == "azure"
        assert e.dialect == "openai-responses"
        assert e.azure_endpoint == AZ_BASE
        assert e.azure_api_key == "k1"
        assert e.deployment == "gpt-5"

    def test_deployment_defaults_to_alias(self):
        res = _parse_azure_resources({"r1": {"base_url": AZ_BASE, "api_key": "k"}})
        models = _parse_models(
            {"my-dep": {"transport": "azure", "dialect": "openai-chat",
                        "azure_resource": "r1"}},
            res,
        )
        assert models["my-dep"].deployment == "my-dep"

    def test_unknown_resource_ref_raises(self):
        with pytest.raises(ValueError, match="unknown azure_resource"):
            _parse_models(
                {"az": {"azure_resource": "nope", "deployment": "x"}},
                {},
            )

    def test_azure_ref_forces_transport(self):
        """Even if transport omitted, a resource ref forces transport=azure."""
        res = _parse_azure_resources({"r1": {"base_url": AZ_BASE, "api_key": "k"}})
        models = _parse_models(
            {"az": {"dialect": "openai-responses", "azure_resource": "r1"}},
            res,
        )
        assert models["az"].transport == "azure"


# ---------------------------------------------------------------------------
# 2. AzureTransport unit
# ---------------------------------------------------------------------------

class TestAzureTransport:
    transport = AzureTransport()

    def _entry(self, endpoint=AZ_BASE):
        return ModelEntry(bedrock_id="gpt-5", transport="azure",
                          azure_endpoint=endpoint, azure_api_key="secret",
                          deployment="gpt-5")

    def test_url_plain_base(self):
        e = self._entry(AZ_BASE)
        assert self.transport.build_url("/responses", "us-east-1", e) == (
            AZ_BASE + "/responses"
        )

    def test_url_preserves_query_string(self):
        """A base with ?api-version=... must keep the query at the very end."""
        e = self._entry(AZ_BASE_VER)
        url = self.transport.build_url("/responses", "us-east-1", e)
        assert url == (
            AZ_BASE + "/responses?api-version=2025-04-01-preview"
        )

    def test_url_chat_operation(self):
        e = self._entry(AZ_BASE)
        assert self.transport.build_url("/chat/completions", "x", e) == (
            AZ_BASE + "/chat/completions"
        )

    def test_auth_headers_use_api_key(self):
        e = self._entry()
        h = self.transport.auth_headers(e)
        assert h["api-key"] == "secret"
        # Azure does NOT use Authorization: Bearer
        assert "Authorization" not in h


class TestTransportOwnsApiRoot:
    """Regression guard for the two-axis boundary: dialects emit a BARE
    operation; each transport prepends its own /openai/v1 root. The final
    composed URLs must match what each cloud actually serves.
    """

    def test_bedrock_mantle_adds_openai_v1_root(self):
        from bedrock_gateway.providers import BedrockTransport
        from bedrock_gateway.providers.openai_responses import (
            ResponsesPassthroughDialect,
        )
        t, d = BedrockTransport(), ResponsesPassthroughDialect()
        e = ModelEntry(bedrock_id="openai.gpt-5.5", transport="bedrock",
                      endpoint="mantle", dialect="openai-responses")
        url = t.build_url(d.operation_path(e, False), "us-east-1", e)
        assert url == (
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
        )

    def test_azure_base_already_has_root(self):
        from bedrock_gateway.providers.openai_responses import (
            ResponsesPassthroughDialect,
        )
        t, d = AzureTransport(), ResponsesPassthroughDialect()
        # production-style base ending in /openai/v1
        base = "https://r.cognitiveservices.azure.com/openai/v1"
        e = ModelEntry(bedrock_id="gpt-5", transport="azure",
                      azure_endpoint=base, dialect="openai-responses")
        url = t.build_url(d.operation_path(e, False), "x", e)
        assert url == base + "/responses"


# ---------------------------------------------------------------------------
# 3. ChatPassthroughDialect unit
# ---------------------------------------------------------------------------

class TestChatPassthroughDialect:
    dialect = ChatPassthroughDialect()

    def _azure_entry(self):
        return ModelEntry(bedrock_id="gpt-5", transport="azure",
                          dialect="openai-chat", deployment="gpt-5")

    def test_operation_path_is_bare_and_cloud_agnostic(self):
        """operation_path returns only the bare operation, regardless of cloud
        — the transport owns the /openai/v1 root prefix. This keeps the two
        axes orthogonal (no `if transport` inside the dialect)."""
        azure = self._azure_entry()
        bedrock = ModelEntry(bedrock_id="x", transport="bedrock",
                            endpoint="mantle", dialect="openai-chat")
        assert self.dialect.operation_path(azure, False) == "/chat/completions"
        assert self.dialect.operation_path(bedrock, False) == "/chat/completions"

    def test_render_sync_verbatim_openai_usage(self):
        upstream = {
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        body, log = self.dialect.render_sync(upstream, "azure-gpt-5-chat")
        assert body is upstream  # passthrough
        assert log["input_tokens"] == 5
        assert log["output_tokens"] == 3
        assert log["finish"] == "stop"

    def test_render_sync_tolerates_missing_fields(self):
        body, log = self.dialect.render_sync({}, "m")
        assert log["input_tokens"] == "?"
        assert log["finish"] == "?"

    async def test_transform_stream_passthrough(self):
        async def it():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        out = "".join([c async for c in self.dialect.transform_stream(
            it(), "m", "id")])
        assert '"content":"hi"' in out
        assert "[DONE]" in out

    async def test_transform_stream_utf8_boundary(self):
        raw = 'data: {"delta":"策略"}\n\n'.encode("utf-8")
        mid = len(raw) // 2

        async def it():
            yield raw[:mid]
            yield raw[mid:]
        out = "".join([c async for c in self.dialect.transform_stream(
            it(), "m", "id")])
        assert "策略" in out
        assert "�" not in out

    def test_stream_error(self):
        s = self.dialect.stream_error("boom", 502)
        assert '"message": "boom"' in s
        assert '"code": 502' in s
        assert s.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# 4. Transport/dialect selection for Azure
# ---------------------------------------------------------------------------

class TestAzureSelection:
    def test_azure_responses_selection(self):
        e = ModelEntry(bedrock_id="gpt-5", transport="azure",
                      dialect="openai-responses")
        assert isinstance(get_transport(e), AzureTransport)
        assert isinstance(get_dialect(e), ResponsesPassthroughDialect)

    def test_azure_chat_selection(self):
        e = ModelEntry(bedrock_id="gpt-5", transport="azure",
                      dialect="openai-chat")
        assert isinstance(get_transport(e), AzureTransport)
        assert isinstance(get_dialect(e), ChatPassthroughDialect)


# ---------------------------------------------------------------------------
# 5. Integration — routing an Azure model through the server
# ---------------------------------------------------------------------------

def _azure_config() -> GatewayConfig:
    resources = {"r1": AzureResource(base_url=AZ_BASE_VER, api_key="az-secret")}
    models = _parse_models({
        "azure-gpt-5": {"transport": "azure", "dialect": "openai-responses",
                        "azure_resource": "r1", "deployment": "gpt-5-dep"},
        "azure-gpt-5-chat": {"transport": "azure", "dialect": "openai-chat",
                             "azure_resource": "r1", "deployment": "gpt-5-dep"},
    }, resources)
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=models,
        azure_resources=resources,
    )


@pytest.fixture
def az_client() -> TestClient:
    return TestClient(create_app(_azure_config()))


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


def _call(mock_cls):
    """Return (url, headers, body) of the mocked upstream POST."""
    c = mock_cls.return_value.post.call_args
    url = c[0][0] if c[0] else c.kwargs.get("url")
    return url, c.kwargs.get("headers", {}), json.loads(
        c.kwargs.get("content", b"{}"))


def _responses_body():
    return {"object": "response", "status": "completed", "model": "gpt-5-dep",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "hi"}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "content_filters": [{"blocked": False}]}


def _chat_body():
    return {"object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"},
                         "content_filter_results": {"hate": {"filtered": False}}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


class TestAzureResponsesIntegration:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_azure_url(self, mock_cls, az_client):
        mock_cls.return_value = _mock_sync_client(_responses_body())
        r = az_client.post("/openai/v1/responses",
                          json={"model": "azure-gpt-5", "input": "hi"})
        assert r.status_code == 200
        url, headers, body = _call(mock_cls)
        assert url == AZ_BASE + "/responses?api-version=2025-04-01-preview"
        # api-key auth header, not Bearer
        assert headers.get("api-key") == "az-secret"
        # model swapped to the deployment name
        assert body["model"] == "gpt-5-dep"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_content_filters_passed_through(self, mock_cls, az_client):
        mock_cls.return_value = _mock_sync_client(_responses_body())
        r = az_client.post("/openai/v1/responses",
                          json={"model": "azure-gpt-5", "input": "hi"})
        assert "content_filters" in r.json()

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_input_image_preserved(self, mock_cls, az_client):
        mock_cls.return_value = _mock_sync_client(_responses_body())
        az_client.post("/openai/v1/responses", json={
            "model": "azure-gpt-5",
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAA"}]}],
        })
        _, _, body = _call(mock_cls)
        blk = body["input"][0]["content"][0]
        assert blk["type"] == "input_image"


class TestAzureChatIntegration:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_azure_chat_url(self, mock_cls, az_client):
        mock_cls.return_value = _mock_sync_client(_chat_body())
        r = az_client.post("/v1/chat/completions", json={
            "model": "azure-gpt-5-chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 200
        url, headers, body = _call(mock_cls)
        assert url == AZ_BASE + "/chat/completions?api-version=2025-04-01-preview"
        assert headers.get("api-key") == "az-secret"
        assert body["model"] == "gpt-5-dep"
        # passthrough: messages forwarded untouched (not converted to Anthropic)
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_response_is_openai_chat_shape(self, mock_cls, az_client):
        mock_cls.return_value = _mock_sync_client(_chat_body())
        r = az_client.post("/v1/chat/completions", json={
            "model": "azure-gpt-5-chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        b = r.json()
        assert b["object"] == "chat.completion"
        assert b["choices"][0]["message"]["content"] == "hi"


class TestAzureGuards:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_responses_model_rejected_on_chat(self, mock_cls, az_client):
        # azure-gpt-5 is a Responses model; misrouting to chat → 400
        r = az_client.post("/v1/chat/completions", json={
            "model": "azure-gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 400
        assert "/openai/v1/responses" in r.json()["error"]["message"]

    def test_chat_model_rejected_on_responses(self, az_client):
        # azure-gpt-5-chat is a Chat model; misrouting to responses → 400
        r = az_client.post("/openai/v1/responses", json={
            "model": "azure-gpt-5-chat", "input": "hi"})
        assert r.status_code == 400
        assert "/v1/chat/completions" in r.json()["error"]["message"]
