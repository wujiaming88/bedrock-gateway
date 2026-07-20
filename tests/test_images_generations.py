"""Tests for /openai/v1/images/generations passthrough.

Covers the new OpenAI Images dialect, Azure prefix routing, endpoint guards,
error paths, and dashboard metrics inclusion. The endpoint is intentionally
sync-only and is primarily exposed through ``azure/<deployment>``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from bedrock_gateway.auth import AuthConfig
from bedrock_gateway.config import (
    AzureResource,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    _DEFAULT_MODELS,
    _parse_models,
)
from bedrock_gateway.dashboard.middleware import _LLM_PATHS
from bedrock_gateway.dashboard.metrics import MetricsCollector
from bedrock_gateway.dashboard.middleware import metrics_middleware_factory
from bedrock_gateway.models import ModelRegistry
from bedrock_gateway.providers import get_dialect
from bedrock_gateway.providers.base import Dialect
from bedrock_gateway.providers.dialect_images import ImagesPassthroughDialect
from bedrock_gateway.server import create_app

AZ_BASE = "https://my-res.cognitiveservices.azure.com/openai/v1"


@pytest.fixture
def client() -> TestClient:
    resources = {
        "r1": AzureResource(base_url=AZ_BASE, api_key="az-secret", prefix="azure"),
    }
    cfg = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=_parse_models(_DEFAULT_MODELS, resources),
        azure_resources=resources,
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


def _mock_bad_json_client():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("bad json")
    resp.text = "not-json"
    resp.content = b"not-json"
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _call(mock_cls):
    call = mock_cls.return_value.post.call_args
    url = call[0][0] if call[0] else call.kwargs.get("url")
    return url, call.kwargs.get("headers", {}), json.loads(
        call.kwargs.get("content", b"{}")
    )


def _image_body():
    return {
        "created": 123,
        "background": "opaque",
        "data": [{"b64_json": "iVBORw0KGgo="}],
    }


class TestImagesDialect:
    def test_identity_properties(self):
        d = ImagesPassthroughDialect()
        entry = ModelEntry(bedrock_id="gpt-image-2", dialect="openai-images")
        assert d.name == "openai-images"
        assert d.supports_stream is False
        assert d.operation_path(entry, stream=False) == "/images/generations"
        assert d.operation_path(entry, stream=True) == "/images/generations"

    def test_build_request_passthrough(self):
        d = ImagesPassthroughDialect()
        entry = ModelEntry(bedrock_id="gpt-image-2", dialect="openai-images")
        body = {"model": "gpt-image-2", "prompt": "x"}
        assert d.build_request(body, entry) is body

    def test_render_sync_passthrough_and_log_info(self):
        d = ImagesPassthroughDialect()
        body = _image_body() | {"status": "completed"}
        rendered, log = d.render_sync(body, "gpt-image-2")
        assert rendered is body
        assert log == {
            "input_tokens": "?",
            "output_tokens": "?",
            "finish": "completed",
        }

    def test_render_sync_default_finish(self):
        d = ImagesPassthroughDialect()
        _, log = d.render_sync(_image_body(), "gpt-image-2")
        assert log["finish"] == "completed"

    @pytest.mark.asyncio
    async def test_stream_methods_not_implemented(self):
        d = ImagesPassthroughDialect()

        async def byte_iter():
            yield b"x"

        with pytest.raises(NotImplementedError):
            async for _ in d.transform_stream(byte_iter(), "m", "id"):
                pass
        with pytest.raises(NotImplementedError):
            d.stream_error("x", 500)

    def test_registered_in_provider_registry(self):
        entry = ModelEntry(bedrock_id="gpt-image-2", dialect="openai-images")
        assert isinstance(get_dialect(entry), ImagesPassthroughDialect)


class TestImagesEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_prefix_routes_to_azure_images_url(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_image_body())
        resp = client.post("/openai/v1/images/generations", json={
            "model": "azure/gpt-image-2",
            "prompt": "a red square",
            "size": "1024x1024",
            "n": 1,
            "quality": "high",
            "background": "opaque",
        })
        assert resp.status_code == 200
        assert resp.json()["data"][0]["b64_json"] == "iVBORw0KGgo="
        url, headers, body = _call(mock_cls)
        assert url == AZ_BASE + "/images/generations"
        assert headers["api-key"] == "az-secret"
        assert body == {
            "model": "gpt-image-2",
            "prompt": "a red square",
            "size": "1024x1024",
            "n": 1,
            "quality": "high",
            "background": "opaque",
        }

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_explicit_model_entry_routes(self, mock_cls):
        resources = {"r1": AzureResource(base_url=AZ_BASE, api_key="az-secret")}
        cfg = GatewayConfig(
            auth=AuthConfig(mode="bearer_token", bearer_token="t"),
            region="us-east-1",
            server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
            retry=RetryConfig(max_retries=1, base_delay=0.01),
            models={
                "image-model": ModelEntry(
                    bedrock_id="gpt-image-2",
                    transport="azure",
                    dialect="openai-images",
                    deployment="gpt-image-2",
                    azure_endpoint=AZ_BASE,
                    azure_api_key="az-secret",
                )
            },
            azure_resources=resources,
        )
        explicit_client = TestClient(create_app(cfg))
        mock_cls.return_value = _mock_sync_client(_image_body())
        resp = explicit_client.post("/openai/v1/images/generations", json={
            "model": "image-model", "prompt": "x"
        })
        assert resp.status_code == 200
        assert _call(mock_cls)[2]["model"] == "gpt-image-2"

    def test_invalid_json(self, client):
        resp = client.post(
            "/openai/v1/images/generations",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Invalid JSON body"

    def test_stream_rejected(self, client):
        resp = client.post("/openai/v1/images/generations", json={
            "model": "azure/gpt-image-2", "prompt": "x", "stream": True
        })
        assert resp.status_code == 400
        assert "does not support streaming" in resp.json()["error"]["message"]

    def test_responses_model_rejected(self, client):
        resp = client.post("/openai/v1/images/generations", json={
            "model": "gpt-5.5", "prompt": "x"
        })
        assert resp.status_code == 400
        assert "/openai/v1/images/generations" in resp.json()["error"]["message"]
        assert "/openai/v1/responses" in resp.json()["error"]["message"]

    def test_claude_model_rejected(self, client):
        resp = client.post("/openai/v1/images/generations", json={
            "model": "claude-haiku", "prompt": "x"
        })
        assert resp.status_code == 400
        assert "not available" in resp.json()["error"]["message"]

    def test_unknown_model_rejected(self, client):
        resp = client.post("/openai/v1/images/generations", json={
            "model": "not-a-model", "prompt": "x"
        })
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_upstream_error_surfaced(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(
            {"error": {"message": "unsupported", "type": "invalid_request_error"}},
            status=400,
        )
        resp = client.post("/openai/v1/images/generations", json={
            "model": "azure/gpt-image-2", "prompt": "x"
        })
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bad_upstream_json_returns_502(self, mock_cls, client):
        mock_cls.return_value = _mock_bad_json_client()
        resp = client.post("/openai/v1/images/generations", json={
            "model": "azure/gpt-image-2", "prompt": "x"
        })
        assert resp.status_code == 502
        assert "malformed" in resp.json()["error"]["message"]


class TestImagesModelResolution:
    def test_prefix_resolution_uses_images_dialect(self, client):
        # Exercise the same ModelRegistry branch the endpoint uses.
        resources = {
            "r1": AzureResource(base_url=AZ_BASE, api_key="az-secret", prefix="azure"),
        }
        registry = ModelRegistry(GatewayConfig(azure_resources=resources))
        entry = registry.resolve_prefixed("azure/gpt-image-2", "openai-images")
        assert entry is not None
        assert entry.transport == "azure"
        assert entry.dialect == "openai-images"
        assert entry.deployment == "gpt-image-2"


class TestImagesMetrics:
    def test_images_path_is_llm_path(self):
        assert "/openai/v1/images/generations" in _LLM_PATHS

    def test_images_request_is_recorded_with_model_and_zero_tokens(self):
        collector = MetricsCollector()
        app = FastAPI()
        app.middleware("http")(metrics_middleware_factory(collector))

        @app.post("/openai/v1/images/generations")
        async def handler():
            return JSONResponse(_image_body())

        c = TestClient(app)
        resp = c.post("/openai/v1/images/generations", json={
            "model": "azure/gpt-image-2", "prompt": "x"
        })
        assert resp.status_code == 200
        recs = collector.recent_requests(limit=1)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["path"] == "/openai/v1/images/generations"
        assert rec["model"] == "azure/gpt-image-2"
        assert rec["prompt_tokens"] == 0
        assert rec["completion_tokens"] == 0
