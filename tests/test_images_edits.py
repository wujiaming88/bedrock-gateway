"""Tests for Azure OpenAI /openai/v1/images/edits passthrough."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.auth import AuthConfig
from bedrock_gateway.config import AzureResource, GatewayConfig, RetryConfig, ServerConfig
from bedrock_gateway.server import create_app

AZ_BASE = "https://my-res.cognitiveservices.azure.com/openai/v1"
PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"
IMAGE_RESPONSE = {
    "created": 123,
    "data": [{"b64_json": "iVBORw0KGgp0ZXN0LWltYWdl"}],
    "provider_extension": {"kept": True},
}


@pytest.fixture
def client() -> TestClient:
    resources = {
        "r1": AzureResource(
            base_url=AZ_BASE,
            api_key="az-secret",
            prefix="azure",
        ),
    }
    cfg = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=2, base_delay=0),
        azure_resources=resources,
    )
    return TestClient(create_app(cfg))


def _response(data: dict = IMAGE_RESPONSE, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.content = resp.text.encode()
    return resp


def _sync_client(*responses):
    inst = AsyncMock()
    inst.post = AsyncMock(side_effect=list(responses))
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _multipart_call(mock_cls, index: int = 0):
    call = mock_cls.return_value.post.call_args_list[index]
    return call.args[0], call.kwargs["headers"], call.kwargs["content"]


def _request(model: str = "azure/gpt-image-2", **fields):
    data = {"model": model, "prompt": "make the square blue", **fields}
    return data, [("image", ("input.png", PNG, "image/png"))]


class TestImageEditsSync:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_canonical_model_routes_multipart(self, mock_cls, client):
        mock_cls.return_value = _sync_client(_response())
        data, files = _request(size="1024x1024", quality="high")
        resp = client.post("/openai/v1/images/edits", data=data, files=files)

        assert resp.status_code == 200
        assert resp.json() == IMAGE_RESPONSE
        url, headers, body = _multipart_call(mock_cls)
        assert url == AZ_BASE + "/images/edits"
        assert headers["api-key"] == "az-secret"
        assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
        assert b'name="model"' in body and b"gpt-image-2" in body
        assert b"azure/gpt-image-2" not in body
        assert b'name="image"; filename="input.png"' in body
        assert PNG in body
        assert b'name="quality"' in body and b"high" in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_custom_deployment_allowed_by_images_api(self, mock_cls, client):
        mock_cls.return_value = _sync_client(_response())
        data, files = _request("azure/prod-image")
        resp = client.post("/openai/v1/images/edits", data=data, files=files)
        assert resp.status_code == 200
        assert b"prod-image" in _multipart_call(mock_cls)[2]

    @pytest.mark.parametrize("model", ["gpt-5.5", "claude-haiku", "not-a-model"])
    def test_wrong_model_rejected(self, client, model):
        data, files = _request(model)
        assert client.post("/openai/v1/images/edits", data=data, files=files).status_code == 400

    def test_requires_multipart_boundary(self, client):
        resp = client.post("/openai/v1/images/edits", json={"model": "azure/gpt-image-2"})
        assert resp.status_code == 415

    @pytest.mark.parametrize(
        "data,files,message",
        [
            ({"prompt": "x"}, [("image", ("x.png", PNG, "image/png"))], "model"),
            ({"model": "azure/gpt-image-2"}, [("image", ("x.png", PNG, "image/png"))], "prompt"),
            ({"model": "azure/gpt-image-2", "prompt": "x"}, [("mask", ("mask.png", PNG, "image/png"))], "image"),
        ],
    )
    def test_required_parts(self, client, data, files, message):
        resp = client.post("/openai/v1/images/edits", data=data, files=files)
        assert resp.status_code == 400
        assert message in resp.json()["error"]["message"]

    def test_invalid_stream_value(self, client):
        data, files = _request(stream="yes")
        resp = client.post("/openai/v1/images/edits", data=data, files=files)
        assert resp.status_code == 400

    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_retry_replays_identical_multipart(self, mock_cls, _sleep, client):
        mock_cls.return_value = _sync_client(
            _response({"error": {"message": "busy"}}, 429), _response()
        )
        data, files = _request()
        resp = client.post("/openai/v1/images/edits", data=data, files=files)
        assert resp.status_code == 200
        first = _multipart_call(mock_cls, 0)
        second = _multipart_call(mock_cls, 1)
        assert first[1]["Content-Type"] == second[1]["Content-Type"]
        assert first[2] == second[2]
        assert PNG in second[2]


class TestImageEditsMultipartFidelity:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_image_array_field_is_accepted(self, mock_cls, client):
        mock_cls.return_value = _sync_client(_response())
        resp = client.post(
            "/openai/v1/images/edits",
            data={"model": "azure/gpt-image-2", "prompt": "x"},
            files=[("image[]", ("one.png", PNG, "image/png"))],
        )
        assert resp.status_code == 200
        assert b'name="image[]"; filename="one.png"' in _multipart_call(mock_cls)[2]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_repeated_images_and_mask_are_preserved(self, mock_cls, client):
        mock_cls.return_value = _sync_client(_response())
        files = [
            ("image", ("one.png", PNG + b"1", "image/png")),
            ("image", ("two.webp", b"RIFF-webp", "image/webp")),
            ("mask", ("mask.png", PNG + b"mask", "image/png")),
        ]
        resp = client.post(
            "/openai/v1/images/edits",
            data={"model": "azure/gpt-image-2", "prompt": "x", "future_field": "kept"},
            files=files,
        )
        assert resp.status_code == 200
        body = _multipart_call(mock_cls)[2]
        assert body.count(b'name="image"; filename=') == 2
        assert b'filename="one.png"' in body
        assert b'filename="two.webp"' in body
        assert b'name="mask"; filename="mask.png"' in body
        assert b'name="future_field"' in body and b"kept" in body
