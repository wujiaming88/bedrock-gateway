"""Tests for /v1/audio/transcriptions passthrough.

Covers the new OpenAI Audio dialect, OpenRouter prefix routing, endpoint guards,
multipart fidelity, privacy-safe logging, and error paths. Phase 1 is sync-only
and exposed through ``openrouter/<vendor>/<model>``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from bedrock_gateway.auth import AuthConfig
from bedrock_gateway.config import (
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    UpstreamResource,
    UpstreamRoute,
)
from bedrock_gateway.providers import get_dialect
from bedrock_gateway.providers.dialect_audio import AudioPassthroughDialect
from bedrock_gateway.server import create_app

OR_BASE = "https://openrouter.ai/api/v1"
AUDIO = b"fake-audio-bytes"
TRANSCRIPTION_RESPONSE = {"text": "hello world", "usage": {"audio_seconds": 1.5}}


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("OPENROUTER_TEST_SECRET", "or-test-secret")
    resource = UpstreamResource(
        prefix="openrouter",
        secret_env="OPENROUTER_TEST_SECRET",
        routes={
            "openai-audio": UpstreamRoute(
                base_url=OR_BASE,
                path="/audio/transcriptions",
                auth="bearer",
            ),
        },
    )
    cfg = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=2, base_delay=0),
        upstream_resources={"openrouter": resource},
    )
    return TestClient(create_app(cfg))


def _response(data: dict = TRANSCRIPTION_RESPONSE, status: int = 200):
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


def _request(model: str = "openrouter/openai/whisper-1", **fields):
    data = {"model": model, **fields}
    return data, [("file", ("audio.wav", AUDIO, "audio/wav"))]


class TestAudioDialect:
    def test_identity_properties(self):
        d = AudioPassthroughDialect()
        entry = ModelEntry(bedrock_id="openai/whisper-1", dialect="openai-audio")
        assert d.name == "openai-audio"
        assert d.supports_stream is True
        assert d.operation_path(entry, stream=False) == "/audio/transcriptions"
        assert d.operation_path(entry, stream=True) == "/audio/transcriptions"
        assert (
            d.operation_path(entry, stream=False, operation="transcriptions")
            == "/audio/transcriptions"
        )
        assert (
            d.operation_path(entry, stream=False, operation="translations")
            == "/audio/translations"
        )

    def test_operation_path_unknown_raises(self):
        d = AudioPassthroughDialect()
        entry = ModelEntry(bedrock_id="x", dialect="openai-audio")
        with pytest.raises(ValueError, match="Unsupported Audio operation"):
            d.operation_path(entry, False, operation="bogus")

    def test_build_request_passthrough(self):
        d = AudioPassthroughDialect()
        entry = ModelEntry(bedrock_id="openai/whisper-1", dialect="openai-audio")
        body = {"model": "openai/whisper-1", "prompt": "x"}
        assert d.build_request(body, entry) is body

    def test_render_sync_passthrough_and_log_info(self):
        d = AudioPassthroughDialect()
        body = {"text": "the transcript", "usage": {"audio_seconds": 1.0}}
        rendered, log = d.render_sync(body, "openai/whisper-1")
        assert rendered is body
        assert log == {
            "input_tokens": "?",
            "output_tokens": "?",
            "finish": "completed",
        }
        # The transcript must never leak into the access-log dict.
        assert "text" not in log

    @pytest.mark.asyncio
    async def test_stream_passthrough_and_error(self):
        d = AudioPassthroughDialect()

        async def byte_iter():
            raw = 'data: {"text": "中文"}\n\n'.encode()
            yield raw[:-2]
            yield raw[-2:]

        chunks = [chunk async for chunk in d.transform_stream(byte_iter(), "m", "id")]
        assert "".join(chunks) == 'data: {"text": "中文"}\n\n'

        # End the stream mid-character: the final flush emits the pending
        # replacement char via ``decoder.decode(b"", final=True)``.
        async def tail_iter():
            yield b"data: " + "中".encode()[:2]

        tail_chunks = [chunk async for chunk in d.transform_stream(tail_iter(), "m", "id")]
        assert "".join(tail_chunks).endswith("�")
        assert d.stream_error("x", 500).startswith("event: error\n")

    def test_registered_in_provider_registry(self):
        entry = ModelEntry(bedrock_id="openai/whisper-1", dialect="openai-audio")
        assert isinstance(get_dialect(entry), AudioPassthroughDialect)


class TestAudioEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_canonical_model_routes_multipart(self, mock_cls, client):
        mock_cls.return_value = _sync_client(_response())
        data, files = _request(language="en", response_format="json")
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)

        assert resp.status_code == 200
        assert resp.json() == TRANSCRIPTION_RESPONSE
        url, headers, body = _multipart_call(mock_cls)
        assert url == OR_BASE + "/audio/transcriptions"
        assert headers["Authorization"] == "Bearer or-test-secret"
        assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
        assert b'name="model"' in body and b"openai/whisper-1" in body
        assert b"openrouter/openai/whisper-1" not in body
        assert b'name="file"; filename="audio.wav"' in body
        assert AUDIO in body
        assert b'name="language"' in body and b"en" in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_explicit_model_entry_routes(self, mock_cls, monkeypatch):
        monkeypatch.setenv("OPENROUTER_TEST_SECRET", "or-test-secret")
        resource = UpstreamResource(
            prefix="openrouter",
            secret_env="OPENROUTER_TEST_SECRET",
            routes={
                "openai-audio": UpstreamRoute(
                    base_url=OR_BASE, path="/audio/transcriptions", auth="bearer"
                ),
            },
        )
        cfg = GatewayConfig(
            auth=AuthConfig(mode="bearer_token", bearer_token="t"),
            region="us-east-1",
            server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
            retry=RetryConfig(max_retries=1, base_delay=0.01),
            upstream_resources={"openrouter": resource},
            models={
                "my-whisper": ModelEntry(
                    bedrock_id="openai/whisper-1",
                    transport="http",
                    dialect="openai-audio",
                    deployment="openai/whisper-1",
                    upstream_resource="openrouter",
                    upstream_base_url=OR_BASE,
                    upstream_path="/audio/transcriptions",
                    upstream_auth="bearer",
                    upstream_secret_env="OPENROUTER_TEST_SECRET",
                ),
            },
        )
        explicit_client = TestClient(create_app(cfg))
        mock_cls.return_value = _sync_client(_response())
        resp = explicit_client.post(
            "/v1/audio/transcriptions",
            data={"model": "my-whisper"},
            files=[("file", ("a.wav", AUDIO, "audio/wav"))],
        )
        assert resp.status_code == 200
        assert b"openai/whisper-1" in _multipart_call(mock_cls)[2]

    def test_non_audio_named_model_rejected(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_TEST_SECRET", "or-test-secret")
        cfg = GatewayConfig(
            auth=AuthConfig(mode="bearer_token", bearer_token="t"),
            region="us-east-1",
            server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
            retry=RetryConfig(max_retries=1, base_delay=0.01),
            models={
                "chat-model": ModelEntry(
                    bedrock_id="some-model", transport="bedrock", dialect="openai-chat"
                ),
            },
        )
        c = TestClient(create_app(cfg))
        resp = c.post(
            "/v1/audio/transcriptions",
            data={"model": "chat-model"},
            files=[("file", ("a.wav", AUDIO, "audio/wav"))],
        )
        assert resp.status_code == 400
        assert "/v1/audio/transcriptions" in resp.json()["error"]["message"]

    def test_unknown_model_rejected(self, client):
        data, files = _request("not-a-model")
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 400
        assert "openrouter/" in resp.json()["error"]["message"]
        assert resp.json()["error"]["code"] == "model_not_found"

    def test_requires_multipart_media_type(self, client):
        resp = client.post("/v1/audio/transcriptions", json={"model": "x"})
        assert resp.status_code == 415

    def test_form_parse_error_is_correlated_and_not_forwarded(self, client, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        with patch.object(Request, "form", new_callable=AsyncMock) as form:
            form.side_effect = ValueError("parser-secret-sentinel")
            data, files = _request()
            resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 400
        assert "Request ID:" in resp.json()["error"]["message"]
        text = resp.text + "\n" + "\n".join(r.getMessage() for r in caplog.records)
        assert "parser-secret-sentinel" not in text
        assert "MULTIPART_PARSE_FAILED" in text

    @pytest.mark.parametrize(
        "data,files,message",
        [
            ({}, [("file", ("a.wav", AUDIO, "audio/wav"))], "model"),
            (
                {"model": "openrouter/openai/whisper-1"},
                [("audio", ("a.wav", AUDIO, "audio/wav"))],
                "file",
            ),
        ],
    )
    def test_required_parts(self, client, data, files, message):
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 400
        assert message in resp.json()["error"]["message"]

    def test_multiple_files_rejected(self, client):
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "openrouter/openai/whisper-1"},
            files=[
                ("file", ("a.wav", AUDIO, "audio/wav")),
                ("file", ("b.wav", AUDIO, "audio/wav")),
            ],
        )
        assert resp.status_code == 400
        assert "Exactly one" in resp.json()["error"]["message"]

    def test_invalid_stream_value(self, client):
        data, files = _request(stream="yes")
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 400
        assert "stream must be true or false" in resp.json()["error"]["message"]

    def test_stream_rejected(self, client):
        data, files = _request(stream="true")
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 400
        assert "does not support streaming" in resp.json()["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_upstream_error_surfaced(self, mock_cls, client):
        mock_cls.return_value = _sync_client(
            _response(
                {"error": {"message": "unsupported", "type": "invalid_request_error"}},
                status=400,
            )
        )
        data, files = _request()
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["error"]["message"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bad_upstream_json_returns_502(self, mock_cls, client):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("bad json")
        resp.text = "not-json"
        resp.content = b"not-json"
        inst = AsyncMock()
        inst.post = AsyncMock(return_value=resp)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst
        data, files = _request()
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 502
        assert "malformed" in resp.json()["error"]["message"]

    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_retry_replays_identical_multipart(self, mock_cls, _sleep, client):
        mock_cls.return_value = _sync_client(
            _response({"error": {"message": "busy"}}, 429), _response()
        )
        data, files = _request()
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 200
        first = _multipart_call(mock_cls, 0)
        second = _multipart_call(mock_cls, 1)
        assert first[1]["Content-Type"] == second[1]["Content-Type"]
        assert first[2] == second[2]
        assert AUDIO in second[2]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_prompt_and_audio_never_logged(self, mock_cls, client, caplog):
        mock_cls.return_value = _sync_client(
            _response({"error": {"message": "boom", "type": "invalid_request_error"}}, status=400)
        )
        caplog.set_level(logging.INFO, logger="bedrock_gateway")
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "openrouter/openai/whisper-1", "prompt": "SENTINEL-PROMPT"},
            files=[("file", ("audio.wav", b"SENTINEL-AUDIO", "audio/wav"))],
        )
        assert resp.status_code == 400
        log = "\n".join(r.getMessage() for r in caplog.records)
        assert "SENTINEL-PROMPT" not in log
        assert "SENTINEL-AUDIO" not in log
        # Passthrough is still faithful: the upstream body carries both.
        body = _multipart_call(mock_cls)[2]
        assert b"SENTINEL-PROMPT" in body
        assert b"SENTINEL-AUDIO" in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_response_verbatim_passthrough(self, mock_cls, client):
        mock_cls.return_value = _sync_client(_response())
        data, files = _request()
        resp = client.post("/v1/audio/transcriptions", data=data, files=files)
        assert resp.status_code == 200
        assert resp.json() == TRANSCRIPTION_RESPONSE
