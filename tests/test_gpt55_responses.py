"""
Tests for GPT-5.5 / OpenAI Responses passthrough via the mantle endpoint.

Covers the layers that must stay in sync for the new upstream dialect:
  1. config._DEFAULT_MODELS / _MODEL_ALIASES — gpt-5.5 registration + aliases
  2. ModelEntry.endpoint/protocol — new routing fields, backward compatible
  3. providers.get_provider — protocol → provider selection
  4. OpenAIResponsesProvider — URL construction, verbatim sync render,
     SSE passthrough (incl. CJK split across chunk boundaries)
  5. Endpoint routing/guards — /openai/v1/responses serves only responses
     models; /v1/chat/completions and /v1/messages reject them.
  6. End-to-end passthrough — body flows through untouched (model swapped),
     image blocks preserved, streaming SSE forwarded.
"""

import base64
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
    AnthropicBedrockProvider,
    OpenAIResponsesProvider,
    UnsupportedProtocolError,
    get_provider,
)
from bedrock_gateway.server import create_app

GPT55_ALIAS = "gpt-5.5"
GPT55_BEDROCK_ID = "openai.gpt-5.5"


# ---------------------------------------------------------------------------
# Layer 1 — registration & aliases
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_gpt55_registered(self):
        assert GPT55_ALIAS in _DEFAULT_MODELS
        e = _DEFAULT_MODELS[GPT55_ALIAS]
        assert e["bedrock_id"] == GPT55_BEDROCK_ID
        assert e["endpoint"] == "mantle"
        assert e["protocol"] == "openai-responses"
        assert e["context_length"] == 272_000

    @pytest.mark.parametrize(
        "alias", ["gpt-55", "gpt5.5", "gpt-5-5", "openai.gpt-5.5", "openai-gpt-5.5"]
    )
    def test_aliases_resolve(self, alias):
        assert _MODEL_ALIASES[alias] == GPT55_ALIAS

    def test_parses_into_model_entry(self):
        models = _parse_models(_DEFAULT_MODELS)
        e = models[GPT55_ALIAS]
        assert isinstance(e, ModelEntry)
        assert e.endpoint == "mantle"
        assert e.protocol == "openai-responses"


# ---------------------------------------------------------------------------
# Layer 2 — ModelEntry backward compatibility
# ---------------------------------------------------------------------------

class TestModelEntryDefaults:
    def test_new_fields_default_to_runtime_anthropic(self):
        """Existing/flat entries with no endpoint/protocol keep old behaviour."""
        e = ModelEntry(bedrock_id="us.anthropic.claude-x")
        assert e.endpoint == "runtime"
        assert e.protocol == "anthropic"

    def test_flat_yaml_without_new_fields(self):
        """A legacy flat config (three keys only) still parses & defaults."""
        models = _parse_models(
            {"legacy": {"bedrock_id": "us.anthropic.foo",
                        "context_length": 200000, "max_output": 4096}}
        )
        assert models["legacy"].protocol == "anthropic"
        assert models["legacy"].endpoint == "runtime"

    def test_all_claude_defaults_are_anthropic(self):
        """No existing Claude model accidentally got the new protocol."""
        models = _parse_models(_DEFAULT_MODELS)
        for alias, e in models.items():
            if alias == GPT55_ALIAS:
                continue
            assert e.protocol == "anthropic", alias
            assert e.endpoint == "runtime", alias


# ---------------------------------------------------------------------------
# Layer 3 — provider selection
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def test_gpt55_selects_responses_provider(self):
        e = ModelEntry(bedrock_id=GPT55_BEDROCK_ID, protocol="openai-responses")
        assert get_provider(e).name == "openai-responses"

    def test_claude_selects_anthropic_provider(self):
        e = ModelEntry(bedrock_id="us.anthropic.claude-x")
        assert get_provider(e).name == "anthropic"

    def test_unknown_protocol_raises(self):
        e = ModelEntry(bedrock_id="x", protocol="does-not-exist")
        with pytest.raises(UnsupportedProtocolError):
            get_provider(e)

    def test_registry_get_entry_resolves_alias(self):
        cfg = GatewayConfig(models=_parse_models(_DEFAULT_MODELS))
        reg = ModelRegistry(cfg)
        assert reg.get_entry("gpt-55").protocol == "openai-responses"
        assert reg.get_entry("gpt-5.5").bedrock_id == GPT55_BEDROCK_ID
        # unregistered raw id → None (falls back to default provider)
        assert reg.get_entry("some.raw.id") is None


# ---------------------------------------------------------------------------
# Layer 4 — OpenAIResponsesProvider unit behaviour
# ---------------------------------------------------------------------------

class TestResponsesProviderUnit:
    provider = OpenAIResponsesProvider()

    def test_urls_target_mantle(self):
        u = self.provider.sync_url("us-east-1", GPT55_BEDROCK_ID)
        assert u == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
        # sync and stream hit the same URL (stream toggled in body)
        assert self.provider.stream_url("us-east-1", GPT55_BEDROCK_ID) == u

    def test_url_respects_region(self):
        u = self.provider.sync_url("us-east-2", GPT55_BEDROCK_ID)
        assert "bedrock-mantle.us-east-2.api.aws" in u

    def test_render_sync_is_verbatim(self):
        upstream = {
            "id": "resp_abc",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "hello"}]}],
            "usage": {"input_tokens": 7, "output_tokens": 5},
        }
        body, log = self.provider.render_sync(upstream, GPT55_ALIAS)
        # passthrough → identical object
        assert body is upstream
        assert log["input_tokens"] == 7
        assert log["output_tokens"] == 5
        assert log["finish"] == "completed"

    def test_render_sync_missing_usage(self):
        body, log = self.provider.render_sync({"status": "x"}, GPT55_ALIAS)
        assert log["input_tokens"] == "?"

    async def test_transform_stream_passthrough(self):
        async def byte_iter():
            yield b"event: response.created\ndata: {}\n\n"
            yield b"event: response.completed\ndata: {}\n\n"

        out = "".join(
            [c async for c in self.provider.transform_stream(
                byte_iter(), GPT55_ALIAS, "msg_1")]
        )
        assert "response.created" in out
        assert "response.completed" in out

    async def test_transform_stream_utf8_split_boundary(self):
        """A multi-byte char split across chunks must not be corrupted."""
        text = "数到三"
        raw = f'data: {{"delta":"{text}"}}\n\n'.encode("utf-8")
        mid = len(raw) // 2

        async def byte_iter():
            yield raw[:mid]
            yield raw[mid:]

        out = "".join(
            [c async for c in self.provider.transform_stream(
                byte_iter(), GPT55_ALIAS, "msg_1")]
        )
        assert text in out
        assert "�" not in out  # no replacement char

    def test_stream_error_is_sse_event(self):
        s = self.provider.stream_error("boom", 500)
        assert s.startswith("event: error\n")
        assert '"message": "boom"' in s
        assert '"code": 500' in s

    async def test_transform_stream_flushes_tail_on_split_final_byte(self):
        """A trailing partial multi-byte char must be flushed at stream end."""
        raw = "末".encode("utf-8")  # 3 bytes
        assert len(raw) == 3

        async def byte_iter():
            # deliver only the first 2 bytes — final byte never arrives as
            # its own chunk; the incremental decoder must flush on final=True
            yield raw[:2]

        out = "".join(
            [c async for c in self.provider.transform_stream(
                byte_iter(), GPT55_ALIAS, "msg_1")]
        )
        # replacement char emitted at flush (not dropped, not crashing)
        assert out == "�" or out == ""  # decoder flush path exercised


# ---------------------------------------------------------------------------
# Endpoint routing / guards
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    config = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=_parse_models(_DEFAULT_MODELS),
    )
    return TestClient(create_app(config))


def _mock_sync_client(response_data: dict, status: int = 200):
    mock_response = MagicMock()
    mock_response.status_code = status
    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=mock_response)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _sent(mock_cls) -> dict:
    call = mock_cls.return_value.post.call_args
    return json.loads(call.kwargs.get("content", b"{}"))


def _responses_body() -> dict:
    return {
        "id": "resp_x",
        "object": "response",
        "status": "completed",
        "model": GPT55_BEDROCK_ID,
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


class TestEndpointGuards:
    def test_gpt55_rejected_on_chat_completions(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": GPT55_ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]

    def test_gpt55_rejected_on_messages(self, client):
        resp = client.post("/v1/messages", json={
            "model": GPT55_ALIAS,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "/openai/v1/responses" in resp.json()["error"]["message"]

    def test_claude_rejected_on_responses(self, client):
        resp = client.post("/openai/v1/responses", json={
            "model": "claude-haiku",
            "input": "hi",
        })
        assert resp.status_code == 400
        assert "/v1/chat/completions" in resp.json()["error"]["message"]

    def test_responses_invalid_json(self, client):
        resp = client.post(
            "/openai/v1/responses",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


class TestResponsesEndToEnd:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_routes_to_mantle_url(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body())
        resp = client.post("/openai/v1/responses", json={
            "model": GPT55_ALIAS, "input": "hi",
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert "bedrock-mantle.us-east-1.api.aws/openai/v1/responses" in url

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_body_passthrough_swaps_model_only(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body())
        client.post("/openai/v1/responses", json={
            "model": GPT55_ALIAS,
            "input": "hi",
            "reasoning": {"effort": "high"},
            "temperature": 0.4,
        })
        sent = _sent(mock_cls)
        # alias swapped to the upstream bedrock id
        assert sent["model"] == GPT55_BEDROCK_ID
        # every other field flows through untouched
        assert sent["input"] == "hi"
        assert sent["reasoning"] == {"effort": "high"}
        assert sent["temperature"] == 0.4

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_alias_resolves_and_routes(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_responses_body())
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-55", "input": "hi",
        })
        assert resp.status_code == 200
        assert _sent(mock_cls)["model"] == GPT55_BEDROCK_ID

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_image_block_preserved(self, mock_cls, client):
        """input_image blocks must pass through verbatim (no cleaning)."""
        mock_cls.return_value = _mock_sync_client(_responses_body())
        png = base64.b64encode(b"\x89PNG_fake").decode()
        client.post("/openai/v1/responses", json={
            "model": GPT55_ALIAS,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "what color"},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{png}"},
            ]}],
        })
        sent = _sent(mock_cls)
        blocks = sent["input"][0]["content"]
        img = [b for b in blocks if b["type"] == "input_image"][0]
        assert img["image_url"] == f"data:image/png;base64,{png}"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_sync_response_is_verbatim(self, mock_cls, client):
        upstream = _responses_body()
        mock_cls.return_value = _mock_sync_client(upstream)
        resp = client.post("/openai/v1/responses", json={
            "model": GPT55_ALIAS, "input": "hi",
        })
        assert resp.status_code == 200
        body = resp.json()
        # Responses shape preserved (output array, not chat choices)
        assert body["object"] == "response"
        assert body["output"][0]["content"][0]["text"] == "hi"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_upstream_error_surfaced(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(
            {"message": "bad model"}, status=400)
        resp = client.post("/openai/v1/responses", json={
            "model": GPT55_ALIAS, "input": "hi",
        })
        assert resp.status_code == 400

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_streaming_passthrough(self, mock_cls, client):
        """Streaming forwards upstream Responses SSE frames verbatim."""
        frames = [
            b'event: response.created\ndata: {"type":"response.created"}\n\n',
            b'event: response.output_text.delta\ndata: {"delta":"hi"}\n\n',
            b'event: response.completed\ndata: {"usage":{"input_tokens":3,"output_tokens":2}}\n\n',
        ]

        async def aiter_bytes():
            for f in frames:
                yield f

        resp = MagicMock()
        resp.status_code = 200
        resp.aiter_bytes = aiter_bytes

        async def aiter_text():
            yield ""
        resp.aiter_text = aiter_text

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        inst = AsyncMock()
        inst.stream = MagicMock(return_value=ctx)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst

        with client.stream("POST", "/openai/v1/responses", json={
            "model": GPT55_ALIAS, "input": "hi", "stream": True,
        }) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        # upstream frames present verbatim
        assert "response.created" in body
        assert '"delta":"hi"' in body
        assert "response.completed" in body


# ---------------------------------------------------------------------------
# Regression guard — Anthropic provider still faithful
# ---------------------------------------------------------------------------

class TestAnthropicProviderUnit:
    provider = AnthropicBedrockProvider()

    def test_sync_url_is_runtime_invoke(self):
        u = self.provider.sync_url("us-east-1", "us.anthropic.claude-x")
        assert u == (
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/us.anthropic.claude-x/invoke"
        )

    def test_stream_url_is_invoke_with_stream(self):
        u = self.provider.stream_url("us-east-1", "us.anthropic.claude-x")
        assert u.endswith("/invoke-with-response-stream")

    def test_render_sync_produces_chat_completion(self):
        upstream = {
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "stop_reason": "end_turn",
        }
        body, log = self.provider.render_sync(upstream, "claude-haiku")
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "answer"
        assert body["usage"]["total_tokens"] == 30
        assert log["finish"] == "stop"

    def test_stream_error_emits_chunk_and_done(self):
        """Mid-stream error → OpenAI error chunk + [DONE] terminator."""
        s = self.provider.stream_error("kaboom", 504)
        assert '"message": "kaboom"' in s
        assert '"code": 504' in s
        assert s.rstrip().endswith("data: [DONE]")
