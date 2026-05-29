"""
Tests for Claude Opus 4.8 support.

Covers the three layers that must stay in sync for a new model:
  1. config._DEFAULT_MODELS  — alias → bedrock_id + context/output specs
  2. config._MODEL_ALIASES   — name variations resolve to the canonical alias
  3. converter adaptive-thinking pattern — 4.8 uses {"type": "adaptive"}

Plus end-to-end request flow asserting every reasoning_effort level maps to
adaptive thinking for 4.8, and that the fallback paths behave safely.
"""

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
from bedrock_gateway.converter import (
    REASONING_EFFORT_MAP,
    _model_supports_adaptive,
    map_reasoning_effort,
)
from bedrock_gateway.models import ModelRegistry, UnknownModelError
from bedrock_gateway.server import create_app

OPUS_4_8_ALIAS = "claude-opus-4.8"
OPUS_4_8_BEDROCK_ID = "us.anthropic.claude-opus-4-8"


# ---------------------------------------------------------------------------
# Layer 1 — model registration in _DEFAULT_MODELS
# ---------------------------------------------------------------------------

class TestDefaultRegistration:
    def test_opus_4_8_registered(self):
        assert OPUS_4_8_ALIAS in _DEFAULT_MODELS
        entry = _DEFAULT_MODELS[OPUS_4_8_ALIAS]
        assert entry["bedrock_id"] == OPUS_4_8_BEDROCK_ID
        assert entry["context_length"] == 1_000_000
        assert entry["max_output"] == 128_000

    def test_parses_into_model_entry(self):
        models = _parse_models(_DEFAULT_MODELS)
        entry = models[OPUS_4_8_ALIAS]
        assert isinstance(entry, ModelEntry)
        assert entry.bedrock_id == OPUS_4_8_BEDROCK_ID
        assert entry.max_output == 128_000

    def test_does_not_disturb_4_7(self):
        """Adding 4.8 must not change the existing 4.7 entry."""
        assert _DEFAULT_MODELS["claude-opus-4.7"]["bedrock_id"] == (
            "us.anthropic.claude-opus-4-7"
        )

    def test_bare_opus_4_alias_still_points_to_4_6(self):
        """Per decision: claude-opus-4 stays on 4-6, NOT bumped to 4.8."""
        assert _DEFAULT_MODELS["claude-opus-4"]["bedrock_id"] == (
            "us.anthropic.claude-opus-4-6-v1"
        )


# ---------------------------------------------------------------------------
# Layer 2 — alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    @pytest.fixture
    def registry(self) -> ModelRegistry:
        config = GatewayConfig(models=_parse_models(_DEFAULT_MODELS))
        return ModelRegistry(config)

    def test_canonical_alias_resolves(self, registry):
        assert registry.resolve(OPUS_4_8_ALIAS) == OPUS_4_8_BEDROCK_ID

    @pytest.mark.parametrize("variant", [
        "claude-opus-4-8",
        "claude-4.8-opus",
        "claude-4-8-opus",
    ])
    def test_variants_resolve(self, registry, variant):
        assert variant in _MODEL_ALIASES
        assert registry.resolve(variant) == OPUS_4_8_BEDROCK_ID

    def test_get_info(self, registry):
        info = registry.get_info(OPUS_4_8_ALIAS)
        assert info is not None
        assert info.bedrock_id == OPUS_4_8_BEDROCK_ID
        assert info.context_length == 1_000_000
        assert info.max_output == 128_000

    def test_get_info_via_variant(self, registry):
        info = registry.get_info("claude-opus-4-8")
        assert info is not None
        assert info.bedrock_id == OPUS_4_8_BEDROCK_ID

    def test_get_max_output(self, registry):
        assert registry.get_max_output(OPUS_4_8_ALIAS) == 128_000
        assert registry.get_max_output("claude-opus-4-8") == 128_000

    def test_full_bedrock_id_passes_through(self, registry):
        """The raw cross-region id (already prefixed) passes through untouched."""
        assert registry.resolve(OPUS_4_8_BEDROCK_ID) == OPUS_4_8_BEDROCK_ID

    def test_unknown_opus_variant_rejected(self, registry):
        """A made-up bare name that isn't an alias must 400, not silently pass."""
        with pytest.raises(UnknownModelError):
            registry.resolve("claude-opus-9-9")

    def test_listed_in_models(self, registry):
        ids = [m["id"] for m in registry.list_models()]
        assert OPUS_4_8_ALIAS in ids


# ---------------------------------------------------------------------------
# Layer 3 — adaptive thinking support
# ---------------------------------------------------------------------------

class TestAdaptiveThinking:
    def test_bedrock_id_supports_adaptive(self):
        assert _model_supports_adaptive(OPUS_4_8_BEDROCK_ID) is True

    @pytest.mark.parametrize("effort", list(REASONING_EFFORT_MAP.keys()))
    def test_every_effort_level_maps_to_adaptive(self, effort):
        """All thinking levels must take effect as adaptive for 4.8."""
        result = map_reasoning_effort(effort, OPUS_4_8_BEDROCK_ID)
        assert result == {"type": "adaptive"}

    def test_unknown_effort_falls_back_to_none(self):
        """Unrecognized effort → None (no thinking injected) — safe fallback."""
        assert map_reasoning_effort("ultra", OPUS_4_8_BEDROCK_ID) is None

    def test_non_adaptive_model_uses_budget_tokens(self):
        """Sanity: a non-4.x model still gets budget_tokens, proving the
        adaptive branch is model-specific and not a blanket override."""
        result = map_reasoning_effort("high", "us.anthropic.claude-3-5-sonnet")
        assert result == {"type": "enabled", "budget_tokens": 4096}


# ---------------------------------------------------------------------------
# End-to-end — full request flow through the gateway
# ---------------------------------------------------------------------------

def _mock_sync_client(response_data: dict):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data
    mock_instance = AsyncMock()
    mock_instance.post = AsyncMock(return_value=mock_response)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    return mock_instance


def _bedrock_thinking_response() -> dict:
    return {
        "content": [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "answer"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "stop_reason": "end_turn",
    }


def _sent_body(mock_client_cls) -> dict:
    """Extract the JSON body posted to Bedrock from the mock."""
    call = mock_client_cls.return_value.post.call_args
    raw = call.kwargs.get("content", call[1].get("content", b"{}"))
    return json.loads(raw)


@pytest.fixture
def client() -> TestClient:
    """Gateway wired with the real default models (incl. 4.8)."""
    config = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=_parse_models(_DEFAULT_MODELS),
    )
    return TestClient(create_app(config))


class TestEndToEnd:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_request_routes_to_4_8_bedrock_id(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_bedrock_thinking_response())
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        # The Bedrock invoke URL must carry the resolved 4.8 id
        url = mock_cls.return_value.post.call_args[0][0]
        assert OPUS_4_8_BEDROCK_ID in url

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_variant_alias_routes_to_4_8(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_bedrock_thinking_response())
        resp = client.post("/v1/chat/completions", json={
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        url = mock_cls.return_value.post.call_args[0][0]
        assert OPUS_4_8_BEDROCK_ID in url

    @pytest.mark.parametrize("effort", list(REASONING_EFFORT_MAP.keys()))
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_all_efforts_send_adaptive_thinking(self, mock_cls, client, effort):
        """Every reasoning_effort level must reach Bedrock as adaptive for 4.8."""
        mock_cls.return_value = _mock_sync_client(_bedrock_thinking_response())
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": effort,
        })
        assert resp.status_code == 200
        body = _sent_body(mock_cls)
        assert body["thinking"] == {"type": "adaptive"}
        # adaptive carries no budget_tokens, so the <1024 clamp must NOT apply
        assert "budget_tokens" not in body["thinking"]
        # temperature is dropped whenever thinking is enabled
        assert "temperature" not in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_explicit_thinking_overrides_reasoning_effort(self, mock_cls, client):
        mock_cls.return_value = _mock_sync_client(_bedrock_thinking_response())
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "think"}],
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "reasoning_effort": "low",
        })
        assert resp.status_code == 200
        assert _sent_body(mock_cls)["thinking"]["budget_tokens"] == 8192

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_explicit_low_budget_clamped_to_1024(self, mock_cls, client):
        """Fallback: an explicit sub-1024 budget is clamped up to Bedrock's min."""
        mock_cls.return_value = _mock_sync_client(_bedrock_thinking_response())
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "think"}],
            "thinking": {"type": "enabled", "budget_tokens": 100},
        })
        assert resp.status_code == 200
        assert _sent_body(mock_cls)["thinking"]["budget_tokens"] == 1024

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_no_thinking_when_no_effort(self, mock_cls, client):
        """Fallback: plain request adds no thinking block at all."""
        mock_cls.return_value = _mock_sync_client({
            "content": [{"type": "text", "text": "plain"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        })
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        assert "thinking" not in _sent_body(mock_cls)

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_unknown_effort_sends_no_thinking(self, mock_cls, client):
        """Fallback: an unrecognized reasoning_effort must not inject thinking."""
        mock_cls.return_value = _mock_sync_client({
            "content": [{"type": "text", "text": "plain"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        })
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "bogus",
        })
        assert resp.status_code == 200
        body = _sent_body(mock_cls)
        assert "thinking" not in body
        # temperature stays since thinking was never enabled
        assert "temperature" in body

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_max_tokens_respects_4_8_default(self, mock_cls, client):
        """With adaptive thinking and no explicit max_tokens, the auto-fill uses
        the 4.8 default_max (128K) since adaptive carries no budget."""
        mock_cls.return_value = _mock_sync_client(_bedrock_thinking_response())
        resp = client.post("/v1/chat/completions", json={
            "model": OPUS_4_8_ALIAS,
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "high",
        })
        assert resp.status_code == 200
        assert _sent_body(mock_cls)["max_tokens"] == 128_000
