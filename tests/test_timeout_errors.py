"""
Tests for the timeout / error-handling audit fixes (P1–P3).

  P1 — a 200 upstream response with an unparseable body is surfaced as 502
       (bad gateway), not a 500 (gateway crash).
  P2 — the single ``timeout`` value splits into a fast connect + generous
       read/write via ``_httpx_timeout``.
  P3 — the whole retry sequence is bounded by a wall-clock deadline
       (``_retry_deadline``), not just the attempt count.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    GatewayConfig,
    RetryConfig,
    ServerConfig,
    _parse_models,
)
from bedrock_gateway.server import (
    _httpx_timeout,
    _retry_deadline,
    create_app,
)


# ---------------------------------------------------------------------------
# P2 — connect/read timeout split
# ---------------------------------------------------------------------------

class TestHttpxTimeout:
    def test_read_uses_full_value_connect_is_capped(self):
        t = _httpx_timeout(300.0)
        assert t.read == 300.0
        assert t.write == 300.0
        assert t.pool == 300.0
        assert t.connect == 10.0          # capped, fast-fail on connect

    def test_connect_never_exceeds_read(self):
        # a tiny timeout must not give connect a larger value than read
        t = _httpx_timeout(3.0)
        assert t.connect == 3.0
        assert t.read == 3.0


# ---------------------------------------------------------------------------
# P3 — total retry-budget deadline
# ---------------------------------------------------------------------------

class TestRetryDeadline:
    def test_scales_with_timeout_and_attempts(self):
        import time
        now = time.monotonic()
        d = _retry_deadline(timeout=10.0, max_retries=3)
        # 10 * 1.5 * 3 = 45s budget (approx; clock advances slightly)
        assert 44.0 <= (d - now) <= 46.0

    def test_at_least_one_attempt_budget(self):
        import time
        now = time.monotonic()
        d = _retry_deadline(timeout=10.0, max_retries=0)
        assert (d - now) >= 10.0          # max(1, retries) floor


# ---------------------------------------------------------------------------
# P1 — malformed 200 body → 502 (integration)
# ---------------------------------------------------------------------------

def _config() -> GatewayConfig:
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="t"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01, timeout=30),
        models=_parse_models(None),   # built-in defaults
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_config()))


def _mock_bad_json_client():
    """A 200 response whose .json() raises (truncated / HTML error page)."""
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"<html>502 from a proxy in front of the model</html>"
    resp.json.side_effect = json.JSONDecodeError("x", "doc", 0)
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


class TestResponsesEndpointIsMetered:
    """Regression: /openai/v1/responses (GPT-5.5 / Grok / Azure) must be in the
    metrics middleware's LLM paths, else those requests are silently untracked
    (model + tokens missing from the dashboard). Bug found during the audit.
    """

    def test_responses_path_in_llm_paths(self):
        from bedrock_gateway.dashboard.middleware import _LLM_PATHS
        assert "/openai/v1/responses" in _LLM_PATHS
        assert "/v1/chat/completions" in _LLM_PATHS
        assert "/v1/messages" in _LLM_PATHS

    def test_json_usage_handles_responses_shape(self):
        """Responses uses input_tokens/output_tokens (not prompt/completion)."""
        from bedrock_gateway.dashboard.middleware import _parse_json_usage
        body = json.dumps({
            "object": "response",
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }).encode()
        assert _parse_json_usage(body) == (12, 7)


class TestMalformedUpstreamBody:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_chat_completions_bad_json_is_502(self, mock_cls, client):
        mock_cls.return_value = _mock_bad_json_client()
        r = client.post("/v1/chat/completions", json={
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 502          # not 500
        assert "malformed" in r.json()["error"]["message"].lower()

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_bad_json_is_502(self, mock_cls, client):
        mock_cls.return_value = _mock_bad_json_client()
        r = client.post("/v1/messages", json={
            "model": "claude-haiku",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 502
        body = r.json()
        # Anthropic-shaped error envelope
        assert body["type"] == "error"
        assert "malformed" in body["error"]["message"].lower()
