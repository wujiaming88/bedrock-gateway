"""Integration tests for the 0.1.2 change — passive upstream health.

The active probe (a periodic GET against bedrock-runtime root) was
removed; ``/api/metrics/health`` now derives its ``upstream``
section from real request traffic.

What we verify end-to-end here:

  1. ``GET /api/metrics/health`` reports ``status: unknown``
     when no traffic has gone through, regardless of upstream reachability.
  2. After a successful request, ``status: healthy`` and ``last_success``
     is set.
  3. After a 5xx burst, ``status: down`` and the success rate reflects it.
  4. After a 401, ``status: auth_failed`` even when most requests succeeded.
  5. No process in the gateway issues a ``GET`` to
     ``bedrock-runtime.<region>.amazonaws.com/`` during normal operation —
     the only outbound calls are real ``POST /model/.../invoke...`` or
     ``invoke-with-response-stream`` requests.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    DashboardConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    StorageConfig,
)
from bedrock_gateway.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config(*, tmp_path) -> GatewayConfig:
    """Dashboard-on config so the health endpoint is reachable."""
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.001),
        dashboard=DashboardConfig(
            enabled=True,
            require_auth=False,
            api_key=None,
            # localhost_only=False so the TestClient (which connects from
            # the same process) isn't blocked.
            localhost_only=False,
            rate_limit=600,
            max_request_log=20,
            storage=StorageConfig(
                enabled=False,
                path=str(tmp_path / "x.db"),
                retain_days=7,
            ),
        ),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-model-v1",
                context_length=200000,
                max_output=4096,
            ),
        },
    )


def _bedrock_response(status: int, body: dict | str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    if isinstance(body, dict):
        resp.json = MagicMock(return_value=body)
        resp.text = json.dumps(body)
    else:
        resp.text = body or ""
    return resp


def _success_body() -> dict:
    return {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _mk_mock_client(post_resp: MagicMock) -> AsyncMock:
    """An ``httpx.AsyncClient`` mock that returns *post_resp* for POSTs.
    GETs are tracked separately so tests can assert nothing in the gateway
    issues a stray ``GET /``.
    """
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=post_resp)
    inst.get = AsyncMock(
        return_value=MagicMock(spec=httpx.Response, status_code=404, text="")
    )
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _post_chat(client: TestClient):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


def _get_health(client: TestClient) -> dict:
    r = client.get("/api/metrics/health")
    assert r.status_code == 200, r.text
    return r.json()["upstream"]


# ---------------------------------------------------------------------------
# End-to-end status transitions
# ---------------------------------------------------------------------------


class TestUpstreamStatusEndToEnd:
    def test_unknown_before_any_traffic(self, tmp_path):
        with TestClient(create_app(_config(tmp_path=tmp_path))) as client:
            ups = _get_health(client)
            assert ups["status"] == "unknown"
            assert ups["total"] == 0
            assert ups["last_success"] is None

    def test_healthy_after_successful_request(self, tmp_path):
        with TestClient(create_app(_config(tmp_path=tmp_path))) as client:
            mock_client = _mk_mock_client(_bedrock_response(200, _success_body()))
            with patch(
                "bedrock_gateway.server.httpx.AsyncClient",
                return_value=mock_client,
            ):
                r = _post_chat(client)
                assert r.status_code == 200, r.text

            ups = _get_health(client)
            assert ups["status"] == "healthy"
            assert ups["success_rate"] == 100.0
            assert ups["total"] >= 1
            assert ups["last_success"] is not None

    def test_down_after_5xx_burst(self, tmp_path):
        with TestClient(create_app(_config(tmp_path=tmp_path))) as client:
            mock_client = _mk_mock_client(
                _bedrock_response(500, '{"message":"upstream blew up"}')
            )
            with patch(
                "bedrock_gateway.server.httpx.AsyncClient",
                return_value=mock_client,
            ):
                # Send several failing requests so the rate is solidly < 80%.
                for _ in range(5):
                    _post_chat(client)

            ups = _get_health(client)
            assert ups["status"] == "down"
            assert ups["success_rate"] is not None
            assert ups["success_rate"] < 80.0

    def test_auth_failed_after_single_401(self, tmp_path):
        with TestClient(create_app(_config(tmp_path=tmp_path))) as client:
            ok_client = _mk_mock_client(_bedrock_response(200, _success_body()))
            bad_client = _mk_mock_client(
                _bedrock_response(401, '{"message":"token expired"}')
            )

            # Mostly successful traffic followed by one auth failure —
            # the auth band must override the otherwise-healthy rate.
            with patch(
                "bedrock_gateway.server.httpx.AsyncClient",
                return_value=ok_client,
            ):
                for _ in range(5):
                    _post_chat(client)

            with patch(
                "bedrock_gateway.server.httpx.AsyncClient",
                return_value=bad_client,
            ):
                r = _post_chat(client)
                assert r.status_code == 401

            ups = _get_health(client)
            assert ups["status"] == "auth_failed"


# ---------------------------------------------------------------------------
# Side-effect contract: no probe-style GETs to bedrock-runtime
# ---------------------------------------------------------------------------


class TestNoProbeSideEffects:
    def test_dashboard_health_endpoint_does_not_trigger_outbound_get(
        self, tmp_path
    ):
        """Hitting the health endpoint is a pure read — it must not trigger
        a probe call to bedrock-runtime root. (The old code path exposed the
        probe via the same data structure, but never *as* a side effect of
        the endpoint; this test guards the same property after the rewrite.)
        """
        # Build a mock client that records all `.get()` calls so we can
        # inspect the URLs the gateway hit during the request.
        get_calls: list[str] = []

        async def _track_get(url, *_, **__):
            get_calls.append(str(url))
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 404
            resp.text = ""
            return resp

        inst = AsyncMock()
        inst.get = _track_get
        inst.post = AsyncMock(
            return_value=_bedrock_response(200, _success_body())
        )
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)

        with TestClient(create_app(_config(tmp_path=tmp_path))) as client:
            with patch(
                "bedrock_gateway.server.httpx.AsyncClient", return_value=inst
            ):
                # Several health-endpoint hits over the app lifetime.
                for _ in range(5):
                    client.get("/api/metrics/health")

        runtime_root_hits = [
            u for u in get_calls
            if u.endswith("bedrock-runtime.us-east-1.amazonaws.com/")
        ]
        assert runtime_root_hits == [], (
            "no probe-style GET / should have been issued, "
            f"saw: {runtime_root_hits!r}"
        )
