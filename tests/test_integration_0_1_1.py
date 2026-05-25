"""Integration tests for the 0.1.1 fixes.

These tests stand up the full FastAPI application (lifespan included) and
exercise it end-to-end with mocked Bedrock responses. The unit-level tests
in ``test_fixes_0_1_1.py`` cover the helper-level contract; the tests here
cover the lifecycle and request paths together, ensuring the fixes hold
when the app is wired up the way it is in production.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
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
from bedrock_gateway.dashboard import health as health_module
from bedrock_gateway.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config(*, dashboard_enabled: bool, tmp_path) -> GatewayConfig:
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.001),
        dashboard=DashboardConfig(
            enabled=dashboard_enabled,
            require_auth=False,
            api_key=None,
            localhost_only=False,
            rate_limit=60,
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


def _bedrock_success_response() -> MagicMock:
    """Build a mocked Bedrock 200 response carrying a minimal Claude payload."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    resp.text = json.dumps(resp.json.return_value)
    return resp


def _mk_mock_async_client(resp):
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.get = AsyncMock(
        return_value=MagicMock(spec=httpx.Response, status_code=404, text="")
    )
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


# ---------------------------------------------------------------------------
# Lifecycle: dashboard off → no upstream probe at all
# ---------------------------------------------------------------------------


class TestLifecycleNoProbeWhenDashboardOff:
    def test_no_upstream_probe_calls_during_app_lifetime(self, tmp_path, caplog):
        """When dashboard is disabled, the periodic GET to bedrock-runtime
        must never run — even briefly during startup.

        Patches :func:`HealthMonitor._probe_once` and asserts it stays at
        zero invocations through the full app lifecycle.
        """
        caplog.set_level(logging.INFO, logger="bedrock_gateway")
        config = _config(dashboard_enabled=False, tmp_path=tmp_path)

        with patch.object(
            health_module.HealthMonitor,
            "_probe_once",
            new=AsyncMock(),
        ) as probe:
            app = create_app(config)
            with TestClient(app) as client:
                # Do real work to give any leaked task a chance to fire.
                assert client.get("/health").status_code == 200
                assert client.get("/v1/models").status_code == 200

            assert probe.call_count == 0, (
                "upstream probe must not run when dashboard is disabled"
            )

    def test_health_endpoint_works_without_probe(self, tmp_path):
        """The gateway's own /health endpoint must keep working even when
        the dashboard's upstream-probe task is gated off (they are
        different concepts: /health is a synchronous status endpoint;
        the probe is the dashboard's background reachability check)."""
        config = _config(dashboard_enabled=False, tmp_path=tmp_path)
        with TestClient(create_app(config)) as client:
            r = client.get("/health")
            assert r.status_code == 200
            payload = r.json()
            assert payload["status"] == "ok"
            assert payload["region"] == "us-east-1"


class TestLifecycleProbeRunsWhenDashboardOn:
    def test_dashboard_enabled_invokes_health_start(self, tmp_path):
        """When the dashboard is on, the startup hook must call
        ``HealthMonitor.start``. We assert at the integration boundary
        (real lifespan) by spying on the bound method; the unit test in
        ``test_fixes_0_1_1.py`` already covers the matching ``stop``.
        """
        config = _config(dashboard_enabled=True, tmp_path=tmp_path)
        app = create_app(config)
        with patch.object(app.state.health, "start") as start_spy, \
             patch.object(app.state.health, "stop", new=AsyncMock()):
            with TestClient(app) as client:
                # Force startup to fire by issuing one request.
                assert client.get("/health").status_code == 200
            assert start_spy.call_count == 1


# ---------------------------------------------------------------------------
# End-to-end: 4xx surfaces as WARNING, 5xx as ERROR, auth-failure tag
# ---------------------------------------------------------------------------


class TestEndToEndLogLevels:
    def _post_chat(self, client: TestClient):
        return client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    def _patched_bedrock(self, status: int, body: str):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status
        resp.text = body
        return patch(
            "bedrock_gateway.server.httpx.AsyncClient",
            return_value=_mk_mock_async_client(resp),
        )

    def test_e2e_400_logs_warning_only(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        with TestClient(create_app(_config(dashboard_enabled=False, tmp_path=tmp_path))) as client:
            with self._patched_bedrock(400, '{"message":"bad image"}'):
                r = self._post_chat(client)
        assert r.status_code == 400
        err_lines = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and rec.message.startswith("ERR ")
        ]
        assert err_lines
        assert all(rec.levelno == logging.WARNING for rec in err_lines)

    def test_e2e_401_tags_auth_failure_at_error_level(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        with TestClient(create_app(_config(dashboard_enabled=False, tmp_path=tmp_path))) as client:
            with self._patched_bedrock(401, '{"message":"token expired"}'):
                r = self._post_chat(client)
        assert r.status_code == 401
        auth_records = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and "[auth-failure]" in rec.message
        ]
        assert auth_records, "expected an [auth-failure] tagged log line"
        assert auth_records[0].levelno == logging.ERROR

    def test_e2e_502_logs_error(self, tmp_path, caplog):
        """A 502 from upstream is *not* in the (429, 529, 503) retry set,
        so it falls through to ``_log_upstream_error`` and must be ERROR."""
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        with TestClient(create_app(_config(dashboard_enabled=False, tmp_path=tmp_path))) as client:
            with self._patched_bedrock(502, '{"message":"bad gateway"}'):
                r = self._post_chat(client)
        assert r.status_code == 502
        err_lines = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and rec.message.startswith("ERR ")
        ]
        assert err_lines
        assert any(rec.levelno == logging.ERROR for rec in err_lines)


# ---------------------------------------------------------------------------
# End-to-end: catch-all preserves traceback
# ---------------------------------------------------------------------------


class TestEndToEndExceptionTraceback:
    def test_runtime_error_through_chat_path(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR, logger="bedrock_gateway")
        config = _config(dashboard_enabled=False, tmp_path=tmp_path)

        inst = AsyncMock()
        inst.post = AsyncMock(side_effect=RuntimeError("synthetic"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)

        with TestClient(create_app(config)) as client:
            with patch(
                "bedrock_gateway.server.httpx.AsyncClient",
                return_value=inst,
            ):
                r = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        assert r.status_code == 500
        unexpected = [
            rec for rec in caplog.records
            if "UNEXPECTED" in rec.message and rec.exc_info is not None
        ]
        assert unexpected
        assert unexpected[0].exc_info[0] is RuntimeError
