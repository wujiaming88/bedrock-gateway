"""Unit tests for the 0.1.1 fixes:

1. HealthMonitor background tasks must not start when ``dashboard.enabled``
   is false (avoids dead-work upstream probes flooding logs).
2. Upstream non-2xx responses must be logged at a level matching the
   severity:
     * 401/403 -> ERROR with ``[auth-failure]`` tag
     * other 4xx -> WARNING
     * 5xx -> ERROR
3. The catch-all ``except Exception`` blocks in the request handlers must
   call ``logger.exception`` so the stack trace is preserved.
"""

from __future__ import annotations

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
from bedrock_gateway.server import _log_upstream_error, create_app


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(*, dashboard_enabled: bool, tmp_path) -> GatewayConfig:
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


def _mk_mock_client_post(resp):
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _mk_mock_client_post_raises(exc):
    inst = AsyncMock()
    inst.post = AsyncMock(side_effect=exc)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


# ---------------------------------------------------------------------------
# Fix 1: probe must not start when dashboard is off
# ---------------------------------------------------------------------------


class TestProbeGatedByDashboard:
    def test_probe_does_not_start_when_dashboard_disabled(self, tmp_path):
        config = _make_config(dashboard_enabled=False, tmp_path=tmp_path)
        app = create_app(config)
        with patch.object(app.state.health, "start") as start_spy, \
             patch.object(app.state.health, "stop", new=AsyncMock()) as stop_spy:
            with TestClient(app):
                # TestClient context fires startup; exiting fires shutdown.
                pass
            assert start_spy.call_count == 0
            assert stop_spy.call_count == 0

    def test_probe_starts_when_dashboard_enabled(self, tmp_path):
        config = _make_config(dashboard_enabled=True, tmp_path=tmp_path)
        app = create_app(config)
        with patch.object(app.state.health, "start") as start_spy, \
             patch.object(app.state.health, "stop", new=AsyncMock()) as stop_spy:
            with TestClient(app):
                pass
            assert start_spy.call_count == 1
            assert stop_spy.call_count == 1

    def test_dashboard_off_logs_skip_message(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="bedrock_gateway")
        config = _make_config(dashboard_enabled=False, tmp_path=tmp_path)
        create_app(config)
        msgs = [r.message for r in caplog.records if r.name == "bedrock_gateway"]
        assert any("dashboard disabled" in m for m in msgs), (
            "expected an INFO line explaining the probe was skipped, "
            f"got {msgs!r}"
        )


# ---------------------------------------------------------------------------
# Fix 2: upstream-error log level dispatch
# ---------------------------------------------------------------------------


class TestUpstreamLogLevel:
    def test_400_logged_as_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        _log_upstream_error(400, "ERR %d msg=%s", 400, "image too big")
        rec = next(r for r in caplog.records if r.name == "bedrock_gateway")
        assert rec.levelno == logging.WARNING
        assert "ERR 400" in rec.message
        assert "auth-failure" not in rec.message

    def test_401_logged_as_error_with_auth_tag(self, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        _log_upstream_error(401, "ERR %d msg=%s", 401, "expired token")
        rec = next(r for r in caplog.records if r.name == "bedrock_gateway")
        assert rec.levelno == logging.ERROR
        assert "[auth-failure]" in rec.message

    def test_403_logged_as_error_with_auth_tag(self, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        _log_upstream_error(403, "ERR %d msg=%s", 403, "denied")
        rec = next(r for r in caplog.records if r.name == "bedrock_gateway")
        assert rec.levelno == logging.ERROR
        assert "[auth-failure]" in rec.message

    def test_404_logged_as_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        _log_upstream_error(404, "ERR %d msg=%s", 404, "not found")
        rec = next(r for r in caplog.records if r.name == "bedrock_gateway")
        assert rec.levelno == logging.WARNING

    def test_500_logged_as_error(self, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        _log_upstream_error(500, "ERR %d msg=%s", 500, "internal")
        rec = next(r for r in caplog.records if r.name == "bedrock_gateway")
        assert rec.levelno == logging.ERROR
        assert "auth-failure" not in rec.message

    def test_502_logged_as_error(self, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        _log_upstream_error(502, "ERR %d msg=%s", 502, "bad gateway")
        rec = next(r for r in caplog.records if r.name == "bedrock_gateway")
        assert rec.levelno == logging.ERROR

    def test_chat_completions_400_emits_warning_not_error(self, tmp_path, caplog):
        """End-to-end through the handler: a Bedrock 400 must surface as
        a WARNING (the original bug logged it as ERROR)."""
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        config = _make_config(dashboard_enabled=False, tmp_path=tmp_path)
        client = TestClient(create_app(config))

        bedrock_resp = MagicMock(spec=httpx.Response)
        bedrock_resp.status_code = 400
        bedrock_resp.text = '{"message": "image exceeds 5 MB maximum"}'

        with patch(
            "bedrock_gateway.server.httpx.AsyncClient",
            return_value=_mk_mock_client_post(bedrock_resp),
        ):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert r.status_code == 400
        err_records = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and rec.message.startswith("ERR ")
        ]
        assert err_records, "expected an ERR log line"
        # All ERR lines for a 4xx must be WARNING.
        for rec in err_records:
            assert rec.levelno == logging.WARNING, (
                f"expected WARNING for 400, got {rec.levelname}: {rec.message}"
            )

    def test_chat_completions_500_still_logs_error(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="bedrock_gateway")
        config = _make_config(dashboard_enabled=False, tmp_path=tmp_path)
        client = TestClient(create_app(config))

        bedrock_resp = MagicMock(spec=httpx.Response)
        bedrock_resp.status_code = 500
        bedrock_resp.text = '{"message": "internal"}'

        with patch(
            "bedrock_gateway.server.httpx.AsyncClient",
            return_value=_mk_mock_client_post(bedrock_resp),
        ):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert r.status_code == 500
        err_records = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and rec.message.startswith("ERR ")
        ]
        assert any(rec.levelno == logging.ERROR for rec in err_records), (
            "expected an ERROR log line for an upstream 5xx"
        )


# ---------------------------------------------------------------------------
# Fix 3: catch-all blocks must use logger.exception
# ---------------------------------------------------------------------------


class TestExceptionTraceback:
    def test_unexpected_exception_in_chat_completions_logs_traceback(
        self, tmp_path, caplog
    ):
        caplog.set_level(logging.ERROR, logger="bedrock_gateway")
        config = _make_config(dashboard_enabled=False, tmp_path=tmp_path)
        client = TestClient(create_app(config))

        # Make every retry attempt raise a non-httpx error so we hit the
        # catch-all ``except Exception`` branch.
        with patch(
            "bedrock_gateway.server.httpx.AsyncClient",
            return_value=_mk_mock_client_post_raises(RuntimeError("boom")),
        ):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert r.status_code == 500
        # Find the UNEXPECTED log record and assert it has exc_info attached.
        unexpected = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and "UNEXPECTED" in rec.message
        ]
        assert unexpected, "expected an UNEXPECTED log line from the catch-all"
        assert unexpected[0].exc_info is not None, (
            "catch-all must use logger.exception so the traceback is preserved"
        )
        assert unexpected[0].exc_info[0] is RuntimeError

    def test_unexpected_exception_in_messages_logs_traceback(
        self, tmp_path, caplog
    ):
        caplog.set_level(logging.ERROR, logger="bedrock_gateway")
        config = _make_config(dashboard_enabled=False, tmp_path=tmp_path)
        client = TestClient(create_app(config))

        with patch(
            "bedrock_gateway.server.httpx.AsyncClient",
            return_value=_mk_mock_client_post_raises(RuntimeError("kapow")),
        ):
            r = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert r.status_code == 500
        unexpected = [
            rec for rec in caplog.records
            if rec.name == "bedrock_gateway"
            and "UNEXPECTED" in rec.message
            and "[messages]" in rec.message
        ]
        assert unexpected, "expected an UNEXPECTED [messages] log line"
        assert unexpected[0].exc_info is not None
        assert unexpected[0].exc_info[0] is RuntimeError
