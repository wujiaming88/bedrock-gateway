"""Coverage tests for bedrock_gateway.dashboard.health —
event-loop-lag task, auth expiry introspection, FD reader.

The active upstream probe was removed in 0.1.2 (upstream health is now
derived passively from request metrics — see ``MetricsCollector.upstream_health``),
so probe tests live in ``test_upstream_health.py`` instead.
"""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import MagicMock, patch

from bedrock_gateway.dashboard import health as health_module
from bedrock_gateway.dashboard.health import HealthMonitor, _iso, _read_fd_info


class TestStartStop:
    async def test_start_creates_tasks_and_stop_cancels(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")

        async def run():
            h.start()
            # Only the event-loop-lag task remains after the upstream
            # probe was removed in 0.1.2.
            assert len(h._tasks) == 1
            # Start again is a no-op.
            h.start()
            assert len(h._tasks) == 1
            await h.stop()
            assert h._tasks == []

        await run()

    def test_start_without_running_loop_is_noop(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")
        # No asyncio loop running in this scope.
        h.start()
        assert h._tasks == []

    async def test_stop_swallows_task_exceptions(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")

        async def _bad():
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        loop = asyncio.get_running_loop()
        h._tasks.append(loop.create_task(_bad()))
        # stop() awaits and swallows the exception.
        await h.stop()
        assert h._tasks == []


class TestEventLoopLagTask:
    async def test_event_loop_lag_updates_gauge(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")
        # Shorten sample interval so the test doesn't sleep for seconds.
        with patch.object(health_module, "_EVENT_LOOP_SLEEP_S", 0.001), \
             patch.object(health_module, "_EVENT_LOOP_SAMPLE_INTERVAL_S", 0.002):
            task = asyncio.create_task(h._event_loop_lag_task())
            # Let it sample a few iterations.
            await asyncio.sleep(0.03)
            h._stopped.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Lag is non-negative (may be 0 on very fast runs).
        assert h._event_loop_lag_ms >= 0

    async def test_event_loop_lag_smoothing_path(self):
        """Force lag_ms to already be >0 so the smoothing branch is taken."""
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")
        h._event_loop_lag_ms = 50.0  # seed non-zero
        with patch.object(health_module, "_EVENT_LOOP_SLEEP_S", 0.001), \
             patch.object(health_module, "_EVENT_LOOP_SAMPLE_INTERVAL_S", 0.002):
            task = asyncio.create_task(h._event_loop_lag_task())
            await asyncio.sleep(0.02)
            h._stopped.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestUpstreamSnapshot:
    """Upstream section of ``snapshot`` is sourced from the metrics
    collector since 0.1.2 — verify the wiring is correct."""

    def test_no_metrics_returns_unknown(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")
        snap = h.snapshot()
        assert snap["upstream"]["status"] == "unknown"
        assert snap["upstream"]["total"] == 0

    def test_metrics_upstream_health_is_used(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")

        class FakeMetrics:
            def consecutive_errors(self):
                return 0

            def upstream_health(self):
                return {
                    "status": "healthy",
                    "success_rate": 100.0,
                    "total": 12,
                    "errors": 0,
                    "window_minutes": 5,
                    "last_success": "2026-05-25T10:00:00Z",
                }

        snap = h.snapshot(metrics=FakeMetrics())
        assert snap["upstream"]["status"] == "healthy"
        assert snap["upstream"]["total"] == 12

    def test_metrics_upstream_health_exception_falls_back(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")

        class BadMetrics:
            def consecutive_errors(self):
                return 0

            def upstream_health(self):
                raise RuntimeError("boom")

        snap = h.snapshot(metrics=BadMetrics())
        # Fallback payload keeps a sane default.
        assert snap["upstream"]["status"] == "unknown"


class TestAuthSnapshot:
    def test_unknown_auth_mode(self):
        h = HealthMonitor(region="us-east-1", auth_mode="weird_mode")
        snap = h.snapshot()
        assert snap["auth"]["mode"] == "weird_mode"
        assert snap["auth"]["status"] == "unknown"

    def test_empty_auth_mode(self):
        h = HealthMonitor(region="us-east-1", auth_mode="")
        snap = h.snapshot()
        # Empty mode falls through to the final branch with mode="-"
        assert snap["auth"]["mode"] == "-"
        assert snap["auth"]["status"] == "unknown"

    def test_iam_role_no_provider(self):
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=None)
        snap = h.snapshot()
        assert snap["auth"]["mode"] == "iam_role"
        assert snap["auth"]["status"] == "unknown"

    def test_iam_role_with_unmaterialised_client(self):
        provider = MagicMock()
        provider._boto3_client = None
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "unknown"

    def test_iam_role_with_no_expiry(self):
        provider = MagicMock()
        creds = MagicMock(spec=[])  # no _expiry_time attribute
        # When _expiry_time is missing, getattr returns None → "valid"
        signer = MagicMock()
        signer._credentials = creds
        client = MagicMock()
        client._request_signer = signer
        provider._boto3_client = client
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "valid"
        assert snap["auth"]["expires_at"] is None

    def test_iam_role_expired_credentials(self):
        provider = MagicMock()
        creds = MagicMock()
        # Force datetime to be interpretable by .timestamp()
        creds._expiry_time = datetime.datetime.now(datetime.timezone.utc) - \
                             datetime.timedelta(seconds=60)
        creds.get_frozen_credentials = MagicMock()
        signer = MagicMock()
        signer._credentials = creds
        client = MagicMock()
        client._request_signer = signer
        provider._boto3_client = client
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "expired"
        assert snap["auth"]["expires_at"] is not None

    def test_iam_role_expiring_soon(self):
        provider = MagicMock()
        creds = MagicMock()
        # 5 minutes from now — within the 15-minute "expiring_soon" window.
        creds._expiry_time = datetime.datetime.now(datetime.timezone.utc) + \
                             datetime.timedelta(minutes=5)
        creds.get_frozen_credentials = MagicMock()
        signer = MagicMock()
        signer._credentials = creds
        client = MagicMock()
        client._request_signer = signer
        provider._boto3_client = client
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "expiring_soon"

    def test_iam_role_valid_with_future_expiry(self):
        provider = MagicMock()
        creds = MagicMock()
        creds._expiry_time = datetime.datetime.now(datetime.timezone.utc) + \
                             datetime.timedelta(hours=12)
        creds.get_frozen_credentials = MagicMock()
        signer = MagicMock()
        signer._credentials = creds
        client = MagicMock()
        client._request_signer = signer
        provider._boto3_client = client
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "valid"
        assert snap["auth"]["expires_at"] is not None

    def test_iam_role_get_frozen_credentials_exception_is_swallowed(self):
        provider = MagicMock()
        creds = MagicMock()
        creds._expiry_time = datetime.datetime.now(datetime.timezone.utc) + \
                             datetime.timedelta(hours=1)
        creds.get_frozen_credentials = MagicMock(side_effect=RuntimeError("bad"))
        signer = MagicMock()
        signer._credentials = creds
        client = MagicMock()
        client._request_signer = signer
        provider._boto3_client = client
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        # Even though refresh failed, we still compute status from expiry.
        assert snap["auth"]["status"] in ("valid", "expiring_soon")

    def test_iam_role_timestamp_exception_returns_valid(self):
        provider = MagicMock()
        creds = MagicMock()
        # Something that raises when .timestamp() is called.
        bad_expiry = MagicMock()
        bad_expiry.timestamp.side_effect = ValueError("bad timestamp")
        creds._expiry_time = bad_expiry
        creds.get_frozen_credentials = MagicMock()
        signer = MagicMock()
        signer._credentials = creds
        client = MagicMock()
        client._request_signer = signer
        provider._boto3_client = client
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "valid"
        assert snap["auth"]["expires_at"] is None

    def test_boto_auth_expiry_outer_exception(self):
        # Force the outer try to fail before even looking at credentials.
        provider = MagicMock()
        # Accessing any attribute raises.
        type(provider)._boto3_client = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("oops"))
        )
        h = HealthMonitor(region="us-east-1", auth_mode="iam_role",
                          auth_provider=provider)
        snap = h.snapshot()
        assert snap["auth"]["status"] == "unknown"


class TestSnapshotMetricsError:
    def test_metrics_consecutive_errors_raises(self):
        h = HealthMonitor(region="us-east-1", auth_mode="bearer_token")

        class BadMetrics:
            def consecutive_errors(self):
                raise RuntimeError("kaboom")

        snap = h.snapshot(metrics=BadMetrics())
        # Exception is swallowed → value defaults to 0.
        assert snap["consecutive_errors"] == 0


class TestReadFDInfo:
    def test_returns_dict_shape(self):
        info = _read_fd_info()
        assert "current" in info
        assert "limit" in info

    def test_listdir_oserror_returns_none_current(self):
        with patch("bedrock_gateway.dashboard.health.os.listdir",
                   side_effect=OSError("nope")):
            info = _read_fd_info()
        assert info["current"] is None

    def test_rlimit_error_returns_none_limit(self):
        with patch("bedrock_gateway.dashboard.health.resource.getrlimit",
                   side_effect=ValueError("bad rlim")):
            info = _read_fd_info()
        assert info["limit"] is None


class TestIso:
    def test_iso_none(self):
        assert _iso(None) is None

    def test_iso_valid(self):
        s = _iso(0.0)  # epoch
        assert s == "1970-01-01T00:00:00Z"

    def test_iso_invalid_returns_none(self):
        # Something that isn't a number
        assert _iso(float("inf")) is None
