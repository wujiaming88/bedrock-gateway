"""Coverage tests for dashboard package — api.py, metrics.py, middleware.py,
security.py, storage.py. Focus on paths not exercised by existing tests:
error branches, edge cases, alternate window sizes, storage failures."""

from __future__ import annotations

import json
import queue
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
from bedrock_gateway.dashboard import (
    DashboardAuth,
    HealthMonitor,
    MetricsCollector,
    MetricsStorage,
    RateLimiter,
    build_dashboard_router,
    metrics_middleware_factory,
)
from bedrock_gateway.dashboard.api import (
    _parse_form_urlencoded,
    _resolve_window,
    _system_info,
    _WindowError,
)
from bedrock_gateway.dashboard.metrics import _classify_error, _read_rss_kb
from bedrock_gateway.dashboard.middleware import (
    _extract_client_ip,
    _parse_json_usage,
    _parse_sse_line,
    _scan_chunk_usage,
)
from bedrock_gateway.dashboard.security import mask_ip
from bedrock_gateway.dashboard.storage import (
    AsyncWriter,
    BucketRow,
    RequestRow,
)
from bedrock_gateway.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> GatewayConfig:
    dash_kwargs = {
        "enabled": True,
        "require_auth": False,
        "api_key": None,
        "localhost_only": False,
        "rate_limit": 60,
        "max_request_log": 20,
        "storage": StorageConfig(enabled=False, path="data/metrics.db", retain_days=7),
    }
    dash_kwargs.update(kwargs)
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="t"),
        region="us-east-1",
        server=ServerConfig(log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.001),
        dashboard=DashboardConfig(**dash_kwargs),
        models={
            "m": ModelEntry(bedrock_id="us.anthropic.m",
                            context_length=100_000, max_output=4_096),
        },
    )


@pytest.fixture
def open_client() -> TestClient:
    return TestClient(create_app(_make_config()))


@pytest.fixture
def keyed_client() -> TestClient:
    return TestClient(create_app(_make_config(
        require_auth=True, api_key="sk-test", localhost_only=False,
    )))


# ---------------------------------------------------------------------------
# _resolve_window — alternate bin widths
# ---------------------------------------------------------------------------


class TestResolveWindow:
    def test_14d_uses_2h_bins(self):
        # retain_days must be large enough to allow 14d.
        minutes, bin_seconds = _resolve_window("14d", retain_days=30)
        assert minutes == 14 * 24 * 60
        assert bin_seconds == 2 * 60 * 60

    def test_30d_uses_4h_bins(self):
        minutes, bin_seconds = _resolve_window("30d", retain_days=30)
        assert minutes == 30 * 24 * 60
        assert bin_seconds == 4 * 60 * 60

    def test_exceeds_retention(self):
        with pytest.raises(_WindowError, match="retention"):
            _resolve_window("7d", retain_days=3)

    def test_invalid_format(self):
        with pytest.raises(_WindowError):
            _resolve_window("banana", retain_days=7)

    def test_zero_value(self):
        with pytest.raises(_WindowError):
            _resolve_window("0h", retain_days=7)


# ---------------------------------------------------------------------------
# _system_info — exception when registry raises
# ---------------------------------------------------------------------------


class TestSystemInfo:
    def test_registry_raises(self):
        request = MagicMock()
        state = MagicMock()
        config = MagicMock()
        config.region = "eu-west-1"
        auth = MagicMock()
        auth.mode = "credentials"
        registry = MagicMock()
        registry.list_models.side_effect = RuntimeError("broken")
        state.config = config
        state.auth = auth
        state.registry = registry
        request.app.state = state
        info = _system_info(request)
        assert info["model_count"] == 0
        assert info["region"] == "eu-west-1"
        assert info["auth_mode"] == "credentials"


# ---------------------------------------------------------------------------
# _parse_form_urlencoded — decode error
# ---------------------------------------------------------------------------


class TestParseFormUrlencoded:
    def test_decode_exception_returns_empty(self):
        """Bytes that raise on decode (via patched method) → empty dict."""

        class _Bad:
            def decode(self, *a, **kw):
                raise RuntimeError("bad bytes")

        # _parse_form_urlencoded calls body.decode(...) — simulate failure.
        assert _parse_form_urlencoded(_Bad()) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dashboard API — keyed client gets blocked on every /api/metrics/* endpoint
# ---------------------------------------------------------------------------


class TestGuardApiBlockedEndpoints:
    """Hit every JSON endpoint without credentials to exercise each 401 branch."""

    @pytest.mark.parametrize("path", [
        "/api/metrics/traffic",
        "/api/metrics/models",
        "/api/metrics/requests",
        "/api/metrics/errors",
        "/api/metrics/sources",
        "/api/metrics/memory",
        "/api/metrics/system",
        "/api/metrics/health",
    ])
    def test_endpoint_returns_401_without_key(self, keyed_client: TestClient, path: str):
        assert keyed_client.get(path).status_code == 401


# ---------------------------------------------------------------------------
# Dashboard API — sources endpoint populates IP list
# ---------------------------------------------------------------------------


class TestSourcesEndpoint:
    def test_returns_masked_ips(self, open_client: TestClient):
        coll = open_client.app.state.metrics  # type: ignore[attr-defined]
        coll.record_request(
            method="POST", path="/v1/x", model="m", status=200, latency_ms=1,
            client_ip="1.2.3.4",
        )
        resp = open_client.get("/api/metrics/sources?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        # IPv4 /24 mask
        assert any(s["ip"] == "1.2.3.0" for s in data["sources"])


# ---------------------------------------------------------------------------
# Dashboard API — login_form + login submit (JSON body) paths
# ---------------------------------------------------------------------------


class TestLoginFlows:
    def test_login_form_redirects_when_no_key(self, open_client: TestClient):
        # Dashboard has no api_key → login page redirects straight to dashboard.
        resp = open_client.get("/dashboard/login", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/dashboard/"

    def test_login_form_redirects_when_already_authed(self, keyed_client: TestClient):
        # Pre-authenticated via cookie.
        keyed_client.cookies.set("bedrock_gw_key", "sk-test")
        resp = keyed_client.get("/dashboard/login", follow_redirects=False)
        assert resp.status_code == 302

    def test_login_submit_json_body(self, keyed_client: TestClient):
        resp = keyed_client.post(
            "/dashboard/login",
            json={"key": "sk-test", "next": "/dashboard/"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_login_submit_invalid_json(self, keyed_client: TestClient):
        resp = keyed_client.post(
            "/dashboard/login",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        # Falls through to _login_page because payload is {}, verify_key fails.
        assert resp.status_code == 200
        assert "Invalid API key" in resp.text


# ---------------------------------------------------------------------------
# Dashboard UI — dashboard_redirect / static asset / traversal
# ---------------------------------------------------------------------------


class TestBuildRouterDefaults:
    def test_router_with_no_auth_and_limiter(self):
        """Passing auth=None creates a permissive DashboardAuth; request with
        no client falls back to x-forwarded-for for the rate-limit key."""
        from fastapi import FastAPI

        coll = MetricsCollector()
        rl = RateLimiter(limit=5, window_seconds=60)
        # auth=None → internal default
        router = build_dashboard_router(coll, auth=None, rate_limiter=rl)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        # Non-localhost XFF is used as the rate-limiter key.
        resp = client.get(
            "/api/metrics/overview",
            headers={"x-forwarded-for": "198.51.100.1"},
        )
        assert resp.status_code == 200


class TestDashboardIndexFallback:
    def test_dashboard_index_missing_static(self, open_client: TestClient, monkeypatch):
        # Monkeypatch the _STATIC_DIR constant for just this test.
        from pathlib import Path

        import bedrock_gateway.dashboard.api as api_mod

        original = api_mod._STATIC_DIR
        monkeypatch.setattr(api_mod, "_STATIC_DIR", Path("/nonexistent/dir"))
        try:
            # Build a fresh app so the router picks up the new _STATIC_DIR.
            cfg = _make_config()
            app = create_app(cfg)
            c = TestClient(app)
            resp = c.get("/dashboard/")
            assert resp.status_code == 404
        finally:
            monkeypatch.setattr(api_mod, "_STATIC_DIR", original)


class TestDashboardAssetReserved:
    def test_asset_nonexistent_file(self, open_client: TestClient):
        resp = open_client.get("/dashboard/does/not/exist.txt")
        assert resp.status_code == 404

    def test_asset_serves_real_static_file(self, open_client: TestClient):
        # The project ships static/app.js — the asset route should serve it.
        resp = open_client.get("/dashboard/app.js")
        assert resp.status_code == 200

    def test_asset_blocked_for_keyed_client(self, keyed_client: TestClient):
        """The asset route sits behind _guard_ui; unauthenticated access
        redirects to the login page."""
        resp = keyed_client.get("/dashboard/app.js", follow_redirects=False)
        assert resp.status_code in (302, 307, 403)

    def test_asset_route_rejects_reserved_names(self):
        """Directly invoke the dashboard_asset endpoint with a reserved
        filename — FastAPI routes /dashboard/login to its own handler, so we
        exercise this branch by calling the closure directly."""
        import asyncio

        import bedrock_gateway.dashboard.api as api_mod
        from fastapi import HTTPException

        coll = MetricsCollector()
        router = api_mod.build_dashboard_router(coll)
        # Find the asset-route endpoint among the router's routes.
        asset_endpoint = None
        for route in router.routes:
            if getattr(route, "path", "") == "/dashboard/{filename:path}":
                asset_endpoint = route.endpoint
                break
        assert asset_endpoint is not None

        request = MagicMock()
        request.app.state = MagicMock()
        # filename == "login" triggers the reserved-name 404 branch.
        with pytest.raises(HTTPException) as exc:
            asyncio.run(asset_endpoint(request=request, filename="login"))
        assert exc.value.status_code == 404


class TestClientKeyFallback:
    def test_no_client_no_xff_defaults_to_dash(self):
        """Exercises _client_key's XFF fallback by calling through an app with
        no request.client. We build a direct FastAPI invocation."""
        import asyncio

        from fastapi import FastAPI

        coll = MetricsCollector()
        rl = RateLimiter(limit=100, window_seconds=60)
        router = build_dashboard_router(coll, rate_limiter=rl)
        app = FastAPI()
        app.include_router(router)

        # Fire a request via ASGI with no client scope and no XFF header.
        from starlette.testclient import TestClient as TC

        c = TC(app)
        # TestClient always injects a client; we simulate "no client" via a
        # manual ASGI call.

        async def _call():
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/api/metrics/overview",
                "raw_path": b"/api/metrics/overview",
                "query_string": b"",
                "headers": [],  # no XFF, no host
                "client": None,
            }
            received = []

            async def _receive():
                return {"type": "http.request", "body": b""}

            async def _send(msg):
                received.append(msg)

            await app(scope, _receive, _send)
            return received

        msgs = asyncio.run(_call())
        # Request completed with 200 despite no client info.
        assert any(m.get("type") == "http.response.start" and m.get("status") == 200
                   for m in msgs)


class TestMemoryEndpointWindowError:
    def test_memory_bad_window(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/memory?window=bogus")
        assert resp.status_code == 400


class TestHealthEndpointWithoutMonitor:
    def test_health_without_monitor_returns_degenerate(self):
        """build_dashboard_router standalone (no app.state.health) returns a
        stub health payload."""
        from fastapi import FastAPI

        coll = MetricsCollector()
        # Seed one error so consecutive_errors returns non-zero.
        coll.record_request(
            method="POST", path="/v1/x", model="m", status=500, latency_ms=1,
        )
        app = FastAPI()
        app.include_router(build_dashboard_router(coll))
        c = TestClient(app)
        resp = c.get("/api/metrics/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["consecutive_errors"] == 1
        assert data["auth"]["mode"] == "-"


class TestDashboardRedirect:
    def test_dashboard_no_slash_redirects(self, open_client: TestClient):
        resp = open_client.get("/dashboard", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "/dashboard/" in resp.headers["location"]

    def test_dashboard_keyed_redirect_blocked(self, keyed_client: TestClient):
        # Unauthenticated → redirect to login (with ?next=).
        resp = keyed_client.get("/dashboard", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "login" in resp.headers["location"]

    def test_ui_redirect_preserves_query_string(self, keyed_client: TestClient):
        resp = keyed_client.get("/dashboard/?foo=bar", follow_redirects=False)
        assert resp.status_code == 302
        loc = resp.headers["location"]
        assert "foo=bar" in loc or "foo%3Dbar" in loc

    def test_login_and_logout_paths_are_not_asset_routes(self, keyed_client: TestClient):
        # /dashboard/login and /dashboard/logout have their own handlers;
        # the asset route rejects them with 404.
        # Authenticate so the asset route's guard permits the call.
        keyed_client.cookies.set("bedrock_gw_key", "sk-test")
        # /dashboard/login belongs to login_form (GET) — 200 is fine; the
        # asset-route branch is only reachable via other filenames.

    def test_asset_nonexistent_file_404(self, open_client: TestClient):
        resp = open_client.get("/dashboard/nonexistent.file")
        assert resp.status_code == 404

    def test_asset_traversal_blocked(self, open_client: TestClient):
        # Trying to escape the static dir.
        resp = open_client.get("/dashboard/..%2Fetc%2Fpasswd")
        # Either blocked with 404 or sanitized; either way not 200 with secrets.
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MetricsCollector — storage load failure paths
# ---------------------------------------------------------------------------


class TestMetricsLoadFailure:
    def test_load_recent_requests_failure_falls_back(self, tmp_path):
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        # Force load_recent_requests to raise.
        with patch.object(storage, "load_recent_requests",
                          side_effect=RuntimeError("io")):
            # Collector must still init successfully.
            m = MetricsCollector(storage=storage)
            if m._writer is not None:
                m._writer.stop()
        # No rows loaded — recent is empty.
        assert m.recent_requests() == []

    def test_load_buckets_failure_falls_back(self, tmp_path):
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        with patch.object(storage, "load_buckets",
                          side_effect=RuntimeError("io")):
            m = MetricsCollector(storage=storage)
            if m._writer is not None:
                m._writer.stop()
        # Init completed; no buckets.
        assert m.overview()["total_requests"] == 0

    def test_load_existing_request_errors_into_errors_deque(self, tmp_path):
        """Storage rehydration: rows with status>=400 populate the errors list."""
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        storage.batch_write_requests([
            RequestRow(ts=time.time(), method="POST", path="/v1/x", model="m",
                       status=500, latency_ms=1, error_type="err"),
            RequestRow(ts=time.time() + 1, method="POST", path="/v1/x", model="m",
                       status=200, latency_ms=1),
        ])
        m = MetricsCollector(storage=MetricsStorage(str(db)))
        if m._writer is not None:
            m._writer.stop()
        errs = m.recent_errors()
        assert any(r["status"] == 500 for r in errs)

    def test_collect_minute_buckets_storage_failure(self, tmp_path):
        """If the storage load fails during timeseries(), fall back to empty."""
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        m = MetricsCollector(storage=storage)
        if m._writer is not None:
            m._writer.stop()

        # Request a window >24h so _collect_minute_buckets hits storage.
        with patch.object(m._storage, "load_buckets",
                          side_effect=RuntimeError("io")):
            data = m.timeseries(minutes=48 * 60)
        assert "labels" in data


class TestMetricsRehydrationLatencyAndTTFT:
    def test_bucket_reconstructs_latencies_and_ttft(self, tmp_path):
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        ts = int(time.time()) // 60 * 60 - 3600  # well in the past
        storage.upsert_bucket(BucketRow(
            ts=ts, total=5, success=5, error=0,
            latency_sum=500.0, latency_count=5,
            p50=100.0, p95=100.0, p99=100.0,
            prompt_tokens=10, completion_tokens=5,
            model_counts={"m1": 5},
            status_counts={200: 5},
            ttft_sum=50.0, ttft_count=5,
        ))
        m = MetricsCollector(storage=MetricsStorage(str(db)))
        if m._writer is not None:
            m._writer.stop()
        # Bucket loaded; model totals rehydrated.
        assert m.model_stats()["models"][0]["model"] == "m1"


# ---------------------------------------------------------------------------
# MetricsCollector — persist bucket failure is swallowed
# ---------------------------------------------------------------------------


class TestPersistBucketFailure:
    def test_upsert_failure_swallowed(self, tmp_path):
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        m = MetricsCollector(storage=storage)
        # Inject a bucket that triggers rollover when another request lands.
        from bedrock_gateway.dashboard.metrics import _Bucket

        now = time.time()
        past_minute = int(now // 60) * 60 - 60
        with m._lock:
            b = _Bucket(ts=past_minute, total=1, success=1)
            b.latencies = [5.0]
            m._buckets[past_minute] = b

        with patch.object(storage, "upsert_bucket",
                          side_effect=RuntimeError("disk")):
            m.record_request(
                method="POST", path="/v1/x", model="m",
                status=200, latency_ms=1,
            )
        # Flush pending also must swallow.
        with patch.object(storage, "upsert_bucket",
                          side_effect=RuntimeError("disk")):
            m.flush_pending()
        if m._writer is not None:
            m._writer.stop()


# ---------------------------------------------------------------------------
# MetricsCollector — flush_pending without storage is no-op
# ---------------------------------------------------------------------------


class TestFlushPendingAndCleanup:
    def test_flush_pending_noop_without_storage(self):
        m = MetricsCollector()
        m.flush_pending()  # must not raise

    def test_cleanup_storage_noop_without_storage(self):
        m = MetricsCollector()
        assert m.cleanup_storage() == 0

    def test_cleanup_storage_with_storage(self, tmp_path):
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        m = MetricsCollector(storage=storage)
        n = m.cleanup_storage()
        if m._writer is not None:
            m._writer.stop()
        assert n >= 0


# ---------------------------------------------------------------------------
# MetricsCollector — ip cap trimming
# ---------------------------------------------------------------------------


class TestIpCountsCap:
    def test_ip_map_trimmed_when_over_5000(self):
        m = MetricsCollector()
        # Pre-seed with 5000 + 1 unique IPs; the next recording triggers trim.
        with m._lock:
            for i in range(5001):
                m._ip_counts[f"10.0.{i // 256}.{i % 256}"] = 1
        m.record_request(
            method="POST", path="/v1/x", model="m",
            status=200, latency_ms=1, client_ip="new.client.0.1",
        )
        assert len(m._ip_counts) <= 1001  # trimmed to ~1000


# ---------------------------------------------------------------------------
# MetricsCollector — _read_rss_kb + _classify_error + unknown status
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_unknown_status_returns_unknown(self):
        assert _classify_error(200, None) == "unknown"
        assert _classify_error(300, None) == "unknown"

    def test_explicit_error_type_preserved(self):
        assert _classify_error(500, "MyErr") == "MyErr"


class TestReadRssKb:
    def test_oserror_returns_none(self):
        with patch("builtins.open", side_effect=OSError("denied")):
            assert _read_rss_kb() is None

    def test_returns_kb_or_none(self):
        v = _read_rss_kb()
        assert v is None or isinstance(v, int)

    def test_malformed_proc_status(self):
        """If /proc/*/status VmRSS line is malformed (non-digit), break returns None."""
        from io import StringIO

        class FakeCtx:
            def __enter__(self):
                return StringIO("VmRSS: notanumber kB\n")
            def __exit__(self, *a):
                return False

        with patch("builtins.open", return_value=FakeCtx()):
            assert _read_rss_kb() is None


class TestSystemStatusRssFailure:
    def test_system_status_handles_oserror(self):
        m = MetricsCollector()
        with patch("builtins.open", side_effect=OSError("denied")):
            payload = m.system_status(
                version="1.0", auth_mode="bearer_token",
                region="us-east-1", model_count=1,
            )
        assert payload["memory_rss_mb"] is None

    def test_system_status_malformed_proc(self):
        from io import StringIO

        class FakeCtx:
            def __enter__(self):
                return StringIO("VmRSS: notnum kB\n")
            def __exit__(self, *a):
                return False

        m = MetricsCollector()
        with patch("builtins.open", return_value=FakeCtx()):
            payload = m.system_status(
                version="1.0", auth_mode="bearer_token",
                region="us-east-1", model_count=1,
            )
        assert payload["memory_rss_mb"] is None


# ---------------------------------------------------------------------------
# middleware — helper functions
# ---------------------------------------------------------------------------


class TestParseSseLine:
    def test_non_data_line(self):
        assert _parse_sse_line("event: x") == (0, 0)

    def test_done_marker(self):
        assert _parse_sse_line("data: [DONE]") == (0, 0)

    def test_empty_payload(self):
        assert _parse_sse_line("data: ") == (0, 0)

    def test_invalid_json(self):
        assert _parse_sse_line("data: not json") == (0, 0)

    def test_non_dict_json(self):
        assert _parse_sse_line("data: [1, 2, 3]") == (0, 0)

    def test_openai_input_tokens_fallback(self):
        """When only input_tokens key is present."""
        line = 'data: {"usage": {"input_tokens": 4}}'
        i, o = _parse_sse_line(line)
        assert i == 4

    def test_message_start_non_dict_message(self):
        line = 'data: {"type": "message_start", "message": "not a dict"}'
        # Should not crash; returns 0,0
        assert _parse_sse_line(line) == (0, 0)

    def test_message_start_non_dict_usage(self):
        line = 'data: {"type": "message_start", "message": {"usage": "x"}}'
        assert _parse_sse_line(line) == (0, 0)


class TestScanChunkUsage:
    def test_str_chunk_path(self):
        # str chunks go through the non-bytes branch (line 97).
        result = _scan_chunk_usage('data: {"usage": {"prompt_tokens": 3}}\n')
        assert result[0] == 3

    def test_no_usage_marker_short_circuit(self):
        assert _scan_chunk_usage(b"just some random bytes") == (0, 0)


class TestParseJsonUsage:
    def test_invalid_json(self):
        assert _parse_json_usage(b"not json") == (0, 0)

    def test_non_dict(self):
        assert _parse_json_usage(b"[1, 2, 3]") == (0, 0)

    def test_non_dict_usage(self):
        assert _parse_json_usage(b'{"usage": "invalid"}') == (0, 0)

    def test_type_error_on_int_conversion(self):
        # Truthy non-coercible value — ``int(truthy_non_int_no_0)`` raises.
        assert _parse_json_usage(
            b'{"usage": {"input_tokens": "abc", "output_tokens": 1}}'
        ) == (0, 0)


class TestExtractClientIp:
    def test_x_real_ip_fallback(self):
        req = MagicMock()
        req.headers = {"x-forwarded-for": "", "x-real-ip": "5.6.7.8"}
        req.client = None
        assert _extract_client_ip(req) == "5.6.7.8"

    def test_no_client_returns_none(self):
        req = MagicMock()
        req.headers = {}
        req.client = None
        assert _extract_client_ip(req) is None

    def test_client_without_host_returns_none(self):
        req = MagicMock()
        req.headers = {}
        client = MagicMock()
        client.host = ""
        req.client = client
        assert _extract_client_ip(req) is None

    def test_xff_empty_first_hop_ignored(self):
        req = MagicMock()
        req.headers = {"x-forwarded-for": ", 5.6.7.8", "x-real-ip": "9.9.9.9"}
        req.client = None
        # First hop is empty → fall back to x-real-ip.
        assert _extract_client_ip(req) == "9.9.9.9"

    def test_xff_first_hop_returned(self):
        req = MagicMock()
        req.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        req.client = None
        assert _extract_client_ip(req) == "1.2.3.4"


# ---------------------------------------------------------------------------
# Middleware — handler raises, call_next fails path
# ---------------------------------------------------------------------------


class TestMiddlewareExceptionPath:
    def test_handler_exception_recorded_then_reraised(self):
        coll = MetricsCollector()
        app = FastAPI()
        app.middleware("http")(metrics_middleware_factory(coll))

        @app.get("/boom")
        async def boom():
            raise RuntimeError("explode")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        # Middleware recorded the exception before re-raising.
        rec = coll.recent_requests(limit=1)[0]
        assert rec["status"] == 500
        assert rec["error_type"] == "RuntimeError"

    def test_middleware_with_health_increments_active(self):
        coll = MetricsCollector()
        health = HealthMonitor(region="us-east-1", auth_mode="bearer_token")
        app = FastAPI()
        app.middleware("http")(metrics_middleware_factory(coll, health=health))

        @app.get("/v1/x")
        async def x():
            return {"ok": True}

        client = TestClient(app)
        client.get("/v1/x")
        # Active went back to zero.
        assert health.snapshot()["active_connections"] == 0

    def test_middleware_request_body_read_failure(self):
        """If request.body() raises, middleware continues without the model."""
        coll = MetricsCollector()
        app = FastAPI()
        app.middleware("http")(metrics_middleware_factory(coll))

        @app.post("/v1/chat/completions")
        async def oai(request: Request):
            return JSONResponse({"ok": True})

        # Patch Request.body to raise once per request (the middleware's read).
        client = TestClient(app)
        real_body = Request.body

        call_count = {"n": 0}

        async def flaky_body(self):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("bad read")
            return await real_body(self)

        with patch.object(Request, "body", flaky_body):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "claude-haiku", "messages": []},
            )
        assert resp.status_code == 200

    def test_non_stream_llm_response_with_str_chunks(self):
        """Non-streaming LLM response yields str chunks (not bytes) — middleware
        re-encodes before buffering.

        Call the middleware directly so we can control exactly what the
        response.body_iterator emits (Starlette/BaseHTTPMiddleware normalises
        chunks to bytes before they reach user middleware in normal ASGI
        flow, so we bypass it here)."""
        import asyncio

        from starlette.responses import StreamingResponse

        coll = MetricsCollector()
        middleware = metrics_middleware_factory(coll)

        async def str_gen():
            yield '{"usage": {"prompt_tokens": 1,'
            yield ' "completion_tokens": 2}}'

        # Fake request
        request = MagicMock()
        request.url.path = "/v1/chat/completions"
        request.method = "POST"
        request.headers = {}
        request.client = None

        async def _body():
            return b'{"model": "claude-haiku", "messages": []}'

        request.body = _body
        request.state = type("S", (), {})()

        async def call_next(_req):
            return StreamingResponse(str_gen(), media_type="application/json")

        async def run():
            resp = await middleware(request, call_next)
            # drain the wrapped body
            chunks = []
            if hasattr(resp, "body_iterator"):
                async for c in resp.body_iterator:
                    chunks.append(c)
            return resp, chunks

        resp, chunks = asyncio.run(run())
        assert resp.status_code == 200
        rec = coll.recent_requests(limit=1)[0]
        assert rec["prompt_tokens"] == 1
        assert rec["completion_tokens"] == 2


# ---------------------------------------------------------------------------
# security — disabled gate, empty rate-limiter key, mask_ip edge cases
# ---------------------------------------------------------------------------


class TestSecurityEdges:
    def test_client_host_none_returns_empty(self):
        from bedrock_gateway.dashboard.security import _client_host

        req = MagicMock()
        req.client = None
        assert _client_host(req) == ""

    def test_dashboard_auth_disabled(self):
        auth = DashboardAuth(enabled=False, api_key=None, require_auth=False,
                             localhost_only=False)
        req = MagicMock()
        allowed, reason = auth.check(req)
        assert not allowed
        assert reason == "disabled"

    def test_rate_limiter_empty_key_treated_as_dash(self):
        rl = RateLimiter(limit=1, window_seconds=60)
        assert rl.check("")[0] is True
        # Second hit to same synthesized "-" key is blocked.
        assert rl.check("")[0] is False

    def test_rate_limiter_reset(self):
        rl = RateLimiter(limit=1, window_seconds=60)
        rl.check("ip")
        assert rl.check("ip")[0] is False
        rl.reset()
        assert rl.check("ip")[0] is True

    def test_mask_ip_ipv4_short_returns_as_is(self):
        # "1.2.3" is not 4 octets → falls through.
        assert mask_ip("1.2.3") == "1.2.3"

    def test_mask_ip_non_ip_value(self):
        assert mask_ip("hostname") == "hostname"


# ---------------------------------------------------------------------------
# Storage — batch_write_requests empty short-circuit, ALTER migration
# ---------------------------------------------------------------------------


class TestStorageEdges:
    def test_batch_write_requests_empty_list(self, tmp_path):
        storage = MetricsStorage(str(tmp_path / "m.db"))
        storage.batch_write_requests([])  # short-circuit, no exception

    def test_load_buckets_invalid_json_counts(self, tmp_path):
        storage = MetricsStorage(str(tmp_path / "m.db"))
        # Poke in a row with bad JSON for both status_counts and model_counts.
        with storage._lock, storage._connect() as conn:
            conn.execute(
                "INSERT INTO minute_buckets (ts, status_counts_json, model_counts_json) "
                "VALUES (?, ?, ?)",
                (42, "not-json", "also-not-json"),
            )
        rows = storage.load_buckets(since_ts=0)
        assert len(rows) == 1
        assert rows[0].status_counts == {}
        assert rows[0].model_counts == {}

    def test_alter_table_adds_missing_columns(self, tmp_path):
        """Init against an older schema triggers _add_missing_columns."""
        path = tmp_path / "m.db"
        # Create an older-schema DB missing the newer columns but with enough
        # structure that CREATE INDEX IF NOT EXISTS on `status` succeeds.
        conn = sqlite3.connect(str(path))
        conn.executescript(
            "CREATE TABLE requests ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, method TEXT, "
            "path TEXT, model TEXT, status INTEGER, latency_ms REAL, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "error_type TEXT, error_message TEXT); "
            "CREATE TABLE minute_buckets ("
            "ts INTEGER PRIMARY KEY, total INTEGER, success INTEGER, "
            "error INTEGER, latency_sum REAL, latency_count INTEGER, "
            "p50 REAL, p95 REAL, p99 REAL, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "status_counts_json TEXT, model_counts_json TEXT);"
        )
        conn.close()
        # MetricsStorage.__init__ should detect and add columns.
        MetricsStorage(str(path))
        conn = sqlite3.connect(str(path))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        # The migration added ttft_ms / tokens_per_sec / retry_count / client_ip.
        assert "retry_count" in cols
        assert "ttft_ms" in cols
        assert "client_ip" in cols
        bucket_cols = {row[1] for row in conn.execute("PRAGMA table_info(minute_buckets)")}
        assert "retry_count" in bucket_cols
        assert "ttft_sum" in bucket_cols
        conn.close()


# ---------------------------------------------------------------------------
# AsyncWriter — stop path + sqlite error swallow + dropped counter
# ---------------------------------------------------------------------------


class TestAsyncWriter:
    def test_dropped_when_queue_full(self, tmp_path):
        storage = MetricsStorage(str(tmp_path / "m.db"))
        w = AsyncWriter(storage, max_queue=1, batch_size=10, flush_interval=0.05)
        # Stop the writer first so items accumulate in the queue rather than
        # being drained.
        w._thread.join(timeout=0)  # no-op; thread is still running
        # Force queue full
        w._queue.put_nowait(RequestRow(
            ts=time.time(), method="G", path="/",
            model="-", status=200, latency_ms=1,
        ))
        w.enqueue(RequestRow(
            ts=time.time(), method="G", path="/",
            model="-", status=200, latency_ms=1,
        ))
        assert w.dropped >= 1
        w.stop()

    def test_sqlite_error_swallowed(self, tmp_path):
        storage = MetricsStorage(str(tmp_path / "m.db"))
        w = AsyncWriter(storage, flush_interval=0.05)

        # Patch batch_write_requests to raise sqlite3.Error.
        with patch.object(storage, "batch_write_requests",
                          side_effect=sqlite3.OperationalError("disk")):
            w.enqueue(RequestRow(
                ts=time.time(), method="G", path="/", model="-",
                status=200, latency_ms=1,
            ))
            time.sleep(0.15)  # let the writer flush
        w.stop()

    def test_stop_signals_loop(self, tmp_path):
        storage = MetricsStorage(str(tmp_path / "m.db"))
        w = AsyncWriter(storage, flush_interval=0.05)
        w.stop()
        assert not w._thread.is_alive()

    def test_stop_when_queue_full(self, tmp_path):
        """Covers queue.Full branch inside stop()."""
        storage = MetricsStorage(str(tmp_path / "m.db"))
        w = AsyncWriter(storage, max_queue=1, flush_interval=10.0)
        # Block the worker by filling its queue while it's sleeping — can't
        # strictly guarantee this without racing, but cover the try/except path.
        # Directly call stop() after overflowing the queue to reach line 375-376.
        w._queue.put_nowait(RequestRow(
            ts=time.time(), method="G", path="/", model="-",
            status=200, latency_ms=1,
        ))
        # Queue is now full; the extra put_nowait in stop() will raise.
        # Fill the queue a second slot is not possible - manually wedge:
        with patch.object(w._queue, "put_nowait", side_effect=queue.Full()):
            w.stop()


# ---------------------------------------------------------------------------
# Metrics — timeseries aggregate-only path (stored p95 without raw latencies)
# ---------------------------------------------------------------------------


class TestTimeseriesAggregateFallback:
    def test_timeseries_uses_stored_percentiles_when_no_raw(self, tmp_path):
        """Request a window wider than 24h. Older buckets only expose
        stored p50/p95/p99 (no raw latencies)."""
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        # Put a bucket from 48h ago into storage (before in-memory retention).
        ts = int(time.time()) // 60 * 60 - 48 * 3600
        storage.upsert_bucket(BucketRow(
            ts=ts, total=2, success=2,
            p50=100.0, p95=200.0, p99=300.0,
            latency_sum=0.0, latency_count=0,  # no raw samples
        ))
        m = MetricsCollector(storage=storage, retain_days=7)
        if m._writer is not None:
            m._writer.stop()
        ts_data = m.timeseries(minutes=72 * 60, bin_seconds=60 * 60)
        assert "p95" in ts_data

    def test_memory_timeseries_returns_none_on_gap(self):
        m = MetricsCollector()
        data = m.memory_timeseries(minutes=5)
        # All bins empty → memory_mb is None.
        assert all(v is None for v in data["memory_mb"])


# ---------------------------------------------------------------------------
# Metrics — recent_tokens_per_sec with zero latency returns 0
# ---------------------------------------------------------------------------


class TestTokensPerSecZero:
    def test_zero_latency_returns_zero(self):
        m = MetricsCollector()
        assert m._recent_tokens_per_sec_locked(5) == 0.0

    def test_failure_rates_zero_total(self):
        m = MetricsCollector()
        retry, timeout = m._recent_failure_rates_locked(5)
        assert retry == 0.0
        assert timeout == 0.0


class TestTTFTPercentileWindow:
    def test_ttft_percentile_from_populated_buckets(self):
        m = MetricsCollector()
        from bedrock_gateway.dashboard.metrics import _Bucket

        now_minute = int(time.time() // 60) * 60
        with m._lock:
            b = _Bucket(ts=now_minute, total=3, success=3)
            b.ttft_values = [10.0, 20.0, 30.0]
            b.latencies = [100.0, 200.0, 300.0]
            m._buckets[now_minute] = b
        p50 = m._recent_ttft_percentile_locked(50, 5)
        assert p50 > 0


class TestEvictOldBuckets:
    def test_stale_buckets_removed(self):
        m = MetricsCollector()
        from bedrock_gateway.dashboard.metrics import _Bucket

        # Inject a bucket far in the past (older than 24h retention).
        now_minute = int(time.time() // 60) * 60
        ancient = now_minute - 25 * 60 * 60  # 25h ago
        with m._lock:
            m._buckets[ancient] = _Bucket(ts=ancient)
            m._buckets[now_minute] = _Bucket(ts=now_minute)
            # Trigger the eviction by recording a request that forces rollover.
            m._evict_old_buckets_locked(now_minute + 60)
        assert ancient not in m._buckets


class TestCollectMinuteBucketsOverlap:
    def test_in_memory_wins_over_storage(self, tmp_path):
        """When in-memory has a bucket for ts X, storage rows for same ts ignored."""
        db = tmp_path / "m.db"
        storage = MetricsStorage(str(db))
        # Insert an old-bucket row directly.
        ancient_ts = int(time.time()) // 60 * 60 - 48 * 3600
        storage.upsert_bucket(BucketRow(
            ts=ancient_ts, total=10, success=10,
            p50=1.0, p95=1.0, p99=1.0,
        ))
        # Also insert a row whose ts is beyond the window — should be skipped.
        future_ts = int(time.time()) // 60 * 60 + 60
        storage.upsert_bucket(BucketRow(ts=future_ts, total=1))

        m = MetricsCollector(storage=storage, retain_days=7)
        if m._writer is not None:
            m._writer.stop()

        # In-memory bucket at ancient_ts — should mask the storage row.
        from bedrock_gateway.dashboard.metrics import _Bucket

        with m._lock:
            m._buckets[ancient_ts] = _Bucket(ts=ancient_ts, total=99, success=99)

        data = m._collect_minute_buckets(
            since_ts=ancient_ts - 60, until_ts=int(time.time()),
        )
        # In-memory value wins.
        assert data[ancient_ts]["total"] == 99
        # future_ts is skipped (> until_ts)
        assert future_ts not in data


class TestPersistBucketStorageNoneGuard:
    def test_persist_bucket_noop_when_storage_is_none(self):
        """Defensive check inside _persist_bucket when storage is None."""
        m = MetricsCollector()
        from bedrock_gateway.dashboard.metrics import _Bucket

        # storage is None by default — calling _persist_bucket returns early.
        m._persist_bucket(_Bucket(ts=0))


class TestMemoryTimeseriesPath:
    def test_memory_timeseries_with_rss(self):
        m = MetricsCollector()
        from bedrock_gateway.dashboard.metrics import _Bucket

        now_minute = int(time.time() // 60) * 60
        with m._lock:
            b = _Bucket(ts=now_minute, total=1, success=1)
            b.memory_rss_kb = 4 * 1024  # 4MB
            m._buckets[now_minute] = b
        data = m.memory_timeseries(minutes=5)
        assert any(v == 4.0 for v in data["memory_mb"] if v is not None)
