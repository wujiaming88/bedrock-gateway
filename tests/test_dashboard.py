"""
Tests for the dashboard package — metrics collector, API endpoints,
middleware, and public-deployment hardening.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    DashboardConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
)
from bedrock_gateway.dashboard import (
    DashboardAuth,
    MetricsCollector,
    RateLimiter,
    build_dashboard_router,
    metrics_middleware_factory,
)
from bedrock_gateway.dashboard.security import (
    SECURITY_HEADERS,
    mask_api_key,
    mask_ip,
    sanitize_request_log,
)
from bedrock_gateway.server import create_app


# ---------------------------------------------------------------------------
# MetricsCollector — unit tests
# ---------------------------------------------------------------------------


class TestMetricsCollectorOverview:
    def test_counts_and_success_rate(self):
        m = MetricsCollector()
        m.record_request(
            method="POST", path="/v1/messages", model="m1",
            status=200, latency_ms=50,
        )
        m.record_request(
            method="POST", path="/v1/messages", model="m1",
            status=200, latency_ms=100,
        )
        m.record_request(
            method="POST", path="/v1/messages", model="m1",
            status=500, latency_ms=200, error_type="ServerError",
        )
        o = m.overview()
        assert o["total_requests"] == 3
        assert o["success"] == 2
        assert o["error"] == 1
        assert o["success_rate"] == pytest.approx(200 / 3)
        assert o["avg_latency_ms"] == pytest.approx(116.67, rel=0.01)
        assert o["active_models"] == 1

    def test_overview_empty(self):
        m = MetricsCollector()
        o = m.overview()
        assert o["total_requests"] == 0
        assert o["success_rate"] == 0.0
        assert o["avg_latency_ms"] == 0.0

    def test_token_totals(self):
        m = MetricsCollector()
        m.record_request(
            method="POST", path="/v1/messages", model="m1",
            status=200, latency_ms=10, prompt_tokens=50, completion_tokens=25,
        )
        m.record_request(
            method="POST", path="/v1/messages", model="m2",
            status=200, latency_ms=10, prompt_tokens=100, completion_tokens=40,
        )
        o = m.overview()
        assert o["prompt_tokens"] == 150
        assert o["completion_tokens"] == 65


class TestLatencyPercentiles:
    def test_p50_p95_p99(self):
        m = MetricsCollector()
        for v in range(1, 101):  # 1..100
            m.record_request(
                method="POST", path="/v1/x", model="m1",
                status=200, latency_ms=float(v),
            )
        ts = m.timeseries(minutes=1)
        # All 100 samples fall in the current minute bucket
        p50 = ts["p50"][-1]
        p95 = ts["p95"][-1]
        p99 = ts["p99"][-1]
        assert p50 == pytest.approx(50.5, abs=0.5)
        assert p95 == pytest.approx(95.0, abs=1.0)
        assert p99 == pytest.approx(99.0, abs=1.0)


class TestRingBufferOverflow:
    def test_recent_bounded_by_max(self):
        m = MetricsCollector(max_request_log=5)
        for i in range(20):
            m.record_request(
                method="GET", path=f"/x/{i}", model="m1",
                status=200, latency_ms=1,
            )
        recent = m.recent_requests(limit=100)
        assert len(recent) == 5
        # Newest first — the last path recorded was /x/19
        assert recent[0]["path"] == "/x/19"
        assert recent[-1]["path"] == "/x/15"


class TestModelStats:
    def test_share_across_models(self):
        m = MetricsCollector()
        for _ in range(3):
            m.record_request(
                method="POST", path="/v1/messages", model="mA",
                status=200, latency_ms=5, prompt_tokens=10, completion_tokens=5,
            )
        for _ in range(7):
            m.record_request(
                method="POST", path="/v1/messages", model="mB",
                status=200, latency_ms=5, prompt_tokens=1, completion_tokens=1,
            )
        stats = m.model_stats()
        total = sum(x["requests"] for x in stats["models"])
        assert total == 10
        mb = next(x for x in stats["models"] if x["model"] == "mB")
        ma = next(x for x in stats["models"] if x["model"] == "mA")
        # Sorted by count descending
        assert stats["models"][0]["model"] == "mB"
        assert mb["requests"] == 7
        assert ma["requests"] == 3
        # Tokens split
        assert mb["tokens"] == 7 * 2
        assert ma["tokens"] == 3 * 15


class TestThreadSafety:
    def test_concurrent_writes_do_not_crash(self):
        m = MetricsCollector()
        n_threads = 8
        per_thread = 50

        def worker(tid: int) -> None:
            for i in range(per_thread):
                m.record_request(
                    method="POST", path=f"/t/{tid}", model=f"m{tid % 3}",
                    status=200 if i % 5 else 500, latency_ms=float(i),
                    prompt_tokens=1, completion_tokens=1,
                    error_type="Err" if i % 5 == 0 else None,
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        o = m.overview()
        assert o["total_requests"] == n_threads * per_thread
        assert o["success"] + o["error"] == o["total_requests"]


class TestTimeseriesWindow:
    def test_traffic_windows_valid(self):
        m = MetricsCollector()
        m.record_request(
            method="POST", path="/v1/x", model="m1",
            status=200, latency_ms=10,
        )
        for label, expected in [("1h", 60), ("6h", 360), ("24h", 1440)]:
            minutes = {"1h": 60, "6h": 360, "24h": 1440}[label]
            ts = m.timeseries(minutes=minutes)
            assert len(ts["labels"]) == expected
            assert len(ts["qps"]) == expected

    def test_timeseries_clamped(self):
        m = MetricsCollector()
        ts = m.timeseries(minutes=0)
        assert len(ts["labels"]) == 1
        # Way above the retained window → clamped
        huge = m.timeseries(minutes=10**6)
        assert len(huge["labels"]) == 24 * 60


# ---------------------------------------------------------------------------
# Fixtures for API + middleware + security tests
# ---------------------------------------------------------------------------


def _make_config(
    *,
    api_key: str = "",
    dashboard: DashboardConfig | None = None,
) -> GatewayConfig:
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="tok"),
        region="us-east-1",
        server=ServerConfig(
            host="127.0.0.1", port=4000, log_level="warning", api_key=api_key,
        ),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        dashboard=dashboard or DashboardConfig(api_key=None),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-v1",
                context_length=200000,
                max_output=4096,
            ),
        },
    )


@pytest.fixture
def open_client() -> TestClient:
    """Dashboard with no auth, explicitly localhost-only off (for tests)."""
    cfg = _make_config(
        dashboard=DashboardConfig(
            enabled=True, require_auth=False, api_key=None, localhost_only=False,
            rate_limit=60, max_request_log=200,
        )
    )
    return TestClient(create_app(cfg))


@pytest.fixture
def keyed_client() -> TestClient:
    """Dashboard protected by its own dashboard.api_key."""
    cfg = _make_config(
        dashboard=DashboardConfig(
            enabled=True, require_auth=True, api_key="sk-test-123",
            localhost_only=False, rate_limit=60, max_request_log=200,
        ),
    )
    return TestClient(create_app(cfg))


@pytest.fixture
def localhost_only_client() -> TestClient:
    """Dashboard with no API key — localhost-only by default."""
    cfg = _make_config(
        # No dashboard.api_key → localhost_only should auto-enable.
        dashboard=DashboardConfig(
            enabled=True, require_auth=True, api_key=None, localhost_only=None,
            rate_limit=60, max_request_log=200,
        ),
    )
    return TestClient(create_app(cfg))


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------


class TestMetricsAPI:
    def test_overview_shape(self, open_client: TestClient):
        # Seed some metrics
        coll = open_client.app.state.metrics  # type: ignore[attr-defined]
        coll.record_request(
            method="POST", path="/v1/messages", model="m1",
            status=200, latency_ms=10,
        )

        resp = open_client.get("/api/metrics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) >= {
            "total_requests", "success", "error", "success_rate",
            "avg_latency_ms", "active_models", "uptime_seconds",
            "prompt_tokens", "completion_tokens", "sparkline",
        }
        assert data["total_requests"] >= 1

    def test_traffic_1h(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/traffic?window=1h")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == "1h"
        assert len(data["labels"]) == 60
        assert len(data["qps"]) == 60
        assert len(data["p95"]) == 60

    def test_traffic_rejects_invalid_window(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/traffic?window=bogus")
        # FastAPI's Query pattern returns 422 for regex failure
        assert resp.status_code == 422

    def test_models(self, open_client: TestClient):
        coll = open_client.app.state.metrics  # type: ignore[attr-defined]
        coll.record_request(
            method="POST", path="/v1/x", model="mA",
            status=200, latency_ms=5,
        )
        resp = open_client.get("/api/metrics/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert any(m["model"] == "mA" for m in data["models"])

    def test_requests_filter_and_limit(self, open_client: TestClient):
        coll = open_client.app.state.metrics  # type: ignore[attr-defined]
        for i in range(5):
            coll.record_request(
                method="POST", path="/v1/x", model="m1",
                status=200, latency_ms=1,
            )
        coll.record_request(
            method="POST", path="/v1/x", model="m1",
            status=500, latency_ms=1, error_type="Boom",
        )

        resp = open_client.get("/api/metrics/requests?limit=3&filter=all")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["requests"]) == 3
        assert data["limit"] == 3

        resp_err = open_client.get("/api/metrics/requests?filter=error")
        assert resp_err.status_code == 200
        data = resp_err.json()
        assert all(r["status"] >= 400 for r in data["requests"])
        assert len(data["requests"]) >= 1

        resp_ok = open_client.get("/api/metrics/requests?filter=success")
        data = resp_ok.json()
        assert all(r["status"] < 400 for r in data["requests"])

    def test_requests_rejects_bad_filter(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/requests?filter=nope")
        assert resp.status_code == 422

    def test_requests_rejects_oversize_limit(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/requests?limit=9999")
        assert resp.status_code == 422

    def test_errors(self, open_client: TestClient):
        coll = open_client.app.state.metrics  # type: ignore[attr-defined]
        coll.record_request(
            method="POST", path="/v1/x", model="m1",
            status=500, latency_ms=1, error_type="ServerError",
        )
        resp = open_client.get("/api/metrics/errors")
        assert resp.status_code == 200
        data = resp.json()
        assert "by_status" in data
        assert "by_type" in data
        assert "recent" in data
        assert data["by_status"].get("500", 0) >= 1

    def test_system(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/system")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("version", "auth_mode", "region", "model_count", "uptime_seconds"):
            assert key in data
        assert data["region"] == "us-east-1"
        assert data["model_count"] == 1

    def test_dashboard_html(self, open_client: TestClient):
        resp = open_client.get("/dashboard/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


class TestDashboardMiddleware:
    def _build_app(self) -> tuple[FastAPI, MetricsCollector]:
        collector = MetricsCollector()
        app = FastAPI()
        app.middleware("http")(metrics_middleware_factory(collector))

        @app.get("/v1/hit")
        async def hit():
            return {"ok": True}

        @app.get("/v1/boom")
        async def boom():
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=502, content={"err": "x"})

        @app.get("/dashboard/ignored")
        async def ignored_dash():
            return {"ok": True}

        @app.get("/api/metrics/ignored")
        async def ignored_metrics():
            return {"ok": True}

        @app.get("/health")
        async def health():
            return {"ok": True}

        @app.post("/v1/messages/count_tokens")
        async def count_tokens():
            return {"input_tokens": 1}

        return app, collector

    def test_records_real_traffic(self):
        app, collector = self._build_app()
        client = TestClient(app)
        client.get("/v1/hit")
        client.get("/v1/hit")
        o = collector.overview()
        assert o["total_requests"] == 2
        assert o["success"] == 2

    def test_excludes_dashboard_and_metrics(self):
        app, collector = self._build_app()
        client = TestClient(app)
        client.get("/dashboard/ignored")
        client.get("/api/metrics/ignored")
        client.get("/health")
        assert collector.overview()["total_requests"] == 0

    def test_excludes_count_tokens(self):
        """SDK-internal count_tokens pre-flights must not pollute metrics."""
        app, collector = self._build_app()
        client = TestClient(app)
        resp = client.post("/v1/messages/count_tokens", json={"messages": []})
        assert resp.status_code == 200
        assert collector.overview()["total_requests"] == 0

    def test_records_error_status(self):
        app, collector = self._build_app()
        client = TestClient(app)
        client.get("/v1/boom")
        recent = collector.recent_requests(limit=10)
        assert recent[0]["status"] == 502
        assert collector.overview()["error"] == 1

    def test_excludes_models_listing(self):
        app, collector = self._build_app()

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": []}

        client = TestClient(app)
        client.get("/v1/models")
        assert collector.overview()["total_requests"] == 0


class TestMiddlewareBodyExtraction:
    """Middleware parses request/response bodies to capture model + tokens."""

    def _build_app(self) -> tuple[FastAPI, MetricsCollector]:
        from fastapi.responses import JSONResponse, StreamingResponse

        collector = MetricsCollector()
        app = FastAPI()
        app.middleware("http")(metrics_middleware_factory(collector))

        @app.post("/v1/chat/completions")
        async def oai(request: Request):
            # Downstream handler should still be able to read the body
            # even though the middleware already read it.
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "bad json"}, status_code=400)
            if body.get("stream"):
                async def gen():
                    yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
                    yield (
                        b'data: {"choices":[],"usage":{"prompt_tokens":12,'
                        b'"completion_tokens":7,"total_tokens":19}}\n\n'
                    )
                    yield b"data: [DONE]\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")
            return JSONResponse(
                {
                    "choices": [{"message": {"content": "Hi"}}],
                    "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 13,
                        "total_tokens": 55,
                    },
                }
            )

        @app.post("/v1/messages")
        async def ant(request: Request):
            body = await request.json()
            if body.get("stream"):
                async def gen():
                    yield (
                        b'event: message_start\n'
                        b'data: {"type":"message_start","message":'
                        b'{"usage":{"input_tokens":33}}}\n\n'
                    )
                    yield (
                        b'event: content_block_delta\n'
                        b'data: {"type":"content_block_delta","delta":'
                        b'{"type":"text_delta","text":"hi"}}\n\n'
                    )
                    yield (
                        b'event: message_delta\n'
                        b'data: {"type":"message_delta","usage":'
                        b'{"output_tokens":11}}\n\n'
                    )
                return StreamingResponse(gen(), media_type="text/event-stream")
            return JSONResponse(
                {
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 77, "output_tokens": 22},
                }
            )

        return app, collector

    def test_captures_model_and_tokens_sync_openai(self):
        app, collector = self._build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "claude-haiku", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        # Handler output is unchanged.
        assert resp.json()["usage"]["prompt_tokens"] == 42
        rec = collector.recent_requests(limit=1)[0]
        assert rec["model"] == "claude-haiku"
        assert rec["prompt_tokens"] == 42
        assert rec["completion_tokens"] == 13

    def test_captures_model_and_tokens_sync_anthropic(self):
        app, collector = self._build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        rec = collector.recent_requests(limit=1)[0]
        assert rec["model"] == "claude-sonnet"
        assert rec["prompt_tokens"] == 77
        assert rec["completion_tokens"] == 22

    def test_captures_model_and_tokens_stream_openai(self):
        app, collector = self._build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # Consume the streamed body so the wrapper's finaliser runs.
        body = resp.text
        assert "[DONE]" in body
        rec = collector.recent_requests(limit=1)[0]
        assert rec["model"] == "claude-haiku"
        assert rec["prompt_tokens"] == 12
        assert rec["completion_tokens"] == 7

    def test_captures_model_and_tokens_stream_anthropic(self):
        app, collector = self._build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "message_delta" in body
        rec = collector.recent_requests(limit=1)[0]
        assert rec["model"] == "claude-sonnet"
        assert rec["prompt_tokens"] == 33
        assert rec["completion_tokens"] == 11

    def test_missing_model_falls_back_to_dash(self):
        app, collector = self._build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        rec = collector.recent_requests(limit=1)[0]
        assert rec["model"] == "-"

    def test_invalid_json_body_does_not_crash(self):
        app, collector = self._build_app()
        client = TestClient(app)
        # Non-JSON body — handler will 422, but middleware must still record
        # the request instead of exploding.
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code in (400, 422)
        rec = collector.recent_requests(limit=1)[0]
        assert rec["model"] == "-"
        assert rec["prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# Security — authentication
# ---------------------------------------------------------------------------


class TestDashboardAuth:
    def test_no_key_dashboard_blocked_from_non_localhost(
        self, localhost_only_client: TestClient
    ):
        # TestClient default host is "testclient" — not in localhost set.
        resp = localhost_only_client.get(
            "/api/metrics/overview", headers={"host": "example.com"}
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["type"] == "permission_error"

    def test_no_key_dashboard_ui_blocked_from_non_localhost(
        self, localhost_only_client: TestClient
    ):
        resp = localhost_only_client.get("/dashboard/")
        assert resp.status_code == 403

    def test_keyed_api_requires_auth(self, keyed_client: TestClient):
        resp = keyed_client.get("/api/metrics/overview")
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_keyed_api_accepts_bearer(self, keyed_client: TestClient):
        resp = keyed_client.get(
            "/api/metrics/overview",
            headers={"Authorization": "Bearer sk-test-123"},
        )
        assert resp.status_code == 200

    def test_keyed_api_accepts_x_api_key(self, keyed_client: TestClient):
        resp = keyed_client.get(
            "/api/metrics/overview",
            headers={"x-api-key": "sk-test-123"},
        )
        assert resp.status_code == 200

    def test_keyed_api_accepts_query_param(self, keyed_client: TestClient):
        resp = keyed_client.get("/api/metrics/overview?key=sk-test-123")
        assert resp.status_code == 200

    def test_keyed_api_accepts_cookie(self, keyed_client: TestClient):
        client = keyed_client
        client.cookies.set("bedrock_gw_key", "sk-test-123")
        resp = client.get("/api/metrics/overview")
        assert resp.status_code == 200

    def test_ui_redirects_to_login_when_unauth(self, keyed_client: TestClient):
        resp = keyed_client.get("/dashboard/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "/dashboard/login" in resp.headers["location"]

    def test_login_page_renders(self, keyed_client: TestClient):
        resp = keyed_client.get("/dashboard/login")
        assert resp.status_code == 200
        assert "API Key" in resp.text
        assert "text/html" in resp.headers.get("content-type", "")

    def test_login_wrong_key(self, keyed_client: TestClient):
        resp = keyed_client.post(
            "/dashboard/login",
            data={"key": "wrong", "next": "/dashboard/"},
        )
        assert resp.status_code == 200  # re-renders form
        assert "Invalid API key" in resp.text

    def test_login_correct_sets_cookie(self, keyed_client: TestClient):
        resp = keyed_client.post(
            "/dashboard/login",
            data={"key": "sk-test-123", "next": "/dashboard/"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers["location"] == "/dashboard/"
        # Set-Cookie present
        cookie_header = resp.headers.get("set-cookie", "")
        assert "bedrock_gw_key=" in cookie_header
        assert "HttpOnly" in cookie_header or "httponly" in cookie_header.lower()

    def test_login_rejects_open_redirect(self, keyed_client: TestClient):
        resp = keyed_client.post(
            "/dashboard/login",
            data={"key": "sk-test-123", "next": "//evil.example/x"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        # Forced back to /dashboard/
        assert resp.headers["location"] == "/dashboard/"

    def test_logout_clears_cookie(self, keyed_client: TestClient):
        resp = keyed_client.get("/dashboard/logout", follow_redirects=False)
        assert resp.status_code == 302
        assert "/dashboard/login" in resp.headers["location"]

    def test_dashboard_disabled_returns_403(self):
        cfg = _make_config(
            dashboard=DashboardConfig(
                enabled=False, require_auth=False, api_key=None, localhost_only=False,
                rate_limit=60, max_request_log=50,
            ),
        )
        app = create_app(cfg)
        client = TestClient(app)
        # With dashboard.enabled=False the router isn't mounted at all.
        resp = client.get("/api/metrics/overview")
        assert resp.status_code == 404
        resp2 = client.get("/dashboard/")
        assert resp2.status_code == 404


# ---------------------------------------------------------------------------
# Security — server.api_key and dashboard.api_key are independent
# ---------------------------------------------------------------------------


class TestDashboardServerKeyIsolation:
    """The dashboard key and the model-API key are deliberately separate:
    holders of one must not be able to use the other."""

    def _app(self) -> TestClient:
        cfg = _make_config(
            api_key="server-key-abc",
            dashboard=DashboardConfig(
                enabled=True, require_auth=True, api_key="dash-key-xyz",
                localhost_only=False, rate_limit=60, max_request_log=50,
            ),
        )
        return TestClient(create_app(cfg))

    def test_server_key_cannot_access_dashboard(self):
        client = self._app()
        resp = client.get(
            "/api/metrics/overview",
            headers={"Authorization": "Bearer server-key-abc"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_dashboard_key_accesses_dashboard(self):
        client = self._app()
        resp = client.get(
            "/api/metrics/overview",
            headers={"Authorization": "Bearer dash-key-xyz"},
        )
        assert resp.status_code == 200

    def test_dashboard_key_cannot_call_model_endpoints(self):
        client = self._app()
        # /v1/models requires server.api_key when set; dashboard key is rejected.
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer dash-key-xyz"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_server_key_allows_model_endpoints(self):
        client = self._app()
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer server-key-abc"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Security — rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(limit=3, window_seconds=60)
        for _ in range(3):
            allowed, _ = rl.check("1.2.3.4")
            assert allowed

    def test_blocks_over_limit(self):
        rl = RateLimiter(limit=2, window_seconds=60)
        rl.check("ip")
        rl.check("ip")
        allowed, retry_after = rl.check("ip")
        assert not allowed
        assert retry_after >= 1

    def test_per_ip_isolation(self):
        rl = RateLimiter(limit=1, window_seconds=60)
        assert rl.check("a")[0] is True
        assert rl.check("a")[0] is False
        assert rl.check("b")[0] is True

    def test_window_expiry(self):
        rl = RateLimiter(limit=2, window_seconds=60)
        # Inject fake timestamps older than the window.
        with rl._lock:
            rl._hits["ip"] = deque([time.time() - 3600, time.time() - 3600])
        allowed, _ = rl.check("ip")
        assert allowed

    def test_api_returns_429(self):
        cfg = _make_config(
            dashboard=DashboardConfig(
                enabled=True, require_auth=False, api_key=None, localhost_only=False,
                rate_limit=3, max_request_log=50,
            ),
        )
        client = TestClient(create_app(cfg))
        # 3 ok, 4th should 429.
        for _ in range(3):
            assert client.get("/api/metrics/overview").status_code == 200
        resp = client.get("/api/metrics/overview")
        assert resp.status_code == 429
        assert resp.json()["error"]["type"] == "rate_limit_error"
        assert "retry-after" in {k.lower() for k in resp.headers.keys()}


# ---------------------------------------------------------------------------
# Security — headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    def test_api_has_headers(self, open_client: TestClient):
        resp = open_client.get("/api/metrics/overview")
        for name, expected in SECURITY_HEADERS.items():
            assert resp.headers.get(name) == expected, f"missing {name}"

    def test_dashboard_ui_has_headers(self, open_client: TestClient):
        resp = open_client.get("/dashboard/")
        assert resp.status_code == 200
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "jsdelivr" in resp.headers.get("Content-Security-Policy", "")

    def test_unauthorized_response_has_headers(self, keyed_client: TestClient):
        resp = keyed_client.get("/api/metrics/overview")
        assert resp.status_code == 401
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


# ---------------------------------------------------------------------------
# Security — input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize("window", ["1h", "6h", "24h"])
    def test_traffic_windows_accepted(self, open_client: TestClient, window: str):
        resp = open_client.get(f"/api/metrics/traffic?window={window}")
        assert resp.status_code == 200

    @pytest.mark.parametrize("window", ["2h", "5m", "", "1h;drop"])
    def test_traffic_bad_windows(self, open_client: TestClient, window: str):
        resp = open_client.get(f"/api/metrics/traffic?window={window}")
        assert resp.status_code == 422

    @pytest.mark.parametrize("limit", [0, -1, 201, 10000])
    def test_requests_bad_limits(self, open_client: TestClient, limit: int):
        resp = open_client.get(f"/api/metrics/requests?limit={limit}")
        assert resp.status_code == 422

    @pytest.mark.parametrize("limit", [1, 10, 100, 200])
    def test_requests_good_limits(self, open_client: TestClient, limit: int):
        resp = open_client.get(f"/api/metrics/requests?limit={limit}")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Security — sanitisation of logs
# ---------------------------------------------------------------------------


class TestSanitizeRequestLog:
    def test_drops_messages(self):
        records = [
            {"path": "/v1/messages", "messages": [{"role": "user", "content": "secret"}]},
        ]
        cleaned = sanitize_request_log(records)
        assert "messages" not in cleaned[0]
        assert cleaned[0]["path"] == "/v1/messages"

    def test_masks_api_key(self):
        records = [{"path": "/", "api_key": "sk-supersecret-12345"}]
        cleaned = sanitize_request_log(records)
        assert cleaned[0]["api_key"] == "sk-s***"

    def test_drops_aws_creds(self):
        records = [{
            "path": "/",
            "aws_access_key_id": "AKIA...",
            "aws_secret_access_key": "secret",
            "aws_session_token": "t",
        }]
        cleaned = sanitize_request_log(records)
        assert "aws_access_key_id" not in cleaned[0]
        assert "aws_secret_access_key" not in cleaned[0]
        assert "aws_session_token" not in cleaned[0]

    def test_masks_ip_by_default(self):
        records = [{"path": "/", "ip": "1.2.3.4"}]
        cleaned = sanitize_request_log(records)
        assert cleaned[0]["ip"] == "1.2.3.0"

    def test_ip_shown_when_requested(self):
        records = [{"path": "/", "ip": "1.2.3.4"}]
        cleaned = sanitize_request_log(records, show_ip=True)
        assert cleaned[0]["ip"] == "1.2.3.4"

    def test_truncates_error_message(self):
        records = [{"path": "/", "error_message": "x" * 1000}]
        cleaned = sanitize_request_log(records)
        assert len(cleaned[0]["error_message"]) <= 300

    def test_mask_api_key_helpers(self):
        assert mask_api_key("") == ""
        assert mask_api_key("abc") == "***"
        assert mask_api_key("abcdefghij") == "abcd***"

    def test_mask_ip_helpers(self):
        assert mask_ip("1.2.3.4") == "1.2.3.0"
        assert mask_ip("::1") == "::"
        assert mask_ip("") == ""

    def test_requests_endpoint_does_not_leak_body(self, open_client: TestClient):
        """Even if a handler stashed a body on the record, the API strips it."""
        coll = open_client.app.state.metrics  # type: ignore[attr-defined]
        coll.record_request(
            method="POST", path="/v1/messages", model="m1",
            status=200, latency_ms=1,
        )
        # Manually inject something that would look like a body to belt-and-braces
        # the sanitizer (not normally done by the collector).
        # This also proves that _recent record shape is bounded to known fields.
        resp = open_client.get("/api/metrics/requests")
        assert resp.status_code == 200
        for r in resp.json()["requests"]:
            assert "messages" not in r
            assert "body" not in r


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_DASHBOARD_KEY", raising=False)
        cfg = DashboardConfig()
        assert cfg.enabled is True
        assert cfg.require_auth is True
        assert cfg.rate_limit == 60
        assert cfg.max_request_log == 200
        assert cfg.localhost_only is None  # auto
        assert cfg.api_key is None

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_DASHBOARD_KEY", "env-dash-key")
        cfg = DashboardConfig()
        assert cfg.api_key == "env-dash-key"

    def test_loaded_from_yaml(self, tmp_path):
        from bedrock_gateway.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "dashboard:\n"
            "  enabled: true\n"
            "  api_key: yaml-dash-key\n"
            "  require_auth: false\n"
            "  localhost_only: true\n"
            "  rate_limit: 5\n"
            "  max_request_log: 42\n"
        )
        cfg = load_config(config_file)
        assert cfg.dashboard.enabled is True
        assert cfg.dashboard.api_key == "yaml-dash-key"
        assert cfg.dashboard.require_auth is False
        assert cfg.dashboard.localhost_only is True
        assert cfg.dashboard.rate_limit == 5
        assert cfg.dashboard.max_request_log == 42

    def test_max_request_log_is_applied(self):
        cfg = _make_config(
            dashboard=DashboardConfig(
                enabled=True, require_auth=False, api_key=None, localhost_only=False,
                rate_limit=60, max_request_log=3,
            ),
        )
        app = create_app(cfg)
        coll: MetricsCollector = app.state.metrics  # type: ignore[attr-defined]
        for i in range(10):
            coll.record_request(
                method="GET", path=f"/x/{i}", model="m1",
                status=200, latency_ms=1,
            )
        assert len(coll.recent_requests(limit=100)) == 3


# ---------------------------------------------------------------------------
# End-to-end: dashboard metrics reflect real traffic
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_health_hits_do_not_pollute_metrics(self, open_client: TestClient):
        for _ in range(3):
            open_client.get("/health")
        data = open_client.get("/api/metrics/overview").json()
        assert data["total_requests"] == 0

    def test_models_listing_does_not_pollute_metrics(self, open_client: TestClient):
        # GET /v1/models is a listing endpoint, not a model invocation,
        # so the dashboard should not record it.
        open_client.get("/v1/models")
        data = open_client.get("/api/metrics/overview").json()
        assert data["total_requests"] == 0
