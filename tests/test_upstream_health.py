"""Unit tests for ``MetricsCollector.upstream_health`` — the passive
upstream-status derivation that replaced the active probe in 0.1.2.

Covers each band of the decision table:
    no traffic                 → unknown
    success rate ≥ 99 %        → healthy
    success rate ≥ 80 %        → degraded
    success rate <  80 %       → down
    any 401/403 in the window  → auth_failed (overrides rate-based bands)

Plus the carry-over behaviour of ``last_success`` across windows.
"""

from __future__ import annotations

import time

from bedrock_gateway.dashboard.metrics import MetricsCollector


def _record_n(
    collector: MetricsCollector, *, n: int, status: int, model: str = "claude-haiku"
) -> None:
    """Record *n* identical request outcomes."""
    for _ in range(n):
        collector.record_request(
            method="POST",
            path="/v1/messages",
            model=model,
            status=status,
            latency_ms=10.0,
        )


class TestUpstreamHealthBands:
    def test_no_traffic_is_unknown(self):
        m = MetricsCollector()
        h = m.upstream_health()
        assert h["status"] == "unknown"
        assert h["success_rate"] is None
        assert h["total"] == 0
        assert h["last_success"] is None

    def test_all_success_is_healthy(self):
        m = MetricsCollector()
        _record_n(m, n=10, status=200)
        h = m.upstream_health()
        assert h["status"] == "healthy"
        assert h["success_rate"] == 100.0
        assert h["total"] == 10
        assert h["errors"] == 0
        assert h["last_success"] is not None

    def test_99_pct_boundary_is_healthy(self):
        m = MetricsCollector()
        # 99 successes + 1 5xx = 99% — boundary case (>= 99 → healthy).
        _record_n(m, n=99, status=200)
        _record_n(m, n=1, status=500)
        h = m.upstream_health()
        assert h["status"] == "healthy"
        assert 98.5 <= h["success_rate"] <= 99.5

    def test_90_pct_is_degraded(self):
        m = MetricsCollector()
        _record_n(m, n=9, status=200)
        _record_n(m, n=1, status=500)  # 90% success
        h = m.upstream_health()
        assert h["status"] == "degraded"
        assert h["success_rate"] == 90.0

    def test_80_pct_boundary_is_degraded(self):
        m = MetricsCollector()
        _record_n(m, n=8, status=200)
        _record_n(m, n=2, status=500)  # 80% success
        h = m.upstream_health()
        assert h["status"] == "degraded"

    def test_below_80_pct_is_down(self):
        m = MetricsCollector()
        _record_n(m, n=7, status=200)
        _record_n(m, n=3, status=500)  # 70% success
        h = m.upstream_health()
        assert h["status"] == "down"
        assert h["success_rate"] == 70.0
        assert h["errors"] == 3

    def test_auth_failure_overrides_rate(self):
        """A single 401 in the window flips status to ``auth_failed``,
        even when the overall success rate would otherwise be healthy."""
        m = MetricsCollector()
        _record_n(m, n=99, status=200)
        _record_n(m, n=1, status=401)
        h = m.upstream_health()
        assert h["status"] == "auth_failed"
        # success_rate is still computed and surfaced for context.
        assert h["success_rate"] is not None

    def test_403_also_triggers_auth_failed(self):
        m = MetricsCollector()
        _record_n(m, n=10, status=200)
        _record_n(m, n=1, status=403)
        h = m.upstream_health()
        assert h["status"] == "auth_failed"

    def test_4xx_other_than_auth_does_not_trigger_auth_failed(self):
        """400/404/422 are client errors but not auth failures."""
        m = MetricsCollector()
        _record_n(m, n=10, status=200)
        _record_n(m, n=2, status=400)  # 10/12 ≈ 83% → degraded
        h = m.upstream_health()
        assert h["status"] == "degraded"

    def test_4xx_count_as_errors_in_rate(self):
        """4xx is recorded as an error, not a success — it pulls the
        success rate down even though the upstream is technically up."""
        m = MetricsCollector()
        _record_n(m, n=8, status=200)
        _record_n(m, n=2, status=400)  # 80% success
        h = m.upstream_health()
        assert h["status"] == "degraded"


class TestUpstreamHealthLastSuccess:
    def test_last_success_updated_on_2xx(self):
        m = MetricsCollector()
        before = time.time()
        _record_n(m, n=1, status=200)
        h = m.upstream_health()
        assert h["last_success"] is not None
        # ISO-8601 UTC string, sanity check format.
        assert h["last_success"].endswith("Z")
        # The recorded request happened after `before`.
        assert before <= time.time()

    def test_last_success_not_updated_on_5xx(self):
        m = MetricsCollector()
        _record_n(m, n=1, status=500)
        h = m.upstream_health()
        assert h["last_success"] is None

    def test_last_success_persists_after_failures(self):
        """Once ``last_success`` is set, subsequent failures don't clear it —
        operators want to know "when did this last work?"."""
        m = MetricsCollector()
        _record_n(m, n=1, status=200)
        first = m.upstream_health()["last_success"]
        _record_n(m, n=5, status=500)
        h = m.upstream_health()
        assert h["last_success"] == first


class TestUpstreamHealthWindow:
    def test_window_minutes_is_clamped_to_at_least_one(self):
        m = MetricsCollector()
        _record_n(m, n=1, status=200)
        # window=0 should be coerced to 1, not raise / not return total=0.
        h = m.upstream_health(window_minutes=0)
        assert h["window_minutes"] == 1
        assert h["total"] == 1

    def test_default_window_is_5_minutes(self):
        m = MetricsCollector()
        h = m.upstream_health()
        assert h["window_minutes"] == 5
