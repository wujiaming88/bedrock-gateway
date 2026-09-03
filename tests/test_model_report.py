"""Tests for the log-only per-model performance report.

Covers ``MetricsCollector.model_performance`` (the per-model aggregation and
summary), ``model_report.log_model_performance`` (the log line), and the
``ModelPerformanceReporter`` background task lifecycle.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from bedrock_gateway.dashboard.metrics import MetricsCollector, _ModelPerf
from bedrock_gateway.model_report import (
    ModelPerformanceReporter,
    log_model_performance,
)


# ---------------------------------------------------------------------------
# model_performance() — per-model summary
# ---------------------------------------------------------------------------


def test_model_performance_empty_collector():
    coll = MetricsCollector()
    assert coll.model_performance() == {"models": [], "window": "since_start"}


def test_model_performance_computes_per_model():
    coll = MetricsCollector()
    # "fast" — 2 successful requests, both with TTFT and completion tokens.
    coll.record_request(
        method="POST", path="/v1/x", model="fast", status=200,
        latency_ms=100.0, prompt_tokens=10, completion_tokens=50, ttft_ms=10.0,
    )
    coll.record_request(
        method="POST", path="/v1/x", model="fast", status=200,
        latency_ms=300.0, prompt_tokens=20, completion_tokens=50, ttft_ms=30.0,
    )
    # "slow" — 1 failed request, no TTFT, no completion tokens.
    coll.record_request(
        method="POST", path="/v1/x", model="slow", status=500,
        latency_ms=1000.0, prompt_tokens=5, completion_tokens=0, ttft_ms=None,
    )
    # "zero" — latency 0 exercises the tokens_per_sec divide-by-zero guard.
    coll.record_request(
        method="POST", path="/v1/x", model="zero", status=200,
        latency_ms=0.0, completion_tokens=10,
    )

    data = coll.model_performance()
    assert data["window"] == "since_start"
    assert [m["model"] for m in data["models"]] == ["fast", "slow", "zero"]

    fast = data["models"][0]
    assert fast["requests"] == 2
    assert fast["success_rate"] == 100.0
    assert fast["avg_latency_ms"] == 200.0
    assert fast["p50_latency_ms"] == 200.0
    assert fast["p95_latency_ms"] == 290.0
    assert fast["p99_latency_ms"] == 298.0
    assert fast["tokens_per_sec"] == 250.0
    assert fast["prompt_tokens"] == 30
    assert fast["completion_tokens"] == 100
    assert fast["avg_ttft_ms"] == 20.0
    assert fast["p50_ttft_ms"] == 20.0

    slow = data["models"][1]
    assert slow["requests"] == 1
    assert slow["success_rate"] == 0.0
    assert slow["avg_latency_ms"] == 1000.0
    assert slow["p50_latency_ms"] == 1000.0
    assert slow["p95_latency_ms"] == 1000.0
    assert slow["p99_latency_ms"] == 1000.0
    assert slow["tokens_per_sec"] == 0.0
    assert slow["prompt_tokens"] == 5
    assert slow["completion_tokens"] == 0
    assert slow["avg_ttft_ms"] is None
    assert slow["p50_ttft_ms"] is None

    zero = data["models"][2]
    assert zero["tokens_per_sec"] == 0.0
    assert zero["avg_latency_ms"] == 0.0


def test_model_performance_skips_zero_request_models():
    coll = MetricsCollector()
    # A _ModelPerf that never saw a record (defensive guard branch).
    coll._model_perf["phantom"] = _ModelPerf()
    coll.record_request(
        method="POST", path="/v1/x", model="real", status=200, latency_ms=10.0,
    )
    models = [m["model"] for m in coll.model_performance()["models"]]
    assert models == ["real"]


# ---------------------------------------------------------------------------
# log_model_performance() — log lines
# ---------------------------------------------------------------------------


def test_log_model_performance_empty(caplog):
    coll = MetricsCollector()
    with caplog.at_level(logging.INFO, logger="bedrock_gateway.model_report"):
        log_model_performance(coll)
    assert "MODEL-PERF no requests recorded since start" in caplog.text


def test_log_model_performance_one_line_per_model(caplog):
    coll = MetricsCollector()
    coll.record_request(
        method="POST", path="/v1/x", model="fast", status=200,
        latency_ms=100.0, prompt_tokens=10, completion_tokens=50, ttft_ms=10.0,
    )
    coll.record_request(
        method="POST", path="/v1/x", model="fast", status=200,
        latency_ms=300.0, prompt_tokens=20, completion_tokens=50, ttft_ms=30.0,
    )
    with caplog.at_level(logging.INFO, logger="bedrock_gateway.model_report"):
        log_model_performance(coll)
    assert "MODEL-PERF model=fast requests=2 ok=100.0%" in caplog.text
    assert "tok/s=250.0" in caplog.text


def test_log_model_performance_ttft_dash_when_absent(caplog):
    coll = MetricsCollector()
    coll.record_request(
        method="POST", path="/v1/x", model="nottft", status=200,
        latency_ms=50.0, completion_tokens=5, ttft_ms=None,
    )
    with caplog.at_level(logging.INFO, logger="bedrock_gateway.model_report"):
        log_model_performance(coll)
    # No TTFT sample → the p50_ttft column renders as "-".
    assert "ttft_p50=-" in caplog.text


# ---------------------------------------------------------------------------
# ModelPerformanceReporter — lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reporter_loop_logs_then_cancels(monkeypatch):
    coll = MetricsCollector()
    logged: list[object] = []
    monkeypatch.setattr(
        "bedrock_gateway.model_report.log_model_performance",
        lambda m: logged.append(m),
    )
    calls = {"n": 0}

    async def fake_sleep(_interval):
        calls["n"] += 1
        if calls["n"] > 1:
            raise asyncio.CancelledError()

    reporter = ModelPerformanceReporter(coll, interval_s=1, sleep=fake_sleep)
    reporter.start()
    task = reporter._task
    assert task is not None

    with pytest.raises(asyncio.CancelledError):
        await task

    assert logged == [coll]
    # stop() is idempotent and safe after the task has already finished.
    await reporter.stop()
    await reporter.stop()


@pytest.mark.asyncio
async def test_reporter_start_is_idempotent():
    coll = MetricsCollector()

    async def never_sleep(_interval):
        await asyncio.sleep(3600)

    reporter = ModelPerformanceReporter(coll, interval_s=1, sleep=never_sleep)
    reporter.start()
    first = reporter._task
    reporter.start()
    assert reporter._task is first
    await reporter.stop()
    assert reporter._task is None


def test_reporter_start_without_running_loop_is_noop():
    reporter = ModelPerformanceReporter(MetricsCollector())
    reporter.start()  # no running event loop → deferred, no task
    assert reporter._task is None
