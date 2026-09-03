"""
Periodic per-model performance logging.

``MetricsCollector`` already records per-request ``latency_ms`` / ``ttft_ms`` /
``prompt_tokens`` / ``completion_tokens`` / ``model`` (computed by the metrics
middleware). This module turns that accumulation into a **log-only** summary:
every ``MODEL_PERF_INTERVAL_S`` seconds (default 30 minutes) it emits one
``MODEL-PERF`` log line per model, so an operator can compare latency and
output speed across models without a dashboard or API change.

Why a background task instead of a timer thread? The gateway runs on asyncio,
and the collector is a plain in-memory object with no I/O. Reusing the loop via
:meth:`ModelPerformanceReporter.start` / :meth:`stop` (the same pattern as
``dashboard.health.HealthMonitor``) keeps the reporter cooperative with the
app's startup/shutdown lifecycle and trivially testable via an injected sleep.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger("bedrock_gateway.model_report")

# Emit the per-model summary this often. 30 minutes keeps steady-state noise low
# while still surfacing a model degradation within a single operator shift.
MODEL_PERF_INTERVAL_S = 30 * 60


def log_model_performance(metrics: Any) -> None:
    """Compute and emit one ``MODEL-PERF`` log line per model.

    ``metrics`` is a :class:`~bedrock_gateway.dashboard.metrics.MetricsCollector`.
    With no recorded traffic a single "no requests" line is emitted so a silent
    journal doesn't read as "logging broken".
    """
    rows = metrics.model_performance()["models"]
    if not rows:
        logger.info("MODEL-PERF no requests recorded since start")
        return
    for r in rows:
        logger.info(
            "MODEL-PERF model=%s requests=%d ok=%.1f%% avg=%.0fms "
            "p50=%.0fms p95=%.0fms p99=%.0fms ttft_p50=%s out=%d tok/s=%.1f",
            r["model"],
            r["requests"],
            r["success_rate"],
            r["avg_latency_ms"],
            r["p50_latency_ms"],
            r["p95_latency_ms"],
            r["p99_latency_ms"],
            "-" if r["p50_ttft_ms"] is None else f'{r["p50_ttft_ms"]:.0f}',
            r["completion_tokens"],
            r["tokens_per_sec"],
        )


class ModelPerformanceReporter:
    """Background task that logs per-model performance on a fixed cadence.

    The ``sleep`` callable is injectable so tests can drive the loop
    deterministically (mirrors ``DailyFileHandler(clock=...)``).
    """

    def __init__(
        self,
        metrics: Any,
        *,
        interval_s: float = MODEL_PERF_INTERVAL_S,
        sleep: Callable[..., Any] = asyncio.sleep,
    ) -> None:
        self._metrics = metrics
        self._interval_s = interval_s
        self._sleep = sleep
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        """Launch the periodic reporter task.

        Safe to call multiple times; no-op if already running.
        """
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. called from a sync context) — defer.
            logger.debug("ModelPerformanceReporter.start called with no running loop")
            return
        self._task = loop.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the reporter task, if any. Idempotent."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    async def _loop(self) -> None:
        while True:
            await self._sleep(self._interval_s)
            log_model_performance(self._metrics)
