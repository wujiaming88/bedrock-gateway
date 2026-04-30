"""
Thread-safe in-memory metrics collector for the Bedrock Gateway dashboard.

Design notes:
  * Retains the last 24h of per-minute buckets for QPS / error-rate / latency.
  * Retains a bounded ring buffer of the most recent request entries for the
    "Request Log" panel.
  * Optionally persists buckets + recent requests via :class:`MetricsStorage`
    so the dashboard survives a process restart.
  * All mutating operations take an internal ``threading.Lock`` so the
    collector is safe to call from the async request path and from other
    threads (e.g. Starlette's threadpool for sync handlers).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .storage import AsyncWriter, BucketRow, MetricsStorage, RequestRow


logger = logging.getLogger("bedrock_gateway.dashboard.metrics")


# Per-minute buckets kept for 24h
_BUCKETS = 24 * 60
_BUCKET_SECONDS = 60

# Max recent requests retained for the log table
_MAX_RECENT_REQUESTS = 200

# Max recent errors retained for the error panel
_MAX_RECENT_ERRORS = 50

# How many recent minute-buckets to aggregate when computing dashboard-gauge
# percentiles. Using a rolling window of several minutes makes the gauge
# meaningful even when the most recent minute had only 1-2 requests.
_GAUGE_WINDOW_MINUTES = 5

# Rolling window (in minute-buckets) used to compute the "tokens/h" gauge —
# "tokens in the last 60 minutes" is what operators actually care about,
# not "tokens since process start".
_TOKEN_RATE_WINDOW_MINUTES = 60


@dataclass
class RequestRecord:
    """Single request summary for the log table."""
    ts: float
    method: str
    path: str
    model: str
    status: int
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "method": self.method,
            "path": self.path,
            "model": self.model,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class _Bucket:
    """Per-minute aggregation bucket."""
    ts: int  # unix seconds at minute boundary
    total: int = 0
    success: int = 0
    error: int = 0
    latencies: list[float] = field(default_factory=list)
    status_counts: dict[int, int] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Set to True once this bucket has been persisted after rotating out of
    # the "current minute" — keeps us from writing it to disk every second.
    flushed: bool = False


def _percentile(values: list[float], p: float) -> float:
    """Return the *p*th percentile (0-100) of *values*."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _classify_error(status: int, error_type: str | None) -> str:
    """
    Normalise an HTTP status + optional exception name into a coarse error
    category suitable for the "BY TYPE" panel.
    """
    if error_type:
        return error_type
    if status == 401 or status == 403:
        return "auth_error"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limit"
    if status in (503, 529):
        return "overloaded"
    if 500 <= status < 600:
        return "internal_error"
    if 400 <= status < 500:
        return "client_error"
    return "unknown"


class MetricsCollector:
    """
    In-memory metrics aggregator.

    Usage::

        m = MetricsCollector()
        m.record_request(
            method="POST", path="/v1/messages", model="claude-haiku",
            status=200, latency_ms=123.4,
            prompt_tokens=100, completion_tokens=50,
        )
        snapshot = m.overview()
    """

    def __init__(
        self,
        *,
        max_request_log: int = _MAX_RECENT_REQUESTS,
        storage: MetricsStorage | None = None,
        retain_days: int = 7,
    ) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._buckets: dict[int, _Bucket] = {}
        self._recent: deque[RequestRecord] = deque(maxlen=max(1, int(max_request_log)))
        self._errors: deque[RequestRecord] = deque(maxlen=_MAX_RECENT_ERRORS)
        # Lifetime totals (survive bucket rotation)
        self._total_requests = 0
        self._total_success = 0
        self._total_error = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._model_totals: dict[str, int] = {}
        self._model_tokens: dict[str, int] = {}

        self._storage = storage
        self._retain_days = max(1, int(retain_days))
        self._writer: AsyncWriter | None = None
        if storage is not None:
            self._writer = AsyncWriter(storage)
            self._load_from_storage(storage)

    def _load_from_storage(self, storage: MetricsStorage) -> None:
        """Rehydrate in-memory state from the persistence layer."""
        # Recent requests (newest first in DB; we push chronologically).
        try:
            recent = storage.load_recent_requests(limit=self._recent.maxlen or 200)
        except Exception:  # noqa: BLE001 — never let restart fail on IO
            logger.warning("failed to load recent requests from storage", exc_info=True)
            recent = []
        for r in reversed(recent):
            self._recent.append(
                RequestRecord(
                    ts=r.ts, method=r.method, path=r.path, model=r.model,
                    status=r.status, latency_ms=r.latency_ms,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    error_type=r.error_type, error_message=r.error_message,
                )
            )
            if r.status >= 400:
                self._errors.append(self._recent[-1])

        # Minute buckets over the retained window.
        since = int(time.time()) - _BUCKETS * _BUCKET_SECONDS
        try:
            buckets = storage.load_buckets(since)
        except Exception:  # noqa: BLE001
            logger.warning("failed to load buckets from storage", exc_info=True)
            buckets = []

        for row in buckets:
            b = _Bucket(
                ts=row.ts, total=row.total, success=row.success, error=row.error,
                status_counts=dict(row.status_counts),
                model_counts=dict(row.model_counts),
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                flushed=True,
            )
            # Recreate a representative latency population so _percentile
            # recomputes sensibly. We only store the aggregates on disk, so
            # plant the mean at ``latency_count`` copies — good enough for
            # the gauge, and the true percentile is served directly from
            # ``row.p50/p95/p99`` via _bucket_percentiles.
            if row.latency_count > 0 and row.latency_sum > 0:
                avg = row.latency_sum / row.latency_count
                b.latencies = [avg] * min(row.latency_count, 200)
            self._buckets[row.ts] = b

            self._total_requests += row.total
            self._total_success += row.success
            self._total_error += row.error
            self._total_prompt_tokens += row.prompt_tokens
            self._total_completion_tokens += row.completion_tokens
            for m, n in row.model_counts.items():
                self._model_totals[m] = self._model_totals.get(m, 0) + n

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        *,
        method: str,
        path: str,
        model: str,
        status: int,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record a completed request."""
        now = time.time()
        minute = int(now // _BUCKET_SECONDS) * _BUCKET_SECONDS

        is_err = status >= 400
        normalised_error_type = (
            _classify_error(status, error_type) if is_err else error_type
        )

        rec = RequestRecord(
            ts=now,
            method=method,
            path=path,
            model=model or "-",
            status=status,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_type=normalised_error_type,
            error_message=error_message,
        )

        to_flush: list[_Bucket] = []
        with self._lock:
            bucket = self._buckets.get(minute)
            if bucket is None:
                bucket = _Bucket(ts=minute)
                self._buckets[minute] = bucket
                # When we cross into a new minute, flush any buckets that
                # are now "closed" so the persistence layer has a durable
                # record of completed minutes.
                to_flush = [
                    b for b in self._buckets.values()
                    if b.ts < minute and not b.flushed
                ]
                for b in to_flush:
                    b.flushed = True
                self._evict_old_buckets_locked(minute)

            bucket.total += 1
            bucket.latencies.append(latency_ms)
            bucket.status_counts[status] = bucket.status_counts.get(status, 0) + 1
            if model:
                bucket.model_counts[model] = bucket.model_counts.get(model, 0) + 1
            bucket.prompt_tokens += prompt_tokens
            bucket.completion_tokens += completion_tokens

            if is_err:
                bucket.error += 1
            else:
                bucket.success += 1

            self._recent.append(rec)
            if is_err:
                self._errors.append(rec)

            self._total_requests += 1
            if is_err:
                self._total_error += 1
            else:
                self._total_success += 1
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            if model:
                self._model_totals[model] = self._model_totals.get(model, 0) + 1
                self._model_tokens[model] = (
                    self._model_tokens.get(model, 0)
                    + prompt_tokens + completion_tokens
                )

        # Persistence is best-effort; drop on overflow rather than blocking.
        if self._writer is not None:
            self._writer.enqueue(
                RequestRow(
                    ts=now, method=method, path=path, model=model or "-",
                    status=status, latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error_type=normalised_error_type,
                    error_message=error_message,
                )
            )
        if self._storage is not None:
            for b in to_flush:
                self._persist_bucket(b)

    def _persist_bucket(self, bucket: _Bucket) -> None:
        if self._storage is None:
            return
        try:
            self._storage.upsert_bucket(
                BucketRow(
                    ts=bucket.ts, total=bucket.total,
                    success=bucket.success, error=bucket.error,
                    latency_sum=sum(bucket.latencies),
                    latency_count=len(bucket.latencies),
                    p50=_percentile(bucket.latencies, 50),
                    p95=_percentile(bucket.latencies, 95),
                    p99=_percentile(bucket.latencies, 99),
                    prompt_tokens=bucket.prompt_tokens,
                    completion_tokens=bucket.completion_tokens,
                    status_counts=dict(bucket.status_counts),
                    model_counts=dict(bucket.model_counts),
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("failed to persist bucket", exc_info=True)

    def _evict_old_buckets_locked(self, current_minute: int) -> None:
        cutoff = current_minute - _BUCKETS * _BUCKET_SECONDS
        stale = [ts for ts in self._buckets if ts < cutoff]
        for ts in stale:
            del self._buckets[ts]

    def flush_pending(self) -> None:
        """Persist the current minute bucket (useful on shutdown / periodic sync)."""
        if self._storage is None:
            return
        with self._lock:
            snapshot = list(self._buckets.values())
        for b in snapshot:
            self._persist_bucket(b)

    def cleanup_storage(self) -> int:
        if self._storage is None:
            return 0
        return self._storage.cleanup(retain_days=self._retain_days)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        """Return top-of-dashboard summary stats."""
        with self._lock:
            total = self._total_requests
            success = self._total_success
            error = self._total_error
            active_models = len(self._model_totals)
            all_latencies: list[float] = []
            for b in self._buckets.values():
                all_latencies.extend(b.latencies)
            avg_latency = (
                sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
            )
            uptime = time.time() - self._start_time

            # 24h sparkline — one datapoint per minute (requests count)
            sparkline = self._timeseries_locked(lambda b: b.total, _BUCKETS)

            # Recent-window QPS: average over the last few minutes, compensating
            # for the current minute being partially elapsed.
            qps_recent = self._recent_qps_locked()
            # Recent-window P95: pool latencies across the last 5 minutes so the
            # gauge isn't noisy when the current minute has only 1-2 requests.
            p95_recent = self._recent_percentile_locked(95, _GAUGE_WINDOW_MINUTES)
            # Tokens/h from a rolling 60-minute window, not lifetime.
            tokens_per_hour = self._recent_tokens_per_hour_locked()

        return {
            "total_requests": total,
            "success": success,
            "error": error,
            "success_rate": (success / total * 100) if total else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "active_models": active_models,
            "uptime_seconds": int(uptime),
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "sparkline": sparkline,
            "qps": round(qps_recent, 3),
            "p95_ms": round(p95_recent, 2),
            "tokens_per_hour": int(tokens_per_hour),
        }

    def _recent_qps_locked(self) -> float:
        """
        Return QPS averaged over the last completed minute if one exists,
        else the in-progress minute's rate scaled by elapsed seconds.

        Dividing the current (still-open) minute's count by 60 always
        under-reports until the minute completes, so we scale by the
        elapsed portion instead.
        """
        now = time.time()
        current_minute = int(now // _BUCKET_SECONDS) * _BUCKET_SECONDS

        last_full = self._buckets.get(current_minute - _BUCKET_SECONDS)
        if last_full is not None and last_full.total > 0:
            return last_full.total / _BUCKET_SECONDS

        current = self._buckets.get(current_minute)
        if current is not None and current.total > 0:
            elapsed = max(1.0, now - current_minute)
            return current.total / elapsed
        return 0.0

    def _recent_percentile_locked(self, p: float, minutes: int) -> float:
        """Pooled percentile across the last *minutes* buckets (inclusive)."""
        now_minute = int(time.time() // _BUCKET_SECONDS) * _BUCKET_SECONDS
        samples: list[float] = []
        for i in range(minutes):
            b = self._buckets.get(now_minute - i * _BUCKET_SECONDS)
            if b is not None and b.latencies:
                samples.extend(b.latencies)
        return _percentile(samples, p)

    def _recent_tokens_per_hour_locked(self) -> float:
        """Sum tokens across the last 60 minute-buckets."""
        now_minute = int(time.time() // _BUCKET_SECONDS) * _BUCKET_SECONDS
        total = 0
        for i in range(_TOKEN_RATE_WINDOW_MINUTES):
            b = self._buckets.get(now_minute - i * _BUCKET_SECONDS)
            if b is not None:
                total += b.prompt_tokens + b.completion_tokens
        return float(total)

    def timeseries(self, minutes: int = 60) -> dict[str, Any]:
        """
        Return per-minute QPS / success / error / latency percentiles
        for the last *minutes* minutes.
        """
        minutes = max(1, min(minutes, _BUCKETS))
        with self._lock:
            labels: list[int] = []
            qps: list[float] = []
            success: list[int] = []
            errors: list[int] = []
            p50: list[float] = []
            p95: list[float] = []
            p99: list[float] = []

            now = time.time()
            now_minute = int(now // _BUCKET_SECONDS) * _BUCKET_SECONDS
            for i in range(minutes - 1, -1, -1):
                ts = now_minute - i * _BUCKET_SECONDS
                b = self._buckets.get(ts)
                labels.append(ts)
                if b is None:
                    qps.append(0.0)
                    success.append(0)
                    errors.append(0)
                    p50.append(0.0)
                    p95.append(0.0)
                    p99.append(0.0)
                else:
                    # The in-progress minute's average QPS uses elapsed time
                    # so the latest point doesn't look artificially low.
                    if ts == now_minute:
                        elapsed = max(1.0, now - ts)
                        qps.append(round(b.total / elapsed, 3))
                    else:
                        qps.append(round(b.total / _BUCKET_SECONDS, 3))
                    success.append(b.success)
                    errors.append(b.error)
                    p50.append(round(_percentile(b.latencies, 50), 2))
                    p95.append(round(_percentile(b.latencies, 95), 2))
                    p99.append(round(_percentile(b.latencies, 99), 2))

        return {
            "labels": labels,
            "qps": qps,
            "success": success,
            "errors": errors,
            "p50": p50,
            "p95": p95,
            "p99": p99,
        }

    def _timeseries_locked(self, fn, minutes: int) -> list[float]:
        now_minute = int(time.time() // _BUCKET_SECONDS) * _BUCKET_SECONDS
        out: list[float] = []
        for i in range(minutes - 1, -1, -1):
            ts = now_minute - i * _BUCKET_SECONDS
            b = self._buckets.get(ts)
            out.append(float(fn(b)) if b else 0.0)
        return out

    def model_stats(self) -> dict[str, Any]:
        """Return per-model usage (requests + tokens)."""
        with self._lock:
            models = [
                {
                    "model": m,
                    "requests": self._model_totals.get(m, 0),
                    "tokens": self._model_tokens.get(m, 0),
                }
                for m in self._model_totals
            ]
        models.sort(key=lambda x: x["requests"], reverse=True)
        return {"models": models}

    def recent_requests(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return most-recent request summaries (newest first)."""
        with self._lock:
            items = list(self._recent)
        items.reverse()
        return [r.to_dict() for r in items[:limit]]

    def recent_errors(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._errors)
        items.reverse()
        return [r.to_dict() for r in items[:limit]]

    def error_breakdown(self) -> dict[str, Any]:
        """Return error-status counts over the retained window."""
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        with self._lock:
            for b in self._buckets.values():
                for code, n in b.status_counts.items():
                    if code >= 400:
                        status_counts[str(code)] = (
                            status_counts.get(str(code), 0) + n
                        )
            for r in self._errors:
                key = r.error_type or _classify_error(r.status, None)
                type_counts[key] = type_counts.get(key, 0) + 1
        return {"by_status": status_counts, "by_type": type_counts}

    def system_status(
        self,
        *,
        version: str,
        auth_mode: str,
        region: str,
        model_count: int,
    ) -> dict[str, Any]:
        """Return static+runtime info for the system-status panel."""
        rss_mb: float | None = None
        try:
            # Best-effort RSS memory read — works on Linux without psutil.
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            rss_mb = round(int(parts[1]) / 1024.0, 1)
                        break
        except OSError:
            rss_mb = None

        with self._lock:
            uptime = time.time() - self._start_time

        return {
            "version": version,
            "auth_mode": auth_mode,
            "region": region,
            "model_count": model_count,
            "uptime_seconds": int(uptime),
            "memory_rss_mb": rss_mb,
            "pid": os.getpid(),
        }
