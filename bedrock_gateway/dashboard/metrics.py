"""
Thread-safe in-memory metrics collector for the Bedrock Gateway dashboard.

Design notes:
  * Retains the last 24h of per-minute buckets for QPS / error-rate / latency.
  * Retains a bounded ring buffer of the most recent request entries for the
    "Request Log" panel.
  * All mutating operations take an internal ``threading.Lock`` so the
    collector is safe to call from the async request path and from other
    threads (e.g. Starlette's threadpool for sync handlers).
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# Per-minute buckets kept for 24h
_BUCKETS = 24 * 60
_BUCKET_SECONDS = 60

# Max recent requests retained for the log table
_MAX_RECENT_REQUESTS = 200

# Max recent errors retained for the error panel
_MAX_RECENT_ERRORS = 50


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

    def __init__(self, *, max_request_log: int = _MAX_RECENT_REQUESTS) -> None:
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

        rec = RequestRecord(
            ts=now,
            method=method,
            path=path,
            model=model or "-",
            status=status,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_type=error_type,
            error_message=error_message,
        )

        with self._lock:
            bucket = self._buckets.get(minute)
            if bucket is None:
                bucket = _Bucket(ts=minute)
                self._buckets[minute] = bucket
                self._evict_old_buckets_locked(minute)

            bucket.total += 1
            bucket.latencies.append(latency_ms)
            bucket.status_counts[status] = bucket.status_counts.get(status, 0) + 1
            if model:
                bucket.model_counts[model] = bucket.model_counts.get(model, 0) + 1
            bucket.prompt_tokens += prompt_tokens
            bucket.completion_tokens += completion_tokens

            is_err = status >= 400
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

    def _evict_old_buckets_locked(self, current_minute: int) -> None:
        cutoff = current_minute - _BUCKETS * _BUCKET_SECONDS
        stale = [ts for ts in self._buckets if ts < cutoff]
        for ts in stale:
            del self._buckets[ts]

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
        }

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

            now_minute = int(time.time() // _BUCKET_SECONDS) * _BUCKET_SECONDS
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
                    qps.append(round(b.total / 60.0, 3))
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
                key = r.error_type or "unknown"
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
