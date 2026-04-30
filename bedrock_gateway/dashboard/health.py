"""
Gateway self-health monitor.

Collects process/runtime indicators that answer "is the gateway itself
healthy right now?" — distinct from the request-metrics surface
(:class:`MetricsCollector`), which answers "what traffic did we serve?".

Indicators:
  * active_connections — in-flight HTTP requests into the gateway
  * upstream_pool      — in-flight requests from the gateway to Bedrock
  * open_fds           — process file-descriptor count vs the soft ulimit
  * auth               — auth-mode status + optional expiry (iam_role/profile)
  * consecutive_errors — pulled from :class:`MetricsCollector`
  * event_loop_lag_ms  — asyncio scheduling lag (staleness of the loop)
  * upstream           — periodic reachability probe against bedrock-runtime

The background tasks (event-loop-lag and upstream-probe) are started via
:meth:`HealthMonitor.start` during the app's startup event and cancelled
via :meth:`HealthMonitor.stop` on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("bedrock_gateway.dashboard.health")


_EVENT_LOOP_SAMPLE_INTERVAL_S = 1.0
_EVENT_LOOP_SLEEP_S = 0.1
_UPSTREAM_PROBE_INTERVAL_S = 30.0
_UPSTREAM_PROBE_TIMEOUT_S = 5.0
_AUTH_EXPIRING_SOON_S = 15 * 60


@dataclass
class _UpstreamState:
    reachable: bool | None = None
    latency_ms: float | None = None
    last_check: float | None = None
    last_success: float | None = None


class HealthMonitor:
    """
    Tracks gateway self-health indicators.

    Concurrency:
      * Atomic counters are guarded by a ``threading.Lock`` so the
        monitor is safe to call from the async request path or from a
        threadpool-dispatched handler.
      * Async probe/lag tasks are managed cooperatively via
        :meth:`start` / :meth:`stop`.
    """

    def __init__(
        self,
        *,
        region: str,
        auth_mode: str = "-",
        auth_provider: Any | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._active_connections = 0
        self._upstream_active = 0
        self._upstream_total = 0  # cumulative, just for "total" display

        self._event_loop_lag_ms: float = 0.0

        self._upstream = _UpstreamState()
        self._upstream_url = (
            f"https://bedrock-runtime.{region}.amazonaws.com/"
        )
        self._region = region
        self._auth_mode = auth_mode
        self._auth_provider = auth_provider

        self._tasks: list[asyncio.Task[Any]] = []
        self._stopped = asyncio.Event()

    # ------------------------------------------------------------------
    # Counters (sync — called from middleware/handlers)
    # ------------------------------------------------------------------

    def inc_active(self) -> None:
        with self._lock:
            self._active_connections += 1

    def dec_active(self) -> None:
        with self._lock:
            if self._active_connections > 0:
                self._active_connections -= 1

    def inc_upstream(self) -> None:
        with self._lock:
            self._upstream_active += 1
            self._upstream_total += 1

    def dec_upstream(self) -> None:
        with self._lock:
            if self._upstream_active > 0:
                self._upstream_active -= 1

    @asynccontextmanager
    async def track_upstream(self):
        """Context manager: increments upstream counter on entry,
        decrements on exit (even if the wrapped block raises)."""
        self.inc_upstream()
        try:
            yield
        finally:
            self.dec_upstream()

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the event-loop-lag + upstream-probe background tasks.

        Safe to call multiple times; no-op if already running.
        """
        if self._tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. called from sync context) — defer.
            logger.debug("HealthMonitor.start called with no running loop")
            return
        self._stopped = asyncio.Event()
        self._tasks.append(loop.create_task(self._event_loop_lag_task()))
        self._tasks.append(loop.create_task(self._upstream_probe_task()))

    async def stop(self) -> None:
        self._stopped.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []

    async def _event_loop_lag_task(self) -> None:
        """Sample scheduling lag of the asyncio event loop once a second."""
        expected_ms = _EVENT_LOOP_SLEEP_S * 1000.0
        while not self._stopped.is_set():
            t0 = time.perf_counter()
            try:
                await asyncio.sleep(_EVENT_LOOP_SLEEP_S)
            except asyncio.CancelledError:
                return
            actual_ms = (time.perf_counter() - t0) * 1000.0
            lag = max(0.0, actual_ms - expected_ms)
            with self._lock:
                # Exponential-ish smoothing so a single GC pause doesn't
                # make the gauge jump to "red" for a minute.
                if self._event_loop_lag_ms <= 0:
                    self._event_loop_lag_ms = lag
                else:
                    self._event_loop_lag_ms = (
                        self._event_loop_lag_ms * 0.7 + lag * 0.3
                    )
            # Pace to ~1Hz independent of how long sleep actually took.
            remaining = _EVENT_LOOP_SAMPLE_INTERVAL_S - _EVENT_LOOP_SLEEP_S
            if remaining > 0:
                try:
                    await asyncio.sleep(remaining)
                except asyncio.CancelledError:
                    return

    async def _upstream_probe_task(self) -> None:
        """Probe the Bedrock runtime endpoint every 30s."""
        while not self._stopped.is_set():
            await self._probe_once()
            try:
                await asyncio.sleep(_UPSTREAM_PROBE_INTERVAL_S)
            except asyncio.CancelledError:
                return

    async def _probe_once(self) -> None:
        now = time.time()
        t0 = time.perf_counter()
        reachable = False
        try:
            async with httpx.AsyncClient(timeout=_UPSTREAM_PROBE_TIMEOUT_S) as c:
                # A GET without auth will get a 403/400 — that's fine; we
                # only care that TCP+TLS reached the upstream.
                resp = await c.get(self._upstream_url)
                reachable = resp.status_code < 600
        except Exception:  # noqa: BLE001 — probe never raises
            reachable = False
        latency_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._upstream.reachable = reachable
            self._upstream.latency_ms = latency_ms
            self._upstream.last_check = now
            if reachable:
                self._upstream.last_success = now

    # ------------------------------------------------------------------
    # Snapshot / introspection
    # ------------------------------------------------------------------

    def snapshot(self, metrics: Any | None = None) -> dict[str, Any]:
        """
        Return a dashboard-friendly dict of all health indicators.

        *metrics* is an optional :class:`MetricsCollector`; when supplied,
        ``consecutive_errors`` is pulled from it.
        """
        fd_info = _read_fd_info()
        auth = self._auth_snapshot()
        with self._lock:
            active = self._active_connections
            up_active = self._upstream_active
            up_total = self._upstream_total
            lag_ms = self._event_loop_lag_ms
            up = _UpstreamState(
                reachable=self._upstream.reachable,
                latency_ms=self._upstream.latency_ms,
                last_check=self._upstream.last_check,
                last_success=self._upstream.last_success,
            )

        consecutive_errors = 0
        if metrics is not None and hasattr(metrics, "consecutive_errors"):
            try:
                consecutive_errors = int(metrics.consecutive_errors())
            except Exception:  # noqa: BLE001 — never fail a snapshot on metrics
                consecutive_errors = 0

        return {
            "active_connections": active,
            "upstream_pool": {
                "active": up_active,
                "total": up_total,
                # "idle" is conceptual here (we don't reuse connections);
                # surface it as 0 so the UI can still show the three-column
                # layout documented by the design.
                "idle": 0,
            },
            "open_fds": fd_info,
            "auth": auth,
            "consecutive_errors": consecutive_errors,
            "event_loop_lag_ms": round(lag_ms, 2),
            "upstream": {
                "reachable": up.reachable,
                "latency_ms": (
                    round(up.latency_ms, 2) if up.latency_ms is not None else None
                ),
                "last_check": _iso(up.last_check),
                "last_success": _iso(up.last_success),
            },
        }

    # ------------------------------------------------------------------
    # Auth introspection
    # ------------------------------------------------------------------

    def _auth_snapshot(self) -> dict[str, Any]:
        """Return {mode, status, expires_at} for the current auth config."""
        mode = self._auth_mode
        # bearer_token / credentials: no expiry surfaced by Bedrock. We
        # report "valid" but leave expires_at null.
        if mode in ("bearer_token", "credentials"):
            return {
                "mode": mode,
                "status": "valid",
                "expires_at": None,
            }

        if mode in ("iam_role", "profile"):
            expires_at, status = self._boto_auth_expiry()
            return {
                "mode": mode,
                "status": status,
                "expires_at": expires_at,
            }

        return {"mode": mode or "-", "status": "unknown", "expires_at": None}

    def _boto_auth_expiry(self) -> tuple[str | None, str]:
        """Best-effort peek at boto3 session credential expiry.

        Returns ``(iso8601_or_none, status)``.
        """
        provider = self._auth_provider
        if provider is None:
            return None, "unknown"
        try:
            client = getattr(provider, "_boto3_client", None)
            if client is None:
                # Not yet materialised — boto auth is lazy. Treat as
                # unknown; once the first request triggers it we'll pick
                # up the expiry on subsequent snapshots.
                return None, "unknown"
            signer = client._request_signer  # type: ignore[attr-defined]
            creds = signer._credentials  # type: ignore[attr-defined]
            # Refreshable credentials expose ``_expiry_time``; static
            # credentials don't. We try frozen credentials first because
            # that triggers a refresh if one is pending.
            try:
                creds.get_frozen_credentials()
            except Exception:  # noqa: BLE001
                pass
            expiry = getattr(creds, "_expiry_time", None)
            if expiry is None:
                # No expiry info — treat as valid/no-expiry.
                return None, "valid"
            # expiry is datetime (aware). Normalise to UTC seconds.
            try:
                expires_ts = expiry.timestamp()
            except Exception:  # noqa: BLE001
                return None, "valid"
            now = time.time()
            if expires_ts <= now:
                return _iso(expires_ts), "expired"
            if expires_ts - now <= _AUTH_EXPIRING_SOON_S:
                return _iso(expires_ts), "expiring_soon"
            return _iso(expires_ts), "valid"
        except Exception:  # noqa: BLE001
            return None, "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_fd_info() -> dict[str, int | None]:
    """Return ``{current, limit}`` for the process file descriptors.

    Falls back to ``None`` on platforms where /proc/self/fd isn't
    available (mostly macOS; the gateway targets Linux).
    """
    current: int | None = None
    try:
        current = len(os.listdir("/proc/self/fd"))
    except OSError:
        current = None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit: int | None = int(soft) if soft > 0 else None
    except (OSError, ValueError):
        limit = None
    return {"current": current, "limit": limit}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
        )
    except (TypeError, ValueError, OverflowError):
        return None
