"""
Security utilities for the dashboard:

  * ``DashboardAuth``        — API-key / localhost-only gate used by the
                                metrics API and the dashboard UI.
  * ``RateLimiter``          — simple in-memory per-IP fixed-window limiter.
  * ``SECURITY_HEADERS``     — response headers attached to dashboard responses.
  * ``sanitize_request_log`` — strip sensitive fields from request records.
  * ``mask_api_key``         — return ``"xxxx***"`` from an API key.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

from fastapi import Request


# Response headers applied to every dashboard / metrics-API response.
# The dashboard UI loads Chart.js from jsdelivr, so we restrict script
# sources to that CDN plus the page itself.
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self' data:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    ),
}


_LOCALHOST_HOSTS = {"127.0.0.1", "localhost", "::1"}

_COOKIE_NAME = "bedrock_gw_key"


# ---------------------------------------------------------------------------
# API-key extraction + auth gate
# ---------------------------------------------------------------------------


def _extract_key(request: Request) -> str:
    """Pull the candidate API key from cookie/header/query, in that order."""
    cookie_key = request.cookies.get(_COOKIE_NAME)
    if cookie_key:
        return cookie_key

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key

    query_key = request.query_params.get("key")
    if query_key:
        return query_key

    return ""


def _client_host(request: Request) -> str:
    client = request.client
    if client is None:
        return ""
    return client.host or ""


class DashboardAuth:
    """
    Gatekeeper for dashboard URLs and metrics APIs.

    Behaviour (evaluated in order):

      1. If ``enabled=False`` → always deny (dashboard disabled).
      2. If ``api_key`` is set **and** ``require_auth`` is true →
         require a matching key in a cookie, ``Authorization: Bearer``,
         ``x-api-key`` header, or ``?key=`` query param.
      3. If ``api_key`` is empty and ``localhost_only`` is true →
         allow only ``127.0.0.1`` / ``::1`` clients.
      4. Otherwise → allow.

    ``localhost_only`` defaults to True when no API key is configured,
    so a dashboard exposed without a key on ``0.0.0.0`` is still safe.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        api_key: str = "",
        require_auth: bool = True,
        localhost_only: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.api_key = api_key or ""
        self.require_auth = require_auth
        # When no API key is configured, default to localhost-only unless
        # the operator has explicitly opted out via config.
        if localhost_only is None:
            localhost_only = not bool(self.api_key)
        self.localhost_only = bool(localhost_only)

    def is_configured_key(self) -> bool:
        return bool(self.api_key)

    def verify_key(self, candidate: str) -> bool:
        """Constant-time comparison of *candidate* against the configured key."""
        if not self.api_key or not candidate:
            return False
        return hmac.compare_digest(candidate, self.api_key)

    def check(self, request: Request) -> tuple[bool, str]:
        """
        Return ``(allowed, reason)``. When ``allowed`` is False, *reason* is
        one of ``"disabled" | "localhost_only" | "auth_required"``.
        """
        if not self.enabled:
            return False, "disabled"

        if self.api_key and self.require_auth:
            candidate = _extract_key(request)
            if not self.verify_key(candidate):
                return False, "auth_required"
            return True, ""

        if self.localhost_only:
            host = _client_host(request)
            if host not in _LOCALHOST_HOSTS:
                return False, "localhost_only"

        return True, ""


# ---------------------------------------------------------------------------
# Rate limiting (fixed window, per IP)
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Fixed-window per-IP rate limiter. Safe to call from multiple threads.

    Not designed for high-scale deployments — this is just a guard against
    abusive polling of ``/api/metrics/*`` by a single client.
    """

    def __init__(self, limit: int = 60, window_seconds: int = 60) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """
        Record a hit for *key* and return ``(allowed, retry_after_seconds)``.
        ``retry_after_seconds`` is only meaningful when ``allowed`` is False.
        """
        if not key:
            key = "-"
        now = time.time()
        window_start = now - self.window_seconds
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
            # Drop entries outside the current window.
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.limit:
                # Earliest remaining hit + window_seconds → next free slot.
                retry_after = max(1, int(bucket[0] + self.window_seconds - now))
                return False, retry_after
            bucket.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# ---------------------------------------------------------------------------
# Request-log sanitisation
# ---------------------------------------------------------------------------


# Known sensitive keys that must never reach the dashboard. The recorded
# request log already avoids storing these, but we defend in depth in case
# a handler stashes enrichment data on ``request.state.metrics_info``.
_SENSITIVE_KEYS = {
    "messages",
    "body",
    "prompt",
    "request_body",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "access_key_id",
    "secret_access_key",
    "session_token",
    "authorization",
    "bearer_token",
    "api_key",
    "x-api-key",
}


def mask_api_key(value: str | None) -> str:
    """Return a masked representation of *value*, e.g. ``sk-a***``."""
    if not value:
        return ""
    v = str(value)
    if len(v) <= 4:
        return "***"
    return v[:4] + "***"


def mask_ip(ip: str | None) -> str:
    """Return a coarse representation of *ip* (first three IPv4 octets)."""
    if not ip:
        return ""
    if "." in ip:
        parts = ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3]) + ".0"
    if ":" in ip:
        return "::"
    return ip


def sanitize_request_log(
    records: Iterable[dict[str, Any]],
    *,
    show_ip: bool = False,
) -> list[dict[str, Any]]:
    """
    Return a copy of *records* with sensitive keys dropped or masked.

    Keys in :data:`_SENSITIVE_KEYS` are dropped outright; ``api_key`` (if
    present) is masked via :func:`mask_api_key`; ``ip`` is masked to a /24
    unless *show_ip* is true.
    """
    cleaned: list[dict[str, Any]] = []
    for rec in records:
        out: dict[str, Any] = {}
        for k, v in rec.items():
            if k in _SENSITIVE_KEYS:
                if k == "api_key":
                    out["api_key"] = mask_api_key(v)
                # else: drop
                continue
            if k == "ip":
                out["ip"] = mask_ip(v) if not show_ip else v
                continue
            out[k] = v
        # Error messages can leak things like full URLs with embedded tokens —
        # truncate aggressively as a belt-and-braces measure.
        if isinstance(out.get("error_message"), str):
            out["error_message"] = out["error_message"][:300]
        cleaned.append(out)
    return cleaned
