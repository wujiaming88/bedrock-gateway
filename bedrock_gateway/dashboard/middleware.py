"""
Starlette HTTP middleware that times every request and records it into the
shared :class:`MetricsCollector`.

The middleware is intentionally lightweight — it does NOT parse request
bodies. Per-request model / token information is supplied by the handlers
themselves via ``request.state.metrics_info`` (a mutable dict), which the
middleware reads after the handler returns.

Handlers may set any of the following keys:
  * ``model``             — resolved model alias (str)
  * ``prompt_tokens``     — int
  * ``completion_tokens`` — int
  * ``error_type``        — str, for non-HTTP errors
  * ``error_message``     — str, for non-HTTP errors

Paths under ``/dashboard``, ``/api/metrics``, ``/health`` and ``/`` are
excluded so the dashboard doesn't pollute its own metrics.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.responses import Response

from .metrics import MetricsCollector


_EXCLUDED_PREFIXES = ("/dashboard", "/api/metrics")
_EXCLUDED_EXACT = {"/health", "/", "/favicon.ico"}


def metrics_middleware_factory(collector: MetricsCollector):
    """Return an ASGI-style middleware function bound to *collector*."""

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path in _EXCLUDED_EXACT or any(
            path.startswith(p) for p in _EXCLUDED_PREFIXES
        ):
            return await call_next(request)

        # Give handlers a place to stash enrichment data.
        request.state.metrics_info = {}

        start = time.perf_counter()
        error_type: str | None = None
        error_message: str | None = None
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:  # pragma: no cover — defensive
            status = 500
            error_type = type(exc).__name__
            error_message = str(exc)[:500]
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            info = getattr(request.state, "metrics_info", {}) or {}
            model = info.get("model", "-")
            prompt_tokens = int(info.get("prompt_tokens", 0) or 0)
            completion_tokens = int(info.get("completion_tokens", 0) or 0)
            etype = error_type or info.get("error_type")
            emsg = error_message or info.get("error_message")
            collector.record_request(
                method=request.method,
                path=path,
                model=model,
                status=status,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error_type=etype,
                error_message=emsg,
            )
        return response

    return middleware
