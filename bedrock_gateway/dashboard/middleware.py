"""
Starlette HTTP middleware that times every request and records it into the
shared :class:`MetricsCollector`.

For model-invocation endpoints (``/v1/chat/completions``, ``/v1/messages``)
the middleware also parses the request body (to capture ``model``) and the
response body / SSE stream (to capture ``usage`` tokens) so the dashboard's
request log shows real values rather than ``-`` / ``0``.

Handlers may still override any of the extracted values by writing to
``request.state.metrics_info`` (``model`` / ``prompt_tokens`` /
``completion_tokens`` / ``error_type`` / ``error_message``).

Paths under ``/dashboard``, ``/api/metrics``, ``/health``, ``/``,
``/v1/models`` (GET listing), and the Anthropic SDK's ``count_tokens``
pre-flight are excluded so the dashboard doesn't pollute its own metrics.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import Request
from starlette.responses import Response, StreamingResponse

from .metrics import MetricsCollector


_EXCLUDED_PREFIXES = ("/dashboard", "/api/metrics")
_EXCLUDED_EXACT = {
    "/",
    "/health",
    "/favicon.ico",
    "/v1/models",
    "/v1/messages/count_tokens",
}

# Paths that carry an LLM request body (JSON with ``model``) and whose
# responses carry ``usage`` tokens worth extracting.
_LLM_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})

# Headers that describe the upstream body we buffer/replace; re-computed
# from the response we return.
_STRIP_HEADERS = {"content-length", "content-encoding"}


def _parse_sse_line(line: str) -> tuple[int, int]:
    """Extract ``(input_tokens, output_tokens)`` from a single SSE ``data:`` line."""
    if not line.startswith("data: "):
        return 0, 0
    payload = line[6:].strip()
    if not payload or payload == "[DONE]":
        return 0, 0
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return 0, 0
    if not isinstance(data, dict):
        return 0, 0

    in_t = 0
    out_t = 0

    # OpenAI chunk (stream_options={include_usage: true}) — top-level usage
    usage = data.get("usage")
    if isinstance(usage, dict):
        in_t = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        out_t = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    etype = data.get("type")

    # Anthropic ``message_start`` — ``message.usage.input_tokens``
    if etype == "message_start":
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            m_usage = msg.get("usage") or {}
            if isinstance(m_usage, dict):
                in_t = m_usage.get("input_tokens", in_t) or in_t

    # Anthropic ``message_delta`` — ``usage.output_tokens``
    if etype == "message_delta":
        m_usage = data.get("usage") or {}
        if isinstance(m_usage, dict):
            out_t = m_usage.get("output_tokens", out_t) or out_t

    return int(in_t or 0), int(out_t or 0)


def _scan_chunk_usage(chunk: bytes | str) -> tuple[int, int]:
    """Scan a chunk of SSE bytes/str for any usage info it carries."""
    if isinstance(chunk, (bytes, bytearray)):
        text = bytes(chunk).decode("utf-8", errors="ignore")
    else:
        text = chunk
    # Cheap short-circuit: skip JSON parsing unless a usage-bearing marker
    # appears in this chunk.
    if (
        "usage" not in text
        and "message_start" not in text
        and "message_delta" not in text
    ):
        return 0, 0
    best_in = 0
    best_out = 0
    for line in text.splitlines():
        i, o = _parse_sse_line(line)
        if i:
            best_in = i
        if o:
            best_out = o
    return best_in, best_out


def _parse_json_usage(body: bytes) -> tuple[int, int]:
    """Extract ``(input, output)`` tokens from a buffered JSON response body."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
        return 0, 0
    if not isinstance(data, dict):
        return 0, 0
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    in_t = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out_t = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    try:
        return int(in_t or 0), int(out_t or 0)
    except (TypeError, ValueError):
        return 0, 0


def _extract_client_ip(request: Request) -> str | None:
    """Return the best-guess client IP for *request*.

    Prefers the first hop in ``X-Forwarded-For`` (typical when running
    behind a proxy), falling back to Starlette's ``request.client.host``.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real_ip = request.headers.get("x-real-ip", "")
    if real_ip:
        return real_ip.strip()
    client = request.client
    if client and client.host:
        return client.host
    return None


def metrics_middleware_factory(collector: MetricsCollector, health: Any = None):
    """Return an ASGI-style middleware function bound to *collector*.

    When *health* is a :class:`HealthMonitor`, the middleware also bumps
    the active-connection counter for the duration of each request.
    """

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path in _EXCLUDED_EXACT or any(
            path.startswith(p) for p in _EXCLUDED_PREFIXES
        ):
            return await call_next(request)

        # Active-connection counter (in-flight, visible to the health API).
        if health is not None:
            health.inc_active()

        # Give handlers a place to stash enrichment data.
        request.state.metrics_info = {}

        client_ip = _extract_client_ip(request)

        # Extract the model from the request body for LLM paths. Starlette's
        # BaseHTTPMiddleware wraps the request in a ``_CachedRequest`` so
        # reading the body here does not prevent the downstream handler
        # from reading it again.
        model_from_body: str | None = None
        if request.method == "POST" and path in _LLM_PATHS:
            try:
                body_bytes = await request.body()
            except Exception:
                body_bytes = b""
            if body_bytes:
                try:
                    data = json.loads(body_bytes)
                    if isinstance(data, dict):
                        m = data.get("model")
                        if isinstance(m, str) and m:
                            model_from_body = m
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass

        start = time.perf_counter()

        active_released = False

        def _release_active() -> None:
            nonlocal active_released
            if not active_released and health is not None:
                health.dec_active()
                active_released = True

        def _record(
            status: int,
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            error_type: str | None = None,
            error_message: str | None = None,
            ttft_ms: float | None = None,
            is_timeout: bool = False,
        ) -> None:
            _release_active()
            latency_ms = (time.perf_counter() - start) * 1000
            info = getattr(request.state, "metrics_info", {}) or {}
            model = info.get("model") or model_from_body or "-"
            pt = int(info.get("prompt_tokens") or prompt_tokens or 0)
            ct = int(info.get("completion_tokens") or completion_tokens or 0)
            etype = error_type or info.get("error_type")
            emsg = error_message or info.get("error_message")
            retry_count = int(info.get("retry_count") or 0)
            ttft = info.get("ttft_ms", ttft_ms)
            # ``ttft_ms`` from metrics_info wins when set; keep our measured
            # value as a fallback.
            if ttft is None:
                ttft = ttft_ms
            tps: float | None = None
            if ct > 0 and latency_ms > 0:
                tps = ct / (latency_ms / 1000.0)
            collector.record_request(
                method=request.method,
                path=path,
                model=model,
                status=status,
                latency_ms=latency_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                error_type=etype,
                error_message=emsg,
                ttft_ms=ttft,
                tokens_per_sec=tps,
                retry_count=retry_count,
                client_ip=client_ip,
                is_timeout=is_timeout or bool(info.get("timeout")),
            )

        try:
            response = await call_next(request)
        except Exception as exc:
            _record(
                status=500,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
            raise

        status = response.status_code

        # Non-LLM path: no body introspection needed; record and return.
        if path not in _LLM_PATHS:
            _record(status)
            return response

        content_type = response.headers.get("content-type", "")
        is_stream = "text/event-stream" in content_type

        if is_stream:
            original_iter = response.body_iterator

            async def wrapped() -> AsyncIterator[bytes]:
                in_t = 0
                out_t = 0
                first_chunk_ms: float | None = None
                try:
                    async for chunk in original_iter:
                        # Record TTFT on the first non-empty chunk that
                        # carries content bytes.
                        if first_chunk_ms is None and chunk:
                            first_chunk_ms = (
                                time.perf_counter() - start
                            ) * 1000
                        ci, co = _scan_chunk_usage(chunk)
                        if ci:
                            in_t = ci
                        if co:
                            out_t = co
                        yield chunk
                finally:
                    _record(status, in_t, out_t, ttft_ms=first_chunk_ms)

            headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in _STRIP_HEADERS
            }
            return StreamingResponse(
                wrapped(),
                status_code=status,
                headers=headers,
                media_type=response.media_type,
            )

        # Non-streaming LLM response: buffer, parse usage, re-emit.
        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            chunks.append(chunk)
        body = b"".join(chunks)

        in_t, out_t = _parse_json_usage(body)
        _record(status, in_t, out_t)

        headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in _STRIP_HEADERS
        }
        return Response(
            content=body,
            status_code=status,
            headers=headers,
            media_type=response.media_type,
        )

    return middleware
