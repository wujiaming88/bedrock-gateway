"""
FastAPI server for Bedrock Gateway.

Exposes an OpenAI-compatible API and an Anthropic Messages API
that proxy requests to AWS Bedrock:
  - POST /v1/chat/completions  (OpenAI format, sync + streaming)
  - POST /v1/messages          (Anthropic Messages format, sync + streaming)
  - GET  /v1/models
  - GET  /health
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from multipart.multipart import parse_options_header  # type: ignore[import-untyped]
from starlette.datastructures import UploadFile

from . import __version__
from .auth import AuthProvider
from .config import GatewayConfig, ModelEntry, load_config
from .converter import (
    convert_tool_choice,
    convert_tools,
    decode_event_stream_chunk,
    extract_system_and_messages,
    format_anthropic_error,
    format_anthropic_response,
    make_anthropic_sse,
    map_reasoning_effort,
    parse_bedrock_error,
)
from .dashboard import (
    DashboardAuth,
    HealthMonitor,
    MetricsCollector,
    RateLimiter,
    build_dashboard_router,
    metrics_middleware_factory,
)
from .dashboard.storage import MetricsStorage
from .embeddings import (
    EmbeddingRequest,
    EmbeddingsAdapter,
    EmbeddingsError,
    EmbeddingsValidationError,
    UnsupportedEmbeddingsModelError,
    apply_default_dimensions,
    parse_request,
    render_openai,
    resolve_profile,
)
from .messages_to_responses import (
    AnthropicStreamAdapter,
    responses_error_to_anthropic,
    to_anthropic_response,
    to_responses_request,
)
from .logging_config import configure_logging
from .models import ModelRegistry, UnknownModelError
from .responses_compatibility import (
    CompatibilityPolicy,
    is_exact_variant_rejection,
    project_mantle_input,
    responses_compat_policy,
)
from .providers import (
    Dialect,
    Transport,
    get_dialect,
    get_transport,
)

logger = logging.getLogger("bedrock_gateway")

# Fallback ModelEntry for aliases that resolve to a raw Bedrock ID with no
# registered entry (e.g. a pass-through vendor ID). Defaults put it on the
# original bedrock-runtime / Anthropic-Messages path.
def _fallback_entry(bedrock_id: str) -> ModelEntry:
    return ModelEntry(bedrock_id=bedrock_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oai_error(
    status: int,
    message: str,
    etype: str = "api_error",
    *,
    headers: dict[str, str] | None = None,
    param: str | None = None,
    code: str | int | None = None,
) -> JSONResponse:
    """Return an OpenAI-style error response."""
    error: dict[str, Any] = {
        "message": message,
        "type": etype,
        "code": status if code is None else code,
    }
    if param is not None:
        error["param"] = param
    return JSONResponse(
        status_code=status,
        content={"error": error},
        headers=headers,
    )


def _multipart_content_type(request: Request) -> tuple[bytes, dict[bytes, bytes], Request]:
    """Parse multipart parameters and normalize only the media type for Starlette."""
    content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type)
    if media_type == b"multipart/form-data" or media_type.lower() != b"multipart/form-data":
        return media_type, options, request

    # Starlette compares the parsed media type case-sensitively. Preserve all
    # parameters while canonicalizing that token, without reading the body.
    _, separator, parameters = content_type.partition(";")
    normalized = "multipart/form-data"
    if separator:
        normalized += separator + parameters
    scope = dict(request.scope)
    headers = [
        (name, normalized.encode("latin-1") if name.lower() == b"content-type" else value)
        for name, value in scope.get("headers", [])
    ]
    scope["headers"] = headers
    normalized_request = Request(scope, receive=request.receive)
    return media_type, options, normalized_request


def _multipart_parse_error(
    request: Request,
    exc: Exception,
    options: dict[bytes, bytes],
) -> JSONResponse:
    """Return a correlated multipart error without logging request values."""
    request_id = str(uuid.uuid4())
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        content_length: int | str = "absent"
    else:
        try:
            content_length = int(raw_length)
            if content_length < 0:
                content_length = "invalid"
        except ValueError:
            content_length = "invalid"
    boundary = options.get(b"boundary")
    logger.warning(
        "MULTIPART_PARSE_FAILED request_id=%s exception=%s content_length=%s "
        "transfer_encoding=%s content_encoding=%s boundary_present=%s boundary_length=%d",
        request_id,
        type(exc).__name__,
        content_length,
        "present" if request.headers.get("transfer-encoding") else "absent",
        "present" if request.headers.get("content-encoding") else "absent",
        boundary is not None,
        len(boundary) if boundary is not None else 0,
    )
    return _oai_error(
        400,
        f"Invalid multipart form data. Request ID: {request_id}",
        "invalid_request_error",
        headers={"X-Request-ID": request_id},
    )


@dataclass(frozen=True, slots=True)
class PreparedRequestBody:
    """Fully encoded, replayable upstream request payload."""

    content: bytes
    content_type: str
    log_body: dict[str, Any] | None = None


def _prepare_request_body(
    body: dict[str, Any] | PreparedRequestBody,
) -> PreparedRequestBody:
    if isinstance(body, PreparedRequestBody):
        return body
    return PreparedRequestBody(
        content=json.dumps(body).encode(),
        content_type="application/json",
        log_body=body,
    )


def _operation_path(
    dialect: Dialect,
    entry: ModelEntry,
    stream: bool,
    operation: str | None,
) -> str:
    if operation is None:
        return dialect.operation_path(entry, stream)
    return dialect.operation_path(entry, stream, operation=operation)


def _payload_headers(
    transport: Transport,
    entry: ModelEntry,
    auth: AuthProvider,
    url: str,
    payload: PreparedRequestBody,
) -> dict[str, str]:
    headers = transport.auth_headers(entry) or auth.get_headers(
        method="POST", url=url, body=payload.content
    )
    headers = dict(headers)
    headers["Content-Type"] = payload.content_type
    return headers


def _transport_payload_headers(
    transport: Transport,
    entry: ModelEntry,
    payload: PreparedRequestBody,
) -> dict[str, str] | None:
    """Return static transport headers; global auth must be signed per attempt."""
    headers = transport.auth_headers(entry)
    if headers is None:
        return None
    headers = dict(headers)
    headers["Content-Type"] = payload.content_type
    return headers


def _note_retry(request: Request | None) -> None:
    """Bump ``request.state.metrics_info['retry_count']`` by 1, if possible."""
    if request is None:
        return
    try:
        info = getattr(request.state, "metrics_info", None)
        if info is None:
            info = {}
            request.state.metrics_info = info
        info["retry_count"] = int(info.get("retry_count") or 0) + 1
    except Exception:  # noqa: BLE001 — never let metrics accounting break a handler
        pass


@asynccontextmanager
async def _track_upstream(health: HealthMonitor | None):
    """Optional upstream-counter bump — no-op when health is None."""
    if health is None:
        yield
        return
    async with health.track_upstream():
        yield


def _note_timeout(request: Request | None) -> None:
    """Flag this request as having timed out for metrics purposes."""
    if request is None:
        return
    try:
        info = getattr(request.state, "metrics_info", None)
        if info is None:
            info = {}
            request.state.metrics_info = info
        info["timeout"] = True
    except Exception:  # noqa: BLE001
        pass


def _log_upstream_error(status_code: int, fmt: str, *args: Any) -> None:
    """Log a non-2xx upstream response at the level matching its severity.

    Severity rules:
      * 401 / 403 → ERROR with an ``[auth-failure]`` tag. These are *not*
        client mistakes — the gateway's own credentials were rejected, so
        on-call should be paged.
      * Other 4xx → WARNING. These are caused by the calling client
        (bad model id, oversized image, malformed body, …); logging them
        at ERROR floods alerting and hides real gateway/upstream faults.
      * 5xx and unknown codes → ERROR.
    """
    if status_code in (401, 403):
        logger.error(fmt + " [auth-failure]", *args)
    elif 400 <= status_code < 500:
        logger.warning(fmt, *args)
    else:
        logger.error(fmt, *args)


def _value_shape(value: Any) -> Any:
    """Return a redacted structural summary of an arbitrary JSON-ish value.

    This is intentionally value-free: it records types/counts/keys only, never
    user text, tool arguments, headers, or credentials. It is safe for logs and
    is useful when an upstream rejects the request body with a schema error.
    """
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(k) for k in value.keys())}
    if isinstance(value, list):
        return {"type": "array", "len": len(value)}
    if isinstance(value, str):
        return {"type": "string", "len": len(value)}
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__}


def _content_shape(content: Any) -> dict[str, Any]:
    """Summarize message/content blocks without exposing text or arguments."""
    if isinstance(content, str):
        return {"type": "string", "len": len(content)}
    if not isinstance(content, list):
        return _value_shape(content)
    blocks: list[dict[str, Any]] = []
    for block in content[:12]:
        if isinstance(block, dict):
            blocks.append({
                "type": block.get("type", "object"),
                "keys": sorted(str(k) for k in block.keys()),
            })
        else:
            blocks.append(_value_shape(block))
    return {"type": "array", "len": len(content), "blocks": blocks}


def _reasoning_summary_shape(value: Any) -> dict[str, Any]:
    """Summarize reasoning summary blocks without exposing their text."""
    if not isinstance(value, list):
        return _value_shape(value)
    block_types = [
        block.get("type", "object") if isinstance(block, dict)
        else _value_shape(block)["type"]
        for block in value
    ]
    valid = all(
        isinstance(block, dict)
        and block.get("type") == "summary_text"
        and isinstance(block.get("text"), str)
        for block in value
    )
    return {"type": "array", "len": len(value), "block_types": block_types, "valid": valid}


def _input_shape(input_value: Any) -> dict[str, Any]:
    """Summarize Responses ``input`` variants for upstream-schema debugging."""
    if isinstance(input_value, str):
        return {"type": "string", "len": len(input_value)}
    if not isinstance(input_value, list):
        return _value_shape(input_value)
    items: list[dict[str, Any]] = []
    reasoning_total = 0
    reasoning_valid = 0
    reasoning_encrypted = 0
    for item in input_value:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            reasoning_total += 1
            if _reasoning_summary_shape(item.get("summary")).get("valid") is True:
                reasoning_valid += 1
            if "encrypted_content" in item:
                reasoning_encrypted += 1
    for item in input_value[:20]:
        if isinstance(item, dict):
            item_type = item.get("type", "message" if "role" in item else "object")
            summary: dict[str, Any] = {
                "type": item_type,
                "role": item.get("role"),
                "keys": sorted(str(k) for k in item.keys()),
            }
            if "content" in item:
                summary["content"] = _content_shape(item.get("content"))
            if item_type == "reasoning":
                summary["summary"] = _reasoning_summary_shape(item.get("summary"))
            items.append(summary)
        else:
            items.append(_value_shape(item))
    return {
        "type": "array",
        "len": len(input_value),
        "items": items,
        "reasoning": {
            "total": reasoning_total,
            "valid_summary": reasoning_valid,
            "invalid_or_missing_summary": reasoning_total - reasoning_valid,
            "encrypted_present": reasoning_encrypted,
        },
    }


def _messages_shape(messages: Any) -> dict[str, Any]:
    """Summarize chat/messages arrays without text."""
    if not isinstance(messages, list):
        return _value_shape(messages)
    out: list[dict[str, Any]] = []
    for msg in messages[:20]:
        if isinstance(msg, dict):
            item = {
                "role": msg.get("role"),
                "keys": sorted(str(k) for k in msg.keys()),
            }
            if "content" in msg:
                item["content"] = _content_shape(msg.get("content"))
            out.append(item)
        else:
            out.append(_value_shape(msg))
    return {"type": "array", "len": len(messages), "messages": out}


def _tools_shape(tools: Any) -> dict[str, Any]:
    """Summarize tool declarations without schemas/arguments."""
    if not isinstance(tools, list):
        return _value_shape(tools)
    out: list[dict[str, Any]] = []
    for tool in tools[:30]:
        if isinstance(tool, dict):
            out.append({
                "type": tool.get("type", "function" if "function" in tool else "object"),
                "name": tool.get("name") or (tool.get("function") or {}).get("name")
                if isinstance(tool.get("function"), dict) else tool.get("name"),
                "keys": sorted(str(k) for k in tool.keys()),
            })
        else:
            out.append(_value_shape(tool))
    return {"type": "array", "len": len(tools), "tools": out}


def _request_shape_summary(body: dict[str, Any]) -> dict[str, Any]:
    """Redacted request-body structure summary for error logs."""
    summary: dict[str, Any] = {
        "top_keys": sorted(str(k) for k in body.keys()),
        "model": body.get("model") if isinstance(body.get("model"), str) else None,
        "stream": body.get("stream") if isinstance(body.get("stream"), bool) else None,
    }
    if "input" in body:
        summary["input"] = _input_shape(body.get("input"))
    if "messages" in body:
        summary["messages"] = _messages_shape(body.get("messages"))
    if "tools" in body:
        summary["tools"] = _tools_shape(body.get("tools"))
    if "reasoning" in body:
        reasoning = body.get("reasoning")
        summary["reasoning"] = (
            {"type": "object", "keys": sorted(str(k) for k in reasoning.keys())}
            if isinstance(reasoning, dict) else _value_shape(reasoning)
        )
    for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
        if key in body:
            summary[key] = _value_shape(body.get(key))
    return summary


def _log_request_shape_for_upstream_error(
    *, status: int, model: str, body: dict[str, Any], err_text: str
) -> None:
    """Log a redacted request structure when upstream rejects the body."""
    try:
        shape = _request_shape_summary(body)
        logger.warning(
            "REQ-SHAPE upstream_error status=%s model=%s err=%s shape=%s",
            status,
            model,
            err_text[:160],
            json.dumps(shape, ensure_ascii=False, sort_keys=True),
        )
    except Exception:  # noqa: BLE001 — diagnostics must never break requests
        logger.exception("REQ-SHAPE failed model=%s", model)


def _compat_projection(
    compat: CompatibilityPolicy | None,
    body: dict[str, Any] | None,
    model: str,
) -> dict[str, Any] | None:
    """Run the one-time safe projection for an exact-variant 400.

    The caller has already validated ``is_exact_variant_rejection``. Returns the
    projected body when it is safe to retry (changed AND no unsafe relation/item),
    else None. Logs a redacted decision record either way — decisions, counts and
    categories only, never request values.
    """
    if compat is None or body is None or not isinstance(body, dict):
        logger.warning("COMPAT model=%s no_projection_body", model)
        return None
    result = project_mantle_input(body)
    logger.info(
        "COMPAT model=%s safe=%s changed=%s decisions=%s unsafe=%s relations=%s",
        model,
        result.safe_to_retry,
        result.changed,
        json.dumps(result.decisions, sort_keys=True),
        json.dumps(list(result.unsafe_reasons), sort_keys=True),
        json.dumps(result.relation_counts, sort_keys=True),
    )
    if result.safe_to_retry and result.changed:
        return result.body
    return None


# Connections should fail fast (DNS/refused/TLS is not something a long read
# timeout should mask); reads must be generous for slow reasoning models. So we
# split the single ``timeout`` value: it governs read/write/pool, while connect
# is capped short. See P2 in the timeout/error-handling audit.
_CONNECT_TIMEOUT = 10.0


def _httpx_timeout(timeout: float) -> httpx.Timeout:
    """Build an httpx.Timeout: fast connect, ``timeout`` for read/write/pool."""
    return httpx.Timeout(timeout, connect=min(_CONNECT_TIMEOUT, timeout))


# Total wall-clock budget for the whole retry sequence (all attempts + backoff
# sleeps), as a multiple of the per-attempt timeout. Prevents a slow upstream
# from stacking N full timeouts into minutes of hang under load. See P3.
_RETRY_BUDGET_FACTOR = 1.5


def _retry_deadline(timeout: float, max_retries: int) -> float:
    """A ``time.monotonic()`` deadline bounding the entire retry sequence."""
    return time.monotonic() + timeout * _RETRY_BUDGET_FACTOR * max(1, max_retries)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: GatewayConfig | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    If *config* is ``None``, configuration is loaded from the default
    locations (``config.yaml`` / env vars).
    """
    if config is None:
        config = load_config()

    app = FastAPI(
        title="MuxLane",
        version=__version__,
        description="The unified lane for multi-cloud AI traffic",
    )

    registry = ModelRegistry(config)
    auth = AuthProvider(config.auth, config.region)
    bedrock_base = f"https://bedrock-runtime.{config.region}.amazonaws.com"
    max_retries = config.retry.max_retries
    retry_base_delay = config.retry.base_delay
    request_timeout = config.retry.timeout

    # Metrics collector (shared across middleware + dashboard router)
    storage: MetricsStorage | None = None
    if config.dashboard.enabled and config.dashboard.storage.enabled:
        try:
            storage = MetricsStorage(config.dashboard.storage.path)
        except Exception:  # noqa: BLE001 — dashboard persistence is optional
            logger.warning(
                "failed to initialise dashboard storage at %s; continuing in-memory",
                config.dashboard.storage.path,
                exc_info=True,
            )
            storage = None
    metrics = MetricsCollector(
        max_request_log=config.dashboard.max_request_log,
        storage=storage,
        retain_days=config.dashboard.storage.retain_days,
    )

    # Self-health monitor (active conns, upstream probe, event loop lag, …).
    # Background tasks are started in the app's ``startup`` handler below.
    health = HealthMonitor(
        region=config.region,
        auth_mode=auth.mode,
        auth_provider=auth,
    )

    # Dashboard auth + rate limiter (public-deployment hardening).
    # dashboard.api_key is deliberately independent of server.api_key:
    # model clients can't reach the dashboard, and dashboard admins can't
    # call the model endpoints.
    dashboard_auth = DashboardAuth(
        enabled=config.dashboard.enabled,
        api_key=config.dashboard.api_key or "",
        require_auth=config.dashboard.require_auth,
        # None → default ("localhost-only when no dashboard.api_key configured");
        # True/False → explicit operator override.
        localhost_only=config.dashboard.localhost_only,
    )
    dashboard_rate_limiter = RateLimiter(
        limit=max(1, config.dashboard.rate_limit), window_seconds=60
    )

    # Store on app.state for testability
    app.state.config = config
    app.state.registry = registry
    app.state.auth = auth
    app.state.metrics = metrics
    app.state.health = health
    app.state.dashboard_auth = dashboard_auth
    app.state.dashboard_rate_limiter = dashboard_rate_limiter

    # Start/stop the event-loop-lag sampler with the app lifecycle.
    # Only runs when the dashboard is enabled — it exists solely to
    # populate the dashboard's loop-lag gauge. Upstream health is now
    # derived passively from request metrics (no probe).
    if config.dashboard.enabled:
        @app.on_event("startup")
        async def _health_startup() -> None:
            health.start()

        @app.on_event("shutdown")
        async def _health_shutdown() -> None:
            await health.stop()
    else:
        logger.info(
            "dashboard disabled — event-loop-lag sampler is not started"
        )

    # ------------------------------------------------------------------
    # API key authentication middleware (opt-in)
    # ------------------------------------------------------------------

    api_key = config.server.api_key

    @app.middleware("http")
    async def api_key_auth(request: Request, call_next):
        # Skip auth when no API key is configured
        if not api_key:
            return await call_next(request)

        # Whitelist: public endpoints (no auth required)
        path = request.url.path
        if path in ("/health", "/") or path.startswith(("/dashboard", "/api/metrics")):
            return await call_next(request)

        # Extract key from Authorization: Bearer <key> or x-api-key header
        key = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            key = auth_header[7:]
        if not key:
            key = request.headers.get("x-api-key")

        # Constant-time comparison to prevent timing attacks
        if not key or not hmac.compare_digest(key, api_key):
            # Return format-appropriate error
            if request.url.path.startswith("/v1/messages"):
                return JSONResponse(
                    status_code=401,
                    content={
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "Invalid API key",
                        },
                    },
                )
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                        "code": 401,
                    }
                },
            )

        return await call_next(request)

    # ------------------------------------------------------------------
    # Metrics middleware (wraps every request for latency + counts)
    # ------------------------------------------------------------------
    app.middleware("http")(metrics_middleware_factory(metrics, health=health))

    # ------------------------------------------------------------------
    # Dashboard UI + metrics JSON API
    # ------------------------------------------------------------------
    if config.dashboard.enabled:
        app.include_router(
            build_dashboard_router(
                metrics,
                auth=dashboard_auth,
                rate_limiter=dashboard_rate_limiter,
            )
        )

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"object": "list", "data": registry.list_models()}

    # ------------------------------------------------------------------
    # GET / (root, for client connectivity checks like Claude Code HEAD /)
    # ------------------------------------------------------------------

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root() -> dict:
        return {"status": "ok"}

    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health_route() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "auth_mode": auth.mode,
            "region": config.region,
            "models": len(registry.list_models()),
        }

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        # Parse body
        try:
            body = await request.json()
        except Exception:
            return _oai_error(400, "Invalid JSON body")

        raw_model = body.get("model", "claude-haiku")

        # Prefix passthrough first: ``azure/<deployment>`` → Azure resource,
        # dialect fixed by this endpoint (chat). Falls through otherwise.
        entry = registry.resolve_prefixed(raw_model, "openai-chat")
        if entry is not None:
            model = entry.deployment
        elif resolved := registry.resolve_for_dialect(raw_model, "openai-chat"):
            model, entry = resolved
        else:
            try:
                model = registry.resolve(raw_model)
            except UnknownModelError as exc:
                return _oai_error(400, str(exc), "invalid_request_error")
            # Unregistered aliases (raw Bedrock IDs passed through) fall back
            # to the default bedrock/anthropic entry.
            entry = registry.get_entry(raw_model) or _fallback_entry(model)

        transport = get_transport(entry)
        dialect = get_dialect(entry)
        stream = body.get("stream", False)

        # This endpoint serves two dialects:
        #   * anthropic   → OpenAI chat converted to Anthropic Messages (below)
        #   * openai-chat → verbatim passthrough (Azure / mantle chat models)
        # The Responses dialect belongs on /openai/v1/responses; reject it here.
        if dialect.name == "openai-chat":
            # Passthrough: swap alias for upstream id (Azure deployment or
            # resolved Bedrock id); forward the body untouched.
            upstream_id = (
                entry.deployment if entry.transport == "azure" else model
            )
            upstream_body = dict(body)
            upstream_body["model"] = upstream_id
            logger.info(
                "REQ [chat-passthrough] model=%s -> %s (%s) stream=%s",
                raw_model, upstream_id, entry.transport, stream,
            )
            if stream:
                return await _handle_stream(
                    transport, dialect, entry, upstream_id, config.region,
                    upstream_body, auth, max_retries, retry_base_delay,
                    timeout=request_timeout, request=request, health=health,
                )
            return await _handle_sync(
                transport, dialect, entry, upstream_id, config.region,
                upstream_body, auth, max_retries, retry_base_delay,
                timeout=request_timeout, request=request, health=health,
            )
        if dialect.name != "anthropic":
            return _oai_error(
                400,
                f"Model '{raw_model}' is not available on /v1/chat/completions; "
                f"use /openai/v1/responses instead.",
                "invalid_request_error",
            )

        logger.info(
            "REQ model=%s -> %s msgs=%d tools=%d stream=%s",
            raw_model,
            model,
            len(body.get("messages", [])),
            len(body.get("tools", [])),
            stream,
        )

        # Model parameters
        default_max = registry.get_max_output(raw_model, 128_000)
        max_tokens = body.get(
            "max_tokens", body.get("max_completion_tokens", default_max)
        )
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p")
        stop = body.get("stop")

        # Convert messages
        system, chat_messages = extract_system_and_messages(
            body.get("messages", [])
        )

        # Build Bedrock payload
        bedrock_body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": chat_messages,
            "temperature": temperature,
        }
        if top_p is not None:
            bedrock_body["top_p"] = top_p
        if stop:
            bedrock_body["stop_sequences"] = (
                stop if isinstance(stop, list) else [stop]
            )
        if system:
            bedrock_body["system"] = system

        # Tools
        tools = body.get("tools", [])
        if tools:
            bedrock_body["tools"] = convert_tools(tools)
            tc = convert_tool_choice(body.get("tool_choice"), True)
            if tc:
                bedrock_body["tool_choice"] = tc

        # Extended thinking
        thinking = body.get("thinking")
        reasoning_effort = body.get("reasoning_effort")

        # reasoning_effort → thinking mapping (thinking takes precedence)
        if not thinking and reasoning_effort:
            thinking = map_reasoning_effort(reasoning_effort, model)

        if thinking:
            # Budget tokens minimum clamp (Bedrock requires >= 1024)
            if thinking.get("budget_tokens", 0) < 1024 and "budget_tokens" in thinking:
                thinking["budget_tokens"] = 1024

            bedrock_body["thinking"] = thinking
            bedrock_body.pop("temperature", None)

            # Auto-fill max_tokens when thinking is enabled
            if "max_tokens" not in body and "max_completion_tokens" not in body:
                budget = thinking.get("budget_tokens", 0)
                bedrock_body["max_tokens"] = budget + default_max if budget else default_max

        if stream:
            return await _handle_stream(
                transport, dialect, entry, model, config.region, bedrock_body,
                auth, max_retries, retry_base_delay,
                timeout=request_timeout, request=request, health=health,
            )
        return await _handle_sync(
            transport, dialect, entry, model, config.region, bedrock_body,
            auth, max_retries, retry_base_delay,
            timeout=request_timeout, request=request, health=health,
        )

    # ------------------------------------------------------------------
    # POST /openai/v1/responses  (OpenAI Responses API — GPT-5.5 via mantle)
    # ------------------------------------------------------------------

    @app.post("/openai/v1/responses")
    async def responses(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            return _oai_error(400, "Invalid JSON body")

        raw_model = body.get("model", "gpt-5.5")

        # Prefix passthrough first: ``azure/<deployment>`` → Azure resource,
        # dialect fixed by this endpoint (responses). Falls through when the
        # model carries no configured resource prefix.
        entry = registry.resolve_prefixed(raw_model, "openai-responses")
        if entry is not None:
            model = entry.deployment
        elif resolved := registry.resolve_for_dialect(raw_model, "openai-responses"):
            model, entry = resolved
        else:
            try:
                model = registry.resolve(raw_model)
            except UnknownModelError as exc:
                return _oai_error(400, str(exc), "invalid_request_error")
            entry = registry.get_entry(raw_model) or _fallback_entry(model)

        dialect = get_dialect(entry)
        transport = get_transport(entry)
        # This endpoint only serves the Responses dialect.
        if dialect.name != "openai-responses":
            return _oai_error(
                400,
                f"Model '{raw_model}' is not available on /openai/v1/responses; "
                f"use /v1/chat/completions instead.",
                "invalid_request_error",
            )
        stream = body.get("stream", False)

        # Passthrough: the client body is already native Responses format.
        # Swap the client-facing alias for the upstream id — the Azure
        # deployment name for Azure models, else the resolved Bedrock id.
        # Every other field (input, image blocks, reasoning, tools) is untouched
        # on the FIRST attempt: the raw client body is sent verbatim (model id
        # swapped only). No eager normalization.
        upstream_id = entry.deployment if entry.transport == "azure" else model
        upstream_body = dict(body)
        upstream_body["model"] = upstream_id
        # One-time safe-projection fallback policy, armed only for Bedrock
        # mantle GPT-5.x native Responses (never a user switch).
        compat = responses_compat_policy(entry.transport, entry.dialect, upstream_id)

        logger.info(
            "REQ [responses] model=%s -> %s (%s) stream=%s",
            raw_model, upstream_id, entry.transport, stream,
        )

        if stream:
            return await _handle_stream(
                transport, dialect, entry, upstream_id, config.region,
                upstream_body, auth, max_retries, retry_base_delay,
                timeout=request_timeout, request=request, health=health,
                compat=compat,
            )
        return await _handle_sync(
            transport, dialect, entry, upstream_id, config.region,
            upstream_body, auth, max_retries, retry_base_delay,
            timeout=request_timeout, request=request, health=health,
            compat=compat,
        )

    # ------------------------------------------------------------------
    # POST /openai/v1/images/generations  (OpenAI Images API — Azure)
    # ------------------------------------------------------------------

    @app.post("/openai/v1/images/generations")
    async def image_generations(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            return _oai_error(400, "Invalid JSON body")

        raw_model = body.get("model", "gpt-image-2")

        # Prefix passthrough first: ``azure/<deployment>`` → Azure resource,
        # dialect fixed by this endpoint (images). Falls through otherwise.
        entry = registry.resolve_prefixed(raw_model, "openai-images")
        if entry is not None:
            model = entry.deployment
        else:
            try:
                model = registry.resolve(raw_model)
            except UnknownModelError as exc:
                return _oai_error(400, str(exc), "invalid_request_error")
            entry = registry.get_entry(raw_model) or _fallback_entry(model)

        dialect = get_dialect(entry)
        transport = get_transport(entry)
        if dialect.name != "openai-images":
            return _oai_error(
                400,
                f"Model '{raw_model}' is not available on /openai/v1/images/generations; "
                f"use /openai/v1/responses instead.",
                "invalid_request_error",
            )
        if body.get("stream", False):
            return _oai_error(
                400,
                "/openai/v1/images/generations does not support streaming.",
                "invalid_request_error",
            )

        upstream_id = entry.deployment if entry.transport == "azure" else model
        upstream_body = dict(body)
        upstream_body["model"] = upstream_id

        logger.info(
            "REQ [images] model=%s -> %s (%s)",
            raw_model, upstream_id, entry.transport,
        )
        return await _handle_sync(
            transport, dialect, entry, upstream_id, config.region,
            upstream_body, auth, max_retries, retry_base_delay,
            operation="generations", timeout=request_timeout,
            request=request, health=health,
        )

    # ------------------------------------------------------------------
    # POST /openai/v1/images/edits  (OpenAI Images API — Azure)
    # ------------------------------------------------------------------

    @app.post("/openai/v1/images/edits")
    async def image_edits(request: Request) -> Any:
        media_type, multipart_options, form_request = _multipart_content_type(request)
        if media_type.lower() != b"multipart/form-data":
            return _oai_error(
                415,
                "/openai/v1/images/edits requires multipart/form-data.",
                "invalid_request_error",
            )

        try:
            form = await form_request.form()
        except Exception as exc:
            return _multipart_parse_error(request, exc, multipart_options)

        raw_model: str | None = None
        prompt_present = False
        stream = False
        image_count = 0
        text_parts: list[tuple[str, str]] = []
        file_parts: list[tuple[str, tuple[str, bytes, str]]] = []
        safe_files: list[dict[str, Any]] = []

        try:
            for field, value in form.multi_items():
                if isinstance(value, UploadFile):
                    raw = await value.read()
                    filename = value.filename or "upload"
                    part_type = value.content_type or "application/octet-stream"
                    file_parts.append((field, (filename, raw, part_type)))
                    safe_files.append({
                        "field": field,
                        "filename": filename,
                        "content_type": part_type,
                        "size": len(raw),
                    })
                    if field in {"image", "image[]"}:
                        image_count += 1
                    continue

                text = str(value)
                if field == "model":
                    raw_model = text.strip()
                    continue
                if field == "prompt":
                    prompt_present = bool(text.strip())
                if field == "stream":
                    normalized = text.strip().lower()
                    if normalized not in {"true", "false"}:
                        stream = None
                    else:
                        stream = normalized == "true"
                text_parts.append((field, text))
        finally:
            await form.close()

        if raw_model:
            request.state.metrics_info["model"] = raw_model
        if not raw_model:
            return _oai_error(400, "model is required", "invalid_request_error")
        if stream is None:
            return _oai_error(400, "stream must be true or false", "invalid_request_error")
        if image_count == 0:
            return _oai_error(400, "At least one image file is required", "invalid_request_error")
        if not prompt_present:
            return _oai_error(400, "prompt is required", "invalid_request_error")

        entry = registry.resolve_prefixed(raw_model, "openai-images")
        if entry is None:
            try:
                registry.resolve(raw_model)
            except UnknownModelError as exc:
                return _oai_error(400, str(exc), "invalid_request_error")
            entry = registry.get_entry(raw_model)

        if entry is None or (
            entry.transport != "azure" or entry.dialect != "openai-images"
        ):
            return _oai_error(
                400,
                f"Model '{raw_model}' is not an Azure OpenAI Images deployment available "
                "on /openai/v1/images/edits.",
                "invalid_request_error",
            )

        upstream_id = entry.deployment
        text_parts = [(name, value) for name, value in text_parts if name != "model"]
        text_parts.append(("model", upstream_id))
        multipart_parts: list[tuple[str, tuple[Any, ...]]] = [
            (name, (None, value)) for name, value in text_parts
        ]
        multipart_parts.extend(file_parts)
        encoded = httpx.Request(
            "POST", "https://multipart.invalid", files=multipart_parts
        )
        body_bytes = encoded.read()
        upstream_content_type = encoded.headers["Content-Type"]
        payload = PreparedRequestBody(
            content=body_bytes,
            content_type=upstream_content_type,
            log_body={
                "model": upstream_id,
                "operation": "edits",
                "fields": [name for name, _ in text_parts],
                "files": safe_files,
            },
        )

        transport = get_transport(entry)
        dialect = get_dialect(entry)
        logger.info(
            "REQ [images.edits] model=%s -> %s (%s) files=%d",
            raw_model, upstream_id, entry.transport, image_count,
        )
        handler = _handle_stream if stream else _handle_sync
        return await handler(
            transport, dialect, entry, upstream_id, config.region,
            payload, auth, max_retries, retry_base_delay,
            operation="edits", log_tag="images.edits",
            timeout=request_timeout, request=request, health=health,
        )

    # ------------------------------------------------------------------
    # POST /v1/embeddings  (OpenAI Embeddings API — Cohere / Titan)
    # ------------------------------------------------------------------

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            return _oai_error(400, "Invalid JSON body")

        raw_model = body.get("model", "cohere-embed-v4")

        try:
            model = registry.resolve(raw_model)
        except UnknownModelError as exc:
            return _oai_error(400, str(exc), "invalid_request_error")
        entry = registry.get_entry(raw_model) or _fallback_entry(model)

        transport = get_transport(entry)
        dialect = get_dialect(entry)
        if dialect.name != "openai-embeddings":
            return _oai_error(
                400,
                f"Model '{raw_model}' is not available on /v1/embeddings; "
                f"use /v1/chat/completions or /openai/v1/responses instead.",
                "invalid_request_error",
            )

        # Shape half: parse/validate the strict OpenAI request and resolve the
        # embeddings adapter (the only place a model→adapter decision is made).
        # The adapter builds the native bodies; the transport/dialect above own
        # URL + auth — never a provider branch in the server.
        try:
            ir = parse_request(body)
            adapter = resolve_profile(entry.embedding_profile)
            ir = apply_default_dimensions(ir, adapter)
            adapter.validate(ir)
        except EmbeddingsValidationError as exc:
            return _oai_error(
                exc.status_code,
                exc.message,
                "invalid_request_error",
                param=exc.param,
                code=exc.code,
            )
        except UnsupportedEmbeddingsModelError as exc:
            return _oai_error(
                exc.status_code,
                str(exc),
                "invalid_request_error",
                param="model",
                code="model_not_found",
            )

        native_bodies = adapter.build_requests(ir)

        logger.info(
            "REQ [embeddings] model=%s -> %s inputs=%d native_requests=%d",
            raw_model, model, len(ir.inputs), len(native_bodies),
        )

        return await _handle_embeddings(
            transport, dialect, entry, model, config.region,
            adapter, ir, native_bodies, auth, max_retries, retry_base_delay,
            timeout=request_timeout, request=request, health=health,
        )

    # ------------------------------------------------------------------
    # POST /v1/messages  (Anthropic Messages API)
    # ------------------------------------------------------------------

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        # Parse body
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content=format_anthropic_error(400, "Invalid JSON body"),
            )

        raw_model = body.get("model", "claude-haiku")

        # Prefer a resource's native Anthropic route; fall back to Responses
        # translation for resources such as Azure that expose no Messages API.
        _prefixed = registry.resolve_prefixed(raw_model, "anthropic-passthrough")
        if _prefixed is not None and _prefixed.transport != "http":
            _prefixed = None
        if _prefixed is None and (resolved := registry.resolve_for_dialect(
            raw_model, "anthropic-passthrough"
        )):
            model, _prefixed = resolved
        if _prefixed is None:
            _prefixed = registry.resolve_prefixed(raw_model, "openai-responses")
        if _prefixed is not None:
            entry = _prefixed
            model = entry.deployment
        else:
            try:
                model = registry.resolve(raw_model)
            except UnknownModelError as exc:
                return JSONResponse(
                    status_code=400,
                    content={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": str(exc),
                        },
                    },
                )
            entry = registry.get_entry(raw_model) or _fallback_entry(model)

        stream = body.get("stream", False)

        # Native HTTP Anthropic upstreams are forwarded without conversion.
        if entry.dialect == "anthropic-passthrough":
            upstream_body = dict(body)
            upstream_body["model"] = model
            transport = get_transport(entry)
            dialect = get_dialect(entry)
            protocol_headers = {
                key: value
                for key in ("anthropic-version", "anthropic-beta")
                if (value := request.headers.get(key)) is not None
            }
            logger.info(
                "REQ [messages-passthrough] model=%s -> %s (%s) stream=%s",
                raw_model, model, entry.transport, stream,
            )
            return await _handle_raw_passthrough(
                transport, dialect, entry, model, config.region, upstream_body,
                auth, max_retries, retry_base_delay, stream=stream,
                protocol_headers=protocol_headers, timeout=request_timeout,
                request=request, health=health,
            )

        # Dialect fork — the ONLY behaviour change on this endpoint:
        #   * anthropic       → existing Claude path (unchanged, below)
        #   * openai-responses → translate Anthropic Messages ⇄ Responses so an
        #     Anthropic-only client can drive GPT-5.5 / Grok / azure/<dep>.
        #   * anything else (openai-chat) → still a client mistake → 400.
        if entry.dialect == "openai-responses":
            client_model = raw_model
            upstream_id = entry.deployment if entry.transport == "azure" else model
            transport = get_transport(entry)
            dialect = get_dialect(entry)
            translated_body = dict(body)
            if "max_tokens" not in translated_body:
                translated_body["max_tokens"] = entry.max_output
            responses_body = to_responses_request(translated_body, upstream_id)
            logger.info(
                "REQ [messages->responses] client_model=%s upstream_model=%s "
                "transport=%s stream=%s max_output_tokens=%s defaulted=%s",
                client_model, upstream_id, entry.transport, stream,
                responses_body.get("max_output_tokens"), "max_tokens" not in body,
            )
            if stream:
                return await _handle_messages_via_responses_stream(
                    transport, dialect, entry, upstream_id, client_model, config.region,
                    responses_body, auth, max_retries, retry_base_delay,
                    timeout=request_timeout, request=request, health=health,
                )
            return await _handle_messages_via_responses_sync(
                transport, dialect, entry, upstream_id, client_model, config.region,
                responses_body, auth, max_retries, retry_base_delay,
                timeout=request_timeout, request=request, health=health,
            )
        if entry.dialect != "anthropic":
            return JSONResponse(
                status_code=400,
                content=format_anthropic_error(
                    400,
                    f"Model '{raw_model}' is not available on /v1/messages; "
                    f"use /openai/v1/responses or /v1/chat/completions instead.",
                ),
            )

        # max_tokens is required by the Anthropic API spec
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = registry.get_max_output(raw_model, 64_000)

        logger.info(
            "REQ [messages] model=%s -> %s msgs=%d stream=%s",
            raw_model,
            model,
            len(body.get("messages", [])),
            stream,
        )

        # Build Bedrock payload — mostly pass-through since Bedrock
        # already uses the Anthropic format internally.
        bedrock_body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": body.get("messages", []),
        }

        # Optional fields
        if "system" in body:
            bedrock_body["system"] = body["system"]
        if "temperature" in body:
            bedrock_body["temperature"] = body["temperature"]
        if "top_p" in body:
            bedrock_body["top_p"] = body["top_p"]
        if "top_k" in body:
            bedrock_body["top_k"] = body["top_k"]
        if "stop_sequences" in body:
            bedrock_body["stop_sequences"] = body["stop_sequences"]
        if "metadata" in body:
            bedrock_body["metadata"] = body["metadata"]

        # Tools
        if "tools" in body:
            bedrock_body["tools"] = body["tools"]
        if "tool_choice" in body:
            bedrock_body["tool_choice"] = body["tool_choice"]

        # Extended thinking
        thinking = body.get("thinking")
        if thinking:
            if thinking.get("budget_tokens", 0) < 1024 and "budget_tokens" in thinking:
                thinking["budget_tokens"] = 1024
            bedrock_body["thinking"] = thinking
            bedrock_body.pop("temperature", None)
            # Auto-fill max_tokens when thinking is enabled
            if "max_tokens" not in body:
                budget = thinking.get("budget_tokens", 0)
                default_max = registry.get_max_output(raw_model, 64_000)
                bedrock_body["max_tokens"] = budget + default_max if budget else default_max

        if stream:
            return await _handle_messages_stream(
                model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay,
                timeout=request_timeout, request=request, health=health,
            )
        return await _handle_messages_sync(
            model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay,
            timeout=request_timeout, request=request, health=health,
        )

    # ------------------------------------------------------------------
    # POST /v1/messages/count_tokens  (Anthropic SDK token pre-flight)
    # ------------------------------------------------------------------

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> JSONResponse:
        # Bedrock doesn't expose a token counter, so we return a rough
        # character-based estimate (~4 chars/token). Good enough for the
        # SDK's budget checks, which is all this endpoint needs to satisfy.
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content=format_anthropic_error(400, "Invalid JSON body"),
            )

        def _content_chars(content: Any) -> int:
            if content is None:
                return 0
            if isinstance(content, str):
                return len(content)
            if isinstance(content, list):
                total = 0
                for block in content:
                    if isinstance(block, dict):
                        # text blocks
                        if "text" in block:
                            total += len(str(block["text"]))
                        # tool_result blocks carry their own content payload
                        elif "content" in block:
                            total += _content_chars(block["content"])
                        else:
                            total += len(json.dumps(block))
                    else:
                        total += len(str(block))
                return total
            return len(str(content))

        total_chars = 0
        for msg in body.get("messages", []) or []:
            if isinstance(msg, dict):
                total_chars += _content_chars(msg.get("content"))

        system = body.get("system")
        total_chars += _content_chars(system)

        # Tools add to the prompt too — include their schemas.
        for tool in body.get("tools", []) or []:
            total_chars += len(json.dumps(tool))

        input_tokens = max(1, total_chars // 4)
        return JSONResponse({"input_tokens": input_tokens})

    return app


# ---------------------------------------------------------------------------
# Native HTTP passthrough handler
# ---------------------------------------------------------------------------

_SAFE_UPSTREAM_HEADERS = {
    "content-type", "request-id", "x-request-id", "retry-after", "cache-control",
}


def _passthrough_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value for key, value in headers.items()
        if key.lower() in _SAFE_UPSTREAM_HEADERS
        or key.lower().startswith(("anthropic-ratelimit-", "ratelimit-", "x-ratelimit-"))
    }


async def _handle_raw_passthrough(
    transport: Transport,
    dialect: Dialect,
    entry: ModelEntry,
    model: str,
    region: str,
    body: dict[str, Any],
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    stream: bool,
    protocol_headers: dict[str, str],
    timeout: float,
    request: Request | None,
    health: HealthMonitor | None,
) -> Response:
    url = transport.build_url(dialect.operation_path(entry, stream), region, entry)
    payload = _prepare_request_body(body)
    headers = _payload_headers(transport, entry, auth, url, payload)
    for key, value in protocol_headers.items():
        headers[key] = value
    headers["Content-Type"] = payload.content_type

    retryable = {429, 503, 529}
    deadline = _retry_deadline(timeout, max_retries)

    if not stream:
        for attempt in range(max_retries):
            try:
                async with _track_upstream(health), httpx.AsyncClient(
                    timeout=_httpx_timeout(timeout)
                ) as client:
                    response = await client.post(
                        url, headers=headers, content=payload.content
                    )
            except httpx.TimeoutException:
                _note_timeout(request)
                if attempt < max_retries - 1 and time.monotonic() < deadline:
                    _note_retry(request)
                    await asyncio.sleep(retry_base_delay * (2**attempt))
                    continue
                return JSONResponse(
                    status_code=504,
                    content=format_anthropic_error(504, "Upstream request timeout"),
                )
            except httpx.HTTPError:
                if attempt < max_retries - 1 and time.monotonic() < deadline:
                    _note_retry(request)
                    await asyncio.sleep(retry_base_delay * (2**attempt))
                    continue
                return JSONResponse(
                    status_code=502,
                    content=format_anthropic_error(502, "Upstream connection failed"),
                )
            if (
                response.status_code in retryable
                and attempt < max_retries - 1
                and time.monotonic() < deadline
            ):
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2**attempt))
                continue
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=_passthrough_response_headers(response.headers),
                media_type=None,
            )

    for attempt in range(max_retries):
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(_track_upstream(health))
            client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=_httpx_timeout(timeout))
            )
            response = await stack.enter_async_context(
                client.stream("POST", url, headers=headers, content=payload.content)
            )
        except httpx.TimeoutException:
            await stack.aclose()
            _note_timeout(request)
            if attempt < max_retries - 1 and time.monotonic() < deadline:
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2**attempt))
                continue
            return JSONResponse(
                status_code=504,
                content=format_anthropic_error(504, "Upstream connect timeout"),
            )
        except httpx.HTTPError:
            await stack.aclose()
            if attempt < max_retries - 1 and time.monotonic() < deadline:
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2**attempt))
                continue
            return JSONResponse(
                status_code=502,
                content=format_anthropic_error(502, "Upstream connection failed"),
            )

        if response.status_code in retryable and attempt < max_retries - 1:
            await response.aread()
            await stack.aclose()
            _note_retry(request)
            await asyncio.sleep(retry_base_delay * (2**attempt))
            continue
        if response.status_code != 200:
            content = await response.aread()
            response_headers = _passthrough_response_headers(response.headers)
            status = response.status_code
            await stack.aclose()
            return Response(
                content, status_code=status, headers=response_headers, media_type=None
            )

        async def generate():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await stack.aclose()

        response_headers = _passthrough_response_headers(response.headers)
        return StreamingResponse(
            generate(), status_code=200, headers=response_headers,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    return JSONResponse(
        status_code=502,
        content=format_anthropic_error(502, "Upstream unavailable"),
    )


# ---------------------------------------------------------------------------
# Sync handler
# ---------------------------------------------------------------------------

async def _handle_sync(
    transport: Transport,
    dialect: Dialect,
    entry: ModelEntry,
    model: str,
    region: str,
    bedrock_body: dict[str, Any] | PreparedRequestBody,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    operation: str | None = None,
    log_tag: str = "chat.completions",
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
    compat: CompatibilityPolicy | None = None,
) -> dict | JSONResponse:
    url = transport.build_url(
        _operation_path(dialect, entry, False, operation), region, entry
    )
    payload = _prepare_request_body(bedrock_body)
    last_error: str | None = None
    deadline = _retry_deadline(timeout, max_retries)
    attempt = 0
    compat_attempted = False

    while True:
        if attempt >= max_retries:
            break
        # P3: stop retrying once the total wall-clock budget is spent, rather
        # than stacking N full per-attempt timeouts into a multi-minute hang.
        if attempt > 0 and time.monotonic() >= deadline:
            logger.warning(
                "RETRY-BUDGET exhausted model=%s attempt=%d/%d", model,
                attempt + 1, max_retries,
            )
            break
        try:
            # Transport-specific headers (e.g. Azure api-key) override the
            # gateway's global auth; None → use the global SigV4/Bearer path.
            headers = _payload_headers(transport, entry, auth, url, payload)
            async with _track_upstream(health), httpx.AsyncClient(timeout=_httpx_timeout(timeout)) as client:
                resp = await client.post(url, headers=headers, content=payload.content)

            if resp.status_code == 200:
                # P1: a 200 with an unparseable body is an upstream fault (502),
                # not a gateway crash (500) — guard the JSON decode.
                try:
                    result = resp.json()
                except (ValueError, json.JSONDecodeError):
                    logger.error(
                        "BADJSON model=%s upstream 200 but body not JSON (%d bytes)",
                        model, len(resp.content),
                    )
                    return _oai_error(
                        502, "Upstream returned a malformed (non-JSON) response",
                        "api_error",
                    )
                client_body, log_info = dialect.render_sync(result, model)
                logger.info(
                    "RES model=%s finish=%s in=%s out=%s attempt=%d",
                    model,
                    log_info.get("finish", "?"),
                    log_info.get("input_tokens", "?"),
                    log_info.get("output_tokens", "?"),
                    attempt + 1,
                )
                return client_body

            if resp.status_code in (429, 529, 503):
                last_error = resp.text[:200]
                delay = retry_base_delay * (2**attempt)
                logger.warning(
                    "RETRY %d model=%s attempt=%d/%d delay=%.1fs",
                    resp.status_code,
                    model,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                _note_retry(request)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            # One-time compatibility fallback: only an exact variant 400 on a
            # Bedrock GPT-5.x native Responses request. Independent of the normal
            # retry budget (but bounded by the shared deadline), hard-capped at 1.
            if (
                compat is not None
                and not compat_attempted
                and resp.status_code == 400
                and is_exact_variant_rejection(400, resp.text)
            ):
                compat_attempted = True
                projected_body = _compat_projection(compat, payload.log_body, model)
                if projected_body is not None and time.monotonic() < deadline:
                    payload = _prepare_request_body(projected_body)
                    continue
                # unsafe / no-change projection → fall through to the original 400

            error = parse_bedrock_error(resp.status_code, resp.text)
            _log_upstream_error(
                resp.status_code,
                "ERR %d model=%s msg=%s",
                resp.status_code,
                model,
                error["message"][:300],
            )
            if payload.log_body is not None:
                _log_request_shape_for_upstream_error(
                    status=resp.status_code,
                    model=model,
                    body=payload.log_body,
                    err_text=error["message"],
                )
            return _oai_error(
                resp.status_code, error["message"], error["type"]
            )

        except httpx.TimeoutException:
            last_error = "Request timeout"
            logger.warning(
                "TIMEOUT model=%s attempt=%d/%d",
                model,
                attempt + 1,
                max_retries,
            )
            _note_retry(request)
            _note_timeout(request)
            await asyncio.sleep(retry_base_delay * (2**attempt))
            attempt += 1

        except Exception as exc:
            logger.exception("UNEXPECTED model=%s during %s", model, log_tag)
            return _oai_error(500, str(exc))

    logger.error(
        "FAILED model=%s all %d retries exhausted: %s",
        model,
        max_retries,
        last_error,
    )
    return _oai_error(502, f"All {max_retries} retries failed: {last_error}")


# ---------------------------------------------------------------------------
# Embeddings fanout handler
# ---------------------------------------------------------------------------

# Cap concurrent native embedding requests so a wide Titan batch can't open an
# unbounded number of upstream connections at once.
_EMBEDDINGS_MAX_CONCURRENCY = 8

# Upstream statuses worth retrying per native request (same set as the chat /
# responses sync path; timeouts are handled as a separate exception below).
_EMBEDDINGS_RETRYABLE = frozenset({429, 503, 529})


class _EmbeddingsUpstreamError(Exception):
    """A native embeddings request failed permanently (after its retries)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


async def _embeddings_attempt(
    *,
    index: int,
    payload: PreparedRequestBody,
    url: str,
    transport: Transport,
    entry: ModelEntry,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    timeout: float,
    deadline: float,
    request: Request | None,
    health: HealthMonitor | None,
    model: str,
) -> dict:
    """Send one native embeddings body, retrying 429/503/529 and timeouts.

    Returns the parsed 200 upstream JSON on success; raises
    :class:`_EmbeddingsUpstreamError` on a permanent failure. The caller
    enforces all-or-nothing across the fan-out.
    """
    last_error: str | None = None
    for attempt in range(max_retries):
        if attempt > 0 and time.monotonic() >= deadline:
            logger.warning(
                "EMB retry-budget exhausted model=%s index=%d attempt=%d/%d",
                model, index, attempt + 1, max_retries,
            )
            break
        try:
            headers = _payload_headers(transport, entry, auth, url, payload)
            async with _track_upstream(health), httpx.AsyncClient(
                timeout=_httpx_timeout(timeout)
            ) as client:
                resp = await client.post(url, headers=headers, content=payload.content)
        except httpx.TimeoutException:
            last_error = "Upstream request timeout"
            _note_timeout(request)
            if attempt < max_retries - 1 and time.monotonic() < deadline:
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2 ** attempt))
                continue
            break
        except httpx.HTTPError as exc:
            last_error = f"Upstream connection failed: {exc}"
            if attempt < max_retries - 1 and time.monotonic() < deadline:
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2 ** attempt))
                continue
            break

        if resp.status_code == 200:
            try:
                return resp.json()
            except (ValueError, json.JSONDecodeError):
                logger.error(
                    "BADJSON [embeddings] model=%s index=%d upstream 200 but "
                    "body not JSON", model, index,
                )
                raise _EmbeddingsUpstreamError(
                    502, "Upstream returned a malformed (non-JSON) response"
                )

        if resp.status_code in _EMBEDDINGS_RETRYABLE:
            last_error = resp.text[:200] or f"upstream {resp.status_code}"
            logger.warning(
                "RETRY [embeddings] %d model=%s index=%d attempt=%d/%d",
                resp.status_code, model, index, attempt + 1, max_retries,
            )
            if attempt < max_retries - 1 and time.monotonic() < deadline:
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2 ** attempt))
                continue
            break

        error = parse_bedrock_error(resp.status_code, resp.text)
        _log_upstream_error(
            resp.status_code,
            "ERR [embeddings] %d model=%s index=%d msg=%s",
            resp.status_code, model, index, error["message"][:300],
        )
        raise _EmbeddingsUpstreamError(resp.status_code, error["message"])

    raise _EmbeddingsUpstreamError(
        502,
        f"All {max_retries} retries failed: {last_error or 'upstream unavailable'}",
    )


async def _handle_embeddings(
    transport: Transport,
    dialect: Dialect,
    entry: ModelEntry,
    model: str,
    region: str,
    adapter: EmbeddingsAdapter,
    ir: EmbeddingRequest,
    native_bodies: list[dict],
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
) -> dict | JSONResponse:
    """Fan out native embedding requests with bounded concurrency.

    All items share one wall-clock deadline; each item retries its own
    429/503/529/timeouts independently. Results are re-ordered to match the
    input, usage is aggregated by the adapter, and any permanent item failure
    fails the whole request (all-or-nothing — never a partial vector list).
    """
    url = transport.build_url(dialect.operation_path(entry, False), region, entry)
    payloads = [_prepare_request_body(body) for body in native_bodies]
    deadline = _retry_deadline(timeout, max_retries)
    semaphore = asyncio.Semaphore(_EMBEDDINGS_MAX_CONCURRENCY)

    async def run(index: int, payload: PreparedRequestBody) -> dict:
        async with semaphore:
            return await _embeddings_attempt(
                index=index,
                payload=payload,
                url=url,
                transport=transport,
                entry=entry,
                auth=auth,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                timeout=timeout,
                deadline=deadline,
                request=request,
                health=health,
                model=model,
            )

    results = await asyncio.gather(
        *(run(i, payload) for i, payload in enumerate(payloads)),
        return_exceptions=True,
    )

    # All-or-nothing: a permanent failure on any native request aborts the
    # whole response rather than returning a partial ``data`` array.
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            if isinstance(result, _EmbeddingsUpstreamError):
                return _oai_error(result.status, result.message, "api_error")
            logger.exception(
                "UNEXPECTED [embeddings] model=%s index=%d", model, index,
            )
            return _oai_error(500, str(result))

    natives = [result for result in results if isinstance(result, dict)]
    try:
        response_ir = adapter.render_responses(natives, ir)
    except EmbeddingsError as exc:
        return _oai_error(502, str(exc), "api_error")

    return render_openai(response_ir, ir.encoding_format)


# ---------------------------------------------------------------------------
# Streaming preflight
# ---------------------------------------------------------------------------

async def _open_upstream_stream(
    url: str,
    payload: PreparedRequestBody,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    request: Request | None,
    health: HealthMonitor | None,
    log_tag: str,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 300.0,
    compat: CompatibilityPolicy | None = None,
) -> tuple[Any, AsyncExitStack | None, dict | None]:
    """Open the Bedrock streaming connection and inspect the HTTP status
    *before* any bytes are handed to the client.

    This is what lets a *pre-stream* failure (bad request, unsupported tool,
    auth, model-not-found, retry-exhausted throttling) be returned as a real
    HTTP error response — exactly like the non-streaming path and like the
    upstream Anthropic API — instead of being disguised as a ``200`` SSE body
    carrying an orphan ``error`` event (which leaves clients hanging).

    Retries 429/503/529 with exponential backoff, mirroring the sync path.

    Returns ``(response, stack, None)`` on success — the caller MUST consume
    ``response.aiter_bytes()`` and ``await stack.aclose()`` when done — or
    ``(None, None, error_dict)`` when the stream could not be opened, where
    ``error_dict`` is ``{"status": int, "type": str, "message": str}``.
    """
    last_status = 502
    last_message = "upstream unavailable"
    deadline = _retry_deadline(timeout, max_retries)
    attempt = 0
    compat_attempted = False
    while True:
        if attempt >= max_retries:
            break
        if attempt > 0 and time.monotonic() >= deadline:  # P3: total budget
            logger.warning(
                "STREAM-OPEN retry-budget exhausted [%s] attempt=%d/%d",
                log_tag, attempt + 1, max_retries,
            )
            break
        stack = AsyncExitStack()
        # Transport-specific headers (Azure api-key) override the global auth.
        headers = extra_headers or auth.get_headers(
            method="POST", url=url, body=payload.content
        )
        if extra_headers is None:
            headers = dict(headers)
            headers["Content-Type"] = payload.content_type
        try:
            await stack.enter_async_context(_track_upstream(health))
            client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=_httpx_timeout(timeout))
            )
            resp = await stack.enter_async_context(
                client.stream("POST", url, headers=headers, content=payload.content)
            )
        except httpx.TimeoutException:
            await stack.aclose()
            _note_timeout(request)
            last_status, last_message = 504, "Upstream connect timeout"
            logger.warning(
                "STREAM-OPEN timeout [%s] attempt=%d/%d",
                log_tag, attempt + 1, max_retries,
            )
            if attempt < max_retries - 1:
                _note_retry(request)
                await asyncio.sleep(retry_base_delay * (2**attempt))
            attempt += 1
            continue
        except Exception as exc:  # noqa: BLE001
            # Connection-level failure before any bytes flowed (DNS, refused,
            # TLS, …). This is a pre-stream error → surface as a real HTTP 500
            # rather than letting it escape as an unhandled 500 with no body.
            await stack.aclose()
            logger.exception("STREAM-OPEN unexpected [%s]", log_tag)
            return None, None, {
                "status": 500,
                "type": "api_error",
                "message": str(exc),
            }

        status = resp.status_code

        if status == 200:
            logger.info("STREAM-OPEN ok [%s] attempt=%d", log_tag, attempt + 1)
            return resp, stack, None

        # Read the upstream error body, then release the connection.
        err_body = ""
        try:
            async for chunk in resp.aiter_text():
                err_body += chunk
        except Exception:  # noqa: BLE001 — best-effort body capture
            pass
        await stack.aclose()

        if status in (429, 503, 529):
            last_status = status
            last_message = err_body[:200] or f"upstream {status}"
            _note_retry(request)
            logger.warning(
                "STREAM-OPEN retryable %d [%s] attempt=%d/%d",
                status, log_tag, attempt + 1, max_retries,
            )
            await asyncio.sleep(retry_base_delay * (2**attempt))
            attempt += 1
            continue

        # One-time compatibility fallback: exact variant 400, before any client
        # SSE byte is emitted, on a Bedrock GPT-5.x native Responses request.
        if (
            compat is not None
            and not compat_attempted
            and status == 400
            and is_exact_variant_rejection(400, err_body)
        ):
            compat_attempted = True
            model = str(log_tag)
            if isinstance(payload.log_body, dict) and payload.log_body.get("model"):
                model = str(payload.log_body["model"])
            projected_body = _compat_projection(compat, payload.log_body, model)
            if projected_body is not None and time.monotonic() < deadline:
                payload = _prepare_request_body(projected_body)
                continue
            # unsafe / no-change projection → fall through to the original error

        # Deterministic, non-retryable failure → surface as real HTTP error.
        error = parse_bedrock_error(status, err_body)
        _log_upstream_error(
            status,
            "STREAM-OPEN error %d [%s] msg=%s",
            status, log_tag, error["message"][:300],
        )
        if payload.log_body is not None:
            _log_request_shape_for_upstream_error(
                status=status,
                model=str(payload.log_body.get("model") or log_tag),
                body=payload.log_body,
                err_text=error["message"],
            )
        return None, None, {
            "status": status,
            "type": error["type"],
            "message": error["message"],
        }

    logger.error(
        "STREAM-OPEN failed [%s] all %d attempts exhausted: %s",
        log_tag, max_retries, last_message,
    )
    return None, None, {
        "status": last_status,
        "type": parse_bedrock_error(last_status, "")["type"],
        "message": f"All {max_retries} attempts failed: {last_message}",
    }


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------

async def _handle_stream(
    transport: Transport,
    dialect: Dialect,
    entry: ModelEntry,
    model: str,
    region: str,
    bedrock_body: dict[str, Any] | PreparedRequestBody,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    operation: str | None = None,
    log_tag: str = "chat",
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
    compat: CompatibilityPolicy | None = None,
) -> JSONResponse | StreamingResponse:
    url = transport.build_url(
        _operation_path(dialect, entry, True, operation), region, entry
    )
    payload = _prepare_request_body(bedrock_body)
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Preflight: open the upstream stream and check the status BEFORE we commit
    # to a 200 SSE response. A pre-stream failure becomes a real HTTP error.
    resp, stack, err = await _open_upstream_stream(
        url, payload, auth, max_retries, retry_base_delay,
        request=request, health=health, log_tag=f"{log_tag} model={model}",
        extra_headers=_transport_payload_headers(transport, entry, payload),
        timeout=timeout,
        compat=compat,
    )
    if err is not None:
        return _oai_error(err["status"], err["message"], err["type"])

    async def generate():
        try:
            # Dialect-specific transform: decode the upstream stream and
            # re-emit client-facing SSE (OpenAI chunks for Anthropic, or a
            # verbatim passthrough for Responses).
            async for chunk in dialect.transform_stream(
                resp.aiter_bytes(), model, msg_id
            ):
                yield chunk
        except httpx.TimeoutException:
            # Connection dropped mid-stream after data may have flowed.
            _note_timeout(request)
            logger.warning("STREAM-MID timeout [chat] model=%s", model)
            yield dialect.stream_error("Upstream timeout mid-stream", 504)
        except Exception as exc:  # noqa: BLE001
            logger.exception("UNEXPECTED [stream] model=%s during chat.completions", model)
            yield dialect.stream_error(str(exc), 500)
        finally:
            await stack.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Anthropic Messages API — Sync handler
# ---------------------------------------------------------------------------

async def _handle_messages_sync(
    model: str,
    bedrock_body: dict,
    bedrock_base: str,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
) -> dict | JSONResponse:
    url = f"{bedrock_base}/model/{model}/invoke"
    body_bytes = json.dumps(bedrock_body).encode()
    last_error: str | None = None
    deadline = _retry_deadline(timeout, max_retries)

    for attempt in range(max_retries):
        if attempt > 0 and time.monotonic() >= deadline:  # P3: total budget
            logger.warning(
                "RETRY-BUDGET exhausted [messages] model=%s attempt=%d/%d",
                model, attempt + 1, max_retries,
            )
            break
        try:
            headers = auth.get_headers(method="POST", url=url, body=body_bytes)
            async with _track_upstream(health), httpx.AsyncClient(timeout=_httpx_timeout(timeout)) as client:
                resp = await client.post(url, headers=headers, content=body_bytes)

            if resp.status_code == 200:
                try:  # P1: malformed 200 body → 502, not a 500 crash
                    result = resp.json()
                except (ValueError, json.JSONDecodeError):
                    logger.error(
                        "BADJSON [messages] model=%s upstream 200 but body not JSON",
                        model,
                    )
                    return JSONResponse(
                        status_code=502,
                        content=format_anthropic_error(
                            502, "Upstream returned a malformed (non-JSON) response"
                        ),
                    )
                usage = result.get("usage", {})
                logger.info(
                    "RES [messages] model=%s stop=%s in=%s out=%s attempt=%d",
                    model,
                    result.get("stop_reason", "?"),
                    usage.get("input_tokens", "?"),
                    usage.get("output_tokens", "?"),
                    attempt + 1,
                )
                return format_anthropic_response(result, model)

            if resp.status_code in (429, 529, 503):
                last_error = resp.text[:200]
                delay = retry_base_delay * (2**attempt)
                logger.warning(
                    "RETRY [messages] %d model=%s attempt=%d/%d delay=%.1fs",
                    resp.status_code,
                    model,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                _note_retry(request)
                await asyncio.sleep(delay)
                continue

            error = parse_bedrock_error(resp.status_code, resp.text)
            _log_upstream_error(
                resp.status_code,
                "ERR [messages] %d model=%s msg=%s",
                resp.status_code,
                model,
                error["message"][:300],
            )
            return JSONResponse(
                status_code=resp.status_code,
                content=format_anthropic_error(
                    resp.status_code, error["message"]
                ),
            )

        except httpx.TimeoutException:
            last_error = "Request timeout"
            logger.warning(
                "TIMEOUT [messages] model=%s attempt=%d/%d",
                model,
                attempt + 1,
                max_retries,
            )
            _note_retry(request)
            _note_timeout(request)
            await asyncio.sleep(retry_base_delay * (2**attempt))

        except Exception as exc:
            logger.exception("UNEXPECTED [messages] model=%s", model)
            return JSONResponse(
                status_code=500,
                content=format_anthropic_error(500, str(exc)),
            )

    logger.error(
        "FAILED [messages] model=%s all %d retries exhausted: %s",
        model,
        max_retries,
        last_error,
    )
    return JSONResponse(
        status_code=502,
        content=format_anthropic_error(
            502, f"All {max_retries} retries failed: {last_error}"
        ),
    )


# ---------------------------------------------------------------------------
# Anthropic Messages API — Streaming handler
# ---------------------------------------------------------------------------

async def _handle_messages_stream(
    model: str,
    bedrock_body: dict,
    bedrock_base: str,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
) -> JSONResponse | StreamingResponse:
    url = f"{bedrock_base}/model/{model}/invoke-with-response-stream"
    body_bytes = json.dumps(bedrock_body).encode()
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Preflight: a pre-stream failure (bad request, unsupported tool, auth,
    # model-not-found, retry-exhausted) is returned as a real HTTP error with
    # the proper status code and a complete Anthropic error envelope — matching
    # the upstream Anthropic API — instead of a 200 SSE body with an orphan
    # error event that leaves clients hanging.
    payload = _prepare_request_body(bedrock_body)
    resp, stack, err = await _open_upstream_stream(
        url, payload, auth, max_retries, retry_base_delay,
        request=request, health=health, log_tag=f"messages model={model}",
        timeout=timeout,
    )
    if err is not None:
        return JSONResponse(
            status_code=err["status"],
            content={
                "type": "error",
                "error": {"type": err["type"], "message": err["message"]},
            },
        )

    async def generate():  # noqa: C901
        started = False  # whether message_start has been emitted
        try:
            buf = b""
            async for raw in resp.aiter_bytes():
                buf += raw
                events, consumed = decode_event_stream_chunk(buf)
                if consumed > 0:
                    buf = buf[consumed:]
                for event in events:
                    etype = event.get("type", "")

                    if etype == "_exception":
                        # Mid-stream upstream fault. Emit a valid Anthropic
                        # error event (a legal stream terminator) so the client
                        # stops cleanly instead of waiting forever for frames
                        # that the regex decoder used to drop silently.
                        _log_upstream_error(
                            event.get("status", 500),
                            "STREAM-MID error [messages] model=%s type=%s msg=%s",
                            model,
                            event.get("exception_type", "?"),
                            event.get("message", "")[:300],
                        )
                        yield make_anthropic_sse(
                            "error",
                            {
                                "type": "error",
                                "error": {
                                    "type": parse_bedrock_error(
                                        event.get("status", 500), ""
                                    )["type"],
                                    "message": event.get(
                                        "message", "upstream stream error"
                                    ),
                                },
                            },
                        )
                        return

                    if etype == "message_start":
                        started = True
                        # Enrich the message_start with our ID & model
                        msg_obj = event.get("message", {})
                        msg_obj["id"] = msg_id
                        msg_obj["model"] = model
                        msg_obj.setdefault("type", "message")
                        msg_obj.setdefault("role", "assistant")
                        msg_obj.setdefault("content", [])
                        msg_obj.setdefault("stop_reason", None)
                        msg_obj.setdefault("stop_sequence", None)
                        yield make_anthropic_sse(
                            "message_start",
                            {"type": "message_start", "message": msg_obj},
                        )

                    elif etype == "content_block_start":
                        yield make_anthropic_sse(
                            "content_block_start", event
                        )

                    elif etype == "content_block_delta":
                        yield make_anthropic_sse(
                            "content_block_delta", event
                        )

                    elif etype == "content_block_stop":
                        yield make_anthropic_sse(
                            "content_block_stop", event
                        )

                    elif etype == "message_delta":
                        yield make_anthropic_sse(
                            "message_delta", event
                        )

                    elif etype == "message_stop":
                        yield make_anthropic_sse(
                            "message_stop",
                            {"type": "message_stop"},
                        )

                    elif etype == "ping":
                        yield make_anthropic_sse(
                            "ping", {"type": "ping"}
                        )

        except httpx.TimeoutException:
            _note_timeout(request)
            logger.warning(
                "STREAM-MID timeout [messages] model=%s started=%s",
                model, started,
            )
            yield make_anthropic_sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Upstream timeout mid-stream",
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("UNEXPECTED [messages-stream] model=%s", model)
            yield make_anthropic_sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
            )
        finally:
            await stack.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Anthropic Messages ⇄ OpenAI Responses (inbound translation)
# ---------------------------------------------------------------------------
#
# Lets an Anthropic-only client (Claude Code via ANTHROPIC_BASE_URL) drive a
# Responses-dialect model. The request was already translated to a Responses
# body by ``to_responses_request``; these handlers reuse the SAME generic
# upstream primitives as every other path (``_handle_sync`` /
# ``_open_upstream_stream`` — retries, timeout budget, preflight, metrics) and
# only add the response-direction translation back to Anthropic Messages.


def _oai_error_to_anthropic(resp: JSONResponse) -> JSONResponse:
    """Re-wrap an OpenAI-style error JSONResponse as an Anthropic error.

    ``_handle_sync`` emits OpenAI-shaped error envelopes; on the /v1/messages
    path the client expects the Anthropic shape, so translate the envelope
    while preserving the status code and message.
    """
    status = resp.status_code
    message = "upstream error"
    try:
        payload = json.loads(bytes(resp.body))
        message = payload.get("error", {}).get("message", message)
    except (ValueError, TypeError, AttributeError):  # noqa: BLE001
        pass
    return JSONResponse(
        status_code=status, content=format_anthropic_error(status, message)
    )


async def _handle_messages_via_responses_sync(
    transport: Transport,
    dialect: Dialect,
    entry: ModelEntry,
    upstream_model: str,
    client_model: str,
    region: str,
    responses_body: dict,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
) -> dict | JSONResponse:
    # Reuse the generic sync path: for the Responses dialect it returns the
    # upstream JSON verbatim (render_sync passthrough) or an error JSONResponse.
    result = await _handle_sync(
        transport, dialect, entry, upstream_model, region, responses_body, auth,
        max_retries, retry_base_delay,
        timeout=timeout, request=request, health=health,
    )
    if isinstance(result, JSONResponse):
        return _oai_error_to_anthropic(result)
    if result.get("status") == "failed" or (
        result.get("error") and result.get("status") not in {"completed", "incomplete"}
    ):
        status, error_type, message = responses_error_to_anthropic(
            result.get("error") or result
        )
        logger.warning(
            "RESPONSES-FAILED client_model=%s upstream_model=%s status=%d "
            "error_type=%s",
            client_model, upstream_model, status, error_type,
        )
        return JSONResponse(
            status_code=status,
            content={
                "type": "error",
                "error": {"type": error_type, "message": message},
            },
        )
    return to_anthropic_response(result, client_model)


async def _handle_messages_via_responses_stream(
    transport: Transport,
    dialect: Dialect,
    entry: ModelEntry,
    upstream_model: str,
    client_model: str,
    region: str,
    responses_body: dict,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    timeout: float = 300.0,
    request: Request | None = None,
    health: HealthMonitor | None = None,
) -> JSONResponse | StreamingResponse:
    responses_body = dict(responses_body)
    responses_body["stream"] = True
    url = transport.build_url(dialect.operation_path(entry, True), region, entry)
    body_bytes = json.dumps(responses_body).encode()
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Same pre-stream preflight as every streaming path: a failure before any
    # bytes flow is returned as a real HTTP error (Anthropic-shaped here).
    payload = _prepare_request_body(responses_body)
    resp, stack, err = await _open_upstream_stream(
        url, payload, auth, max_retries, retry_base_delay,
        request=request, health=health,
        log_tag=f"messages->responses model={upstream_model}",
        extra_headers=_transport_payload_headers(transport, entry, payload),
        timeout=timeout,
    )
    if err is not None:
        return JSONResponse(
            status_code=err["status"],
            content={
                "type": "error",
                "error": {"type": err["type"], "message": err["message"]},
            },
        )

    adapter = AnthropicStreamAdapter(client_model, msg_id)

    async def generate():
        try:
            async for frame in adapter.translate(resp.aiter_bytes()):
                yield frame
        except httpx.TimeoutException:
            _note_timeout(request)
            logger.warning(
                "STREAM-MID timeout [messages->responses] client_model=%s "
                "upstream_model=%s terminal=%s",
                client_model, upstream_model, adapter.terminal,
            )
            for frame in adapter.terminate_error("Upstream timeout mid-stream"):
                yield frame
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "UNEXPECTED [messages->responses stream] client_model=%s "
                "upstream_model=%s terminal=%s",
                client_model, upstream_model, adapter.terminal,
            )
            for frame in adapter.terminate_error(str(exc)):
                yield frame
        finally:
            await stack.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run(config: GatewayConfig | None = None) -> None:
    """Start the gateway server (blocking)."""
    if config is None:
        config = load_config()

    configure_logging(config)

    app = create_app(config)
    logger.info(
        "MuxLane v%s starting on %s:%d (%d models, auth=%s, region=%s)",
        __version__,
        config.server.host,
        config.server.port,
        len(config.models),
        config.auth.mode,
        config.region,
    )
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        log_config=None,
    )
