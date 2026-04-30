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
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .auth import AuthProvider
from .config import GatewayConfig, load_config
from .converter import (
    convert_tool_choice,
    convert_tools,
    convert_usage,
    decode_event_stream_chunk,
    extract_system_and_messages,
    format_anthropic_error,
    format_anthropic_response,
    make_anthropic_sse,
    make_stream_chunk,
    map_reasoning_effort,
    parse_bedrock_error,
    parse_bedrock_response,
)
from .dashboard import (
    DashboardAuth,
    MetricsCollector,
    RateLimiter,
    build_dashboard_router,
    metrics_middleware_factory,
)
from .dashboard.storage import MetricsStorage
from .models import ModelRegistry

logger = logging.getLogger("bedrock_gateway")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oai_error(status: int, message: str, etype: str = "api_error") -> JSONResponse:
    """Return an OpenAI-style error response."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "code": status}},
    )


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
        title="Bedrock Gateway",
        version=__version__,
        description="OpenAI-compatible proxy for AWS Bedrock",
    )

    registry = ModelRegistry(config)
    auth = AuthProvider(config.auth, config.region)
    bedrock_base = f"https://bedrock-runtime.{config.region}.amazonaws.com"
    max_retries = config.retry.max_retries
    retry_base_delay = config.retry.base_delay

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
    app.state.dashboard_auth = dashboard_auth
    app.state.dashboard_rate_limiter = dashboard_rate_limiter

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
    app.middleware("http")(metrics_middleware_factory(metrics))

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
    async def health() -> dict:
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
        model = registry.resolve(raw_model)
        stream = body.get("stream", False)

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
                model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay,
                request=request,
            )
        return await _handle_sync(
            model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay,
            request=request,
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
        model = registry.resolve(raw_model)
        stream = body.get("stream", False)

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
                request=request,
            )
        return await _handle_messages_sync(
            model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay,
            request=request,
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
# Sync handler
# ---------------------------------------------------------------------------

async def _handle_sync(
    model: str,
    bedrock_body: dict,
    bedrock_base: str,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    request: Request | None = None,
) -> dict | JSONResponse:
    url = f"{bedrock_base}/model/{model}/invoke"
    body_bytes = json.dumps(bedrock_body).encode()
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            headers = auth.get_headers(method="POST", url=url, body=body_bytes)
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(url, headers=headers, content=body_bytes)

            if resp.status_code == 200:
                result = resp.json()
                message, finish = parse_bedrock_response(result)
                usage = result.get("usage", {})
                logger.info(
                    "RES model=%s finish=%s in=%s out=%s attempt=%d",
                    model,
                    finish,
                    usage.get("input_tokens", "?"),
                    usage.get("output_tokens", "?"),
                    attempt + 1,
                )
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": convert_usage(usage),
                }

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
                continue

            error = parse_bedrock_error(resp.status_code, resp.text)
            logger.error(
                "ERR %d model=%s msg=%s",
                resp.status_code,
                model,
                error["message"][:300],
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

        except Exception as exc:
            return _oai_error(500, str(exc))

    logger.error(
        "FAILED model=%s all %d retries exhausted: %s",
        model,
        max_retries,
        last_error,
    )
    return _oai_error(502, f"All {max_retries} retries failed: {last_error}")


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------

async def _handle_stream(
    model: str,
    bedrock_body: dict,
    bedrock_base: str,
    auth: AuthProvider,
    max_retries: int,
    retry_base_delay: float,
    *,
    request: Request | None = None,
) -> StreamingResponse:
    url = f"{bedrock_base}/model/{model}/invoke-with-response-stream"
    body_bytes = json.dumps(bedrock_body).encode()
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def generate():  # noqa: C901
        for attempt in range(max_retries):
            try:
                headers = auth.get_headers(method="POST", url=url, body=body_bytes)
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream(
                        "POST", url, headers=headers, content=body_bytes
                    ) as resp:
                        if resp.status_code in (429, 529, 503):
                            _note_retry(request)
                            await asyncio.sleep(retry_base_delay * (2**attempt))
                            continue

                        if resp.status_code != 200:
                            err = ""
                            async for chunk in resp.aiter_text():
                                err += chunk
                            error = parse_bedrock_error(resp.status_code, err)
                            yield f'data: {json.dumps({"error": error})}\n\n'
                            yield "data: [DONE]\n\n"
                            return

                        buf = b""
                        stream_input_tokens = 0
                        stream_output_tokens = 0
                        current_tool_id: str | None = None
                        current_tool_name: str | None = None

                        async for raw in resp.aiter_bytes():
                            buf += raw
                            events, consumed = decode_event_stream_chunk(buf)
                            if consumed > 0:
                                buf = buf[consumed:]
                            for event in events:
                                etype = event.get("type", "")

                                if etype == "message_start":
                                    _mu = event.get("message", {}).get("usage", {})
                                    stream_input_tokens = _mu.get("input_tokens", 0)
                                    # Send initial role chunk (OpenAI spec)
                                    yield make_stream_chunk(
                                        msg_id, model, {"role": "assistant"}
                                    )

                                elif etype == "content_block_start":
                                    cb = event.get("content_block", {})
                                    if cb.get("type") == "tool_use":
                                        current_tool_id = cb.get("id", "")
                                        current_tool_name = cb.get("name", "")
                                        yield make_stream_chunk(
                                            msg_id,
                                            model,
                                            {
                                                "tool_calls": [{
                                                    "index": 0,
                                                    "id": current_tool_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": current_tool_name,
                                                        "arguments": "",
                                                    },
                                                }]
                                            },
                                        )
                                    elif cb.get("type") == "thinking":
                                        # Start of a thinking block — no output needed
                                        pass

                                elif etype == "content_block_delta":
                                    delta = event.get("delta", {})
                                    dtype = delta.get("type", "")
                                    if dtype == "text_delta":
                                        yield make_stream_chunk(
                                            msg_id,
                                            model,
                                            {"content": delta.get("text", "")},
                                        )
                                    elif dtype == "input_json_delta":
                                        partial = delta.get("partial_json", "")
                                        yield make_stream_chunk(
                                            msg_id,
                                            model,
                                            {
                                                "tool_calls": [{
                                                    "index": 0,
                                                    "function": {
                                                        "arguments": partial,
                                                    },
                                                }]
                                            },
                                        )
                                    elif dtype == "thinking_delta":
                                        yield make_stream_chunk(
                                            msg_id,
                                            model,
                                            {
                                                "reasoning_content": delta.get(
                                                    "thinking", ""
                                                )
                                            },
                                        )
                                    elif dtype == "signature_delta":
                                        # Signature associated with thinking block;
                                        # no user-visible output needed.
                                        pass

                                elif etype == "content_block_stop":
                                    current_tool_id = None
                                    current_tool_name = None

                                elif etype == "message_delta":
                                    sr = event.get("delta", {}).get(
                                        "stop_reason", "end_turn"
                                    )
                                    fr = "tool_calls" if sr == "tool_use" else "stop"
                                    _du = event.get("usage", {})
                                    if _du.get("output_tokens"):
                                        stream_output_tokens = _du["output_tokens"]
                                    if _du.get("input_tokens"):
                                        stream_input_tokens = _du["input_tokens"]
                                    yield make_stream_chunk(
                                        msg_id, model, {}, fr
                                    )
                                    # Send separate usage-only chunk (OpenAI stream_options format)
                                    _usage = {
                                        "prompt_tokens": stream_input_tokens,
                                        "completion_tokens": stream_output_tokens,
                                        "total_tokens": stream_input_tokens + stream_output_tokens,
                                    }
                                    yield f'data: {json.dumps({"id": msg_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [], "usage": _usage})}\n\n'



                yield "data: [DONE]\n\n"
                return

            except httpx.TimeoutException:
                _note_timeout(request)
                if attempt < max_retries - 1:
                    _note_retry(request)
                    await asyncio.sleep(retry_base_delay * (2**attempt))
                    continue
                yield f'data: {json.dumps({"error": {"message": "Timeout after retries"}})}\n\n'
                yield "data: [DONE]\n\n"
                return

            except Exception as exc:
                yield f'data: {json.dumps({"error": {"message": str(exc)}})}\n\n'
                yield "data: [DONE]\n\n"
                return

        yield f'data: {json.dumps({"error": {"message": "All retries failed"}})}\n\n'
        yield "data: [DONE]\n\n"

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
    request: Request | None = None,
) -> dict | JSONResponse:
    url = f"{bedrock_base}/model/{model}/invoke"
    body_bytes = json.dumps(bedrock_body).encode()
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            headers = auth.get_headers(method="POST", url=url, body=body_bytes)
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(url, headers=headers, content=body_bytes)

            if resp.status_code == 200:
                result = resp.json()
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
            logger.error(
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
    request: Request | None = None,
) -> StreamingResponse:
    url = f"{bedrock_base}/model/{model}/invoke-with-response-stream"
    body_bytes = json.dumps(bedrock_body).encode()
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    async def generate():  # noqa: C901
        for attempt in range(max_retries):
            try:
                headers = auth.get_headers(method="POST", url=url, body=body_bytes)
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream(
                        "POST", url, headers=headers, content=body_bytes
                    ) as resp:
                        if resp.status_code in (429, 529, 503):
                            _note_retry(request)
                            await asyncio.sleep(retry_base_delay * (2**attempt))
                            continue

                        if resp.status_code != 200:
                            err = ""
                            async for chunk in resp.aiter_text():
                                err += chunk
                            error = parse_bedrock_error(resp.status_code, err)
                            yield make_anthropic_sse(
                                "error",
                                {
                                    "type": "error",
                                    "error": {
                                        "type": error["type"],
                                        "message": error["message"],
                                    },
                                },
                            )
                            return

                        buf = b""
                        async for raw in resp.aiter_bytes():
                            buf += raw
                            events, consumed = decode_event_stream_chunk(buf)
                            if consumed > 0:
                                buf = buf[consumed:]
                            for event in events:
                                etype = event.get("type", "")

                                if etype == "message_start":
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

                return  # success, exit retry loop

            except httpx.TimeoutException:
                _note_timeout(request)
                if attempt < max_retries - 1:
                    _note_retry(request)
                    await asyncio.sleep(retry_base_delay * (2**attempt))
                    continue
                yield make_anthropic_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "Timeout after retries",
                        },
                    },
                )
                return

            except Exception as exc:
                yield make_anthropic_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": str(exc),
                        },
                    },
                )
                return

        # All retries exhausted
        yield make_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "All retries failed",
                },
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run(config: GatewayConfig | None = None) -> None:
    """Start the gateway server (blocking)."""
    if config is None:
        config = load_config()

    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = create_app(config)
    logger.info(
        "Bedrock Gateway v%s starting on %s:%d (%d models, auth=%s, region=%s)",
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
    )
