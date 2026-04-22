"""
FastAPI server for Bedrock Gateway.

Exposes an OpenAI-compatible API that proxies requests to AWS Bedrock:
  - POST /v1/chat/completions  (sync + streaming)
  - GET  /v1/models
  - GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
    decode_event_stream_chunk,
    extract_system_and_messages,
    make_stream_chunk,
    parse_bedrock_response,
)
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

    # Store on app.state for testability
    app.state.config = config
    app.state.registry = registry
    app.state.auth = auth

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"object": "list", "data": registry.list_models()}

    # ------------------------------------------------------------------
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
        if thinking:
            bedrock_body["thinking"] = thinking
            bedrock_body.pop("temperature", None)

        if stream:
            return await _handle_stream(
                model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay
            )
        return await _handle_sync(
            model, bedrock_body, bedrock_base, auth, max_retries, retry_base_delay
        )

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
                    "usage": usage,
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
                await asyncio.sleep(delay)
                continue

            logger.error(
                "ERR %d model=%s body=%s",
                resp.status_code,
                model,
                resp.text[:300],
            )
            return _oai_error(resp.status_code, resp.text)

        except httpx.TimeoutException:
            last_error = "Request timeout"
            logger.warning(
                "TIMEOUT model=%s attempt=%d/%d",
                model,
                attempt + 1,
                max_retries,
            )
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
                            await asyncio.sleep(retry_base_delay * (2**attempt))
                            continue

                        if resp.status_code != 200:
                            err = ""
                            async for chunk in resp.aiter_text():
                                err += chunk
                            yield make_stream_chunk(
                                msg_id, model, {}, None
                            ).replace(
                                "data: ",
                                f'data: {json.dumps({"error": {"message": err, "code": resp.status_code}})}\n\ndata: ',
                                1,
                            )
                            yield "data: [DONE]\n\n"
                            return

                        buf = b""
                        processed = 0
                        current_tool_id: str | None = None
                        current_tool_name: str | None = None

                        async for raw in resp.aiter_bytes():
                            buf += raw
                            events = decode_event_stream_chunk(buf[processed:])
                            for event in events:
                                etype = event.get("type", "")

                                if etype == "content_block_start":
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

                                elif etype == "content_block_stop":
                                    current_tool_id = None
                                    current_tool_name = None

                                elif etype == "message_delta":
                                    sr = event.get("delta", {}).get(
                                        "stop_reason", "end_turn"
                                    )
                                    fr = "tool_calls" if sr == "tool_use" else "stop"
                                    yield make_stream_chunk(
                                        msg_id, model, {}, fr
                                    )

                            processed = len(buf)

                yield "data: [DONE]\n\n"
                return

            except httpx.TimeoutException:
                if attempt < max_retries - 1:
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
