"""Redacted request-shape diagnostics for upstream schema errors."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import AuthConfig, GatewayConfig, RetryConfig, ServerConfig, _DEFAULT_MODELS, _parse_models
from bedrock_gateway.server import _request_shape_summary, create_app


@pytest.fixture
def client() -> TestClient:
    cfg = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models=_parse_models(_DEFAULT_MODELS),
    )
    return TestClient(create_app(cfg))


def test_request_shape_summary_redacts_text_and_tool_args():
    body = {
        "model": "gpt-5.6-sol",
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "SECRET USER TEXT"},
                    {"type": "input_text", "text": "MORE SECRET"},
                ],
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "SECRET TOOL RESULT"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "danger", "parameters": {"secret": "VALUE"}}}
        ],
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 32,
    }
    shape = _request_shape_summary(body)
    rendered = json.dumps(shape, ensure_ascii=False)
    assert "SECRET" not in rendered
    assert "VALUE" not in rendered
    assert shape["input"]["items"][0]["content"]["blocks"][0]["type"] == "text"
    assert shape["input"]["items"][0]["content"]["blocks"][1]["type"] == "input_text"
    assert shape["input"]["items"][1]["type"] == "function_call_output"
    assert shape["tools"]["tools"][0]["name"] == "danger"
    assert shape["reasoning"]["keys"] == ["effort"]
    assert shape["input"]["reasoning"] == {
        "total": 0, "valid_summary": 0, "invalid_or_missing_summary": 0,
        "encrypted_present": 0,
    }


def test_reasoning_shape_reports_structure_and_full_array_aggregate():
    input_items = [
        {"type": "message", "role": "user", "content": "safe"}
        for _ in range(20)
    ] + [
        {
            "type": "reasoning", "id": "SECRET-ID", "encrypted_content": "rsn_SECRET",
            "summary": [{"type": "summary_text", "text": "SECRET SUMMARY"}],
        },
        {"type": "reasoning", "summary": None},
        {"type": "reasoning", "summary": [{"type": "text", "text": "SECRET BAD"}, "SECRET"]},
    ]
    shape = _request_shape_summary({"model": "gpt-5.6-sol", "input": input_items})
    rendered = json.dumps(shape)
    assert shape["input"]["reasoning"] == {
        "total": 3, "valid_summary": 1, "invalid_or_missing_summary": 2,
        "encrypted_present": 1,
    }
    assert len(shape["input"]["items"]) == 20
    assert "SECRET" not in rendered


def test_reasoning_summary_shapes_do_not_include_text():
    body = {"input": [
        {"type": "reasoning"},
        {"type": "reasoning", "summary": "SECRET"},
        {"type": "reasoning", "summary": []},
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "SECRET TEXT"},
            {"type": "text", "text": "SECRET BAD"},
            "SECRET SCALAR",
        ]},
    ]}
    shape = _request_shape_summary(body)["input"]
    summaries = [item["summary"] for item in shape["items"]]
    assert summaries[0] == {"type": "null"}
    assert summaries[1] == {"type": "string", "len": 6}
    assert summaries[2] == {"type": "array", "len": 0, "block_types": [], "valid": True}
    assert summaries[3] == {
        "type": "array", "len": 3,
        "block_types": ["summary_text", "text", "string"], "valid": False,
    }
    assert "SECRET" not in json.dumps(shape)


def _mock_sync_error_client(status: int = 400):
    resp = MagicMock()
    resp.status_code = status
    resp.text = json.dumps({"error": {"message": "invalid request body: Invalid 'input'", "type": "invalid_request_error"}})
    resp.content = resp.text.encode()
    inst = AsyncMock()
    inst.post = AsyncMock(return_value=resp)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _mock_stream_error_client(status: int = 400):
    async def aiter_text():
        yield json.dumps({"error": {"message": "invalid request body: Invalid 'input'", "type": "invalid_request_error"}})

    resp = MagicMock()
    resp.status_code = status
    resp.aiter_text = aiter_text
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    inst = AsyncMock()
    inst.stream = MagicMock(return_value=ctx)
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_sync_upstream_error_logs_request_shape(mock_cls, client, caplog):
    mock_cls.return_value = _mock_sync_error_client()
    with caplog.at_level("WARNING", logger="bedrock_gateway"):
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.6-sol",
            "input": [{"role": "user", "content": [{"type": "text", "text": "DO NOT LOG"}]}],
        })
    assert resp.status_code == 400
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "REQ-SHAPE upstream_error" in text
    assert "gpt-5.6-sol" in text
    assert "DO NOT LOG" not in text
    assert "content" in text


@patch("bedrock_gateway.server.httpx.AsyncClient")
def test_stream_open_error_logs_request_shape(mock_cls, client, caplog):
    mock_cls.return_value = _mock_stream_error_client()
    with caplog.at_level("WARNING", logger="bedrock_gateway"):
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.6-sol",
            "stream": True,
            "input": [{"role": "user", "content": [{"type": "text", "text": "DO NOT LOG"}]}],
        })
    assert resp.status_code == 400
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "REQ-SHAPE upstream_error" in text
    assert "gpt-5.6-sol" in text
    assert "DO NOT LOG" not in text
