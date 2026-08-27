#!/usr/bin/env python3
"""
End-to-end smoke test — starts a real gateway and calls every provider path
against the live upstreams, asserting on real responses.

This is NOT a unit test (it needs real credentials + network) and is excluded
from the pytest suite. Run it manually before a release or after touching the
transport/dialect layer.

Required env:
  AWS_BEARER_TOKEN_BEDROCK   Bedrock bearer token (Claude / GPT-5.5 / Grok)
  AZURE_OPENAI_KEY           Azure OpenAI resource api-key (optional)
  AZURE_OPENAI_ENDPOINT      Azure resource base incl. /openai (optional;
                             e.g. https://<res>.cognitiveservices.azure.com/openai)
  GATEWAY_API_KEY            Gateway ingress key (defaults to the Bedrock token)

Usage:
  python scripts/smoke_e2e.py            # run all reachable checks
  python scripts/smoke_e2e.py --port 4199

Exit code 0 = all attempted checks passed; non-zero = at least one failed.
Checks whose credentials are absent are SKIPPED (not failed).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error
import uuid
import zlib

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  {mark} [{status}] {name}" + (f" — {detail}" if detail else ""))


def _gateway_headers(content_type: str) -> dict[str, str]:
    headers = {"Content-Type": content_type}
    token = os.environ.get("GATEWAY_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post(url: str, body: dict, stream: bool = False, timeout: int = 60):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers=_gateway_headers("application/json"), method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp.read().decode("utf-8", "replace")
    return json.loads(resp.read())


def _multipart_post(url: str, fields: dict[str, str], files: list[tuple[str, str, str, bytes]], timeout: int = 300):
    boundary = f"----bedrock-gateway-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(), b"\r\n",
        ]
    for name, filename, content_type, content in files:
        chunks += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content, b"\r\n",
        ]
    chunks.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=b"".join(chunks),
        headers=_gateway_headers(f"multipart/form-data; boundary={boundary}"),
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.headers.get_content_type(), resp.read()


def _png(width: int = 256, height: int = 256) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + bytes([230, 40, 40]) * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _assert_image_response(raw: bytes) -> None:
    d = json.loads(raw)
    assert isinstance(d.get("created"), int), d
    assert isinstance(d.get("data"), list) and d["data"], d
    for item in d["data"]:
        encoded = item.get("b64_json")
        assert isinstance(encoded, str) and encoded, item
        image = base64.b64decode(encoded, validate=True)
        assert image.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"RIFF")), image[:12]


def _responses_text(d: dict) -> str:
    return "".join(
        c.get("text", "")
        for o in d.get("output", [])
        for c in o.get("content", [])
        if c.get("type") == "output_text"
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_health(base: str) -> None:
    d = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())
    assert d["status"] == "ok", d
    record(f"health (v{d.get('version')}, {d.get('models')} models)", PASS)


def check_bedrock_claude(base: str) -> None:
    d = _post(base + "/v1/chat/completions", {
        "model": "claude-haiku",
        "messages": [{"role": "user", "content": "reply one word: ok"}],
        "max_tokens": 10,
    })
    assert d["choices"][0]["message"]["content"], d
    record("bedrock claude — chat/completions sync", PASS)


def check_gpt55(base: str) -> None:
    d = _post(base + "/openai/v1/responses",
              {"model": "gpt-5.5", "input": "one word: ok"})
    assert d.get("status") == "completed", d
    record("bedrock gpt-5.5 — responses sync", PASS)


def check_gpt56_messages(base: str) -> None:
    d = _post(base + "/v1/messages", {
        "model": "gpt-5.6-sol",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "reply one word: ok"}],
    })
    assert d.get("type") == "message" and d.get("model") == "gpt-5.6-sol", d
    usage = d.get("usage") or {}
    assert all(key in usage for key in (
        "input_tokens", "output_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )), usage
    record("bedrock gpt-5.6 — messages sync", PASS)


def check_gpt56_messages_stream(base: str) -> None:
    raw = _post(base + "/v1/messages", {
        "model": "gpt-5.6-sol",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "reply one word: ok"}],
        "stream": True,
    }, stream=True)
    frames = [frame for frame in raw.replace("\r\n", "\n").split("\n\n") if frame]
    events = []
    for frame in frames:
        event = next((line[7:] for line in frame.splitlines() if line.startswith("event: ")), None)
        data = next((line[6:] for line in frame.splitlines() if line.startswith("data: ")), None)
        assert event and data, frame
        events.append((event, json.loads(data)))
    assert events[0][0] == "message_start" and events[-1][0] == "message_stop", events
    start = events[0][1]["message"]
    assert start["model"] == "gpt-5.6-sol", start
    delta = next(data for event, data in events if event == "message_delta")
    assert all(key in delta["usage"] for key in (
        "input_tokens", "output_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )), delta
    record("bedrock gpt-5.6 — messages stream", PASS)


def check_grok(base: str) -> None:
    for model in ("grok-4.3", "grok-4.6"):
        d = _post(base + "/openai/v1/responses",
                  {"model": model, "input": "one word: ok"})
        assert d.get("status") == "completed", d
        record(f"bedrock {model} — responses sync", PASS)


def check_azure_responses(base: str) -> None:
    d = _post(base + "/openai/v1/responses",
              {"model": "azure-gpt-5", "input": "one word: ok"})
    assert d.get("status") == "completed", d
    assert _responses_text(d), d
    record("azure gpt-5 — responses sync", PASS)


def check_azure_responses_stream(base: str) -> None:
    sse = _post(base + "/openai/v1/responses",
                {"model": "azure-gpt-5", "input": "count 1 to 3", "stream": True},
                stream=True)
    assert "response.completed" in sse, sse[:200]
    assert "response.output_text.delta" in sse, sse[:200]
    record("azure gpt-5 — responses stream", PASS)


def check_azure_chat(base: str) -> None:
    d = _post(base + "/v1/chat/completions", {
        "model": "azure-gpt-5-chat",
        "messages": [{"role": "user", "content": "reply one word: ok"}],
    })
    assert d["object"] == "chat.completion", d
    assert d["choices"][0]["message"]["content"], d
    record("azure gpt-5 — chat/completions sync", PASS)


def check_azure_chat_stream(base: str) -> None:
    sse = _post(base + "/v1/chat/completions", {
        "model": "azure-gpt-5-chat",
        "messages": [{"role": "user", "content": "count 1 to 3"}],
        "stream": True,
    }, stream=True)
    assert "[DONE]" in sse, sse[:200]
    assert '"delta"' in sse, sse[:200]
    record("azure gpt-5 — chat/completions stream", PASS)


def check_azure_image_edit(base: str) -> None:
    content_type, raw = _multipart_post(
        base + "/openai/v1/images/edits",
        {"model": "azure/gpt-image-2", "prompt": "change the square to blue", "n": "1"},
        [("image", "input.png", "image/png", _png(width=1024, height=1024))],
    )
    assert content_type == "application/json", content_type
    _assert_image_response(raw)
    record("azure gpt-image-2 — images/edits sync", PASS)


def check_azure_image_edit_stream(base: str) -> None:
    content_type, raw = _multipart_post(
        base + "/openai/v1/images/edits",
        {
            "model": "azure/gpt-image-2",
            "prompt": "change the square to green",
            "stream": "true",
            "partial_images": "1",
        },
        [("image[]", "input.png", "image/png", _png())],
    )
    assert content_type == "text/event-stream", content_type
    text = raw.decode("utf-8")
    frames = [frame for frame in text.replace("\r\n", "\n").split("\n\n") if frame]
    completed = 0
    partial_indices: list[int] = []
    for frame in frames:
        event_lines = [line[7:] for line in frame.splitlines() if line.startswith("event: ")]
        data_lines = [line[6:] for line in frame.splitlines() if line.startswith("data: ")]
        assert len(event_lines) == 1 and len(data_lines) == 1, frame[:200]
        payload = json.loads(data_lines[0])
        assert payload.get("type") == event_lines[0], payload
        encoded = payload.get("b64_json")
        if encoded:
            image = base64.b64decode(encoded, validate=True)
            assert image.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"RIFF"))
        if event_lines[0] == "image_edit.partial_image":
            partial_indices.append(payload["partial_image_index"])
        if event_lines[0] == "image_edit.completed":
            completed += 1
    assert partial_indices == sorted(set(partial_indices)), partial_indices
    assert completed == 1, text[-500:]
    record("azure gpt-image-2 — images/edits stream", PASS)


def check_guard_gpt55_on_chat(base: str) -> None:
    try:
        _post(base + "/v1/chat/completions",
              {"model": "gpt-5.5", "messages": [{"role": "user", "content": "x"}]})
    except urllib.error.HTTPError as e:
        assert e.code == 400, e.code
        record("guard — gpt-5.5 rejected on chat/completions (400)", PASS)
        return
    raise AssertionError("expected 400")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def build_config(port: int, tmp_path: str) -> tuple[str, dict]:
    """Write a smoke config; return (config_path, env)."""
    env = dict(os.environ)
    lines = [
        "auth:",
        "  mode: bearer_token",
        "  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}",
        "region: us-east-1",
        "server:",
        "  host: 127.0.0.1",
        f"  port: {port}",
        "  log_level: warning",
        "dashboard:",
        "  enabled: false",
    ]
    az_ep = os.environ.get("AZURE_OPENAI_ENDPOINT")
    az_key = os.environ.get("AZURE_OPENAI_KEY")
    if az_ep and az_key:
        # A single /openai/v1 resource serves both dialects (responses + chat);
        # no api-version needed. AZURE_OPENAI_ENDPOINT is the resource base up
        # to /openai. ``models:`` merges with the built-in Bedrock defaults.
        lines += [
            "azure_resources:",
            "  az:",
            f"    base_url: {az_ep}/v1",
            "    api_key: ${AZURE_OPENAI_KEY}",
            "    prefix: azure",
            "models:",
            "  azure-gpt-5:",
            "    transport: azure",
            "    dialect: openai-responses",
            "    azure_resource: az",
            "    deployment: gpt-5",
            "  azure-gpt-5-chat:",
            "    transport: azure",
            "    dialect: openai-chat",
            "    azure_resource: az",
            "    deployment: gpt-5",
        ]
    cfg = os.path.join(tmp_path, "smoke_config.yaml")
    with open(cfg, "w") as f:
        f.write("\n".join(lines) + "\n")
    return cfg, env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4199)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    have_bedrock = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    have_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT")
                      and os.environ.get("AZURE_OPENAI_KEY"))
    if not have_bedrock:
        print("AWS_BEARER_TOKEN_BEDROCK not set — cannot start gateway", file=sys.stderr)
        return 2

    cfg, env = build_config(args.port, "/tmp")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "from bedrock_gateway.config import load_config;"
         "from bedrock_gateway.server import run;"
         f"run(load_config({cfg!r}))"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # wait for startup
        for _ in range(20):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)

        print("\n=== Bedrock ===")
        for fn in (check_health, check_bedrock_claude, check_gpt55,
                   check_gpt56_messages, check_gpt56_messages_stream, check_grok,
                   check_guard_gpt55_on_chat):
            _run(fn, base)

        print("\n=== Azure ===")
        az_checks = (check_azure_responses, check_azure_responses_stream,
                     check_azure_chat, check_azure_chat_stream,
                     check_azure_image_edit, check_azure_image_edit_stream)
        if have_azure:
            for fn in az_checks:
                _run(fn, base)
        else:
            for fn in az_checks:
                record(fn.__name__, SKIP, "AZURE_* env not set")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"\n{'='*50}\nSMOKE: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


def _run(fn, base: str) -> None:
    try:
        fn(base)
    except Exception as exc:  # noqa: BLE001
        record(fn.__name__, FAIL, str(exc)[:120])


if __name__ == "__main__":
    sys.exit(main())
