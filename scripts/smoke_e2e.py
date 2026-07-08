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

Usage:
  python scripts/smoke_e2e.py            # run all reachable checks
  python scripts/smoke_e2e.py --port 4199

Exit code 0 = all attempted checks passed; non-zero = at least one failed.
Checks whose credentials are absent are SKIPPED (not failed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  {mark} [{status}] {name}" + (f" — {detail}" if detail else ""))


def _post(url: str, body: dict, stream: bool = False, timeout: int = 60):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp.read().decode("utf-8", "replace")
    return json.loads(resp.read())


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


def check_grok(base: str) -> None:
    d = _post(base + "/openai/v1/responses",
              {"model": "grok-4.3", "input": "one word: ok"})
    assert d.get("status") == "completed", d
    record("bedrock grok-4.3 — responses sync", PASS)


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
        # NOTE: a ``models:`` section REPLACES the built-in defaults, so the
        # Bedrock defaults (claude/gpt-5.5/grok) must be re-listed here to keep
        # them available alongside the Azure additions.
        lines += [
            "azure_resources:",
            "  az_resp:",
            f"    base_url: {az_ep}?api-version=2025-04-01-preview",
            "    api_key: ${AZURE_OPENAI_KEY}",
            "  az_chat:",
            f"    base_url: {az_ep}/v1",
            "    api_key: ${AZURE_OPENAI_KEY}",
            "models:",
            # Bedrock defaults (re-listed because models: overrides defaults)
            "  claude-haiku:",
            "    bedrock_id: us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "  gpt-5.5:",
            "    bedrock_id: openai.gpt-5.5",
            "    endpoint: mantle",
            "    protocol: openai-responses",
            "  grok-4.3:",
            "    bedrock_id: xai.grok-4.3",
            "    endpoint: mantle",
            "    protocol: openai-responses",
            # Azure additions
            "  azure-gpt-5:",
            "    transport: azure",
            "    dialect: openai-responses",
            "    azure_resource: az_resp",
            "    deployment: gpt-5",
            "  azure-gpt-5-chat:",
            "    transport: azure",
            "    dialect: openai-chat",
            "    azure_resource: az_chat",
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
        for fn in (check_health, check_bedrock_claude, check_gpt55, check_grok,
                   check_guard_gpt55_on_chat):
            _run(fn, base)

        print("\n=== Azure ===")
        az_checks = (check_azure_responses, check_azure_responses_stream,
                     check_azure_chat, check_azure_chat_stream)
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
