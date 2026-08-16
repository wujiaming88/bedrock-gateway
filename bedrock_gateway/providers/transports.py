"""
Transports — the *where + how to authenticate* axis.

  * :class:`BedrockTransport` — AWS Bedrock hosts (``bedrock-runtime`` for the
    Anthropic dialect, ``bedrock-mantle`` for OpenAI dialects). Auth is handled
    by the gateway's global :class:`AuthProvider` (SigV4 / Bearer), so
    ``auth_headers`` returns ``None``.
  * :class:`AzureTransport` — a per-resource Azure OpenAI endpoint with an
    ``api-key`` header. The endpoint (incl. any ``api-version`` query string)
    is resolved onto ``entry.azure_endpoint`` by the config loader.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import Transport

if TYPE_CHECKING:
    from ..config import ModelEntry


class BedrockTransport(Transport):
    """AWS Bedrock. Host + API root depend on the ``endpoint`` hint:

      * ``mantle`` → ``bedrock-mantle.{region}.api.aws`` serving the
        OpenAI-compatible API under ``/openai/v1`` (Responses / Chat dialects).
      * ``runtime`` (default) → ``bedrock-runtime.{region}.amazonaws.com`` with
        Bedrock's native ``/model/{id}/...`` paths (Anthropic dialect, whose
        ``operation_path`` already carries the full native path).

    ``operation_path`` from OpenAI-compat dialects is a bare operation
    (``/responses``); this transport prepends the ``/openai/v1`` root. The
    Anthropic dialect returns its full native path, which is used as-is.
    Auth is the gateway global (SigV4 / Bearer), so ``auth_headers`` is None.
    """

    name = "bedrock"

    def build_url(
        self, operation_path: str, region: str, entry: "ModelEntry"
    ) -> str:
        if entry.endpoint == "mantle":
            # OpenAI-compatible surface: dialect gives a bare op, we add the root.
            return (
                f"https://bedrock-mantle.{region}.api.aws/openai/v1"
                + operation_path
            )
        # runtime: native Bedrock path, already complete from the dialect.
        return f"https://bedrock-runtime.{region}.amazonaws.com" + operation_path


class HttpTransport(Transport):
    """Generic HTTP upstream configured entirely by resolved model fields."""

    name = "http"

    def build_url(
        self, operation_path: str, region: str, entry: "ModelEntry"
    ) -> str:
        # The configured route owns its path. ``operation_path`` only selects the
        # route through the dialect and must not impose provider-specific URLs.
        return f"{entry.upstream_base_url.rstrip('/')}{entry.upstream_path}"

    def auth_headers(self, entry: "ModelEntry") -> dict[str, str]:
        secret = os.environ.get(entry.upstream_secret_env, "")
        if not secret:
            raise RuntimeError(
                f"upstream credential environment variable "
                f"{entry.upstream_secret_env!r} is not set"
            )
        headers = dict(entry.upstream_default_headers)
        if entry.upstream_auth == "bearer":
            headers["Authorization"] = f"Bearer {secret}"
        elif entry.upstream_auth == "x-api-key":
            headers["x-api-key"] = secret
        return headers


class AzureTransport(Transport):
    """Azure OpenAI. Per-resource endpoint + ``api-key`` header.

    ``entry.azure_endpoint`` is the resource base (up to ``/openai``), possibly
    already carrying an ``api-version`` query string on some operations. The
    dialect's ``operation_path`` is appended; if the base already has a query
    string it is preserved by splitting before appending the path.
    """

    name = "azure"

    def build_url(
        self, operation_path: str, region: str, entry: "ModelEntry"
    ) -> str:
        base = entry.azure_endpoint
        # Split off any pre-existing query string on the base, re-attach after
        # the operation path so ?api-version=... stays at the end.
        if "?" in base:
            base_path, query = base.split("?", 1)
            return f"{base_path.rstrip('/')}{operation_path}?{query}"
        return f"{base.rstrip('/')}{operation_path}"

    def auth_headers(self, entry: "ModelEntry") -> dict[str, str]:
        return {
            "api-key": entry.azure_api_key,
            "Content-Type": "application/json",
        }
