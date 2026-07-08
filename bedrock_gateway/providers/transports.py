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

from typing import TYPE_CHECKING

from .base import Transport

if TYPE_CHECKING:
    from ..config import ModelEntry


class BedrockTransport(Transport):
    """AWS Bedrock. Host depends on the dialect's endpoint hint
    (``runtime`` → ``bedrock-runtime``; ``mantle`` → ``bedrock-mantle``).
    Auth is the gateway global (SigV4 / Bearer), so ``auth_headers`` is None.
    """

    name = "bedrock"

    def build_url(
        self, operation_path: str, region: str, entry: "ModelEntry"
    ) -> str:
        if entry.endpoint == "mantle":
            host = f"https://bedrock-mantle.{region}.api.aws"
        else:  # "runtime" (default)
            host = f"https://bedrock-runtime.{region}.amazonaws.com"
        return host + operation_path


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
