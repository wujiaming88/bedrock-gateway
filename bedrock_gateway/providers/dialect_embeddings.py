"""
OpenAI Embeddings dialect — the transport-side marker for ``/v1/embeddings``.

The *shape* half — OpenAI request parsing, capability validation, native body
building, and response rendering — lives in :mod:`bedrock_gateway.embeddings`
(the embeddings adapter layer). This dialect is deliberately minimal: it tells
the transport the upstream operation path (Bedrock's native ``invoke`` for the
Cohere / Titan embedding models) and declares the modality non-streaming.

The server's embeddings endpoint owns the orchestration: it resolves the model
entry (transport + dialect) *and* the embeddings adapter, then fans out the
adapter-built native bodies. As with the Anthropic dialect, ``build_request`` /
``render_sync`` here are identity shims — the adapter does the real shaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class EmbeddingsPassthroughDialect(Dialect):
    """OpenAI Embeddings API on Bedrock-native embedding models (Cohere/Titan)."""

    name = "openai-embeddings"
    supports_stream = False

    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        # Bedrock-native invoke path, like the Anthropic dialect. Cohere embed
        # and Titan are bedrock-runtime models, not mantle OpenAI-compat models.
        return f"/model/{entry.bedrock_id}/invoke"

    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        # Native request bodies are built by the embeddings adapter layer.
        return client_body

    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        # Unused: the endpoint renders via the embeddings adapter.
        return upstream_json, {
            "input_tokens": "?",
            "output_tokens": "?",
            "finish": "?",
        }
