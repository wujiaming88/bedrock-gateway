"""
OpenAI Images Generations provider — passthrough.

Azure OpenAI image deployments (for example ``gpt-image-2``) are exposed via
``/openai/v1/images/generations`` rather than the Responses API. The client
already speaks the native OpenAI Images dialect, so this dialect is an identity
map: request body, response body, and image payloads (``b64_json`` / URLs) flow
through unchanged except for the server-side model/deployment swap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Dialect

if TYPE_CHECKING:
    from ..config import ModelEntry


class ImagesPassthroughDialect(Dialect):
    """OpenAI Images Generations API, verbatim passthrough."""

    name = "openai-images"
    supports_stream = False

    def operation_path(self, entry: "ModelEntry", stream: bool) -> str:
        # Bare operation, relative to the OpenAI-compat API root. The transport
        # owns the root prefix (Azure's base already ends in /openai/v1).
        return "/images/generations"

    def build_request(self, client_body: dict, entry: "ModelEntry") -> dict:
        # Server already swapped model→upstream id; pure passthrough here.
        return client_body

    def render_sync(
        self, upstream_json: dict, model: str
    ) -> tuple[dict, dict]:
        log_info = {
            "input_tokens": "?",
            "output_tokens": "?",
            "finish": upstream_json.get("status", "completed"),
        }
        return upstream_json, log_info
