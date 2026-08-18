"""
Compatibility normalization facade for Bedrock OpenAI Responses requests.

The rules live in :mod:`bedrock_gateway.responses_compatibility` (a pure,
versioned profile). This module re-exports the two historical public entry
points for backward compatibility; the server no longer applies the eager
normalizer before the first request — the fixed behaviour is "raw request
first, one-time safe projection only after an exact variant 400".
"""

from __future__ import annotations

from .responses_compatibility import (
    is_bedrock_gpt5x_responses_model,
    normalize_bedrock_gpt5x_responses_request,
)

__all__ = [
    "is_bedrock_gpt5x_responses_model",
    "normalize_bedrock_gpt5x_responses_request",
]
