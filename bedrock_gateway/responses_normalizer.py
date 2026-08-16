"""
Compatibility normalization for Bedrock OpenAI Responses requests.

Bedrock mantle implements the OpenAI Responses surface, but its GPT-5.x request
validator accepts a narrower set of ``input`` item variants than some clients
(Codex in particular) emit. This module performs small, semantics-preserving
normalizations at the gateway boundary, before forwarding to Bedrock:

* ``additional_tools`` input items are lifted into top-level ``tools``.
* ``developer`` messages are folded into top-level ``instructions``.
* Codex ``reasoning.context`` objects are reduced to Bedrock's supported
  ``"auto"`` selector.
* Unknown encrypted reasoning/content blobs are dropped unless they carry a
  Bedrock-recognized ``rsn_`` or ``smry_`` prefix; reasoning replay summaries
  are normalized to Mantle's required array shape.
* Unsupported ``search_content_types`` filters are removed from ``web_search``
  tools while preserving the tool and its other options.

The normalizer is deliberately conservative: it only touches known incompatible
Codex extension shapes (``input`` plus the corresponding top-level ``tools`` /
``instructions`` destinations, and ``reasoning.context``), preserves
user/assistant/tool items, never rewrites text values, and is idempotent.
"""

from __future__ import annotations

from typing import Any

_DROP = object()
_BEDROCK_ENCRYPTED_PREFIXES = ("rsn_", "smry_")


def is_bedrock_gpt5x_responses_model(transport: str, dialect: str, model: str) -> bool:
    """Return True for Bedrock OpenAI GPT-5.x Responses models.

    Azure deployments and non-Responses dialects are intentionally excluded: the
    compatibility gap is observed on Bedrock mantle's GPT-5.x validator.
    """
    return (
        transport == "bedrock"
        and dialect == "openai-responses"
        and model.startswith("openai.gpt-5")
    )


def normalize_bedrock_gpt5x_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize Codex/OpenAI-extended Responses input for Bedrock GPT-5.x.

    Returns a shallow copy when a change is needed; otherwise returns a copy with
    equivalent content. The input object is never mutated.
    """
    out = dict(body)
    if "tools" in out:
        out["tools"] = _normalize_tools(out.get("tools"))
    input_value = body.get("input")
    if not isinstance(input_value, list):
        return out

    kept_input: list[Any] = []
    lifted_tools: list[Any] = []
    developer_texts: list[str] = []

    for item in input_value:
        if not isinstance(item, dict):
            kept_input.append(item)
            continue

        item_type = item.get("type")
        role = item.get("role")

        if item_type == "additional_tools":
            tools = item.get("tools")
            if isinstance(tools, list):
                lifted_tools.extend(tools)
            continue

        if item_type == "message" and role == "developer":
            text = _extract_text_from_content(item.get("content"))
            if text:
                developer_texts.append(text)
            continue

        kept_input.append(_normalize_input_item(item))

    if lifted_tools:
        out["tools"] = _merge_tools(out.get("tools"), lifted_tools)
    if "tools" in out:
        out["tools"] = _normalize_tools(out.get("tools"))
    if developer_texts:
        out["instructions"] = _merge_instructions(out.get("instructions"), developer_texts)
    if "reasoning" in out:
        out["reasoning"] = _normalize_reasoning(out.get("reasoning"))
    out = _filter_opaque_state(out)
    filtered_input = _filter_opaque_state(kept_input)
    out["input"] = [
        normalized
        for item in filtered_input
        if (normalized := _normalize_reasoning_replay_item(item)) is not _DROP
    ]
    return out


_BEDROCK_REASONING_CONTEXT_VALUES = {"auto", "current_turn", "all_turns"}


def _valid_bedrock_encrypted_content(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_BEDROCK_ENCRYPTED_PREFIXES)


def _valid_reasoning_summary(value: Any) -> bool:
    """Return whether a reasoning summary matches Mantle's replay schema."""
    return isinstance(value, list) and all(
        isinstance(block, dict)
        and block.get("type") == "summary_text"
        and isinstance(block.get("text"), str)
        for block in value
    )


def _filter_opaque_state(value: Any) -> Any:
    """Drop encrypted state blobs not minted by Bedrock.

    ``encrypted_content`` is opaque provider-private state. Bedrock GPT-5.x only
    recognizes blobs with ``rsn_`` or ``smry_`` prefixes; forwarding any other
    issuer's blob makes the whole request fail with a 400. This filter removes
    unrecognized blobs and drops now-empty reasoning items.
    """
    if isinstance(value, list):
        filtered = []
        for item in value:
            child = _filter_opaque_state(item)
            if child is not _DROP:
                filtered.append(child)
        return filtered
    if not isinstance(value, dict):
        return value

    out: dict[str, Any] = {}
    for key, child in value.items():
        if key == "encrypted_content":
            if _valid_bedrock_encrypted_content(child):
                out[key] = child
            continue
        filtered_child = _filter_opaque_state(child)
        if filtered_child is not _DROP:
            out[key] = filtered_child

    return out


def _normalize_reasoning_replay_item(item: Any) -> Any:
    """Validate only top-level Responses input reasoning items for replay."""
    if not isinstance(item, dict) or item.get("type") != "reasoning":
        return item
    if _valid_bedrock_encrypted_content(item.get("encrypted_content")):
        if _valid_reasoning_summary(item.get("summary")):
            return item
        out = dict(item)
        out["summary"] = []
        return out
    if _valid_reasoning_summary(item.get("summary")):
        return item
    return _DROP


def _normalize_reasoning(reasoning: Any) -> Any:
    """Normalize Codex reasoning context to Bedrock's accepted enum values."""
    if not isinstance(reasoning, dict):
        return reasoning
    out = dict(reasoning)
    context = out.get("context")
    if context is not None and (
        not isinstance(context, str) or context not in _BEDROCK_REASONING_CONTEXT_VALUES
    ):
        out["context"] = "auto"
    return out


def _normalize_input_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize known compatible message content while preserving item keys."""
    if "content" not in item:
        return dict(item)
    copy = dict(item)
    copy["content"] = _normalize_content(copy.get("content"))
    return copy


def _normalize_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    return [_normalize_content_block(block) for block in content]


def _normalize_content_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    out = dict(block)
    if out.get("type") == "text":
        out["type"] = "input_text"
    return out


def _extract_text_from_content(content: Any) -> str:
    """Extract developer instruction text without inventing new content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"input_text", "text"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif isinstance(block, str) and block:
            parts.append(block)
    return "\n".join(parts)


def _merge_instructions(existing: Any, developer_texts: list[str]) -> str:
    additions = [t for t in developer_texts if t]
    if isinstance(existing, str) and existing:
        return "\n\n".join([existing, *additions]) if additions else existing
    return "\n\n".join(additions)


def _tool_key(tool: Any) -> tuple[str, str]:
    """Best-effort stable key for tool de-duplication."""
    if not isinstance(tool, dict):
        return (type(tool).__name__, repr(tool))
    name = tool.get("name")
    if isinstance(name, str) and name:
        return (str(tool.get("type") or ""), name)
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return (str(tool.get("type") or "function"), fn["name"])
    return (str(tool.get("type") or ""), repr(sorted(tool.keys())))


def _normalize_tools(tools: Any) -> Any:
    """Remove Bedrock-unsupported options from Responses web search tools."""
    if not isinstance(tools, list):
        return tools
    normalized: list[Any] = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "web_search":
            copy = dict(tool)
            copy.pop("search_content_types", None)
            normalized.append(copy)
        else:
            normalized.append(tool)
    return normalized


def _merge_tools(existing: Any, lifted: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for tool in (existing if isinstance(existing, list) else []):
        key = _tool_key(tool)
        if key not in seen:
            merged.append(tool)
            seen.add(key)
    for tool in lifted:
        key = _tool_key(tool)
        if key not in seen:
            merged.append(tool)
            seen.add(key)
    return merged
