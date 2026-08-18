"""
Pure compatibility rules for Bedrock mantle OpenAI Responses.

This is the single source of truth for the live-proven differences between what
OpenAI Responses clients (Codex in particular) emit and what the Bedrock mantle
GPT-5.x request validator accepts. It is deliberately **pure**: no I/O, no
mutable global state, no network, no logging of request values.

Three layers, all versioned together by :data:`MantleResponsesProfile`:

* :func:`is_exact_variant_rejection` — recognises the *deserialization-stage*
  HTTP 400 that Mantle returns for an unexpected ``input`` item/field variant
  (``agent_message``, ``local_shell_call``, ``code_interpreter_call``, a bare
  top-level ``input_text``, object-typed ``custom_tool_call.input`` /
  ``function_call.arguments``, an illegal ``custom_tool_call.status``, or a
  missing required field). Relationship errors ("no tool output / call found")
  are a *different* failure class with different text and are never eligible.
* :func:`project_mantle_input` — a copy-on-write, deterministic **safe
  projection** of the ``input`` array. It only repairs shapes that can be mapped
  losslessly; anything it cannot repair (side-effect items, orphan/unmatched
  calls, unknown visible semantics, missing required fields) marks the result
  ``safe_to_retry=False`` so the caller returns the original error unchanged.
* :func:`analyze_history` — a value-free structural fingerprint (item/field
  counts, call/output relations, payload/status enums) for redacted diagnostics.

The full eager normalizer (:func:`normalize_bedrock_gpt5x_responses_request`)
remains available for callers that want the historical whole-body rewrite, but
the server no longer applies it before the first request — the fixed behaviour
is "raw request first, one-time safe projection only after an exact variant
400". See the ``responses_normalizer`` facade module for the legacy imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PROFILE_VERSION = 1

_DROP = object()

_BEDROCK_ENCRYPTED_PREFIXES = ("rsn_", "smry_")
_BEDROCK_REASONING_CONTEXT_VALUES = frozenset({"auto", "current_turn", "all_turns"})

# Item-type taxonomy for the Responses ``input`` array.
_TEXT_ITEM_TYPES = frozenset({"input_text", "output_text", "text"})
_SIDE_EFFECT_ITEM_TYPES = frozenset({"local_shell_call", "code_interpreter_call"})
_CALL_ITEM_TYPES = frozenset({"function_call", "custom_tool_call"})
_OUTPUT_ITEM_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
_ACCEPTED_ITEM_TYPES = frozenset(
    {
        "message",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "reasoning",
    }
)

# ``custom_tool_call`` statuses Mantle accepts on *replayed* calls. Any other
# status is pure output metadata (the call already happened; its result is the
# paired output item) and is removed without changing call semantics.
_CUSTOM_STATUS_KEEP = frozenset({"completed", "in_progress"})


@dataclass(frozen=True)
class MantleResponsesProfile:
    """Versioned declaration of the live-proven mantle Responses differences.

    No I/O: this is an immutable in-code profile. A schema drift gate (live
    probe) can bump :data:`PROFILE_VERSION` and update these tables together.
    """

    version: int = PROFILE_VERSION
    accepted_item_types: frozenset = _ACCEPTED_ITEM_TYPES
    text_item_types: frozenset = _TEXT_ITEM_TYPES
    side_effect_item_types: frozenset = _SIDE_EFFECT_ITEM_TYPES
    call_item_types: frozenset = _CALL_ITEM_TYPES
    output_item_types: frozenset = _OUTPUT_ITEM_TYPES
    custom_status_keep: frozenset = _CUSTOM_STATUS_KEEP
    # Exact-variant rejection signatures (deserialization-stage 400s). Only
    # these, combined with HTTP 400, may ever trigger a projection fallback.
    exact_variant_signatures: tuple[str, ...] = (
        "invalid 'input': value did not match any expected variant",
    )
    # Relationship-error signatures. These must NEVER trigger a fallback — they
    # are a post-deserialization failure class whose text differs.
    relationship_signatures: tuple[str, ...] = (
        "no tool output",
        "no tool call",
        "tool output without",
        "tool call without",
    )


MANTLE_RESPONSES_PROFILE = MantleResponsesProfile()


@dataclass(frozen=True)
class CompatibilityPolicy:
    """Internal (non-user) policy arming the one-time exact-variant fallback.

    Derived automatically from ``(transport=bedrock, dialect=openai-responses,
    upstream model=openai.gpt-5*)`` — never a user switch.
    """

    profile: MantleResponsesProfile = MANTLE_RESPONSES_PROFILE


@dataclass(frozen=True)
class ProjectionResult:
    """Outcome of :func:`project_mantle_input`.

    ``body`` is the projected request (a copy; the input is never mutated).
    Diagnostics carry only types, field presence, counts and relation/decision
    categories — never text, arguments, outputs, ids, blobs or headers.
    """

    body: dict[str, Any]
    changed: bool
    safe_to_retry: bool
    item_counts: dict[str, int]
    decisions: dict[str, int]
    relation_counts: dict[str, int]
    unsafe_reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Rejection classification
# ---------------------------------------------------------------------------


def is_exact_variant_rejection(status: int, error_text: str | None) -> bool:
    """Return True iff this is a mantle deserialization-stage 400.

    Only HTTP 400 carrying a declared exact-variant signature qualifies, and a
    relationship-error signature always wins (returns False) even when it shares
    the generic "Invalid 'input'" prefix.
    """
    if status != 400 or not error_text:
        return False
    text = error_text.lower()
    profile = MANTLE_RESPONSES_PROFILE
    if any(sig in text for sig in profile.relationship_signatures):
        return False
    return any(sig in text for sig in profile.exact_variant_signatures)


def is_bedrock_gpt5x_responses_model(transport: str, dialect: str, model: str) -> bool:
    """Return True for Bedrock mantle GPT-5.x Responses models.

    Azure deployments, generic HTTP upstreams and non-Responses dialects are
    excluded: the compatibility gap is observed only on Bedrock mantle's GPT-5.x
    validator.
    """
    return (
        transport == "bedrock"
        and dialect == "openai-responses"
        and model.startswith("openai.gpt-5")
    )


def responses_compat_policy(
    transport: str, dialect: str, model: str
) -> CompatibilityPolicy | None:
    """Return the fallback policy for Bedrock GPT-5.x native Responses, else None."""
    if is_bedrock_gpt5x_responses_model(transport, dialect, model):
        return CompatibilityPolicy()
    return None


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _count(d: dict[str, int], key: Any) -> None:
    d[key] = d.get(key, 0) + 1


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _item_type(item: dict[str, Any]) -> str:
    itype = item.get("type")
    if itype is not None:
        return str(itype)
    if "role" in item:
        return "message"
    return "unknown"


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


def _stable_json_string(value: Any) -> str:
    """Stable compact JSON: UTF-8, sorted keys."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_json_string(value: Any) -> Any:
    """Encode a JSON container/scalar to a stable compact string; strings as-is."""
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float, list, dict)):
        return _stable_json_string(value)
    return value


def _strip_foreign_opaque(value: Any) -> tuple[Any, bool]:
    """Drop ``encrypted_content`` not minted by Bedrock, recursively.

    Returns ``(value, changed)``; returns the *same* object when nothing changed
    so callers can detect no-op projections cheaply.
    """
    if isinstance(value, list):
        out: list[Any] = []
        changed = False
        for child in value:
            stripped, c = _strip_foreign_opaque(child)
            out.append(stripped)
            changed = changed or c
        return (value, False) if not changed else (out, True)
    if not isinstance(value, dict):
        return value, False
    out_dict: dict[str, Any] = {}
    changed = False
    for key, child in value.items():
        if key == "encrypted_content":
            if _valid_bedrock_encrypted_content(child):
                out_dict[key] = child
            else:
                changed = True
            continue
        stripped, c = _strip_foreign_opaque(child)
        out_dict[key] = stripped
        changed = changed or c
    return (value, False) if not changed else (out_dict, True)


# ---------------------------------------------------------------------------
# History analysis (value-free fingerprint)
# ---------------------------------------------------------------------------

def analyze_history(items: Any) -> dict[str, Any]:
    """Return a value-free structural fingerprint of a Responses ``input`` array.

    Records only types, field-name sets, counts and relation categories — no
    text, arguments, outputs, ids, blobs or headers.
    """
    result: dict[str, Any] = {
        "item_type_counts": {},
        "field_fingerprints": {},
        "relations": {
            "calls": 0,
            "outputs": 0,
            "matched": 0,
            "unmatched_calls": 0,
            "orphan_outputs": 0,
            "duplicate_call_ids": 0,
        },
        "payload_types": {},
        "status_enums": {},
        "unknown_item_types": {"count": 0, "first_index": None},
    }
    if not isinstance(items, list):
        return result

    call_ids: list[str] = []
    output_ids: list[str] = []
    call_by_id: dict[str, int] = {}
    output_by_id: dict[str, int] = {}

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            _count(result["item_type_counts"], "scalar")
            continue
        itype = _item_type(item)
        _count(result["item_type_counts"], itype)
        fingerprint = ",".join(sorted(str(k) for k in item.keys()))
        _count(result["field_fingerprints"].setdefault(itype, {}), fingerprint)

        if itype in _CALL_ITEM_TYPES:
            rel = result["relations"]
            rel["calls"] += 1
            cid = item.get("call_id")
            if isinstance(cid, str) and cid:
                call_ids.append(cid)
                call_by_id[cid] = call_by_id.get(cid, 0) + 1
            for key in ("input", "arguments"):
                if key in item:
                    _count(result["payload_types"], _json_type(item.get(key)))
            if itype == "custom_tool_call" and "status" in item:
                _count(result["status_enums"], str(item.get("status")))
        elif itype in _OUTPUT_ITEM_TYPES:
            result["relations"]["outputs"] += 1
            cid = item.get("call_id")
            if isinstance(cid, str) and cid:
                output_ids.append(cid)
                output_by_id[cid] = output_by_id.get(cid, 0) + 1
            if "output" in item:
                _count(result["payload_types"], _json_type(item.get("output")))
        elif itype == "unknown":
            unknown = result["unknown_item_types"]
            unknown["count"] += 1
            if unknown["first_index"] is None:
                unknown["first_index"] = idx

    rel = result["relations"]
    call_set = set(call_ids)
    output_set = set(output_ids)
    for cid in call_set:
        if cid in output_set:
            if call_by_id[cid] == 1 and output_by_id[cid] == 1:
                rel["matched"] += 1
            else:
                rel["duplicate_call_ids"] += 1
        else:
            rel["unmatched_calls"] += 1
    for cid in output_set:
        if cid not in call_set:
            rel["orphan_outputs"] += 1

    return result


# ---------------------------------------------------------------------------
# Safe projection
# ---------------------------------------------------------------------------

def project_mantle_input(body: dict[str, Any]) -> ProjectionResult:
    """Project a Responses body's ``input`` for mantle GPT-5.x, copy-on-write.

    Returns a :class:`ProjectionResult` whose ``body`` is a new dict (input never
    mutated). ``safe_to_retry`` is True only when at least one lossless repair
    was made AND no unsafe item/relation remains; the caller must otherwise
    return the original upstream error unchanged.
    """
    input_value = body.get("input")
    if not isinstance(input_value, list):
        return ProjectionResult(
            body=dict(body),
            changed=False,
            safe_to_retry=False,
            item_counts={},
            decisions={},
            relation_counts={},
            unsafe_reasons=("input_not_list",),
        )

    analysis = analyze_history(input_value)
    projected_items: list[Any] = []
    decisions: dict[str, int] = {}
    unsafe_reasons: list[str] = []
    changed = False

    for item in input_value:
        new_item, decision, unsafe = _project_item(item)
        if decision is not None:
            _count(decisions, decision)
        if unsafe is not None:
            unsafe_reasons.append(unsafe)
        if new_item is _DROP:
            changed = True
            continue
        if new_item is not item:
            changed = True
        projected_items.append(new_item)

    rel = analysis["relations"]
    if rel["unmatched_calls"]:
        unsafe_reasons.append("unmatched_call")
    if rel["orphan_outputs"]:
        unsafe_reasons.append("orphan_output")
    if rel["duplicate_call_ids"]:
        unsafe_reasons.append("duplicate_call_id")

    out = dict(body)
    out["input"] = projected_items
    return ProjectionResult(
        body=out,
        changed=changed,
        safe_to_retry=changed and not unsafe_reasons,
        item_counts=dict(analysis["item_type_counts"]),
        decisions=decisions,
        relation_counts=dict(rel),
        unsafe_reasons=tuple(dict.fromkeys(unsafe_reasons)),
    )


def _project_item(item: Any) -> tuple[Any, str | None, str | None]:
    """Project one input item.

    Returns ``(new_item, decision, unsafe_reason)``. ``decision`` is a stable
    category token (change or unsafe) or None for a no-op; ``unsafe_reason`` is
    a value-free reason or None. ``_DROP`` means the item is removed.
    """
    if not isinstance(item, dict):
        return item, None, None
    itype = item.get("type")
    role = item.get("role")

    if itype in _TEXT_ITEM_TYPES:
        return _project_bare_text(item)
    if itype == "agent_message":
        return _project_agent_message(item)
    if itype in _SIDE_EFFECT_ITEM_TYPES:
        return item, "unsafe_side_effect", f"side_effect:{itype}"
    if itype in _CALL_ITEM_TYPES:
        return _project_call(item)
    if itype in _OUTPUT_ITEM_TYPES:
        return _project_output(item)
    if itype == "reasoning":
        return _project_reasoning(item)
    if itype == "message" or (itype is None and role is not None):
        return _project_message(item)
    return item, "unsafe_unknown", f"unknown:{itype}"


def _project_bare_text(item: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    text = item.get("text")
    if not isinstance(text, str):
        return item, "unsafe_bare_text", "bare_text_non_string"
    role = "assistant" if item.get("type") == "output_text" else "user"
    message = {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }
    return message, "wrap_text_item", None


def _project_agent_message(item: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    """Map ``agent_message`` to a stable user message, preserving visible text.

    Unsafe when any visible block is an image, binary or unknown block — content
    is never dropped to "improve" a retry.
    """
    text = item.get("text")
    content = item.get("content")
    if isinstance(text, str):
        return _agent_user_message([{"type": "input_text", "text": text}])
    if isinstance(content, str):
        return _agent_user_message([{"type": "input_text", "text": content}])
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "input_text", "text": block})
                continue
            if isinstance(block, dict) and block.get("type") in {
                "input_text", "output_text", "text",
            }:
                t = block.get("text")
                if isinstance(t, str):
                    blocks.append({"type": "input_text", "text": t})
                    continue
                return item, "unsafe_agent_message", "agent_message_non_text_block"
            return item, "unsafe_agent_message", "agent_message_unmappable_block"
        if not blocks:
            return item, "unsafe_agent_message", "agent_message_no_visible_text"
        return _agent_user_message(blocks)
    return item, "unsafe_agent_message", "agent_message_unidentifiable"


def _agent_user_message(blocks: list[dict[str, Any]]) -> tuple[Any, str | None, str | None]:
    return {"type": "message", "role": "user", "content": blocks}, "agent_message_to_user", None


def _project_call(item: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    itype = item.get("type")
    payload_key = "input" if itype == "custom_tool_call" else "arguments"
    call_id = item.get("call_id")
    name = item.get("name")
    if not isinstance(call_id, str) or not call_id:
        return item, "unsafe_missing_required", f"{itype}_missing_call_id"
    if not isinstance(name, str) or not name:
        return item, "unsafe_missing_required", f"{itype}_missing_name"
    if payload_key not in item or item.get(payload_key) is None:
        return item, "unsafe_missing_required", f"{itype}_missing_{payload_key}"

    out = dict(item)
    decisions: list[str] = []
    coerced = _coerce_json_string(item.get(payload_key))
    if coerced is not item.get(payload_key):
        out[payload_key] = coerced
        decisions.append("coerce_input" if payload_key == "input" else "coerce_arguments")
    if itype == "custom_tool_call" and "status" in out:
        if out.get("status") not in _CUSTOM_STATUS_KEEP:
            out.pop("status")
            decisions.append("drop_status")

    if not decisions:
        return item, None, None
    return out, "_".join(decisions), None


def _project_output(item: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    itype = item.get("type")
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return item, "unsafe_missing_required", f"{itype}_missing_call_id"
    if "output" not in item or item.get("output") is None:
        return item, "unsafe_missing_required", f"{itype}_missing_output"
    return item, None, None


def _project_reasoning(item: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    """Apply the live-proven reasoning summary/opaque rules.

    Foreign opaque state is dropped; an id-only reasoning item (no Bedrock state
    and no valid summary) is dropped; a Bedrock-prefixed opaque item with an
    invalid summary gets an empty summary. These rules are shared with the
    historical eager normalizer and do not touch visible text.
    """
    filtered, opaque_changed = _strip_foreign_opaque(item)
    if _valid_bedrock_encrypted_content(filtered.get("encrypted_content")):
        if _valid_reasoning_summary(filtered.get("summary")):
            if opaque_changed:
                return filtered, "drop_foreign_opaque", None
            return item, None, None
        out = dict(filtered)
        out["summary"] = []
        return out, "set_reasoning_summary", None
    if _valid_reasoning_summary(filtered.get("summary")):
        if opaque_changed:
            return filtered, "drop_foreign_opaque", None
        return item, None, None
    return _DROP, "drop_reasoning_item", None


def _project_message(item: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    filtered, changed = _strip_foreign_opaque(item)
    if changed:
        return filtered, "drop_foreign_opaque", None
    return item, None, None


# ---------------------------------------------------------------------------
# Historical eager normalizer (kept for compatibility; no longer eager-called)
# ---------------------------------------------------------------------------

def normalize_bedrock_gpt5x_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize Codex/OpenAI-extended Responses input for Bedrock GPT-5.x.

    Historical whole-body rewrite. The server no longer calls this before the
    first request — it is retained as a public pure function (and the source of
    the reasoning/opaque/developer/additional-tools rules) for callers and tests.
    Returns a shallow copy when a change is needed; the input is never mutated.
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


def _filter_opaque_state(value: Any) -> Any:
    """Drop encrypted state blobs not minted by Bedrock (facade over stripper)."""
    return _strip_foreign_opaque(value)[0]


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


__all__ = [
    "PROFILE_VERSION",
    "MantleResponsesProfile",
    "MANTLE_RESPONSES_PROFILE",
    "CompatibilityPolicy",
    "ProjectionResult",
    "is_exact_variant_rejection",
    "is_bedrock_gpt5x_responses_model",
    "responses_compat_policy",
    "analyze_history",
    "project_mantle_input",
    "normalize_bedrock_gpt5x_responses_request",
]
