"""
Pure, generic remediation for upstream "unsupported field/tool" 400s.

This is the single source of truth for **Class A** divergence: a model's
OpenAI-compatible validator rejects a whole field it does not implement, naming
it precisely in the error text. Two upstream signatures are recognised:

* Bedrock mantle — ``Unsupported parameter: 'X'``, where ``X`` is the full
  dotted path, e.g. ::

      Unsupported parameter: 'reasoning.summary' is not supported with the
      'openai.gpt-6-astra' model.

  or a field that was merely renamed across OpenAI API generations::

      Unsupported parameter: 'max_tokens' is not supported with this model.

* Generic HTTP upstreams (ark / Volcengine, etc.) — Go's encoding/json error,
  where ``X`` is the **leaf key only**, e.g. ::

      json: unknown field "summary"

  Here the error names just the leaf, not its parent, so remediation must locate
  the key by a depth-first search rather than a fixed dotted path.

* Tool-type rejection — the model names a tool *type* it does not implement, e.g. ::

      Tool type 'web_search' is not supported for model `xai.grok-4.7`.

  (Grok 4.7 / Kimi K3 reject the Responses ``web_search`` server tool this way.)
  Here the named thing is an entry in ``body["tools"]``, not a field, so
  remediation drops every ``{"type": "web_search"}`` tool rather than a field.

Because the upstream names the exact field/tool it rejects, removing (or, for a
small lossless rename table, renaming) that field is inherently safe: the request
was already going to 400, and a field/tool the model does not implement cannot
carry semantics for it. This module is deliberately **pure** — no I/O, no mutable
global state, no logging of request values — mirroring
:mod:`responses_compatibility` (which handles **Class B**, ``Invalid 'input'``
shape divergence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lossless renames: the client used an older OpenAI name and the model only knows
# the newer one. Keyed by the full dotted field path named in the 400. The rename
# preserves the value; it is only a name change across API generations.
_RENAMES: dict[str, str] = {
    "max_tokens": "max_completion_tokens",
}

# Upper bound on how many distinct unsupported fields may be stripped/renamed in
# one request. Each iteration removes exactly one more field, so a chain of
# unsupported fields converges within this bound instead of looping.
MAX_UNSUPPORTED_STRIPS = 3

# Bedrock mantle names the full dotted path.
_UNSUPPORTED_PARAM_RE = re.compile(r"Unsupported parameter:\s*'([^']+)'", re.IGNORECASE)

# Go's encoding/json (ark / Volcengine) names only the leaf key. The upstream
# error body is JSON, so the field name may arrive quoted (decoded) or
# backslash-escaped (raw ``resp.text``); accept both.
_UNKNOWN_FIELD_RE = re.compile(r'json:\s*unknown field\s+\\?"([^"\\]+)\\?"', re.IGNORECASE)

# Tool-type rejection (Grok / Kimi K3 Responses): ``Tool type 'web_search' is not
# supported for model ...``. The named string is a ``body["tools"]`` entry type,
# not a field. Matches regardless of the surrounding model/`this model` wording.
_UNSUPPORTED_TOOL_RE = re.compile(r"tool type\s+'([^']+)'\s+is not supported", re.IGNORECASE)


@dataclass(frozen=True)
class UnsupportedParam:
    """A single thing the upstream named as unsupported.

    ``kind`` discriminates the remediation shape:

    * ``"field"`` — a field the validator rejects. ``field_path`` is the path
      into the request body — either the full dotted path (``reasoning.summary``,
      from Bedrock mantle) or a bare leaf key (``summary``, from Go's ``unknown
      field`` error). ``action`` is ``"drop"`` or ``"rename"``; ``rename_to`` is
      the replacement leaf key for a rename, else ``None``.
    * ``"tool"`` — a tool *type* the model rejects (``Tool type 'web_search' is
      not supported``). ``field_path`` carries the tool type; ``action`` is
      always ``"drop"`` and ``rename_to`` is ``None``. Remediation removes every
      ``{"type": tool_type}`` entry from ``body["tools"]``.
    """

    field_path: str
    action: str
    rename_to: str | None
    kind: str = "field"


def parse_unsupported_param(error_text: str | None) -> UnsupportedParam | None:
    """Parse an upstream 400 for the field/tool it declared unsupported.

    Recognises the Bedrock mantle ``Unsupported parameter: 'X'`` signature, the
    generic-HTTP ``json: unknown field "X"`` signature, and the tool-type
    ``Tool type 'X' is not supported`` signature, choosing ``rename`` when a
    field ``X`` is in :data:`_RENAMES` and ``drop`` otherwise (a tool type is
    always ``drop``). Returns ``None`` for anything else — non-400 text, the
    ``Invalid 'input'`` variant rejection, relationship errors, or no match — so
    callers only ever act on a precisely named field/tool.
    """
    if not error_text:
        return None
    match = _UNSUPPORTED_PARAM_RE.search(error_text)
    if match is None:
        match = _UNKNOWN_FIELD_RE.search(error_text)
    if match is not None:
        field_path = match.group(1)
        rename_to = _RENAMES.get(field_path)
        if rename_to is not None:
            return UnsupportedParam(
                field_path=field_path, action="rename", rename_to=rename_to
            )
        return UnsupportedParam(field_path=field_path, action="drop", rename_to=None)
    tool_match = _UNSUPPORTED_TOOL_RE.search(error_text)
    if tool_match is not None:
        return UnsupportedParam(
            field_path=tool_match.group(1), action="drop", rename_to=None, kind="tool"
        )
    return None


def _mutate_leaf(node: dict, leaf: str, action: str, rename_to: str | None) -> None:
    """Drop/rename ``leaf`` **in place** on ``node`` (which must be a copy)."""
    if action == "rename" and rename_to is not None:
        if rename_to in node:
            del node[leaf]
        else:
            node[rename_to] = node.pop(leaf)
    else:  # drop
        del node[leaf]


def _apply_at_path(
    body: dict, parts: list[str], action: str, rename_to: str | None
) -> tuple[dict, bool]:
    """Drop/rename along an exact dotted path, copy-on-write.

    Returns ``(new_body, changed)``; ``changed`` is ``False`` when the path does
    not resolve to a present key (a non-dict intermediate or missing leaf).
    """
    out = dict(body)
    node = out
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            return body, False
        copied = dict(child)
        node[part] = copied
        node = copied
    leaf = parts[-1]
    if leaf not in node:
        return body, False
    _mutate_leaf(node, leaf, action, rename_to)
    return out, True


def _apply_leaf(
    body: dict, leaf: str, action: str, rename_to: str | None
) -> tuple[dict, bool]:
    """Drop/rename ``leaf`` wherever it first appears, copy-on-write.

    Depth-first, left-to-right, top-level match wins. Each ancestor along the
    found path is copied, so the original ``body`` is never mutated. Returns
    ``(new_body, changed)``; ``changed`` is ``False`` when ``leaf`` is absent.
    """
    if leaf in body:
        out = dict(body)
        _mutate_leaf(out, leaf, action, rename_to)
        return out, True
    for key, value in body.items():
        if isinstance(value, dict):
            child, changed = _apply_leaf(value, leaf, action, rename_to)
            if changed:
                out = dict(body)
                out[key] = child
                return out, True
    return body, False


def strip_unsupported_tool(body: dict, tool_type: str) -> tuple[dict, bool]:
    """Remove every ``{"type": tool_type}`` entry from ``body["tools"]``.

    Copy-on-write, so the original body is never mutated. Returns ``(new_body,
    changed)``; ``changed`` is ``False`` when there is no matching tool (nothing
    safe to strip), leaving the caller to surface the original error unchanged.
    """
    if not isinstance(body, dict):
        return body, False
    tools = body.get("tools")
    if not isinstance(tools, list):
        return body, False
    kept = [
        tool
        for tool in tools
        if not (isinstance(tool, dict) and tool.get("type") == tool_type)
    ]
    if len(kept) == len(tools):
        return body, False
    out = dict(body)
    out["tools"] = kept
    return out, True


def apply_unsupported_remediation(
    body: dict, unsupported: UnsupportedParam
) -> tuple[dict, bool]:
    """Drop or rename the named field/tool in a copy of ``body``.

    Copy-on-write, so the original body is never mutated. A ``"tool"`` kind drops
    every matching ``body["tools"]`` entry (see :func:`strip_unsupported_tool`);
    a ``"field"`` kind drops/renames a field. For fields, a dotted ``field_path``
    (``reasoning.summary``) is resolved exactly; a bare leaf key (``summary``) is
    located by a depth-first search, which is how generic-HTTP ``unknown field``
    errors arrive. ``drop`` removes the terminal key; ``rename`` moves its value
    to ``rename_to`` — unless ``rename_to`` is already present, in which case the
    source key is dropped instead (the client already expressed the intent in the
    newer name, and overwriting it would silently discard the client's value).

    Returns ``(new_body, changed)``. ``changed`` is ``False`` when the field/tool
    is not present anywhere, meaning there is nothing safe to strip; the caller
    must then surface the original error unchanged.
    """
    if not isinstance(body, dict):
        return body, False
    if unsupported.kind == "tool":
        return strip_unsupported_tool(body, unsupported.field_path)
    parts = unsupported.field_path.split(".")
    if len(parts) == 1:
        return _apply_leaf(
            body, unsupported.field_path, unsupported.action, unsupported.rename_to
        )
    return _apply_at_path(body, parts, unsupported.action, unsupported.rename_to)


__all__ = [
    "MAX_UNSUPPORTED_STRIPS",
    "UnsupportedParam",
    "parse_unsupported_param",
    "apply_unsupported_remediation",
    "strip_unsupported_tool",
]
