"""
Pure, generic remediation for Bedrock mantle "Unsupported parameter" 400s.

This is the single source of truth for **Class A** divergence: a model's
OpenAI-compatible validator rejects a whole field it does not implement, naming
it precisely in the error text, e.g. ::

    Unsupported parameter: 'reasoning.summary' is not supported with the
    'openai.gpt-6-astra' model.

or, for a field that was merely renamed across OpenAI API generations::

    Unsupported parameter: 'max_tokens' is not supported with this model.

Because the upstream names the exact field it rejects, removing (or, for a small
lossless rename table, renaming) that field is inherently safe: the request was
already going to 400, and a field the model does not implement cannot carry
semantics for it. This module is deliberately **pure** — no I/O, no mutable
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

_UNSUPPORTED_PARAM_RE = re.compile(r"Unsupported parameter:\s*'([^']+)'", re.IGNORECASE)


@dataclass(frozen=True)
class UnsupportedParam:
    """A single field the upstream named as unsupported.

    ``field_path`` is the dotted path into the request body (``reasoning.summary``
    or ``max_tokens``). ``action`` is ``"drop"`` or ``"rename"``; ``rename_to`` is
    the replacement leaf key for a rename, else ``None``.
    """

    field_path: str
    action: str
    rename_to: str | None


def parse_unsupported_param(error_text: str | None) -> UnsupportedParam | None:
    """Parse an upstream 400 for the field it declared unsupported.

    Returns an :class:`UnsupportedParam` for the exact ``Unsupported parameter:
    'X'`` signature, choosing ``rename`` when ``X`` is in :data:`_RENAMES` and
    ``drop`` otherwise. Returns ``None`` for anything else — non-400 text, the
    ``Invalid 'input'`` variant rejection, relationship errors, or no match —
    so callers only ever act on a precisely named field.
    """
    if not error_text:
        return None
    match = _UNSUPPORTED_PARAM_RE.search(error_text)
    if match is None:
        return None
    field_path = match.group(1)
    rename_to = _RENAMES.get(field_path)
    if rename_to is not None:
        return UnsupportedParam(field_path=field_path, action="rename", rename_to=rename_to)
    return UnsupportedParam(field_path=field_path, action="drop", rename_to=None)


def apply_unsupported_remediation(
    body: dict, unsupported: UnsupportedParam
) -> tuple[dict, bool]:
    """Drop or rename the named field in a copy of ``body``.

    Copy-on-write along the dotted ``field_path``: each intermediate dict is
    copied before descending, so the original body is never mutated. ``drop``
    removes the terminal key; ``rename`` moves its value to ``rename_to`` —
    unless ``rename_to`` is already present, in which case the source key is
    dropped instead (the client already expressed the intent in the newer name,
    and overwriting it would silently discard the client's value).

    Returns ``(new_body, changed)``. ``changed`` is ``False`` when the field
    path does not resolve to a present key (or the body is not a dict), meaning
    there is nothing safe to strip; the caller must then surface the original
    error unchanged.
    """
    if not isinstance(body, dict):
        return body, False
    parts = unsupported.field_path.split(".")
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
    if unsupported.action == "rename" and unsupported.rename_to is not None:
        if unsupported.rename_to in node:
            del node[leaf]
        else:
            node[unsupported.rename_to] = node.pop(leaf)
    else:  # drop
        del node[leaf]
    return out, True


__all__ = [
    "MAX_UNSUPPORTED_STRIPS",
    "UnsupportedParam",
    "parse_unsupported_param",
    "apply_unsupported_remediation",
]
