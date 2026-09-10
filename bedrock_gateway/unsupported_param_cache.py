"""In-memory, TTL-bounded memoization of per-model unsupported fields.

The raw-first self-heal in :mod:`bedrock_gateway.server` learns which fields a
model rejects only *after* an upstream 400. This module remembers that lesson —
per model, with a TTL — so later requests can drop/rename the field *before*
sending and succeed on the first attempt instead of paying a 400 round-trip.

Thread safety: a single ``threading.Lock`` guards the entry table. Every lock
hold is a short, non-blocking dict operation; the pure remediation in
:meth:`LearnedUnsupportedCache.strip` runs *outside* the lock. There is exactly
one lock, it is never acquired re-entrantly, and nothing under the lock calls
back out — so the cache cannot deadlock, and concurrent readers only contend for
a few microseconds.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from bedrock_gateway.unsupported_param import (
    UnsupportedParam,
    apply_unsupported_remediation,
)

DEFAULT_TTL_SECONDS = 86400.0  # 24 hours


@dataclass(frozen=True)
class _Entry:
    unsupported: UnsupportedParam
    expires_at: float


class LearnedUnsupportedCache:
    """Per-model set of learned unsupported fields, each with a TTL.

    :meth:`record` remembers that ``model`` rejected a ``field_path``;
    :meth:`strip` applies every non-expired learned remediation for a model to a
    request body (pure copy-on-write — the original body is never mutated).
    Entries expire after ``ttl_seconds`` so a field the upstream later grows
    support for is only pessimistically stripped for a bounded window.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        *,
        clock=time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: dict[str, dict[str, _Entry]] = {}
        self._lock = threading.Lock()

    def record(self, model: str, unsupported: UnsupportedParam) -> None:
        """Remember that ``model`` rejected ``unsupported.field_path``."""
        with self._lock:
            expires_at = self._clock() + self._ttl
            self._entries.setdefault(model, {})[unsupported.field_path] = _Entry(
                unsupported, expires_at
            )

    def lookup(self, model: str) -> dict[str, UnsupportedParam]:
        """Return the non-expired ``{field_path: UnsupportedParam}`` for ``model``.

        Expired entries are pruned lazily. The returned dict is a detached copy;
        mutating it cannot corrupt the cache.
        """
        with self._lock:
            now = self._clock()
            raw = self._entries.get(model)
            if not raw:
                return {}
            live = {
                path: entry.unsupported
                for path, entry in raw.items()
                if entry.expires_at > now
            }
            # Prune only when something actually expired, avoiding a needless
            # rebuild on the common hot path (nothing expired).
            if len(live) != len(raw):
                if live:
                    self._entries[model] = {
                        path: entry
                        for path, entry in raw.items()
                        if entry.expires_at > now
                    }
                else:
                    self._entries.pop(model, None)
            return live

    def strip(self, model: str, body: dict) -> tuple[dict, bool]:
        """Apply every learned remediation for ``model`` to ``body``.

        Returns ``(new_body, changed)``; ``changed`` is ``False`` when there is
        nothing learned (or nothing applicable) to strip. Copy-on-write: the
        input ``body`` is never mutated.
        """
        if not isinstance(body, dict):
            return body, False
        live = self.lookup(model)
        if not live:
            return body, False
        out = body
        changed_any = False
        for unsupported in live.values():
            out, changed = apply_unsupported_remediation(out, unsupported)
            changed_any = changed_any or changed
        return out, changed_any

    def clear(self) -> None:
        """Drop all learned entries (used by tests to reset process state)."""
        with self._lock:
            self._entries.clear()
