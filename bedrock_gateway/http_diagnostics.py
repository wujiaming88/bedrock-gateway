"""Peer-level diagnostics for requests uvicorn's HTTP parser rejects.

When httptools cannot parse an inbound request, uvicorn logs a bare
``Invalid HTTP request received.`` — no peer, no payload. An operator reading
that cannot tell whether the caller is an internet scanner speaking TLS to the
plaintext port, a client with a malformed request line, or a binary protocol
probe. The 400 uvicorn returns is already correct; only the *explanation* is
missing.

This module wraps uvicorn's parser so the same rejection also records the peer
address and a bounded preview of what arrived. It changes no control flow: the
parser is proxied, the exception is re-raised, and uvicorn still answers 400
exactly as before. Diagnostics that fail are swallowed — logging must never
mask the real parse error.
"""

from __future__ import annotations

import logging
from typing import Any

try:  # httptools ships with uvicorn[standard]; degrade to uvicorn's auto choice.
    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
except ImportError:  # pragma: no cover - only in a minimal uvicorn install
    HttpToolsProtocol = None  # type: ignore[assignment,misc]

logger = logging.getLogger("bedrock_gateway")

# Enough to fingerprint a TLS ClientHello or a malformed request line without
# writing an attacker's whole payload into the log.
_PREVIEW_BYTES = 64


def _classify(data: bytes) -> str:
    """Name what most likely sent ``data``, so the cause is readable at a glance."""
    if not data:
        return "empty"
    # TLS record header: content type 0x16 (handshake) + major version 0x03.
    if data[0] == 0x16 and len(data) >= 3 and data[1] == 0x03:
        return "tls-client-hello (HTTPS sent to a plaintext HTTP port?)"
    head = data[:_PREVIEW_BYTES]
    printable = sum(1 for b in head if 0x20 <= b < 0x7F or b in (0x09, 0x0A, 0x0D))
    if printable == len(head):
        return "text (malformed HTTP request line/headers?)"
    if printable == 0:
        return "binary"
    return "mixed text/binary"


def _preview(data: bytes) -> str:
    """A one-line, log-safe rendering: bytes repr, then hex for exactness."""
    head = data[:_PREVIEW_BYTES]
    truncated = "…" if len(data) > _PREVIEW_BYTES else ""
    # bytes repr escapes control and non-ASCII bytes exactly once (``\xa5``)
    # and keeps printable ASCII literal, which is what a wire preview needs.
    return f"{head!r}{truncated} hex={head.hex()}"


def log_rejected(protocol: Any, data: bytes) -> None:
    """Record who sent an unparseable request and what it looked like."""
    client = getattr(protocol, "client", None)
    peer = f"{client[0]}:{client[1]}" if client else "unknown"
    logger.warning(
        "[DN] REJECT invalid_http peer=%s kind=%s bytes=%d preview=%s",
        peer,
        _classify(data),
        len(data),
        _preview(data),
    )


class _DiagnosticParser:
    """Transparent proxy around httptools' parser that logs rejected input.

    uvicorn calls exactly one method on the parser while serving traffic
    (``feed_data``); everything else is forwarded unchanged, so wiring this in
    leaves uvicorn's own error handling — the 400 response included — intact.
    """

    def __init__(self, inner: Any, protocol: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_protocol", protocol)

    def feed_data(self, data: bytes) -> None:
        try:
            self._inner.feed_data(data)
        except Exception:
            try:
                log_rejected(self._protocol, data)
            except Exception:  # pragma: no cover - never mask the parse error
                pass
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


if HttpToolsProtocol is not None:

    class DiagnosticHttpToolsProtocol(HttpToolsProtocol):  # type: ignore[misc,valid-type]
        """uvicorn protocol that names the peer when a request fails to parse."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Wrap only after the base constructor configured the parser
            # (callbacks, leniencies), so nothing is lost in the hand-off.
            self.parser = _DiagnosticParser(self.parser, self)

else:  # pragma: no cover - minimal uvicorn install
    DiagnosticHttpToolsProtocol = None  # type: ignore[assignment,misc]


def http_protocol_class() -> Any:
    """The ``http=`` value for ``uvicorn.run`` that enables these diagnostics.

    Falls back to uvicorn's own auto-selection when httptools is unavailable.
    """
    return DiagnosticHttpToolsProtocol or "auto"
