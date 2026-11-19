"""Peer-level diagnostics for requests uvicorn's parser rejects.

uvicorn's stock warning ("Invalid HTTP request received.") carries neither the
peer nor the payload, which makes an operator unable to tell an internet
scanner from a broken client. These tests pin the replacement: peer, a kind
classification, and a bounded hex/repr preview — while the original parse
exception must still propagate so uvicorn's 400 path is unchanged.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from bedrock_gateway.http_diagnostics import (
    DiagnosticHttpToolsProtocol,
    _classify,
    _DiagnosticParser,
    _preview,
    http_protocol_class,
    log_rejected,
)

TLS_CLIENT_HELLO = b"\x16\x03\x01\x00\xa5\x01\x00\x00\xa1\x03\x03"


class TestClassify:
    def test_empty(self):
        assert _classify(b"") == "empty"

    def test_tls_client_hello(self):
        assert "tls-client-hello" in _classify(TLS_CLIENT_HELLO)

    def test_text(self):
        assert _classify(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n").startswith("text")

    def test_binary(self):
        assert _classify(bytes([0x00, 0x01, 0x02, 0xFF])) == "binary"

    def test_mixed(self):
        assert _classify(b"GET \x00\x01\x02\xff/bin") == "mixed text/binary"


class TestPreview:
    def test_escapes_control_chars_onto_one_line(self):
        preview = _preview(b"GET / HTTP/1.1\r\n")
        assert "\\r\\n" in preview
        assert "\n" not in preview
        assert "hex=" in preview

    def test_truncates_long_payload(self):
        preview = _preview(b"A" * 200)
        assert "…" in preview
        # 64 preview bytes -> 128 hex chars
        assert len(preview.rsplit("hex=", 1)[1]) == 128

    def test_does_not_mark_short_payload(self):
        assert "…" not in _preview(b"short")

    def test_high_bytes_escape_exactly_once(self):
        preview = _preview(TLS_CLIENT_HELLO)
        assert "\\xa5" in preview
        assert "\\\\xa5" not in preview


class _BoomParser:
    def __init__(self) -> None:
        self.seen: list[bytes] = []

    def feed_data(self, data: bytes) -> None:
        self.seen.append(data)
        raise ValueError("parse failed")


class _RecordingParser:
    def __init__(self) -> None:
        self.seen: list[bytes] = []
        self.flavor = "vanilla"

    def feed_data(self, data: bytes) -> None:
        self.seen.append(data)


class _Peer:
    client = ("9.9.9.9", 1234)


class _NoPeer:
    client = None


class TestLogRejected:
    def test_includes_peer_and_kind(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bedrock_gateway"):
            log_rejected(_Peer(), b"GET / HTTP/1.1\r\n")
        message = caplog.records[-1].getMessage()
        assert "REJECT invalid_http" in message
        assert "peer=9.9.9.9:1234" in message
        assert "kind=text" in message

    def test_missing_peer_is_labelled(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bedrock_gateway"):
            log_rejected(_NoPeer(), b"\x00\x01")
        assert "peer=unknown" in caplog.records[-1].getMessage()


class TestDiagnosticParser:
    def test_logs_then_reraises(self, caplog):
        inner = _BoomParser()
        proxy = _DiagnosticParser(inner, _Peer())
        with caplog.at_level(logging.WARNING, logger="bedrock_gateway"):
            with pytest.raises(ValueError):
                proxy.feed_data(b"\x00\x01")
        assert "REJECT invalid_http" in caplog.records[-1].getMessage()
        assert inner.seen == [b"\x00\x01"]

    def test_success_path_is_transparent(self):
        inner = _RecordingParser()
        proxy = _DiagnosticParser(inner, _Peer())
        proxy.feed_data(b"GET / HTTP/1.1\r\n")
        assert inner.seen == [b"GET / HTTP/1.1\r\n"]

    def test_attributes_forward_both_ways(self):
        inner = _RecordingParser()
        proxy = _DiagnosticParser(inner, _Peer())
        assert proxy.flavor == "vanilla"
        proxy.flavor = "chocolate"
        assert inner.flavor == "chocolate"

    def test_logging_failure_never_masks_parse_error(self, monkeypatch):
        import bedrock_gateway.http_diagnostics as hd

        def _explode(*_args):
            raise RuntimeError("logger exploded")

        monkeypatch.setattr(hd, "log_rejected", _explode)
        proxy = _DiagnosticParser(_BoomParser(), _Peer())
        with pytest.raises(ValueError):
            proxy.feed_data(b"x")


class TestProtocolWiring:
    @pytest.mark.skipif(
        DiagnosticHttpToolsProtocol is None,
        reason="httptools not installed (minimal uvicorn) — fallback path only",
    )
    def test_protocol_class_wraps_the_parser(self):
        from uvicorn.config import Config
        from uvicorn.server import ServerState

        cfg = Config(
            app=lambda scope, receive, send: None,
            http=DiagnosticHttpToolsProtocol,
            loop="asyncio",
            lifespan="off",
        )
        cfg.load()
        # Pass the loop explicitly: another test may have closed the current
        # one, and the protocol constructor defaults to get_event_loop().
        loop = asyncio.new_event_loop()
        try:
            protocol = cfg.http_protocol_class(cfg, ServerState(), {}, _loop=loop)
        finally:
            loop.close()
        assert isinstance(protocol.parser, _DiagnosticParser)

    def test_http_protocol_class_returns_the_diagnostic_protocol(self):
        selected = http_protocol_class()
        if DiagnosticHttpToolsProtocol is None:
            assert selected == "auto"
        else:
            assert selected is DiagnosticHttpToolsProtocol
