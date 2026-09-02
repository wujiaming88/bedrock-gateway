"""Daily file logging configuration tests."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from bedrock_gateway.config import GatewayConfig, LoggingConfig, ServerConfig
from bedrock_gateway.logging_config import (
    DailyFileHandler,
    DirectionFilter,
    configure_logging,
)


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


_LOG_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} ")


def test_shared_formatter_uses_fixed_three_digit_milliseconds(tmp_path):
    cfg = GatewayConfig(
        server=ServerConfig(log_level="info"),
        logging=LoggingConfig(
            file_enabled=True, directory=str(tmp_path), retention_days=30
        ),
    )
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_logging(cfg)
        formatter = next(
            handler.formatter
            for handler in root.handlers
            if handler not in original_handlers
        )
        assert formatter is not None
        record = logging.LogRecord(
            "bedrock_gateway", logging.INFO, __file__, 1, "marker", (), None
        )
        record.created = _timestamp("2026-08-27 12:34:56")
        record.msecs = 7.0
        assert formatter.format(record).startswith("2026-08-27 12:34:56.007 ")
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_daily_handler_switches_files_at_local_midnight(tmp_path):
    now = [_timestamp("2026-08-11 23:59:59")]
    handler = DailyFileHandler(tmp_path, 30, clock=lambda: now[0])
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.daily.switch")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.info("before")
        now[0] = _timestamp("2026-08-12 00:00:01")
        logger.info("after")
    finally:
        handler.close()
        logger.handlers.clear()
        logger.propagate = True

    assert (tmp_path / "bedrock-gateway-2026-08-11.log").read_text().strip() == "before"
    assert (tmp_path / "bedrock-gateway-2026-08-12.log").read_text().strip() == "after"


def test_daily_handler_deletes_only_expired_matching_files(tmp_path):
    for name in (
        "bedrock-gateway-2026-07-11.log",
        "bedrock-gateway-2026-07-12.log",
        "bedrock-gateway-invalid.log",
        "other-2026-01-01.log",
    ):
        (tmp_path / name).write_text("x")

    handler = DailyFileHandler(
        tmp_path, 30, clock=lambda: _timestamp("2026-08-11 12:00:00")
    )
    handler.close()

    assert not (tmp_path / "bedrock-gateway-2026-07-11.log").exists()
    assert (tmp_path / "bedrock-gateway-2026-07-12.log").exists()
    assert (tmp_path / "bedrock-gateway-invalid.log").exists()
    assert (tmp_path / "other-2026-01-01.log").exists()


def test_configure_logging_collects_all_loggers_without_duplicates(
    tmp_path, capsys
):
    cfg = GatewayConfig(
        server=ServerConfig(log_level="info"),
        logging=LoggingConfig(
            file_enabled=True, directory=str(tmp_path), retention_days=30
        ),
    )
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_logging(cfg)
        configure_logging(cfg)
        names = (
            "bedrock_gateway",
            "bedrock_gateway.dashboard.metrics",
            "uvicorn.error",
            "uvicorn.access",
            "httpx",
        )
        for index, name in enumerate(names):
            logging.getLogger(name).info("unique-marker-%d", index)

        text = next(tmp_path.glob("bedrock-gateway-*.log")).read_text()
        console = capsys.readouterr().err
        for index in range(len(names)):
            marker = f"unique-marker-{index}"
            assert text.count(marker) == 1
            assert console.count(marker) == 1
        assert "[test_logging_config.py:" in text
        assert all(_LOG_PREFIX.match(line) for line in text.splitlines())
        assert all(_LOG_PREFIX.match(line) for line in console.splitlines())
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def _record(name: str, message: str) -> logging.LogRecord:
    return logging.LogRecord(name, logging.INFO, __file__, 1, message, (), None)


def test_direction_filter_tags_third_party_loggers():
    filt = DirectionFilter()

    tagged = [
        ("httpx", "[UP] "),
        ("httpx._client", "[UP] "),
        ("httpcore", "[UP] "),
        ("httpcore.http11", "[UP] "),
        ("uvicorn.access", "[DN] "),
    ]
    for name, tag in tagged:
        record = _record(name, "hello")
        assert filt.filter(record) is True
        assert record.getMessage() == f"{tag}hello"

    for name in ("bedrock_gateway", "uvicorn.error", "uvicorn.asgi"):
        record = _record(name, "hello")
        assert filt.filter(record) is True
        assert record.getMessage() == "hello"


def test_direction_filter_is_idempotent():
    filt = DirectionFilter()
    for message in ("[UP] hello", "[DN] hello"):
        record = _record("httpx", message)
        assert filt.filter(record) is True
        assert record.getMessage() == message
