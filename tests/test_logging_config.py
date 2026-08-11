"""Daily file logging configuration tests."""

from __future__ import annotations

import logging
from datetime import datetime

from bedrock_gateway.config import GatewayConfig, LoggingConfig, ServerConfig
from bedrock_gateway.logging_config import DailyFileHandler, configure_logging


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


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


def test_configure_logging_collects_all_loggers_without_duplicates(tmp_path):
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
        for index in range(len(names)):
            assert text.count(f"unique-marker-{index}") == 1
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
