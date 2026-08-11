"""Logging setup with daily files and console output for journald."""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from .config import GatewayConfig

_LOG_NAME = re.compile(r"^bedrock-gateway-(\d{4}-\d{2}-\d{2})\.log$")
_MANAGED = "_bedrock_gateway_managed"


class DailyFileHandler(logging.Handler):
    """Write directly to a local-date log file and switch after midnight."""

    def __init__(
        self,
        directory: str | Path,
        retention_days: int,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.retention_days = retention_days
        self._clock = clock
        self._current_date: date | None = None
        self._stream = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._switch_if_needed()

    @staticmethod
    def _local_date(timestamp: float) -> date:
        return datetime.fromtimestamp(timestamp).date()

    def _path_for(self, day: date) -> Path:
        return self.directory / f"bedrock-gateway-{day.isoformat()}.log"

    def _switch_if_needed(self) -> None:
        today = self._local_date(self._clock())
        if today == self._current_date and self._stream is not None:
            return
        if self._stream is not None:
            self._stream.close()
        self._stream = self._path_for(today).open("a", encoding="utf-8")
        self._current_date = today
        self._delete_expired(today)

    def _delete_expired(self, today: date) -> None:
        cutoff = today - timedelta(days=self.retention_days)
        for path in self.directory.iterdir():
            match = _LOG_NAME.fullmatch(path.name)
            if not match or not path.is_file():
                continue
            try:
                file_date = date.fromisoformat(match.group(1))
            except ValueError:
                continue
            if file_date < cutoff:
                path.unlink()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.acquire()
            self._switch_if_needed()
            assert self._stream is not None
            self._stream.write(self.format(record) + self.terminator)
            self._stream.flush()
        except Exception:
            self.handleError(record)
        finally:
            self.release()

    @property
    def terminator(self) -> str:
        return "\n"

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            self.release()
        super().close()


def configure_logging(config: GatewayConfig) -> None:
    """Configure one console path plus an optional daily file path."""
    level = getattr(logging, config.server.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED, False):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    setattr(console, _MANAGED, True)
    root.addHandler(console)

    if config.logging.file_enabled:
        file_handler = DailyFileHandler(
            config.logging.directory,
            config.logging.retention_days,
        )
        file_handler.setFormatter(formatter)
        setattr(file_handler, _MANAGED, True)
        root.addHandler(file_handler)

    root.setLevel(level)
    for name in ("bedrock_gateway", "httpx", "httpcore"):
        child = logging.getLogger(name)
        child.setLevel(level)
        child.propagate = True

    # Uvicorn normally owns handlers with propagate=False. The standalone runner
    # passes log_config=None, so route every Uvicorn record through root instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        child = logging.getLogger(name)
        child.handlers.clear()
        child.setLevel(level)
        child.propagate = True
