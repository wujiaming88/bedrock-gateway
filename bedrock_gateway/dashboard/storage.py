"""
SQLite-backed persistence for dashboard metrics.

Kept deliberately minimal:
  * A ``requests`` table retaining the most recent request records, used to
    repopulate the request-log panel after a restart.
  * A ``minute_buckets`` table holding per-minute aggregates (QPS, success
    counts, latency percentiles, tokens) so the traffic chart survives a
    restart.
  * Writes are coalesced through an :class:`AsyncWriter` so the metrics
    middleware never blocks on disk I/O.

SQLite is enough here: the gateway is a single-instance process, the
volume is low (one row per request), and ``sqlite3`` ships with Python
so we don't pull a new dependency.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class RequestRow:
    ts: float
    method: str
    path: str
    model: str
    status: int
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class BucketRow:
    ts: int  # minute-boundary unix timestamp
    total: int = 0
    success: int = 0
    error: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetricsStorage
# ---------------------------------------------------------------------------


class MetricsStorage:
    """File-backed store used by :class:`MetricsCollector` to survive restarts."""

    def __init__(self, db_path: str | Path = "metrics.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # --- connection --------------------------------------------------------

    @contextmanager
    def _connect(self) -> Any:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    method TEXT,
                    path TEXT,
                    model TEXT,
                    status INTEGER,
                    latency_ms REAL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    error_type TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS minute_buckets (
                    ts INTEGER PRIMARY KEY,
                    total INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 0,
                    error INTEGER DEFAULT 0,
                    latency_sum REAL DEFAULT 0,
                    latency_count INTEGER DEFAULT 0,
                    p50 REAL DEFAULT 0,
                    p95 REAL DEFAULT 0,
                    p99 REAL DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    status_counts_json TEXT DEFAULT '{}',
                    model_counts_json TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
                CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
                """
            )

    # --- writes ------------------------------------------------------------

    def batch_write_requests(self, rows: Iterable[RequestRow]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO requests "
                "(ts, method, path, model, status, latency_ms, "
                "prompt_tokens, completion_tokens, error_type, error_message) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r.ts, r.method, r.path, r.model, r.status, r.latency_ms,
                        r.prompt_tokens, r.completion_tokens,
                        r.error_type, r.error_message,
                    )
                    for r in rows
                ],
            )

    def upsert_bucket(self, bucket: BucketRow) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO minute_buckets "
                "(ts, total, success, error, latency_sum, latency_count, "
                "p50, p95, p99, prompt_tokens, completion_tokens, "
                "status_counts_json, model_counts_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(ts) DO UPDATE SET "
                "total=excluded.total, success=excluded.success, error=excluded.error, "
                "latency_sum=excluded.latency_sum, latency_count=excluded.latency_count, "
                "p50=excluded.p50, p95=excluded.p95, p99=excluded.p99, "
                "prompt_tokens=excluded.prompt_tokens, "
                "completion_tokens=excluded.completion_tokens, "
                "status_counts_json=excluded.status_counts_json, "
                "model_counts_json=excluded.model_counts_json",
                (
                    bucket.ts, bucket.total, bucket.success, bucket.error,
                    bucket.latency_sum, bucket.latency_count,
                    bucket.p50, bucket.p95, bucket.p99,
                    bucket.prompt_tokens, bucket.completion_tokens,
                    json.dumps({str(k): v for k, v in bucket.status_counts.items()}),
                    json.dumps(bucket.model_counts),
                ),
            )

    # --- reads -------------------------------------------------------------

    def load_recent_requests(self, limit: int = 200) -> list[RequestRow]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, method, path, model, status, latency_ms, "
                "prompt_tokens, completion_tokens, error_type, error_message "
                "FROM requests ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            RequestRow(
                ts=float(r[0]), method=r[1] or "", path=r[2] or "",
                model=r[3] or "-", status=int(r[4] or 0),
                latency_ms=float(r[5] or 0.0),
                prompt_tokens=int(r[6] or 0), completion_tokens=int(r[7] or 0),
                error_type=r[8], error_message=r[9],
            )
            for r in rows
        ]

    def load_buckets(self, since_ts: int) -> list[BucketRow]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, total, success, error, latency_sum, latency_count, "
                "p50, p95, p99, prompt_tokens, completion_tokens, "
                "status_counts_json, model_counts_json "
                "FROM minute_buckets WHERE ts >= ? ORDER BY ts ASC",
                (int(since_ts),),
            ).fetchall()
        out: list[BucketRow] = []
        for r in rows:
            try:
                sc_raw = json.loads(r[11] or "{}")
                status_counts = {int(k): int(v) for k, v in sc_raw.items()}
            except (ValueError, TypeError):
                status_counts = {}
            try:
                model_counts = {
                    str(k): int(v) for k, v in json.loads(r[12] or "{}").items()
                }
            except (ValueError, TypeError):
                model_counts = {}
            out.append(
                BucketRow(
                    ts=int(r[0]), total=int(r[1] or 0),
                    success=int(r[2] or 0), error=int(r[3] or 0),
                    latency_sum=float(r[4] or 0.0),
                    latency_count=int(r[5] or 0),
                    p50=float(r[6] or 0.0), p95=float(r[7] or 0.0),
                    p99=float(r[8] or 0.0),
                    prompt_tokens=int(r[9] or 0),
                    completion_tokens=int(r[10] or 0),
                    status_counts=status_counts,
                    model_counts=model_counts,
                )
            )
        return out

    def cleanup(self, retain_days: int = 7) -> int:
        """Delete rows older than *retain_days*. Returns total deleted."""
        cutoff = time.time() - retain_days * 86400
        with self._lock, self._connect() as conn:
            n1 = conn.execute(
                "DELETE FROM requests WHERE ts < ?", (cutoff,)
            ).rowcount or 0
            n2 = conn.execute(
                "DELETE FROM minute_buckets WHERE ts < ?", (int(cutoff),)
            ).rowcount or 0
        return int(n1) + int(n2)


# ---------------------------------------------------------------------------
# Background writer
# ---------------------------------------------------------------------------


class AsyncWriter:
    """
    Background thread that drains a bounded queue of ``RequestRow`` entries
    into :class:`MetricsStorage`.

    Designed to never block the hot path: ``enqueue`` drops writes when the
    queue is full rather than stalling request handling.
    """

    _STOP = object()

    def __init__(
        self,
        storage: MetricsStorage,
        *,
        max_queue: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 1.0,
    ) -> None:
        self._storage = storage
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.05, flush_interval)
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._loop, name="dashboard-storage-writer", daemon=True
        )
        self._thread.start()

    def enqueue(self, row: RequestRow) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1

    def stop(self) -> None:
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    @property
    def dropped(self) -> int:
        return self._dropped

    def _loop(self) -> None:
        while True:
            batch: list[RequestRow] = []
            try:
                item = self._queue.get(timeout=self._flush_interval)
            except queue.Empty:
                continue
            if item is self._STOP:
                return
            batch.append(item)  # type: ignore[arg-type]
            while len(batch) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is self._STOP:
                    break
                batch.append(item)  # type: ignore[arg-type]
            try:
                self._storage.batch_write_requests(batch)
            except sqlite3.Error:
                # Never let a storage error kill the writer thread.
                pass
            if item is self._STOP:
                return
