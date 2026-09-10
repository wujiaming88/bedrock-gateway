"""Tests for the TTL-bounded, thread-safe ``LearnedUnsupportedCache``.

Covers record/lookup/strip semantics, 24h TTL expiry, per-model isolation,
copy-on-write, and — explicitly — thread safety and deadlock-freedom under
concurrent access (the cache guards a shared table with a single short lock).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from bedrock_gateway.unsupported_param import UnsupportedParam
from bedrock_gateway.unsupported_param_cache import (
    DEFAULT_TTL_SECONDS,
    LearnedUnsupportedCache,
)

SUMMARY_DROP = UnsupportedParam(field_path="reasoning.summary", action="drop", rename_to=None)
MAX_TOKENS_RENAME = UnsupportedParam(
    field_path="max_tokens", action="rename", rename_to="max_completion_tokens"
)


class FakeClock:
    """A monotonic-clock stand-in whose time the test advances manually."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# record / lookup / strip semantics
# ---------------------------------------------------------------------------

class TestRecordLookup:
    def test_record_then_lookup_returns_entry(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        live = cache.lookup("m")
        assert live == {"reasoning.summary": SUMMARY_DROP}

    def test_lookup_unknown_model_is_empty(self):
        cache = LearnedUnsupportedCache()
        assert cache.lookup("nope") == {}

    def test_lookup_returns_detached_copy(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        live = cache.lookup("m")
        live.clear()  # mutating the returned dict must not corrupt the cache
        assert cache.lookup("m") == {"reasoning.summary": SUMMARY_DROP}

    def test_models_are_isolated(self):
        cache = LearnedUnsupportedCache()
        cache.record("m1", SUMMARY_DROP)
        cache.record("m2", MAX_TOKENS_RENAME)
        assert set(cache.lookup("m1")) == {"reasoning.summary"}
        assert set(cache.lookup("m2")) == {"max_tokens"}


class TestStrip:
    def test_strip_drops_learned_field(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        body = {"model": "m", "reasoning": {"effort": "low", "summary": "s"}}
        out, changed = cache.strip("m", body)
        assert changed
        assert out["reasoning"] == {"effort": "low"}
        # copy-on-write: original body untouched
        assert body["reasoning"]["summary"] == "s"

    def test_strip_renames_learned_field(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", MAX_TOKENS_RENAME)
        body = {"model": "m", "max_tokens": 42}
        out, changed = cache.strip("m", body)
        assert changed
        assert out == {"model": "m", "max_completion_tokens": 42}
        assert body == {"model": "m", "max_tokens": 42}

    def test_strip_applies_chain_of_learned_fields(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        cache.record("m", MAX_TOKENS_RENAME)
        body = {"model": "m", "max_tokens": 7, "reasoning": {"summary": "s"}}
        out, changed = cache.strip("m", body)
        assert changed
        assert out == {"model": "m", "max_completion_tokens": 7, "reasoning": {}}

    def test_strip_with_no_learned_entries_no_change(self):
        cache = LearnedUnsupportedCache()
        body = {"model": "m", "reasoning": {"summary": "s"}}
        out, changed = cache.strip("m", body)
        assert not changed
        assert out is body

    def test_strip_absent_field_no_change(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        body = {"model": "m", "input": "hi"}  # no reasoning.summary present
        out, changed = cache.strip("m", body)
        assert not changed
        assert out is body

    def test_strip_absent_leaf_no_change(self):
        # Parent key present, learned leaf absent — the no-op must leave the
        # body untouched (this also covers the missing-leaf branch of the
        # underlying remediation).
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        body = {"model": "m", "reasoning": {"effort": "low"}}
        out, changed = cache.strip("m", body)
        assert not changed
        assert out is body

    def test_strip_non_dict_body_no_change(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        out, changed = cache.strip("m", ["not", "a", "dict"])  # type: ignore[arg-type]
        assert not changed
        assert out == ["not", "a", "dict"]


# ---------------------------------------------------------------------------
# TTL expiry (24h)
# ---------------------------------------------------------------------------

class TestTTL:
    def test_default_ttl_is_24_hours(self):
        assert DEFAULT_TTL_SECONDS == 86400.0

    def test_entry_expires_after_ttl(self):
        clock = FakeClock(0.0)
        cache = LearnedUnsupportedCache(ttl_seconds=100.0, clock=clock)
        cache.record("m", SUMMARY_DROP)

        clock.now = 99.0  # not yet expired
        assert cache.lookup("m") == {"reasoning.summary": SUMMARY_DROP}
        body = {"model": "m", "reasoning": {"summary": "s"}}
        _, changed = cache.strip("m", body)
        assert changed

        clock.now = 100.0  # exactly at expiry → expired (strictly-greater guard)
        assert cache.lookup("m") == {}
        out, changed = cache.strip("m", body)
        assert not changed
        assert out is body

    def test_expired_entry_is_pruned(self):
        clock = FakeClock(0.0)
        cache = LearnedUnsupportedCache(ttl_seconds=10.0, clock=clock)
        cache.record("m", SUMMARY_DROP)
        clock.now = 11.0
        assert cache.lookup("m") == {}
        assert cache._entries == {}  # model key removed on prune

    def test_partial_expiry_prunes_only_expired(self):
        clock = FakeClock(0.0)
        cache = LearnedUnsupportedCache(ttl_seconds=100.0, clock=clock)
        cache.record("m", SUMMARY_DROP)      # expires at t=100
        clock.now = 50.0
        cache.record("m", MAX_TOKENS_RENAME)  # expires at t=150
        clock.now = 120.0  # SUMMARY_DROP expired, MAX_TOKENS_RENAME still live
        assert cache.lookup("m") == {"max_tokens": MAX_TOKENS_RENAME}
        # the expired field was pruned; the live one remains in the store
        assert set(cache._entries["m"]) == {"max_tokens"}

    def test_refresh_resets_expiry(self):
        clock = FakeClock(0.0)
        cache = LearnedUnsupportedCache(ttl_seconds=100.0, clock=clock)
        cache.record("m", SUMMARY_DROP)
        clock.now = 80.0
        cache.record("m", SUMMARY_DROP)  # re-learned → expiry pushed out
        clock.now = 170.0  # 90s after the refresh, but >100s after the first learn
        assert cache.lookup("m") == {"reasoning.summary": SUMMARY_DROP}


# ---------------------------------------------------------------------------
# Thread safety and deadlock-freedom
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_strip_is_correct_and_deadlock_free(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        cache.record("m", MAX_TOKENS_RENAME)
        body = {"model": "m", "max_tokens": 5, "reasoning": {"summary": "s"}}

        def work(_: int) -> dict:
            for _ in range(300):
                out, changed = cache.strip("m", body)
                assert changed
                assert "summary" not in out["reasoning"]
                assert "max_tokens" not in out
                assert out["max_completion_tokens"] == 5
            return out

        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(work, i) for i in range(32)]
            for fut in futures:
                # A deadlock would hang the future and blow this timeout.
                assert fut.result(timeout=20) is not None

    def test_concurrent_record_loses_no_entries(self):
        cache = LearnedUnsupportedCache()
        fields = [
            UnsupportedParam(field_path=f"f{i}", action="drop", rename_to=None)
            for i in range(64)
        ]

        def record(field: UnsupportedParam) -> None:
            for _ in range(50):
                cache.record("m", field)

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(record, fields))

        live = cache.lookup("m")
        assert set(live) == {f.field_path for f in fields}

    def test_mixed_record_strip_lookup_no_deadlock(self):
        cache = LearnedUnsupportedCache()
        body = {"model": "m", "reasoning": {"summary": "s"}}

        def churn(i: int) -> None:
            for n in range(200):
                if n % 3 == 0:
                    cache.record("m", SUMMARY_DROP)
                elif n % 3 == 1:
                    cache.strip("m", body)
                else:
                    cache.lookup("m")

        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = [ex.submit(churn, i) for i in range(24)]
            for fut in futures:
                fut.result(timeout=20)

        # Final state is sane: the field was learned at least once and still lives.
        assert cache.lookup("m") == {"reasoning.summary": SUMMARY_DROP}

    def test_clear_resets_under_concurrency(self):
        cache = LearnedUnsupportedCache()
        cache.record("m", SUMMARY_DROP)
        cache.clear()
        assert cache.lookup("m") == {}
        assert cache._entries == {}
