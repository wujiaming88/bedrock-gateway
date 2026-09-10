"""Shared test fixtures.

The learned-unsupported cache is process-global state; an autouse fixture clears
it around every test so one test's recorded lesson never leaks into another's
raw-first expectation.
"""

from __future__ import annotations

import pytest

from bedrock_gateway.server import _LEARNED_UNSUPPORTED


@pytest.fixture(autouse=True)
def _reset_learned_unsupported():
    _LEARNED_UNSUPPORTED.clear()
    yield
    _LEARNED_UNSUPPORTED.clear()
