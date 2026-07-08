"""Coverage tests for small uncovered branches in config.py and converter.py."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import bedrock_gateway
from bedrock_gateway.config import _deep_resolve, load_config
from bedrock_gateway.converter import (
    _convert_image_url,
    convert_tool_choice,
    decode_event_stream_chunk,
)


class TestVersionConsistency:
    """The version lives in two files that must never drift apart:
    ``bedrock_gateway/__init__.py`` (runtime / /health) and
    ``pyproject.toml`` (packaging / the pip-installed dist name).
    A mismatch silently ships a wheel labelled with the wrong version."""

    def test_init_matches_pyproject(self):
        pyproject = (
            Path(__file__).resolve().parent.parent / "pyproject.toml"
        ).read_text()
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
        assert m, "no version found in pyproject.toml [project]"
        assert m.group(1) == bedrock_gateway.__version__, (
            f"pyproject version {m.group(1)!r} != "
            f"__init__ {bedrock_gateway.__version__!r}"
        )


class TestDeepResolveList:
    def test_resolves_list_of_strings(self):
        # Covers the ``isinstance(obj, list)`` branch of _deep_resolve.
        os.environ["TEST_XYZ_COV"] = "hello"
        try:
            out = _deep_resolve(["literal", "${TEST_XYZ_COV}"])
            assert out == ["literal", "hello"]
        finally:
            del os.environ["TEST_XYZ_COV"]


class TestLoadConfigFindsDefaultConfig:
    def test_discovers_config_yaml_in_cwd(self, tmp_path, monkeypatch):
        """When path is None and config.yaml exists in cwd, it's picked up."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "region: eu-west-1\n"
            "auth:\n  mode: bearer_token\n  bearer_token: yaml-token\n"
        )
        monkeypatch.chdir(tmp_path)
        # AWS_REGION env var takes precedence via __post_init__; clear it.
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = load_config(None)
        assert cfg.region == "eu-west-1"
        assert cfg.auth.bearer_token == "yaml-token"


class TestConvertImageUrl:
    def test_empty_url_returns_none(self):
        assert _convert_image_url({"image_url": {"url": ""}}) is None

    def test_malformed_data_url_returns_none(self):
        # ``data:`` prefix but without the image/<mime>;base64, structure.
        assert _convert_image_url(
            {"image_url": {"url": "data:application/json;base64,xyz"}}
        ) is None


class TestConvertToolChoiceDefault:
    def test_unknown_shape_returns_none(self):
        # Has tools but dict without "function" → falls through.
        assert convert_tool_choice({"random": "shape"}, True) is None


class TestDecodeEventStreamChunkMalformed:
    def test_malformed_base64_is_skipped(self):
        """Regex matches, but base64.b64decode raises (odd length) or JSON parse
        fails — both are swallowed and the event is skipped."""
        # The regex is "[A-Za-z0-9+/=]+" — an odd-length all-valid-character
        # string still matches but base64 decode / JSON parse will fail.
        buf = b'"bytes":"Zm9v"'  # matches regex; decodes to b"foo" → not JSON
        events, consumed = decode_event_stream_chunk(buf)
        assert events == []
        assert consumed > 0
