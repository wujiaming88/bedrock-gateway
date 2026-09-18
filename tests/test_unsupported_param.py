"""Tests for the pure ``unsupported_param`` module and its server fallback.

Covers Class A divergence — ``Unsupported parameter: 'X'`` 400s — which is
remediated by dropping or losslessly renaming the exact field the upstream
named, then retrying. Also covers the arming predicate boundaries.
"""

from __future__ import annotations

import pytest

from bedrock_gateway.responses_compatibility import (
    is_bedrock_gpt5x_responses_model,
    is_bedrock_gpt_responses_model,
    is_bedrock_mantle_openai_model,
    is_openai_compatible_model,
)
from bedrock_gateway.unsupported_param import (
    MAX_UNSUPPORTED_STRIPS,
    UnsupportedParam,
    apply_unsupported_remediation,
    parse_unsupported_param,
)

REASONING_SUMMARY_400 = (
    "Unsupported parameter: 'reasoning.summary' is not supported with the "
    "'openai.gpt-6-astra' model."
)
MAX_TOKENS_400 = "Unsupported parameter: 'max_tokens' is not supported with this model."
ARK_SUMMARY_400 = 'json: unknown field "summary" Request id: 0217897032abcdef'
ARK_VERBOSITY_400 = 'json: unknown field "verbosity" Request id: 0217897032fedcba'
# The upstream error body is JSON, so the field name is backslash-escaped in the
# raw ``resp.text`` the server hands to the parser. These are byte-for-byte the
# live ark /responses 400 bodies (verified 2026-09-18 against the real endpoint).
ARK_SUMMARY_400_ESCAPED = (
    r'{"error":{"code":"InvalidParameter","message":"json: unknown field \"summary\"'
    r' Request id: 021789705923845cb4471f0f910e4739f2e6b993dd0a535411e1c",'
    r'"param":"","type":"BadRequest"}}'
)
ARK_VERBOSITY_400_ESCAPED = (
    r'{"error":{"code":"InvalidParameter","message":"json: unknown field \"verbosity\"'
    r' Request id: 021789705923845cb4471f0f910e4739f2e6b993dd0a535411e1c",'
    r'"param":"","type":"BadRequest"}}'
)


# ---------------------------------------------------------------------------
# parse_unsupported_param
# ---------------------------------------------------------------------------

class TestParseUnsupportedParam:
    def test_reasoning_summary_is_drop(self):
        u = parse_unsupported_param(REASONING_SUMMARY_400)
        assert u is not None
        assert u.field_path == "reasoning.summary"
        assert u.action == "drop"
        assert u.rename_to is None

    def test_max_tokens_is_rename(self):
        u = parse_unsupported_param(MAX_TOKENS_400)
        assert u is not None
        assert u.field_path == "max_tokens"
        assert u.action == "rename"
        assert u.rename_to == "max_completion_tokens"

    def test_unknown_field_summary_is_drop(self):
        u = parse_unsupported_param(ARK_SUMMARY_400)
        assert u is not None
        assert u.field_path == "summary"
        assert u.action == "drop"
        assert u.rename_to is None

    def test_unknown_field_verbosity_is_drop(self):
        u = parse_unsupported_param(ARK_VERBOSITY_400)
        assert u is not None
        assert u.field_path == "verbosity"
        assert u.action == "drop"
        assert u.rename_to is None

    def test_unknown_field_summary_is_drop_escaped(self):
        # Raw JSON error body: the field name is backslash-escaped. The server
        # passes ``resp.text`` verbatim, so this must still parse.
        u = parse_unsupported_param(ARK_SUMMARY_400_ESCAPED)
        assert u is not None
        assert u.field_path == "summary"
        assert u.action == "drop"
        assert u.rename_to is None

    def test_unknown_field_verbosity_is_drop_escaped(self):
        u = parse_unsupported_param(ARK_VERBOSITY_400_ESCAPED)
        assert u is not None
        assert u.field_path == "verbosity"
        assert u.action == "drop"
        assert u.rename_to is None

    @pytest.mark.parametrize("text", [
        None,
        "",
        "Invalid 'input': value did not match any expected variant",
        "No tool output found for call 'abc'",
        "context length exceeded",
        "invalid api key",
        "some other 400 without the signature",
        "json: cannot unmarshal number into Go struct field max_output_tokens",
    ])
    def test_non_signature_returns_none(self, text):
        assert parse_unsupported_param(text) is None

    def test_case_insensitive(self):
        u = parse_unsupported_param("UNSUPPORTED PARAMETER: 'max_tokens'")
        assert u is not None
        assert u.field_path == "max_tokens"
        assert u.action == "rename"


# ---------------------------------------------------------------------------
# apply_unsupported_remediation
# ---------------------------------------------------------------------------

class TestApplyUnsupportedRemediation:
    def test_drop_top_level(self):
        body = {"model": "m", "max_tokens": 10, "input": "x"}
        u = UnsupportedParam(field_path="max_tokens", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert "max_tokens" not in new
        assert new["input"] == "x"
        # copy-on-write
        assert "max_tokens" in body

    def test_drop_nested(self):
        body = {"reasoning": {"effort": "low", "summary": "s"}, "input": "x"}
        u = UnsupportedParam(field_path="reasoning.summary", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert "summary" not in new["reasoning"]
        assert new["reasoning"]["effort"] == "low"
        # nested copy-on-write: original nested dict untouched
        assert "summary" in body["reasoning"]

    def test_drop_leaf_located_by_search(self):
        # ark names only the leaf ("summary"); locate it under `reasoning`.
        body = {"reasoning": {"effort": "low", "summary": "s"}, "input": "x"}
        u = UnsupportedParam(field_path="summary", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert "summary" not in new["reasoning"]
        assert new["reasoning"]["effort"] == "low"
        # copy-on-write: original nested dict untouched
        assert "summary" in body["reasoning"]

    def test_drop_leaf_top_level_wins_over_nested(self):
        # When the leaf name collides, the top-level key is the one dropped.
        body = {"verbosity": "medium", "nested": {"verbosity": "high"}}
        u = UnsupportedParam(field_path="verbosity", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert "verbosity" not in new
        assert new["nested"]["verbosity"] == "high"

    def test_drop_leaf_missing_no_change(self):
        body = {"input": "x"}
        u = UnsupportedParam(field_path="summary", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert not changed
        assert new is body

    def test_rename_preserves_value(self):
        body = {"max_tokens": 42}
        u = UnsupportedParam(field_path="max_tokens", action="rename",
                             rename_to="max_completion_tokens")
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert new["max_completion_tokens"] == 42
        assert "max_tokens" not in new
        assert body == {"max_tokens": 42}

    def test_rename_when_target_present_drops_source(self):
        body = {"max_tokens": 10, "max_completion_tokens": 20}
        u = UnsupportedParam(field_path="max_tokens", action="rename",
                             rename_to="max_completion_tokens")
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert new["max_completion_tokens"] == 20  # client's value preserved
        assert "max_tokens" not in new

    def test_missing_field_no_change(self):
        body = {"input": "x"}
        u = UnsupportedParam(field_path="reasoning.summary", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert not changed
        assert new is body

    def test_path_not_dict_no_change(self):
        body = {"reasoning": "not-a-dict"}
        u = UnsupportedParam(field_path="reasoning.summary", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert not changed
        assert new is body

    def test_non_dict_body_no_change(self):
        u = UnsupportedParam(field_path="a.b", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation([1, 2, 3], u)  # type: ignore[arg-type]
        assert not changed
        assert new == [1, 2, 3]

    def test_nested_copy_on_write_isolates_original(self):
        body = {"a": {"b": {"c": 1, "d": 2}}}
        u = UnsupportedParam(field_path="a.b.c", action="drop", rename_to=None)
        new, changed = apply_unsupported_remediation(body, u)
        assert changed
        assert "c" not in new["a"]["b"]
        # original untouched at every level
        assert body["a"]["b"] == {"c": 1, "d": 2}


# ---------------------------------------------------------------------------
# Arming predicates
# ---------------------------------------------------------------------------

class TestArmingPredicates:
    @pytest.mark.parametrize("transport,dialect", [
        ("bedrock", "openai-responses"),
        ("bedrock", "openai-chat"),
        ("http", "openai-responses"),
        ("http", "openai-chat"),
    ])
    def test_is_openai_compatible_model_true(self, transport, dialect):
        assert is_openai_compatible_model(transport, dialect)

    @pytest.mark.parametrize("transport,dialect", [
        ("azure", "openai-responses"),
        ("azure", "openai-chat"),
        ("bedrock", "anthropic"),
        ("bedrock", "openai-images"),
        ("bedrock", "openai-embeddings"),
        ("http", "anthropic-passthrough"),
        ("http", "openai-embeddings"),
    ])
    def test_is_openai_compatible_model_false(self, transport, dialect):
        assert not is_openai_compatible_model(transport, dialect)

    def test_bedrock_mantle_alias_tracks_openai_compatible(self):
        # The historical name is a backward-compatible alias; it now also arms
        # generic HTTP upstreams (ark) that exhibit the same divergence.
        assert is_bedrock_mantle_openai_model is is_openai_compatible_model
        assert is_bedrock_mantle_openai_model("http", "openai-responses")
        assert not is_bedrock_mantle_openai_model("azure", "openai-chat")

    def test_gpt_responses_model_gate_includes_gpt6(self):
        assert is_bedrock_gpt_responses_model("bedrock", "openai-responses", "openai.gpt-5.5")
        assert is_bedrock_gpt_responses_model("bedrock", "openai-responses", "openai.gpt-6-astra")
        assert not is_bedrock_gpt_responses_model("bedrock", "openai-responses", "xai.grok-4.3")
        assert not is_bedrock_gpt_responses_model("bedrock", "openai-chat", "openai.gpt-6-astra")

    def test_gpt5x_alias_still_works(self):
        # Backward-compatible alias for the historical name.
        assert is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "openai.gpt-6-astra")
        assert not is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "xai.grok-4.3")

    def test_max_strips_bounded_constant(self):
        assert MAX_UNSUPPORTED_STRIPS >= 1
