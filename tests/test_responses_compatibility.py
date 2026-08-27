"""Tests for the pure Responses compatibility module and its server fallback.

Covers the three layers in ``responses_compatibility.py`` (profile / analyzer /
projector), the exact-variant-400 one-time fallback in ``_handle_sync`` and
``_open_upstream_stream``, and the isolation guarantees (no fallback for
Azure/DeepSeek/Grok/other endpoints).
"""

from __future__ import annotations

import copy
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.auth import AuthProvider
from bedrock_gateway.config import (
    AuthConfig,
    AzureResource,
    GatewayConfig,
    RetryConfig,
    ServerConfig,
    _DEFAULT_MODELS,
    _parse_models,
)
from bedrock_gateway.responses_compatibility import (
    MANTLE_RESPONSES_PROFILE,
    CompatibilityPolicy,
    ProjectionResult,
    PROFILE_VERSION,
    analyze_history,
    is_bedrock_gpt5x_responses_model,
    is_exact_variant_rejection,
    project_mantle_input,
    responses_compat_policy,
)
from bedrock_gateway.server import (
    _compat_projection,
    _open_upstream_stream,
    _prepare_request_body,
    create_app,
)

VARIANT_400_TEXT = "invalid request body: Invalid 'input': value did not match any expected variant"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_version_declared(self):
        assert MANTLE_RESPONSES_PROFILE.version == PROFILE_VERSION

    def test_signatures_are_stable_categories(self):
        assert VARIANT_400_TEXT.split(": ", 1)[1].lower() in (
            signature.lower()
            for signature in MANTLE_RESPONSES_PROFILE.exact_variant_signatures
        )
        assert MANTLE_RESPONSES_PROFILE.relationship_signatures

    def test_policy_defaults_to_mantle_profile(self):
        assert CompatibilityPolicy().profile is MANTLE_RESPONSES_PROFILE


# ---------------------------------------------------------------------------
# Rejection classification
# ---------------------------------------------------------------------------

class TestIsExactVariantRejection:
    def test_exact_variant_400_matches(self):
        assert is_exact_variant_rejection(400, VARIANT_400_TEXT)

    @pytest.mark.parametrize("message", [
        "Invalid 'input' field",
        "bad request: invalid 'input' object",
        "Invalid 'input': missing tool output",
    ])
    def test_broad_invalid_input_messages_do_not_match(self, message):
        assert not is_exact_variant_rejection(400, message)

    def test_relationship_signatures_win(self):
        # Relationship errors may share the generic prefix but must NEVER fallback.
        assert not is_exact_variant_rejection(
            400, "No tool output found for call 'abc': invalid 'input'"
        )
        assert not is_exact_variant_rejection(400, "no tool call found for output")
        assert not is_exact_variant_rejection(400, "tool output without a matching call")
        assert not is_exact_variant_rejection(400, "tool call without an output")

    def test_non_400_never_matches(self):
        assert not is_exact_variant_rejection(422, VARIANT_400_TEXT)
        assert not is_exact_variant_rejection(500, VARIANT_400_TEXT)
        assert not is_exact_variant_rejection(429, VARIANT_400_TEXT)

    def test_empty_or_unrelated_text(self):
        assert not is_exact_variant_rejection(400, "")
        assert not is_exact_variant_rejection(400, None)
        assert not is_exact_variant_rejection(400, "context length exceeded")
        assert not is_exact_variant_rejection(400, "invalid api key")

    def test_case_insensitive(self):
        assert is_exact_variant_rejection(
            400, "INVALID 'INPUT': VALUE DID NOT MATCH ANY EXPECTED VARIANT"
        )


# ---------------------------------------------------------------------------
# Policy gating (no user switch; derived automatically)
# ---------------------------------------------------------------------------

class TestPolicyGating:
    @pytest.mark.parametrize("model", ["openai.gpt-5.5", "openai.gpt-5.6-sol",
                                       "openai.gpt-5.7"])
    def test_bedrock_gpt5x_armed(self, model):
        policy = responses_compat_policy("bedrock", "openai-responses", model)
        assert policy is not None
        assert isinstance(policy, CompatibilityPolicy)

    @pytest.mark.parametrize("transport,dialect,model", [
        ("azure", "openai-responses", "gpt-5.5"),
        ("http", "openai-responses", "deepseek-responses"),
        ("bedrock", "openai-responses", "xai.grok-4.3"),
        ("bedrock", "openai-responses", "xai.grok-4.6"),
        ("bedrock", "openai-chat", "openai.gpt-5.5"),
        ("bedrock", "anthropic", "openai.gpt-5.5"),
        ("bedrock", "openai-images", "openai.gpt-5.5"),
    ])
    def test_not_armed(self, transport, dialect, model):
        assert responses_compat_policy(transport, dialect, model) is None

    def test_model_gate_matches_policy(self):
        assert is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "openai.gpt-5.5")
        assert not is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "xai.grok-4.3")
        assert not is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "xai.grok-4.6")


# ---------------------------------------------------------------------------
# History analysis (value-free)
# ---------------------------------------------------------------------------

class TestAnalyzeHistory:
    def test_value_free(self):
        items = [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "TOP-SECRET-TEXT"},
            ]},
            {"type": "custom_tool_call", "call_id": "SECRET-CALL-ID", "name": "n",
             "input": "SECRET-ARGS"},
            {"type": "custom_tool_call_output", "call_id": "SECRET-CALL-ID",
             "output": "SECRET-OUTPUT"},
        ]
        rendered = json.dumps(analyze_history(items), ensure_ascii=False, sort_keys=True)
        for secret in ("TOP-SECRET-TEXT", "SECRET-CALL-ID", "SECRET-ARGS", "SECRET-OUTPUT"):
            assert secret not in rendered

    def test_relations_counted(self):
        items = [
            {"type": "custom_tool_call", "call_id": "c1", "name": "n", "input": "{}"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "n", "input": "{}"},
            {"type": "custom_tool_call_output", "call_id": "c3", "output": "orphan"},
            {"type": "custom_tool_call", "call_id": "dup", "name": "n", "input": "{}"},
            {"type": "custom_tool_call", "call_id": "dup", "name": "n", "input": "{}"},
            {"type": "custom_tool_call_output", "call_id": "dup", "output": "ok"},
        ]
        rel = analyze_history(items)["relations"]
        assert rel["calls"] == 4
        assert rel["outputs"] == 3
        assert rel["matched"] == 1           # c1
        assert rel["unmatched_calls"] == 1   # c2 (a call with no output)
        assert rel["orphan_outputs"] == 1    # c3
        assert rel["duplicate_call_ids"] == 1  # dup: 2 calls, 1 output

    def test_payload_and_status_enums(self):
        items = [
            {"type": "custom_tool_call", "call_id": "c", "name": "n",
             "input": {"a": 1}, "status": "queued"},
        ]
        analysis = analyze_history(items)
        assert analysis["payload_types"] == {"object": 1}
        assert analysis["status_enums"] == {"queued": 1}

    def test_unknown_item_types(self):
        # "unknown" = item with neither a ``type`` nor a ``role`` field.
        items = [{"type": "message", "role": "user", "content": "x"},
                 {"foo": 1}]
        analysis = analyze_history(items)
        assert analysis["unknown_item_types"] == {"count": 1, "first_index": 1}

    def test_non_list_returns_empty_structure(self):
        analysis = analyze_history("not-a-list")
        assert analysis["item_type_counts"] == {}
        assert analysis["relations"]["calls"] == 0


# ---------------------------------------------------------------------------
# Safe projection — the 8 exact-variant trigger shapes
# ---------------------------------------------------------------------------

class TestProjectionTriggerMatrix:
    """Each live-proven exact-variant shape must produce the right decision."""

    @pytest.mark.parametrize("item,decision,expected_changed", [
        ({"type": "agent_message", "text": "hi"}, "agent_message_to_user", True),
        ({"type": "local_shell_call", "call_id": "c"}, "unsafe_side_effect", False),
        ({"type": "code_interpreter_call", "call_id": "c"}, "unsafe_side_effect", False),
        ({"type": "input_text", "text": "bare"}, "wrap_text_item", True),
        ({"type": "custom_tool_call", "call_id": "c", "name": "t", "input": {"a": 1}},
         "coerce_input", True),
        ({"type": "function_call", "call_id": "c", "name": "t", "arguments": {"a": 1}},
         "coerce_arguments", True),
        ({"type": "custom_tool_call", "call_id": "c", "name": "t", "input": "{}",
          "status": "queued"}, "drop_status", True),
        ({"type": "custom_tool_call", "name": "t", "input": "{}"},
         "unsafe_missing_required", False),
    ])
    def test_each_shape_triggers_projection(self, item, decision, expected_changed):
        result = project_mantle_input({"input": [item]})
        assert decision in result.decisions
        assert result.changed is expected_changed

    def test_safe_shapes_only(self):
        """Lossless repairs with no dangling relation stay safe_to_retry."""
        cases = [
            [{"type": "input_text", "text": "bare"}],
            [{"type": "agent_message", "text": "hello world"}],
            [
                {"type": "custom_tool_call", "call_id": "c1", "name": "t",
                 "input": {"a": 1}},
                {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
            ],
            [
                {"type": "function_call", "call_id": "c1", "name": "t",
                 "arguments": [1, 2]},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
            [
                {"type": "custom_tool_call", "call_id": "c1", "name": "t", "input": "{}",
                 "status": "queued"},
                {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
            ],
        ]
        for items in cases:
            result = project_mantle_input({"input": items})
            assert result.changed is True, items
            assert result.safe_to_retry is True, items
            assert result.unsafe_reasons == (), items

    def test_unsafe_shapes(self):
        cases = [
            # side-effect items cannot be losslessly mapped
            [{"type": "local_shell_call", "call_id": "c"}],
            [{"type": "code_interpreter_call", "call_id": "c"}],
            # missing required field
            [{"type": "custom_tool_call", "name": "t", "input": "{}"}],
            # unmatched call
            [{"type": "custom_tool_call", "call_id": "c", "name": "t", "input": "{}"}],
            # orphan output
            [{"type": "custom_tool_call_output", "call_id": "c", "output": "ok"}],
            # unknown item with visible semantics
            [{"type": "mystery_item", "text": "visible"}],
        ]
        for items in cases:
            result = project_mantle_input({"input": items})
            assert result.safe_to_retry is False, items
            assert result.unsafe_reasons, items


# ---------------------------------------------------------------------------
# Projection details
# ---------------------------------------------------------------------------

class TestLegacyCompatibilityRulesInFallback:
    def test_additional_tools_and_developer_messages_are_folded(self):
        body = {
            "instructions": "base",
            "tools": [{"type": "function", "name": "existing"}],
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [
                    {"type": "function", "name": "existing"},
                    {"type": "function", "name": "shell"},
                ]},
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "dev rule"},
                ]},
                {"type": "agent_message", "text": "visible"},
            ],
        }
        result = project_mantle_input(body)
        assert result.safe_to_retry
        assert result.body["tools"] == [
            {"type": "function", "name": "existing"},
            {"type": "function", "name": "shell"},
        ]
        assert result.body["instructions"] == "base\n\ndev rule"
        assert result.body["input"] == [{
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "visible"}],
        }]
        assert result.decisions["lift_additional_tools"] == 1
        assert result.decisions["fold_developer_message"] == 1

    def test_malformed_additional_tools_and_developer_are_unsafe(self):
        for item, reason in [
            ({"type": "additional_tools", "tools": {}}, "additional_tools_not_list"),
            ({"type": "message", "role": "developer", "content": []},
             "developer_message_unmappable"),
        ]:
            result = project_mantle_input({"input": [item]})
            assert not result.safe_to_retry
            assert reason in result.unsafe_reasons

    def test_message_text_and_web_search_tool_legacy_rules(self):
        body = {
            "tools": [{
                "type": "web_search", "search_content_types": ["text"],
                "external_web_access": True,
            }],
            "input": [{
                "type": "message", "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }],
        }
        result = project_mantle_input(body)
        assert result.safe_to_retry
        assert result.body["input"][0]["content"][0]["type"] == "input_text"
        assert result.body["tools"] == [{
            "type": "web_search", "external_web_access": True,
        }]

    def test_top_level_reasoning_and_nested_foreign_opaque_are_normalized(self):
        body = {
            "reasoning": {"context": {"new": "shape"}, "effort": "high"},
            "input": [{
                "type": "message", "role": "user",
                "content": [{
                    "type": "input_text", "text": "visible",
                    "encrypted_content": "foreign",
                }],
            }],
        }
        result = project_mantle_input(body)
        assert result.safe_to_retry
        assert result.body["reasoning"]["context"] == "auto"
        assert "encrypted_content" not in result.body["input"][0]["content"][0]

    def test_item_reference_and_web_search_call_are_preserved(self):
        items = [
            {"type": "item_reference", "id": "ref"},
            {"type": "web_search_call", "id": "search", "status": "completed"},
            {"type": "agent_message", "text": "visible"},
        ]
        result = project_mantle_input({"input": items})
        assert result.safe_to_retry
        assert result.body["input"][:2] == items[:2]


class TestProjectionDetails:
    def test_object_arguments_stable_json_string(self):
        body = {"input": [
            {"type": "function_call", "call_id": "c", "name": "t",
             "arguments": {"b": 2, "a": [1, {"z": True}]}},
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
        ]}
        result = project_mantle_input(body)
        assert result.safe_to_retry
        encoded = result.body["input"][0]["arguments"]
        assert encoded == '{"a":[1,{"z":true}],"b":2}'  # sorted keys, compact

    def test_arguments_already_string_untouched(self):
        item = {"type": "function_call", "call_id": "c", "name": "t", "arguments": "x"}
        result = project_mantle_input({"input": [item]})
        assert result.changed is False  # only unmatched_call unsafe; but unchanged
        assert result.body["input"][0]["arguments"] == "x"

    def test_bare_text_wrap_role(self):
        result = project_mantle_input({"input": [
            {"type": "input_text", "text": "user says"},
            {"type": "output_text", "text": "assistant says"},
        ]})
        assert result.safe_to_retry
        items = result.body["input"]
        assert items[0] == {"type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": "user says"}]}
        assert items[1] == {"type": "message", "role": "assistant",
                            "content": [{"type": "input_text", "text": "assistant says"}]}

    def test_agent_message_safe_and_unsafe(self):
        safe = {"type": "agent_message", "content": [
            {"type": "input_text", "text": "a"},
            {"type": "output_text", "text": "b"},
        ]}
        result = project_mantle_input({"input": [safe]})
        assert result.safe_to_retry
        blocks = result.body["input"][0]["content"]
        assert [b["text"] for b in blocks] == ["a", "b"]

        for unsafe in [
            {"type": "agent_message", "content": [{"type": "input_image", "image_url": "x"}]},
            {"type": "agent_message", "content": [{"type": "binary", "data": "x"}]},
            {"type": "agent_message", "content": []},
            {"type": "agent_message"},
        ]:
            r = project_mantle_input({"input": [unsafe]})
            assert r.safe_to_retry is False, unsafe
            assert "agent_message" in r.unsafe_reasons[0]

    def test_agent_message_visible_text_survives_pure_opaque_block(self):
        item = {
            "type": "agent_message", "author": "agent", "recipient": "user",
            "content": [
                {"type": "input_text", "text": "visible"},
                {"type": "encrypted_content", "encrypted_content": "foreign"},
            ],
        }
        result = project_mantle_input({"input": [item]})
        assert result.safe_to_retry
        assert result.body["input"] == [{
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "visible"}],
        }]

    def test_agent_message_opaque_only_or_opaque_metadata_is_unsafe(self):
        for item, reason in [
            ({"type": "agent_message", "content": [
                {"type": "encrypted_content", "encrypted_content": "foreign"},
            ]}, "agent_message_no_visible_text"),
            ({"type": "agent_message", "content": [
                {"type": "input_text", "text": "visible"},
                {"type": "encrypted_content", "future": "unknown"},
            ]}, "agent_message_opaque_with_metadata"),
        ]:
            result = project_mantle_input({"input": [item]})
            assert not result.safe_to_retry
            assert reason in result.unsafe_reasons

    def test_namespace_phase_caller_preserved(self):
        body = {"input": [
            {"type": "message", "role": "assistant", "phase": "final_answer",
             "caller": "codex", "content": []},
            {"type": "custom_tool_call", "call_id": "c1", "name": "t",
             "namespace": "functions", "input": {"a": 1}},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
        ]}
        result = project_mantle_input(body)
        assert result.safe_to_retry
        items = result.body["input"]
        assert items[0]["phase"] == "final_answer"
        assert items[0]["caller"] == "codex"
        assert items[1]["namespace"] == "functions"

    def test_reasoning_summary_and_opaque_rules(self):
        # foreign opaque dropped; bedrock-prefixed with bad summary → empty summary
        body = {"input": [
            {"type": "reasoning", "encrypted_content": "foreign_blob", "summary": []},
            {"type": "reasoning", "encrypted_content": "rsn_valid", "summary": "bad"},
            {"type": "reasoning", "summary": []},
            {"type": "reasoning", "id": "id_only"},
        ]}
        result = project_mantle_input(body)
        assert result.safe_to_retry
        items = result.body["input"]
        # foreign blob dropped (kept summary), rsn summary emptied, summary kept, id-only dropped
        assert items == [
            {"type": "reasoning", "summary": []},
            {"type": "reasoning", "encrypted_content": "rsn_valid", "summary": []},
            {"type": "reasoning", "summary": []},
        ]

    def test_copy_on_write_and_idempotent(self):
        body = {"model": "openai.gpt-5.5",
                "input": [{"type": "input_text", "text": "hi"}]}
        original = copy.deepcopy(body)
        r1 = project_mantle_input(body)
        assert body == original
        assert r1.body is not body
        assert r1.changed and r1.safe_to_retry
        r2 = project_mantle_input(r1.body)
        assert r2.changed is False  # already canonical

    def test_non_list_input_unsafe(self):
        result = project_mantle_input({"input": "just a string"})
        assert result.changed is False
        assert result.safe_to_retry is False
        assert result.unsafe_reasons == ("input_not_list",)

    def test_result_is_frozen_dataclass(self):
        r = project_mantle_input({"input": []})
        assert isinstance(r, ProjectionResult)
        with pytest.raises(Exception):
            r.changed = True  # frozen


# ---------------------------------------------------------------------------
# Long histories
# ---------------------------------------------------------------------------

class TestLongHistories:
    def test_53_item_mixed_history(self):
        items = []
        for i in range(53):
            if i % 5 == 0:
                items.append({"type": "input_text", "text": f"bare {i}"})
            elif i % 3 == 0:
                items.append({"type": "message", "role": "user",
                              "content": [{"type": "input_text", "text": f"m {i}"}]})
            else:
                items.append({"type": "custom_tool_call", "call_id": f"c{i}",
                              "name": "t", "input": "{}"})
                items.append({"type": "custom_tool_call_output", "call_id": f"c{i}",
                              "output": "ok"})
        body = {"input": items}
        result = project_mantle_input(body)
        assert sum(result.item_counts.values()) == len(items)
        assert result.changed is True
        assert result.safe_to_retry is True

    def test_200_plus_message_history_no_change(self):
        items = [{"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": f"m{i}"}]}
                 for i in range(220)]
        result = project_mantle_input({"input": items})
        assert result.changed is False
        assert len(result.body["input"]) == 220
        assert sum(result.item_counts.values()) == 220


# ---------------------------------------------------------------------------
# Sync fallback state machine
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    cfg = GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=1, base_delay=0.001),
        models=_parse_models(_DEFAULT_MODELS),
    )
    return TestClient(create_app(cfg))


def _responses_body(model: str) -> dict:
    return {
        "id": "resp_x",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _err_resp(status: int, text: str) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


def _ok_resp(body: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = body
    return r


def _sync_inst(responses) -> AsyncMock:
    inst = AsyncMock()
    inst.post = AsyncMock(side_effect=list(responses))
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _nth_sent(mock_cls, n: int) -> dict:
    return json.loads(mock_cls.return_value.post.call_args_list[n].kwargs["content"])


class TestSyncFallbackStateMachine:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_safe_fallback_retries_once(self, mock_cls, client):
        mock_cls.return_value = _sync_inst([
            _err_resp(400, VARIANT_400_TEXT),
            _ok_resp(_responses_body("openai.gpt-5.5")),
        ])
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        assert resp.status_code == 200
        assert mock_cls.return_value.post.call_count == 2
        # first attempt: raw passthrough; second: projected (wrapped) body
        assert _nth_sent(mock_cls, 0)["input"] == [{"type": "input_text", "text": "hi"}]
        assert _nth_sent(mock_cls, 1)["input"] == [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_second_400_is_terminal(self, mock_cls, client):
        mock_cls.return_value = _sync_inst([
            _err_resp(400, VARIANT_400_TEXT),
            _err_resp(400, VARIANT_400_TEXT),
        ])
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        assert resp.status_code == 400
        assert mock_cls.return_value.post.call_count == 2  # no recursion

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_unsafe_projection_does_not_retry(self, mock_cls, client):
        mock_cls.return_value = _sync_inst([_err_resp(400, VARIANT_400_TEXT)])
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.5",
            "input": [{"type": "local_shell_call", "call_id": "c"}],
        })
        assert resp.status_code == 400
        assert mock_cls.return_value.post.call_count == 1

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_normal_request_is_single_call_and_verbatim(self, mock_cls, client):
        body = {
            "model": "gpt-5.5",
            "input": "hi",
            "reasoning": {"effort": "high"},
            "temperature": 0.4,
        }
        mock_cls.return_value = _sync_inst([_ok_resp(_responses_body("openai.gpt-5.5"))])
        resp = client.post("/openai/v1/responses", json=body)
        assert resp.status_code == 200
        assert mock_cls.return_value.post.call_count == 1
        sent = _nth_sent(mock_cls, 0)
        assert sent["model"] == "openai.gpt-5.5"
        assert sent["input"] == "hi"
        assert sent["reasoning"] == {"effort": "high"}
        assert sent["temperature"] == 0.4

    @pytest.mark.parametrize("status,text,expected_status", [
        (400, "No tool output found for call 'c'", 400),
        (400, "context length exceeded", 400),
        (400, "invalid api key", 400),
        (401, VARIANT_400_TEXT, 401),
        (422, VARIANT_400_TEXT, 422),
        (429, VARIANT_400_TEXT, 502),  # retryable, then exhausted (max_retries=1)
        (503, VARIANT_400_TEXT, 502),
        (500, VARIANT_400_TEXT, 500),
    ])
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_no_fallback_on_other_errors(self, mock_cls, client, status, text,
                                         expected_status):
        mock_cls.return_value = _sync_inst([_err_resp(status, text)])
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        assert resp.status_code == expected_status
        assert mock_cls.return_value.post.call_count == 1

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_no_fallback_on_upstream_timeout(self, mock_cls, client):
        import httpx
        inst = AsyncMock()
        inst.post = AsyncMock(side_effect=httpx.TimeoutException("boom"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst
        resp = client.post("/openai/v1/responses", json={
            "model": "gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        assert resp.status_code == 502
        # a timeout never reaches the compat path (no 400 to inspect)
        assert inst.post.call_count == 1


# ---------------------------------------------------------------------------
# Stream preflight fallback state machine (direct)
# ---------------------------------------------------------------------------

def _stream_resp(status: int, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status

    async def aiter_text():
        if text:
            yield text

    r.aiter_text = aiter_text
    return r


def _ctx(resp) -> AsyncMock:
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _stream_inst(responses) -> AsyncMock:
    inst = AsyncMock()
    # ``stream`` must return each async context manager synchronously (MagicMock,
    # not AsyncMock — the latter wraps the return in a coroutine).
    inst.stream = MagicMock(side_effect=[_ctx(r) for r in responses])
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    return inst


def _auth() -> AuthProvider:
    return AuthProvider(AuthConfig(mode="bearer_token", bearer_token="x"), "us-east-1")


class TestStreamPreflightStateMachine:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    async def test_safe_fallback_retries_once(self, mock_cls):
        inst = _stream_inst([
            _stream_resp(400, VARIANT_400_TEXT),
            _stream_resp(200),
        ])
        mock_cls.return_value = inst
        payload = _prepare_request_body({
            "model": "openai.gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        resp, stack, err = await _open_upstream_stream(
            "https://example/openai/v1/responses", payload, _auth(), 1, 0.001,
            request=None, health=None, log_tag="t", timeout=30.0,
            compat=CompatibilityPolicy(),
        )
        assert err is None and resp is not None
        assert inst.stream.call_count == 2
        second = json.loads(inst.stream.call_args_list[1].kwargs["content"])
        assert second["input"][0]["type"] == "message"
        await stack.aclose()

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    async def test_second_400_is_terminal(self, mock_cls):
        inst = _stream_inst([
            _stream_resp(400, VARIANT_400_TEXT),
            _stream_resp(400, VARIANT_400_TEXT),
        ])
        mock_cls.return_value = inst
        payload = _prepare_request_body({
            "model": "openai.gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        resp, stack, err = await _open_upstream_stream(
            "https://example/", payload, _auth(), 1, 0.001,
            request=None, health=None, log_tag="t", timeout=30.0,
            compat=CompatibilityPolicy(),
        )
        assert resp is None and stack is None
        assert err is not None and err["status"] == 400
        assert inst.stream.call_count == 2

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    async def test_unsafe_projection_does_not_retry(self, mock_cls):
        inst = _stream_inst([_stream_resp(400, VARIANT_400_TEXT)])
        mock_cls.return_value = inst
        payload = _prepare_request_body({
            "model": "openai.gpt-5.5",
            "input": [{"type": "local_shell_call", "call_id": "c"}],
        })
        resp, stack, err = await _open_upstream_stream(
            "https://example/", payload, _auth(), 1, 0.001,
            request=None, health=None, log_tag="t", timeout=30.0,
            compat=CompatibilityPolicy(),
        )
        assert resp is None and err is not None
        assert inst.stream.call_count == 1

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    async def test_no_compat_policy_no_fallback(self, mock_cls):
        inst = _stream_inst([_stream_resp(400, VARIANT_400_TEXT)])
        mock_cls.return_value = inst
        payload = _prepare_request_body({"input": [{"type": "input_text", "text": "hi"}]})
        resp, stack, err = await _open_upstream_stream(
            "https://example/", payload, _auth(), 1, 0.001,
            request=None, health=None, log_tag="t", timeout=30.0,
            compat=None,
        )
        assert resp is None and err is not None
        assert inst.stream.call_count == 1


# ---------------------------------------------------------------------------
# Isolation and regression
# ---------------------------------------------------------------------------

class TestIsolation:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_azure_same_400_text_does_not_fallback(self, mock_cls):
        resources = {"r1": AzureResource(
            base_url="https://az.example/openai/v1", api_key="az", prefix="azure"
        )}
        cfg = GatewayConfig(
            auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
            region="us-east-1",
            server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
            retry=RetryConfig(max_retries=1, base_delay=0.001),
            models=_parse_models(_DEFAULT_MODELS, resources),
            azure_resources=resources,
        )
        c = TestClient(create_app(cfg))
        inst = _sync_inst([_err_resp(400, VARIANT_400_TEXT)])
        mock_cls.return_value = inst
        resp = c.post("/openai/v1/responses", json={
            "model": "azure/gpt-5.5",
            "input": [{"type": "input_text", "text": "hi"}],
        })
        assert resp.status_code == 400
        assert inst.post.call_count == 1

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_endpoint_never_carries_compat(self, mock_cls, client):
        # /v1/messages → Responses translation is a different path; even a GPT-5.x
        # model there must not get the native-Responses compat fallback.
        inst = _sync_inst([_err_resp(400, VARIANT_400_TEXT)])
        mock_cls.return_value = inst
        resp = client.post("/v1/messages", json={
            "model": "gpt-5.5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert inst.post.call_count == 1

    def test_compat_projection_without_body_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bedrock_gateway"):
            assert _compat_projection(CompatibilityPolicy(), None, "model") is None
        assert "no_projection_body" in caplog.records[-1].getMessage()

    def test_compat_log_sentinel_has_no_secrets(self, caplog):
        body = {
            "model": "openai.gpt-5.5",
            "input": [{"type": "input_text", "text": "TOPSECRET-PAYLOAD"}],
        }
        with caplog.at_level(logging.INFO, logger="bedrock_gateway"):
            out = _compat_projection(CompatibilityPolicy(), body, "openai.gpt-5.5")
        assert out is not None
        records = [r for r in caplog.records if r.getMessage().startswith("COMPAT")]
        assert records
        for r in records:
            assert "TOPSECRET-PAYLOAD" not in r.getMessage()
            assert "input_text" not in r.getMessage()  # decisions are categories only
