"""Bedrock GPT-5.x Responses request normalizer tests."""

from __future__ import annotations

import copy
import json

from bedrock_gateway.responses_normalizer import (
    is_bedrock_gpt5x_responses_model,
    normalize_bedrock_gpt5x_responses_request,
)


def test_model_gate_only_bedrock_gpt5x_responses():
    assert is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "openai.gpt-5.6-sol")
    assert is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "openai.gpt-5.5")
    assert not is_bedrock_gpt5x_responses_model("azure", "openai-responses", "gpt-5.6-sol")
    assert not is_bedrock_gpt5x_responses_model("bedrock", "openai-chat", "openai.gpt-5.6-sol")
    assert not is_bedrock_gpt5x_responses_model("bedrock", "openai-responses", "xai.grok-4.3")


def test_string_input_unchanged_but_copied():
    body = {"model": "openai.gpt-5.6-sol", "input": "hello", "stream": True}
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out == body
    assert out is not body


def test_additional_tools_lifted_to_top_level_tools():
    tool_a = {"type": "function", "name": "apply_patch", "description": "x"}
    tool_b = {"type": "function", "name": "shell", "description": "y"}
    body = {
        "model": "openai.gpt-5.6-sol",
        "tools": [tool_a],
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [tool_a, tool_b]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["tools"] == [tool_a, tool_b]
    assert out["input"] == [body["input"][1]]


def test_developer_messages_fold_into_instructions():
    body = {
        "model": "openai.gpt-5.6-sol",
        "instructions": "base",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev one"}]},
            {"type": "message", "role": "developer", "content": [{"type": "text", "text": "dev two"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "user"}]},
        ],
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["instructions"] == "base\n\ndev one\n\ndev two"
    assert len(out["input"]) == 1
    assert out["input"][0]["role"] == "user"


def test_text_blocks_are_normalized_to_input_text():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello", "other": 1}]},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    block = out["input"][0]["content"][0]
    assert block == {"type": "input_text", "text": "hello", "other": 1}


def test_non_message_items_preserved_unless_additional_tools():
    body = {
        "input": [
            {"type": "function_call_output", "call_id": "c", "output": "result"},
            "raw",
            123,
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"] == body["input"]


def test_existing_tools_non_list_replaced_by_lifted_tools():
    tool = {"type": "function", "function": {"name": "f"}}
    body = {"tools": {"bad": True}, "input": [{"type": "additional_tools", "tools": [tool]}]}
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["tools"] == [tool]


def test_developer_content_string_and_non_text_blocks():
    body = {
        "input": [
            {"type": "message", "role": "developer", "content": "string dev"},
            {"type": "message", "role": "developer", "content": [{"type": "input_image", "image_url": "x"}]},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["instructions"] == "string dev"
    assert out["input"] == []


def test_idempotent_and_does_not_mutate_input():
    body = {
        "input": [
            {"type": "additional_tools", "tools": [{"type": "function", "name": "f"}]},
            {"type": "message", "role": "developer", "content": [{"type": "text", "text": "dev"}]},
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "user"}]},
        ],
    }
    original = copy.deepcopy(body)
    once = normalize_bedrock_gpt5x_responses_request(body)
    twice = normalize_bedrock_gpt5x_responses_request(once)
    assert body == original
    assert once == twice


def test_no_secret_or_text_lost_except_expected_relocation():
    body = {
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "developer text"}]},
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "user text"}]},
        ],
        "reasoning": {"context": {"foo": "bar"}, "effort": "medium"},
        "stream": True,
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    rendered = json.dumps(out, ensure_ascii=False)
    assert "developer text" in out["instructions"]
    assert "user text" in rendered
    assert out["reasoning"] == {"context": "auto", "effort": "medium"}
    assert out["stream"] is True


def test_web_search_unsupported_content_types_removed():
    body = {
        "input": "find news",
        "tools": [
            {
                "type": "web_search",
                "external_web_access": True,
                "search_content_types": ["text", "images"],
            },
            {"type": "function", "name": "lookup", "search_content_types": ["keep"]},
        ],
    }
    original = copy.deepcopy(body)
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["tools"][0] == {"type": "web_search", "external_web_access": True}
    assert out["tools"][1] == body["tools"][1]
    assert body == original
    assert normalize_bedrock_gpt5x_responses_request(out) == out


def test_lifted_web_search_tool_is_normalized():
    body = {
        "input": [{
            "type": "additional_tools",
            "tools": [{
                "type": "web_search",
                "external_web_access": False,
                "search_content_types": ["text"],
            }],
        }]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["tools"] == [{"type": "web_search", "external_web_access": False}]


def test_nested_search_content_types_is_not_removed():
    body = {
        "input": [],
        "tools": [{
            "type": "web_search",
            "metadata": {"search_content_types": ["keep"]},
        }],
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["tools"][0]["metadata"] == {"search_content_types": ["keep"]}


def test_reasoning_context_supported_values_preserved_and_non_dict_untouched():
    for value in ["auto", "current_turn", "all_turns"]:
        body = {"input": [], "reasoning": {"context": value, "effort": "medium"}}
        assert normalize_bedrock_gpt5x_responses_request(body)["reasoning"]["context"] == value
    body = {"input": [], "reasoning": "medium"}
    assert normalize_bedrock_gpt5x_responses_request(body)["reasoning"] == "medium"


def test_reasoning_context_object_becomes_auto():
    body = {"input": [], "reasoning": {"context": {"summary": "x"}, "effort": "medium"}}
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["reasoning"] == {"context": "auto", "effort": "medium"}


def test_boundary_non_list_content_and_scalar_blocks_are_preserved():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "plain content"},
            {"type": "message", "role": "user", "content": ["scalar block"]},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"][0]["content"] == "plain content"
    assert out["input"][1]["content"] == ["scalar block"]


def test_boundary_developer_non_list_content_ignored_and_string_block_extracted():
    body = {
        "input": [
            {"type": "message", "role": "developer", "content": {"not": "text"}},
            {"type": "message", "role": "developer", "content": ["string block"]},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["instructions"] == "string block"
    assert out["input"] == []


def test_boundary_tool_dedup_keys_for_non_dict_function_and_key_only_tools():
    body = {
        "tools": ["scalar-tool", {"type": "function", "function": {"name": "fn"}}],
        "input": [
            {"type": "additional_tools", "tools": [
                "scalar-tool",
                {"type": "function", "function": {"name": "fn"}},
                {"type": "custom", "description": "no name"},
            ]}
        ],
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["tools"] == [
        "scalar-tool",
        {"type": "function", "function": {"name": "fn"}},
        {"type": "custom", "description": "no name"},
    ]


def test_invalid_encrypted_reasoning_item_is_dropped():
    body = {
        "input": [
            {"type": "reasoning", "encrypted_content": "enc_bad_foreign_blob"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"] == [body["input"][1]]


def test_valid_bedrock_encrypted_prefixes_get_required_summary():
    for prefix in ["rsn_", "smry_"]:
        item = {"type": "reasoning", "encrypted_content": prefix + "abc"}
        out = normalize_bedrock_gpt5x_responses_request({"input": [item]})
        assert out["input"] == [{**item, "summary": []}]


def test_invalid_nested_encrypted_content_is_removed_but_semantic_text_stays():
    body = {
        "input": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "visible", "encrypted_content": "foreign_blob"}
            ]}
        ],
        "reasoning": {"effort": "medium", "encrypted_content": "bad_blob"},
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert "encrypted_content" not in out["reasoning"]
    block = out["input"][0]["content"][0]
    assert block == {"type": "output_text", "text": "visible"}


def test_reasoning_with_foreign_blob_and_valid_summary_survives_without_blob():
    body = {"input": [{"type": "reasoning", "encrypted_content": "bad", "summary": []}]}
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"] == [{"type": "reasoning", "summary": []}]


def test_bedrock_reasoning_malformed_summaries_become_empty_list():
    malformed = [None, "text", {}, ["text"], [{}], [{"type": "text", "text": "x"}],
                 [{"type": "summary_text"}], [{"type": "summary_text", "text": 1}]]
    for summary in malformed:
        item = {
            "type": "reasoning",
            "id": "reasoning-id",
            "encrypted_content": "rsn_valid",
            "summary": summary,
            "future": "kept",
        }
        original = copy.deepcopy(item)
        out = normalize_bedrock_gpt5x_responses_request({"input": [item]})
        assert out["input"] == [{**item, "summary": []}]
        assert item == original
        assert normalize_bedrock_gpt5x_responses_request(out) == out


def test_valid_reasoning_summaries_are_preserved_with_extra_keys():
    summaries = [
        [],
        [{"type": "summary_text", "text": "brief", "future": True}],
        [
            {"type": "summary_text", "text": "one"},
            {"type": "summary_text", "text": "two"},
        ],
    ]
    for summary in summaries:
        item = {"type": "reasoning", "encrypted_content": "smry_valid", "summary": summary}
        out = normalize_bedrock_gpt5x_responses_request({"input": [item]})
        assert out["input"] == [item]


def test_reasoning_without_bedrock_state_requires_valid_summary():
    invalid_items = [
        {"type": "reasoning"},
        {"type": "reasoning", "id": "id"},
        {"type": "reasoning", "content": None},
        {
            "type": "reasoning",
            "id": "id",
            "namespace": "ns",
            "status": "completed",
            "phase": "analysis",
            "encrypted_content": "foreign_blob",
        },
        {"type": "reasoning", "summary": None},
        {"type": "reasoning", "summary": [{"type": "text", "text": "bad"}]},
    ]
    body = {
        "input": [
            *invalid_items,
            {"type": "reasoning", "id": "kept", "summary": []},
            {"type": "message", "role": "user", "content": "hello"},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"] == [
        {"type": "reasoning", "id": "kept", "summary": []},
        {"type": "message", "role": "user", "content": "hello"},
    ]


def test_nested_non_input_reasoning_metadata_is_not_dropped():
    body = {
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "hello",
                "metadata": {"type": "reasoning", "ticket": "kept"},
            }],
        }],
        "metadata": {"type": "reasoning", "ticket": "also-kept"},
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"][0]["content"][0]["metadata"] == {
        "type": "reasoning", "ticket": "kept"
    }
    assert out["metadata"] == {"type": "reasoning", "ticket": "also-kept"}


def test_mixed_history_keeps_custom_tools_and_drops_only_invalid_reasoning():
    body = {
        "input": [
            {"type": "reasoning", "id": "poison", "encrypted_content": "foreign"},
            {
                "type": "custom_tool_call",
                "id": "call-item",
                "call_id": "call-1",
                "name": "tool",
                "namespace": "functions",
                "status": "completed",
                "input": "{}",
            },
            {"type": "custom_tool_call_output", "id": "out", "call_id": "call-1", "output": "ok"},
            {"type": "reasoning", "encrypted_content": "rsn_good", "summary": None},
        ]
    }
    out = normalize_bedrock_gpt5x_responses_request(body)
    assert out["input"] == [body["input"][1], body["input"][2], {**body["input"][3], "summary": []}]
