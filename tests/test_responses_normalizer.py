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
