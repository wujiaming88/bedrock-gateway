"""Coverage tests for bedrock_gateway.models — get_info / get_max_output
resolution paths that aren't exercised by the main server tests."""

from __future__ import annotations

import pytest

from bedrock_gateway.config import GatewayConfig, ModelEntry
from bedrock_gateway.models import ModelInfo, ModelRegistry, UnknownModelError


@pytest.fixture
def registry() -> ModelRegistry:
    cfg = GatewayConfig(
        models={
            "claude-haiku": ModelEntry(
                bedrock_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                context_length=200_000,
                max_output=64_000,
            ),
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-model-v1",
                context_length=100_000,
                max_output=8_000,
            ),
        }
    )
    return ModelRegistry(cfg)


class TestGetInfo:
    def test_exact_match(self, registry: ModelRegistry):
        info = registry.get_info("claude-haiku")
        assert isinstance(info, ModelInfo)
        assert info.alias == "claude-haiku"
        assert info.bedrock_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert info.context_length == 200_000
        assert info.max_output == 64_000

    def test_alias_table_resolves(self, registry: ModelRegistry):
        # claude-3-5-haiku → claude-haiku via _MODEL_ALIASES
        info = registry.get_info("claude-3-5-haiku")
        assert info is not None
        # The alias is re-mapped to the canonical name
        assert info.alias == "claude-haiku"
        assert info.max_output == 64_000

    def test_unknown_returns_none(self, registry: ModelRegistry):
        assert registry.get_info("completely-unknown") is None

    def test_alias_without_registered_canonical(self):
        # Alias table points at a name that isn't in the registry → None
        cfg = GatewayConfig(models={})  # empty — no canonical entry
        reg = ModelRegistry(cfg)
        assert reg.get_info("claude-3-5-haiku") is None


class TestGetMaxOutput:
    def test_exact_match(self, registry: ModelRegistry):
        assert registry.get_max_output("claude-haiku") == 64_000

    def test_alias_table_resolves(self, registry: ModelRegistry):
        # claude-3-5-haiku alias → claude-haiku → 64_000
        assert registry.get_max_output("claude-3-5-haiku") == 64_000

    def test_alias_without_registered_canonical_uses_default(self):
        cfg = GatewayConfig(models={})
        reg = ModelRegistry(cfg)
        # Alias resolves to "claude-haiku" but it isn't registered → default
        assert reg.get_max_output("claude-3-5-haiku", default=123) == 123

    def test_unknown_uses_default(self, registry: ModelRegistry):
        assert registry.get_max_output("nope", default=999) == 999


class TestResolveUnknown:
    def test_raises_on_garbage(self, registry: ModelRegistry):
        with pytest.raises(UnknownModelError):
            registry.resolve("totally-not-a-model")

    def test_unknown_error_has_model_attr(self):
        err = UnknownModelError("x-y-z")
        assert err.model == "x-y-z"
        assert "x-y-z" in str(err)
