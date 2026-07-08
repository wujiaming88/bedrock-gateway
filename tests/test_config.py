"""
Tests for bedrock_gateway.config — configuration loading.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bedrock_gateway.config import (
    AuthConfig,
    GatewayConfig,
    ModelEntry,
    load_config,
)


class TestEnvVarInterpolation:
    """Environment variable interpolation in YAML values."""

    def test_load_with_env_vars(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "auth:\n"
            "  mode: bearer_token\n"
            "  bearer_token: ${TEST_BG_TOKEN}\n"
            "region: us-west-2\n"
        )
        env_override = {"TEST_BG_TOKEN": "my-secret-token", "AWS_REGION": "us-west-2"}
        with patch.dict(os.environ, env_override):
            cfg = load_config(config_file)
        assert cfg.auth.bearer_token == "my-secret-token"
        assert cfg.region == "us-west-2"

    def test_missing_env_var_becomes_empty(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "auth:\n"
            "  mode: bearer_token\n"
            "  bearer_token: ${NONEXISTENT_VAR_XYZ}\n"
        )
        # Clear the env var fallback too
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEXISTENT_VAR_XYZ", None)
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            cfg = load_config(config_file)
        assert cfg.auth.bearer_token == ""


class TestDefaultConfig:
    """Default configuration when no file exists."""

    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            cfg = load_config("/nonexistent/config.yaml")
        assert cfg.auth.mode == "bearer_token"
        assert cfg.server.port == 4000
        assert cfg.retry.max_retries == 3
        assert len(cfg.models) > 0  # built-in defaults

    def test_default_models_include_claude(self):
        cfg = load_config("/nonexistent/config.yaml")
        assert "claude-haiku" in cfg.models
        assert "claude-opus-4" in cfg.models


class TestModelAliases:
    """Model alias table for common name variations."""

    def test_aliases_exist(self):
        from bedrock_gateway.config import _MODEL_ALIASES
        assert "claude-3.5-sonnet" in _MODEL_ALIASES
        assert "claude-3-5-haiku" in _MODEL_ALIASES
        assert "claude-opus" in _MODEL_ALIASES
        assert "claude-sonnet" in _MODEL_ALIASES

    def test_aliases_point_to_valid_defaults(self):
        from bedrock_gateway.config import _MODEL_ALIASES, _DEFAULT_MODELS
        for alias, canonical in _MODEL_ALIASES.items():
            assert canonical in _DEFAULT_MODELS, (
                f"Alias {alias!r} -> {canonical!r} not in _DEFAULT_MODELS"
            )


class TestModelParsing:
    """Model entry parsing from YAML."""

    def test_custom_models(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "models:\n"
            "  my-model:\n"
            "    bedrock_id: us.my-org.my-model-v1\n"
            "    context_length: 50000\n"
            "    max_output: 8192\n"
        )
        cfg = load_config(config_file)
        assert "my-model" in cfg.models
        entry = cfg.models["my-model"]
        assert entry.bedrock_id == "us.my-org.my-model-v1"
        assert entry.context_length == 50000
        assert entry.max_output == 8192

    def test_custom_models_merge_with_defaults(self, tmp_path: Path):
        """A ``models:`` section is ADDITIVE — custom models are merged on top
        of the built-in defaults, which remain available.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "models:\n"
            "  extra-model:\n"
            "    bedrock_id: us.x.y\n"
        )
        cfg = load_config(config_file)
        assert "extra-model" in cfg.models          # custom added
        assert "claude-haiku" in cfg.models          # defaults still present
        assert "gpt-5.5" in cfg.models

    def test_custom_model_overrides_default_on_clash(self, tmp_path: Path):
        """A custom entry sharing a default alias overrides it."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "models:\n"
            "  claude-haiku:\n"
            "    bedrock_id: us.custom.haiku-override\n"
        )
        cfg = load_config(config_file)
        assert cfg.models["claude-haiku"].bedrock_id == "us.custom.haiku-override"

    def test_use_default_models_false_disables_defaults(self, tmp_path: Path):
        """use_default_models: false → only the configured models are exposed."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "use_default_models: false\n"
            "models:\n"
            "  only-model:\n"
            "    bedrock_id: us.x.y\n"
        )
        cfg = load_config(config_file)
        assert set(cfg.models) == {"only-model"}
        assert "claude-haiku" not in cfg.models


class TestAuthConfig:
    """AuthConfig post-init env fallback."""

    def test_credentials_env_fallback(self):
        env = {
            "AWS_ACCESS_KEY_ID": "ak_from_env",
            "AWS_SECRET_ACCESS_KEY": "sk_from_env",
            "AWS_SESSION_TOKEN": "st_from_env",
        }
        with patch.dict(os.environ, env):
            cfg = AuthConfig(mode="credentials")
        assert cfg.access_key_id == "ak_from_env"
        assert cfg.secret_access_key == "sk_from_env"
        assert cfg.session_token == "st_from_env"


class TestSigV4TransportWarning:
    """SigV4 auth modes can't reach mantle/Azure — startup should warn (not
    fail, since runtime/Bedrock models still work)."""

    def test_credentials_mode_warns_on_mantle_model(self, tmp_path, caplog):
        import logging
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "auth:\n"
            "  mode: credentials\n"
            "  access_key_id: AKIA_x\n"
            "  secret_access_key: secret_x\n"
            "models:\n"
            "  gpt-5.5:\n"
            "    bedrock_id: openai.gpt-5.5\n"
            "    endpoint: mantle\n"
            "    protocol: openai-responses\n"
        )
        with caplog.at_level(logging.WARNING, logger="bedrock_gateway"):
            load_config(cfg)
        assert any("will NOT authenticate" in r.message for r in caplog.records)
        assert any("gpt-5.5" in r.message for r in caplog.records)

    def test_bearer_mode_no_warning(self, tmp_path, caplog):
        import logging
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "auth:\n"
            "  mode: bearer_token\n"
            "  bearer_token: tok\n"
            "models:\n"
            "  gpt-5.5:\n"
            "    bedrock_id: openai.gpt-5.5\n"
            "    endpoint: mantle\n"
            "    protocol: openai-responses\n"
        )
        with caplog.at_level(logging.WARNING, logger="bedrock_gateway"):
            load_config(cfg)
        assert not any("will NOT authenticate" in r.message for r in caplog.records)
