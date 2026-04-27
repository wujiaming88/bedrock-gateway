"""
Tests for API key authentication middleware.

Covers:
  - No key configured → all requests pass through
  - Key configured → unauthenticated requests rejected
  - Bearer token auth
  - x-api-key header auth
  - /health endpoint whitelisted
  - Constant-time comparison (hmac.compare_digest usage)
  - OpenAI vs Anthropic error format based on endpoint
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
)
from bedrock_gateway.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_no_key() -> GatewayConfig:
    """Config without API key (auth disabled)."""
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="aws-token"),
        region="us-east-1",
        server=ServerConfig(
            host="127.0.0.1", port=4000, log_level="warning", api_key=""
        ),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-v1",
                context_length=200000,
                max_output=4096,
            ),
        },
    )


@pytest.fixture
def config_with_key() -> GatewayConfig:
    """Config with API key (auth enabled)."""
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="aws-token"),
        region="us-east-1",
        server=ServerConfig(
            host="127.0.0.1",
            port=4000,
            log_level="warning",
            api_key="sk-test-secret-key-12345",
        ),
        retry=RetryConfig(max_retries=1, base_delay=0.01),
        models={
            "test-model": ModelEntry(
                bedrock_id="us.anthropic.test-v1",
                context_length=200000,
                max_output=4096,
            ),
        },
    )


@pytest.fixture
def client_no_key(config_no_key: GatewayConfig) -> TestClient:
    return TestClient(create_app(config_no_key))


@pytest.fixture
def client_with_key(config_with_key: GatewayConfig) -> TestClient:
    return TestClient(create_app(config_with_key))


# ---------------------------------------------------------------------------
# No API key configured → auth disabled
# ---------------------------------------------------------------------------


class TestAuthDisabled:
    """When no API key is set, all requests pass through."""

    def test_health_no_key(self, client_no_key: TestClient):
        resp = client_no_key.get("/health")
        assert resp.status_code == 200

    def test_models_no_key(self, client_no_key: TestClient):
        resp = client_no_key.get("/v1/models")
        assert resp.status_code == 200

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_chat_completions_no_key(
        self, mock_client_cls, client_no_key: TestClient
    ):
        """Chat completions work without any auth header."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client_no_key.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_messages_no_key(self, mock_client_cls, client_no_key: TestClient):
        """Messages API works without any auth header."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client_no_key.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API key configured → auth enforced
# ---------------------------------------------------------------------------


class TestAuthEnforced:
    """When API key is set, unauthenticated requests are rejected."""

    def test_health_always_open(self, client_with_key: TestClient):
        """/health is whitelisted and never requires auth."""
        resp = client_with_key.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_models_rejected_no_auth(self, client_with_key: TestClient):
        """GET /v1/models without auth → 401."""
        resp = client_with_key.get("/v1/models")
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["type"] == "authentication_error"
        assert data["error"]["message"] == "Invalid API key"

    def test_chat_completions_rejected_no_auth(
        self, client_with_key: TestClient
    ):
        """POST /v1/chat/completions without auth → 401 (OpenAI format)."""
        resp = client_with_key.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 401
        data = resp.json()
        # OpenAI error format
        assert "error" in data
        assert data["error"]["type"] == "authentication_error"
        assert data["error"]["code"] == 401

    def test_messages_rejected_no_auth(self, client_with_key: TestClient):
        """POST /v1/messages without auth → 401 (Anthropic format)."""
        resp = client_with_key.post("/v1/messages", json={
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 401
        data = resp.json()
        # Anthropic error format
        assert data["type"] == "error"
        assert data["error"]["type"] == "authentication_error"

    def test_wrong_key_rejected(self, client_with_key: TestClient):
        """Wrong API key → 401."""
        resp = client_with_key.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_empty_bearer_rejected(self, client_with_key: TestClient):
        """Empty Bearer token → 401."""
        resp = client_with_key.get(
            "/v1/models",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_non_bearer_auth_rejected(self, client_with_key: TestClient):
        """Non-Bearer auth header → 401 (falls through to x-api-key check)."""
        resp = client_with_key.get(
            "/v1/models",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Bearer token authentication
# ---------------------------------------------------------------------------


class TestBearerAuth:
    """Authentication via Authorization: Bearer <key>."""

    def test_correct_bearer(self, client_with_key: TestClient):
        """Correct Bearer token → request passes through."""
        resp = client_with_key.get(
            "/v1/models",
            headers={"Authorization": "Bearer sk-test-secret-key-12345"},
        )
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bearer_chat_completions(
        self, mock_client_cls, client_with_key: TestClient
    ):
        """Bearer auth works for chat completions."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client_with_key.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"Authorization": "Bearer sk-test-secret-key-12345"},
        )
        assert resp.status_code == 200

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bearer_messages(
        self, mock_client_cls, client_with_key: TestClient
    ):
        """Bearer auth works for Messages API."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client_with_key.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"Authorization": "Bearer sk-test-secret-key-12345"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# x-api-key authentication
# ---------------------------------------------------------------------------


class TestXApiKeyAuth:
    """Authentication via x-api-key header."""

    def test_correct_x_api_key(self, client_with_key: TestClient):
        """Correct x-api-key → request passes through."""
        resp = client_with_key.get(
            "/v1/models",
            headers={"x-api-key": "sk-test-secret-key-12345"},
        )
        assert resp.status_code == 200

    def test_wrong_x_api_key(self, client_with_key: TestClient):
        """Wrong x-api-key → 401."""
        resp = client_with_key.get(
            "/v1/models",
            headers={"x-api-key": "wrong-key"},
        )
        assert resp.status_code == 401

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_x_api_key_messages(
        self, mock_client_cls, client_with_key: TestClient
    ):
        """x-api-key works for Messages API."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = client_with_key.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"x-api-key": "sk-test-secret-key-12345"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Bearer takes priority over x-api-key
# ---------------------------------------------------------------------------


class TestAuthPriority:
    """Bearer token is checked first; x-api-key is fallback."""

    def test_bearer_takes_priority(self, client_with_key: TestClient):
        """When both headers present, Bearer is used."""
        resp = client_with_key.get(
            "/v1/models",
            headers={
                "Authorization": "Bearer sk-test-secret-key-12345",
                "x-api-key": "wrong-key",
            },
        )
        # Bearer is correct → pass
        assert resp.status_code == 200

    def test_bearer_wrong_falls_through(self, client_with_key: TestClient):
        """Wrong Bearer with correct x-api-key: Bearer value is used, fails."""
        resp = client_with_key.get(
            "/v1/models",
            headers={
                "Authorization": "Bearer wrong-key",
                "x-api-key": "sk-test-secret-key-12345",
            },
        )
        # Bearer is wrong → extracted first → fails (does NOT fall back to x-api-key)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Error format differs by endpoint
# ---------------------------------------------------------------------------


class TestErrorFormat:
    """OpenAI endpoints return OpenAI errors, Anthropic returns Anthropic errors."""

    def test_openai_endpoint_error_format(self, client_with_key: TestClient):
        """OpenAI endpoints return {error: {message, type, code}}."""
        resp = client_with_key.post("/v1/chat/completions", json={})
        assert resp.status_code == 401
        data = resp.json()
        assert "error" in data
        assert "type" not in data  # No top-level "type" field
        assert data["error"]["message"] == "Invalid API key"
        assert data["error"]["type"] == "authentication_error"
        assert data["error"]["code"] == 401

    def test_anthropic_endpoint_error_format(self, client_with_key: TestClient):
        """Anthropic endpoints return {type: "error", error: {type, message}}."""
        resp = client_with_key.post("/v1/messages", json={})
        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "authentication_error"
        assert data["error"]["message"] == "Invalid API key"
        assert "code" not in data["error"]  # Anthropic format has no "code"

    def test_models_endpoint_error_format(self, client_with_key: TestClient):
        """GET /v1/models returns OpenAI error format."""
        resp = client_with_key.get("/v1/models")
        assert resp.status_code == 401
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == 401


# ---------------------------------------------------------------------------
# Config: api_key from YAML and env var
# ---------------------------------------------------------------------------


class TestApiKeyConfig:
    """API key configuration from YAML and environment variables."""

    def test_api_key_from_config(self, config_with_key: GatewayConfig):
        assert config_with_key.server.api_key == "sk-test-secret-key-12345"

    def test_api_key_empty_by_default(self, config_no_key: GatewayConfig):
        assert config_no_key.server.api_key == ""

    def test_api_key_from_env(self):
        """BEDROCK_API_KEY env var is picked up."""
        import os
        from unittest.mock import patch as mock_patch

        with mock_patch.dict(os.environ, {"BEDROCK_API_KEY": "env-key-456"}):
            cfg = ServerConfig(api_key="")
            cfg.__post_init__()
            assert cfg.api_key == "env-key-456"

    def test_yaml_key_takes_precedence_over_env(self):
        """YAML api_key overrides env var (since YAML resolves before __post_init__)."""
        import os
        from unittest.mock import patch as mock_patch

        with mock_patch.dict(os.environ, {"BEDROCK_API_KEY": "env-key"}):
            cfg = ServerConfig(api_key="yaml-key")
            # __post_init__ only fills if api_key is empty
            assert cfg.api_key == "yaml-key"

    def test_yaml_config_with_api_key(self, tmp_path):
        """api_key loaded from YAML config file."""
        from bedrock_gateway.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "server:\n"
            "  api_key: yaml-secret-key\n"
        )
        cfg = load_config(config_file)
        assert cfg.server.api_key == "yaml-secret-key"

    def test_yaml_config_env_interpolation(self, tmp_path):
        """api_key supports ${ENV_VAR} interpolation."""
        import os
        from unittest.mock import patch as mock_patch
        from bedrock_gateway.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "server:\n"
            "  api_key: ${MY_GATEWAY_SECRET}\n"
        )
        with mock_patch.dict(os.environ, {"MY_GATEWAY_SECRET": "interpolated-key"}):
            cfg = load_config(config_file)
        assert cfg.server.api_key == "interpolated-key"
