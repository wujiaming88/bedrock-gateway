"""
Tests for bedrock_gateway.auth — authentication module.
"""

import os
from unittest.mock import patch

import pytest

from bedrock_gateway.auth import AuthProvider, sign_v4
from bedrock_gateway.config import AuthConfig


# ─── Bearer Token ─────────────────────────────────────────────────────


class TestBearerTokenAuth:
    """Bearer token mode produces correct Authorization header."""

    def test_basic_bearer(self):
        cfg = AuthConfig(mode="bearer_token", bearer_token="tok_abc123")
        provider = AuthProvider(cfg, region="us-east-1")
        headers = provider.get_headers()
        assert headers["Authorization"] == "Bearer tok_abc123"
        assert headers["Content-Type"] == "application/json"

    def test_bearer_from_env(self):
        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "env_token"}, clear=False):
            cfg = AuthConfig(mode="bearer_token", bearer_token="")
            cfg.__post_init__()  # re-trigger env fallback
            provider = AuthProvider(cfg, region="us-east-1")
            headers = provider.get_headers()
            assert headers["Authorization"] == "Bearer env_token"

    def test_mode_property(self):
        cfg = AuthConfig(mode="bearer_token")
        provider = AuthProvider(cfg, region="us-east-1")
        assert provider.mode == "bearer_token"


# ─── SigV4 Signing ───────────────────────────────────────────────────


class TestSigV4:
    """Test standalone SigV4 signing function."""

    def test_produces_authorization_header(self):
        headers = sign_v4(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
            headers={"Content-Type": "application/json"},
            body=b'{"test": true}',
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="bedrock",
        )
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "x-amz-date" in headers
        assert "x-amz-content-sha256" in headers

    def test_with_session_token(self):
        headers = sign_v4(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
            headers={},
            body=b"{}",
            access_key="AKID",
            secret_key="SECRET",
            region="us-east-1",
            service="bedrock",
            session_token="SESSION_TOKEN_XYZ",
        )
        assert headers["x-amz-security-token"] == "SESSION_TOKEN_XYZ"

    def test_different_bodies_different_signatures(self):
        common = dict(
            method="POST",
            url="https://example.com/invoke",
            headers={},
            access_key="AK",
            secret_key="SK",
            region="us-east-1",
            service="bedrock",
        )
        h1 = sign_v4(body=b'{"a": 1}', **common)
        h2 = sign_v4(body=b'{"b": 2}', **common)
        # Content hashes must differ
        assert h1["x-amz-content-sha256"] != h2["x-amz-content-sha256"]


# ─── Credentials Mode ────────────────────────────────────────────────


class TestCredentialsAuth:
    """Credentials mode uses SigV4 signing."""

    def test_credentials_produces_sigv4(self):
        cfg = AuthConfig(
            mode="credentials",
            access_key_id="AKIAEXAMPLE",
            secret_access_key="SecretExample",
        )
        provider = AuthProvider(cfg, region="us-west-2")
        headers = provider.get_headers(
            method="POST",
            url="https://bedrock-runtime.us-west-2.amazonaws.com/model/test/invoke",
            body=b'{"messages": []}',
        )
        assert "Authorization" in headers
        assert "AWS4-HMAC-SHA256" in headers["Authorization"]
        assert "us-west-2/bedrock/aws4_request" in headers["Authorization"]

    def test_credentials_from_env(self):
        cfg = AuthConfig(mode="credentials")
        env = {
            "AWS_ACCESS_KEY_ID": "ENV_AK",
            "AWS_SECRET_ACCESS_KEY": "ENV_SK",
        }
        with patch.dict(os.environ, env):
            cfg.__post_init__()
            assert cfg.access_key_id == "ENV_AK"
            assert cfg.secret_access_key == "ENV_SK"


# ─── IAM Role / Profile ──────────────────────────────────────────────


class TestIAMAndProfileAuth:
    """IAM role and profile modes require boto3."""

    def test_iam_role_requires_boto3(self):
        cfg = AuthConfig(mode="iam_role")
        provider = AuthProvider(cfg, region="us-east-1")
        # If boto3 is not available, should raise ImportError with helpful message
        # If boto3 IS available, it would try to get credentials (and likely fail in test)
        # We test the error path by mocking the import
        import sys
        original = sys.modules.get("boto3")
        sys.modules["boto3"] = None  # type: ignore
        try:
            with pytest.raises((ImportError, TypeError)):
                provider.get_headers(
                    method="POST",
                    url="https://example.com/invoke",
                    body=b"{}",
                )
        finally:
            if original is not None:
                sys.modules["boto3"] = original
            else:
                sys.modules.pop("boto3", None)


# ─── Unknown Mode ─────────────────────────────────────────────────────


class TestUnknownMode:
    """Unknown auth mode raises ValueError."""

    def test_unknown_mode_raises(self):
        cfg = AuthConfig(mode="magic")
        provider = AuthProvider(cfg, region="us-east-1")
        with pytest.raises(ValueError, match="Unknown auth mode"):
            provider.get_headers()
