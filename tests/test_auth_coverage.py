"""Coverage tests for bedrock_gateway.auth — boto3-backed modes
(iam_role, profile) and the fail-path for missing boto3."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from bedrock_gateway.auth import AuthProvider
from bedrock_gateway.config import AuthConfig


class TestBoto3HeadersIAMRole:
    def test_iam_role_uses_boto3_session(self):
        # Fake out the boto3 imports that auth.py does lazily.
        fake_boto3 = MagicMock()
        fake_botocore_auth = MagicMock()
        fake_botocore_awsrequest = MagicMock()

        # ``sigv4.add_auth`` mutates request.headers in place.
        def _add_auth(request):
            request.headers["Authorization"] = (
                "AWS4-HMAC-SHA256 Credential=fake/..., "
                "SignedHeaders=..., Signature=abc"
            )
            request.headers["x-amz-date"] = "20260101T000000Z"

        sigv4 = MagicMock()
        sigv4.add_auth.side_effect = _add_auth
        fake_botocore_auth.SigV4Auth.return_value = sigv4

        class FakeAWSRequest:
            def __init__(self, method, url, data):
                self.method = method
                self.url = url
                self.data = data
                self.headers = {}

        fake_botocore_awsrequest.AWSRequest = FakeAWSRequest

        with patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "botocore": MagicMock(),
                "botocore.auth": fake_botocore_auth,
                "botocore.awsrequest": fake_botocore_awsrequest,
            },
        ):
            cfg = AuthConfig(mode="iam_role")
            provider = AuthProvider(cfg, region="us-west-2")
            headers = provider.get_headers(
                method="POST",
                url="https://bedrock-runtime.us-west-2.amazonaws.com/model/x/invoke",
                body=b'{"ok": true}',
            )
            # The forged SigV4Auth wrote into request.headers → our result.
            assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
            assert headers["Content-Type"] == "application/json"
            # Second call: client was cached → no new Session.
            headers2 = provider.get_headers(
                method="POST",
                url="https://bedrock-runtime.us-west-2.amazonaws.com/model/x/invoke",
                body=b"{}",
            )
            assert "Authorization" in headers2
            # boto3.Session was only instantiated once (client cached).
            assert fake_boto3.Session.call_count == 1

    def test_profile_mode_passes_profile_name(self):
        fake_boto3 = MagicMock()

        def _add_auth(request):
            request.headers["Authorization"] = "signed"

        sigv4 = MagicMock()
        sigv4.add_auth.side_effect = _add_auth

        class FakeAWSRequest:
            def __init__(self, method, url, data):
                self.headers = {}

        fake_botocore_auth = MagicMock()
        fake_botocore_auth.SigV4Auth.return_value = sigv4
        fake_botocore_awsrequest = MagicMock()
        fake_botocore_awsrequest.AWSRequest = FakeAWSRequest

        with patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "botocore": MagicMock(),
                "botocore.auth": fake_botocore_auth,
                "botocore.awsrequest": fake_botocore_awsrequest,
            },
        ):
            cfg = AuthConfig(mode="profile", profile="dev-sso")
            provider = AuthProvider(cfg, region="us-east-1")
            provider.get_headers(
                method="POST", url="https://example.com/model/x/invoke", body=b"{}"
            )
        # boto3.Session(profile_name="dev-sso") was called.
        fake_boto3.Session.assert_called_once_with(profile_name="dev-sso")

    def test_profile_mode_without_name_uses_default_session(self):
        fake_boto3 = MagicMock()

        def _add_auth(request):
            request.headers["Authorization"] = "signed"

        sigv4 = MagicMock()
        sigv4.add_auth.side_effect = _add_auth

        class FakeAWSRequest:
            def __init__(self, method, url, data):
                self.headers = {}

        fake_botocore_auth = MagicMock()
        fake_botocore_auth.SigV4Auth.return_value = sigv4
        fake_botocore_awsrequest = MagicMock()
        fake_botocore_awsrequest.AWSRequest = FakeAWSRequest

        with patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "botocore": MagicMock(),
                "botocore.auth": fake_botocore_auth,
                "botocore.awsrequest": fake_botocore_awsrequest,
            },
        ):
            cfg = AuthConfig(mode="profile", profile="")
            provider = AuthProvider(cfg, region="us-east-1")
            provider.get_headers(
                method="POST", url="https://example.com/model/x/invoke", body=b"{}"
            )
        # No profile_name kwarg → default session.
        fake_boto3.Session.assert_called_once_with()


class TestBoto3ImportFailure:
    def test_import_error_message_is_helpful(self):
        cfg = AuthConfig(mode="iam_role")
        provider = AuthProvider(cfg, region="us-east-1")

        # Block boto3 imports.
        with patch.dict(sys.modules, {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                provider.get_headers(
                    method="POST",
                    url="https://example.com/x",
                    body=b"{}",
                )


class TestAuthHeaders:
    """AuthHeaders dataclass is a container; trivial but covers import."""

    def test_auth_headers_carries_dict(self):
        from bedrock_gateway.auth import AuthHeaders

        ah = AuthHeaders(headers={"a": "b"})
        assert ah.headers == {"a": "b"}
