"""
Authentication module for Bedrock Gateway.

Supports four authentication modes:
  1. **bearer_token** — AWS Bearer Token (ABSK) via Authorization header
  2. **credentials** — AWS Access Key / Secret Key via SigV4 signing
  3. **iam_role** — Automatic IAM role from EC2/ECS/Lambda metadata
  4. **profile** — Named AWS CLI profile (uses boto3)

Each mode produces an ``httpx``-compatible set of headers (or an
``httpx.Auth`` instance for SigV4).
"""

from __future__ import annotations

import hashlib
import hmac
import datetime
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, quote

from .config import AuthConfig


# ---------------------------------------------------------------------------
# SigV4 Signer (standalone, no boto3 dependency for credentials mode)
# ---------------------------------------------------------------------------

def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(
    secret: str, date_stamp: str, region: str, service: str
) -> bytes:
    k_date = _sign(("AWS4" + secret).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


def sign_v4(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "bedrock",
    session_token: str = "",
) -> dict[str, str]:
    """
    Compute AWS Signature V4 headers and return the *additional* headers
    that must be merged into the request.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = quote(parsed.path or "/", safe="/")

    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    # Canonical request
    canonical_querystring = ""
    payload_hash = hashlib.sha256(body).hexdigest()

    signed_header_keys = ["host", "x-amz-content-sha256", "x-amz-date"]
    if session_token:
        signed_header_keys.append("x-amz-security-token")
    signed_header_keys.sort()

    canonical_headers_map = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if session_token:
        canonical_headers_map["x-amz-security-token"] = session_token

    canonical_headers = "".join(
        f"{k}:{canonical_headers_map[k]}\n" for k in signed_header_keys
    )
    signed_headers = ";".join(signed_header_keys)

    canonical_request = "\n".join([
        method,
        path,
        canonical_querystring,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    # String to sign
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        algorithm,
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    # Signature
    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        f"{algorithm} "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    result: dict[str, str] = {
        "Authorization": authorization,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }
    if session_token:
        result["x-amz-security-token"] = session_token
    return result


# ---------------------------------------------------------------------------
# Auth provider
# ---------------------------------------------------------------------------

@dataclass
class AuthHeaders:
    """Container for authentication headers to be applied to requests."""
    headers: dict[str, str]


class AuthProvider:
    """
    Produces authentication headers/credentials for outgoing Bedrock requests.

    Usage::

        provider = AuthProvider(config.auth, config.region)
        headers = provider.get_headers(method="POST", url=url, body=body_bytes)
    """

    def __init__(self, auth_config: AuthConfig, region: str) -> None:
        self._config = auth_config
        self._region = region
        self._boto3_client: Any = None  # lazy

    @property
    def mode(self) -> str:
        return self._config.mode

    def get_headers(
        self,
        method: str = "POST",
        url: str = "",
        body: bytes = b"",
    ) -> dict[str, str]:
        """Return headers dict that authenticates the request."""
        mode = self._config.mode

        if mode == "bearer_token":
            return {
                "Authorization": f"Bearer {self._config.bearer_token}",
                "Content-Type": "application/json",
            }

        if mode == "credentials":
            base = {"Content-Type": "application/json"}
            sig_headers = sign_v4(
                method=method,
                url=url,
                headers=base,
                body=body,
                access_key=self._config.access_key_id,
                secret_key=self._config.secret_access_key,
                region=self._region,
                service="bedrock",
                session_token=self._config.session_token,
            )
            base.update(sig_headers)
            return base

        if mode in ("iam_role", "profile"):
            return self._get_boto3_headers(method, url, body)

        raise ValueError(f"Unknown auth mode: {mode!r}")

    # ------------------------------------------------------------------
    # boto3-based auth (iam_role / profile)
    # ------------------------------------------------------------------

    def _get_boto3_headers(
        self, method: str, url: str, body: bytes
    ) -> dict[str, str]:
        """Use boto3/botocore to sign the request (IAM role or profile)."""
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for iam_role/profile auth mode. "
                "Install it with: pip install boto3"
            ) from exc

        if self._boto3_client is None:
            session_kwargs: dict[str, Any] = {}
            if self._config.mode == "profile" and self._config.profile:
                session_kwargs["profile_name"] = self._config.profile
            session = boto3.Session(**session_kwargs)
            self._boto3_client = session.client(
                "bedrock-runtime", region_name=self._region
            )

        credentials = (
            self._boto3_client._request_signer._credentials  # type: ignore[attr-defined]
        )
        request = AWSRequest(method=method, url=url, data=body)
        SigV4Auth(credentials, "bedrock", self._region).add_auth(request)

        headers = dict(request.headers)
        headers["Content-Type"] = "application/json"
        return headers
