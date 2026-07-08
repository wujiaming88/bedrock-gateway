"""
Configuration loader for Bedrock Gateway.

Supports YAML config files with environment variable interpolation,
environment variable overrides, and sensible defaults.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return _ENV_PATTERN.sub(_replacer, value)


def _deep_resolve(obj: Any) -> Any:
    """Recursively resolve environment variables in a nested structure."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _deep_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_resolve(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AuthConfig:
    """Authentication configuration."""
    mode: str = "bearer_token"  # bearer_token | credentials | iam_role | profile
    bearer_token: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    profile: str = ""

    def __post_init__(self) -> None:
        # Allow env-var fallbacks when fields are empty
        if self.mode == "bearer_token" and not self.bearer_token:
            self.bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
        if self.mode == "credentials":
            if not self.access_key_id:
                self.access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
            if not self.secret_access_key:
                self.secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            if not self.session_token:
                self.session_token = os.environ.get("AWS_SESSION_TOKEN", "")


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 4000
    log_level: str = "info"
    api_key: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("BEDROCK_API_KEY", "")


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    base_delay: float = 1.0


@dataclass
class StorageConfig:
    """Dashboard metrics persistence (SQLite) configuration."""
    enabled: bool = True
    path: str = "data/metrics.db"
    retain_days: int = 7


@dataclass
class DashboardConfig:
    """Dashboard (metrics UI + API) configuration.

    ``api_key`` is the dashboard's own authentication key — deliberately
    independent of ``server.api_key`` so that model-calling clients and
    dashboard operators can hold separate credentials.

    ``localhost_only`` is a tri-state: ``None`` means "auto" — the
    dashboard restricts itself to localhost when no ``dashboard.api_key``
    is configured, and is unrestricted otherwise. Set it explicitly to
    ``True`` / ``False`` in ``config.yaml`` to override.
    """
    enabled: bool = True
    require_auth: bool = True
    api_key: str | None = None
    localhost_only: bool | None = None
    rate_limit: int = 60
    max_request_log: int = 200
    storage: StorageConfig = field(default_factory=StorageConfig)

    def __post_init__(self) -> None:
        if not self.api_key:
            env_key = os.environ.get("BEDROCK_DASHBOARD_KEY", "")
            self.api_key = env_key or None


@dataclass
class AzureResource:
    """A single Azure OpenAI resource: its endpoint + api-key.

    Azure credentials are resource-scoped (each resource has its own endpoint
    and key; keys are not interchangeable across resources). One resource can
    host many deployments that all share this endpoint + key.

    ``base_url`` is the full Azure endpoint up to (but not including) the
    operation, e.g.::

        https://my-res.cognitiveservices.azure.com/openai/v1

    or the api-version-style base a specific resource exposes. The provider
    appends the operation path (``/responses``, ``/embeddings`` …) and any
    query string already present is preserved.

    ``prefix`` enables **passthrough-by-model-prefix**: when set (e.g.
    ``azure``), any client model of the form ``azure/<deployment>`` is routed
    to this resource with ``<deployment>`` as the Azure deployment name — no
    per-model config needed. The dialect is chosen by the endpoint the request
    hits (``/openai/v1/responses`` → responses, ``/v1/chat/completions`` →
    chat). Leave empty to disable prefix routing for this resource.
    """
    base_url: str
    api_key: str = ""
    prefix: str = ""


@dataclass
class ModelEntry:
    """A single model's metadata.

    The upstream is described on two orthogonal axes (see
    ``docs/multi-cloud-multimodal-design.md``):

      * ``transport`` — *where + how to auth*: ``bedrock`` (default) | ``azure``
      * ``dialect``   — *request/response/stream shape*: ``anthropic`` (default)
        | ``openai-responses`` | ``openai-chat`` | ``embeddings``

    ``endpoint`` is a transport-scoped hint (Bedrock: ``runtime`` | ``mantle``).

    **Backward compatibility**: the legacy ``protocol`` field is still accepted
    and mapped onto ``transport``/``dialect`` by :func:`_parse_models`, so
    existing models and flat YAML keep working unchanged.

    Azure models reference an :class:`AzureResource`; the loader resolves its
    endpoint + key onto ``azure_endpoint`` / ``azure_api_key`` and sets
    ``deployment`` (the name that goes into the request body's ``model`` field).
    """
    bedrock_id: str
    context_length: int = 200000
    max_output: int = 64000
    endpoint: str = "runtime"     # transport hint: "runtime" | "mantle"
    protocol: str = "anthropic"   # LEGACY — mapped to transport/dialect
    transport: str = "bedrock"    # "bedrock" | "azure"
    dialect: str = "anthropic"    # "anthropic" | "openai-responses" | "openai-chat" | "embeddings"
    # ── Azure-only fields (unused for Bedrock models) ──
    deployment: str = ""          # Azure deployment name → request body "model"
    azure_endpoint: str = ""      # resolved from AzureResource.base_url
    azure_api_key: str = ""       # resolved from AzureResource.api_key


@dataclass
class GatewayConfig:
    """Top-level configuration for the gateway."""
    auth: AuthConfig = field(default_factory=AuthConfig)
    region: str = "us-east-1"
    server: ServerConfig = field(default_factory=ServerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    azure_resources: dict[str, AzureResource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Env-var override for region
        self.region = os.environ.get("AWS_REGION", self.region)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    # ── Opus ──────────────────────────────────────────────────────────
    "claude-opus-4.8": {
        "bedrock_id": "us.anthropic.claude-opus-4-8",
        "context_length": 1_000_000,
        "max_output": 128_000,
    },
    "claude-opus-4.7": {
        "bedrock_id": "us.anthropic.claude-opus-4-7",
        "context_length": 1_000_000,
        "max_output": 128_000,
    },
    "claude-opus-4": {
        "bedrock_id": "us.anthropic.claude-opus-4-6-v1",
        "context_length": 1_000_000,
        "max_output": 128_000,
    },
    # ── Sonnet 4.x ───────────────────────────────────────────────────
    "claude-sonnet-4.6": {
        "bedrock_id": "us.anthropic.claude-sonnet-4-6",
        "context_length": 1_000_000,
        "max_output": 64_000,
    },
    "claude-sonnet-4": {
        "bedrock_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "context_length": 200_000,
        "max_output": 64_000,
    },
    # ── Haiku ─────────────────────────────────────────────────────────
    "claude-haiku": {
        "bedrock_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "context_length": 200_000,
        "max_output": 64_000,
    },
    # ── Sonnet 3.5 ────────────────────────────────────────────────────
    "claude-sonnet-3.5": {
        "bedrock_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "context_length": 200_000,
        "max_output": 64_000,
    },
    # ── OpenAI GPT-5.5 (mantle endpoint, Responses API) ───────────────
    "gpt-5.5": {
        "bedrock_id": "openai.gpt-5.5",
        "context_length": 272_000,
        "max_output": 64_000,
        "endpoint": "mantle",
        "protocol": "openai-responses",
    },
    # ── xAI Grok 4.3 (mantle endpoint, Responses API) ─────────────────
    "grok-4.3": {
        "bedrock_id": "xai.grok-4.3",
        "context_length": 1_000_000,
        "max_output": 131_072,
        "endpoint": "mantle",
        "protocol": "openai-responses",
    },
}

# Common model name variations → canonical alias
_MODEL_ALIASES: dict[str, str] = {
    # Opus variations
    "claude-opus": "claude-opus-4",
    "claude-4-opus": "claude-opus-4",
    "claude-3-opus": "claude-opus-4",
    "claude-3-opus-20240229": "claude-opus-4",
    # Sonnet 4 variations
    "claude-sonnet": "claude-sonnet-4",
    "claude-4-sonnet": "claude-sonnet-4",
    # Haiku variations
    "claude-3-haiku": "claude-haiku",
    "claude-3.5-haiku": "claude-haiku",
    "claude-3-5-haiku": "claude-haiku",
    "claude-haiku-3.5": "claude-haiku",
    "claude-haiku-3-5": "claude-haiku",
    "claude-4.5-haiku": "claude-haiku",
    "claude-4-5-haiku": "claude-haiku",
    "claude-haiku-4.5": "claude-haiku",
    "claude-haiku-4-5": "claude-haiku",
    "claude-3-5-haiku-20241022": "claude-haiku",
    # Sonnet 3.5 variations
    "claude-3.5-sonnet": "claude-sonnet-3.5",
    "claude-3-5-sonnet": "claude-sonnet-3.5",
    "claude-3-5-sonnet-v2": "claude-sonnet-3.5",
    "claude-3-5-sonnet-20241022": "claude-sonnet-3.5",
    "claude-sonnet-3-5": "claude-sonnet-3.5",
    # Anthropic API names (as sent by some SDKs)
    "claude-3-5-sonnet-latest": "claude-sonnet-3.5",
    "claude-3-5-haiku-latest": "claude-haiku",
    "claude-sonnet-4-0-20250514": "claude-sonnet-4",
    # Anthropic official model names (sent by Claude Code / Anthropic SDK)
    "claude-haiku-4-5-20251001": "claude-haiku",
    "claude-3-5-haiku-20251022": "claude-haiku",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-opus-4-20250115": "claude-opus-4",
    "claude-opus-4-7-20250428": "claude-opus-4.7",
    "claude-sonnet-4-6-20250627": "claude-sonnet-4.6",
    # Opus 4.8 variations
    "claude-opus-4-8": "claude-opus-4.8",
    "claude-4.8-opus": "claude-opus-4.8",
    "claude-4-8-opus": "claude-opus-4.8",
    # GPT-5.5 variations
    "gpt-55": "gpt-5.5",
    "gpt5.5": "gpt-5.5",
    "gpt-5-5": "gpt-5.5",
    "openai.gpt-5.5": "gpt-5.5",
    "openai-gpt-5.5": "gpt-5.5",
    # Grok 4.3 variations
    "grok": "grok-4.3",
    "grok-4": "grok-4.3",
    "grok4.3": "grok-4.3",
    "grok-4-3": "grok-4.3",
    "xai.grok-4.3": "grok-4.3",
    "xai-grok-4.3": "grok-4.3",
}


def _parse_azure_resources(
    raw: dict[str, Any] | None,
) -> dict[str, AzureResource]:
    """Parse the ``azure_resources`` section into AzureResource objects."""
    resources: dict[str, AzureResource] = {}
    for name, info in (raw or {}).items():
        if isinstance(info, dict):
            resources[name] = AzureResource(
                base_url=str(info.get("base_url", "")).rstrip("/"),
                api_key=str(info.get("api_key", "")),
                prefix=str(info.get("prefix", "")),
            )
    return resources


def _build_entry(
    name: str, info: dict[str, Any], resources: dict[str, AzureResource]
) -> ModelEntry:
    """Construct a single :class:`ModelEntry` from a raw config dict.

    Azure models reference a resource by name via ``azure_resource``; the
    referenced :class:`AzureResource`'s endpoint + key are resolved onto the
    entry here so downstream providers stay stateless. An unknown reference
    raises ``ValueError`` rather than silently producing an unauthenticated
    entry.
    """
    endpoint = info.get("endpoint", "runtime")
    protocol = info.get("protocol", "anthropic")
    transport, dialect = _resolve_axes(info, protocol)
    entry = ModelEntry(
        bedrock_id=info.get("bedrock_id", name),
        context_length=int(info.get("context_length", 200_000)),
        max_output=int(info.get("max_output", 64_000)),
        endpoint=endpoint,
        protocol=protocol,
        transport=transport,
        dialect=dialect,
    )
    # Resolve Azure resource reference → concrete endpoint + key.
    res_ref = info.get("azure_resource")
    if res_ref is not None:
        if res_ref not in resources:
            raise ValueError(
                f"model {name!r} references unknown azure_resource "
                f"{res_ref!r}; defined resources: {list(resources)}"
            )
        res = resources[res_ref]
        entry.azure_endpoint = res.base_url
        entry.azure_api_key = res.api_key
        entry.transport = "azure"
        # deployment defaults to the alias when omitted
        entry.deployment = str(info.get("deployment", name))
    return entry


def _parse_models(
    raw: dict[str, Any] | None,
    azure_resources: dict[str, AzureResource] | None = None,
    *,
    use_defaults: bool = True,
) -> dict[str, ModelEntry]:
    """Parse model entries, **merging** built-in defaults with user config.

    The built-in :data:`_DEFAULT_MODELS` form the base; entries from *raw* are
    added on top, an entry sharing a default's alias **overriding** it. This is
    additive: adding one custom model does not drop the Claude / GPT-5.5 / Grok
    defaults. Set ``use_defaults=False`` (config ``use_default_models: false``)
    to start from an empty base and expose *only* the configured models.
    """
    resources = azure_resources or {}
    models: dict[str, ModelEntry] = {}
    if use_defaults:
        for name, info in _DEFAULT_MODELS.items():
            models[name] = _build_entry(name, info, resources)
    for name, info in (raw or {}).items():
        if not isinstance(info, dict):
            continue
        models[name] = _build_entry(name, info, resources)  # override on clash
    return models


# Legacy ``protocol`` → (transport, dialect) mapping. Keeps existing configs
# and _DEFAULT_MODELS working after the two-axis split (see design doc §4.2).
_PROTOCOL_TO_AXES: dict[str, tuple[str, str]] = {
    "anthropic": ("bedrock", "anthropic"),
    "openai-responses": ("bedrock", "openai-responses"),
}


def _resolve_axes(info: dict[str, Any], protocol: str) -> tuple[str, str]:
    """Determine (transport, dialect) for a model entry.

    Explicit ``transport``/``dialect`` keys win; otherwise fall back to
    mapping the legacy ``protocol`` value. An Azure resource reference forces
    ``transport=azure`` (applied by the caller after resource resolution).
    """
    default_t, default_d = _PROTOCOL_TO_AXES.get(protocol, ("bedrock", protocol))
    return info.get("transport", default_t), info.get("dialect", default_d)


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """
    Load configuration from a YAML file with env-var interpolation.

    If *path* is ``None``, attempts ``config.yaml`` in CWD, then falls back
    to pure environment-variable / default configuration.
    """
    raw: dict[str, Any] = {}

    if path is None:
        candidate = Path("config.yaml")
        if candidate.exists():
            path = candidate

    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
            raw = _deep_resolve(raw)

    # Auth
    auth_raw = raw.get("auth", {})
    auth = AuthConfig(
        mode=auth_raw.get("mode", os.environ.get("BEDROCK_AUTH_MODE", "bearer_token")),
        bearer_token=auth_raw.get("bearer_token", ""),
        access_key_id=auth_raw.get("access_key_id", ""),
        secret_access_key=auth_raw.get("secret_access_key", ""),
        session_token=auth_raw.get("session_token", ""),
        profile=auth_raw.get("profile", ""),
    )

    # Server
    srv_raw = raw.get("server", {})
    server = ServerConfig(
        host=srv_raw.get("host", os.environ.get("BEDROCK_HOST", "127.0.0.1")),
        port=int(srv_raw.get("port", os.environ.get("BEDROCK_PORT", "4000"))),
        log_level=srv_raw.get("log_level", os.environ.get("BEDROCK_LOG_LEVEL", "info")),
        api_key=srv_raw.get("api_key", ""),
    )

    # Retry
    retry_raw = raw.get("retry", {})
    retry = RetryConfig(
        max_retries=int(retry_raw.get("max_retries", os.environ.get("BEDROCK_MAX_RETRIES", "3"))),
        base_delay=float(retry_raw.get("base_delay", "1.0")),
    )

    # Dashboard
    dash_raw = raw.get("dashboard", {})
    lh_raw = dash_raw.get("localhost_only", None)
    localhost_only: bool | None
    if lh_raw is None:
        localhost_only = None
    else:
        localhost_only = bool(lh_raw)
    dash_api_key_raw = dash_raw.get("api_key", None)
    storage_raw = dash_raw.get("storage", {}) or {}
    storage = StorageConfig(
        enabled=bool(storage_raw.get("enabled", True)),
        path=str(storage_raw.get("path", "data/metrics.db")),
        retain_days=int(storage_raw.get("retain_days", 7)),
    )
    dashboard = DashboardConfig(
        enabled=bool(dash_raw.get("enabled", True)),
        require_auth=bool(dash_raw.get("require_auth", True)),
        api_key=dash_api_key_raw if dash_api_key_raw else None,
        localhost_only=localhost_only,
        rate_limit=int(dash_raw.get("rate_limit", 60)),
        max_request_log=int(dash_raw.get("max_request_log", 200)),
        storage=storage,
    )

    # Azure resources (parsed before models so references can resolve)
    azure_resources = _parse_azure_resources(raw.get("azure_resources"))

    # Models — built-in defaults merged with user config (user overrides on
    # alias clash). ``use_default_models: false`` starts from an empty base.
    models = _parse_models(
        raw.get("models"),
        azure_resources,
        use_defaults=bool(raw.get("use_default_models", True)),
    )

    return GatewayConfig(
        auth=auth,
        region=raw.get("region", os.environ.get("AWS_REGION", "us-east-1")),
        server=server,
        retry=retry,
        dashboard=dashboard,
        models=models,
        azure_resources=azure_resources,
    )
