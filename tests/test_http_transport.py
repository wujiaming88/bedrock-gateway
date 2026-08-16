"""Tests for generic upstream resources and HTTP transport."""

import os
from unittest.mock import patch

import pytest

from bedrock_gateway.config import ModelEntry, _parse_upstream_resources, load_config
from bedrock_gateway.models import ModelRegistry
from bedrock_gateway.providers import HttpTransport, get_transport


ROUTE = {
    "prefix": "vendor",
    "secret_env": "VENDOR_TEST_SECRET",
    "routes": {
        "openai-responses": {
            "base_url": "https://api.example.test/v1/",
            "path": "/responses",
            "auth": "bearer",
            "default_headers": {"X-Client": "gateway"},
        },
        "openai-chat": {
            "base_url": "https://api.example.test/v1",
            "path": "/chat/completions",
            "auth": "x-api-key",
        },
    },
}


class TestUpstreamConfig:
    def test_parse_resource_and_routes(self):
        resource = _parse_upstream_resources({"primary": ROUTE})["primary"]
        assert resource.prefix == "vendor"
        assert resource.secret_env == "VENDOR_TEST_SECRET"
        assert resource.routes["openai-responses"].base_url == (
            "https://api.example.test/v1"
        )
        assert resource.routes["openai-responses"].default_headers == {
            "X-Client": "gateway"
        }

    @pytest.mark.parametrize(
        "change,match",
        [
            ({"prefix": "bad/value"}, "prefix"),
            ({"secret_env": ""}, "secret_env"),
            ({"routes": {}}, "routes"),
        ],
    )
    def test_invalid_resource_rejected(self, change, match):
        raw = dict(ROUTE)
        raw.update(change)
        with pytest.raises(ValueError, match=match):
            _parse_upstream_resources({"bad": raw})

    @pytest.mark.parametrize(
        "change,match",
        [
            ({"base_url": "ftp://example.test"}, "base_url"),
            ({"path": "responses"}, "path"),
            ({"auth": "basic"}, "auth"),
        ],
    )
    def test_invalid_route_rejected(self, change, match):
        raw = dict(ROUTE)
        raw["routes"] = {
            "openai-responses": {**ROUTE["routes"]["openai-responses"], **change}
        }
        with pytest.raises(ValueError, match=match):
            _parse_upstream_resources({"bad": raw})

    def test_explicit_model_resolves_generic_fields(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "use_default_models: false\n"
            "upstream_resources:\n"
            "  primary:\n"
            "    prefix: vendor\n"
            "    secret_env: VENDOR_TEST_SECRET\n"
            "    routes:\n"
            "      openai-responses:\n"
            "        base_url: https://api.example.test/v1\n"
            "        path: /responses\n"
            "        auth: bearer\n"
            "models:\n"
            "  named-model:\n"
            "    upstream_resource: primary\n"
            "    dialect: openai-responses\n"
            "    deployment: upstream-name\n"
        )
        entry = load_config(config_file).models["named-model"]
        assert entry.transport == "http"
        assert entry.deployment == "upstream-name"
        assert entry.upstream_base_url == "https://api.example.test/v1"
        assert entry.upstream_path == "/responses"
        assert entry.upstream_secret_env == "VENDOR_TEST_SECRET"

    @pytest.mark.parametrize(
        "raw,match",
        [
            ({"bad": "scalar"}, "must be a mapping"),
            ({"bad": {"prefix": "p", "secret_env": "S", "routes": "bad"}}, "routes must be"),
            ({"bad": {"prefix": "p", "secret_env": "S", "routes": {"openai-chat": "bad"}}}, "route .* mapping"),
            ({"bad": {"prefix": "p", "secret_env": "S", "routes": {"openai-chat": {"base_url": "https://x", "path": "/x", "default_headers": []}}}}, "default_headers"),
            ({"a": ROUTE, "b": ROUTE}, "duplicate"),
        ],
    )
    def test_resource_mapping_validation(self, raw, match):
        with pytest.raises(ValueError, match=match):
            _parse_upstream_resources(raw)

    def test_explicit_model_unknown_resource(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "use_default_models: false\nmodels:\n  x:\n"
            "    upstream_resource: missing\n    dialect: openai-chat\n"
        )
        with pytest.raises(ValueError, match="unknown upstream_resource"):
            load_config(config_file)

    def test_explicit_model_requires_matching_route(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "use_default_models: false\n"
            "upstream_resources:\n"
            "  primary:\n"
            "    prefix: vendor\n"
            "    secret_env: VENDOR_TEST_SECRET\n"
            "    routes:\n"
            "      openai-chat:\n"
            "        base_url: https://api.example.test/v1\n"
            "        path: /chat/completions\n"
            "models:\n"
            "  named-model:\n"
            "    upstream_resource: primary\n"
            "    dialect: openai-responses\n"
        )
        with pytest.raises(ValueError, match="has no route"):
            load_config(config_file)


class TestGenericPrefixResolution:
    def _registry(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "upstream_resources:\n"
            "  primary:\n"
            "    prefix: vendor\n"
            "    secret_env: VENDOR_TEST_SECRET\n"
            "    routes:\n"
            "      openai-responses:\n"
            "        base_url: https://api.example.test/v1\n"
            "        path: /responses\n"
            "        auth: bearer\n"
            "      openai-chat:\n"
            "        base_url: https://chat.example.test\n"
            "        path: /v2/chat\n"
            "        auth: x-api-key\n"
        )
        return ModelRegistry(load_config(config_file))

    def test_prefixed_resolution_selects_route_by_dialect(self, tmp_path):
        registry = self._registry(tmp_path)
        responses = registry.resolve_prefixed("vendor/model/a", "openai-responses")
        chat = registry.resolve_prefixed("vendor/model/a", "openai-chat")
        assert responses.transport == "http"
        assert responses.deployment == "model/a"
        assert responses.upstream_path == "/responses"
        assert chat.upstream_base_url == "https://chat.example.test"
        assert chat.upstream_auth == "x-api-key"

    def test_missing_dialect_route_does_not_resolve(self, tmp_path):
        registry = self._registry(tmp_path)
        assert registry.resolve_prefixed("vendor/model", "openai-images") is None
        assert registry.resolve_prefixed("vendor", "openai-chat") is None

    def test_explicit_model_missing_route_does_not_resolve(self, tmp_path):
        registry = self._registry(tmp_path)
        assert registry.resolve_for_dialect("gpt-5.5", "openai-chat") is None

    def test_explicit_http_model_missing_alternate_route(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "use_default_models: false\nupstream_resources:\n  primary:\n"
            "    prefix: vendor\n    secret_env: VENDOR_TEST_SECRET\n    routes:\n"
            "      openai-chat:\n        base_url: https://chat.example.test\n"
            "        path: /chat/completions\nmodels:\n  named:\n"
            "    upstream_resource: primary\n    dialect: openai-chat\n"
        )
        registry = ModelRegistry(load_config(config_file))
        assert registry.resolve_for_dialect("named", "openai-responses") is None

    def test_missing_resource_reference_on_runtime_entry_returns_none(self, tmp_path):
        registry = self._registry(tmp_path)
        entry = ModelEntry(
            bedrock_id="x", transport="http", dialect="openai-chat",
            upstream_resource="missing",
        )
        registry._models["broken"] = entry
        assert registry.resolve_for_dialect("broken", "openai-chat") is None


class TestHttpTransport:
    def _entry(self, auth="bearer"):
        return ModelEntry(
            bedrock_id="model",
            transport="http",
            upstream_base_url="https://api.example.test/v1",
            upstream_path="/responses",
            upstream_auth=auth,
            upstream_secret_env="VENDOR_TEST_SECRET",
            upstream_default_headers={"X-Client": "gateway"},
        )

    def test_registered_and_builds_configured_url(self):
        entry = self._entry()
        transport = get_transport(entry)
        assert isinstance(transport, HttpTransport)
        assert transport.build_url("/ignored", "ignored", entry) == (
            "https://api.example.test/v1/responses"
        )

    def test_reads_bearer_secret_at_request_time(self):
        transport = HttpTransport()
        entry = self._entry()
        with patch.dict(os.environ, {"VENDOR_TEST_SECRET": "first"}):
            assert transport.auth_headers(entry)["Authorization"] == "Bearer first"
        with patch.dict(os.environ, {"VENDOR_TEST_SECRET": "second"}):
            headers = transport.auth_headers(entry)
        assert headers == {
            "X-Client": "gateway",
            "Authorization": "Bearer second",
        }
        assert "first" not in repr(entry)

    def test_x_api_key_auth(self):
        with patch.dict(os.environ, {"VENDOR_TEST_SECRET": "key-value"}):
            headers = HttpTransport().auth_headers(self._entry("x-api-key"))
        assert headers["x-api-key"] == "key-value"
        assert "Authorization" not in headers

    def test_missing_secret_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            RuntimeError, match="VENDOR_TEST_SECRET"
        ):
            HttpTransport().auth_headers(self._entry())
