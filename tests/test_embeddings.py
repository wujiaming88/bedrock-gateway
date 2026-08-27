"""
Unit tests for the embeddings provider layer (bedrock_gateway/embeddings.py).

Pure, transport-agnostic module — no server/config/mocking required. Covers the
OpenAI request parser, capability validation, the Cohere embed v4 document/query
and Titan Embeddings V2 adapters, the adapter registry, and float/base64
rendering.
"""

import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from bedrock_gateway.config import (
    AuthConfig,
    DashboardConfig,
    GatewayConfig,
    ModelEntry,
    RetryConfig,
    ServerConfig,
    load_config,
)
from bedrock_gateway.embeddings import (
    DEFAULT_REGISTRY,
    CohereEmbedV4DocumentAdapter,
    CohereEmbedV4DynamicAdapter,
    CohereEmbedV4QueryAdapter,
    EmbeddingsAdapter,
    EmbeddingsAdapterRegistry,
    EmbeddingsValidationError,
    EmbeddingRequestExtensionDecoder,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingTask,
    TaskPolicy,
    TitanEmbedV2Adapter,
    UnsupportedEmbeddingsModelError,
    decode_input_type,
    encode_base64,
    parse_request,
    prepare_request,
    render_openai,
    resolve_adapter,
    resolve_profile,
)
from bedrock_gateway.models import ModelRegistry
from bedrock_gateway.providers import get_dialect, get_transport
from bedrock_gateway.providers.dialect_embeddings import EmbeddingsPassthroughDialect
from bedrock_gateway.server import create_app


def _body(**overrides):
    body = {"model": "cohere.embed-v4:0", "input": "hello world"}
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# 1. Strict OpenAI request parser
# ---------------------------------------------------------------------------

class TestParseRequest:
    def test_string_input_is_normalised_to_single(self):
        ir = parse_request(_body(input="hello"))
        assert ir.inputs == ("hello",)
        assert ir.encoding_format == "float"
        assert ir.dimensions is None

    def test_list_of_strings(self):
        ir = parse_request(_body(input=["a", "b", "c"]))
        assert ir.inputs == ("a", "b", "c")

    def test_model_is_stripped(self):
        ir = parse_request(_body(model="  cohere.embed-v4  "))
        assert ir.model == "cohere.embed-v4"

    def test_base64_encoding_format(self):
        ir = parse_request(_body(encoding_format="base64"))
        assert ir.encoding_format == "base64"

    def test_dimensions_positive_int(self):
        ir = parse_request(_body(dimensions=512))
        assert ir.dimensions == 512

    def test_missing_model(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request({"input": "hello"})
        assert exc.value.param == "model"

    def test_empty_model(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(model="   "))

    def test_missing_input(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request({"model": "cohere.embed-v4:0"})
        assert exc.value.param == "input"

    def test_non_dict_body(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request("not a dict")  # type: ignore[arg-type]

    def test_empty_list_input(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(input=[]))

    def test_empty_string_input(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(input="   "))

    def test_list_with_empty_string(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(input=["ok", ""]))

    def test_token_array_rejected(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request(_body(input=[1, 2, 3]))
        assert exc.value.param == "input"

    def test_nested_token_array_rejected(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(input=[[1, 2], [3, 4]]))

    def test_mixed_input_rejected(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(input=["ok", 1]))

    def test_numeric_input_rejected(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(input=42))

    def test_invalid_encoding_format(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request(_body(encoding_format="int8"))
        assert exc.value.param == "encoding_format"

    def test_zero_dimensions_rejected(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(dimensions=0))

    def test_negative_dimensions_rejected(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(dimensions=-1))

    def test_bool_dimensions_rejected(self):
        # bool is an int subclass but must not pass as a dimension.
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(dimensions=True))

    def test_float_dimensions_rejected(self):
        with pytest.raises(EmbeddingsValidationError):
            parse_request(_body(dimensions=512.5))

    def test_user_is_accepted_and_unknown_fields_rejected(self):
        ir = parse_request(_body(user="alice"))
        assert ir.user == "alice"
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request(_body(extra="rejected"))
        assert exc.value.code == "unknown_parameter"
        assert exc.value.param == "extra"

    def test_input_type_is_accepted_and_decoded(self):
        ir = parse_request(_body(input_type="query"))
        assert ir.task is EmbeddingTask.RETRIEVAL_QUERY

    def test_input_type_document_decoded(self):
        ir = parse_request(_body(input_type="document"))
        assert ir.task is EmbeddingTask.RETRIEVAL_DOCUMENT

    def test_input_type_classification_and_clustering(self):
        assert parse_request(_body(input_type="classification")).task is EmbeddingTask.CLASSIFICATION
        assert parse_request(_body(input_type="clustering")).task is EmbeddingTask.CLUSTERING

    def test_input_type_is_case_and_space_insensitive(self):
        assert parse_request(_body(input_type="  QUERY ")).task is EmbeddingTask.RETRIEVAL_QUERY

    def test_input_type_absent_is_none(self):
        assert parse_request(_body()).task is None

    def test_invalid_input_type_rejected(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request(_body(input_type="bogus"))
        assert exc.value.param == "input_type"

    def test_empty_input_type_rejected(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request(_body(input_type="   "))
        assert exc.value.param == "input_type"

    def test_non_string_input_type_rejected(self):
        with pytest.raises(EmbeddingsValidationError) as exc:
            parse_request(_body(input_type=42))
        assert exc.value.param == "input_type"


class TestExtensionDecoderArchitecture:
    def test_openclaw_decoder_is_registered_without_provider_terms_in_ir(self):
        from bedrock_gateway import embeddings

        decoder = embeddings._EXTENSION_DECODERS["input_type"]
        assert isinstance(decoder, EmbeddingRequestExtensionDecoder)
        assert decoder.decode("query") is EmbeddingTask.RETRIEVAL_QUERY

    def test_new_decoder_can_map_another_wire_field_without_changing_ir(self):
        from bedrock_gateway import embeddings

        class TaskTypeDecoder(EmbeddingRequestExtensionDecoder):
            field = "task_type"

            def decode(self, value):
                if value != "RETRIEVAL_QUERY":
                    raise EmbeddingsValidationError("bad task", param=self.field)
                return EmbeddingTask.RETRIEVAL_QUERY

        original = dict(embeddings._EXTENSION_DECODERS)
        try:
            embeddings._EXTENSION_DECODERS["task_type"] = TaskTypeDecoder()
            ir = parse_request(_body(task_type="RETRIEVAL_QUERY"))
            assert ir.task is EmbeddingTask.RETRIEVAL_QUERY
        finally:
            embeddings._EXTENSION_DECODERS.clear()
            embeddings._EXTENSION_DECODERS.update(original)


class TestDecodeInputType:
    def test_maps_openclaw_wire_terms(self):
        assert decode_input_type("query") is EmbeddingTask.RETRIEVAL_QUERY
        assert decode_input_type("document") is EmbeddingTask.RETRIEVAL_DOCUMENT
        assert decode_input_type("classification") is EmbeddingTask.CLASSIFICATION
        assert decode_input_type("clustering") is EmbeddingTask.CLUSTERING

    def test_rejects_unknown_term(self):
        with pytest.raises(EmbeddingsValidationError):
            decode_input_type("search_document")


# ---------------------------------------------------------------------------
# 2. Capability validation
# ---------------------------------------------------------------------------

class TestCapabilityValidation:
    def test_titan_multiple_inputs_are_valid_for_fanout(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input=["a", "b"]))
        adapter.validate(ir)
        assert len(adapter.build_requests(ir)) == 2

    def test_titan_accepts_single_input(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input="a"))
        adapter.validate(ir)  # must not raise

    def test_titan_allowed_dimensions(self):
        adapter = TitanEmbedV2Adapter()
        for dim in (256, 512, 1024):
            ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", dimensions=dim))
            adapter.validate(ir)

    def test_titan_rejects_unsupported_dimension(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", dimensions=768))
        with pytest.raises(EmbeddingsValidationError) as exc:
            adapter.validate(ir)
        assert exc.value.param == "dimensions"

    def test_cohere_rejects_too_many_inputs(self):
        adapter = CohereEmbedV4DocumentAdapter()
        ir = parse_request(_body(input=[f"t{i}" for i in range(97)]))
        with pytest.raises(EmbeddingsValidationError):
            adapter.validate(ir)

    def test_cohere_accepts_96_inputs(self):
        adapter = CohereEmbedV4DocumentAdapter()
        ir = parse_request(_body(input=[f"t{i}" for i in range(96)]))
        adapter.validate(ir)

    def test_cohere_allowed_dimensions(self):
        adapter = CohereEmbedV4DocumentAdapter()
        for dim in (256, 512, 1024, 1536):
            ir = parse_request(_body(dimensions=dim))
            adapter.validate(ir)

    def test_cohere_rejects_non_multiple_of_32(self):
        adapter = CohereEmbedV4DocumentAdapter()
        ir = parse_request(_body(dimensions=100))
        with pytest.raises(EmbeddingsValidationError) as exc:
            adapter.validate(ir)
        assert exc.value.param == "dimensions"

    def test_cohere_rejects_out_of_range_dimension(self):
        adapter = CohereEmbedV4DocumentAdapter()
        ir = parse_request(_body(dimensions=8192))
        with pytest.raises(EmbeddingsValidationError):
            adapter.validate(ir)


# ---------------------------------------------------------------------------
# 2b. Task policy — fixed / accepted / symmetric
# ---------------------------------------------------------------------------

class TestTaskPolicy:
    def test_fixed_document_rejects_input_type(self):
        adapter = CohereEmbedV4DocumentAdapter()
        ir = parse_request(_body(input_type="query"))
        with pytest.raises(EmbeddingsValidationError) as exc:
            adapter.validate(ir)
        assert exc.value.param == "input_type"

    def test_fixed_query_rejects_input_type(self):
        adapter = CohereEmbedV4QueryAdapter()
        ir = parse_request(_body(input_type="document"))
        with pytest.raises(EmbeddingsValidationError):
            adapter.validate(ir)

    def test_fixed_accepts_when_no_input_type(self):
        CohereEmbedV4DocumentAdapter().validate(parse_request(_body()))
        CohereEmbedV4QueryAdapter().validate(parse_request(_body()))

    def test_dynamic_requires_input_type(self):
        adapter = CohereEmbedV4DynamicAdapter()
        ir = parse_request(_body())
        with pytest.raises(EmbeddingsValidationError) as exc:
            adapter.validate(ir)
        assert exc.value.param == "input_type"

    def test_dynamic_accepts_all_four_tasks(self):
        adapter = CohereEmbedV4DynamicAdapter()
        for value in ("document", "query", "classification", "clustering"):
            adapter.validate(parse_request(_body(input_type=value)))

    def test_dynamic_accepted_tasks_are_the_four_cohere_tasks(self):
        adapter = CohereEmbedV4DynamicAdapter()
        assert adapter.capabilities.task_policy is TaskPolicy.ACCEPTED
        assert adapter.capabilities.accepted_tasks == frozenset(
            {
                EmbeddingTask.RETRIEVAL_DOCUMENT,
                EmbeddingTask.RETRIEVAL_QUERY,
                EmbeddingTask.CLASSIFICATION,
                EmbeddingTask.CLUSTERING,
            }
        )

    def test_titan_symmetric_rejects_any_input_type(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input_type="query"))
        with pytest.raises(EmbeddingsValidationError) as exc:
            adapter.validate(ir)
        assert exc.value.param == "input_type"

    def test_titan_symmetric_accepts_no_input_type(self):
        adapter = TitanEmbedV2Adapter()
        adapter.validate(parse_request(_body(model="amazon.titan-embed-text-v2:0")))

    def test_policy_defaults_are_fixed(self):
        # An adapter that never opts in stays FIXED: it rejects input_type.
        class BareAdapter(EmbeddingsAdapter):
            name = "bare"
            def build_request(self, ir):
                return {}
            def render_response(self, native, ir):
                return EmbeddingResponse(model=ir.model, data=[], prompt_tokens=0, total_tokens=0)

        adapter = BareAdapter()
        assert adapter.capabilities.task_policy is TaskPolicy.FIXED
        with pytest.raises(EmbeddingsValidationError):
            adapter.validate(parse_request(_body(input_type="query")))


# ---------------------------------------------------------------------------
# 3. Cohere embed v4 — document adapter
# ---------------------------------------------------------------------------

class TestCohereDocumentAdapter:
    def setup_method(self):
        self.adapter = CohereEmbedV4DocumentAdapter()

    def test_build_request_without_dimensions(self):
        ir = parse_request(_body(input=["a", "b"]))
        body = self.adapter.build_request(ir)
        assert body == {
            "texts": ["a", "b"],
            "input_type": "search_document",
            "embedding_types": ["float"],
        }

    def test_build_request_with_output_dimension(self):
        ir = parse_request(_body(input="a", dimensions=512))
        body = self.adapter.build_request(ir)
        assert body["output_dimension"] == 512
        assert body["texts"] == ["a"]

    def test_render_list_embeddings(self):
        ir = parse_request(_body(input=["a", "b"]))
        response = self.adapter.render_response(
            {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}, ir
        )
        assert [d.index for d in response.data] == [0, 1]
        assert response.data[0].embedding == [0.1, 0.2]
        assert response.data[1].embedding == [0.3, 0.4]

    def test_render_dict_embeddings(self):
        ir = parse_request(_body(input="a"))
        response = self.adapter.render_response(
            {"embeddings": {"float": [[0.5, 0.6]]}}, ir
        )
        assert response.data[0].embedding == [0.5, 0.6]

    def test_usage_is_zero(self):
        ir = parse_request(_body(input="a"))
        response = self.adapter.render_response(
            {"embeddings": [[0.5, 0.6]], "meta": {"billed_units": {"input_tokens": 7}}},
            ir,
        )
        assert response.prompt_tokens == 0
        assert response.total_tokens == 0

    def test_missing_embeddings_raises(self):
        ir = parse_request(_body(input="a"))
        with pytest.raises(Exception):
            self.adapter.render_response({}, ir)

    def test_model_echoed(self):
        ir = parse_request(_body(input="a"))
        response = self.adapter.render_response({"embeddings": [[0.0]]}, ir)
        assert response.model == "cohere.embed-v4:0"


# ---------------------------------------------------------------------------
# 4. Cohere embed v4 — query adapter
# ---------------------------------------------------------------------------

class TestCohereQueryAdapter:
    def test_build_request_uses_search_query(self):
        adapter = CohereEmbedV4QueryAdapter()
        ir = parse_request(_body(input="a"))
        assert adapter.build_request(ir)["input_type"] == "search_query"

    def test_matches_query_model(self):
        adapter = CohereEmbedV4QueryAdapter()
        assert adapter.matches("cohere.embed-v4-query")
        assert adapter.matches("embed-v-4-0-query")

    def test_document_does_not_match_query_model(self):
        document = CohereEmbedV4DocumentAdapter()
        assert not document.matches("cohere.embed-v4-query")


# ---------------------------------------------------------------------------
# 4b. Cohere embed v4 — dynamic adapter (OpenClaw input_type)
# ---------------------------------------------------------------------------

class TestCohereDynamicAdapter:
    def setup_method(self):
        self.adapter = CohereEmbedV4DynamicAdapter()

    @pytest.mark.parametrize(
        "wire,native",
        [
            ("document", "search_document"),
            ("query", "search_query"),
            ("classification", "classification"),
            ("clustering", "clustering"),
        ],
    )
    def test_maps_openclaw_task_to_cohere_native(self, wire, native):
        ir = parse_request(_body(input="a", input_type=wire))
        assert self.adapter.build_request(ir)["input_type"] == native

    def test_requires_input_type_before_building(self):
        ir = parse_request(_body(input="a"))
        with pytest.raises(EmbeddingsValidationError):
            self.adapter.validate(ir)

    def test_output_dimension_applied(self):
        ir = parse_request(_body(input="a", input_type="query", dimensions=512))
        assert self.adapter.build_request(ir)["output_dimension"] == 512

    def test_does_not_claim_model_names(self):
        # The dynamic profile is selected explicitly by profile name; it never
        # claims a model name, so legacy name resolution stays fixed-document.
        assert not self.adapter.matches("cohere-embed-v4")
        assert not self.adapter.matches("cohere.embed-v4:0")


# ---------------------------------------------------------------------------
# 5. Amazon Titan Embeddings V2 adapter
# ---------------------------------------------------------------------------

class TestTitanAdapter:
    def test_build_request(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(
            _body(model="amazon.titan-embed-text-v2:0", input="hello", dimensions=512)
        )
        body = adapter.build_request(ir)
        assert body == {
            "inputText": "hello",
            "normalize": True,
            "embeddingTypes": ["float"],
            "dimensions": 512,
        }

    def test_build_request_omits_dimensions_when_unset(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input="hello"))
        body = adapter.build_request(ir)
        assert "dimensions" not in body

    def test_normalize_flag_configurable(self):
        adapter = TitanEmbedV2Adapter(normalize=False)
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input="hello"))
        assert adapter.build_request(ir)["normalize"] is False

    def test_render_response(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input="hello"))
        response = adapter.render_response(
            {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 4}, ir
        )
        assert response.data[0].index == 0
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.prompt_tokens == 4
        assert response.total_tokens == 4

    def test_missing_token_count_defaults_to_zero(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input="hello"))
        response = adapter.render_response({"embedding": [0.1]}, ir)
        assert response.prompt_tokens == 0
        assert response.total_tokens == 0

    def test_missing_embedding_raises(self):
        adapter = TitanEmbedV2Adapter()
        ir = parse_request(_body(model="amazon.titan-embed-text-v2:0", input="hello"))
        with pytest.raises(Exception):
            adapter.render_response({"inputTextTokenCount": 4}, ir)


# ---------------------------------------------------------------------------
# 6. Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_resolve_cohere_document(self):
        assert isinstance(
            resolve_adapter("cohere.embed-v4:0"), CohereEmbedV4DocumentAdapter
        )

    def test_resolve_cohere_query(self):
        assert isinstance(
            resolve_adapter("cohere.embed-v4-query"), CohereEmbedV4QueryAdapter
        )

    def test_resolve_titan(self):
        assert isinstance(
            resolve_adapter("amazon.titan-embed-text-v2:0"), TitanEmbedV2Adapter
        )

    def test_unknown_model_raises(self):
        with pytest.raises(UnsupportedEmbeddingsModelError) as exc:
            resolve_adapter("gpt-5.5")
        assert exc.value.model == "gpt-5.5"

    def test_custom_registry_and_adapter(self):
        class FakeAdapter(EmbeddingsAdapter):
            name = "fake"
            def matches(self, model: str) -> bool:
                return model == "fake-model"
            def build_request(self, ir):
                return {"texts": list(ir.inputs)}
            def render_response(self, native, ir):
                return EmbeddingResponse(
                    model=ir.model,
                    data=[EmbeddingData(index=0, embedding=native["embeddings"][0])],
                    prompt_tokens=0,
                    total_tokens=0,
                )

        registry = EmbeddingsAdapterRegistry()
        registry.register(FakeAdapter())
        assert isinstance(registry.resolve("fake-model"), FakeAdapter)
        assert len(registry) == 1

    def test_default_registry_len(self):
        assert len(DEFAULT_REGISTRY) == 4

    def test_resolve_dynamic_profile(self):
        assert isinstance(
            resolve_profile("cohere-embed-v4"), CohereEmbedV4DynamicAdapter
        )

    def test_resolve_document_profile(self):
        assert isinstance(
            resolve_profile("cohere-embed-v4-document"), CohereEmbedV4DocumentAdapter
        )

    def test_resolve_query_profile(self):
        assert isinstance(
            resolve_profile("cohere-embed-v4-query"), CohereEmbedV4QueryAdapter
        )

    def test_resolve_titan_profile(self):
        assert isinstance(resolve_profile("amazon-titan-embed-v2"), TitanEmbedV2Adapter)

    def test_unknown_profile_raises(self):
        with pytest.raises(UnsupportedEmbeddingsModelError):
            resolve_profile("unknown-profile")

    def test_raw_model_still_resolves_to_fixed_document(self):
        # The raw Bedrock id keeps resolving by name to the fixed document
        # adapter (not the dynamic profile, which needs an explicit profile).
        assert isinstance(
            resolve_adapter("cohere.embed-v4:0"), CohereEmbedV4DocumentAdapter
        )


# ---------------------------------------------------------------------------
# 7. float / base64 rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def _response(self):
        return EmbeddingResponse(
            model="cohere.embed-v4:0",
            data=[
                EmbeddingData(index=0, embedding=[1.0, 2.0, 3.0]),
                EmbeddingData(index=1, embedding=[4.0, 5.0]),
            ],
            prompt_tokens=3,
            total_tokens=3,
        )

    def test_float_rendering(self):
        body = render_openai(self._response(), "float")
        assert body["object"] == "list"
        assert body["model"] == "cohere.embed-v4:0"
        assert body["data"][0] == {
            "object": "embedding",
            "index": 0,
            "embedding": [1.0, 2.0, 3.0],
        }
        assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

    def test_base64_rendering_round_trips_float32(self):
        body = render_openai(self._response(), "base64")
        raw = base64.b64decode(body["data"][0]["embedding"])
        decoded = struct.unpack("<3f", raw)
        assert decoded == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)
        assert body["data"][1]["index"] == 1

    def test_encode_base64_matches_manual_pack(self):
        assert encode_base64([1.0, -2.5]) == base64.b64encode(
            struct.pack("<2f", 1.0, -2.5)
        ).decode("ascii")

    def test_invalid_encoding_format_raises(self):
        with pytest.raises(EmbeddingsValidationError):
            render_openai(self._response(), "int8")


# ---------------------------------------------------------------------------
# 8. One-shot request preparation (integration of the pure flow)
# ---------------------------------------------------------------------------

class TestPrepareRequest:
    def test_cohere_end_to_end_request_side(self):
        adapter, ir, native = prepare_request(
            {"model": "cohere.embed-v4:0", "input": ["a", "b"], "dimensions": 1024}
        )
        assert isinstance(adapter, CohereEmbedV4DocumentAdapter)
        assert ir.inputs == ("a", "b")
        assert native == {
            "texts": ["a", "b"],
            "input_type": "search_document",
            "embedding_types": ["float"],
            "output_dimension": 1024,
        }

    def test_titan_end_to_end_request_side(self):
        adapter, ir, native = prepare_request(
            {"model": "amazon.titan-embed-text-v2:0", "input": "hello", "dimensions": 256}
        )
        assert isinstance(adapter, TitanEmbedV2Adapter)
        assert native == {
            "inputText": "hello",
            "normalize": True,
            "embeddingTypes": ["float"],
            "dimensions": 256,
        }

    def test_full_round_trip_base64(self):
        body = {"model": "cohere.embed-v4:0", "input": "hi", "encoding_format": "base64"}
        adapter, ir, _native = prepare_request(body)
        response_ir = adapter.render_response({"embeddings": [[0.25, 0.75]]}, ir)
        rendered = render_openai(response_ir, ir.encoding_format)
        assert isinstance(rendered["data"][0]["embedding"], str)
        assert rendered["data"][0]["index"] == 0


# ---------------------------------------------------------------------------
# 9. Config — embedding_profile defaults + model entries + aliases
# ---------------------------------------------------------------------------

class TestEmbeddingsConfig:
    def test_default_models_carry_embedding_profile(self):
        cfg = load_config("/nonexistent/config.yaml")
        assert cfg.models["cohere-embed-v4-document"].bedrock_id == "cohere.embed-v4:0"
        assert (
            cfg.models["cohere-embed-v4-document"].embedding_profile
            == "cohere-embed-v4-document"
        )
        assert (
            cfg.models["cohere-embed-v4-query"].embedding_profile
            == "cohere-embed-v4-query"
        )
        assert cfg.models["cohere-embed-v4-query"].bedrock_id == "cohere.embed-v4:0"
        # The dynamic profile is a distinct default model sharing the Bedrock id.
        dynamic = cfg.models["cohere-embed-v4"]
        assert dynamic.bedrock_id == "cohere.embed-v4:0"
        assert dynamic.dialect == "openai-embeddings"
        assert dynamic.embedding_profile == "cohere-embed-v4"
        titan = cfg.models["titan-embed-text-v2"]
        assert titan.bedrock_id == "amazon.titan-embed-text-v2:0"
        assert titan.dialect == "openai-embeddings"
        assert titan.embedding_profile == "amazon-titan-embed-v2"
        assert cfg.models["cohere-embed-v4-document"].dialect == "openai-embeddings"

    def test_aliases_resolve_through_registry(self):
        registry = ModelRegistry(load_config("/nonexistent/config.yaml"))
        assert registry.resolve("cohere.embed-v4:0") == "cohere.embed-v4:0"
        assert registry.resolve("cohere-embed-v4-query") == "cohere.embed-v4:0"
        assert registry.resolve("embed-v4") == "cohere.embed-v4:0"
        # Bare "cohere-embed-v4" is now a registered *model* (dynamic profile),
        # not an alias — and the raw Bedrock id still resolves to the fixed
        # document alias.
        assert registry.resolve("cohere-embed-v4") == "cohere.embed-v4:0"
        assert (
            registry.get_entry("cohere.embed-v4:0").embedding_profile
            == "cohere-embed-v4-document"
        )
        assert (
            registry.get_entry("cohere-embed-v4").embedding_profile
            == "cohere-embed-v4"
        )
        assert (
            registry.resolve("amazon.titan-embed-text-v2:0")
            == "amazon.titan-embed-text-v2:0"
        )
        assert registry.get_entry("cohere.embed-v4:0").dialect == "openai-embeddings"
        assert (
            registry.get_entry("titan-embed-v2").dialect == "openai-embeddings"
        )

    def test_embedding_model_requires_profile(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "use_default_models: false\nmodels:\n  broken:\n"
            "    bedrock_id: cohere.embed-v4:0\n"
            "    dialect: openai-embeddings\n"
        )
        with pytest.raises(ValueError, match="requires an embedding_profile"):
            load_config(path)

    def test_embeddings_dialect_marker_registered(self):
        entry = ModelEntry(bedrock_id="cohere.embed-v4:0", dialect="openai-embeddings")
        dialect = get_dialect(entry)
        assert isinstance(dialect, EmbeddingsPassthroughDialect)
        assert dialect.name == "openai-embeddings"
        assert dialect.supports_stream is False
        assert dialect.operation_path(entry, False) == "/model/cohere.embed-v4:0/invoke"
        assert get_transport(entry).name == "bedrock"


# ---------------------------------------------------------------------------
# 10. Endpoint / fanout / errors / metrics (server integration)
# ---------------------------------------------------------------------------

def _embeddings_config() -> GatewayConfig:
    return GatewayConfig(
        auth=AuthConfig(mode="bearer_token", bearer_token="test-token"),
        region="us-east-1",
        server=ServerConfig(host="127.0.0.1", port=4000, log_level="warning"),
        retry=RetryConfig(max_retries=2, base_delay=0.01),
        dashboard=DashboardConfig(enabled=False),
        models={
            "cohere-embed-v4-document": ModelEntry(
                bedrock_id="cohere.embed-v4:0",
                dialect="openai-embeddings",
                embedding_profile="cohere-embed-v4-document",
            ),
            "cohere-embed-v4-query": ModelEntry(
                bedrock_id="cohere.embed-v4:0",
                dialect="openai-embeddings",
                embedding_profile="cohere-embed-v4-query",
            ),
            "cohere-embed-v4": ModelEntry(
                bedrock_id="cohere.embed-v4:0",
                dialect="openai-embeddings",
                embedding_profile="cohere-embed-v4",
            ),
            "titan-embed-text-v2": ModelEntry(
                bedrock_id="amazon.titan-embed-text-v2:0",
                dialect="openai-embeddings",
                embedding_profile="amazon-titan-embed-v2",
            ),
            "custom-embed": ModelEntry(
                bedrock_id="custom.embed-model",
                dialect="openai-embeddings",
                embedding_profile="unknown-profile",
            ),
            "chat-model": ModelEntry(bedrock_id="us.anthropic.chat-model-v1"),
        },
    )


@pytest.fixture
def emb_client() -> TestClient:
    return TestClient(create_app(_embeddings_config()))


def _sync_mock(client_cls, return_value):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = ""
    mock_response.json.return_value = return_value
    mock_instance = AsyncMock()
    mock_instance.post = AsyncMock(return_value=mock_response)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    client_cls.return_value = mock_instance
    return mock_instance


class TestEmbeddingsEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_cohere_document_round_trip(self, mock_client_cls, emb_client):
        mock_instance = _sync_mock(
            mock_client_cls, {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        )
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": ["a", "b"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["model"] == "cohere-embed-v4-document"
        assert [d["index"] for d in data["data"]] == [0, 1]
        assert data["data"][0]["embedding"] == [0.1, 0.2]

        call = mock_instance.post.call_args
        assert call.args[0].endswith("/model/cohere.embed-v4:0/invoke")
        sent = json.loads(call.kwargs["content"])
        assert sent["input_type"] == "search_document"
        assert sent["texts"] == ["a", "b"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_cohere_query_uses_search_query(self, mock_client_cls, emb_client):
        mock_instance = _sync_mock(mock_client_cls, {"embeddings": [[0.0]]})
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-query", "input": "hi"},
        )
        assert resp.status_code == 200
        sent = json.loads(mock_instance.post.call_args.kwargs["content"])
        assert sent["input_type"] == "search_query"

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_base64_encoding_format(self, mock_client_cls, emb_client):
        _sync_mock(mock_client_cls, {"embeddings": [[1.0, 2.0]]})
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi", "encoding_format": "base64"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"][0]["embedding"], str)

    def test_invalid_json(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_unknown_model_400(self, emb_client):
        resp = emb_client.post("/v1/embeddings", json={"model": "nope", "input": "x"})
        assert resp.status_code == 400

    def test_non_embeddings_dialect_400(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings", json={"model": "chat-model", "input": "x"}
        )
        assert resp.status_code == 400
        assert "not available on /v1/embeddings" in resp.json()["error"]["message"]

    def test_unknown_field_400(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "x", "bogus": 1},
        )
        assert resp.status_code == 400

    def test_unsupported_dimensions_400(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "titan-embed-text-v2", "input": "x", "dimensions": 768},
        )
        assert resp.status_code == 400

    def test_unknown_embeddings_model_404(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings", json={"model": "custom-embed", "input": "x"}
        )
        assert resp.status_code == 404


class TestOpenClawInputTypeEndpoint:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_exact_openclaw_query_request(self, mock_client_cls, emb_client):
        mock_instance = _sync_mock(mock_client_cls, {"embeddings": [[0.1, 0.2]]})
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4", "input": "hi", "input_type": "query"},
        )
        assert resp.status_code == 200
        sent = json.loads(mock_instance.post.call_args.kwargs["content"])
        assert sent["input_type"] == "search_query"
        assert sent["texts"] == ["hi"]

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @pytest.mark.parametrize(
        "wire,native",
        [
            ("document", "search_document"),
            ("classification", "classification"),
            ("clustering", "clustering"),
        ],
    )
    def test_openclaw_tasks_map_to_cohere_native(
        self, mock_client_cls, emb_client, wire, native
    ):
        mock_instance = _sync_mock(mock_client_cls, {"embeddings": [[0.5]]})
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4", "input": "hi", "input_type": wire},
        )
        assert resp.status_code == 200
        sent = json.loads(mock_instance.post.call_args.kwargs["content"])
        assert sent["input_type"] == native

    def test_fixed_document_rejects_input_type(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            json={
                "model": "cohere-embed-v4-document",
                "input": "x",
                "input_type": "query",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input_type"

    def test_fixed_query_rejects_input_type(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            json={
                "model": "cohere-embed-v4-query",
                "input": "x",
                "input_type": "document",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input_type"

    def test_dynamic_requires_input_type(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings", json={"model": "cohere-embed-v4", "input": "x"}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input_type"

    def test_invalid_input_type_400(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4", "input": "x", "input_type": "bogus"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input_type"

    def test_titan_symmetric_rejects_input_type(self, emb_client):
        resp = emb_client.post(
            "/v1/embeddings",
            json={
                "model": "titan-embed-text-v2",
                "input": "x",
                "input_type": "query",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input_type"


class TestEmbeddingsFanout:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_titan_fanout_order_and_usage(self, mock_client_cls, emb_client):
        async def post_side_effect(url=None, headers=None, content=None, **kwargs):
            text = json.loads(content)["inputText"]
            resp = MagicMock()
            resp.status_code = 200
            resp.text = ""
            resp.json.return_value = {
                "embedding": [float(len(text)), 0.0],
                "inputTextTokenCount": len(text),
            }
            return resp

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=post_side_effect)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "titan-embed-text-v2", "input": ["a", "bb", "ccc"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert [d["index"] for d in data["data"]] == [0, 1, 2]
        assert data["data"][0]["embedding"] == [1.0, 0.0]
        assert data["data"][1]["embedding"] == [2.0, 0.0]
        assert data["data"][2]["embedding"] == [3.0, 0.0]
        assert data["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
        assert mock_instance.post.call_count == 3

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_many_inputs_stay_ordered(self, mock_client_cls, emb_client):
        from bedrock_gateway import server

        assert server._EMBEDDINGS_MAX_CONCURRENCY == 8
        texts = [f"t{i}" for i in range(12)]

        async def post_side_effect(url=None, headers=None, content=None, **kwargs):
            text = json.loads(content)["inputText"]
            idx = int(text[1:])
            resp = MagicMock()
            resp.status_code = 200
            resp.text = ""
            resp.json.return_value = {
                "embedding": [float(idx)],
                "inputTextTokenCount": idx + 1,
            }
            return resp

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=post_side_effect)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "titan-embed-text-v2", "input": texts},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert [d["index"] for d in data["data"]] == list(range(12))
        assert [d["embedding"] for d in data["data"]] == [[float(i)] for i in range(12)]
        assert mock_instance.post.call_count == 12

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_all_or_nothing_on_item_failure(self, mock_client_cls, emb_client):
        async def post_side_effect(url=None, headers=None, content=None, **kwargs):
            text = json.loads(content)["inputText"]
            resp = MagicMock()
            if text == "bad":
                resp.status_code = 500
                resp.text = json.dumps({"message": "boom"})
            else:
                resp.status_code = 200
                resp.text = ""
                resp.json.return_value = {"embedding": [1.0], "inputTextTokenCount": 1}
            return resp

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=post_side_effect)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "titan-embed-text-v2", "input": ["ok", "bad"]},
        )
        assert resp.status_code == 500
        assert "error" in resp.json()

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_retry_on_429_then_success(self, mock_sleep, mock_client_cls, emb_client):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "rate limited"
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.text = ""
        mock_200.json.return_value = {"embeddings": [[0.5]]}

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=[mock_429, mock_200])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = emb_client.post(
            "/v1/embeddings", json={"model": "cohere-embed-v4-document", "input": "hi"}
        )
        assert resp.status_code == 200
        assert mock_instance.post.call_count == 2

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_timeout_retry_then_success(self, mock_sleep, mock_client_cls, emb_client):
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.text = ""
        mock_200.json.return_value = {"embeddings": [[0.5]]}

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(
            side_effect=[httpx.TimeoutException("slow"), mock_200]
        )
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = emb_client.post(
            "/v1/embeddings", json={"model": "cohere-embed-v4-document", "input": "hi"}
        )
        assert resp.status_code == 200
        assert mock_instance.post.call_count == 2

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_final_timeout_is_502(self, mock_client_cls, emb_client):
        instance = AsyncMock()
        instance.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi"},
        )
        assert resp.status_code == 502

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_final_connection_error_is_502(self, mock_client_cls, emb_client):
        instance = AsyncMock()
        instance.post = AsyncMock(side_effect=httpx.ConnectError("down"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi"},
        )
        assert resp.status_code == 502

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_connection_error_retry_then_success(
        self, mock_sleep, mock_client_cls, emb_client
    ):
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.text = ""
        mock_200.json.return_value = {"embeddings": [[0.5]]}
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(
            side_effect=[httpx.ConnectError("down"), mock_200]
        )
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi"},
        )
        assert resp.status_code == 200
        assert mock_instance.post.call_count == 2

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_bad_upstream_json_is_502(self, mock_client_cls, emb_client):
        response = MagicMock()
        response.status_code = 200
        response.json.side_effect = ValueError("bad json")
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi"},
        )
        assert resp.status_code == 502

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    @patch("bedrock_gateway.server.asyncio.sleep", new_callable=AsyncMock)
    def test_all_retries_exhausted_502(self, mock_sleep, mock_client_cls, emb_client):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "rate limited"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_429)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        resp = emb_client.post(
            "/v1/embeddings", json={"model": "cohere-embed-v4-document", "input": "hi"}
        )
        assert resp.status_code == 502
        assert mock_instance.post.call_count == 2


class TestEmbeddingsUnexpectedAndRenderErrors:
    @patch("bedrock_gateway.server._embeddings_attempt", new_callable=AsyncMock)
    def test_unexpected_task_exception_is_500(self, attempt, emb_client):
        attempt.side_effect = RuntimeError("unexpected")
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi"},
        )
        assert resp.status_code == 500

    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_mismatched_native_response_is_502(self, mock_client_cls, emb_client):
        _sync_mock(mock_client_cls, {"unexpected": []})
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "cohere-embed-v4-document", "input": "hi"},
        )
        assert resp.status_code == 502


class TestEmbeddingsMetrics:
    @patch("bedrock_gateway.server.httpx.AsyncClient")
    def test_request_recorded_with_tokens(self, mock_client_cls, emb_client):
        mock_instance = _sync_mock(
            mock_client_cls, {"embedding": [0.1, 0.2], "inputTextTokenCount": 4}
        )
        resp = emb_client.post(
            "/v1/embeddings",
            json={"model": "titan-embed-text-v2", "input": "hello"},
        )
        assert resp.status_code == 200

        records = emb_client.app.state.metrics.recent_requests()
        assert records
        rec = records[0]
        assert rec["path"] == "/v1/embeddings"
        assert rec["model"] == "titan-embed-text-v2"
        assert rec["status"] == 200
        assert rec["prompt_tokens"] == 4

