"""
Embeddings provider layer — OpenAI-compatible ``/v1/embeddings`` → Bedrock-native.

Pure, transport-agnostic module. It parses a client's OpenAI embeddings request
into an intermediate representation (IR), validates it against the resolved
provider adapter's *capabilities*, builds the native upstream request body, and
renders the native response back into the OpenAI shape — as ``float`` vectors or
``base64``-encoded float32, per the client's ``encoding_format``.

Everything provider-specific — body building, response rendering, capability
constraints, and model matching — lives inside an :class:`EmbeddingsAdapter`.
The :class:`EmbeddingsAdapterRegistry` is the *only* place that picks an adapter
for a model; no caller ever branches on a provider name.

This is the "shape" half of the ``embeddings`` dialect (see
``docs/multi-cloud-multimodal-design.md``). The transport/server half — URL
assembly, auth, the model-id swap — is deliberately out of scope here.

Supported upstreams:

  * **Cohere embed v4** (``embed-v4.0`` / ``cohere.embed-v4``) — *document*,
    *query*, *classification*, and *clustering* tasks. Cohere's ``input_type``
    has no OpenAI equivalent, so the gateway carries a provider-neutral
    :class:`EmbeddingTask` in the IR. The fixed ``-document`` / ``-query``
    aliases keep serving standard OpenAI clients (no ``input_type`` on the
    wire); the dynamic ``cohere-embed-v4`` profile accepts an OpenClaw
    ``input_type`` extension and maps each task to Cohere's native term
    (``search_document`` / ``search_query`` / ``classification`` /
    ``clustering``).
  * **Amazon Titan Embeddings V2** (``amazon.titan-embed-text-v2:0``) — a single
    symmetric text embedding per request, optional ``dimensions`` in
    {256, 512, 1024}. Symmetric models accept no ``input_type``.
"""

from __future__ import annotations

import base64
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EmbeddingsError(Exception):
    """Base class for every embeddings-layer error."""


class EmbeddingsValidationError(EmbeddingsError):
    """The client request is invalid or unsupported (maps to HTTP 400).

    ``code`` / ``param`` mirror the OpenAI error shape so the server can emit a
    response the client SDK already knows how to decode.
    """

    status_code = 400

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param


class UnsupportedEmbeddingsModelError(EmbeddingsError):
    """No embeddings adapter is registered for the requested model (HTTP 404)."""

    status_code = 404

    def __init__(self, model: str) -> None:
        super().__init__(f"No embeddings adapter registered for model {model!r}")
        self.model = model


# ---------------------------------------------------------------------------
# Intermediate representation + capabilities
# ---------------------------------------------------------------------------

class EmbeddingTask(str, Enum):
    """Provider-neutral semantic embedding task carried in the IR."""

    RETRIEVAL_DOCUMENT = "retrieval_document"
    RETRIEVAL_QUERY = "retrieval_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class TaskPolicy(str, Enum):
    """How an adapter treats a client-supplied ``input_type`` (the OpenClaw
    wire extension that decodes into an :class:`EmbeddingTask`).

      * ``FIXED`` — the task is baked into the profile (Cohere document/query
        aliases); any client ``input_type`` is rejected so the alias stays a
        standard OpenAI model.
      * ``ACCEPTED`` — the client *must* supply ``input_type``; it is validated
        against :attr:`Capabilities.accepted_tasks`.
      * ``SYMMETRIC`` — the model has no task concept and rejects any
        ``input_type``.
    """

    FIXED = "fixed"
    ACCEPTED = "accepted"
    SYMMETRIC = "symmetric"


@dataclass(frozen=True)
class EmbeddingRequest:
    """Normalised OpenAI embeddings request (the IR handed to an adapter)."""

    model: str
    inputs: tuple[str, ...]
    encoding_format: str = "float"  # "float" | "base64"
    dimensions: int | None = None
    user: str | None = None
    task: EmbeddingTask | None = None  # decoded OpenClaw input_type, if any


@dataclass(frozen=True)
class Capabilities:
    """Static constraints of one adapter, used for generic request validation.

    ``allowed_dimensions`` is empty when any positive dimension is accepted
    (provider-specific rules, e.g. Cohere's multiple-of-32, are enforced by the
    adapter's :meth:`EmbeddingsAdapter._validate` hook).

    ``task_policy`` / ``accepted_tasks`` / ``fixed_task`` describe how the
    adapter treats the IR's provider-neutral :class:`EmbeddingTask` (see
    :class:`TaskPolicy`).
    """

    supports_multiple_inputs: bool = True
    max_inputs: int | None = None
    supports_dimensions: bool = False
    allowed_dimensions: tuple[int, ...] = ()
    default_dimensions: int | None = None
    task_policy: TaskPolicy = TaskPolicy.FIXED
    accepted_tasks: frozenset[EmbeddingTask] = frozenset()
    fixed_task: EmbeddingTask | None = None


@dataclass(frozen=True)
class EmbeddingData:
    """One normalised embedding vector (always floats at this layer)."""

    index: int
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingResponse:
    """Normalised embeddings response — float vectors plus token usage."""

    model: str
    data: list[EmbeddingData]
    prompt_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Strict OpenAI request parser
# ---------------------------------------------------------------------------

_WIRE_INPUT_TYPES: dict[str, EmbeddingTask] = {
    "document": EmbeddingTask.RETRIEVAL_DOCUMENT,
    "query": EmbeddingTask.RETRIEVAL_QUERY,
    "classification": EmbeddingTask.CLASSIFICATION,
    "clustering": EmbeddingTask.CLUSTERING,
}


class EmbeddingRequestExtensionDecoder(ABC):
    """Decode one client compatibility extension into a neutral task."""

    field: str

    @abstractmethod
    def decode(self, value: object) -> EmbeddingTask: ...


class OpenClawInputTypeDecoder(EmbeddingRequestExtensionDecoder):
    field = "input_type"

    def decode(self, value: object) -> EmbeddingTask:
        return decode_input_type(value)


_EXTENSION_DECODERS: dict[str, EmbeddingRequestExtensionDecoder] = {
    OpenClawInputTypeDecoder.field: OpenClawInputTypeDecoder(),
}


def decode_input_type(value: object) -> EmbeddingTask:
    """Decode the OpenClaw ``input_type`` wire extension into a neutral task.

    This is the *only* OpenClaw-specific code in the parser: it accepts the
    provider-neutral task words the OpenClaw client sends and returns an
    :class:`EmbeddingTask`. Whether a given adapter accepts that task is decided
    later by its :class:`Capabilities` task policy — the decoder never knows a
    provider.
    """
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingsValidationError(
            f"'input_type' must be one of {sorted(_WIRE_INPUT_TYPES)}",
            param="input_type",
        )
    task = _WIRE_INPUT_TYPES.get(value.strip().lower())
    if task is None:
        raise EmbeddingsValidationError(
            f"'input_type' must be one of {sorted(_WIRE_INPUT_TYPES)} "
            f"(got {value!r})",
            param="input_type",
        )
    return task


def parse_request(body: dict) -> EmbeddingRequest:
    """Validate an OpenAI embeddings request and normalise it into an IR.

    Strict about the OpenAI schema: ``model`` (non-empty str), ``input`` (str or
    str[]), ``encoding_format`` (``float`` | ``base64``), ``dimensions``
    (positive int). Token-array input (``int[]`` / ``int[][]``) is rejected —
    every supported upstream embeds text, not token ids.

    ``input_type`` is a non-standard OpenClaw extension: it is decoded into the
    IR's neutral :class:`EmbeddingTask`, then accepted or rejected by the
    resolved adapter's task policy.
    """
    if not isinstance(body, dict):
        raise EmbeddingsValidationError("request body must be a JSON object")
    standard = {"model", "input", "encoding_format", "dimensions", "user"}
    allowed = standard | set(_EXTENSION_DECODERS)
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise EmbeddingsValidationError(
            f"unknown request field: {unknown[0]!r}",
            code="unknown_parameter",
            param=unknown[0],
        )

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise EmbeddingsValidationError(
            "'model' is required and must be a non-empty string", param="model"
        )

    inputs = tuple(_normalize_input(body.get("input")))

    encoding_format = body.get("encoding_format", "float")
    if encoding_format not in ("float", "base64"):
        raise EmbeddingsValidationError(
            "'encoding_format' must be 'float' or 'base64'",
            param="encoding_format",
        )

    dimensions = body.get("dimensions")
    if dimensions is not None and (
        not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0
    ):
        raise EmbeddingsValidationError(
            "'dimensions' must be a positive integer", param="dimensions"
        )

    user = body.get("user")
    if user is not None and not isinstance(user, str):
        raise EmbeddingsValidationError("'user' must be a string", param="user")

    present_extensions = [field for field in _EXTENSION_DECODERS if field in body]
    if len(present_extensions) > 1:
        raise EmbeddingsValidationError(
            "only one embedding task extension may be supplied",
            code="conflicting_parameters",
        )
    task = None
    if present_extensions:
        field = present_extensions[0]
        task = _EXTENSION_DECODERS[field].decode(body[field])

    return EmbeddingRequest(
        model=model.strip(),
        inputs=inputs,
        encoding_format=encoding_format,
        dimensions=dimensions,
        user=user,
        task=task,
    )


def _normalize_input(raw: object) -> list[str]:
    """Coerce the OpenAI ``input`` field into a list of non-empty strings."""
    if isinstance(raw, str):
        texts: list[object] = [raw]
    elif isinstance(raw, list):
        texts = raw
    else:
        raise EmbeddingsValidationError(
            "'input' must be a string or an array of strings", param="input"
        )

    if not texts:
        raise EmbeddingsValidationError("'input' must not be empty", param="input")

    if all(isinstance(item, str) for item in texts):
        if any(not item.strip() for item in texts):
            raise EmbeddingsValidationError(
                "'input' must not contain empty strings", param="input"
            )
        return list(texts)  # type: ignore[return-value]

    # int[] / int[][] token-array forms — not embeddable by text providers.
    if all(isinstance(item, int) for item in texts) or all(
        isinstance(item, list) for item in texts
    ):
        raise EmbeddingsValidationError(
            "token-array input is not supported; provide text strings", param="input"
        )
    raise EmbeddingsValidationError(
        "'input' must be a string or an array of strings", param="input"
    )


# ---------------------------------------------------------------------------
# Adapter ABC
# ---------------------------------------------------------------------------

class EmbeddingsAdapter(ABC):
    """Turns an IR into a native request body and a native response into an IR.

    Subclasses set :attr:`name` and :attr:`capabilities`, implement
    :meth:`build_request` / :meth:`render_response`, and optionally narrow model
    matching via :meth:`matches` and add provider-specific checks via
    :meth:`_validate`.
    """

    name: str = "base"
    capabilities: Capabilities = Capabilities()

    # -- model matching (used by the registry) --------------------------

    def matches(self, model: str) -> bool:
        """Return True if this adapter claims *model*. Provider-specific."""
        return False

    # -- validation -----------------------------------------------------

    def validate(self, ir: EmbeddingRequest) -> None:
        """Apply generic capability checks, then the adapter's own rules.

        Raises :class:`EmbeddingsValidationError` when the IR cannot be served.
        """
        cap = self.capabilities
        count = len(ir.inputs)

        if not cap.supports_multiple_inputs and count > 1:
            raise EmbeddingsValidationError(
                f"provider {self.name!r} accepts a single input per request",
                param="input",
            )
        if cap.max_inputs is not None and count > cap.max_inputs:
            raise EmbeddingsValidationError(
                f"provider {self.name!r} accepts at most {cap.max_inputs} inputs "
                f"per request (got {count})",
                param="input",
            )
        if ir.dimensions is not None:
            if not cap.supports_dimensions:
                raise EmbeddingsValidationError(
                    f"provider {self.name!r} does not support 'dimensions'",
                    param="dimensions",
                )
            if cap.allowed_dimensions and ir.dimensions not in cap.allowed_dimensions:
                raise EmbeddingsValidationError(
                    f"provider {self.name!r} supports dimensions "
                    f"{sorted(cap.allowed_dimensions)} (got {ir.dimensions})",
                    param="dimensions",
                )
        self._validate_task(ir)
        self._validate(ir)

    def _validate_task(self, ir: EmbeddingRequest) -> None:
        """Enforce the adapter's task policy against the IR's neutral task.

        The policy (fixed / accepted / symmetric) is the single place a
        client-supplied ``input_type`` is accepted or rejected; adapters never
        branch on the provider wire term here.
        """
        cap = self.capabilities
        if cap.task_policy is TaskPolicy.FIXED:
            if ir.task is not None:
                raise EmbeddingsValidationError(
                    f"provider {self.name!r} uses a fixed task and does not "
                    "accept 'input_type'; use the dynamic profile to select a "
                    "task",
                    param="input_type",
                )
        elif cap.task_policy is TaskPolicy.SYMMETRIC:
            if ir.task is not None:
                raise EmbeddingsValidationError(
                    f"provider {self.name!r} is symmetric and does not accept "
                    "'input_type'",
                    param="input_type",
                )
        elif cap.task_policy is TaskPolicy.ACCEPTED:
            if ir.task is None:
                raise EmbeddingsValidationError(
                    f"provider {self.name!r} requires 'input_type'",
                    param="input_type",
                )
            if cap.accepted_tasks and ir.task not in cap.accepted_tasks:
                allowed = sorted(task.value for task in cap.accepted_tasks)
                raise EmbeddingsValidationError(
                    f"provider {self.name!r} accepts 'input_type' in {allowed} "
                    f"(got {ir.task.value!r})",
                    param="input_type",
                )

    def _validate(self, ir: EmbeddingRequest) -> None:
        """Hook for provider-specific dimension / input rules. Default: none."""

    # -- native body building + response rendering ----------------------

    @abstractmethod
    def build_request(self, ir: EmbeddingRequest) -> dict:
        """Build the native upstream request body from the IR."""

    @abstractmethod
    def render_response(self, native: dict, ir: EmbeddingRequest) -> EmbeddingResponse:
        """Render a native upstream response into a normalised response IR."""

    # -- fanout (multiple native requests / responses) -----------------
    #
    # The generic fanout handler in ``server.py`` sends N native bodies and
    # collects N native responses without knowing which provider it is talking
    # to. Batch providers (Cohere) produce one body; per-input providers
    # (Titan) produce one body per input — both expose the same list-shaped
    # interface here.

    def build_requests(self, ir: EmbeddingRequest) -> list[dict]:
        """Return the native request bodies to fan out (default: one body)."""
        return [self.build_request(ir)]

    def render_responses(
        self, natives: list[dict], ir: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Render native responses into a response IR (default: a single body)."""
        if len(natives) != 1:
            raise EmbeddingsError(
                f"provider {self.name!r} expected one native response, got "
                f"{len(natives)}"
            )
        return self.render_response(natives[0], ir)


# ---------------------------------------------------------------------------
# Cohere embed v4 — document / query / classification / clustering
# ---------------------------------------------------------------------------

_COHERE_EMBED_V4_MARKERS = ("embed-v4", "embed-v-4")
_COHERE_QUERY_MARKERS = ("query",)

# Provider-neutral task → Cohere's native ``input_type`` wire term. This is the
# *only* place the Cohere vocabulary appears; adapters look up the IR task here
# instead of branching on the wire string.
_COHERE_INPUT_TYPES: dict[EmbeddingTask, str] = {
    EmbeddingTask.RETRIEVAL_DOCUMENT: "search_document",
    EmbeddingTask.RETRIEVAL_QUERY: "search_query",
    EmbeddingTask.CLASSIFICATION: "classification",
    EmbeddingTask.CLUSTERING: "clustering",
}


def _is_cohere_embed_v4(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in _COHERE_EMBED_V4_MARKERS)


def _is_query_model(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in _COHERE_QUERY_MARKERS)


def _cohere_float_vectors(native: dict) -> list[list[float]]:
    """Pull float vectors out of a Cohere v4 response.

    Cohere v4 returns ``embeddings`` as a flat list of float vectors when no
    ``embedding_types`` was requested, or a dict keyed by type (``{"float": …}``)
    when one was. Handle both.
    """
    embeddings = native.get("embeddings")
    if isinstance(embeddings, dict):
        vectors = embeddings.get("float")
        if vectors is None:
            raise EmbeddingsError(
                "Cohere response 'embeddings' dict is missing the 'float' key"
            )
    elif isinstance(embeddings, list):
        vectors = embeddings
    else:
        raise EmbeddingsError("Cohere response is missing 'embeddings'")
    if not isinstance(vectors, list):
        raise EmbeddingsError("Cohere 'embeddings' is not a list of vectors")
    return [[float(x) for x in vector] for vector in vectors]


class CohereEmbedV4Adapter(EmbeddingsAdapter):
    """Shared Cohere embed v4 behaviour; subclasses fix the task policy.

    The shared request builder derives Cohere's native ``input_type`` from the
    IR's :class:`EmbeddingTask` (a fixed task for the aliases, the client task
    for the dynamic profile) — never from a provider wire term baked into the
    adapter.
    """

    capabilities = Capabilities(
        supports_multiple_inputs=True,
        max_inputs=96,  # Cohere embed v4 caps a batch at 96 texts.
        supports_dimensions=True,
    )

    _ALLOWED_OUTPUT_DIMENSIONS = (256, 512, 1024, 1536)

    def _resolve_task(self, ir: EmbeddingRequest) -> EmbeddingTask:
        """Return the neutral task to serve: client task, else the fixed one."""
        task = ir.task if ir.task is not None else self.capabilities.fixed_task
        if task is None:
            raise EmbeddingsError(
                f"provider {self.name!r} cannot build a Cohere request without "
                "a task"
            )
        return task

    def _cohere_input_type(self, ir: EmbeddingRequest) -> str:
        return _COHERE_INPUT_TYPES[self._resolve_task(ir)]

    def _validate(self, ir: EmbeddingRequest) -> None:
        dim = ir.dimensions
        if dim is not None and dim not in self._ALLOWED_OUTPUT_DIMENSIONS:
            raise EmbeddingsValidationError(
                "'dimensions' must be one of "
                f"{list(self._ALLOWED_OUTPUT_DIMENSIONS)} for Cohere Embed v4 "
                f"(got {dim})",
                param="dimensions",
            )

    def build_request(self, ir: EmbeddingRequest) -> dict:
        body: dict = {
            "texts": list(ir.inputs),
            "input_type": self._cohere_input_type(ir),
            "embedding_types": ["float"],
        }
        if ir.dimensions is not None:
            body["output_dimension"] = ir.dimensions
        return body

    def render_response(self, native: dict, ir: EmbeddingRequest) -> EmbeddingResponse:
        vectors = _cohere_float_vectors(native)
        data = [
            EmbeddingData(index=i, embedding=vector)
            for i, vector in enumerate(vectors)
        ]
        # Cohere v4 does not expose an OpenAI-usable token count — report 0.
        return EmbeddingResponse(
            model=ir.model, data=data, prompt_tokens=0, total_tokens=0
        )


class CohereEmbedV4DocumentAdapter(CohereEmbedV4Adapter):
    """Cohere embed v4 fixed to ``search_document`` (indexing).

    A standard OpenAI alias: the task is selected by model name, so a client
    ``input_type`` would conflict and is rejected (FIXED policy).
    """

    name = "cohere-embed-v4-document"
    capabilities = Capabilities(
        supports_multiple_inputs=True,
        max_inputs=96,
        supports_dimensions=True,
        default_dimensions=1024,
        task_policy=TaskPolicy.FIXED,
        fixed_task=EmbeddingTask.RETRIEVAL_DOCUMENT,
    )

    def matches(self, model: str) -> bool:
        return _is_cohere_embed_v4(model) and not _is_query_model(model)


class CohereEmbedV4QueryAdapter(CohereEmbedV4Adapter):
    """Cohere embed v4 fixed to ``search_query`` (retrieval)."""

    name = "cohere-embed-v4-query"
    capabilities = Capabilities(
        supports_multiple_inputs=True,
        max_inputs=96,
        supports_dimensions=True,
        default_dimensions=1024,
        task_policy=TaskPolicy.FIXED,
        fixed_task=EmbeddingTask.RETRIEVAL_QUERY,
    )

    def matches(self, model: str) -> bool:
        return _is_cohere_embed_v4(model) and _is_query_model(model)


class CohereEmbedV4DynamicAdapter(CohereEmbedV4Adapter):
    """Cohere embed v4 with a client-selected ``input_type``.

    The dynamic ``cohere-embed-v4`` profile *requires* the OpenClaw
    ``input_type`` extension and maps query/document/classification/clustering
    to Cohere's native terms. It is selected explicitly by profile name — it
    never claims model names via :meth:`matches`, because a bare name cannot
    express which task is wanted.
    """

    name = "cohere-embed-v4"
    capabilities = Capabilities(
        supports_multiple_inputs=True,
        max_inputs=96,
        supports_dimensions=True,
        default_dimensions=1024,
        task_policy=TaskPolicy.ACCEPTED,
        accepted_tasks=frozenset(
            {
                EmbeddingTask.RETRIEVAL_DOCUMENT,
                EmbeddingTask.RETRIEVAL_QUERY,
                EmbeddingTask.CLASSIFICATION,
                EmbeddingTask.CLUSTERING,
            }
        ),
    )


# ---------------------------------------------------------------------------
# Amazon Titan Embeddings V2
# ---------------------------------------------------------------------------

_TITAN_EMBED_V2_DIMENSIONS = (256, 512, 1024)


class TitanEmbedV2Adapter(EmbeddingsAdapter):
    """Amazon Titan Embeddings V2 (``amazon.titan-embed-text-v2:0``).

    Takes a single ``inputText`` (the native API has no batch form), an optional
    ``dimensions`` in {256, 512, 1024}, and returns one ``embedding`` float
    vector plus ``inputTextTokenCount``. ``normalize`` defaults to True (the
    native default) and can be flipped for testing.

    Titan is a *symmetric* model: it has no task concept, so any client
    ``input_type`` is rejected (SYMMETRIC policy).
    """

    name = "amazon-titan-embed-v2"

    capabilities = Capabilities(
        supports_multiple_inputs=True,
        max_inputs=None,
        supports_dimensions=True,
        allowed_dimensions=_TITAN_EMBED_V2_DIMENSIONS,
        default_dimensions=1024,
        task_policy=TaskPolicy.SYMMETRIC,
    )

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def matches(self, model: str) -> bool:
        return "titan-embed" in model.lower()

    def build_request(self, ir: EmbeddingRequest) -> dict:
        if len(ir.inputs) != 1:
            raise EmbeddingsValidationError(
                "Titan Embed V2 native requests require one input; use build_requests",
                param="input",
            )
        body: dict = {
            "inputText": ir.inputs[0],
            "normalize": self.normalize,
            "embeddingTypes": ["float"],
        }
        if ir.dimensions is not None:
            body["dimensions"] = ir.dimensions
        return body

    def build_requests(self, ir: EmbeddingRequest) -> list[dict]:
        return [
            self.build_request(
                EmbeddingRequest(
                    model=ir.model,
                    inputs=(text,),
                    encoding_format=ir.encoding_format,
                    dimensions=ir.dimensions,
                )
            )
            for text in ir.inputs
        ]

    def render_response(self, native: dict, ir: EmbeddingRequest) -> EmbeddingResponse:
        return self.render_responses([native], ir)

    def render_responses(
        self, natives: list[dict], ir: EmbeddingRequest
    ) -> EmbeddingResponse:
        if len(natives) != len(ir.inputs):
            raise EmbeddingsError("Titan response count does not match input count")
        data: list[EmbeddingData] = []
        total_tokens = 0
        for index, native in enumerate(natives):
            embedding = native.get("embedding")
            if not isinstance(embedding, list):
                raise EmbeddingsError("Titan response is missing 'embedding'")
            data.append(
                EmbeddingData(index=index, embedding=[float(x) for x in embedding])
            )
            token_count = native.get("inputTextTokenCount", 0)
            if isinstance(token_count, int):
                total_tokens += token_count
        return EmbeddingResponse(
            model=ir.model,
            data=data,
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class EmbeddingsAdapterRegistry:
    """Maps a client model name to an embeddings adapter.

    Order-insensitive: adapters claim models via :meth:`EmbeddingsAdapter.matches`
    and each provider's matchers are mutually exclusive. Iteration order only
    breaks ties, which never arise with the built-in adapters.
    """

    def __init__(self) -> None:
        self._adapters: list[EmbeddingsAdapter] = []

    def register(self, adapter: EmbeddingsAdapter) -> None:
        """Add an adapter. Later-registered adapters are checked first."""
        self._adapters.insert(0, adapter)

    def resolve(self, model: str) -> EmbeddingsAdapter:
        """Return the adapter that claims *model* (legacy convenience lookup)."""
        for adapter in self._adapters:
            if adapter.matches(model):
                return adapter
        raise UnsupportedEmbeddingsModelError(model)

    def resolve_profile(self, profile: str) -> EmbeddingsAdapter:
        """Resolve an explicit ModelEntry profile without provider guessing."""
        for adapter in self._adapters:
            if adapter.name == profile:
                return adapter
        raise UnsupportedEmbeddingsModelError(profile)

    def __len__(self) -> int:
        return len(self._adapters)


# The process-wide registry, pre-loaded with the built-in adapters. The query
# adapter is registered before the document adapter so ``…-query`` models resolve
# to it (document's matcher already excludes them, but ordering makes it obvious).
# The dynamic adapter never claims model names — it is reached via
# :meth:`EmbeddingsAdapterRegistry.resolve_profile` by its name.
DEFAULT_REGISTRY = EmbeddingsAdapterRegistry()
DEFAULT_REGISTRY.register(CohereEmbedV4QueryAdapter())
DEFAULT_REGISTRY.register(CohereEmbedV4DocumentAdapter())
DEFAULT_REGISTRY.register(CohereEmbedV4DynamicAdapter())
DEFAULT_REGISTRY.register(TitanEmbedV2Adapter())


def resolve_adapter(model: str) -> EmbeddingsAdapter:
    """Resolve *model* against the default registry (legacy convenience API)."""
    return DEFAULT_REGISTRY.resolve(model)


def resolve_profile(profile: str) -> EmbeddingsAdapter:
    """Resolve a ModelEntry embedding profile against the default registry."""
    return DEFAULT_REGISTRY.resolve_profile(profile)


def apply_default_dimensions(
    request: EmbeddingRequest, adapter: EmbeddingsAdapter
) -> EmbeddingRequest:
    """Apply the stable public default dimension declared by one profile."""
    if request.dimensions is not None:
        return request
    dimension = adapter.capabilities.default_dimensions
    return request if dimension is None else replace(request, dimensions=dimension)


# ---------------------------------------------------------------------------
# float / base64 rendering
# ---------------------------------------------------------------------------

def encode_base64(vector: Sequence[float]) -> str:
    """Encode a float vector as an OpenAI-style base64 string (float32 LE)."""
    packed = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(packed).decode("ascii")


def render_openai(response: EmbeddingResponse, encoding_format: str = "float") -> dict:
    """Render a normalised response IR into the OpenAI ``/v1/embeddings`` shape.

    ``encoding_format="base64"`` re-encodes each float vector as base64-encoded
    float32 — the transform OpenAI performs server-side — so any float adapter
    (Cohere or Titan) can satisfy a base64 client request without provider help.
    """
    if encoding_format not in ("float", "base64"):
        raise EmbeddingsValidationError(
            "'encoding_format' must be 'float' or 'base64'",
            param="encoding_format",
        )
    data = [
        {
            "object": "embedding",
            "index": item.index,
            "embedding": (
                item.embedding if encoding_format == "float" else encode_base64(item.embedding)
            ),
        }
        for item in response.data
    ]
    return {
        "object": "list",
        "data": data,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "total_tokens": response.total_tokens,
        },
    }


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def prepare_request(body: dict) -> tuple[EmbeddingsAdapter, EmbeddingRequest, dict]:
    """Legacy convenience flow using model-name adapter discovery.

    Production endpoint routing uses ``ModelEntry.embedding_profile`` via
    :func:`resolve_profile`; this helper remains useful for standalone callers.
    """
    ir = parse_request(body)
    adapter = resolve_adapter(ir.model)
    ir = apply_default_dimensions(ir, adapter)
    adapter.validate(ir)
    return adapter, ir, adapter.build_request(ir)


__all__ = [
    "EmbeddingsError",
    "EmbeddingsValidationError",
    "UnsupportedEmbeddingsModelError",
    "EmbeddingTask",
    "TaskPolicy",
    "EmbeddingRequestExtensionDecoder",
    "OpenClawInputTypeDecoder",
    "EmbeddingRequest",
    "Capabilities",
    "EmbeddingData",
    "EmbeddingResponse",
    "EmbeddingsAdapter",
    "CohereEmbedV4Adapter",
    "CohereEmbedV4DocumentAdapter",
    "CohereEmbedV4QueryAdapter",
    "CohereEmbedV4DynamicAdapter",
    "TitanEmbedV2Adapter",
    "EmbeddingsAdapterRegistry",
    "DEFAULT_REGISTRY",
    "decode_input_type",
    "parse_request",
    "resolve_adapter",
    "resolve_profile",
    "apply_default_dimensions",
    "encode_base64",
    "render_openai",
    "prepare_request",
]
