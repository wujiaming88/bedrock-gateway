# CLAUDE.md

Guidance for AI agents (and humans) working in this repository.

## What this is

An OpenAI/Anthropic-compatible **proxy gateway** for multiple LLM clouds. A
client points its SDK's `base_url` at this gateway and calls models on AWS
Bedrock or Azure OpenAI without changing code. FastAPI + httpx, no boto3
required for the common (bearer-token / api-key) auth paths.

## Architecture: Transport × Dialect (read this first)

The core abstraction is **two orthogonal axes** — internalise this before
touching provider code. See `docs/multi-cloud-multimodal-design.md` for the
full rationale.

- **Transport** (`bedrock_gateway/providers/transports.py`) — *where + how to
  authenticate*: builds the upstream URL and returns auth headers.
  `BedrockTransport` (runtime/mantle hosts, global SigV4/Bearer auth →
  `auth_headers` returns `None`), `AzureTransport` (per-resource endpoint +
  `api-key` header).
- **Dialect** (`anthropic_bedrock.py`, `openai_responses.py`, `openai_chat.py`)
  — *request/response/stream shape*: `operation_path`, `build_request`,
  `render_sync`, `transform_stream`, `stream_error`.
  `anthropic` converts OpenAI↔Anthropic; the `openai-*` dialects are verbatim
  passthrough.

A model = one `(transport, dialect)` pair, chosen from `ModelEntry`. **Adding a
cloud = one Transport; adding a wire format = one Dialect.** Never add an
`if azure:` branch to a dialect or an `if responses:` branch to a transport —
that reintroduces the N×M coupling this design removed.

Boundary rule for URLs: a dialect's `operation_path` returns the **bare
operation** (`/responses`, `/chat/completions`, `/images/generations`) — never a cloud-specific
prefix. The **transport** owns the API root: `BedrockTransport` adds
`/openai/v1` for the mantle endpoint; `AzureTransport` uses whatever the
resource `base_url` already ends in. If you find yourself checking
`entry.transport` inside a dialect, the prefix belongs in the transport instead.

The server (`server.py`) owns everything cross-cutting — retries, backoff,
timeouts, metrics, the pre-stream error preflight (`_open_upstream_stream`),
error-severity logging — and is transport-/dialect-agnostic. Handlers thread
`(transport, dialect, entry)`.

Generic official-compatible upstreams use `upstream_resources`: each resource maps
one prefix and secret env name to dialect-specific base/path/auth routes. The generic
`http` transport performs URL/auth only; it never branches on provider. DeepSeek is
configured this way and uses native Chat, Responses, and Anthropic passthrough.

### Endpoints → dialects
- `POST /v1/chat/completions` — branches by dialect: `anthropic` (Claude,
  converted) vs `openai-chat` (Azure/mantle, passthrough).
- `POST /openai/v1/responses` — `openai-responses` dialect (GPT-5.x, Grok, Azure). Bedrock GPT-5.x requests pass through a small compatibility normalizer (`responses_normalizer.py`) before upstream dispatch: Codex `additional_tools` items are lifted to top-level `tools`, developer messages become `instructions`, and text blocks are normalized to `input_text`. Keep this dialect-adjacent; do not put it in transport.
- `POST /openai/v1/images/generations` — `openai-images` dialect (Azure
  `gpt-image-2`, passthrough, sync JSON only; no streaming).
- `POST /openai/v1/images/edits` — the same `openai-images` dialect with an
  explicit `edits` operation (Azure `gpt-image-2`, replayable multipart,
  sync JSON or SSE). The endpoint is constrained by Azure transport + Images
  dialect, not a model-name allowlist; variations and Bedrock image models are absent.
- `POST /v1/messages` — branches by dialect: `anthropic` (Bedrock Claude),
  `anthropic-passthrough` (native HTTP Messages) vs `openai-responses` (GPT-5.5/Grok/`azure/<dep>`, **translated**
  Anthropic Messages ⇄ Responses via `messages_to_responses.py`). This is what
  lets Claude Code (`ANTHROPIC_BASE_URL`) use non-Anthropic models. The inbound
  translation reuses the same generic `_handle_sync` / `_open_upstream_stream`
  primitives — it is transport-agnostic, so Bedrock mantle and Azure share it.
- Wrong-endpoint-for-model returns a 400 with guidance (dialect guards); an
  `openai-chat` model on `/v1/messages` is still a 400 (only responses is
  translated there).

### Inbound vs outbound translation (the two directions)
- `converter.py` — OpenAI Chat → Anthropic (for Claude on `/v1/chat/completions`).
- `messages_to_responses.py` — Anthropic Messages → OpenAI Responses (mirror; for
  any-model on `/v1/messages`). Kept as an isolated, 100%-covered module; if a
  second inbound format ever appears, this is the first instance to generalise
  into an InboundProtocol abstraction — do NOT abstract prematurely for one case.

## Adding a model

1. Bedrock model, same dialect as an existing one → add one entry to
   `_DEFAULT_MODELS` in `config.py` + aliases in `_MODEL_ALIASES`. Zero code.
2. New Bedrock vendor prefix → also add it to `_BEDROCK_ID_PREFIXES` in
   `models.py` so raw-id passthrough works.
3. Azure model → add an `azure_resources` entry (endpoint + key) and a model
   entry referencing it with a `deployment` name.
4. New cloud or new wire format → add a Transport or Dialect, register it in
   `providers/__init__.py`.

**Always verify a new model actually resolves before writing config**: some
require a Marketplace subscription / IAM permission the account may lack
(Sonnet 5 → 403, GPT-5.4 → 401 as of this writing). curl the upstream with the
bearer/api-key first.

## Commands

```bash
pip install -e ".[dev]"                     # dev setup (needs pytest-asyncio)
python -m pytest -q                          # full suite (671 tests)
python -m pytest tests/test_azure.py -q      # one file
ruff check bedrock_gateway/ tests/
python -m bedrock_gateway                    # run locally (reads ./config.yaml)

# Clean-install packaging smoke (creates a temporary venv; needs package-index access):
python scripts/smoke_install.py

# Real end-to-end smoke (needs live creds; NOT part of pytest):
AWS_BEARER_TOKEN_BEDROCK=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_KEY=... \
  python scripts/smoke_e2e.py --port 4199
```

## Gotchas (hard-won — don't relearn these)

- **Version lives in two files**: bump BOTH `bedrock_gateway/__init__.py` and
  `pyproject.toml`. `TestVersionConsistency` enforces it; a mismatch ships a
  wheel with the wrong dist label.
- **`models:` MERGES with the built-in defaults** (custom entries add on top;
  same-alias overrides). `use_default_models: false` starts from empty. Pinned
  by `test_custom_models_merge_with_defaults` / `..._overrides_default_on_clash`
  / `..._use_default_models_false_disables_defaults`.
- **Running e2e/the gateway writes `data/metrics.db`**, which then pollutes the
  dashboard tests. `rm -f data/metrics.db*` before `pytest`.
- **Azure auth is `api-key:` header, not `Authorization: Bearer`.** Azure
  endpoint URL forms differ per operation (responses/chat/images may share a
  `/openai/v1` root; legacy bases may carry `?api-version=...`) — keep
  `azure_endpoint` fully configurable, don't hardcode a form.
- **Image generation models don't go through Responses.** `gpt-image-2` returns
  "operation unsupported" on `/responses`; use `/openai/v1/images/generations`
  with the `openai-images` dialect.
- **Passthrough dialects must not strip unknown fields** — Azure's
  `content_filter_results`, Responses `input_image`, and Images `b64_json` blocks
  flow through verbatim. No field whitelisting.
- **Streaming passthrough uses an incremental UTF-8 decoder** — multi-byte
  (CJK) chars split across upstream chunks would otherwise corrupt.
- **Verifying the Claude Code → gateway path: `CLAUDE_CODE_USE_BEDROCK` wins
  over `ANTHROPIC_BASE_URL`.** If that env var (or `CLAUDE_CODE_USE_VERTEX`) is
  present and non-empty — *any* value, including `"0"` — Claude Code connects
  directly to the cloud and ignores the gateway entirely; `gpt-5.5` then 400s
  client-side with "provided model identifier is invalid". It must be *unset*,
  not `0`. Also: a `claude` subprocess spawned *inside* a Claude Code session
  does NOT honour a custom `ANTHROPIC_BASE_URL` (the outer harness shadows its
  networking — point it at a dead port and it still answers). So the reliable
  E2E check is `curl` straight at the gateway's `/v1/messages`, not spawning
  `claude` from within a session.

## Deployment (this box)

systemd `bedrock-gateway.service`, port 4000, `User=bedrock`, runtime venv at
`/opt/bedrock-gateway` (**regular pip install, not `-e`**).

```bash
# Install an explicit release and its dependencies; never skip dependency
# resolution for a normal upgrade or rollback.
TARGET_VERSION=v0.4.14
/opt/bedrock-gateway/bin/python -m pip install --upgrade --force-reinstall \
  "git+<repo-url>@${TARGET_VERSION}"

# Preflight before stopping the currently working process.
/opt/bedrock-gateway/bin/python -m pip check
cd /tmp && /opt/bedrock-gateway/bin/python -c \
  "import bedrock_gateway; import bedrock_gateway.server; print(bedrock_gateway.__version__, bedrock_gateway.__file__)"

systemctl restart bedrock-gateway
curl -fsS http://127.0.0.1:4000/health | /opt/bedrock-gateway/bin/python -c \
  'import json,sys; d=json.load(sys.stdin); assert d["status"]=="ok" and "v"+d["version"]==sys.argv[1], d; print(d)' "$TARGET_VERSION"
systemctl status bedrock-gateway --no-pager -l
# On failure: journalctl -u bedrock-gateway -n 100 --no-pager -l
```

Always verify from a **neutral directory** (`cd /tmp`), else Python may load the
source-tree copy. Dependency-bypassing installs are reserved for environments
whose dependencies are fully managed and validated by an external system.

## Conventions

- Match the surrounding code's comment density and docstring style — modules
  carry a purpose docstring; non-obvious logic gets a *why* comment.
- Every change keeps the suite green; new behaviour gets tests. Passthrough and
  streaming paths especially need boundary tests.
- Config changes: update `config.example.yaml` + README + CHANGELOG together.
