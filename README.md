# Bedrock Gateway

**Lightweight OpenAI-compatible proxy that lets any OpenAI client access AWS Bedrock models.**

No SDK changes. No vendor lock-in. Just point your `OPENAI_BASE_URL` and go.

```
┌────────────────┐     OpenAI API     ┌──────────────────┐    Bedrock API    ┌─────────────┐
│  OpenAI Client │ ──────────────────▶ │ Bedrock Gateway  │ ────────────────▶ │ AWS Bedrock │
│  (any library) │ ◀────────────────── │  (this project)  │ ◀──────────────── │   Claude    │
└────────────────┘                     └──────────────────┘                    └─────────────┘
```

## Features

- 🔌 **Drop-in replacement** — OpenAI-compatible `/v1/chat/completions` and `/v1/models`
- 🔐 **Multiple auth modes** — Bearer Token, AK/SK (SigV4), IAM Role, AWS Profile
- 🔄 **Full protocol translation** — messages, tools, images, streaming, thinking
- 🏗️ **Production ready** — retry with backoff, structured logging, health checks
- 📦 **Zero config** — works with environment variables alone, or use YAML for full control
- 🐳 **Docker first** — single container, 50MB image

## Quick Start

### Option 1: pip install

```bash
pip install bedrock-gateway
```

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-token-here"
bedrock-gateway
# → listening on http://127.0.0.1:4000
```

### Option 2: Docker

```bash
docker run -p 4000:4000 \
  -e AWS_BEARER_TOKEN_BEDROCK="your-token" \
  bedrock-gateway
```

### Option 3: From source

```bash
git clone https://github.com/bedrock-gateway/bedrock-gateway.git
cd bedrock-gateway
pip install -e .
python -m bedrock_gateway
```

### Use it

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4000/v1",
    api_key="anything",  # not used, but required by the SDK
)

response = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Authentication Modes

| Mode | Config | Description |
|------|--------|-------------|
| `bearer_token` | `AWS_BEARER_TOKEN_BEDROCK` env var | AWS Bearer Token (ABSK). Simplest setup. |
| `credentials` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | Standard AK/SK with SigV4 signing. |
| `iam_role` | (automatic) | Auto-detect from EC2/ECS/Lambda metadata. Requires `boto3`. |
| `profile` | `AWS_PROFILE` or config | Named AWS CLI profile. Requires `boto3`. |

For `iam_role` or `profile` mode, install with boto3:

```bash
pip install bedrock-gateway[boto3]
```

## Supported Models

| Alias | Bedrock Model ID | Context | Max Output |
|-------|-----------------|---------|------------|
| `claude-opus-4.7` | `us.anthropic.claude-opus-4-7` | 1M | 128K |
| `claude-opus-4` | `us.anthropic.claude-opus-4-6-v1` | 1M | 128K |
| `claude-sonnet-4.6` | `us.anthropic.claude-sonnet-4-6` | 1M | 64K |
| `claude-sonnet-4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | 200K | 64K |
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | 64K |
| `claude-sonnet-3.5` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | 64K |

You can also pass a raw Bedrock model ID directly (e.g., `us.anthropic.claude-3-haiku-20240307-v1:0`).

Add custom models in `config.yaml`:

```yaml
models:
  my-custom-model:
    bedrock_id: us.my-org.my-model-v1
    context_length: 100000
    max_output: 8192
```

## Configuration

### Environment Variables (zero-config)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_BEARER_TOKEN_BEDROCK` | — | Bearer token for authentication |
| `AWS_REGION` | `us-east-1` | AWS region |
| `BEDROCK_HOST` | `127.0.0.1` | Server bind address |
| `BEDROCK_PORT` | `4000` | Server port |
| `BEDROCK_LOG_LEVEL` | `info` | Log level (debug/info/warning/error) |
| `BEDROCK_AUTH_MODE` | `bearer_token` | Auth mode override |
| `BEDROCK_MAX_RETRIES` | `3` | Max retry attempts |

### YAML Configuration

Copy `config.example.yaml` to `config.yaml` for full control:

```yaml
auth:
  mode: bearer_token
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}  # env var interpolation

region: us-east-1

server:
  host: 0.0.0.0
  port: 4000
  log_level: info

retry:
  max_retries: 3
  base_delay: 1.0

models:
  claude-sonnet-4:
    bedrock_id: us.anthropic.claude-sonnet-4-20250514-v1:0
    context_length: 200000
    max_output: 64000
```

YAML values support `${ENV_VAR}` syntax for environment variable interpolation.

## API Reference

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint. Supports:

- ✅ Synchronous and streaming responses
- ✅ System messages
- ✅ Multi-turn conversations
- ✅ Tool calling (function calling)
- ✅ Multimodal (images via base64 or URL)
- ✅ Extended thinking (`thinking` parameter)
- ✅ Stop sequences
- ✅ Temperature, top_p

### GET /v1/models

Returns the list of available models in OpenAI format.

### GET /health

Health check endpoint. Returns:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "auth_mode": "bearer_token",
  "region": "us-east-1",
  "models": 6
}
```

## How It Works

1. Receives OpenAI-format requests
2. Converts messages, tools, and images to Anthropic/Bedrock format
3. Authenticates and forwards to AWS Bedrock
4. Converts the response (or stream) back to OpenAI format
5. Retries automatically on 429/503/529 with exponential backoff

### Protocol Translation Details

| OpenAI | Bedrock (Anthropic) |
|--------|-------------------|
| `messages[role=system]` | `system` top-level field |
| `messages[role=tool]` | `messages[role=user]` with `tool_result` block |
| `tool_calls` | `tool_use` content blocks |
| `tools[type=function]` | `tools` with `input_schema` |
| `tool_choice: "required"` | `tool_choice: {type: "any"}` |
| `image_url` (base64/URL) | `image` source block |
| `stream: true` | `/invoke-with-response-stream` + event-stream parsing |

## Bedrock Gateway vs. LiteLLM

| | Bedrock Gateway | LiteLLM |
|---|---|---|
| **Focus** | AWS Bedrock only | 100+ providers |
| **Dependencies** | 4 (fastapi, uvicorn, httpx, pyyaml) | 50+ |
| **Docker image** | ~50MB | ~500MB |
| **Auth modes** | Bearer Token, AK/SK, IAM, Profile | AK/SK, IAM |
| **Config** | YAML + env vars | YAML + env vars |
| **Setup time** | 30 seconds | Minutes |
| **Best for** | Teams using Bedrock exclusively | Multi-provider routing |

Choose Bedrock Gateway if you only need Bedrock and want minimal overhead.
Choose LiteLLM if you need to route across multiple LLM providers.

## Development

```bash
git clone https://github.com/bedrock-gateway/bedrock-gateway.git
cd bedrock-gateway
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check bedrock_gateway/ tests/

# Type check
mypy bedrock_gateway/ --ignore-missing-imports
```

## License

MIT — see [LICENSE](LICENSE).
