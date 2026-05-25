# Bedrock Gateway

Forward OpenAI / Anthropic API requests to AWS Bedrock.

[中文文档](README.md) · [Changelog](CHANGELOG.md)

## Quick start

```bash
pip install git+https://github.com/wujiaming88/bedrock-gateway.git

export AWS_BEARER_TOKEN_BEDROCK="your-aws-bearer-token"
bedrock-gateway
# listening on http://127.0.0.1:4000
```

Verify:

```bash
curl http://127.0.0.1:4000/health

curl http://127.0.0.1:4000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
```

Use from an SDK:

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="anything")
client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "hello"}],
)

# Anthropic SDK
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:4000", api_key="anything")
client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

## Configuration

All config lives in `config.yaml` next to the process (or at `--config /path/to/config.yaml`). String values support `${VAR}` interpolation from the process environment — missing variables expand to an empty string.

### `auth`

AWS credential source. Exactly one mode at a time.

**`bearer_token`** — Bedrock API key (simplest; no SigV4, no boto3).

```yaml
auth:
  mode: bearer_token
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}
```

**`credentials`** — static AK/SK, request is signed with SigV4 in-process.

```yaml
auth:
  mode: credentials
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  session_token: ${AWS_SESSION_TOKEN}   # optional, temporary credentials
```

**`iam_role`** — pick up credentials from the EC2 / ECS / Lambda metadata service. Requires `boto3`: `pip install "bedrock-gateway[boto3]"`.

```yaml
auth:
  mode: iam_role
```

**`profile`** — named profile from `~/.aws/credentials`. Also requires `boto3`.

```yaml
auth:
  mode: profile
  profile: default
```

### `server`

```yaml
server:
  host: 0.0.0.0          # default 127.0.0.1
  port: 4000             # default 4000
  log_level: info        # debug | info | warning | error
  api_key: ${BEDROCK_API_KEY}   # optional; if set, /v1/* requires it
```

When `api_key` is set, clients must send either `Authorization: Bearer <key>` or `x-api-key: <key>`. `/health` and `/` stay public. Key comparison uses `hmac.compare_digest`.

### `region`

```yaml
region: us-east-1
```

### `retry`

```yaml
retry:
  max_retries: 3     # total attempts before giving up
  base_delay: 1.0    # seconds; actual delay = base_delay * 2^attempt
```

Retries fire on HTTP `429`, `503`, `529`, and timeouts. Everything else returns to the client unchanged.

### `models`

Maps a user-facing alias to a Bedrock model ID. Omit the whole section to use the built-in defaults (see [Model aliases](#model-aliases)). You can also always pass a raw Bedrock model ID as the `model` field — the gateway treats anything starting with `us.`, `anthropic.`, etc. as a direct passthrough.

```yaml
models:
  my-model:
    bedrock_id: us.my-org.my-model-v1
    context_length: 100000
    max_output: 8192
```

### `dashboard`

```yaml
dashboard:
  enabled: true                       # mount /dashboard/ and /api/metrics/*
  api_key: ${BEDROCK_DASHBOARD_KEY}   # dashboard auth, independent of server.api_key
  require_auth: true                  # require dashboard.api_key when one is set
  localhost_only: false               # optional override; see below
  rate_limit: 60                      # /api/metrics/* requests per IP per minute
  max_request_log: 200                # rows kept in the recent-requests panel
```

`dashboard.api_key` is deliberately independent of `server.api_key` — model clients cannot reach the dashboard, and dashboard operators cannot call the model endpoints. `localhost_only` defaults to `true` when no `dashboard.api_key` is configured, and `false` when one is — set it explicitly to override. Set `enabled: false` to not mount the dashboard routes at all.

### Environment-variable shortcuts

Used only when the matching `config.yaml` field is absent.

| Variable | Default | Field |
|---|---|---|
| `BEDROCK_API_KEY` | — | `server.api_key` |
| `BEDROCK_DASHBOARD_KEY` | — | `dashboard.api_key` |
| `AWS_BEARER_TOKEN_BEDROCK` | — | `auth.bearer_token` |
| `AWS_REGION` | `us-east-1` | `region` |
| `BEDROCK_HOST` | `127.0.0.1` | `server.host` |
| `BEDROCK_PORT` | `4000` | `server.port` |
| `BEDROCK_LOG_LEVEL` | `info` | `server.log_level` |
| `BEDROCK_AUTH_MODE` | `bearer_token` | `auth.mode` |
| `BEDROCK_MAX_RETRIES` | `3` | `retry.max_retries` |

## Deployment

### Local

```bash
git clone https://github.com/wujiaming88/bedrock-gateway.git
cd bedrock-gateway
pip install -e .

export AWS_BEARER_TOKEN_BEDROCK="your-token"
python -m bedrock_gateway
```

### systemd

```bash
# deps + venv
apt install -y python3.12-venv git
python3 -m venv /opt/bedrock-gateway
/opt/bedrock-gateway/bin/pip install git+https://github.com/wujiaming88/bedrock-gateway.git
ln -s /opt/bedrock-gateway/bin/bedrock-gateway /usr/local/bin/bedrock-gateway

# secrets
cat > /opt/bedrock-gateway/.env << 'EOF'
AWS_BEARER_TOKEN_BEDROCK=your-aws-bearer-token
BEDROCK_API_KEY=bgw-your-generated-key
BEDROCK_DASHBOARD_KEY=bgw-dash-your-generated-key
EOF
chmod 600 /opt/bedrock-gateway/.env

# config
cat > /opt/bedrock-gateway/config.yaml << 'EOF'
auth:
  mode: bearer_token
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}

region: us-east-1

server:
  host: 0.0.0.0
  port: 4000
  log_level: info
  api_key: ${BEDROCK_API_KEY}

retry:
  max_retries: 3
  base_delay: 1.0

dashboard:
  enabled: true
  api_key: ${BEDROCK_DASHBOARD_KEY}
  require_auth: true
  rate_limit: 60
  max_request_log: 500
EOF
```

`/etc/systemd/system/bedrock-gateway.service`:

```ini
[Unit]
Description=Bedrock Gateway
After=network.target

[Service]
Type=simple
EnvironmentFile=/opt/bedrock-gateway/.env
WorkingDirectory=/opt/bedrock-gateway
ExecStart=/opt/bedrock-gateway/bin/bedrock-gateway
Restart=always
RestartSec=3
User=bedrock
Group=bedrock

[Install]
WantedBy=multi-user.target
```

```bash
useradd --system --no-create-home bedrock
systemctl daemon-reload
systemctl enable --now bedrock-gateway
journalctl -u bedrock-gateway -f
```

### Docker

```bash
docker build -t bedrock-gateway .

docker run -d --name bedrock-gateway \
  -p 4000:4000 \
  -e AWS_BEARER_TOKEN_BEDROCK="your-token" \
  -e BEDROCK_API_KEY="bgw-your-key" \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  bedrock-gateway
```

### docker-compose

The shipped `docker-compose.yml` mounts `./config.yaml` into the container and reads `AWS_BEARER_TOKEN_BEDROCK` / `AWS_REGION` from the environment.

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-token"
docker compose up -d
docker compose logs -f
```

### Nginx reverse proxy

```nginx
upstream bedrock_gateway {
    server 127.0.0.1:4000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name gateway.example.com;

    ssl_certificate     /etc/ssl/gateway.crt;
    ssl_certificate_key /etc/ssl/gateway.key;

    client_max_body_size 32m;

    location / {
        proxy_pass              http://bedrock_gateway;
        proxy_http_version      1.1;
        proxy_set_header        Host $host;
        proxy_set_header        X-Real-IP $remote_addr;
        proxy_set_header        X-Forwarded-For $proxy_add_x_forwarded_for;

        # streaming
        proxy_buffering         off;
        proxy_cache             off;
        proxy_read_timeout      600s;
        proxy_send_timeout      600s;
    }
}
```

## Dashboard

Live request metrics at `/dashboard/`. JSON endpoints at `/api/metrics/*`.

**Access.** The dashboard uses its own `dashboard.api_key`, separate from `server.api_key`: someone with the model-calling key cannot reach the dashboard, and a dashboard operator cannot call `/v1/*`. Four ways to authenticate when `dashboard.api_key` is set: login cookie (via the form at `/dashboard/login`), `Authorization: Bearer`, `x-api-key` header, or `?key=` query parameter.

**Auth rules.**

| Condition | Behavior |
|---|---|
| `dashboard.api_key` set and `dashboard.require_auth: true` | Dashboard key required by any method above |
| `dashboard.api_key` unset | Serves only to `127.0.0.1` / `::1` (unless `localhost_only: false`) |
| `dashboard.enabled: false` | Routes not mounted |

**What it shows.** Top gauges (QPS, success rate, p50/p95 latency, tokens/min); per-model request and token distribution; a 1H / 6H / 24H traffic and latency time series; recent-requests table filterable by status; errors grouped by status code and by error type; header bar with version, region, auth mode, uptime, RSS.

All dashboard responses carry `Content-Security-Policy`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: no-referrer`. `/api/metrics/*` is rate-limited per IP (default 60/min, `dashboard.rate_limit`).

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI chat completions, sync and streaming |
| `POST` | `/v1/messages` | Anthropic messages, sync and streaming |
| `GET`  | `/v1/models` | Model list (OpenAI format) |
| `GET`  | `/health` | Liveness, no auth |
| `GET`  | `/dashboard/` | UI |
| `GET`  | `/api/metrics/*` | Dashboard JSON |

### OpenAI parameters (`/v1/chat/completions`)

| Parameter | Notes |
|---|---|
| `messages` | `role=system` extracted to Bedrock `system` field |
| `model` | Alias or raw Bedrock model ID |
| `stream` | Boolean, SSE |
| `max_tokens` / `max_completion_tokens` | Falls back to model default |
| `temperature`, `top_p` | Passthrough |
| `stop` | String or list → `stop_sequences` |
| `tools`, `tool_choice` | Converted to Anthropic `tool_use` |
| `reasoning_effort` | Mapped to `thinking` budget |
| `thinking` | Passthrough |
| `image_url` | base64 data URLs and remote URLs |

### Anthropic parameters (`/v1/messages`)

Passthrough: `messages`, `system` (string or block array), `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `metadata`, `tools`, `tool_choice`, `thinking`, `stream`. Extended-thinking stream events (`thinking_delta`, `signature_delta`, `redacted_thinking`) are forwarded. Cache-token usage is surfaced when the underlying model reports it.

### Extended thinking

```json
{
  "model": "claude-sonnet-4",
  "max_tokens": 4096,
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "messages": [...]
}
```

When `thinking` is set, `temperature` is stripped (Bedrock rejects it) and `budget_tokens` is clamped to ≥ 1024. On `/v1/chat/completions`, `reasoning_effort: "low" | "medium" | "high"` maps to a `thinking` budget.

### Model aliases

| Alias | Bedrock ID | Context | Max output |
|---|---|---|---|
| `claude-opus-4.7` | `us.anthropic.claude-opus-4-7` | 1M | 128K |
| `claude-opus-4` | `us.anthropic.claude-opus-4-6-v1` | 1M | 128K |
| `claude-sonnet-4.6` | `us.anthropic.claude-sonnet-4-6` | 1M | 64K |
| `claude-sonnet-4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | 200K | 64K |
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | 64K |
| `claude-sonnet-3.5` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | 64K |

Common name variants (e.g. `claude-3-5-sonnet-latest`, `claude-sonnet-4-20250514`) resolve to the canonical alias, so stock Anthropic SDK defaults work as-is.

## Security

Production checklist:

- [ ] `BEDROCK_API_KEY` set to a strong random value (`bgw-$(openssl rand -base64 48)`)
- [ ] `BEDROCK_DASHBOARD_KEY` set to a strong random value — separate from `BEDROCK_API_KEY`
- [ ] Secrets in `.env` with mode `600`, not in `config.yaml`
- [ ] TLS in front of the gateway when binding to `0.0.0.0` (Nginx / ALB / Cloudflare)
- [ ] `dashboard.api_key` set (or `dashboard.enabled: false`), and `dashboard.require_auth: true`
- [ ] Process runs as a non-root user (systemd `User=`, Docker `appuser`)
- [ ] Logs centralised (`journalctl`, container log driver)
- [ ] Bedrock IAM principal limited to the minimum `bedrock:InvokeModel*` actions

## Development

```bash
git clone https://github.com/wujiaming88/bedrock-gateway.git
cd bedrock-gateway
pip install -e ".[dev]"

pytest -v
ruff check bedrock_gateway/ tests/
mypy bedrock_gateway/ --ignore-missing-imports
```

## License

MIT — see [LICENSE](LICENSE).
