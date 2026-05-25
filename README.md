# Bedrock Gateway

把发往 OpenAI / Anthropic 的 API 请求转发到 AWS Bedrock。
任何兼容 OpenAI 或 Anthropic SDK 的客户端，都可以**不改一行代码**接入 Bedrock。

[English](README.en.md) · [更新日志](CHANGELOG.md)

---

## 它解决了什么问题

- **现有应用想切到 Bedrock**：把 `base_url` 改到本网关即可，OpenAI / Anthropic 两套 SDK 都兼容。
- **不想引入 boto3 / SigV4**：Bearer Token 模式开箱即用，进程启动后直接转发。
- **凭据集中托管**：客户端只看到一个网关 API key，AWS 凭据不下发到调用方。
- **可观测性**：自带 dashboard，按模型、状态、时间维度展示请求与延迟。

---

## 快速开始

```bash
pip install git+https://github.com/wujiaming88/bedrock-gateway.git

export AWS_BEARER_TOKEN_BEDROCK="你的-aws-bearer-token"
bedrock-gateway
# listening on http://127.0.0.1:4000
```

验证服务：

```bash
curl http://127.0.0.1:4000/health

curl http://127.0.0.1:4000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
```

OpenAI SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="anything")
client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "hello"}],
)
```

Anthropic SDK：

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:4000", api_key="anything")
client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

> 当 `server.api_key` 为空时，`api_key` 字段可填任意值；一旦设置则必须传真实 key。

---

## 配置

所有配置项位于进程同目录下的 `config.yaml`，也可以通过 `--config /path/to/config.yaml` 显式指定。
字符串值支持 `${VAR}` 形式从环境变量插值；变量未定义时展开为空字符串。

### 1. 认证 `auth`

同一时间只能启用一种 AWS 凭据来源。

| 模式 | 适用场景 | 是否需要 boto3 |
|---|---|:---:|
| `bearer_token` | Bedrock API key，最简单，不走 SigV4 | 否 |
| `credentials` | 静态 AK/SK，进程内做 SigV4 签名 | 否 |
| `iam_role` | EC2 / ECS / Lambda 元数据服务自动取凭据 | **是** |
| `profile` | 使用 `~/.aws/credentials` 中的命名 profile | **是** |

需要 boto3 时安装：`pip install "bedrock-gateway[boto3]"`。

```yaml
# bearer_token（推荐入门）
auth:
  mode: bearer_token
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}

# credentials
auth:
  mode: credentials
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  session_token: ${AWS_SESSION_TOKEN}   # 可选，临时凭证用

# iam_role
auth:
  mode: iam_role

# profile
auth:
  mode: profile
  profile: default
```

### 2. 服务端 `server`

```yaml
server:
  host: 0.0.0.0          # 默认 127.0.0.1
  port: 4000             # 默认 4000
  log_level: info        # debug | info | warning | error
  api_key: ${BEDROCK_API_KEY}   # 可选：设置后 /v1/* 强制鉴权
```

设置 `api_key` 后，调用方需带 `Authorization: Bearer <key>` 或 `x-api-key: <key>`。
`/health` 与 `/` 始终公开，便于探活。比较使用 `hmac.compare_digest`，避免时序攻击。

### 3. 区域 `region`

```yaml
region: us-east-1
```

### 4. 重试 `retry`

```yaml
retry:
  max_retries: 3     # 总尝试次数
  base_delay: 1.0    # 秒；实际延迟 = base_delay * 2^attempt
```

仅对 `429` / `503` / `529` 与超时触发指数退避重试。其他状态码原样返回客户端，不做补救。

### 5. 模型别名 `models`

将易读的别名映射到 Bedrock 模型 ID。整段省略时使用[内置默认](#内置模型别名)。
请求里 `model` 字段也可以直接传原始 Bedrock ID（以 `us.`、`anthropic.` 等开头的会按 passthrough 处理）。

```yaml
models:
  my-model:
    bedrock_id: us.my-org.my-model-v1
    context_length: 100000
    max_output: 8192
```

### 6. 监控面板 `dashboard`

```yaml
dashboard:
  enabled: true                       # false 则完全不挂载 dashboard 路由
  api_key: ${BEDROCK_DASHBOARD_KEY}   # 独立鉴权，与 server.api_key 完全无关
  require_auth: true                  # 已设 api_key 时是否强制要求该 key
  localhost_only: false               # 显式覆盖；见下方
  rate_limit: 60                      # /api/metrics/* 每 IP 每分钟限流
  max_request_log: 200                # 请求日志面板保留的最近条数
```

`localhost_only` 缺省时的默认行为：

- 未配置 `dashboard.api_key` → 默认 `true`，仅本机可访问
- 已配置 `dashboard.api_key` → 默认 `false`，可远程访问但需鉴权

> 自 0.1.1 起：`dashboard.enabled: false` 时，相关后台任务（事件循环延迟采样、上游可达性探针）也不会启动，避免无消费者的空转与日志噪音。

### 环境变量快捷配置

仅在 `config.yaml` 对应字段缺省时生效。

| 变量 | 默认值 | 对应字段 |
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

---

## 部署

### 本地开发

```bash
git clone https://github.com/wujiaming88/bedrock-gateway.git
cd bedrock-gateway
pip install -e .

export AWS_BEARER_TOKEN_BEDROCK="你的令牌"
python -m bedrock_gateway
```

### systemd（推荐生产部署方式）

```bash
# 1. 依赖与虚拟环境
apt install -y python3.12-venv git
python3 -m venv /opt/bedrock-gateway
/opt/bedrock-gateway/bin/pip install git+https://github.com/wujiaming88/bedrock-gateway.git
ln -s /opt/bedrock-gateway/bin/bedrock-gateway /usr/local/bin/bedrock-gateway

# 2. 密钥（与配置分离）
cat > /opt/bedrock-gateway/.env << 'EOF'
AWS_BEARER_TOKEN_BEDROCK=你的-aws-bearer-token
BEDROCK_API_KEY=bgw-生成的密钥
BEDROCK_DASHBOARD_KEY=bgw-dash-生成的密钥
EOF
chmod 600 /opt/bedrock-gateway/.env

# 3. 配置
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

`/etc/systemd/system/bedrock-gateway.service`：

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
  -e AWS_BEARER_TOKEN_BEDROCK="你的令牌" \
  -e BEDROCK_API_KEY="bgw-你的密钥" \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  bedrock-gateway
```

### docker-compose

仓库自带的 `docker-compose.yml` 会挂载 `./config.yaml` 到容器，并从环境读取 `AWS_BEARER_TOKEN_BEDROCK` / `AWS_REGION`。

```bash
export AWS_BEARER_TOKEN_BEDROCK="你的令牌"
docker compose up -d
docker compose logs -f
```

### Nginx 反向代理（含 SSE 流式优化）

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

        # 流式必备
        proxy_buffering         off;
        proxy_cache             off;
        proxy_read_timeout      600s;
        proxy_send_timeout      600s;
    }
}
```

---

## Dashboard

`/dashboard/` 是实时请求监控的 Web 界面，对应的 JSON 接口在 `/api/metrics/*`。

### 鉴权方式

Dashboard 使用独立的 `dashboard.api_key`，与 `server.api_key` **完全分开**：

- 持有模型调用 key 的客户端无法访问 dashboard
- dashboard 管理员也无法调用 `/v1/*`

配置 `dashboard.api_key` 后，以下四种方式任选其一：

1. 登录 cookie（通过 `/dashboard/login` 表单）
2. `Authorization: Bearer <key>`
3. `x-api-key: <key>` 头
4. `?key=<key>` query 参数

### 访问规则

| 条件 | 行为 |
|---|---|
| `dashboard.api_key` 已设 + `require_auth: true` | 必须用任一方式鉴权 |
| `dashboard.api_key` 未设 | 仅 `127.0.0.1` / `::1` 可访问（除非 `localhost_only: false`） |
| `dashboard.enabled: false` | 不挂载路由，相关后台任务也不会启动 |

### 界面内容

- 顶部仪表盘：QPS、成功率、p50/p95 延迟、tokens/分钟
- 按模型的请求与 tokens 分布
- 1H / 6H / 24H 流量与延迟时序
- 可按状态过滤的最近请求表
- 按状态码与错误类型分组的错误面板
- 顶部状态条：版本、region、认证模式、uptime、RSS

所有 dashboard 响应自带安全头：`Content-Security-Policy`、`X-Frame-Options: DENY`、`X-Content-Type-Options: nosniff`、`X-XSS-Protection`、`Referrer-Policy: no-referrer`。
`/api/metrics/*` 按 IP 限流，默认 60 次/分钟，由 `dashboard.rate_limit` 控制。

---

## API

### 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI 聊天补全（同步与流式） |
| `POST` | `/v1/messages` | Anthropic messages（同步与流式） |
| `GET`  | `/v1/models` | 模型列表（OpenAI 格式） |
| `GET`  | `/health` | 健康检查（公开） |
| `GET`  | `/dashboard/` | 监控 UI |
| `GET`  | `/api/metrics/*` | 监控 JSON |

### OpenAI 参数（`/v1/chat/completions`）

| 参数 | 处理方式 |
|---|---|
| `messages` | `role=system` 抽出到 Bedrock `system` 字段 |
| `model` | 别名或原始 Bedrock 模型 ID |
| `stream` | 布尔，SSE |
| `max_tokens` / `max_completion_tokens` | 缺省时使用模型默认值 |
| `temperature` / `top_p` | 透传 |
| `stop` | 字符串或数组 → `stop_sequences` |
| `tools` / `tool_choice` | 转换为 Anthropic `tool_use` |
| `reasoning_effort` | 映射为 `thinking` budget |
| `thinking` | 透传 |
| `image_url` | base64 data URL 与远程 URL 都支持 |

### Anthropic 参数（`/v1/messages`）

直接透传：`messages`、`system`（字符串或 block 数组）、`max_tokens`、`temperature`、`top_p`、`top_k`、`stop_sequences`、`metadata`、`tools`、`tool_choice`、`thinking`、`stream`。
扩展思考流事件（`thinking_delta`、`signature_delta`、`redacted_thinking`）原样转发。底层模型返回 cache-token usage 时也会保留。

### 扩展思考（Extended Thinking）

```json
{
  "model": "claude-sonnet-4",
  "max_tokens": 4096,
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "messages": [...]
}
```

启用 `thinking` 时的两个细节：

- `temperature` 会被自动移除（Bedrock 在 thinking 模式下不接受）
- `budget_tokens` 会钳制到 ≥ 1024

`/v1/chat/completions` 接口里，`reasoning_effort: "low" | "medium" | "high"` 会自动映射为对应 `thinking` budget。

### 内置模型别名

| 别名 | Bedrock ID | 上下文 | 最大输出 |
|---|---|---|---|
| `claude-opus-4.7` | `us.anthropic.claude-opus-4-7` | 1M | 128K |
| `claude-opus-4` | `us.anthropic.claude-opus-4-6-v1` | 1M | 128K |
| `claude-sonnet-4.6` | `us.anthropic.claude-sonnet-4-6` | 1M | 64K |
| `claude-sonnet-4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | 200K | 64K |
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | 64K |
| `claude-sonnet-3.5` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | 64K |

常见变体（如 `claude-3-5-sonnet-latest`、`claude-sonnet-4-20250514`）会自动解析到规范别名，Anthropic SDK 默认模型名可以直接用。

---

## 日志与可观测性

网关的日志级别分配遵循"**ERROR 必须可告警**"原则：

| 现象 | 日志级别 | 含义 |
|---|---|---|
| 上游 200 OK / 业务请求 | INFO（一行 `REQ ...`） | 正常流量 |
| 上游 4xx（非 401/403） | WARNING | 客户端原因（图片过大、错误模型名等），无需告警 |
| 上游 401 / 403 | **ERROR + `[auth-failure]`** | 网关凭据失效，运维必须立刻处理 |
| 上游 5xx 或重试耗尽 | ERROR | 真正的故障 |
| 网关代码意外异常 | ERROR + traceback | 完整栈帧随日志写入 journald |

> 在 systemd 部署下，`journalctl -u bedrock-gateway -p err` 可只看真正的错误。

---

## 安全 Checklist

生产部署前请逐项确认：

- [ ] `BEDROCK_API_KEY` 设为强随机值（`bgw-$(openssl rand -base64 48)`）
- [ ] `BEDROCK_DASHBOARD_KEY` 设为强随机值，且与 `BEDROCK_API_KEY` 不同
- [ ] 密钥仅写入 `.env`（权限 `600`），不写入 `config.yaml`
- [ ] 绑定 `0.0.0.0` 时前面**必须**有 TLS 终止（Nginx / ALB / Cloudflare）
- [ ] 设置 `dashboard.api_key`（或直接 `dashboard.enabled: false`），并保持 `require_auth: true`
- [ ] 非 root 运行（systemd `User=` 或 Docker `appuser`）
- [ ] 日志接入集中收集（journalctl、容器日志驱动）
- [ ] Bedrock IAM 账号最小权限，仅保留必要的 `bedrock:InvokeModel*`

---

## 开发

```bash
git clone https://github.com/wujiaming88/bedrock-gateway.git
cd bedrock-gateway
pip install -e ".[dev]"

pytest -v
ruff check bedrock_gateway/ tests/
mypy bedrock_gateway/ --ignore-missing-imports
```

测试覆盖：531 个用例，`bedrock_gateway/server.py` 行覆盖 100%。

---

## License

MIT — 见 [LICENSE](LICENSE)。
