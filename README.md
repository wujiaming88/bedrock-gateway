# Bedrock Gateway

把发往 **OpenAI / Anthropic** 的 API 请求转发到 **AWS Bedrock**。
任何兼容 OpenAI 或 Anthropic SDK 的客户端，都可以**不改一行业务代码**接入 Bedrock —— 只需把 `base_url` 指向本网关。

[English](README.en.md) · [更新日志](CHANGELOG.md)

---

## 目录

- [为什么用它](#为什么用它)
- [支持的模型与端点](#支持的模型与端点)
- [快速开始](#快速开始)
- [安装与更新](#安装与更新)
- [配置](#配置)
- [使用](#使用)
- [部署（生产）](#部署生产)
- [监控面板 Dashboard](#监控面板-dashboard)
- [日志与可观测性](#日志与可观测性)
- [安全 Checklist](#安全-checklist)
- [开发](#开发)

---

## 为什么用它

- **现有应用零改造切到 Bedrock**：把 `base_url` 改到本网关即可，OpenAI / Anthropic 两套 SDK 都兼容。
- **免 boto3 / SigV4**：Bearer Token 模式开箱即用；也支持 AK/SK、IAM Role、Profile。
- **凭据集中托管**：客户端只持有一个网关 API key，AWS 凭据不下发到调用方。
- **多上游方言**：Claude 系走 Messages / Chat Completions，OpenAI GPT-5.5 走 Responses API，同一网关统一入口。
- **自带可观测性**：内置 dashboard，按模型 / 状态 / 时间维度展示请求量与延迟。

---

## 支持的模型与端点

网关按上游"方言"把请求分流到对应端点，**模型打错端点会返回带指引的 400**。

| 模型别名 | Bedrock ID | 上下文 | 最大输出 | 调用端点 |
|---|---|---|---|---|
| `claude-opus-4.8` | `us.anthropic.claude-opus-4-8` | 1M | 128K | `/v1/chat/completions`、`/v1/messages` |
| `claude-opus-4.7` | `us.anthropic.claude-opus-4-7` | 1M | 128K | 同上 |
| `claude-opus-4` | `us.anthropic.claude-opus-4-6-v1` | 1M | 128K | 同上 |
| `claude-sonnet-4.6` | `us.anthropic.claude-sonnet-4-6` | 1M | 64K | 同上 |
| `claude-sonnet-4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | 200K | 64K | 同上 |
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | 64K | 同上 |
| `claude-sonnet-3.5` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | 64K | 同上 |
| `gpt-5.5` | `openai.gpt-5.5` | 272K | 64K | **`/openai/v1/responses`** |
| `grok-4.3` | `xai.grok-4.3` | 1M | 128K | **`/openai/v1/responses`** |

- Claude 系别名有大量常见变体自动解析（如 `claude-3-5-sonnet-latest`、`claude-sonnet-4-20250514`、Anthropic SDK 默认模型名）。
- GPT-5.5 别名：`gpt-5.5` / `gpt-55` / `gpt5.5` / `gpt-5-5`。
- Grok 4.3 别名：`grok-4.3` / `grok` / `grok-4` / `grok4.3` / `grok-4-3`。
- 请求 `model` 也可直接传原始 Bedrock ID（以 `us.` / `anthropic.` / `openai.` / `xai.` 等开头的按 passthrough 处理）。

**端点速查**

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions（Claude 系，同步 + 流式） |
| `POST` | `/v1/messages` | Anthropic Messages（Claude 系，同步 + 流式） |
| `POST` | `/openai/v1/responses` | OpenAI Responses（GPT-5.5 / Grok 4.3，透传，同步 + 流式） |
| `GET` | `/v1/models` | 模型列表（OpenAI 格式） |
| `GET` | `/health` | 健康检查（公开，无需鉴权） |
| `GET` | `/dashboard/` | 监控界面 |
| `GET` | `/api/metrics/*` | 监控 JSON API |

---

## 快速开始

```bash
pip install git+https://github.com/wujiaming88/bedrock-gateway.git

export AWS_BEARER_TOKEN_BEDROCK="你的-aws-bearer-token"
bedrock-gateway
# listening on http://127.0.0.1:4000
```

验证：

```bash
curl http://127.0.0.1:4000/health
# {"status":"ok","version":"0.2.0","auth_mode":"bearer_token","region":"us-east-1","models":8}
```

发一条 Claude 请求：

```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
```

---

## 安装与更新

### 首次安装

从 GitHub 直接安装（推荐用虚拟环境隔离）：

```bash
python3 -m venv /opt/bedrock-gateway
/opt/bedrock-gateway/bin/pip install git+https://github.com/wujiaming88/bedrock-gateway.git

# 启动脚本会装到 venv 的 bin/ 下
/opt/bedrock-gateway/bin/bedrock-gateway
```

需要 IAM Role / Profile 鉴权时，额外装 boto3：

```bash
/opt/bedrock-gateway/bin/pip install "bedrock-gateway[boto3] @ git+https://github.com/wujiaming88/bedrock-gateway.git"
```

### 更新到新版本

网关以常规包形式装在 venv 的 `site-packages` 里，更新 = **重装包 + 重启进程**。运行中的进程加载的是启动时的代码，装完新版必须重启才生效。

**方式 A — 从 GitHub 拉最新代码更新（生产常用）**

```bash
# 1) 记录当前版本（便于回滚核对）
curl -s http://127.0.0.1:4000/health | grep -o '"version":"[^"]*"'

# 2) 重装（--upgrade 强制拉最新；--no-deps 跳过依赖重装，加快速度）
/opt/bedrock-gateway/bin/pip install --upgrade --force-reinstall --no-deps \
  git+https://github.com/wujiaming88/bedrock-gateway.git

# 3) 校验 site-packages 已是新版（在中立目录执行，避免被当前目录源码干扰）
cd /tmp && /opt/bedrock-gateway/bin/python -c \
  "import bedrock_gateway; print(bedrock_gateway.__version__, bedrock_gateway.__file__)"

# 4) 重启服务
systemctl restart bedrock-gateway

# 5) 验证新版本已生效
curl -s http://127.0.0.1:4000/health
```

**方式 B — 从本地源码目录更新（开发 / 已 clone 仓库）**

```bash
cd /path/to/bedrock-gateway
git pull

/opt/bedrock-gateway/bin/pip install --no-deps .   # 装当前目录源码到 venv
systemctl restart bedrock-gateway
```

> ⚠️ **校验版本时务必切到中立目录（如 `/tmp`）再执行 `python -c "import bedrock_gateway"`**。
> 若在源码目录（含 `bedrock_gateway/` 子目录）下执行，Python 会优先加载当前目录的源码，
> 显示的版本可能并非 venv 里真正安装的版本，造成"看着已更新、实际没装进去"的假象。

### 回滚

```bash
# 回滚到某个已发布 tag / commit
/opt/bedrock-gateway/bin/pip install --upgrade --force-reinstall --no-deps \
  "git+https://github.com/wujiaming88/bedrock-gateway.git@v0.1.4"
systemctl restart bedrock-gateway
```

配置文件（`config.yaml` / `.env`）与代码解耦，升级 / 回滚都不会动它们。

---

## 配置

配置来自**进程工作目录下的 `config.yaml`**（启动前 `cd` 到该目录，或在 systemd 里用 `WorkingDirectory=` 指定）。
字符串值支持 `${VAR}` 从环境变量插值（变量未定义时展开为空）。整份配置可省略——省略的部分走内置默认。

### 最小可用配置

```yaml
# config.yaml —— 只用 Bearer Token，其余全默认
auth:
  mode: bearer_token
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}
region: us-east-1
```

### 完整配置样例（含注释）

```yaml
# ── 认证：同一时间只启用一种凭据来源 ──
auth:
  mode: bearer_token                    # bearer_token | credentials | iam_role | profile
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}

  # mode: credentials —— 静态 AK/SK，进程内做 SigV4
  # access_key_id: ${AWS_ACCESS_KEY_ID}
  # secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  # session_token: ${AWS_SESSION_TOKEN}   # 临时凭证才需要

  # mode: iam_role                        # EC2/ECS/Lambda 自动取凭据（需 boto3）
  # mode: profile                         # 用 ~/.aws/credentials（需 boto3）
  # profile: default

region: us-east-1

server:
  host: 0.0.0.0                           # 默认 127.0.0.1；对外暴露才用 0.0.0.0
  port: 4000
  log_level: info                         # debug | info | warning | error
  api_key: ${BEDROCK_API_KEY}             # 设置后 /v1/*、/openai/* 强制鉴权

retry:
  max_retries: 3                          # 总尝试次数
  base_delay: 1.0                         # 秒；实际延迟 = base_delay * 2^attempt

dashboard:
  enabled: true                           # false 则完全不挂载 dashboard 路由
  api_key: ${BEDROCK_DASHBOARD_KEY}       # 独立鉴权，与 server.api_key 无关
  require_auth: true
  # localhost_only: false                 # 显式覆盖默认（见下）
  rate_limit: 60                          # /api/metrics/* 每 IP 每分钟限流
  max_request_log: 500                    # 请求日志面板保留的最近条数

# ── 模型别名（省略整段则用内置默认）──
# models:
#   my-model:
#     bedrock_id: us.my-org.my-model-v1
#     context_length: 100000
#     max_output: 8192
#   # mantle 端点 + Responses 协议模型示例（如 GPT-5.5）：
#   gpt-5.5:
#     bedrock_id: openai.gpt-5.5
#     context_length: 272000
#     max_output: 64000
#     endpoint: mantle                    # runtime（默认）| mantle
#     protocol: openai-responses          # anthropic（默认）| openai-responses
```

### 认证模式一览

| 模式 | 适用场景 | 需 boto3 |
|---|---|:---:|
| `bearer_token` | Bedrock API key，最简单，不走 SigV4 | 否 |
| `credentials` | 静态 AK/SK，进程内做 SigV4 签名 | 否 |
| `iam_role` | EC2 / ECS / Lambda 元数据服务自动取凭据 | **是** |
| `profile` | 使用 `~/.aws/credentials` 中的命名 profile | **是** |

### 鉴权与访问控制

- 设置 `server.api_key` 后，调用 `/v1/*`、`/openai/*` 需带 `Authorization: Bearer <key>` 或 `x-api-key: <key>`；比较用 `hmac.compare_digest`，防时序攻击。
- `/health` 与 `/` 始终公开，便于探活。
- `dashboard.localhost_only` 缺省行为：未配 `dashboard.api_key` → 默认仅本机可访问；已配 → 默认可远程访问但需鉴权。

### 环境变量快捷配置

仅在 `config.yaml` 对应字段缺省时生效。

| 变量 | 默认值 | 对应字段 |
|---|---|---|
| `AWS_BEARER_TOKEN_BEDROCK` | — | `auth.bearer_token` |
| `BEDROCK_API_KEY` | — | `server.api_key` |
| `BEDROCK_DASHBOARD_KEY` | — | `dashboard.api_key` |
| `AWS_REGION` | `us-east-1` | `region` |
| `BEDROCK_HOST` | `127.0.0.1` | `server.host` |
| `BEDROCK_PORT` | `4000` | `server.port` |
| `BEDROCK_LOG_LEVEL` | `info` | `server.log_level` |
| `BEDROCK_AUTH_MODE` | `bearer_token` | `auth.mode` |
| `BEDROCK_MAX_RETRIES` | `3` | `retry.max_retries` |

---

## 使用

下列样例假设网关跑在 `http://127.0.0.1:4000`，且已设 `server.api_key`（未设时 `api_key` 字段填任意值即可）。

### Claude —— OpenAI SDK（Chat Completions）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="<gateway-key>")
resp = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

### Claude —— Anthropic SDK（Messages）

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:4000", api_key="<gateway-key>")
msg = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content[0].text)
```

### GPT-5.5 / Grok 4.3 —— OpenAI SDK（Responses）

这类模型在 Bedrock 上经 `bedrock-mantle` 的 OpenAI Responses API 提供；网关**原样透传**请求（仅把模型别名替换为上游 ID），响应与 SSE 流也原样回吐。注意 base_url 用 **`/openai/v1`**：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4000/openai/v1", api_key="<gateway-key>")
resp = client.responses.create(model="gpt-5.5", input="用一句话解释 ETF")
print(resp.output[0].content[0].text)

# Grok 4.3 同一用法，换 model 即可（官方定位金融/法律文档分析）
resp = client.responses.create(model="grok-4.3", input="分析这份财报的关键风险")
```

**图片输入**（`input_image` 支持 `data:` base64 与 `s3://`，**不支持公网 http(s) URL**）：

```python
resp = client.responses.create(
    model="gpt-5.5",
    input=[{"role": "user", "content": [
        {"type": "input_text", "text": "这张图什么颜色？"},
        {"type": "input_image", "image_url": "data:image/png;base64,<BASE64>"},
    ]}],
)
```

**流式**：`stream=True`，标准 SSE（`response.output_text.delta` 增量、末尾 `response.completed` 携带 usage）。请求体不做字段清洗，`reasoning`、`tools`、`input_image` 等原样转发。

### 扩展思考（Extended Thinking，Claude）

```json
{
  "model": "claude-sonnet-4",
  "max_tokens": 4096,
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "messages": [{"role": "user", "content": "..."}]
}
```

- 启用 `thinking` 时 `temperature` 会被自动移除（Bedrock 在 thinking 模式下不接受），`budget_tokens` 钳制到 ≥ 1024。
- `/v1/chat/completions` 里 `reasoning_effort: "low" | "medium" | "high"` 会自动映射为对应 `thinking` budget；4.6/4.7/4.8 等自适应模型映射为 `{"type":"adaptive"}`。

### 各端点参数处理

**`/v1/chat/completions`（OpenAI → Anthropic）**

| 参数 | 处理方式 |
|---|---|
| `messages` | `role=system` 抽出到 Bedrock `system` 字段 |
| `model` | 别名或原始 Bedrock ID |
| `stream` | 布尔，SSE |
| `max_tokens` / `max_completion_tokens` | 缺省时用模型默认值 |
| `temperature` / `top_p` | 透传 |
| `stop` | 字符串或数组 → `stop_sequences` |
| `tools` / `tool_choice` | 转换为 Anthropic `tool_use` |
| `reasoning_effort` | 映射为 `thinking` budget |
| `thinking` | 透传 |
| `image_url` | base64 data URL 与远程 URL 都支持 |

**`/v1/messages`（Anthropic 透传）**：`messages`、`system`、`max_tokens`、`temperature`、`top_p`、`top_k`、`stop_sequences`、`metadata`、`tools`、`tool_choice`、`thinking`、`stream` 均直接透传；思考流事件（`thinking_delta` / `signature_delta` / `redacted_thinking`）与 cache-token usage 原样保留。

**`/openai/v1/responses`（Responses 透传）**：请求体除模型别名替换外不做改写，字段原样转发上游。

---

## 部署（生产）

### systemd（推荐）

```bash
# 1) 虚拟环境与安装
apt install -y python3-venv git
python3 -m venv /opt/bedrock-gateway
/opt/bedrock-gateway/bin/pip install git+https://github.com/wujiaming88/bedrock-gateway.git

# 2) 密钥（与配置分离，权限 600）
cat > /opt/bedrock-gateway/.env << 'EOF'
AWS_BEARER_TOKEN_BEDROCK=你的-aws-bearer-token
BEDROCK_API_KEY=bgw-生成的密钥
BEDROCK_DASHBOARD_KEY=bgw-dash-生成的密钥
EOF
chmod 600 /opt/bedrock-gateway/.env

# 3) 配置
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

> 更新此部署时，见上文 [安装与更新](#更新到新版本)。

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

`docker-compose`：仓库自带 `docker-compose.yml` 挂载 `./config.yaml` 并从环境读取 `AWS_BEARER_TOKEN_BEDROCK` / `AWS_REGION`。

```bash
export AWS_BEARER_TOKEN_BEDROCK="你的令牌"
docker compose up -d && docker compose logs -f
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
        proxy_pass         http://bedrock_gateway;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        # 流式必备
        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

---

## 监控面板 Dashboard

`/dashboard/` 是实时请求监控界面，JSON 接口在 `/api/metrics/*`。

**鉴权**：dashboard 用独立的 `dashboard.api_key`，与 `server.api_key` **完全分开**——持有模型调用 key 的客户端无法访问 dashboard，反之亦然。配置后可用四种方式之一：登录 cookie（`/dashboard/login`）、`Authorization: Bearer <key>`、`x-api-key: <key>` 头、`?key=<key>` query。

| 条件 | 行为 |
|---|---|
| `dashboard.api_key` 已设 + `require_auth: true` | 必须鉴权 |
| `dashboard.api_key` 未设 | 仅 `127.0.0.1` / `::1` 可访问（除非 `localhost_only: false`） |
| `dashboard.enabled: false` | 不挂载路由，相关后台任务也不启动 |

**界面内容**：顶部 QPS / 成功率 / p50·p95 延迟 / tokens·分钟；按模型的请求与 token 分布；1H/6H/24H 时序；可按状态过滤的最近请求表；按状态码与错误类型分组的错误面板；状态条（版本、region、认证模式、uptime、RSS）。

所有 dashboard 响应自带安全头（CSP、`X-Frame-Options: DENY`、`X-Content-Type-Options: nosniff` 等）；`/api/metrics/*` 按 IP 限流（默认 60 次/分，`dashboard.rate_limit` 控制）。

---

## 日志与可观测性

日志级别遵循"**ERROR 必须可告警**"原则：

| 现象 | 级别 | 含义 |
|---|---|---|
| 上游 200 / 业务请求 | INFO（一行 `REQ ...`） | 正常流量 |
| 上游 4xx（非 401/403） | WARNING | 客户端原因（图片过大、错误模型名等），无需告警 |
| 上游 401 / 403 | **ERROR + `[auth-failure]`** | 网关凭据失效，运维须立即处理 |
| 上游 5xx 或重试耗尽 | ERROR | 真正的故障 |
| 网关代码意外异常 | ERROR + traceback | 完整栈帧写入 journald |

> systemd 部署下，`journalctl -u bedrock-gateway -p err` 只看真正的错误。

---

## 安全 Checklist

生产部署前逐项确认：

- [ ] `BEDROCK_API_KEY` 设为强随机值（`bgw-$(openssl rand -base64 48)`）
- [ ] `BEDROCK_DASHBOARD_KEY` 设为强随机值，且与 `BEDROCK_API_KEY` 不同
- [ ] 密钥仅写入 `.env`（权限 `600`），不写入 `config.yaml`
- [ ] 绑定 `0.0.0.0` 时前面**必须**有 TLS 终止（Nginx / ALB / Cloudflare）
- [ ] 设 `dashboard.api_key`（或 `dashboard.enabled: false`），并保持 `require_auth: true`
- [ ] 非 root 运行（systemd `User=` 或 Docker 非 root 用户）
- [ ] 日志接入集中收集（journalctl、容器日志驱动）
- [ ] Bedrock IAM 账号最小权限，仅保留必要的 `bedrock:InvokeModel*`

---

## 开发

```bash
git clone https://github.com/wujiaming88/bedrock-gateway.git
cd bedrock-gateway
pip install -e ".[dev]"

pytest -q                                   # 632 用例
ruff check bedrock_gateway/ tests/
mypy bedrock_gateway/ --ignore-missing-imports
```

**架构**：新增模型 / 上游厂商 = 加一个 Provider（`bedrock_gateway/providers/`）+ 一条 `ModelEntry`，`server.py` 的编排逻辑（重试 / 预检 / 超时 / metrics）不用改。Provider 封装"如何与某个上游对话"（URL 构造、请求 / 响应渲染、流转换）：`AnthropicBedrockProvider` 走 `bedrock-runtime` + Anthropic 格式，`OpenAIResponsesProvider` 走 `bedrock-mantle` + Responses 透传。设计细节见 [`docs/provider-abstraction-design.md`](docs/provider-abstraction-design.md)。

---

## License

MIT —— 见 [LICENSE](LICENSE)。
