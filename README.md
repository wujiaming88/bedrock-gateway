# MuxLane

**The unified lane for AI traffic.**

> 原项目名 Bedrock Gateway。整个 `0.x` 保留 `bedrock-gateway` CLI、`bedrock_gateway` Python import、systemd unit、部署目录、日志与环境变量兼容；新命令推荐使用 `muxlane`。

把发往 **OpenAI / Anthropic** 的 API 请求转发到 **AWS Bedrock** 与 **Azure OpenAI**。
任何兼容 OpenAI 或 Anthropic SDK 的客户端，都可以**不改一行业务代码**接入多云模型 —— 只需把 `base_url` 指向本网关。

[English](README.en.md) · [更新日志](CHANGELOG.md)

---

## 目录

- [为什么用它](#为什么用它)
- [支持的模型与端点](#支持的模型与端点)
- [快速开始](#快速开始)
- [安装与更新](#安装与更新)
- [配置](#配置)
- [多云与 Azure](#多云与-azure)
- [使用](#使用)
- [Claude Code 接入任意模型](#claude-code-接入任意模型)
- [部署（生产）](#部署生产)
- [监控面板 Dashboard](#监控面板-dashboard)
- [日志与可观测性](#日志与可观测性)
- [安全 Checklist](#安全-checklist)
- [开发](#开发)

---

## 为什么用它

- **现有应用零改造切换后端**：把 `base_url` 改到本网关即可，OpenAI / Anthropic 两套 SDK 都兼容。
- **多云统一入口**：AWS Bedrock（Claude / GPT-5.5 / Grok）与 Azure OpenAI 经同一网关调用，客户端不感知底层是哪个云。
- **免 boto3 / SigV4**：Bearer Token 模式开箱即用；也支持 AK/SK、IAM Role、Profile；Azure 用 api-key。
- **凭据集中托管**：客户端只持有一个网关 API key，云厂商凭据不下发到调用方。
- **多上游方言**：Claude 走 Messages / Chat Completions（转换），OpenAI / Azure 走 Responses / Chat Completions（透传），由 Transport × Dialect 两轴统一编排。
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
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | 64K | 同上 |
| `gpt-5.5` | `openai.gpt-5.5` | 1.05M | 128K | **`/openai/v1/responses`** |
| `gpt-5.6-sol` | `openai.gpt-5.6-sol` | 1M | 128K* | **`/openai/v1/responses`** |
| `gpt-5.6-terra` | `openai.gpt-5.6-terra` | 1M | 128K* | **`/openai/v1/responses`** |
| `gpt-5.6-luna` | `openai.gpt-5.6-luna` | 1M | 128K* | **`/openai/v1/responses`** |
| `grok-4.3` | `xai.grok-4.3` | 1M | 128K | **`/openai/v1/responses`** |
| Azure 模型（自配） | Azure deployment | — | — | 按 dialect：Responses → `/openai/v1/responses`；Chat → `/v1/chat/completions` |

- Claude 系别名有大量常见变体自动解析：Opus 的点号与连字符写法均可（`claude-opus-4.7` ＝ `claude-opus-4-7`，4.8 / 4.6 同理），以及 Anthropic SDK 默认模型名、带日期的官方名（如 `claude-opus-4-7-20250428`）。裸 `claude-sonnet` 解析到当前最新的 Sonnet（4.6）。
- GPT-5.5 别名：`gpt-5.5` / `gpt-55` / `gpt5.5` / `gpt-5-5`；GPT-5.6 家族别名：`gpt-5.6-sol` / `gpt-56-sol` / `gpt5.6-sol` / `gpt-5-6-sol`（Terra/Luna 同理）；Grok 4.3 别名：`grok-4.3` / `grok` / `grok-4` / `grok4.3` / `grok-4-3`。
- `gpt-5.6-*` 在 Bedrock mantle 上均为 1M 上下文；128K 最大输出字段仍为 advisory，若官方 model card 后续给出不同规格，应同步修正。
- **Azure OpenAI**：多云支持，需在 config 里配 `azure_resources`（endpoint + key）+ 模型条目（见 [多云与 Azure](#多云与-azure)）。
- 请求 `model` 也可直接传原始 Bedrock ID（以 `us.` / `anthropic.` / `openai.` / `xai.` 等开头的按 passthrough 处理）。

**端点速查**

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions（Claude 转换 / Azure·mantle 透传，同步 + 流式） |
| `POST` | `/v1/messages` | Anthropic Messages。Claude 系透传；**GPT-5.5 / Grok / `azure/<dep>` 自动翻译**为 Responses（让 Claude Code 等 Anthropic-only 客户端调任意模型，见 [Claude Code 接入](#claude-code-接入任意模型)） |
| `POST` | `/openai/v1/responses` | OpenAI Responses（Bedrock GPT-5.x / Grok 4.3 / Azure，同步 + 流式；Bedrock GPT-5.x 会做最小兼容 normalizer 以适配 Codex input） |
| `POST` | `/openai/v1/images/generations` | OpenAI Images Generations（Azure `gpt-image-2`，透传，同步） |
| `POST` | `/openai/v1/images/edits` | OpenAI Images Edits（Azure `gpt-image-2`，multipart，同步 + SSE） |
| `GET` | `/v1/models` | 模型列表（OpenAI 格式） |
| `GET` | `/health` | 健康检查（公开，无需鉴权） |
| `GET` | `/dashboard/` | 监控界面 |
| `GET` | `/api/metrics/*` | 监控 JSON API |

**架构：Transport × Dialect 两轴**。每个模型 = 一对 `(transport, dialect)`：
`transport` 决定去哪个云、怎么鉴权（`bedrock` 默认 / `azure`）；`dialect` 决定请求/响应形状（`anthropic` 默认 / `openai-responses` / `openai-chat`）。加一个云 = 加一个 transport；加一种格式 = 加一个 dialect。详见 [`docs/multi-cloud-multimodal-design.md`](docs/multi-cloud-multimodal-design.md)。

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
# {"status":"ok","version":"0.4.2","auth_mode":"bearer_token","region":"us-east-1","models":7}
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

网关以常规包形式装在 venv 的 `site-packages` 里，更新 = **安装包及依赖 + 重启进程**。生产更新请指定已发布 tag；不要跳过依赖安装，否则旧环境可能缺少新版本增加的运行时依赖。

```bash
TARGET_VERSION=v0.4.14

# 1) 记录当前版本（便于回滚核对）
curl -fsS http://127.0.0.1:4000/health

# 2) 安装目标版本并同步依赖
/opt/bedrock-gateway/bin/python -m pip install --upgrade --force-reinstall \
  "git+https://github.com/wujiaming88/bedrock-gateway.git@${TARGET_VERSION}"

# 3) 在停止旧进程前检查依赖和启动导入
/opt/bedrock-gateway/bin/python -m pip check
cd /tmp
/opt/bedrock-gateway/bin/python -c \
  "import bedrock_gateway; import bedrock_gateway.server; print(bedrock_gateway.__version__, bedrock_gateway.__file__)"

# 4) 预检通过后才重启
systemctl restart bedrock-gateway

# 5) 验证服务和目标版本；失败时立即查看完整日志
curl -fsS http://127.0.0.1:4000/health | /opt/bedrock-gateway/bin/python -c \
  'import json,sys; d=json.load(sys.stdin); assert d["status"]=="ok" and "v"+d["version"]==sys.argv[1], d; print(d)' "$TARGET_VERSION"
systemctl status bedrock-gateway --no-pager -l
# journalctl -u bedrock-gateway -n 100 --no-pager -l
```

从本地源码目录更新时同样必须安装依赖并执行预检：

```bash
cd /path/to/bedrock-gateway
git pull
/opt/bedrock-gateway/bin/python -m pip install --upgrade --force-reinstall .
/opt/bedrock-gateway/bin/python -m pip check
cd /tmp && /opt/bedrock-gateway/bin/python -c \
  "import bedrock_gateway; import bedrock_gateway.server; print(bedrock_gateway.__version__, bedrock_gateway.__file__)"
systemctl restart bedrock-gateway
curl -fsS http://127.0.0.1:4000/health
```

> ⚠️ **校验版本时务必切到中立目录（如 `/tmp`）**。在源码目录中导入可能加载当前目录的源码，而不是 venv 中实际安装的包。只有依赖由外部配置管理系统完整维护并另有等效校验时，才可自行跳过 pip 的依赖解析；这不是通用升级方式。

### 回滚

回滚也必须同步目标版本的依赖，并在重启前完成相同预检：

```bash
TARGET_VERSION=v0.4.13
/opt/bedrock-gateway/bin/python -m pip install --upgrade --force-reinstall \
  "git+https://github.com/wujiaming88/bedrock-gateway.git@${TARGET_VERSION}"
/opt/bedrock-gateway/bin/python -m pip check
cd /tmp && /opt/bedrock-gateway/bin/python -c \
  "import bedrock_gateway; import bedrock_gateway.server; print(bedrock_gateway.__version__, bedrock_gateway.__file__)"
systemctl restart bedrock-gateway
curl -fsS http://127.0.0.1:4000/health
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

logging:
  file_enabled: true                      # 默认开启；显式设 false 可关闭
  directory: /var/log/bedrock-gateway     # 每天一个 bedrock-gateway-YYYY-MM-DD.log
  retention_days: 30                      # 自动删除超过 30 天的日志

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

# ── 模型别名（与内置默认合并；同名覆盖）──
# use_default_models: true                # false 则只用下面配的，不含内置默认
# models:
#   my-model:                             # 新增一个模型，默认模型仍保留
#     bedrock_id: us.my-org.my-model-v1
#     context_length: 100000
#     max_output: 8192
```

### 认证模式一览

| 模式 | 适用场景 | 需 boto3 |
|---|---|:---:|
| `bearer_token` | Bedrock API key，最简单，不走 SigV4 | 否 |
| `credentials` | 静态 AK/SK，进程内做 SigV4 签名 | 否 |
| `iam_role` | EC2 / ECS / Lambda 元数据服务自动取凭据 | **是** |
| `profile` | 使用 `~/.aws/credentials` 中的命名 profile | **是** |

> ⚠️ **SigV4 模式（`credentials` / `iam_role` / `profile`）只适用于 Bedrock runtime 模型**（Claude 系）。它们对 **mantle 端点（GPT-5.5 / Grok）和 Azure 签不出有效凭据**——mantle 主机/服务不同、Azure 要 `api-key` header。要用 mantle 模型请配 `bearer_token`，Azure 由资源的 `api_key` 承担。若在 SigV4 模式下配了 mantle/Azure 模型，网关启动时会 WARNING 列出这些不可达的模型（服务照常运行，Bedrock 模型不受影响）。

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

## 多云与 Azure

网关按 **Transport × Dialect 两轴** 支持多云。除 AWS Bedrock 外，可接入 **Azure OpenAI**。

### 概念

- **Azure 凭据是资源级的**：每个 Azure OpenAI 资源有自己独立的 endpoint 和 api-key（跨资源不通用）。一个资源下可部署多个模型（deployment），共享该资源的 endpoint + key。
- 配置分两层：`azure_resources`（资源 = endpoint + key）+ `models`（模型引用资源 + 指定 deployment）。

### 配置样例

```yaml
# 资源级：endpoint + key（一个资源配一次）
# base_url 用 v1 根（.../openai/v1）：同时支持 Responses 与 Chat，无需 api-version
azure_resources:
  my-azure:
    base_url: https://<resource>.cognitiveservices.azure.com/openai/v1
    api_key: ${AZURE_OPENAI_KEY}

models:
  # Responses 模型 → 走 /openai/v1/responses
  azure-gpt-5:
    transport: azure
    dialect: openai-responses
    azure_resource: my-azure       # 引用上面的资源
    deployment: gpt-5              # 你的 Azure deployment 名 → 请求体 model 字段

  # Chat Completions 模型 → 走 /v1/chat/completions
  azure-gpt-4o:
    transport: azure
    dialect: openai-chat
    azure_resource: my-azure
    deployment: gpt-4o
```

> `models:` 与内置默认模型**合并**：只需列出要新增的模型，Claude/GPT-5.5/Grok 等默认仍可用；同名条目会覆盖默认。想只暴露自己配的模型，设 `use_default_models: false`。

### 前缀透传（零模型登记，推荐）

给资源加 `prefix`，即可**不逐个登记模型**——客户端用 `<prefix>/<deployment>` 调用，网关自动路由到该资源、剥离前缀作为 deployment，dialect 由端点决定（`/openai/v1/responses` → responses，`/v1/chat/completions` → chat，`/openai/v1/images/generations` → images）：

```yaml
azure_resources:
  az:
    base_url: https://<resource>.cognitiveservices.azure.com/openai/v1
    api_key: ${AZURE_OPENAI_KEY}
    prefix: azure          # 启用前缀透传
# 无需 models: 段
```

```python
# 客户端：deployment 想调哪个就写哪个，网关不用改
client.responses.create(model="azure/gpt-5.5", input="...")          # → responses
client.chat.completions.create(model="azure/gpt-5", messages=[...])  # → chat
client.images.generate(model="azure/gpt-image-2", prompt="...")     # → images
```

deployment 增删换名都不用动网关配置。代价：`/v1/models` 列不出这些透传模型（网关事先不知道有哪些）。需要别名/模型清单/dashboard 分模型统计时，用上面的显式 `models:` 登记；两者可共存。

### 调用

Azure 模型的对外端点由其 `dialect` 决定，客户端用法与 Bedrock 上同类模型完全一致——只是 `model` 换成你配的别名：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4000/openai/v1", api_key="<gateway-key>")

# Azure Responses 模型
client.responses.create(model="azure-gpt-5", input="分析这份财报")

# Azure Chat 模型（base_url 用 /v1）
OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="<gateway-key>") \
    .chat.completions.create(model="azure-gpt-4o",
        messages=[{"role": "user", "content": "hi"}])
```

网关对 Azure 做**透传**（仅把别名换成 deployment 名），响应含 Azure 的 `content_filter_results` 字段原样保留；鉴权自动用 `api-key` header。URL 里保留资源 base 上的 `?api-version=...`（若有）。

### 通用 HTTP 上游与 DeepSeek

`upstream_resources` 可把一个官方上游的多种原生协议注册到统一网关。以下配置让同一模型别名同时服务 Chat、Responses 和 Anthropic Messages，网关只做路由、鉴权和 model 替换，不做重复协议转换：

```yaml
upstream_resources:
  deepseek:
    prefix: deepseek
    secret_env: DEEPSEEK_API_KEY
    routes:
      openai-chat:
        base_url: https://api.deepseek.com
        path: /chat/completions
        auth: bearer
      openai-responses:
        base_url: https://api.deepseek.com
        path: /responses
        auth: bearer
      anthropic-passthrough:
        base_url: https://api.deepseek.com/anthropic
        path: /v1/messages
        auth: x-api-key
        default_headers:
          anthropic-version: "2023-06-01"

models:
  deepseek-v4-flash:
    upstream_resource: deepseek
    upstream_id: deepseek-v4-flash
    dialect: openai-chat
    context_length: 1000000
    max_output: 393216
  deepseek-v4-pro:
    upstream_resource: deepseek
    upstream_id: deepseek-v4-pro
    dialect: openai-chat
    context_length: 1000000
    max_output: 393216
```

把 Key 只放入受保护的环境文件：

```bash
DEEPSEEK_API_KEY=...
chmod 600 /opt/bedrock-gateway/.env
```

调用端点：

- `/v1/chat/completions`：DeepSeek 原生 Chat，保留 thinking、`reasoning_content`、tools、JSON、cache usage 和 `[DONE]`。
- `/openai/v1/responses`：原生无状态 Responses，保留 typed SSE，且不会应用 Bedrock Mantle normalizer。
- `/v1/messages`：原生 Anthropic passthrough，保留 Messages JSON/SSE 与上游错误状态。

也可使用 `deepseek/<model>` 动态前缀，不逐项登记模型；显式登记才能出现在 `/v1/models`。当前官方模型是 `deepseek-v4-flash` 和 `deepseek-v4-pro`，旧 `deepseek-chat` / `deepseek-reasoner` 不作为默认 alias。DeepSeek Beta Strict Tools、Prefix Completion 和 FIM 暂未接入。

> Thinking + Tools 时必须完整保存并回传 assistant 的 `reasoning_content`；官方不支持该模式下的强制 `tool_choice`。网关按官方 wire 原样透传，不会伪造或补齐这段历史。

---

## 使用

下列样例假设网关跑在 `http://127.0.0.1:4000`，且已设 `server.api_key`（未设时 `api_key` 字段填任意值即可）。

### Claude —— OpenAI SDK（Chat Completions）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="<gateway-key>")
resp = client.chat.completions.create(
    model="claude-sonnet-4.6",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

### Claude —— Anthropic SDK（Messages）

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:4000", api_key="<gateway-key>")
msg = client.messages.create(
    model="claude-sonnet-4.6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content[0].text)
```

### GPT-5.x / Grok 4.3 —— OpenAI SDK（Responses）

这类模型在 Bedrock 上经 `bedrock-mantle` 的 OpenAI Responses API 提供；网关转发到上游前会把模型别名替换为上游 ID。对 Bedrock GPT-5.x,网关还会做一层**最小兼容 normalizer**：Codex 的 `additional_tools` input item 会提升到 top-level `tools`,developer message 会合并到 `instructions`,并把 `text` content block 标准化为 `input_text`。其他字段（`reasoning`、`tool_choice`、`stream` 等）保持透传。注意 base_url 用 **`/openai/v1`**：

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

### Azure 图片生成 —— OpenAI SDK（Images）

Azure 的图片模型（如 `gpt-image-2`）不走 Responses 文本端点，而是走 **`/openai/v1/images/generations`**。如果 `azure_resources` 配了 `prefix: azure`，无需登记模型，直接用 `azure/<deployment>`：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4000/openai/v1", api_key="<gateway-key>")
img = client.images.generate(
    model="azure/gpt-image-2",
    prompt="a small red square icon",
    size="1024x1024",
    n=1,
)
print(img.data[0].b64_json[:40])
```

网关对 Images 请求做**透传**（仅把 `azure/gpt-image-2` 剥离成上游 deployment `gpt-image-2`），响应的 `b64_json` / URL 字段原样返回。generations 为同步 JSON，不支持 `stream=true`。

编辑使用 multipart 文件上传，并支持 Azure SSE 原样透传：

```python
with open("input.png", "rb") as image:
    edited = client.images.edit(
        model="azure/gpt-image-2",
        image=image,
        prompt="change the red square to blue",
        n=1,
    )
print(edited.data[0].b64_json[:40])
```

`/openai/v1/images/edits` 面向 Azure OpenAI Images deployment，不按模型名称建立 allowlist；具体 deployment 是否支持 edits 由 Azure 上游判定并返回规范错误。当前以 `gpt-image-2` 做严格真机验证。暂不支持 Images Variations、Nova Canvas 或 Titan Image Generator。

### 扩展思考（Extended Thinking，Claude）

```json
{
  "model": "claude-sonnet-4.6",
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

**`/openai/v1/responses`（Responses）**：请求体除模型别名替换外基本透传；Bedrock GPT-5.x 会做最小兼容 normalizer（`additional_tools`→`tools`、developer message→`instructions`、`text` block→`input_text`）以适配 Codex/Bedrock 方言差异。

**`/openai/v1/images/generations`（Images 透传）**：JSON 请求除模型别名/前缀剥离外不做改写，字段原样转发上游；响应的 `data[].b64_json` / URL 原样返回；不支持 `stream=true`。

**`/openai/v1/images/edits`（Images 透传）**：multipart 的重复 `image`、可选 `mask`、未知文本字段和文件 metadata 均保留，仅替换 model deployment；同步 JSON 与 `stream=true` SSE 均原样返回。网关按 Azure Images API 能力路由，不按具体模型名称限制。

---

## Claude Code 接入任意模型

Claude Code 只会说 Anthropic Messages 协议（它把请求发往 `/v1/messages`）。网关在这个端点上按目标模型**自动分流**：Claude 系模型原样透传；而 **GPT-5.5 / Grok / `azure/<deployment>`** 这类 Responses 模型会被**翻译**（Anthropic Messages ⇄ OpenAI Responses）后转发。于是把 `ANTHROPIC_BASE_URL` 指向网关，就能让 Claude Code（或任何 Anthropic-only 的 agent）用上非 Anthropic 模型——业务代码零改动。

> ⚠️ **最容易踩的坑：`CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX`。**
> 只要这些云厂商直连开关**存在且非空**，Claude Code 就会直连该云、**完全忽略
> `ANTHROPIC_BASE_URL`**，请求根本不经过网关（表现为它自己报
> `400 The provided model identifier is invalid`——因为 `gpt-5.5` 不是合法的
> Bedrock 模型名）。而且它对**任何非空值都视为开启**，设 `"0"` 也没用——
> **必须整个删掉这个变量**。接入网关前先确认：
> ```bash
> env | grep -E 'CLAUDE_CODE_USE_(BEDROCK|VERTEX)|ANTHROPIC_(BASE_URL|MODEL)'
> unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX   # 若存在，必须清掉
> ```

**方式 A —— 环境变量（临时试用）**

```bash
# 让 Claude Code 用 GPT-5.5（经 Bedrock mantle）
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX    # 关键：清掉云直连开关
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_AUTH_TOKEN="<gateway-key>"     # 网关的 server.api_key（→ Bearer）
export ANTHROPIC_MODEL="gpt-5.5"                # 也可 grok-4.3
claude

# 或用 Azure 上的 GPT-5.5（前缀透传，需资源配 prefix: azure）
export ANTHROPIC_MODEL="azure/gpt-5.5"
```

`ANTHROPIC_AUTH_TOKEN` 走 `Authorization: Bearer`，`ANTHROPIC_API_KEY` 走
`x-api-key`——网关两者都收，任配其一即可。未设 `server.api_key` 时随便填一个。

**方式 B —— settings.json（推荐，持久且不受环境残留影响）**

写入 `~/.claude/settings.json`（或某项目的 `.claude/settings.json`）。**关键：不要
在 `env` 里写 `CLAUDE_CODE_USE_BEDROCK`**——它一旦出现就压过 `ANTHROPIC_BASE_URL`：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "ANTHROPIC_AUTH_TOKEN": "<gateway-key>",
    "ANTHROPIC_MODEL": "gpt-5.5"
  },
  "model": "gpt-5.5"
}
```

**先验证网关这条链路通不通（不依赖 Claude Code）**：直接 curl `/v1/messages`，
成功即说明翻译链路 OK，接下来只是把 Claude Code 指过来的问题：

```bash
curl -s http://127.0.0.1:4000/v1/messages \
  -H "x-api-key: <gateway-key>" -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.5","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
# 返回 {"type":"message","role":"assistant","content":[{"type":"text",...}]} 即通
```

工作机制：Claude Code POST `/v1/messages` → 网关识别 `gpt-5.5` 是 Responses 模型 → 把请求（system / messages / tools / tool_use / tool_result / 图片）翻译成 OpenAI Responses → 经对应 transport 转发（Bedrock mantle 或 Azure api-key）→ 把响应与 SSE 流翻译回 Anthropic 事件（`message_start` / `content_block_*` / `message_delta` / `message_stop`，工具调用用 `input_json_delta` 拼装）。工具调用（Claude Code 的 Read/Edit/Bash 等）完整往返。

**限制（当前版本）**：

- 仅翻译到 **Responses** 目标（GPT-5.5 / Grok / Azure Responses 部署）。Anthropic 端点上的 openai-chat 模型仍返回 400。
- `thinking` / reasoning 块**不回传**（无签名往返）；请求中的 `cache_control` 被剥离，但 Responses 返回的 `cached_tokens` 会拆分成 Anthropic `input_tokens + cache_read_input_tokens`，保证上下文总量不丢失。
- 翻译流只有收到 Responses completed/incomplete 才正常 `message_stop`；failed/error/异常 EOF 以 Anthropic error 终止，context overflow 保留原始 message 并规范化为 `invalid_request_error`。
- 未显式传 `max_tokens` 时使用模型 registry 的 `max_output`；响应的 `model` 保留客户端请求 alias，不暴露上游 ID/deployment。
- `count_tokens` 当前使用字符估算做兼容兜底，属于 approximate：本机探测到 Mantle Responses 不提供该操作、Azure 取决于 deployment、Claude Mantle 需要额外 `bedrock-mantle:CountTokens` 权限。估算结果不用于本地 context 硬限制。
- SigV4 鉴权模式（`credentials`/`iam_role`/`profile`）签不了 mantle/Azure；用 `bearer_token`（mantle）或资源 `api_key`（Azure）。

Claude 模型本身（`ANTHROPIC_MODEL=claude-opus-4.8` 等）在此端点走原生透传路径，行为与直连 Anthropic 一致。

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
logging:
  file_enabled: true
  directory: /var/log/bedrock-gateway
  retention_days: 30
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
install -d -o bedrock -g bedrock -m 0750 /var/log/bedrock-gateway
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
| 网关代码意外异常 | ERROR + traceback | 完整栈帧写入文件与 journald |

文件日志默认开启；仅显式设置 `logging.file_enabled: false` 时关闭。网关业务日志、Uvicorn 启动/错误/访问日志和 httpx 上游请求日志会同时写入 `/var/log/bedrock-gateway/bedrock-gateway-YYYY-MM-DD.log`。文件按主机本地午夜切分并自动保留最近 30 天；console 输出仍保留，因此 journald 不受影响：

```bash
tail -f /var/log/bedrock-gateway/bedrock-gateway-$(date +%F).log
journalctl -u bedrock-gateway -f
journalctl -u bedrock-gateway -p err
```

不要再对这些文件配置外部 `logrotate` rename 轮转，以免和进程内按日切分冲突。当前服务为单进程；若未来启用多 worker，应改用集中日志收集方案。

---

## 安全 Checklist

生产部署前逐项确认：

- [ ] `BEDROCK_API_KEY` 设为强随机值（`bgw-$(openssl rand -base64 48)`）
- [ ] `BEDROCK_DASHBOARD_KEY` 设为强随机值，且与 `BEDROCK_API_KEY` 不同
- [ ] 所有云凭据（`AWS_BEARER_TOKEN_BEDROCK`、`AZURE_OPENAI_KEY` 等）仅写入 `.env`（权限 `600`），`config.yaml` 里只用 `${VAR}` 插值引用，不写明文
- [ ] 绑定 `0.0.0.0` 时前面**必须**有 TLS 终止（Nginx / ALB / Cloudflare）
- [ ] 设 `dashboard.api_key`（或 `dashboard.enabled: false`），并保持 `require_auth: true`
- [ ] 非 root 运行（systemd `User=` 或 Docker 非 root 用户）
- [ ] 文件日志目录属于 `bedrock:bedrock` 且权限为 `0750`；日志同时接入集中收集（journalctl、容器日志驱动）
- [ ] Bedrock IAM 账号最小权限，仅保留必要的 `bedrock:InvokeModel*`；Azure api-key 定期轮换（资源有 key1/key2 可无缝切换）

---

## 开发

```bash
git clone https://github.com/wujiaming88/bedrock-gateway.git
cd bedrock-gateway
pip install -e ".[dev]"

pytest -q
ruff check bedrock_gateway/ tests/
mypy bedrock_gateway/ --ignore-missing-imports
```

**架构**：模型由正交的 `Transport × Dialect` 组合描述。新增云或认证方式只增加 Transport；新增 wire format 只增加 Dialect；同格式的新模型通常只需增加 `ModelEntry`。重试、预检、超时和 metrics 继续由 `server.py` 统一编排。设计细节见 [`docs/multi-cloud-multimodal-design.md`](docs/multi-cloud-multimodal-design.md)。

---

## License

MIT —— 见 [LICENSE](LICENSE)。
