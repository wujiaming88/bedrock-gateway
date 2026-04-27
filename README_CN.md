# Bedrock Gateway

**轻量级 OpenAI 兼容代理，让任何 OpenAI / Anthropic 客户端无缝访问 AWS Bedrock。**

无需修改 SDK，无需厂商锁定，只需设置 `OPENAI_BASE_URL` 或 `ANTHROPIC_BASE_URL` 即可开始使用。

```
┌────────────────┐     OpenAI API     ┌──────────────────┐    Bedrock API    ┌─────────────┐
│  OpenAI 客户端  │ ──────────────────▶ │ Bedrock Gateway  │ ────────────────▶ │ AWS Bedrock │
│  (任意 SDK)     │ ◀────────────────── │   (本项目)        │ ◀──────────────── │   Claude    │
└────────────────┘                     └──────────────────┘                    └─────────────┘

┌────────────────┐   Anthropic API    ┌──────────────────┐    Bedrock API    ┌─────────────┐
│ Anthropic 客户端│ ──────────────────▶ │ Bedrock Gateway  │ ────────────────▶ │ AWS Bedrock │
│ (Claude Code等) │ ◀────────────────── │   (本项目)        │ ◀──────────────── │   Claude    │
└────────────────┘                     └──────────────────┘                    └─────────────┘
```

## 特性

- 🔌 **即插即用** — 完全兼容 OpenAI `/v1/chat/completions` 和 `/v1/models` 接口
- 🔮 **Anthropic Messages API** — 原生 `/v1/messages` 端点（同步 + 流式）
- 🔐 **多种认证** — Bearer Token、AK/SK (SigV4)、IAM Role、AWS Profile
- 🔒 **API Key 鉴权** — 可选的网关鉴权，防止未授权访问
- 🔄 **完整协议转换** — 消息、工具调用、图片、流式、思考模式全支持
- 🏗️ **生产就绪** — 自动重试退避、结构化日志、健康检查
- 📦 **零配置启动** — 仅需环境变量即可运行，或使用 YAML 精细控制
- 🐳 **容器优先** — 单容器部署，镜像仅 50MB

## 快速开始

### 方式一：pip 安装

```bash
pip install bedrock-gateway
```

```bash
export AWS_BEARER_TOKEN_BEDROCK="你的令牌"
bedrock-gateway
# → 监听 http://127.0.0.1:4000
```

### 方式二：Docker

```bash
docker run -p 4000:4000 \
  -e AWS_BEARER_TOKEN_BEDROCK="你的令牌" \
  bedrock-gateway
```

### 方式三：源码运行

```bash
git clone https://github.com/bedrock-gateway/bedrock-gateway.git
cd bedrock-gateway
pip install -e .
python -m bedrock_gateway
```

### 使用示例

**OpenAI SDK：**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4000/v1",
    api_key="任意值",  # SDK 要求但不使用
)

response = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "你好！"}],
)
print(response.choices[0].message.content)
```

**Anthropic SDK：**

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:4000",
    api_key="任意值",  # SDK 要求但不使用
)

message = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好！"}],
)
print(message.content[0].text)
```

**curl：**

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

## 认证方式

| 模式 | 配置 | 说明 |
|------|------|------|
| `bearer_token` | `AWS_BEARER_TOKEN_BEDROCK` 环境变量 | AWS Bearer Token (ABSK)，最简单的方式 |
| `credentials` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | 标准 AK/SK，SigV4 签名 |
| `iam_role` | （自动获取） | 从 EC2/ECS/Lambda 元数据自动获取，需要 `boto3` |
| `profile` | `AWS_PROFILE` 或配置文件 | AWS CLI 命名 Profile，需要 `boto3` |

使用 `iam_role` 或 `profile` 模式时，安装 boto3 依赖：

```bash
pip install bedrock-gateway[boto3]
```

## 支持的模型

| 别名 | Bedrock 模型 ID | 上下文 | 最大输出 |
|------|-----------------|--------|----------|
| `claude-opus-4.7` | `us.anthropic.claude-opus-4-7` | 1M | 128K |
| `claude-opus-4` | `us.anthropic.claude-opus-4-6-v1` | 1M | 128K |
| `claude-sonnet-4.6` | `us.anthropic.claude-sonnet-4-6` | 1M | 64K |
| `claude-sonnet-4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | 200K | 64K |
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | 64K |
| `claude-sonnet-3.5` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | 64K |

也可以直接传入原始的 Bedrock 模型 ID。在 `config.yaml` 中添加自定义模型：

```yaml
models:
  my-custom-model:
    bedrock_id: us.my-org.my-model-v1
    context_length: 100000
    max_output: 8192
```

## 配置

### 环境变量（零配置）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BEDROCK_API_KEY` | — | 网关 API Key 鉴权（可选，设置后启用鉴权） |
| `AWS_BEARER_TOKEN_BEDROCK` | — | AWS Bedrock 认证令牌 |
| `AWS_REGION` | `us-east-1` | AWS 区域 |
| `BEDROCK_HOST` | `127.0.0.1` | 服务绑定地址 |
| `BEDROCK_PORT` | `4000` | 服务端口 |
| `BEDROCK_LOG_LEVEL` | `info` | 日志级别 |
| `BEDROCK_AUTH_MODE` | `bearer_token` | 认证模式 |
| `BEDROCK_MAX_RETRIES` | `3` | 最大重试次数 |

### YAML 配置文件

复制 `config.example.yaml` 为 `config.yaml` 进行精细配置：

```yaml
auth:
  mode: bearer_token
  bearer_token: ${AWS_BEARER_TOKEN_BEDROCK}  # 支持环境变量引用

region: us-east-1

server:
  host: 0.0.0.0
  port: 4000
  log_level: info
  api_key: ${BEDROCK_API_KEY}  # 可选：设置后需要鉴权才能访问 API

retry:
  max_retries: 3
  base_delay: 1.0

models:
  claude-sonnet-4:
    bedrock_id: us.anthropic.claude-sonnet-4-20250514-v1:0
    context_length: 200000
    max_output: 64000
```

## API 参考

### POST /v1/chat/completions

OpenAI 兼容的聊天补全接口，支持：

- ✅ 同步和流式响应
- ✅ System 消息
- ✅ 多轮对话
- ✅ 工具调用（Function Calling）
- ✅ 多模态（Base64 和 URL 图片）
- ✅ 扩展思考（`thinking` 参数）
- ✅ 推理力度映射（`reasoning_effort` → `thinking`）
- ✅ 停止序列、温度、top_p 等参数

### POST /v1/messages

Anthropic Messages API 端点，接受原生 Anthropic 格式：

- ✅ 同步和流式响应
- ✅ System 提示词（字符串或块数组）
- ✅ 多轮对话
- ✅ 工具调用（原生 Anthropic 格式）
- ✅ 扩展思考（`thinking` 参数，流式 `thinking_delta`）
- ✅ 思考块（`thinking`、`redacted_thinking`、`signature_delta`）
- ✅ 停止序列、温度、top_p、top_k
- ✅ 缓存 Token 统计
- ✅ 模型别名解析

```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

### GET /v1/models

返回可用模型列表（OpenAI 格式）。

### GET /health

健康检查端点。

## 安全：API Key 鉴权

默认情况下，网关接受所有请求（适用于本地开发）。当网关暴露在网络上时，建议启用鉴权：

```bash
export BEDROCK_API_KEY="你的强密码"
bedrock-gateway
```

或在 `config.yaml` 中配置：

```yaml
server:
  api_key: ${BEDROCK_API_KEY}
```

启用后，所有 API 端点都需要鉴权（`/health` 除外）。支持两种 Header 格式：

```bash
# Authorization: Bearer 方式
curl -H "Authorization: Bearer 你的密钥" http://localhost:4000/v1/models

# x-api-key 方式
curl -H "x-api-key: 你的密钥" http://localhost:4000/v1/models
```

**安全措施：**

| 措施 | 说明 |
|------|------|
| `hmac.compare_digest` | 恒定时间比较，防止时序攻击 |
| `/health` 白名单 | 监控探针无需鉴权 |
| Bearer + x-api-key 双支持 | 兼容 OpenAI SDK 和 Anthropic SDK |
| 按需启用 | 不配置 Key 则不需要鉴权 |

**配合 Claude Code 使用：**

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://你的VPS:4000",
    "ANTHROPIC_AUTH_TOKEN": "你设置的同一个密钥",
    "ANTHROPIC_MODEL": "claude-opus-4.7"
  }
}
```

Claude Code 会自动将 `ANTHROPIC_AUTH_TOKEN` 作为 `Authorization: Bearer <key>` 发送。

## Bedrock Gateway vs. LiteLLM

| | Bedrock Gateway | LiteLLM |
|---|---|---|
| **定位** | 专注 AWS Bedrock | 100+ 提供商 |
| **依赖** | 4 个 | 50+ |
| **镜像大小** | ~50MB | ~500MB |
| **认证方式** | Bearer Token、AK/SK、IAM、Profile | AK/SK、IAM |
| **API 兼容** | OpenAI + Anthropic 双协议 | OpenAI |
| **启动时间** | 30 秒 | 分钟级 |
| **适用场景** | 专用 Bedrock 的团队 | 多供应商路由 |

如果你只用 Bedrock，选择 Bedrock Gateway 获得最小开销。
如果需要在多个 LLM 提供商之间路由，选择 LiteLLM。

## 开发

```bash
git clone https://github.com/bedrock-gateway/bedrock-gateway.git
cd bedrock-gateway
pip install -e ".[dev]"

# 运行测试
pytest -v

# 代码检查
ruff check bedrock_gateway/ tests/

# 类型检查
mypy bedrock_gateway/ --ignore-missing-imports
```

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
