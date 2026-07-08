# 设计文档：多云 + 全模态架构（Transport / Dialect 两轴）

状态：待评审 | 作者：小强 | 日期：2026-07-08 | 目标版本：0.3.0

---

## 1. 背景与动机

### 1.1 目标演进
网关定位从"AWS Bedrock 文本对话代理"扩展为 **多云、全模态 LLM 统一网关**：
- **多云**：Bedrock（现有）+ Azure OpenAI（本期）+ 未来 OpenAI 直连 / Google 等。
- **全模态**：对话（chat / responses）+ embedding（本期）+ 图像 / 语音（后续）。

### 1.2 现有架构的天花板
当前 `Provider` 抽象（0.2.x）把**两件正交的事捆在一个类里**：

| 关注点 | 现有 Provider 承担的 | 例子 |
|---|---|---|
| **传输**：在哪、怎么鉴权、base URL | ✅ | bedrock-runtime + SigV4/Bearer；bedrock-mantle |
| **方言**：请求/响应/流的形状 | ✅ | Anthropic Messages 转换；OpenAI Responses 透传 |

捆绑的后果是**组合爆炸**：每个（云 × 方言）都要一个新 Provider。

```
现有：AnthropicBedrock、OpenAIResponses(Bedrock mantle)
接 Azure 全模态若沿用旧模式，需要新增：
  AzureResponses, AzureChat, AzureEmbeddings, AzureImages …
再接 OpenAI 直连：OpenAIChat, OpenAIResponses, OpenAIEmbeddings …
```

**N 个云 × M 个方言 = N×M 个类**。这正是本次要避免的补丁式增长。

### 1.3 关键洞察（已实测）
Azure 的各模态端点与标准 OpenAI **高度兼容**，差异只有三处，且都属于"传输层"：
1. **URL 形态**：`{resource}.cognitiveservices.azure.com/openai/...`，per-资源 endpoint。
2. **鉴权**：`api-key: <key>` header（Bedrock 用 `Authorization: Bearer`）。
3. 响应多 `content_filter_results` 字段（透传无害，方言层不关心）。

实测确认（用真实 Azure 资源）：
- `gpt-5` 走 `{base}/openai/responses?api-version=2025-04-01-preview` → 200，Responses 格式。
- `gpt-5` 走 `{base}/openai/v1/chat/completions` → 200，标准 OpenAI chat 格式。
- 两条路径 URL 形态不同 → **endpoint 必须完全可配，不能硬编码形态**。

**结论**：Azure 与 Bedrock mantle 的"方言"其实**一模一样**（都是 OpenAI Responses / Chat 透传），差的只是"传输"。既然方言可复用，就该把两轴拆开。

---

## 2. 核心设计：Transport × Dialect 两轴

把 `Provider` 拆成两个正交的抽象，一个模型 = 一对 `(Transport, Dialect)` 组合：

```
        Transport（传输层）                    Dialect（方言层）
     "在哪调、怎么鉴权、URL"                "请求/响应/流长什么形状"
   ┌────────────────────────┐          ┌──────────────────────────────┐
   │ BedrockTransport        │          │ AnthropicMessagesDialect      │
   │  · host: runtime/mantle │          │  (OpenAI chat → Anthropic 转换)│
   │  · SigV4 / Bearer       │          │ ResponsesPassthroughDialect   │
   │                         │    ×     │  (OpenAI Responses 透传)       │
   │ AzureTransport          │          │ ChatPassthroughDialect        │
   │  · per-资源 endpoint     │          │  (OpenAI Chat 透传)            │
   │  · api-key header        │          │ EmbeddingsPassthroughDialect  │
   │                         │          │  (Embeddings 透传)             │
   └────────────────────────┘          └──────────────────────────────┘
            │                                        │
            └────────────  一个模型  ────────────────┘
              ModelEntry { transport, dialect, ... }
```

### 2.1 矩阵塌缩：N×M → N+M
| 要接的东西 | 旧模式（补丁） | 新模式（两轴） |
|---|---|---|
| Azure Responses | 新写 AzureResponses | `AzureTransport` + **复用** ResponsesDialect |
| Azure Chat | 新写 AzureChat | `AzureTransport` + **复用** ChatDialect |
| Azure Embeddings | 新写 AzureEmbeddings | `AzureTransport` + 新 EmbeddingsDialect |
| OpenAI 直连（未来） | 新写 3 个 | 新 `OpenAITransport` + **复用 3 个方言** |

**加一个云 = 加 1 个 Transport；加一个模态 = 加 1 个 Dialect。** 加法，不是乘法。

### 2.2 职责边界

**Transport** —— 与"内容"无关，只管"怎么把字节送到正确的上游并通过鉴权"：
- `build_url(operation, entry, region, stream) -> str`：给定操作（`responses`/`chat/completions`/`embeddings`）拼出完整 URL。Bedrock 用 region + bedrock_id；Azure 用 `entry.azure_endpoint` + operation。
- `auth_headers(entry) -> dict | None`：返回鉴权 header。Bedrock 返回 `None`（用全局 `AuthProvider` 的 SigV4/Bearer）；Azure 返回 `{"api-key": ...}`。

**Dialect** —— 与"在哪"无关，只管"请求/响应/流的形状"：
- `build_request(body, entry) -> dict`：把客户端入参塑成上游请求体（Anthropic 方言做转换；透传方言仅替换 model/deployment）。
- `render_sync(upstream_json, model) -> (client_body, log_info)`。
- `transform_stream(byte_iter, model, msg_id) -> AsyncIterator[str]`：流式塑形（透传方言字节直通；Anthropic 方言解码 event-stream 重编 SSE）。
- `stream_error(message, status) -> str`。
- `supports_stream: bool`：embedding 方言为 False。

> **正交性验证**：`AzureTransport` 不知道自己在传 Responses 还是 Embeddings；`ResponsesPassthroughDialect` 不知道自己在 Bedrock mantle 还是 Azure。两者只通过 `operation` 字符串和 `ModelEntry` 数据耦合。这就是"正交"。

### 2.3 server 编排层不变
重试、退避、超时、metrics、`_open_upstream_stream` 预检、错误分级日志 —— **全部保留、所有组合共享**。编排层从"调 provider"改为：
```
transport = get_transport(entry)      # 选云
dialect   = get_dialect(entry)        # 选方言
url  = transport.build_url(dialect.operation, entry, region, stream)
hdrs = transport.auth_headers(entry) or global_auth.get_headers(...)
body = dialect.build_request(client_body, entry)
# ... 共享的重试/预检/流式编排 ...
resp = dialect.render_sync(...) / dialect.transform_stream(...)
```

---

## 3. audio：唯一不进"JSON 方言"的模态

chat / responses / embeddings / images 都是 **JSON 进、JSON 出、可选 SSE 文本流**，共用一套 Dialect 接口即可。但 **audio 不是**：
- **请求**：TTS 是 JSON，但 STT（transcriptions/translations）是 **multipart/form-data 文件上传**。
- **响应**：TTS 返回 **二进制音频流**（非 SSE、非 JSON）。

**决策**：不把 audio 硬塞进 JSON Dialect 接口（会用 `Any`/可选字段污染抽象）。为它**显式预留一个独立的 `BinaryDialect` / `MultipartDialect` 缝**，等真正做 audio 时再实现。本期不做，但架构上承认它的存在，不假装它和 chat 一样。

> images 的响应是 JSON（含 URL/base64），可归入 JSON Dialect；仅 audio 需要二进制缝。

---

## 4. 配置模型（已在 0.2.x 落地资源层，本期加 transport/dialect）

### 4.1 两层 Azure 配置（已实现）
```yaml
# 资源级：endpoint + key（一个 Azure 资源配一次）
azure_resources:
  wangmeng-1016:
    base_url: https://wangmeng-1016-resource.cognitiveservices.azure.com/openai
    api_key: ${AZURE_OPENAI_KEY}

# 模型级：引用资源 + deployment + transport/dialect
models:
  azure-gpt-5:
    transport: azure                 # 新轴（默认 bedrock）
    dialect: responses               # 新轴（默认 anthropic）
    azure_resource: wangmeng-1016
    deployment: gpt-5                # → 请求体 model 字段
  azure-gpt-5-chat:
    transport: azure
    dialect: chat
    azure_resource: wangmeng-1016
    deployment: gpt-5
```

### 4.2 向后兼容（关键约束）
现有 `ModelEntry` 有 `endpoint` / `protocol` 两个旧字段，7+2 个 Bedrock 模型和存量 YAML 都在用。**不能破坏**。方案：
- 新增 `transport` / `dialect` 字段，默认 `bedrock` / `anthropic`。
- **保留 `endpoint`/`protocol` 作为兼容映射**：加载时把旧值翻译成新轴：
  - `protocol=anthropic` → `transport=bedrock, dialect=anthropic`
  - `protocol=openai-responses`（GPT-5.5/Grok）→ `transport=bedrock(mantle), dialect=responses`
- 旧配置零改动继续工作；测试断言这个映射。

---

## 5. 迁移步骤（分阶段，每步测试全绿）

| 阶段 | 内容 | 验证 |
|---|---|---|
| **M0** | 本文档评审通过（当前） | 用户确认 |
| **M1** | 定义 `Transport` / `Dialect` 接口；把现有 2 个 Provider 拆成 `BedrockTransport` + `AnthropicMessagesDialect` / `ResponsesPassthroughDialect`；server 编排改为 transport×dialect；旧 `protocol` 映射到新轴 | 643 测试零回归（现有 Bedrock/GPT-5.5/Grok 全绿） |
| **M2** | `AzureTransport`（api-key + per-资源 endpoint）；`azure-gpt-5` 走 Responses 方言 | 单测 + 真机冒烟（实测已通） |
| **M3** | `ChatPassthroughDialect`；Azure + Bedrock 都能走 `/v1/chat/completions` 透传（与现有 Anthropic 转换版分流共存） | 单测 + 真机（gpt-5 chat 实测已通） |
| **M4** | `EmbeddingsPassthroughDialect` + `/v1/embeddings` 端点（无流式） | 单测 + 真机 |
| **M5** | 文档、config.example、CHANGELOG(0.3.0)、README | 一致 |
| **后续** | 图像（JSON 方言，较易）、audio（二进制缝，单独设计） | 独立立项 |

**M1 是纯重构、零行为变化**——先证明两轴骨架能无损承载现有全部功能，再往上加 Azure 和新模态。这是整个演进的安全阀。

---

## 6. 风险与权衡

| 风险 | 影响 | 缓解 |
|---|---|---|
| **M1 大重构引入回归** | 现有 9 个模型受影响 | M1 不加任何新功能，只搬运；643 测试逐一验证；分 transport/dialect 两个小 commit |
| **旧 protocol 字段兼容遗漏** | 存量配置/生产失效 | 保留旧字段 + 映射层 + 专门的兼容测试；生产（0.2.1）不动直到 M5 验证 |
| **过度设计** | 两轴对只有 Bedrock 时是负担 | 但目标已明确是多云全模态，N×M 爆炸是真实的；两轴是必要而非投机 |
| **audio 抽象未定** | 未来做 audio 可能要再调整 | 本期显式不碰，只留缝；不为未定的 audio 预先设计错的接口 |
| **Azure endpoint 形态多样** | responses 带 api-version、chat 用 v1，形态不一 | `base_url` 配到 `/openai`，各 dialect 拼自己的 operation 路径 + 保留已有 query |
| **content_filter 字段** | 客户端可能不认 | 透传（用户已定），不改写响应体 |

---

## 7. 决策记录

- **D1**：拆 Transport × Dialect 两轴 —— 避免 N×M 组合爆炸，加云/加模态变加法。用户 2026-07-08 要求"架构长远思考"。
- **D2**：M1 先做纯重构（现有功能无损搬进两轴）再加新功能 —— 安全阀。
- **D3**：保留 `endpoint`/`protocol` 旧字段做兼容映射 —— 生产在跑 0.2.1，不破坏。
- **D4**：audio 不进 JSON 方言，留独立二进制缝，本期不实现 —— 不为未定接口预先设计。
- **D5**：本期范围 = Azure 对话（responses + chat）+ embedding；图像/语音后续 —— 用户 2026-07-08 定。
- **D6**：Azure endpoint 完全可配（base_url + 各 dialect 拼 operation）—— 实测两条路径形态不同，不能硬编码。
- **D7**：content_filter_results 原样透传 —— 用户 2026-07-08 定。
