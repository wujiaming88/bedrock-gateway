# 设计文档：Provider 抽象与 GPT-5.5（mantle / OpenAI Responses）接入

状态：待评审 | 作者：小强 | 日期：2026-07-08 | 目标版本：0.2.0

---

## 1. 背景与动机

### 1.1 触发需求
接入 AWS Bedrock 上的 **OpenAI GPT-5.5**（`openai.gpt-5.5`）。

已实测确认（用运行时 bearer token 直连）：
- 端点 `https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses`，`openai.gpt-5.5` 返回 **HTTP 200**，正常推理。
- 无订阅阻塞（与当年 Fable5 卡 403 不同），**可端到端实测联调**。
- 返回体是 **OpenAI Responses API** 结构（`output[].content[].output_text`、`usage.input_tokens/output_tokens`、`reasoning.effort`），标准 SSE 流式——与网关现有的 Anthropic 二进制 event-stream 完全不同。

### 1.2 为什么不能"加一条配置就完事"
现网关本质是**单形态代理**：`bedrock-runtime` 端点 + Anthropic Messages 协议，两处硬编码：

| 维度 | 现状（硬编码位置） | GPT-5.5 需要 |
|---|---|---|
| 端点 base | `server.py:152` 写死 `bedrock-runtime.{region}.amazonaws.com` | `bedrock-mantle.{region}.api.aws` |
| 请求路径 | `/model/{id}/invoke` `/invoke-with-response-stream` | `/openai/v1/responses` |
| 请求格式 | Anthropic（`anthropic_version`、messages、thinking 块） | OpenAI Responses（`input`、`reasoning`） |
| 响应解析 | `parse_bedrock_response`（Anthropic content 块） | Responses（`output[]`） |
| 流式解码 | `decode_event_stream_chunk`（AWS 二进制 event-stream） | 标准 SSE |
| ModelEntry | 只有 `bedrock_id/context_length/max_output` | 缺"端点/协议"维度 |

若走"最小补丁"（在 `server.py` 加 `if protocol==...` 分支 + 复制一套 handler），会让 server 里并行 handler 从 4 个膨胀到 6+ 个，协议差异散落进本应协议无关的编排逻辑（重试/超时/metrics/流式预检）。这与已确认的 **ModelEntry 单一事实来源重构** 的目标（消除散落硬编码）正相反。

### 1.3 目标与非目标

**目标**
- G1：网关**新增 `/openai/v1/responses` 端点**，客户端直发 OpenAI Responses 原生格式，网关透传给 mantle 调 GPT-5.5（同步 + 流式）。与 OpenAI SDK 约定一致：客户端设 `OPENAI_BASE_URL=http://<gateway>/openai/v1` 即可直用。
- G2：引入 **Provider 抽象**，把"如何与某个上游对话"与"请求编排"解耦。
- G3：现有 6 个 Claude 模型行为 **零回归**（590 个通过的测试全绿）。
- G4：日后新增模型/厂商 = 加一个 Provider + 一条 ModelEntry，`server.py` 不改。
- G5：GPT-5.5 支持**图片输入**（文本 + 图片多模态）—— Responses 原生格式，透传天然对齐，零转换损耗。

**非目标（本次不做）**
- N1：`/v1/messages`（Anthropic 端点）暴露 GPT-5.5。
- N2：`/v1/chat/completions` 暴露 GPT-5.5 —— 见 D2，主链路走 `/openai/v1/responses` 透传。
- N3：thinking/采样护栏那一层的重构（`_ADAPTIVE_THINKING_PATTERNS` 子串匹配）—— 仍是独立待办，本次不顺手改，控制爆炸半径。
- N4：credentials(SigV4) 模式对 mantle 的签名适配 —— 运行时用 bearer，SigV4 对 mantle 域名的 service/host 差异留待需要时再处理（见 §7 风险）。
- N5：图片**输出**、音频、视频 —— 模型卡明确不支持，无需处理。

---

## 2. 架构总览

```
  POST /v1/chat/completions ─┐   POST /openai/v1/responses ─┐
  (OpenAI chat, 现有)        │   (Responses 原生, 新增)       │
                             ▼                                ▼
                        ┌─────────────────────────────────────┐
                        │              server.py               │
                        │   (协议无关的编排层)                  │
                        │  · 解析入参  · 重试/退避  · 超时       │
                        │  · metrics   · 流式预检(_open_stream) │
                        └───────────────┬─────────────────────┘
                                        │ registry.get_entry(alias)
                                        ▼
                              ModelEntry { endpoint, protocol }
                                        │  选择 provider
                        ┌───────────────┴───────────────┐
                        ▼                                ▼
        ┌───────────────────────────┐   ┌────────────────────────────┐
        │ AnthropicBedrockProvider   │   │ OpenAIResponsesProvider      │
        │ (runtime, 现有逻辑封装)     │   │ (mantle, 新增)               │
        ├───────────────────────────┤   ├────────────────────────────┤
        │ sync_url/stream_url        │   │ sync_url/stream_url          │
        │  → bedrock-runtime...      │   │  → bedrock-mantle...api.aws  │
        │  → /model/{id}/invoke...   │   │  → /openai/v1/responses      │
        │ build_request(body)        │   │ build_request(body)          │
        │  → Anthropic messages      │   │  → 恒等透传(仅补 model)      │
        │ parse_response(raw)        │   │ parse_response(raw)          │
        │  → 复用 converter 现有函数  │   │  → 原样透传 Responses body   │
        │ iter_stream_events(bytes)  │   │ iter_stream_events(sse)      │
        │  → AWS 二进制 event-stream │   │  → 标准 SSE 透传             │
        └───────────────────────────┘   └────────────────────────────┘
                        │                                │
                        ▼                                ▼
              bedrock-runtime API              bedrock-mantle API
              (Anthropic 格式)                 (OpenAI Responses 格式)
```

核心思想：**server 编排一次写好，协议差异下沉到 Provider**。auth 层不动（bearer token 对两个域名都有效，已实测）。

> **两条端点、两种 provider 职责不同**：
> - `/v1/chat/completions`（现有）→ anthropic provider → runtime（Claude 全家）。**不接 GPT-5.5**。
> - `/openai/v1/responses`（新增）→ openai-responses provider → mantle（GPT-5.5）。透传，几乎无格式转换。
>
> 透传路径下 provider 的 `build_request`/`parse_response` 近乎**恒等映射**：请求仅补 `model` 字段后原样转发，响应/流式原样回吐。真正的价值是复用 server 的编排层（重试/超时/metrics/流式预检），而非格式转换。

---

## 3. 接口设计

### 3.1 Provider 协议（`bedrock_gateway/providers/base.py`）

```python
from typing import Protocol, Any, Iterator

class StreamEvent:
    """协议无关的流式事件（归一化后交给 server 生成 OpenAI SSE）。"""
    # 类型之一：role_start | text_delta | tool_call_start | tool_args_delta
    #          | finish | usage | error
    kind: str
    data: dict[str, Any]

class Provider(Protocol):
    """把'如何与某个上游对话'抽象成统一接口。

    server 只依赖本协议，不感知 runtime/mantle、Anthropic/OpenAI 的差异。
    """
    name: str  # "anthropic-bedrock" | "openai-responses"

    def base_url(self, region: str) -> str:
        """上游 base，如 https://bedrock-runtime.{region}.amazonaws.com。"""

    def sync_url(self, region: str, bedrock_id: str) -> str:
        """同步调用完整 URL。"""

    def stream_url(self, region: str, bedrock_id: str) -> str:
        """流式调用完整 URL。"""

    def build_request(
        self, *, bedrock_id: str, body: dict, max_output: int,
        stream: bool,
    ) -> dict:
        """把客户端入参构造成上游请求体。
        - anthropic provider：OpenAI chat → Anthropic messages（现有转换）。
        - openai-responses provider：恒等透传，仅把 model 换成 bedrock_id
          （原生 Responses body 不做字段清洗，见 §3.4）。"""

    def parse_response(self, raw: dict) -> Response:
        """把上游响应转成对外响应。
        - anthropic provider：Bedrock(Anthropic) → OpenAI chat.completion。
        - openai-responses provider：原样透传 Responses body（含 output/usage/
          reasoning），仅按需补齐 id 等。"""

    def iter_stream_bytes(
        self, upstream: bytes,
    ) -> bytes:
        """流式转发。
        - anthropic provider：AWS 二进制 event-stream → OpenAI SSE chunk。
        - openai-responses provider：标准 SSE 原样透传（可零解码直通）。"""
```

> **两种 provider 的流式模型不同**：anthropic provider 要把 AWS 二进制 event-stream 解码再重编成 OpenAI SSE；openai-responses provider 上下游都是标准 SSE，**可字节级透传**（甚至无需解析每个 event，除非要统计 usage/记 metrics）。因此接口用"字节转发"抽象而非强制归一化事件，避免为透传路径徒增解析开销。

### 3.2 ModelEntry 扩展（向后兼容）

```python
@dataclass
class ModelEntry:
    bedrock_id: str
    context_length: int = 200_000
    max_output: int = 64_000
    endpoint: str = "runtime"     # "runtime" | "mantle"      —— 新增，默认不变
    protocol: str = "anthropic"   # "anthropic" | "openai-responses" —— 新增，默认不变
```

- **向后兼容**：现有 6 个模型与旧 YAML（扁平三字段）不写新字段 → 默认 `runtime/anthropic` → 走 `AnthropicBedrockProvider` → 行为完全不变。
- `_parse_models` 增读 `endpoint`/`protocol`，缺省用默认值。

### 3.3 Provider 选择（`providers/__init__.py`）

```python
_REGISTRY = {
    "anthropic": AnthropicBedrockProvider(),   # 无状态，单例即可
    "openai-responses": OpenAIResponsesProvider(),
}
def get_provider(entry: ModelEntry) -> Provider:
    try:
        return _REGISTRY[entry.protocol]
    except KeyError:
        raise UnknownModelError(f"unsupported protocol: {entry.protocol}")
```

Provider 无状态（纯函数式转换），全局单例，无并发问题。

### 3.4 图片输入（透传下天然支持）

**透传路径下网关不做多模态格式转换** —— 客户端直接按 OpenAI Responses 原生格式发图片，网关原样转发给 mantle：
```json
{"model":"gpt-5.5","input":[
  {"role":"user","content":[
    {"type":"input_text","text":"这张图是什么？"},
    {"type":"input_image","image_url":"data:image/png;base64,iVBOR..."}
  ]}
]}
```

因此图片支持 = **透传不破坏 + 实测验证能通**，无需写任何 image_url→input_image 转换代码（那是 chat/completions 路径才需要的，本次非目标 N2）。

要点：
- **图片形态（P4 实测结论）**：`data:image/...;base64,...`（data URI）**可用**，已实测识别红色 PNG 返回"红色"。**https URL 不可用** —— mantle 明确报 `unsupported image_url scheme: must be `data:` or `s3://``。即图片只支持 **base64 data URI** 或 **s3://** 两种 scheme，不支持公网 http(s) URL。这是上游模型限制，非网关问题；该 400 错误被网关正确透传并记为客户端级日志。
- 网关不解析、不重写 `input` 内容 —— body 透传，图片字段由客户端和 mantle 自行约定，天然对齐、零信息损耗。这正是选透传方案的核心收益（见 D2）。
- 唯一需注意：透传时不能因"清洗未知字段"误删 `input_image`。透传 provider 的 `build_request` 是**恒等映射**（仅补 `model` 字段），不做字段白名单过滤。

---

## 4. server.py 的变化（编排层收敛）

### 4.1 URL/请求/响应 provider 化
现有 `_handle_sync` / `_handle_stream` 里需要 provider 化的三处（把写死的部分委托出去）：
1. `url = f"{bedrock_base}/model/{model}/invoke"` → `provider.sync_url(region, bedrock_id)`
2. 请求体构造 → `provider.build_request(...)`
3. 响应解析 → `provider.parse_response(raw)` / 流式 `provider.iter_stream_bytes(...)`

重试、退避、超时、metrics、`_open_upstream_stream` 预检、错误分级日志 —— **全部原样保留、两个 provider 共享**。`_open_upstream_stream` 尤其宝贵（pre-stream 错误变真实 HTTP 状态码），mantle 路径直接复用。

### 4.2 新增 `/openai/v1/responses` 端点
```python
@app.post("/openai/v1/responses")
async def responses(request: Request):
    body = await request.json()
    raw_model = body.get("model", "gpt-5.5")
    entry = registry.get_entry(raw_model)          # 新增
    provider = get_provider(entry)                 # openai-responses
    # 防御：本端点只服务 responses 协议模型
    if provider.name != "openai-responses":
        return _oai_error(400, f"{raw_model} 不支持 Responses API，请用 /v1/chat/completions")
    stream = body.get("stream", False)
    if stream:
        return await _handle_stream(provider, entry, body, region, auth, ...)
    return await _handle_sync(provider, entry, body, region, auth, ...)
```
透传下 `build_request` 恒等（仅把 `model` 替成 `bedrock_id`）、`parse_response` 原样回吐、流式字节直通——handler 编排复用，改动集中在这一个新端点。

### 4.3 现有端点不变
- `/v1/chat/completions`：仍走 anthropic provider 服务 Claude 全家；**不接 GPT-5.5**（D2）。
- `/v1/messages`：Anthropic 端点，不接 GPT-5.5（N1）。
- 防御：`/v1/chat/completions` 与 `/v1/messages` 若被传入 `protocol != anthropic` 的模型，返回 400「该模型请用 /openai/v1/responses」，避免误用。

### 4.4 认证中间件放行
`server.py:238` 的 api_key 白名单目前放行 `/health`、`/`、`/dashboard`、`/api/metrics`。新端点 `/openai/v1/responses` **需要鉴权**（与 `/v1/chat/completions` 一致），不加入白名单——现有中间件逻辑对任意非白名单路径都会校验 key，无需改动即自动覆盖。

---

## 5. 目录结构

```
bedrock_gateway/
├── providers/                    # 新增包
│   ├── __init__.py               # get_provider() + 单例注册表
│   ├── base.py                   # Provider Protocol + StreamEvent
│   ├── anthropic_bedrock.py      # 现有逻辑封装（薄封装，转调 converter）
│   └── openai_responses.py       # 新增：mantle + Responses 透传（恒等 build/parse）
├── config.py                     # ModelEntry +endpoint +protocol
├── models.py                     # +get_entry()；_looks_like_bedrock_id +"openai."
├── converter.py                  # 不动（透传路径几乎不经 converter）
└── server.py                     # +/openai/v1/responses 端点；handler 参数化 provider；编排逻辑不变
```

`converter.py` 现有函数保持原位，`anthropic_bedrock.py` 只做转调薄封装——**不搬运、不重写现有 Anthropic 逻辑**，把回归风险降到最低。

---

## 6. 迁移步骤（分阶段，每步可独立验证）

| 阶段 | 内容 | 验证 |
|---|---|---|
| **P0** | 修复测试环境：装齐 fastapi 等依赖，配好 pytest-asyncio（现 5 个 async 失败是环境问题，非代码） | 590 → 595 全绿基线 |
| **P1** | `providers/base.py` 接口 + `anthropic_bedrock.py` 薄封装；server handler 参数化 provider；现有模型全部走 anthropic provider | 全套测试零回归 |
| **P2** | ModelEntry +endpoint/protocol；`_parse_models` 兼容旧格式 | test_config 加新旧格式用例 |
| **P3** | `openai_responses.py`：sync_url/stream_url（mantle + `/openai/v1/responses`）、build_request（恒等透传，仅补 model）、parse_response（原样回吐）、流式字节直通；新增 `/openai/v1/responses` 端点；注册 gpt-5.5 别名 | 新增单测（URL 构造、透传恒等性、端点鉴权、协议防御 400） |
| **P4** | e2e：用运行时 bearer token 起网关，经 `/openai/v1/responses` 实调 gpt-5.5（同步 + 流式 + **图片：base64 与 URL 各一张**） | 真实 200 响应，图片被正确识别，token 计数正确，SSE 完整 |
| **P5** | 更新 config.example.yaml、README、CHANGELOG(0.2.0) | 文档一致 |

每阶段独立提交，P1 完成即证明抽象层零回归后再引入新协议。

---

## 7. 风险与权衡

| 风险 | 影响 | 缓解 |
|---|---|---|
| **SigV4 对 mantle 未适配** | credentials 模式调 GPT-5.5 会签错（service/host 不同） | 运行时用 bearer（已实测通）。文档标注：GPT-5.5 目前仅支持 bearer_token 模式；credentials 模式调用返回明确 400。留待有需求再补 SigV4 mantle 适配 |
| **透传流式是否需解析** | 若只字节直通，usage/metrics 拿不到 | 默认字节直通；若 dashboard 要记 GPT-5.5 的 token 数，P3 增一个轻量 SSE 旁路解析只提取末尾 `response.completed` 的 usage，不阻断转发 |
| **抽象层引入回归** | 现有 6 模型受影响 | anthropic provider 只做转调薄封装，不重写；P1 单独验证零回归后才动新协议 |
| **透传误删字段** | 白名单清洗会破坏图片/reasoning | 透传 provider 不做字段过滤，body 原样转发（仅替换 model）；§3.4 已明确 |
| **图片透传实际能否通** | 字段约定错会 400 | 图片由客户端按 Responses 原生格式发，P4 用 base64 与 URL 各一张真图 e2e 验证 |
| **max_output N/A** | 模型卡未给上限 | 透传不主动塞 max_output_tokens，让 mantle 用默认；客户端可自带 |

---

## 8. 决策记录

- **D1**：选 Provider 抽象而非 server 内 if 分流 —— 理由见 §1.2，可维护性与可扩展性。
- **D2**：GPT-5.5 经**新增的 `/openai/v1/responses` 透传端点**暴露，**不**经 `/v1/chat/completions` —— 用户 2026-07-08 决定。透传保真、零格式损耗、实现最简。
- **D3**：同步 + 流式一次做完 —— 用户 2026-07-08 决定（agent 团队实跑需流式）。
- **D4**：支持图片输入（文本+图片多模态）—— 用户 2026-07-08 决定。透传天然支持，见 §3.4。
- **D5**：不并入 thinking/采样护栏重构 —— 控制爆炸半径（N3）。Provider 抽象本身是 ModelEntry 重构的一部分，二者合流；护栏重构仍独立。
- **D6**：auth 层不动 —— bearer token 对 runtime/mantle 两域名均已实测有效。
```

