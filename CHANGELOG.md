# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [0.2.1] — 2026-07-08

### 新增

- **接入 xAI Grok 4.3**（`xai.grok-4.3`，1M 上下文）。与 GPT-5.5 同一
  `bedrock-mantle` + Responses 透传通道，经 `POST /openai/v1/responses` 调用，
  纯配置接入、零新代码。别名：`grok-4.3` / `grok` / `grok-4` / `grok4.3` /
  `grok-4-3`。官方定位金融 / 法律文档分析，适配投研场景。
- `xai.` 前缀纳入 passthrough 识别（可直接传原始 Bedrock ID）。

### 修复

- `pyproject.toml` 版本号补齐到位（0.2.0 时遗漏，导致构建的 wheel 元数据版本
  标签滞后）。新增版本一致性测试，防止 `__init__.py` 与 `pyproject.toml` 再脱节。

## [0.2.0] — 2026-07-08

### 新增

- **接入 OpenAI GPT-5.5**（`openai.gpt-5.5`，272K 上下文）。经 Bedrock 的
  `bedrock-mantle` 端点 + OpenAI Responses API 调用，**不同于**现有 Claude 走
  的 `bedrock-runtime` + Anthropic Messages 链路。
  - 新增对外端点 **`POST /openai/v1/responses`**：客户端直发 OpenAI Responses
    原生格式，网关**透传**给 mantle（仅把模型别名替换为上游 ID，其余字段原样
    转发）。图片、reasoning、工具字段天然对齐、零转换损耗。客户端设
    `OPENAI_BASE_URL=http://<gateway>/openai/v1` 即可直用。
  - 支持**图片输入**（`input_image`）：base64 data URI 与 `s3://` 两种 scheme
    可用；公网 http(s) URL 不受 mantle 支持（会返回 400）。
  - 同步 + 流式（标准 SSE 透传，UTF-8 增量解码保证 CJK 不跨块损坏）。
- **Provider 抽象层**（`bedrock_gateway/providers/`）：将"如何与某个上游对话"
  （URL 构造、响应渲染、流转换）从 server 的编排逻辑（重试/退避/超时/metrics/
  流式预检）中解耦。新增模型/厂商 = 加一个 Provider + 一条 `ModelEntry`，
  `server.py` 不再改。现有 Claude 全家收敛为 `AnthropicBedrockProvider`，
  GPT-5.5 为 `OpenAIResponsesProvider`。
- **`ModelEntry` 新增 `endpoint` / `protocol` 字段**（默认 `runtime` /
  `anthropic`），向后兼容：现有模型与旧扁平 YAML 不写这两个字段即保持原行为。

### 变更

- `_handle_sync` / `_handle_stream` 参数化 `provider`，URL 与响应渲染委托给
  provider；重试/预检/超时/metrics 逻辑不变、两个 provider 共享。
- 端点守卫：`/v1/chat/completions`、`/v1/messages` 拒绝 Responses 协议模型；
  `/openai/v1/responses` 拒绝 Anthropic 协议模型，均返回带指引的 400。

### 测试

- 新增 35 个用例（`test_gpt55_responses.py`）：注册/别名、ModelEntry 向后兼容、
  provider 选择、Responses provider 单元行为（URL/透传/UTF-8 跨块/错误帧）、
  端点路由与守卫、透传端到端（模型替换、图片块保留、流式透传）。
- 真机端到端验证：经网关实调 GPT-5.5 同步/流式/图片（base64），Claude 回归。
- 总测试 **630 个全部通过**（595 基线零回归 + 35 新增）。

## [0.1.4] — 2026-05-29

### 修复

- **流式错误不再吞没、不再卡死客户端**：流式请求遇到上游错误时，过去会
  被伪装成 `200` 的 SSE 响应并塞入一个孤立的 `error` 事件，导致客户端状态机
  收不到合法的流终止信号而永久挂起（表现为「对话几次后卡住不回复」）。现按
  错误发生的阶段区分处理，与上游 Anthropic API 行为对齐：
  - **流建立前的错误**（如不支持的 `web_search_20250305` 工具触发的 400、
    auth、model-not-found、重试穷尽）→ 返回**真实 HTTP 状态码 + 完整错误体**，
    不再进入流式。新增预检逻辑 `_open_upstream_stream`，开流前验状态。
    （`server.py`）
  - **流建立后的中途错误** → 发出协议合法的终止帧：`/v1/messages` 发
    `event: error`，`/v1/chat/completions` 发 error chunk + `[DONE]`，
    各自对齐自身协议，客户端能正常收尾。
- **识别 Bedrock 流式异常帧**：`decode_event_stream_chunk` 过去只匹配
  `"bytes"` 正常事件帧，会**静默丢弃**流中途的 `:exception-type` 异常帧
  （throttling / internalServer / modelStreamError 等），使客户端无错误、
  无终止地卡死。现解析异常帧为合成 `_exception` 事件并映射到 HTTP 状态码
  （未知类型兜底 500）。（`converter.py`）
- **消除两条流式路径的处理分歧**：`/v1/messages` 与 `/v1/chat/completions`
  统一预检 + 中途异常收尾逻辑。

### 新增

- **关键排查日志**：`STREAM-OPEN ok/error/retryable/timeout/failed`（开流阶段）
  与 `STREAM-MID error/timeout`（流中途故障），含路径、模型、状态码、原因，
  便于定位「卡住」类问题。（`server.py`）

### 测试

- 新增 12 个用例：解码器异常帧识别（6）、异常类型→状态码映射（2）、两条
  路径流中途异常的合法收尾（3）、web_search 400 真机回归（1）。
- 改写 7 个旧用例：原断言「假 200 + SSE」的错误契约，现按真实 HTTP 状态码
  断言。总测试 595 个全部通过。

## [0.1.3] — 2026-05-29

### 新增

- **支持 Claude Opus 4.8**：注册 `claude-opus-4.8` →
  `us.anthropic.claude-opus-4-8`（context 1M / max output 128K），
  并加入名称变体别名 `claude-opus-4-8` / `claude-4.8-opus` /
  `claude-4-8-opus`。裸别名 `claude-opus-4` 维持指向 4-6 不变。
  （`config.py:_DEFAULT_MODELS`、`config.py:_MODEL_ALIASES`）
- **4.8 启用 adaptive thinking**：在 `_ADAPTIVE_THINKING_PATTERNS` 加入
  `claude-opus-4-8`，使 `reasoning_effort` 各档位映射为
  `{"type": "adaptive"}`，而非退化的固定 `budget_tokens`。
  （`converter.py`）

### 测试

- 新增 `tests/test_opus_4_8.py`（32 个用例），覆盖三层一致性（模型注册、
  别名解析、adaptive thinking）、全部 `reasoning_effort` 档位，以及兜底
  路径（<1024 budget 上钳、未知 effort 不注入 thinking、无 effort 不注入）。
  端到端真实请求确认 model ID `us.anthropic.claude-opus-4-8` 有效（返回
  200），流式与非流式路径均验证通过。总测试 583 个全部通过。

## [0.1.2] — 2026-05-25

### 修复

- **删除针对 Bedrock 上游的主动可达性探针**：原实现每 30 秒向
  `https://bedrock-runtime.<region>.amazonaws.com/` 发一次 GET，
  期望以「TCP+TLS 通就算上游健康」作为判定依据。该端点没有根资源，
  AWS 设计上必然返回 404，所以这个探针：
  - 区分不出"网络不通"以外的故障（凭据失效 / 限流 / 模型下线全部漏报）；
  - 在 dashboard 关闭时仍每 30 秒打一条 httpx INFO 日志（0.1.1 已通过
    `dashboard.enabled` 开关压制，但 dashboard 开着的实例仍被刷屏）；
  - 与已有的请求级 metrics 信号重复，没有新增信息量。

  0.1.2 直接删除该探针。dashboard 的「Upstream」面板改为基于真实请求
  统计被动推导（最近 5 分钟窗口）：

  | 条件 | 状态 |
  | --- | --- |
  | 窗口内无流量 | `unknown` |
  | 出现 401/403 | `auth_failed`（凭据问题，单独标记） |
  | 成功率 ≥ 99% | `healthy` |
  | 成功率 ≥ 80% | `degraded` |
  | 成功率 < 80% | `down` |

  （`metrics.py:upstream_health`、`health.py:snapshot`、`api.py`、
  `static/app.js` 渲染层）

### 移除

- `HealthMonitor._probe_once` 与 `_upstream_probe_task`、相关的
  `_UpstreamState` / `_UPSTREAM_PROBE_INTERVAL_S` / `_UPSTREAM_PROBE_TIMEOUT_S`
  常量、health 模块中对 `httpx` 的依赖。
- 旧响应字段 `upstream.reachable` / `latency_ms` / `last_check`，
  替换为 `upstream.status` / `success_rate` / `total` / `errors` /
  `window_minutes` / `last_success`。

### 测试

- 新增 `tests/test_upstream_health.py`（15 个用例，覆盖各成功率边界、
  401/403 覆盖逻辑、`last_success` 持续性、窗口参数）。
- 新增 `tests/test_integration_0_1_2.py`（5 个用例，端到端验证
  `unknown → healthy → down → auth_failed` 状态切换，并断言
  dashboard 健康端点不会触发任何指向 bedrock-runtime 根路径的 GET）。
- 更新 `tests/test_health_coverage.py` 与 `tests/test_integration_0_1_1.py`
  以反映探针删除；总测试 551 个全部通过。

## [0.1.1] — 2026-05-25

### 修复

- **dashboard 关闭时不再启动后台任务**：`HealthMonitor` 的事件循环延迟采样
  与 Bedrock 上游可达性探针仅在 `dashboard.enabled: true` 时启动。此前两
  者无条件启动，但 dashboard 关闭时没有任何代码读取它们采集的数据，相当
  于做了无用功，并且每 30 秒一次的探针让 httpx 输出 INFO 级 404 日志，
  在长期运行的实例上可占到日志总量的 87% 以上。
  （`server.py:create_app`）

- **上游 4xx 不再误报为 ERROR**：上游返回 4xx（客户端原因，如图片超过
  5 MB、未知模型）现在以 WARNING 记录；5xx 与未知状态码仍为 ERROR。
  401/403 单独处理 —— 它们意味着网关自身凭据失效，仍为 ERROR 并附
  `[auth-failure]` 标签便于告警识别。
  （`server.py:_log_upstream_error`）

- **意外异常不再丢失栈帧**：所有 catch-all `except Exception` 分支改用
  `logger.exception`，traceback 会随日志一起写入，便于事后定位。
  （`server.py` 中三处 chat / messages / streaming 兜底）

### 新增

- 启动时若 dashboard 关闭，明确写入一行 INFO，说明探针与延迟采样未启动
  以及原因，便于运维确认。

### 测试

- 新增 `tests/test_fixes_0_1_1.py` 与 `tests/test_integration_0_1_1.py`，
  覆盖上述三处修复的契约与端到端行为；总测试 531 个，`server.py` 行覆盖
  100%。

## [0.1.0]

- 初始公开版本。
