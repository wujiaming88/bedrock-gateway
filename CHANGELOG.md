# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [0.4.13] — 2026-08-14

### 修复

- 修正 Anthropic Messages → OpenAI Responses 的 cached usage、终态累计 usage、model alias、缺省 `max_tokens`、context overflow 和异常流终止语义。
- `response.failed`、错误事件和异常 EOF 不再伪装成正常 `end_turn`；诊断日志不记录 prompt、tool 参数或完整上游错误 message。

### 测试

- GPT-5.6 Messages sync/stream 真机通过；全量测试 893 项通过，本次生产代码变更可执行行覆盖率 100%。

## [0.4.12] — 2026-08-14

### 新增

- 新增 Azure OpenAI `POST /openai/v1/images/edits`：支持 OpenAI multipart 多图/mask、同步 JSON 与 SSE 原样透传。
- Edits 按 Azure transport + `openai-images` API 能力路由，不绑定具体模型名称；以 `gpt-image-2` 作为严格真机验证基线。

### 改进

- Anthropic Messages → OpenAI Responses 转换现在正确拆分 cached input usage，并在流终态返回累计的 input/cache/output usage，供 Claude Code context 预算与 dashboard 使用。
- 转换流的 `response.failed`、error 和异常 EOF 以唯一 Anthropic error 终止，不再伪装成正常 `end_turn`；context overflow 规范化为 `invalid_request_error`。
- 转换路径保留客户端 model alias，同时将缺省 `max_tokens` 按模型 registry 默认值映射为 `max_output_tokens`。
- 增加不记录 prompt/tool 内容的诊断日志，记录 client/upstream model、输出预算、usage/cache 和 stream terminal 状态。
- `/v1/messages/count_tokens` 当前环境的精确上游候选经探测不可用（Mantle Responses 405、Azure deployment unsupported、Claude CountTokens 权限不足），因此保留兼容性近似估算；该估算不用于本地 context 强制校验。
- 通用上游 dispatch 支持预编码、可重放 payload；multipart 在重试时复用同一 body 与 boundary，现有 JSON API 保持原行为。
- Images edits metrics 不再由 middleware 预读大文件，模型信息由 handler 安全注入。

### 测试

- 覆盖 multipart 字段/文件保真、严格模型 guard、重试重放、同步响应和既有 Images/Azure 回归；真实 Azure smoke 增加有效 PNG 与 Base64 图片校验。

## [0.4.11] — 2026-08-13

### 修复

- 修复 Bedrock mantle GPT-5.x 对 Responses `web_search` 工具返回 400：仅在 Bedrock GPT-5.x normalizer 中移除 mantle 不支持的 `search_content_types` 字段，保留 `web_search`、`external_web_access` 及其他选项。
- Azure Responses、Grok、Claude 与非 `web_search` 工具继续原样透传，不受兼容降级影响。

### 测试

- 覆盖普通与 `additional_tools` 提升后的 web search、幂等/不修改原请求、嵌套同名字段保留、Bedrock 集成及 Azure 不归一化回归。

## [0.4.10] — 2026-08-11

### 改进

- 每日文件日志改为默认开启；未配置 `logging` 段时直接写入 `/var/log/bedrock-gateway` 并保留 30 天。仅在显式设置 `logging.file_enabled: false` 时关闭。

### 测试

- 更新默认配置断言，确保文件日志默认启用。

## [0.4.9] — 2026-08-11

### 改进

- 每条 console/journald 与每日文件日志增加代码来源位置，格式为 `[文件名.py:行号]`；网关日志可直接定位到业务代码，Uvicorn/httpx 日志可定位到对应依赖内部调用点。

### 测试

- 增加 formatter 回归断言，确保文件名和行号持续出现在日志中。

## [0.4.8] — 2026-08-11

### 新增

- 增加可选每日文件日志：网关业务、Uvicorn access/error 与 httpx 上游请求日志统一写入 `bedrock-gateway-YYYY-MM-DD.log`，按主机本地午夜切分，默认保留 30 天。
- 新增 `logging.file_enabled`、`logging.directory`、`logging.retention_days` 配置；console 输出继续保留，systemd journald 不受影响。
- systemd 推荐日志目录为 `/var/log/bedrock-gateway`，由 `bedrock:bedrock` 持有并使用 `0750` 权限。

### 测试

- 覆盖跨午夜文件切换、30 天清理边界、多 logger 统一采集、重复初始化去重和 Uvicorn 启动配置。

## [0.4.7] — 2026-08-11

### 修复

- 将 Bedrock mantle 上 `gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-5.6-luna`
  的上下文元数据由 272K 修正为 1M，并同步模型清单、配置示例和文档。
- 此变更只修正 `/v1/models` 展示的 advisory metadata；网关不在本地限制或截断
  上下文，实际请求限制仍由 Bedrock mantle 执行。

### 测试

- 更新 GPT-5.6 默认模型注册断言，覆盖三个变体的 1M 上下文元数据。

## [0.4.6] — 2026-07-29

### 修复

- **修复 Bedrock GPT-5.x 对 Codex/Responses encrypted reasoning 的 400**。
  Codex 会在历史 input 中回放 `reasoning.encrypted_content` 这类 opaque 状态；
  Bedrock GPT-5.x 只接受自己签发、前缀为 `rsn_` / `smry_` 的 blob，遇到其它
  issuer 的 encrypted content 会返回：
  `encrypted content missing recognized prefix (expected rsn_ or smry_)`。
- `responses_normalizer.py` 现会在 Bedrock GPT-5.x Responses 路径中过滤 opaque state：
  - 保留 `encrypted_content` 以 `rsn_` / `smry_` 开头的项；
  - 删除其它非法 encrypted content；
  - 若 reasoning item 删除非法 blob 后只剩空壳，则删除该 item；
  - 嵌套非法 encrypted_content 会被移除，但可见文本/语义内容保留。
- 作用范围仍限制在 `transport=bedrock` + `dialect=openai-responses` + `openai.gpt-5*`，
  Azure prefix、Grok、Claude、图片端点不受影响。

### 测试

- 扩展 `test_responses_normalizer.py` 和 `test_gpt56_bedrock.py`：覆盖非法 encrypted
  content 删除、`rsn_` / `smry_` 前缀保留、嵌套 encrypted content 过滤、空 reasoning
  item 删除、Azure prefix 不被 normalise。`responses_normalizer.py` 行覆盖保持 100%。
- 真机验证：包含非法 encrypted content 的 Codex-like 请求由 400 变为 HTTP 200；
  GPT-5.5、GPT-5.6 Sol、Azure prefix、`/v1/messages` 入向翻译和 images/generations
  回归均正常。全量测试通过。

## [0.4.5] — 2026-07-23

### 修复

- **修复 Codex 调 Bedrock GPT-5.6 Sol 的 Responses `input` schema 400**。Codex 会在
  `/openai/v1/responses` 的 `input` 中发送扩展 item，例如 `additional_tools` 与
  `role=developer` 的 message；Bedrock mantle GPT-5.6 的 validator 不接受这些
  variant，返回 `Invalid 'input': value did not match any expected variant`。
- 新增 `responses_normalizer.py`，仅对 **Bedrock + OpenAI GPT-5.x + Responses** 路径启用：
  - `additional_tools` input item → 合并到 top-level `tools`；
  - `developer` message → 合并到 top-level `instructions`；
  - message content 中 `type=text` → `type=input_text`；
  - 保留 user/assistant/tool 等其它 input item；不改 `reasoning`、`stream`、`tool_choice`、`metadata` 等字段；
  - 幂等、保守、只在 Bedrock GPT-5.x 上启用，Azure prefix passthrough 不受影响。

### 测试

- 新增 `test_responses_normalizer.py`，normalizer 行覆盖 **100%**，覆盖真实 Codex-like
  input、additional_tools 提升、developer 指令提升、text→input_text、幂等、边界分支。
- 扩展 `test_gpt56_bedrock.py`，覆盖 `/openai/v1/responses` 端点中 Bedrock GPT-5.x
  会启用 normalizer，且 Azure prefix 模型不会被误 normalise。全量测试通过。

## [0.4.4] — 2026-07-23

### 新增

- **上游 schema 错误的脱敏请求结构日志**：当上游返回非重试错误（同步 `_handle_sync`
  或流式 `_open_upstream_stream` 开流前失败）时，网关现在额外打印 `REQ-SHAPE`
  诊断行，记录 request body 的结构摘要：top-level keys、model、stream、Responses
  `input` item types/roles/content block types、messages/tools/reasoning/max token 字段。
- 诊断日志**不记录正文、tool 参数值、headers、密钥**，只记录类型/长度/keys，便于定位
  `Invalid 'input': value did not match any expected variant` 这类上游 schema 错误，
  同时避免泄露用户内容。

### 测试

- 新增 `test_request_shape_logging.py`：覆盖结构摘要脱敏、同步上游错误日志、流式开流
  上游错误日志，断言用户正文不会出现在日志里。全量测试通过。

## [0.4.3] — 2026-07-23

### 新增

- **接入 Bedrock mantle 上的 OpenAI GPT-5.6 family**：新增默认模型
  `gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna`，分别映射到
  `openai.gpt-5.6-sol` / `openai.gpt-5.6-terra` / `openai.gpt-5.6-luna`。
  三个模型均经 Bedrock mantle `/openai/v1/responses` 真机实探返回 HTTP 200。
- 补齐常见别名：`gpt-56-sol` / `gpt5.6-sol` / `gpt-5-6-sol` /
  `openai.gpt-5.6-sol` / `openai-gpt-5.6-sol`（Terra/Luna 同理）。
  这些模型走 `endpoint: mantle` + `protocol: openai-responses`，与 `gpt-5.5`
  同一条 Responses 透传链路。

### 测试

- 新增 `test_gpt56_bedrock.py`：覆盖默认模型注册、别名解析、Responses 路由到
  `bedrock-mantle.../openai/v1/responses`、以及错误端点 guard。全量测试通过。
- 注：`context_length` / `max_output` 暂按 GPT-5.x 现有 1.05M / 128K advisory
  值登记；若后续官方 model card 给出不同规格,应同步修正。

## [0.4.2] — 2026-07-20

### 新增

- **支持 Azure OpenAI 图片生成透传**：新增 `POST /openai/v1/images/generations`，
  用于 `gpt-image-2` 这类 OpenAI Images Generations 模型。现有 Azure 前缀透传
  直接复用：客户端传 `model: "azure/gpt-image-2"`，网关剥离为上游 deployment
  `gpt-image-2`，经 `AzureTransport` 使用 `api-key` 转发到
  `.../openai/v1/images/generations`，响应中的 `data[].b64_json` / URL 原样返回。
  图片端点为同步 JSON，不支持 `stream=true`。
- 新增 `openai-images` dialect，继续保持 **Transport × Dialect** 职责边界：图片
  格式归 dialect，Azure endpoint/auth 归 transport；不改 Responses/Chat/Anthropic
  既有路径。
- dashboard metrics 纳入 `/openai/v1/images/generations`，可统计 model/status/latency
  和错误；图片接口无 token usage 时 token 计数保持 0。

### 测试

- 新增 `test_images_generations.py`：覆盖 images dialect、provider registry、Azure
  prefix 路由、非法 JSON、stream 拒绝、错误模型 guard、上游错误/坏 JSON、metrics。
- 变更行覆盖率 **100%**（`dialect_images.py` 100%；server 新 endpoint 变更区 100%；
  middleware 新路径变更行 100%）。全量测试通过。
- 真机验证：上游 Azure `gpt-image-2` `/images/generations` 返回 `b64_json`；本机
  网关 `/openai/v1/images/generations` 将在部署步骤中验证。

## [0.4.1] — 2026-07-15

### 移除

- **下线两个失效的 Claude 默认模型**（经真机探测确认不可用）：
  - `claude-sonnet-3.5`（`...claude-3-5-sonnet-20241022-v2:0`）——AWS 已标记
    **EOL**（reached end of life）。
  - `claude-sonnet-4`（`...claude-sonnet-4-20250514-v1:0`）——被标记 **Legacy**
    且账号 30 天未调用而锁定，已有 4.6 / 4.8 替代。
  - 指向这两者的所有别名一并移除；裸 `claude-sonnet` / `claude-4-sonnet` 改指向
    当前最新的 **`claude-sonnet-4.6`**。保留 `claude-haiku`（探测 200 正常）。
  - 仍可用原始 Bedrock ID 直传（passthrough 不受影响）。

### 修复

- **Opus 别名遗漏导致的 `Unknown model` 400**：`claude-opus-4.7`（点号）可用，但
  `claude-opus-4-7`（连字符）报 400——别名表只为 4.8 补了连字符写法，4.7 / 4.6 漏了。
  现补齐 Opus 全家族两种写法：`claude-opus-4-7` / `claude-4-7-opus` / `claude-4.7-opus`
  → 4.7；`claude-opus-4-6` / `claude-opus-4.6` → opus-4；并加 `claude-sonnet-4-6`
  → sonnet-4.6。真机验证 `claude-opus-4-7` 与 `claude-opus-4.7` 均 200 且解析到同一
  上游 id。

### 测试

- 新增回归：Opus 点号/连字符解析一致、被删模型条目+别名值双清零、裸 `claude-sonnet`
  重定向到存活模型;既有别名一致性测试保证所有别名指向有效默认。全绿。

## [0.4.0] — 2026-07-09

### 新增

- **Anthropic 客户端可调任意 Responses 模型**（Claude Code 接任意模型）。
  `POST /v1/messages` 现按目标模型的 dialect 分叉：Claude（`anthropic`）走原
  路径**完全不变**；GPT-5.5 / Grok / `azure/<deployment>`（`openai-responses`）
  被**翻译**（Anthropic Messages ⇄ OpenAI Responses）后转发。这样只会说 Anthropic
  Messages 协议的客户端（如 Claude Code 经 `ANTHROPIC_BASE_URL`）可直接用非
  Anthropic 模型——设 `ANTHROPIC_MODEL=gpt-5.5` 或 `ANTHROPIC_MODEL=azure/gpt-5.5`
  即可。**唯一的行为变化**：这类模型此前在 `/v1/messages` 返回 400，现改为翻译转发；
  `/openai/v1/responses`、`/v1/chat/completions`、Claude 的 `/v1/messages` 一律零改动。
  - 新增翻译模块 `messages_to_responses.py`（`converter.py` 的镜像方向）：请求
    （system→instructions、messages→input、tool_use↔function_call、tool_result→
    function_call_output、tools/tool_choice、图片、max_tokens→max_output_tokens）、
    同步响应、以及**有状态流适配器**（Responses SSE → Anthropic message_start /
    content_block_* / message_delta / message_stop，含工具调用 `input_json_delta`
    增量拼装）。
  - 复用现有 Transport × Dialect 的全部横切逻辑（重试 / 超时预算 / 预检 / metrics）
    与两种 transport：`gpt-5.5`（Bedrock mantle）与 `azure/gpt-5.5`（Azure api-key）
    **共用同一份翻译代码**，后者零额外成本——两轴设计的直接回报。
  - **范围（v1）**：仅 Anthropic-in → Responses-out。`thinking`/reasoning 块丢弃
    （不回传签名）、`cache_control` 剥离（prompt caching 不生效），均为无损降级。
    暂不含 Chat 目标格式。
  - **接入注意**：Claude Code 的 `CLAUDE_CODE_USE_BEDROCK`/`CLAUDE_CODE_USE_VERTEX`
    一旦存在且非空（含 `"0"`）就会直连云、忽略 `ANTHROPIC_BASE_URL`——必须整个
    unset 才能走网关。README「Claude Code 接入」小节含完整配置与验证步骤。

### 测试

- 新增 `test_messages_to_responses.py`（80 例，模块**行覆盖 100%**）：请求/响应
  各类块翻译 + **Anthropic Messages 协议一致性校验器**（严格断言事件顺序、
  content block 生命周期、合法 `stop_reason`、`input_json_delta` 拼装、UTF-8 跨块）。
- 新增 `test_messages_inbound_integration.py`（16 例）：端点分叉端到端（mantle +
  Azure 两条 transport、同步 + 流式）、上游错误→Anthropic 信封、流中途超时/异常
  收尾、**Claude 回归（原生 invoke 路径不变）**、openai-chat 模型仍拒绝。
- 变更行覆盖率 **100%**（新模块 + server.py 新增分支/处理器）。总测试 **786 全绿**
  （691 基线零回归 + 95 新增）。

## [0.3.5] — 2026-07-08

### 修复

- **GPT-5.5 规格修正**：`context_length` 64K→**1,050,000**、`max_output`
  64K→**128,000**，采用 OpenAI 官方 model spec（此前的 272K/64K 中 64K 是接入时
  的保守猜测，model card 标 N/A）。这两个字段是 advisory（`/v1/models` 展示 +
  客户端省略 max_tokens 时的默认值），非硬限制。Grok 4.3（1M / 131072）经核实
  准确，未改。

## [0.3.4] — 2026-07-08

### 修复

- **`/openai/v1/responses` 请求此前完全未计入 dashboard metrics**（`_LLM_PATHS`
  漏了该端点，自 0.2.0 起）。GPT-5.5 / Grok / Azure 的调用现被正确统计，model
  名与 token 归属正常（token 提取已兼容 Responses 的 `input_tokens/output_tokens`）。
- **上游 200 但响应体非法 JSON** 现返回 **502**（bad gateway）而非 500（伪装成
  网关崩溃）——上游坏数据不再表现为网关自身错误。（chat + messages 两条 sync 路径）

### 变更（超时/错误处理审查）

- **连接超时与读取超时分离**：`retry.timeout` 现只管 read/write/pool，connect
  固定 ≤10s 快速失败——连不上上游不再干等满整个 timeout。
- **重试总时长封顶**：整个重试序列受墙钟预算约束（`timeout × 1.5 × max_retries`），
  避免慢上游把 N 个完整超时叠成数分钟挂起。
- **SigV4 auth 模式与 mantle/Azure 不兼容时启动告警**：`credentials`/`iam_role`/
  `profile` 只能签 Bedrock runtime；配了 mantle/Azure 模型会在启动时 WARNING 列出
  （服务照常运行，Bedrock 模型不受影响）。README 标注该限制。

### 测试

- 新增 `test_timeout_errors.py`（坏 JSON→502、超时拆分、重试预算、responses 端点
  纳入 metrics）+ config SigV4 告警测试。总测试 **691 全绿**。

## [0.3.3] — 2026-07-08

### 新增

- **Azure 前缀透传（零模型登记）**：`azure_resources` 条目可加 `prefix`（如
  `azure`），之后客户端用 `azure/<deployment>` 调用即自动路由到该资源——无需
  为每个 deployment 写 `models:` 条目。前缀被剥离作为 Azure deployment 名，
  dialect 由命中的端点决定（responses / chat）。deployment 增删换名不用动网关。
  真机验证：`azure/gpt-5.5`(responses) 与 `azure/gpt-5`(chat) 在无 `models:`
  段的配置下均调通。（显式 `models:` 登记仍可用，二者共存；透传模型不进
  `/v1/models` 列表。）

### 测试

- 新增前缀路由单测 + 集成测试（解析/剥离/端点定 dialect/不命中回退/与默认共存）。
  总测试 **681 全绿**。

## [0.3.2] — 2026-07-08

### 变更（架构）

- **收紧 Transport × Dialect 职责边界**：dialect 的 `operation_path` 现只返回
  裸操作（`/responses`、`/chat/completions`），不再含 `if transport == "azure"`
  判断；`/openai/v1` 根前缀的拼接移交给 transport（`BedrockTransport` 为 mantle
  端点补 `/openai/v1`，`AzureTransport` 用资源 base 已有的根）。两轴恢复完全正交
  ——dialect 不再感知云。**最终 URL 与行为完全不变**，纯内部重构。
- **Azure 单资源即可**：确认 `.../openai/v1` 根同时服务 Responses 与 Chat，无需
  api-version、无需为两种 dialect 拆两个资源。config.example / README 改为推荐
  v1 根；smoke 脚本简化为单资源。（带 `?api-version=` 的旧 base 仍兼容。）

### 测试

- 新增 `TestTransportOwnsApiRoot`（钉死两云拼出的完整 URL）+ operation_path 云
  无关性断言。总测试 **674 全绿**，真机冒烟 9/9。

## [0.3.1] — 2026-07-08

### 变更

- **`models:` 配置改为与内置默认合并**（原为整体替换）。现在自定义模型是在
  Claude/GPT-5.5/Grok 等默认之上**追加**，同名条目**覆盖**默认——加一个模型
  不再意外丢失所有默认。新增 `use_default_models`（默认 `true`）开关，设
  `false` 可只暴露自己配的模型。修正了一个反直觉、易踩的行为。

## [0.3.0] — 2026-07-08

### 新增

- **多云支持：接入 Azure OpenAI**。新增 `azure` transport（`api-key` 鉴权 +
  per-资源 endpoint），可经现有 `/openai/v1/responses`（Responses 模型）和
  `/v1/chat/completions`（Chat 模型）调用 Azure 部署，均为透传。
  - **两层配置**：`azure_resources`（资源 = endpoint + key）+ 模型条目引用资源
    并指定 deployment（→ 上游请求体 `model` 字段）。资源引用未定义时报错。
  - 响应的 Azure `content_filter_results` 字段原样透传；URL 保留资源 base 上的
    `?api-version=...`（若有）。
  - 真机端到端验证：Azure gpt-5 走 responses 与 chat 两条路径，同步 + 流式均通过。
- **`openai-chat` dialect**：OpenAI Chat Completions 透传（Azure + mantle），
  `/v1/chat/completions` 现按 dialect 分流——`anthropic`（Claude，转换）与
  `openai-chat`（透传）共存。

### 变更（架构）

- **Provider 抽象重构为 Transport × Dialect 两轴**（见
  `docs/multi-cloud-multimodal-design.md`）。`transport`（云 + 鉴权）与
  `dialect`（请求/响应/流形状）正交，加云/加格式从 N×M 组合塌缩为 N+M 加法。
  纯重构、零行为变化：现有 Claude / GPT-5.5 / Grok 全部路径无损迁移。
  - `ModelEntry` 新增 `transport` / `dialect` 字段；旧 `protocol` / `endpoint`
    保留并映射到新轴，存量配置零改动继续工作。

### 测试

- 新增 `test_azure.py`（25 例：config 两层解析 / AzureTransport URL·auth /
  ChatPassthroughDialect / 选择 / responses·chat 集成 / 守卫 / content_filter
  透传）+ config 覆盖行为固化测试。
- 新增可重复端到端冒烟脚本 `scripts/smoke_e2e.py`（Bedrock + Azure，同步 +
  流式，带断言），真机 9/9 通过。
- 总测试 **671 全绿**。

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
