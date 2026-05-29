# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

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
