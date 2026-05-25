# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

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
