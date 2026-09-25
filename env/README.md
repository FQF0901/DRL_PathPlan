# env

目的：非侵入式封装 MetaDrive，提供场景、观测、专家与动作执行层。

## 场景（`scenario/`）
- `spec.py`：`ScenarioSpec` 数据契约（字段/单位/JSON 往返/validate）；`taxonomy.py` 标签常量与 block 映射。
- `generator.py`：几何分层 + 难度分层抽样；可抽样几何 11 类（`bidirection` 因 S→B 接缝不可通行被排除）。
- `validator.py`：逐条实例化校验（导航/限速/几何/事件触发），输出覆盖率与失败分类 JSON 报告。
- `behaviors.py`：脚本化 cut-in / cut-out（补位 WaypointPolicy 直写位姿，机动结束后交回 IDMPolicy）。
- `labels.py`：逐步可观测标签（router 监督）；`cli.py`：spec 生成入口（经 `tools/gene_env.sh` 调用）。

## MetaDrive 封装（`metadrive_env.py`）
- 显式 `BIG_BLOCK_SEQUENCE` 建图；`store_map=False`（地图每次 reset 重建，但不泄漏 env 实例）。
- 建图后逐 lane 写限速（m/s），并覆写 `reset()` 在**每次重建后重放**后处理与 behaviors；可选 ≤32 张地图 LRU（默认关）。
- 自车 spawn 车道显式（`ego.spawn_lane_index`，关闭随机车道），保证事件请求侧邻车道存在。

## 观测（`obs/`）
- 通道：ego(8)、od(16,9)、ld(16,7)、nav(11)、signal(4) + 掩码；OD/LD 为盒式 scope 前 100 / 后 50 / 左 25 / 右 25 m。
- 记忆：6 帧 @0.5 s（每 5 个 env step 采 1 帧），SE(2) 对齐到**当前** ego 系；预热复制最旧真实帧并用 `hist_valid` 标 0。
- 上一策略动作经 `env.prev_policy_action` 注入 ego 末 2 维（形状不变）；未注入保持 0。

## 专家与跟踪
- `expert/pure_pursuit_idm.py`：纯跟踪 + IDM 规则基线（KPI 冻结参照；确定性、无随机数）。
- BC 数据源：MetaDrive 内置 `IDMPolicy`，经 p2-contract §8.2 过滤规则采集（`--expert pure_pursuit` 可切换）。
- `tracking.py`：`interpolate`（(ds,dθ)→10 Hz 常曲率圆弧）是运动学单一真源；
  `ExactTracker`（阶段 A/B 精确执行）、`LqrTracker`（阶段 C 闭环，预瞄 LQR）；
  `roundtrip_error` 供 BC 数据横向保真检查。
