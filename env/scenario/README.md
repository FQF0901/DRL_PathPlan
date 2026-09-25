# env/scenario

目的：场景 spec 的生成、校验与运行期行为；由显式 BIG block 序列构建小地图（3–5 块）。

## 文件职责
- `taxonomy.py`：标签常量与映射——12 个几何标签中 `bidirection` 因 S→B 接缝不可通行被排除，
  **11 类可抽样**；交通形态/控制/转向/难度集合、block 字符表与分层配额。
- `spec.py`：`ScenarioSpec` 契约（11 字段 `id/seed/split/blocks/geometry/traffic/limits/nav/ego/
  difficulty/labels`），JSON 往返 + `validate()`（单位一律 m/s，标签与字段交叉对账）。
- `generator.py`：几何分层（保底+均摊）+ 难度分层（密度/事件间隙/交叉口数派生）；自车 spawn 车道
  显式指定（首块中间车道，保证事件请求侧邻车道存在，关闭随机车道）；内容只由 seed 决定。
- `behaviors.py`：cut-in/cut-out 脚本事件的安装与运行（补位 `WaypointPolicy` 直写位姿，
  机动 +0.5 s 后交回 `IDMPolicy`）；同文件提供车道/投影/地图查询工具。
- `labels.py`：9 个逐步可观测标签（router 监督；不按 spec 类别直接置位）。
- `validator.py`：逐条实例化校验 + 覆盖率/失败分类报告；`cli.py`：spec 生成入口（由 `tools/gene_env.sh` 转发）。

## 事件与已知非确定性
- 事件窗口 = `[trigger_step, trigger_step+duration_steps)`（10 Hz env step）；字段 side/gap_m/speed_mps。
- 少数脚本事件场景**同 seed 运行间有微小差异**（如 `cutin_active` 样本数 11 vs 12）；`PYTHONHASHSEED`
  已排除，归因 MetaDrive 内部线程/挂钟时序 → 事件类 KPI 带运行间噪声，评测解释需计入；
  候选修复（事件时序改为步进式交接）**待补**。

## 校验结果（`env/specs/validation_*.json`）
- train 10,000/10,000、val 1,000/1,000、slice200 200/200、slice50 50/50 → **11,250/11,250 通过，0 失败**
  （`summary.passed` == `n_specs`，`n_failed=0`，`resample_suggestions` 为空）。
