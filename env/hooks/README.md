# env/hooks

目的：hook/probe 层（场景标签、MoE 路由、KPI 记录）。

## 现状（P0 占位，未实现）
- 本包 `__init__.py` 为空，无任何回调实现/注册表；全仓无 `import env.hooks`。
- 旧版 README 描述的 `on_reset/on_step/on_episode_end` 三回调**在代码中不存在**；经 `env/hooks` 的注册方式与新增 hook 流程：**待补**。

## 当前实际生效的等价机制
- **场景标签**：`env/scenario/labels.compute_step_labels(env, spec)`（9 标签）。
  训练侧由 `pipeline.vector_env.py` worker（record `router_labels`）与 `pipeline/trainer.py`
  的 `_router_labels_from_env`（LocalEnvPool）逐步调用；可视化由 `tools/visualize.py` 调用。
- **MoE 路由统计**：训练用 `pipeline/trainer.py::RouterMonitor`（有效专家数 Σw、权重/输出范数、
  token 数），随 `update()` 的嵌套指标汇总；
  `pipeline/monitoring.py::TrainingMonitor.on_moe_step` 为对外 hook，但当前无调用点（待接线）。
- **KPI 记录**：训练侧 `stages.py` 调用 `TrainingMonitor.on_train_step` + `flush`（`metrics.csv` +
  tensorboard，42 条诊断序列）；评测侧 `pipeline/eval_runner.py` 直接写 `metrics.json`/`episodes.csv`。
  同理 `on_scene_step`/`on_episode` 已实现但当前无调用点（待接线）。
- **MetaDrive 生命周期钩子**：`env/scenario/behaviors.py::ScenarioBehaviorManager`
  （before_step/after_step/reset/after_reset，PRIORITY=11）是脚本事件接入引擎的真实机制。

## 新增一个 hook（当前可行做法）
- 在 `TrainingMonitor` 增加 `on_*` 方法：训练/评测调用点显式调用；监控为 hook 式、非侵入
  （输入只读、tensorboard 不可用自动退化为 CSV、异常不中断训练）。
- 若要恢复 `env/hooks` 回调层：接口与注册表**待补**（需先设计，不与现有 monitoring 重复）。
