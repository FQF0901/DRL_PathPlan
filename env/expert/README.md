# env/expert

目的：专家/参考策略层——KPI 冻结基线 与 BC 冷启动数据源。

## PurePursuitIDMPolicy（规则基线，KPI 协议必需件）
- `pure_pursuit_idm.py`：横向 pure-pursuit 跟踪导航路线（自车车道锚定 + 后继车道链兜底），
  纵向 IDM 跟车 + 限速约束；输出 `[steer, throttle] ∈ [-1,1]²`；无随机数、同状态同动作。
- 接口（MetaDrive `BasePolicy`）：构造 `(control_object, random_seed)`；`act()` 为逐步控制入口；
  `reset()` 清空诊断，但**引擎不会自动调用**，调用方须在每次 env reset 后显式调用。
  关键参数可配（wheelbase/max_steer/lookahead/IDM）；`params()` 导出生效快照写入评测 meta。
- 调用：`tools/test.py --policy baseline`（eval_runner 经 `engine.add_policy(ego.id, cls, ego, seed)` 注册）；
  批量统计用 `tools/baseline_eval.py`。
- 冻结参照：`runs/baseline_eval/val_reference.json`（1000 条 val：success 0.748 / off-road 0.061）
  与按主标签的 `val_reference_by_primary.json`。

## IDMPolicy（BC 数据源，MetaDrive 内置）
- `tools/collect_expert.py --expert idm`（默认）：同样走 `add_policy` 注册；每 5 个 env step（0.5 s）
  记录一帧；专家只用于冷启动，最终性能靠 RL；`--expert pure_pursuit` 可切换做对照。

## 采集过滤与产出（p2-contract §8.2）
- 规则：终末截断；3 s 目标窗口内不得 crash/off-road/arrive；专家须 `on_lane`；6 点目标在
  执行器运动学下 round-trip 可复现；事件帧仅在 `fired ∧ actor_alive` 时保留；难度×几何配平。
- 实测 `runs/bc_expert_2k`（2000 spec）：**103,938 样本，产出率 0.718**（144,785 候选）；
  过滤计数 `roundtrip_fail 28,982` + `terminal_window 11,865`；8 个受监督标签正样本均 ≥50；
  `kinematics_source=env.tracking.interpolate`、`history_storage=per_frame`。
