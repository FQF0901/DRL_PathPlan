# reward_model

可插拔规则奖励（训练用）+ 同口径 KPI 纯函数实现。纯 Python/NumPy，不 import env/metadrive，可脱离仿真单测。
对外 API：`register_term / make_term / build_terms / available_terms`、`DEFAULT_TERM_CONFIGS`、`RewardAggregator / AggregationConfig / StepReward`、`compute_kpis / wilson_ci / Thresholds`（`reward_model/__init__.py:1-13`）。

## 接口与结构

- **奖励项**（`registry.py:34-69`）：`Term` 子类，暴露 `name`、`weight`、`kind`（`dense` / `terminating`）、`shaping`、`reason`，以及 `compute(step_ctx) -> float`。`compute` 返回**未加权原始值**（惩罚 ≥0；势能塑形可负），聚合器按 `weight` 加权求和。有跨步状态的项可覆写 `reset()`（episode 边界由聚合器调用）。
- **配置**（`registry.py:107-153`）：`TermConfig(enabled=False)` 单项关闭；`build_terms` 也接受扁平 dict（`name/weight/enabled/params` + 其余键并入 params）。
- **聚合**（`aggregation.py:239-356`）：`RewardAggregator.step(ctx) -> StepReward`；内部维护上一帧 `route_completion`（势能塑形）与步计数，`reset()` 清状态并调各 term 的 `reset`。
- **KPI**（`kpi.py:45` `KPI_NAMES` 等关键定义）：`KPI_NAMES` 与 `config/eval.yaml::kpis` 同名同序 12 项；`episode_kpi / compute_kpis / compute_kpis_by_primary / wilson_ci / Thresholds`。生产评测口径在 `pipeline/eval_runner.py`，本模块为同口径纯函数实现（单测/离线消费）。

## 奖励项（17 项注册；`terms.py`）

默认启用 10 项（`DEFAULT_TERM_CONFIGS`，`terms.py:819-830`）：

| 项 | kind | 默认权重 | 口径 | 主要输入键 |
| --- | --- | --- | --- | --- |
| `route_completion` | dense, shaping | +1.0 | 势能塑形 `γΦ(s')−Φ(s)`，γ=1，`Φ=clip(rc,0,1)` | `route_completion`（+聚合器注入 `route_completion_prev`） |
| `speed_ratio` | dense | +0.4 | `clip(speed_ratio, 0, cap=1.0)` 收益，防爬行 | `speed_ratio` 或 `speed`/`speed_limit_mps` |
| `low_speed` | dense | −0.2 | `clamp(1 − v/2.0, 0, 1)`（阈值与评测 KPI 对齐） | `speed`/`velocity` |
| `comfort_lon` | dense | −0.05 | 死区 2.5 后二次软罚 `max(0,\|a_lon\|−2.5)²` | `a_lon`/`accel_lon` |
| `comfort_lat` | dense | −0.05 | 死区 2.0 后二次软罚 | `a_lat`/`accel_lat` |
| `comfort_jerk` | dense | −0.005 | 死区 5.0 后二次软罚 | `jerk` |
| `solid_line` | dense | −2.0 | 压/跨连续实线（0/1）；线型 id `{2,3,6,7,8}` | `solid_line_crossing`/`crossed_solid_line`、`on_white/yellow_continuous_line`、`left/right_line_type_id` |
| `speed_limit` | dense | −5.0 | 超速量 `max(0, ratio−(1+tolerance 0.05))` | 同 `speed_ratio` |
| `crash` | terminating | −10.0 | 0/1（任一 `crash*` 或 `collision`）→ 终止 | `crash*`/`collision` |
| `out_of_road` | terminating | −8.0 | 0/1 → 终止 | `out_of_road` |

默认关闭但已注册 7 项（经 config/CLI 启用；权重为类默认值）：

| 项 | 默认权重 | 口径 | 主要输入键 |
| --- | --- | --- | --- |
| `ttc` | −0.5 | `raw = max(0, 1/max(ttc,0.5) − 1/2.0)`；`ttc = gap/max(v_ego−v_lead, ε)`；无前车/未接近 = 0（`terms.py:260-331`） | `lead_gap_m`、`lead_speed_mps`、自车速度 |
| `lead_gap` | −1.0 | `clamp((gap_ref 8 − gap)/8, 0, 1)`（等速跟车也罚，与 ttc 互补）（`terms.py:334-374`） | `lead_gap_m` |
| `comfort_jerk_win` | −0.1 | 最近 20 步 `\|jerk\|` 均值 `clamp((mean−5)/5, 0, 1)`；有状态，`reset` 清窗（`terms.py:426-483`） | `jerk` |
| `lane_center` | −0.1 | `max(0, min(\|d_lat\|,3) − 0.25)`（`terms.py:486-511`） | `d_lat` 等 |
| `lane_boundary` | −0.2 | `max(0, 0.5 − max(half_width−\|d_lat\|, 0))`（`terms.py:514-565`） | `d_lat`、`lane_half_width_m` |
| `off_road_edge` | −1.0 | BC-SAC 式 `clip(1 + d_edge/edge_scale_m, 0, 2)`；缺 `d_edge` 时等效量 `\|d_lat\|−lane_half_width_m`（`terms.py:568-609`） | `d_edge` 或 `d_lat`+`lane_half_width_m` |
| `speed_deficit` | −0.3 | 全速度区间限速缺口 `clamp(1 − v/limit, 0, 1)`（`terms.py:673-716`） | `speed`、`speed_limit_mps` |

## 聚合口径（`aggregation.py`）

- 单步奖励（`aggregation.py:334-340`）：

  `reward = dense_positive_sum × multiplier + dense_negative_sum + terminating_sum + carl_penalty + terminal_value`

  稠密和按**该帧贡献符号**拆正/负；CaRL 乘子只作用于正向部分（违规帧取消收益/塑形，但保留同帧已发生的罚分；无违规时退化为旧口径，`aggregation.py:14-28`）。
- **CaRL 规则**：默认 `crash` / `out_of_road` → `factor=0.0`（清零正向）+ `terminate=True`；规则可带 `penalty`（`aggregation.py:128-133`）。终止型项原始值 >0 即 `done=True` 并记录原因。
- **终局值机制**：终局键判定优先级 `arrive_dest → collision → out_of_road → error → max_step → terminal_reason/reason`（`aggregation.py:222-236`）；`arrive_dest/collision/out_of_road/error/max_step` 命中即置 `done`，`terminal_value` **仅终局步加一次**（`RewardAdapter` 先算奖励再 `reset`，`trainer.py:1348`）。
- **代码默认终局值**（`aggregation.py:103-109`）：`arrive_dest +30 / collision −19 / out_of_road −15 / max_step −23 / error −5`。
- **塑形退火**：`AggregationConfig.shaping_decay` 默认 `None` → 恒 1.0；可配 `ShapingDecay(kind=constant|linear|exponential, start/end/steps)`（`aggregation.py:136-173`）。
- **信用分配**：`assign_credits(mode="dense")` = 逐步原样（PPO/GAE）；`"grouped_discounted"` = 按组折扣回报广播回组内（GRPO 消融）（`aggregation.py:397-421`）。

## 当前采用配方（Stage C）

v7 P2 首臂（arm1）= **v5 bundle + `off_road_edge`**，臂文件 `config/arms/v7_arm1_offroad.yaml:27-46`（预注册 `docs/v7_program_prereg.md:127-132`）：

- terms = 默认 10 项 + `off_road_edge{weight: −0.5, edge_scale_m: 1.0}`（半量级）；
- `aggregation.terminal_values = {arrive_dest: 29, collision: −22, out_of_road: −14, max_step: −46, error: −5}`（来源 `docs/reward_audit/ebeta2/config_draft_rc1.yaml`；与代码默认不同）。

注意：`config/train.yaml` 的 `stages.C.reward` 段当前被注释；**未提供 reward 配置时** Stage C 使用 `DEFAULT_TERM_CONFIGS` + `default_terminal_values()`（+30/−19/−15/−23/−5）。若 `route_completion` 权重 ≠1 而终局值仍为 rc=1 默认，`build_reward_adapter` 会 fail-fast（rc 档配对守卫，`trainer.py:1569-1592`）。

## 上下文键（`pipeline/trainer.RewardAdapter`）

- `step_ctx = MetaDrive info + 派生量`（缺省补齐，不覆盖 info 已有键；`trainer.py:1411-1479`）：`a_lon`（优先策略步速度差，否则 obs ego 第 1 维）、`a_lat`（obs ego 第 2 维）、`jerk`（相邻策略步 `a_lon` 差分 / dt=0.5）、`speed_limit_mps`（`info["lane_speed_limit_mps"]` → obs LD slot 0 第 4 维；终局帧回退本 env 最后有效值）、`speed_ratio = velocity/speed_limit_mps`。
- 车道上下文键 `LANE_CTX_KEYS = ("lead_gap_m", "lead_speed_mps", "lane_half_width_m")`（`trainer.py:1099`，由 `lane_reward_info` 注入；不可用时置 `None` + 一次性告警）。
- `d_lat` 缺省可由 `(dist_to_left_side − dist_to_right_side)/2` 折算；`lane_half_width_m` 可由 `(dist_to_left_side + dist_to_right_side)/2` 折算（`terms.py:143-188`）。

## 配置入口与启用方式

- config `stages.C.reward.terms`（项列表，支持扁平参数）与 `stages.C.reward.aggregation.terminal_values`；CLI `--reward-term-weight NAME=W`（可重复）：已配置项改权重；已注册未配置项（如 `lane_center`）→ 追加启用；未注册 → fail-fast（`trainer.py:1595-1662`，`stages.py:4409-4419`）。
- 例：`--reward-term-weight ttc=-0.5 --reward-term-weight lane_boundary=-0.2`。

## 测试入口

`tools/venv-python -m pytest tests/test_reward_terms.py tests/test_reward_adapter_hook.py tests/test_reward_ctx_lane_keys.py tests/test_reward_ctx_speed_limit.py tests/test_rc_tier_pairing.py tests/test_reward_audit.py tests/test_arm_configs.py -q`

## 约定与坑

- `compute` 只读 `Mapping`，不 import env；缺键 / 非法值 → 0；带 `_WarnMissingInputMixin` 的项（ttc/lead_gap/comfort_jerk_win/lane_boundary/off_road_edge/speed_deficit/low_speed）**每实例只告警一次**（`terms.py:248-257`）；`lead_gap <= 0`（含 −1 哨兵）= 无前车属正常语义、不告警。
- `low_speed` 阈值 2.0 m/s 与 `pipeline.eval_runner.CRAWL_SPEED_MPS` 对齐（`terms.py:80-83`）；`solid_line` 线型 id 与 `env/obs/ld.py` 一致（2/3/6/7/8）。
- `crash` 聚合任意 `crash*` 标志；`Term.kind` 只有 `dense` / `terminating` 两种。
- 有状态项（`comfort_jerk_win`）依赖聚合器 `reset` 的 episode 边界；单独调 `compute` 时窗口会跨调用累积。
