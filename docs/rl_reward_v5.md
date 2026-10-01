# 奖励 v5 规格（剖面 C）· 冻结

> **状态：冻结（2026-10-01；Gate0 复核修正版）**。P2 实现与 Gate2 审计以本文件为准。**终局值表（§1）为临时值（provisional），以 P2 重解定稿为准。**
> 关联：[`docs/v6_program_prereg.md`](v6_program_prereg.md)（P2/Gate2）、[`docs/v6_net_design.md`](v6_net_design.md)（新基座）。
> 上游：`.slim/deepwork/v6-net-retrain.md`（剖面 C 已锁）；反解输入 = 50 场景重建（基座行为；**无归档**，见 §1/§3）。

## 0. 聚合结构（不变式）

```
reward = dense_positive_sum × carl_multiplier
       + dense_negative_sum
       + terminating_sum
       + carl_penalty
       + terminal_value
```

- 每 **0.5 s 策略步**结算一次；稠密项按该帧贡献符号拆分（实现与理由见 `reward_model/aggregation.py` 模块 docstring）。
- **CaRL 已修**：违规帧只清零**正向**稠密和，负向罚分保留（旧式 `dense_sum × 0` 会抹掉同帧 `solid_line` 等已发生代价）——commit `514a834`，见 `runs/reward_viz/README.md` §0.1（#1）与 `docs/rl_stage_c_v4_report.md` §0 ⑥。
- 不变量：无违规帧（`multiplier == 1`）时逐位退化为旧口径。

## 1. 终局值（剖面 C；**临时值（provisional）**，P2 重解后定稿）

| 终局 | 旧默认 | v5 临时值 |
|---|---|---|
| `arrive_dest` | +10 | **+31** |
| `collision` | −5 | **−17** |
| `out_of_road` | −5 | **−11** |
| `max_step` | −2 | **−19** |
| `error` | −5 | **−5**（保留） |

- 落点：`reward_model/aggregation.py::default_terminal_values`（旧默认见该函数；本表整体替换）。
- **临时值说明（Gate0 裁定）**：本表反解输入（50 场景重建）**无归档** ⇒ 全表标记 **临时值（provisional）**；由 P2 审计工具以**完整 v5 项集（含 `low_speed`）**在真实聚合器上重放重解后定稿（§3/§7）；重解前不得作为 Gate2 定稿依据、不得跨档复用。
- 注：终局值随 `route_completion` 权重档重解（§3）；本表 = 基准档（§3 反解输入口径），定稿以 P2 重解结果为准。

## 2. 权重

| 项 | v5 | 现行 | 说明 |
|---|---|---|---|
| `speed_ratio` | **0.4** | 1.0 | 唯一改动权重的保持项 |
| `route_completion` | **扫档 3 / 10 / 30** | 1.0 | **固定剖面、终局值为因变量**：每档重解终局值（§3）；档间单变量 |
| 其余 | 保持 | — | §5 |

## 3. `route_completion` 档位反解（程序）

**反解式**：`终局值 = 目标总量 − 稠密贡献 − 终止项`

- **目标剖面 C（固定，不随档变）**：成功 **+50** / 碰撞 **−20** / 出界 **−15** / 超时 **−10**（`error` 不在剖面内，终局值 −5 保留）。
- **档位语义（Gate0 裁定）**：rc 扫档 = **固定剖面、终局值为因变量**——目标剖面 C 不随档变，每档只改变 `route_completion` 权重并**重解该档终局值**；档间比较必须同档配对。
- **反解输入状态（Gate0 裁定）**：50 场景重建**无归档**（仅上游档有摘要数字）⇒ 现表（§1）为**临时值**。反解必须由 P2 审计工具以**完整 v5 项集（含 `low_speed`）**在真实聚合器上重放重解（稠密贡献随 rc 权重与项集变化），结果落 `runs/` 后才定稿；**重解样本与审计样本互斥**（§7）；**dense 语义（`dense_positive_sum × carl_multiplier + dense_negative_sum` vs 正稠密）在审计中拆开报告**。
- **临时输入摘要（仅作待验证对照，不作定稿依据）**：
  - 稠密贡献：成功 **+18.88** / 碰撞 **+7.45** / 出界 **+4.31** / 超时 **+9.23**；
  - 终止项：成功 **0** / 碰撞 **−10** / 出界 **−8** / 超时 **0**（来源 `reward_model/terms.py::DEFAULT_TERM_CONFIGS`：crash −10 / out_of_road −8）。
- **基准档反解**（与 §1 临时值一致；P2 重解后替换）：

| 终局 | 目标 | 稠密 | 终止 | 反解 | 复算差 |
|---|---|---|---|---|---|
| 成功 | +50 | +18.88 | 0 | 50 − 18.88 − 0 = **+31.12 → +31** | 0.12 |
| 碰撞 | −20 | +7.45 | −10 | −20 − 7.45 + 10 = **−17.45 → −17** | 0.45 |
| 出界 | −15 | +4.31 | −8 | −15 − 4.31 + 8 = **−11.31 → −11** | 0.31 |
| 超时 | −10 | +9.23 | 0 | −10 − 9.23 − 0 = **−19.23 → −19** | 0.23 |

- 复算差冻结容差 **≤ 0.5**（P2 重解同口径）。
- **每档（3 / 10 / 30）必须以完整 v5 项集（含 `low_speed`）重跑反解样本**（≥ 50 条、与审计样本互斥；rc 权重改变稠密贡献），按同式反解产出该档终局值；**不得复用其他档的终局值**。
- 执行/归档：P2 审计工具（§7）以真实聚合器复算并落 `runs/`（含样本清单与 sha256）。

## 4. 新项 `low_speed`（需实现）

- 规格：自车速度 **v < 2 m/s** 时，每策略步 **−0.2 × (1 − v/2)**；v ≥ 2 m/s → 0。
  - 原始值 `raw = clamp(1 − v/2, 0, 1)`；权重 **−0.2**（dense；默认启用 = 剖面 C 的一部分）。
- 口径：速度取 `reward_model/terms.py::ego_speed_from_ctx`（`speed` → `velocity`）；阈值与评测 KPI `CRAWL_SPEED_MPS = 2.0`（`pipeline/eval_runner.py`）对齐。
- 缺键：按现有 `_WarnMissingInputMixin` 约定告警一次并记 0。
- 单测：v=0 → −0.2；v=1 → −0.1；v=2 → 0；v>2 → 0；缺键 → 0 + 告警一次。

## 5. 臂（默认关；P4 单变量）

| 项 | 权重 | 参数 | 现状 |
|---|---|---|---|
| `ttc` | −0.5 | threshold 2.0 s / floor 0.5 s | 已实现（`f619ac2`/`71712d8`；`reward_model/terms.py::TTCLeadPenalty`） |
| `lane_boundary` | −0.2 | margin 0.5 m | 已实现（`LaneBoundaryPenalty`） |
| `lane_center` | −0.1 | deadband 0.25 m | 已实现（`LaneCenterPenalty`） |

- 默认**不进** `DEFAULT_TERM_CONFIGS`；P4 作为预注册臂开启（见 [`docs/v6_program_prereg.md`](v6_program_prereg.md) §7.1；两文档臂集一致：`ttc`/`lane_boundary`/`lane_center`）。
- **`ttc` 臂前置（Gate0 裁定）**：先**离线证伪**——用已归档 collision/off_road episodes 重算 would-be 触发率；若 ≈ 0（当前证据：id680 碰撞前 `ttc`=0，前车 29 m）则改项或不做，不得直接开臂。
- 臂依据（would-be 量化）：`runs/reward_viz/INDEX.md`（id179 `lane_center` ≈ −10.7；id31 `lane_boundary` 10/69 策略步；id680 碰撞前 `ttc`=0）。

## 6. 保持项与已修项

- **保持**（`DEFAULT_TERM_CONFIGS`；除 §2 权重外参数不变）：`comfort_lon` −0.05 / 2.5 m/s²、`comfort_lat` −0.05 / 2.0、`comfort_jerk` −0.005 / 5.0、`solid_line` −2.0、`speed_limit` −5.0（tolerance 0.05）、`crash` −10、`out_of_road` −8、`route_completion` γ=1.0 势能塑形、`speed_ratio` cap 1.0。
- **已修（v4；直接引用，勿回退）**：
  - 限速源对齐 KPI（`6ab70b3` ⑥a；`runs/reward_viz/README.md` §0.1 #2）；
  - 终局帧掩码回退（`6ab70b3` ⑥b；§0.1 #3）；
  - ctx 新键 `lead_gap_m` / `lead_speed_mps` / `lane_half_width_m`（`3268cc7` ⑦）；
  - CaRL 正向清零（`514a834`；§0.1 #1）。
  - 汇总：`docs/rl_stage_c_v4_report.md` §0、`runs/reward_viz/README.md` §0.1、`runs/reward_viz/INDEX.md`。

## 7. 奖励审计工具规格（Gate2；P3 后/P4 前复算）

- 工具：`tools/reward_audit.py`（名 P2 定，接口冻结）。
- 输入：ckpt + spec + 场景数 **≥ 50**（真实 rollout）；**真实** `RewardAggregator` + `RewardAdapter._build_ctx`（禁用 `implied_reward.py` 类近似）。
- **样本纪律（Gate0 裁定）**：
  - **样本与反解互斥**：审计样本（验证剖面）与 rc 档反解样本（产出终局值）必须来自**互斥**的场景集合（同一批不得既反解又审计）；
  - **每终局类 n ≥ 10**（成功 / 碰撞 / 出界 / 超时 / error；不足则该类只报 n 与区间，不做达标判定）；
  - **完整项集**：含 `low_speed`（不得用旧项集近似）。
- 输出：`reward_audit.{md,json}`：
  - 每终局类：n、total reward 均值、`dense_positive_sum` / `dense_negative_sum`、terminating、carl_penalty、terminal_value；
  - **分类剖面 vs 目标**（+50 / −20 / −15 / −10）与 |Δ|；逐项贡献 top（`speed_ratio` / `route_completion` / `low_speed` / `comfort_*` / `solid_line` / `speed_limit`）；
  - **分层报告**：① **折扣 / 未折扣**两列（γ 折扣回报 vs 原始累计）；② **dense 语义拆分**（`pos×mult+neg` 聚合 vs 正稠密口径）；③ `low_speed` 按**有无前车**拆分；④ **E-β′ vs E-β″ 两列**（新旧基座各一列）；
  - **rc 各档重算**（3 / 10 / 30）：完整项集下的稠密贡献重算与终局值反解由同一工具完成（§3）；
  - 违规帧统计、CaRL 乘子触发帧、终局帧掩码回退命中。
- 判定：每类 **|剖面 − 目标| ≤ 容差**（P2 预注册冻结；建议绝对 ≤ 5）。
- **执行时点**：Gate2 一次（重解定稿终局值 + 剖面审计）；**P3 后、P4 前再跑一次**（新基座 E-β″ 上重解 + 复算，作为 P4 奖励口径）。

## 8. 验收清单（P2 / Gate2）

- 单测：聚合公式（含"无违规帧逐位不变"）、终局值表（临时值→P2 重解定稿）、`low_speed` 边界、审计工具 dry-run（小样本 n ≥ 2）。
- 审计：≥ 50 rollout 的分类剖面 vs 目标在容差内（每类 n ≥ 10；样本与反解互斥）；否则不得进入 P3（Gate2 判据，见 [`docs/v6_program_prereg.md`](v6_program_prereg.md) §3）。
- **复算（P3 后、P4 前）**：在 E-β″ 上重跑同一审计/重解，产出 P4 奖励口径；未复算不得开 P4 臂。
