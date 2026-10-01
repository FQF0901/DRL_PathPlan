# 奖励 v5 规格（剖面 C）· 冻结

> **状态：冻结（2026-10-01；Gate0 复核修正版 + P2 审计回填 + Gate2 P4 前置-A 修订）**。P2 实现与 Gate2 审计以本文件为准。**终局值表（§1）已由 P2 审计重解定稿**（原临时值保留对照，见 §1/§3）。
> 关联：[`docs/v6_program_prereg.md`](v6_program_prereg.md)（P2/Gate2；E-β″ 复算规格 §7.1）、[`docs/v6_net_design.md`](v6_net_design.md)（新基座）。
> 上游：`.slim/deepwork/v6-net-retrain.md`（剖面 C 已锁）；反解输入 = E-β′ 1000 episode 真实 rollout（P2 审计，见 §3；原 50 场景重建临时输入作废）。
> P2 审计报告（入库）：`docs/reward_audit/reward_audit.{md,json}`（+ `config_draft_rc{1,3,10,30}.yaml`；原始 `runs/reward_audit/report/` 路径 + sha256 对照见 `docs/reward_audit/MANIFEST.md`；验收镜像 `/tmp/opencode/v6_reward_audit.md`）。

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

## 1. 终局值（剖面 C；**P2 审计重解定稿**）

**基准档（rc 权重 = 1.0；代码默认档）定稿值**（落点：`reward_model/aggregation.py::default_terminal_values`）：

| 终局 | 旧默认 | 临时值（provisional） | **定稿（审计反解）** | 与临时值差 |
|---|---|---|---|---|
| `arrive_dest` | +10 | +31 | **+30** | −1 |
| `collision` | −5 | −17 | **−19** | −2 |
| `out_of_road` | −5 | −11 | **−15** | −4 |
| `max_step` | −2 | −19 | **−23** | −4 |
| `error` | −5 | −5 | **−5**（保留） | 0 |

- 反解输入 = E-β′ 1000 episode 真实 rollout（`env/specs/scenarios_val.json`；pre-v6 快照 `031cc1c`），
  HEAD 完整 v5 项集（含 `low_speed`）离线重放；**审计样本 501 / 反解样本 499（互斥）**；
  基准档反解复算差 max **0.494 ≤ 0.5**（报告 §3）。
- **临时值说明（已作废）**：Gate0 的 50 场景重建（无归档）仅作待验证对照；P2 审计（完整项集、真实
  聚合器）已重解定稿，**临时值不再作为任何档依据**。
- **每档独立**：`route_completion` 扫档 3 / 10 / 30 各自反解一套终局值（§3），**不得跨档复用**：

| 档 | `arrive_dest` | `collision` | `out_of_road` | `max_step` | `error` | 剖面 |Δ|max |
|---|---|---|---|---|---|
| rc=1.0（基准/代码默认） | +30 | −19 | −15 | −23 | −5 | 3.66 |
| rc=3 | +28 | −20 | −16 | −24 | −5 | 3.57 |
| rc=10 | +22 | −23 | −18 | −29 | −5 | 2.76 |
| rc=30 | +2 | −33 | −26 | −42 | −5 | 1.88 |

- 配置草案（入库）：`docs/reward_audit/config_draft_rc{1,3,10,30}.yaml`（原始
  `runs/reward_audit/report/` 副本 + sha256 见 `docs/reward_audit/MANIFEST.md`；经
  `stages.C.reward.aggregation.terminal_values` 或对应 arm 配置启用）。
- **E-β″ 基座复算备注（P4 前置-C；2026-10-01）**：v6 基座（ckpt `723db1c2…`）四档定稿值见
  `docs/reward_audit/ebeta2/`——rc=1 **+29/−22/−14**、rc=3 +27/−23/−15、rc=10 +20/−28/−18、
  rc=30 +0/−41/−26（非 max_step 类两向 |Δ|≤5）；**max_step 采用 E-β″ 点估计
  −46/−48/−54/−71**（n=8/向、不判通过；裁定 + **horizon 对齐 = 训练截断 200 策略步
  （=100 s，与审计/评测一致；旧 600=300 s=3×）** 见 `docs/v6_program_prereg.md` §7.1/§7.4）。
  P4 臂配置从 `config/arms/`（源：
  `docs/reward_audit/ebeta2/config_draft_rc*.yaml`）加载。
- **训练侧 max_step 接线（P4 前置-B，2026-10-01 已实现 + 测试）**：截断（env timeout /
  `max_episode_steps`）且 info 无终局键时注入 `max_step=True`（`LocalEnvPool.step` 入
  `_record` 前；Vector 路径在 `collect_rollout` 统一兜底）→ 奖励侧 `terminal_key=max_step`、
  终局值按档结算（rc=1 定稿 −23），与监视口径 `episode_termination_reason` 一致；修复前
  truncated 不注入 ⇒ `terminal_value=0`（"超时"变正收益）。测试：`tests/test_max_step_wiring.py`
  （含"不接线则 0"对照）。
- **rc 档配对守卫（P4 前置-B，2026-10-01 已实现 + 测试）**：`route_completion` 权重 ≠ 1
  （config terms / CLI `--reward-term-weight`）而 `aggregation.terminal_values` 缺省或仍为
  rc=1 默认 → `build_reward_adapter` fail-fast（跨档复用，Gate2 发现④）；P4 各臂配置必须从
  `docs/reward_audit/config_draft_rc*.yaml` 同档加载。测试：`tests/test_rc_tier_pairing.py`。

## 2. 权重

| 项 | v5 | 现行 | 说明 |
|---|---|---|---|
| `speed_ratio` | **0.4** | 1.0 | 唯一改动权重的保持项 |
| `route_completion` | **扫档 3 / 10 / 30** | 1.0 | **固定剖面、终局值为因变量**：每档重解终局值（§3）；档间单变量 |
| 其余 | 保持 | — | §5 |

- 代码默认（`DEFAULT_TERM_CONFIGS`）保持 rc=1.0（基准档，§1 定稿值配套）；P4 各臂经
  config `stages.C.reward.terms` / CLI `--reward-term-weight route_completion=<档>` 切档，
  并加载该档的 `terminal_values`（§1 表 / 配置草案）。

## 3. `route_completion` 档位反解（程序）

**反解式**：`终局值 = 目标总量 − 稠密贡献 − 终止项`

- **目标剖面 C（固定，不随档变）**：成功 **+50** / 碰撞 **−20** / 出界 **−15** / 超时 **−10**（`error` 不在剖面内，终局值 −5 保留）。
- **档位语义（Gate0 裁定）**：rc 扫档 = **固定剖面、终局值为因变量**——目标剖面 C 不随档变，每档只改变 `route_completion` 权重并**重解该档终局值**；档间比较必须同档配对。
- **反解输入状态（P2 已执行）**：Gate0 的 50 场景重建**无归档**、只作对照；实际反解由 P2 审计工具
  （`tools/reward_audit.py`）以**完整 v5 项集（含 `low_speed`）**在真实聚合器上离线重放完成：
  E-β′ 1000 episode（`scenarios_val.json`；pre-v6 快照 `031cc1c`；rollout 与评测同口径），
  **反解样本 499 条与审计样本 501 条互斥**（同类内交替分配；id/seed/ctx_sha256/文件 sha256 见报告 §1）；
  **dense 语义**（`dense_positive_sum × carl_multiplier + dense_negative_sum` vs 正稠密）已拆开报告（报告 §4）。
- **临时输入摘要（已作废，仅存档对照）**：稠密贡献 成功 +18.88 / 碰撞 +7.45 / 出界 +4.31 / 超时 +9.23；
  终止项 0 / −10 / −8 / 0。
- **基准档（rc=1.0）反解**（P2 实测，反解样本 n = 220 / 17 / 245 / 17）：

| 终局 | 目标 | 稠密贡献（effective） | 终止项 | 反解 raw | 定稿（整数） | 复算 | 复算差 |
|---|---|---|---|---|---|---|---|
| 成功 | +50 | +19.591 | 0 | +30.409 | **+30** | +49.591 | −0.409 |
| 碰撞 | −20 | +8.846 | −10 | −18.846 | **−19** | −20.154 | −0.154 |
| 出界 | −15 | +7.705 | −8 | −14.705 | **−15** | −15.295 | −0.295 |
| 超时 | −10 | +12.506 | 0 | −22.506 | **−23** | −10.494 | −0.494 |

- 复算差冻结容差 **≤ 0.5**（P2 实测 max 0.494）。
- **每档（3 / 10 / 30）以完整 v5 项集重跑反解样本**（与审计样本互斥；rc 权重改变稠密贡献），
  按同式反解产出该档终局值；**不得复用其他档的终局值**。各档反解表/剖面见报告 §2/§3。
- 执行/归档：`tools/reward_audit.py analyze` 以真实聚合器复算并落 `runs/reward_audit/report/`
  （含样本清单与 sha256）；P2 报告/配置草案已入库 `docs/reward_audit/`（原始路径 + sha256 对照见
  `docs/reward_audit/MANIFEST.md`）。

## 4. 新项 `low_speed`（P2 已实现，默认启用）

- 规格：自车速度 **v < 2 m/s** 时，每策略步 **−0.2 × (1 − v/2)**；v ≥ 2 m/s → 0。
  - 原始值 `raw = clamp(1 − v/2, 0, 1)`；权重 **−0.2**（dense；默认启用 = 剖面 C 的一部分）。
- 口径：速度取 `reward_model/terms.py::ego_speed_from_ctx`（`speed` → `velocity`）；阈值与评测 KPI `CRAWL_SPEED_MPS = 2.0`（`pipeline/eval_runner.py`）对齐。
- 缺键：按现有 `_WarnMissingInputMixin` 约定告警一次并记 0。
- 单测：v=0 → −0.2；v=1 → −0.1；v=2 → 0；v>2 → 0；缺键 → 0 + 告警一次。
- **P2 审计实测（E-β′，审计样本 rc=1）**：触发帧 arrive 1041 / collision 734 / out_of_road 599 /
  max_step 1551；类均贡献 −0.292 / −4.465 / −0.204 / −6.866；按有无前车拆分（报告 §5）：
  罚分集中在**有前车**步（如 max_step −6.574 vs 无前车 −0.291）——低速主要发生在跟车/拥堵，
  符合"防蠕动"意图，不额外惩罚静止让行（无前车低速占比小）。

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
  **默认排除 `env/specs/scenarios_eval500.json`**（collect 池过滤 + analyze 兜底过滤；`--no-exclude` 仅显式关闭；
  `--exclude ""` 报错——空值曾导致 P2 池与 eval500 全量重叠）。
- **样本纪律（Gate0 裁定；Gate2 P4 前置-A 修订）**：
  - **样本与反解互斥**：审计样本（验证剖面）与 rc 档反解样本（产出终局值）必须来自**互斥**的场景集合（同一批不得既反解又审计）；划分须**按终局类分层随机**（`--split-seed` 固定记录；P2 的按序号交替两半难度不可交换，已作废），并以 `--swap-ab` 做 A/B 互换交叉验证（见 §7.1）；
  - **每终局类 n ≥ 10**（成功 / 碰撞 / 出界 / 超时 / error；不足则该类只报 n 与区间，不做达标判定）；E-β″ 复算提高到 **≥ 50**（§7.1）；
  - **完整项集**：含 `low_speed`（不得用旧项集近似）。
- 输出：`reward_audit.{md,json}`：
  - 报告 meta：**HEAD commit sha**、split seed（`--split-seed`）、exclude 集、每类最小样本数（`--min-per-class`，E-β″ 复算用 50）；
  - 每终局类：n、total reward 均值、`dense_positive_sum` / `dense_negative_sum`、terminating、carl_penalty、terminal_value；
  - **分类剖面 vs 目标**（+50 / −20 / −15 / −10）与 |Δ|；逐项贡献 top（`speed_ratio` / `route_completion` / `low_speed` / `comfort_*` / `solid_line` / `speed_limit`）；
  - **分层报告**：① **折扣 / 未折扣**两列（γ 折扣回报 vs 原始累计）；② **dense 语义拆分**（`pos×mult+neg` 聚合 vs 正稠密口径）；③ `low_speed` 按**有无前车**拆分；④ **E-β′ vs E-β″ 两列**（新旧基座各一列）；
  - **rc 各档重算**（3 / 10 / 30）：完整项集下的稠密贡献重算与终局值反解由同一工具完成（§3）；
  - 违规帧统计、CaRL 乘子触发帧、终局帧掩码回退命中。
- 判定：每类 **|剖面 − 目标| ≤ 容差**（P2 预注册冻结；建议绝对 ≤ 5）。反解复算差单独冻结 **≤ 0.5**。
- **执行时点**：Gate2 一次（重解定稿终局值 + 剖面审计）；**P3 后、P4 前再跑一次**（新基座 E-β″ 上重解 + 复算，作为 P4 奖励口径）。
- **执行记录（P2 / Gate2，2026-10-01）**：
  - 采集：E-β′（`runs/_refs_rlbase/e_beta_prime/final.pt`，sha256 `c9e2d31e…`）在 pre-v6 快照
    `031cc1c`（`/tmp/opencode/v6_pre`）下 rollout **1000 episode**（`scenarios_val.json`；分类
    arrive 441 / out_of_road 490 / collision 34 / max_step 35；rollout 与评测同口径）；
  - 重放/反解：HEAD `tools/reward_audit.py analyze`（完整 v5 项集含 `low_speed`）；审计 501 /
    反解 499（互斥）；报告 `docs/reward_audit/reward_audit.{md,json}`（原始 `runs/` 副本 + sha256 见
    `docs/reward_audit/MANIFEST.md`）+ 4 份配置草案；
  - 判定：**四档（rc=1/3/10/30）剖面均通过**（|Δ|max 3.66 / 3.57 / 2.76 / 1.88 ≤ 5；每类 n ≥ 10：
    audit collision 17 / max_step 18）；基准档反解复算差 max 0.494 ≤ 0.5；
  - **池重叠声明（Gate2 P4 MUST ①）**：P2 审计池 = `scenarios_val.json` 1000 条，与
    `scenarios_eval500.json` **重叠全部 500 条**（采集时 `--exclude` 因空值静默失效，
    `collect_summary.exclude=null`）⇒ 本报告含 eval500 场景。**E-β″ 复算将排除 eval500**（工具默认
    排除；空值报错；analyze 侧兜底过滤），并执行 §7.1 完整规格（分层随机划分 + A/B 互换 + 每类 n≥50 +
    max_step 接线后 + 记录 HEAD commit sha/split seed）。
  - E-β″ 列：P3 重训后补（同一工具按 §7.1 规格复算；见 §7 执行时点）。

## 8. 验收清单（P2 / Gate2）

- 单测：聚合公式（含"无违规帧逐位不变"）、终局值表（临时值→P2 重解定稿）、`low_speed` 边界、审计工具 dry-run（小样本 n ≥ 2）。✅（`tests/test_reward_terms.py`、`tests/test_reward_audit.py`、`tests/test_stage_c_lam.py`）
- 审计：≥ 50 rollout 的分类剖面 vs 目标在容差内（每类 n ≥ 10；样本与反解互斥）；否则不得进入 P3（Gate2 判据，见 [`docs/v6_program_prereg.md`](v6_program_prereg.md) §3）。✅（1000 episode；四档全通过）
- **复算（P3 后、P4 前）**：在 E-β″ 上重跑同一审计/重解，产出 P4 奖励口径；未复算不得开 P4 臂。
