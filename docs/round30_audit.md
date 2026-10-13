# Round-30 Reality Audit（R30-0..5；缩减版）

> 依据：`docs/deepseekv4p1_argue.md` §5.30（CG5.6 终审 `acf1b8f`：批准缩减版 Reality Audit、训练全冻结）。
> 纪律：**零训练**；只读评测 + 诊断脚本 + 文档。产物：`runs/p1_v2/round30/`、`.slim/deepwork/p1_v2/round30/`。
> 状态：**第一轮完成**（R30-0..6b 全执行 ✅；R30-5：消费审计 ✅ + 契约 v1 ✅；reward 复采 + 复现性 8/8 ✅=§1.5）。等待 round-31 裁决。本文随结果滚动更新。

---

## 0. 覆盖对账（CG5.6 终审 `acf1b8f` 要求 ↔ 执行证据）

口径：**静态**=代码级核验（C1–C7 行号证据、C4 键表、契约）；**动态**=下表各 run（全部零训练、零生产代码改动）。

| # | 要求（终审原文要点） | 状态 | 证据 |
|---|---|---|---|
| C1 | 配置契约 → `stage_a_config_consumption.md`（键级 CONSUMED/IGNORED/LEGACY/DERIVED）；删键/fail-fast=审计后动作 | ✅ 文档；修复后置 | `docs/stage_a_config_consumption.md`（70 键：51/14/2/3） |
| C2 | h1/h3/h6 独立 backward 账本（不删 detach） | ✅ | §4（三布尔 + 模块表） |
| C3 | 动作干预 probe 覆盖 future latent + physical decoder + next Policy action | ✅ | §3（9 变异 × k=1..6；三类目标全录） |
| C4 | optimizer 契约（只记录；白名单建组后置） | ✅ 记录；修复后置 | §5.2 + `stages.py:1810-1819` |
| C5 | 架构命名（文档级）：PlanHead=ContextFusion/PolicyBackbone；PolicyHead=唯一动作器；WM=唯一状态预测器；rollout 编排 | ✅ | §5.2（class 名/ckpt 键未动） |
| 结构 A | Stage B 预测态监督 → R4 三分解 ×h、分层路型/结局 | ✅（+R30-6b 超出） | §2.1–2.2、§2.5 |
| 结构 B | Stage C 四格矩阵（同 ckpt/episode；succ/off/coll/tmo/rc/reward；paired 转换） | ✅（reward 复采完成 + 复现性 8/8 一致，§1.5） | §1、`matrix_report.{json,md}`、§1.5 |
| P1 | R4：E_z/E_policy/E_action × h1..h6；报告 interface/deploy 比值 | ✅（E_z h6=0.332 补报；action 族上限 h5） | §2.1 |
| P2 | WM fidelity（ds×/dθ·κ±/zero/mirror；6 条相对门） | ✅ 实测（mirror 门构造不可判 → 重造提案） | §3 |
| P3 | 账本 4 问（共享 transition 梯度 / 早期无后段 credit / policy→WM / B 冻结） | ✅ | §4.2 |
| P4 | oracle/frozen-t0 三臂（R4 显著即上，不可 backlog） | ✅ 离线臂已执行 | §2.2 |
| R30-0 | 冻结与产物复用索引 | ✅ | §7 |
| R30-5 | 训练/部署执行契约 | ✅ v1 | §5.2 |
| 禁训 | 训练新主模型/T2 蒸馏/删 detach/BPTT/action token/DAgger/MoE/PPO·GRPO/LQR 修改/扩数据 | ✅ 全守 | 零训练；`.slim` 仅诊断脚本 |
| 反驳2 | 三类指标（表示/物理/决策充分性）；baseline/常速/oracle gap 锁门 | ◐ 实测已给；常速 gap 与门锁定=round-31 | §3、§2.5 |
| 反驳3 | 2-step canary 前置三条件（误差随 h 累积/接口误差增长/detach 结构截断） | ✅ 三条件实测满足（资格具备，未执行） | §4.2 + §2.1/§2.5 |

---

## 1. R30-1 执行栈矩阵（Priority 0）——完成 ✅

**口径**：calib50（50 episodes，确定性；`runs/p1_v2/round25/specs_calibration50.json`）。两组策略点：
- **K1e2** = `runs/p1_v2/round28/K1/ckpt_epoch002.pt`（semantic + static_role_only + kappa-compose；当前闭环最优候选）；
- **pri512** = `runs/BTC20261009-0824_sw_pri512/stage_b/ckpt_epoch020.pt`（baseline 口径；P1 起点族）。

格子 = {Exact, LQR} × {plan(chain), repeat}。

**结构判定（先读）**：exact 执行器在 0.5s 决策协议下**仅消费 `plan[0]`**（5×0.1s 子步后 re-arm，尾段被丢弃）
⇒ **exact×plan ≡ exact×repeat**（两个策略点均逐位一致）。四格实为 **3 个系统**：
- **S1 = Exact + mu**（设计原意：单步动作 + 精确执行）
- **S2 = LQR + plan**（正式部署口径）
- **S3 = LQR + repeat**（Stage C 现行训练口径）

### 1.1 每格汇总（calib50）

| 策略点 | 系统 | succ | off | coll | tmo | rc |
|---|---|---|---|---|---|---|
| K1e2 | S1 exact | **0.540** | 0.240 | 0.100 | 0.120 | 0.802 |
| K1e2 | S2 lqr+plan | **0.520** | 0.120 | 0.080 | 0.280 | 0.859 |
| K1e2 | S3 lqr+repeat | **0.020** | 0.120 | 0.020 | 0.840 | 0.657 |
| pri512 | S1 exact | **0.260** | 0.660 | 0.120 | 0.000 | 0.555 |
| pri512 | S2 lqr+plan | **0.360** | 0.500 | 0.020 | 0.140 | 0.702 |
| pri512 | S3 lqr+repeat | **0.000** | 1.000 | 0.000 | 0.000 | 0.076 |

（配对明细见 `runs/p1_v2/round30/matrix_report.md`：success 与 timeout 两套 28 对。）

### 1.2 配对要点（success；翻正=败→成）

- K1e2 **S1 vs S2**：5 翻正 / 6 翻负 / 21 双成 / 18 双败 ⇒ **平局**（26 vs 27）。
- K1e2 **S3 vs S2**：0 翻正 / 25 翻负 / 1 双成 ⇒ **严格劣**。
- pri512 **S1 vs S2**：6 翻正 / 1 翻负 / 12 双成 ⇒ exact 更差。
- pri512 **S3 vs S2**：0 翻正 / 18 翻负 / 0 双成 ⇒ **严格劣**。

### 1.3 结论（首轮，实测；阈值不另锁）

1. **"控制解耦"非免费**：S1（精确单步）对 K1e2 打平（0.54 vs 0.52），对 pri512 明显更差（0.26 vs 0.36；off 0.50→0.66）；
   失败模式系统性迁移（**tmo 降、off 升**）——"隔离规划与控制"的收益假设**未获数据支持**；
   pri512 S1 off=0.66 与 §5.26 F4（尾段×LQR 预览）×§5.28 T 族呼应："精确执行"以压线换超时。
2. **S3 与 S2 不是同一系统（严格劣、跨策略一致）**：同策略下 S3 0 翻正；K1e2 0.52→0.02、pri512 0.36→0.00（off=1.00 全灭）。
   K1e2 在 S3 下进入**低速平衡**（速度 2.9→1.8 m/s、策略 ds 需求 1.9→1.3 m、throttle 0.18、crawl 68s）——
   "repeat 参考无加速预览 ⇒ 匀速跟踪耗速 ⇒ 策略 ds 反馈收缩"假设（待 R4/机制探针复核）；
   pri512 S3 off=1.00 与 §5.5 round17 A3"确定性尾全 0/off 1.0"同族。
   ⇒ 终审"训练口径≠部署口径"指控**成立且量级严重**；Stage C 主训练口径切换（或替代路线）应列为结构级决策。
3. **边界**：两策略均非 S3 训练产物 ⇒ 本矩阵刻画"跨栈不兼容性"，不直接外推"Stage C 训练后表现"；
   现栈 S3-trained ckpt 无存活（仅 v7p2 代，跨代码代不可用）。
4. **收口判断（供裁决）**：审查提案"先确认三者差距"已兑现——S1≈S2（K1）或 S1<S2（pri512）、S3≪S2；
   下一步（Stage C 口径修复、Exact 主训练前置）依据 §5.30 执行注记 2 与 R4 结果决定。

### 1.4 产物 / 复现

- `runs/p1_v2/round30/matrix_{K1e2,pri512}_{lqr,exact}_{plan,repeat}/`（8 run；`_episodes/*.npz` + `episodes.csv` + `summary.json`）；
- `runs/p1_v2/round30/matrix_report.{json,md}`（生成器：`.slim/deepwork/p1_v2/round30/r30_matrix_report.py`）；
- 运行链：`.slim/deepwork/p1_v2/logs/round30_chain.log`；探索/jouranl：`.slim/deepwork/p1_v2/round30/journal.md`。

### 1.5 v2 复跑：reward 补采 + 复现性（2026-10-13 23:48；8 格全部 rc=0）

**动机**：`eval_runner._run_episode` 丢弃 env reward（`_, _, terminated, truncated, info = env.step(action)`）⇒ 既有记录无 reward 列。R30 在**自家 eval_a3** 内包 `env.step` 只读旁录（生产代码未动；`reward.csv` 逐 episode），8 格全部复跑（独立目录 `runs/p1_v2/round30/v2/`）。

**复现性**：**8 格 succ/off/coll/tmo/rc 与 v1 完全一致（Δ=0.000），逐 episode success 一致率 50/50 × 8** ⇒ 确定性复现通过（本协议下未见引擎漂移）。

**reward（MetaDrive 原始口径；episode 和 与 per-step）**：

| cell | succ（v1=v2） | reward_sum（succ\|fail） | per-step（succ\|fail） | steps 均（succ\|fail） |
|---|---|---|---|---|
| K1e2 lqr_plan | 0.520 | 265.57（270.72\|259.99） | 0.334（0.348\|0.318） | 796（779\|814） |
| K1e2 lqr_repeat | 0.020 | 191.80（234.61\|190.93） | 0.196（0.241\|0.195） | 980（972\|980） |
| K1e2 exact（plan=repeat） | 0.540 | 200.62（232.21\|163.54） | 0.512（0.576\|0.438） | 447（428\|471） |
| pri512 lqr_plan | 0.360 | 210.25（262.29\|180.98） | 0.330（0.353\|0.317） | 636（746\|574） |
| pri512 lqr_repeat | 0.000 | 12.11（—\|12.11） | 0.259 | 55 |
| pri512 exact（plan=repeat） | 0.260 | 127.43（221.59\|94.35） | 0.536（0.621\|0.506） | 245（385\|196） |

**口径注记（重要）**：`reward_sum` 受**时长混淆**（超时/长 episode 累积更多步：K1e2 lqr_repeat steps≈980 vs pri512 lqr_repeat≈55）⇒ 跨格比较须看 per-step 或按结局分层；即便 per-step 仍受终止结构影响（pri512 exact off=0.66/steps≈245，per-step 0.536 > 其 lqr_plan 0.330，但 succ 更低）——**reward 不可替代 succ 作唯一判据**。cell 内 succ-vs-fail 区分度（per-step 差）：K1e2 exact +0.138、pri512 exact +0.115、pri512 lqr_plan +0.036、K1e2 lqr_plan +0.030 —— reward 对结局有可测但不均匀的区分度（若作 RL 稠密信号需注意）。

**产物**：`runs/p1_v2/round30/v2/*/reward.csv`、`v2/matrix_report.{json,md}`（v2 扫描；数值与 v1 相同）。

---

## 2. R30-2 R4 rollout 一致性（完成 ✅）＋ R30-6 三臂（条件触发，已执行）

**口径**：`r4_probe.py` 部署链录制（K1e2；calib50；kappa-compose=deploy 口径；50 ep / 7,970 决策步 → 0.57 GB npz）；
离线 `r30_r4_analyze.py` 分解。产物：`runs/p1_v2/round30/r4_K1e2/`（`r4_metrics.{json,md}`）。
独立重算核验：F2 selfcheck 22/22 + 分析师抽样重算一致 + 本报告复算 E_if(k1)/E_dep(k1) 一致。

### 2.1 主表（跨 episode mean；ds=m，dθ=rad）

| metric | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| E_real ds（策略@真态 vs expert） | 0.408 | 0.407 | 0.405 | 0.404 | 0.404 |
| E_if ds（策略@WM态 vs 策略@真态） | 0.642 | 1.108 | 1.469 | 1.699 | 1.743 |
| E_deploy ds（策略@WM态 vs expert） | 0.417 | 0.828 | 1.169 | 1.385 | 1.430 |
| E_z_ego 1−cos（ẑ_k vs z_real(i+k)） | 0.203 | 0.275 | 0.322 | 0.316 | 0.316 |
| E_z_od 1−cos | 0.207 | 0.222 | 0.223 | 0.237 | 0.314 |
| E_z_ld 1−cos | 0.049 | 0.088 | 0.099 | 0.117 | 0.146 |

**补报（h6）**：E_z_ego(k=6)=**0.332**、E_z_ld(k=6)=0.160（od k=6 掩码稀疏；口径同掩码均值）；action 族上限 k=5（plan 长 6、首步 t0 已执行）。

**判定（§5.30 预注册带：interface/deploy <20% / 20–50% / >50%）**：E_if/E_deploy = **1.54（k1）→ 1.22（k5）**，
远超 50% ⇒ **"WM/预测 latent 为接口主瓶颈"成立**；条件触发 R30-6 —— 已执行（下节，离线三臂）。

### 2.2 R30-6 三臂（oracle / WM / frozen-hold；由录制数据离线计算，零训练）

误差 = |尾段动作 − expert(i+k)|（ds，m；跨 episode mean）：

| k | WM=plan[i][k] | frozen-hold=mu[i] | oracle=mu[i+k] |
|---|---|---|---|
| 1 | 0.417 | 0.415 | 0.408 |
| 2 | 0.828 | 0.416 | 0.407 |
| 3 | 1.169 | 0.426 | 0.405 |
| 4 | 1.385 | 0.441 | 0.404 |
| 5 | 1.430 | 0.451 | 0.404 |

（episode 级配对：WM 优于 frozen 的比例 k1=50% → k2=10% → k≥3=0%。）
⇒ **动作贴近度口径**：k≥2 起 WM 尾段不如 frozen-hold 且随 k 单调恶化；oracle 平坦 ≈0.40（策略自身误差下限）。
⇒ **双口径陈述（必须成对读）**：闭环 KPI 口径相反——LQR+plan 0.52 ≫ hold 系（repeat 0.02）；
"换 hold 参考"不可行（S3 改变系统），但"WM 尾段漂移"是真实接口缺陷——下一步修 WM/尾段监督，**不是**换 hold。

### 2.3 机制补充（两个独立口径）

- **尾段 ds 上冲（overshoot）**：deploy plan ds 2.56→3.64 / raw 2.82→4.02（k=1→5），而真态 mu 恒 1.94（raw 2.16）、
  expert 恒 ~2.30 ⇒ 六步计划的**速度剖面系统性上扬**。
- **δ-方向对齐（ego）**：cos(z_pred_k−z0, z_real(i+k)−z0) = +0.70/+0.58/+0.31/+0.38/+0.53
  （median 0.78/0.64/0.38/0.47/0.58）；幅度比 |δp|/|δr| median = 0.60/0.73/1.46/1.23/0.84
  ⇒ **一步方向尚可、2–3 步后方向失配 + 位移幅度膨胀**（k3 中位 1.46×）。
- 与 §5.26 F4（尾段×LQR 预览）、§5.28 T 族同源：**WM rollout 误差形态 = 方向失配 + 幅度/速度膨胀（overshoot drift）**。

### 2.4 边界与口径

- E_z/E_if 为**部署链**一致性（含重规划替换尾参考与 LQR 跟踪残差），非纯 WM 模型误差；纯 TF 消歧 = R30-6b（构建中）。
- 分层已存（路型 10 类 / 结局 4 类 / 速度 3 带；见 `r4_metrics.md`）：tollgate 的 E_real 0.85→1.30（策略误差本身最高）；
  roundabout 的 E_z_od 0.65（k5）最大。
- kappa-compose deploy 口径=下发量；`--variant raw` 对照在同文件可复算。

### 2.5 R30-6b：teacher-forced 纯 WM 消歧（完成 ✅）

目的：从 R4 部署链误差中拆出"WM 固有预测误差"。做法：专家 val 行 + **GT 专家动作链 teacher forcing**，
纯 WM 前向 k=1..5，与真实第 i+k 行编码对比（**全量 35,695 anchors**；pri512 stage_a + M0 双模型，均 0/0/0 载入；
前向自检与 `model._rollout` step0 逐位一致 max|Δ|=0；运动学 sanity canonical frac>1m=0）。

| 口径（ego 1−cos） | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| TF 纯 WM（pri512 stage_a，all） | 0.021 | 0.030 | 0.035 | 0.039 | 0.042 |
| TF 纯 WM（M0，all） | 0.032 | 0.045 | 0.052 | 0.062 | 0.074 |
| R4 部署链（K1e2） | 0.203 | 0.275 | 0.322 | 0.316 | 0.316 |

（canonical 更低：pri512 k1=0.012 / k5=0.029；pred/static 比 0.52–0.53 ⇒ WM 优于 static 基线 ~2×。）

⇒ **判定**：WM 固有 TF 误差（0.02–0.04）比部署链误差（0.20–0.32）小 **8–10×**（k1 近 10×）：
**WM 模型能力不是主瓶颈**；主瓶颈 = **部署链一致性**——策略动作/状态分布漂移（learner vs expert）+ 0.5s 重规划替换尾参考（open-loop 语义）
+ LQR 跟踪残差。R4 的"接口主瓶颈"判定应据此细化为：**想象 vs 现实的崩坏主要在链路语义，而非 WM 转移函数本身**。
局限：TF 用 pri512/M0（K1e2 为 semantic 模式、val 集无 lane_semantic 键，不能直连）；跨模型差在次要量级。

## 3. R30-3 WM 动作条件化（完成 ✅，M0 记录）

**口径**：`wm_action_probe.py`；M0 e2（baseline）；真实状态 N=200（`expert500val` 等距抽样）；**z0 固定 + pose_k 累积**
（隔离"动作条件化"，无 latent 递归漂移）；base=mu；9 变异 × k=1..6。
产物：`runs/p1_v2/round30/wm_action_probe_M0.{json,md}`（cuda，2.5s；原始 10,800 entries）。

### 3.1 关键数字（k=6，1−cos vs base；对照=跨状态自然差异）

| 变异 | 1−cos_ego(k6) | \|Δz_ego\|(k6) | 对照 |
|---|---|---|---|
| ds±20% | **1.0e-4** | 0.15–0.16 | 跨状态（同 variant）：1−cos mean **0.107** / median 0.018 / p10 0.001 |
| dθ±0.05 | 2.9–3.8e-4 | 0.31 | `‖z_ego‖`≈10.9（Δz 相对 ~1.4–3%） |
| κ±0.01 | 1.8–2.3e-4 | — | |
| zero | 3.6e-3 | 0.80 | |
| mirror | 2.2e-3 | 0.31 | |

⇒ **动作诱导 Δz 弱于状态自然差异 1–3 个数量级**；ds 通道最弱（1e-4 低于跨状态 p10=1e-3）。
"动作影响主要停留在 pose 字段"的判断**方向成立**（latent 通道增量极小；物理解码/动作头有可测但小量级响应）。

### 3.2 方向与弱信号（只报数，不设门）

- `a_next.dθ` 对 dθ+/κ+ 扰动几近**反向**（↑率 0.09–0.10）＝**修正极性**（与闭环纠偏一致）；
- `a_next.ds` 对 ds 顺序单调率 **919/1200 ≈ 77%**；
- **mirror dθ 反号率 ≈0**：**该门按现构造不可判**——只镜像动作、未镜像场景；mirror 值应检测"全场景镜像"下的
  等变性（需新变体：镜像 obs 通道 + 动作），列入 round-31 修正；
- zero：11% 状态 near-base（几乎无响应）；decode 响应 |Δod|≈0.12 / |Δld|≈0.34（原始全槽均值）。

### 3.3 判定（首轮）

1. 动作条件化**存在但弱**：latent 通道对动作的响应远小于场景/状态差异（ds 通道尤弱）；
2. "是否足够"**不由 Δz 幅度单独裁决**——由 R4（π(ẑ)≈π(z) 与部署闭环后果）裁决；
3. 审查验收门修正 2 条：①mirror ≥95% 需改为"全场景镜像"变体重造；②ds 单调门实测 ≈77%（若保留，须先议题化）。

### 3.4 产物 / 复现

- `runs/p1_v2/round30/wm_action_probe_M0.{json,md}`；探针：`.slim/deepwork/p1_v2/round30/wm_action_probe.py`；
- 正式命令：`tools/venv-python .slim/deepwork/p1_v2/round30/wm_action_probe.py --n 200 --device cuda --batch-size 100 --out runs/p1_v2/round30/wm_action_probe_M0.json`。

## 4. R30-4 梯度账本（h1/h3/h6 独立 backward；完成 ✅）

探针 `grad_ledger.py`（B=32；pri512 `stage_a/final.pt` + 该 run `model.snapshot.yaml` 严格载入 0/0/0——
`config/model.yaml` 因 primary 512/768 布局差异拒载，已记录；MoE=off 镜像 Stage A）。
产物：`runs/p1_v2/round30/grad_ledger.{json,csv}`；控制台含全部 20 组与对照。

### 4.1 模块级梯度（grad L2，B=32）

| module_group | h1 | h3 | h6 |
|---|---|---|---|
| st_gnn.spatial | 0.405 | 1.592 | 2.096 |
| st_gnn.ego_transition | 1.97e-3 | 3.42e-3 | 8.59e-3 |
| st_gnn.od_transition | 2.94e-3 | 2.62e-3 | 3.86e-3 |
| st_gnn.ld_transition | 1.49e-2 | 8.92e-3 | 7.97e-3 |
| st_gnn.od_head | 0.237 | 0.880 | 0.865 |
| st_gnn.ld_head | 0.049 | 0.055 | 0.132 |
| st_gnn.step_embed·presence·entry | 0.043 | 0.074 | 0.077 |
| plan_head.fusion | 0.052 | 0.087 | 0.158 |
| plan_head.norm | 1.69e-3 | 2.20e-3 | 4.35e-3 |
| plan_head.ego_next | 0.034 | 0.032 | 0.066 |
| plan_head.moe.primary | 0.012 | 0.011 | 0.020 |
| encoders(其余) | 0.084 | 0.030 | 0.043 |
| encoders.embed_nav | 7.55e-3 | 8.67e-3 | 1.13e-2 |
| mem_encoder | 0.012 | 0.020 | 0.028 |
| policy / value | 0 | 0 | 0 |
| **loss total / global grad** | **0.103 / 0.957** | **0.157 / 2.806** | **0.254 / 3.518** |

⇒ **共享转移头/MP 在全部 horizon 获有效梯度**（ego/od/ld transition 各 k 非零）；后段 loss 不能回传前段状态（见 4.2）。
MoE-on 对照（L_6）：experts 9.7e-3 非零（Stage A 零初始化保持），router/residual_scale=0。

### 4.2 三条布尔结论（终审要求；均实证）

1. **detach 截断=True（结构）**：`grad(L_6, state[5])`、`grad(L_3, state[2])` 均 None（h≥2 喂入为 detached 叶子）；
   `grad(L_1, state[0])` 连通。注记：h3/h6 的 encoders/mem 非零梯度来自 t0 anchor / nav 重建 / 静态上下文，**不经状态链**。
2. **policy action loss→WM=False（结构不可达）**：action-only backward 后 st_gnn 全 0/None；
   mu-扰动可达性对照仍全 0（cheap path 不跑 st_gnn，结构性不可达而非被零初始化遮挡）。
3. **Stage B WM 完全冻结=True**：`apply_freeze_prefixes(_SPECIFIC_PHASE_FREEZE)` 冻结 129 参（st_gnn 45/45 `requires_grad=False`）；
   action / action+traj backward 后 st_gnn 全 0/None；决定性对照：st_gnn 单步 WM loss 冻结=0.0 vs 未冻结=2.226。

## 5. R30-5 执行契约文档

- ✅ `docs/stage_a_config_consumption.md`（键级 70 行：CONSUMED 51 / IGNORED 14 / LEGACY 2 / DERIVED 3；
  重点：`world_model.loss` 全仓零读取且 metrics 硬编码旧名；`trainable` 未消费（死参）；`stages.C.kl_anchor*` YAML 零消费；
  `epochs/batch/micro` 双真相；`--plan-noise` metrics 不回写开关）。
- ✅ **执行契约（口径表 v1，R30 结论）**：

| 系统 | 定义 | 角色 | 证据 |
|---|---|---|---|
| S1 Exact+mu | action_mu 精确运动学执行；**exact 每决策仅消费 plan[0]**（5 子步后 re-arm，尾段丢弃） | 设计原意（单步） | §1（F1 结构判定 + 双策略点逐位复现） |
| S2 LQR+plan | 6 步 plan → LQR 预览跟踪 | 部署/评测主口径 | §1 |
| S3 LQR+repeat | repeat(a_t,6) 参考 + LQR | Stage C 收集/训练现行 | §1（严格劣：跨策略 0 翻正） |
| Stage A 图 | 6 步 free-running latent 输入 + 每步局部监督 + **跨步状态链截断（固定）**；h≥2 输入=自预测 | Stage A 训练 | §4 |
| Stage B 监督 | 首步动作回归 + traj6（0.1）+ 冻结 WM rollout；尾 5 步无直接动作标签 | Stage B | §4 + argue §5.29 |
| Stage C 更新 | policy/value + MoE experts/residual_scale（design allowlist）；WM 全冻 | Stage C PPO | §4 |

  声明未消费键清单（冻结期只记录、后置修复）：`world_model.loss`、`world_model.trainable`、`stages.C.kl_anchor*`、
  `stages.C.tracker=mpc_lqr`（非法取值）等——见 §5.1 文档逐键表与建议。
  旁记（R30-2 录制时发现）：`others` 通道 road_class one-hot 在 **First block 恒为 0**（与 obs 通道实现一致）；
  R30 起路型分层改用 block_id 映射（S/C/R/O/I→straight/curve/ramp/roundabout/first），消费该通道的下游需注意。
- **命名（C5，文档级）**：PlanHead=历史命名（职责=ContextFusion/PolicyBackbone）；PolicyHead=唯一动作输出器；
  World Model=唯一状态预测器；rollout=编排 Policy×WM（class 名/ckpt 键冻结不改）。

## 6. 条件触发 / 下一步（更新）

- ✅ **R30-6 已按预注册触发并执行**（离线三臂，§2.2）；✅ **R30-6b 消歧完成**（§2.5）：
  - R4 预注册带判定："WM 接口=主瓶颈"成立（E_if/E_deploy 1.54→1.22）；
  - TF 消歧把根因细化为：**WM 固有能力 OK（0.02–0.04）；崩坏在部署链语义（0.20–0.32，8–10×）**。
- **round-31 候选裁决单（供用户/CG 复核）**：
  1. **链路一致性修复（新主问题）**：
     a. 训练分布侧：learner-state / DAgger 类数据（策略动作与状态入训；覆盖 WM/Stage-B 尾段监督候选）；
     b. 协议侧：重规划 cadence 与"参考尾段消费语义"审计（exact=仅 plan[0]；LQR 预览取点；每 0.5s 替换尾段）；
     c. 监督侧：overshoot drift 直接指标化（尾段 ds/几何一致性项）后预注册。
  2. **Stage C 执行契约**：S3≪S2（结构级）⇒ 键级 fail-fast + 口径重选；Exact 主训练须过 §5.30 注记 2 前置。
  3. **工程尾巴**：R30-3 mirror 门重造（全场景镜像）；配置 fail-fast/删键批次（§5.1 清单）；exact 参考消费语义文档化。
- 结论入口：本文 + `docs/stage_a_config_consumption.md`；产物索引见 `runs/p1_v2/round30/` 与 `.slim/deepwork/p1_v2/round30/journal.md`。

---

## 7. R30-0 产物复用索引（冻结与复用清单）

- **评测协议**：calib50 = `runs/p1_v2/round25/specs_calibration50.json`（50 ep、确定性；与 R28/R29 同协议）；eval 栈=`eval_a3.py`（R30 增 `--tracker {lqr,exact}`，默认 lqr 零行为变化；v2 增 env reward 旁录 `reward.csv`，不动生产代码）。
- **ckpt 复用**：K1e2=`runs/p1_v2/round28/K1/ckpt_epoch002.pt`（semantic+static_role_only+kappa-compose）；pri512=`runs/BTC20261009-0824_sw_pri512/stage_b/ckpt_epoch020.pt`（baseline）；M0=`runs/p1_v2/round22/M0/ckpt_epoch002.pt`；pri512 stage_a=`runs/BTC20261009-0824_sw_pri512/stage_a/final.pt`（+run 旁 `model.snapshot.yaml` 严格载入）。
- **数据复用**：`datasets/BTC20261007-2202_expert500val_v8`（R4/WM/TF 探针）；`datasets/BTC20261007-2202_expert5k_v8`（账本装配）；`datasets/BTC20260926-2343_expert5k`（候选）。
- **工具复用**：`tools/diagnostics/forensics_closed_loop.py`（lane_lat/road_class 口径）；`tools/dagger_collect._make_labeler`（专家空问）；`pipeline.eval_runner`（控制器/汇总/CSV）；`pipeline.stages`（FrameWindows 等 Stage A 装配）。
- **本轮新增（`.slim/deepwork/p1_v2/round30/`，gitignore 工作区）**：`{r4_probe, r30_r4_analyze, wm_action_probe, grad_ledger, wm_tf_check, r30_matrix_report}.py` + `run_r30_*.sh` + journal/monitor。
- **冻结**：无训练、无生产代码改动；`runs/` 只增不改（v2 独立目录 `runs/p1_v2/round30/v2/`）。
