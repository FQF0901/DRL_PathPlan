# v5+K-anchor 链失败根因诊断（只读 + 评测）

- 日期：2026-10-05 · 仓库 HEAD `0065264a87f79f84d0355743bb2e325b1888ff5f`（工作区干净，未改 repo；配置副本/评测输出全在 `/tmp/opencode/v7_fail_diag/`）
- 诊断对象：
  - Stage B：`runs/BTC20261005-0856_v7struct_v5/stage_b/ckpt_epoch010.pt`（sha `2f463a21…`）
  - phase3：`runs/BTC20261005-0856_v7struct_v5_p3/stage_b/ckpt_epoch005.pt`（sha `842b8e34…`）
- 评测口径：clean150（`/tmp/opencode/phase3_diag/exp/specs_val_only150.json`）、`--workers 6 --tracker lqr --device cuda`；anchor 开关配置副本 `/tmp/opencode/v7_fail_diag/v7_struct_b_v5_eval_noanchor.yaml`（仅 `plan_anchor.enabled: false`）。

## 0. 结论（TL;DR）

| 问题 | 判定 |
|---|---|
| E1（关锚→走 rollout）是否恢复？ | **否**。b10：0.0067（锚开 0.26）；p3：0.0067（锚开 0.0）。按预设判据字面 → **不是"K-anchor plan 路径单独"的根因**；但该实验有重要混淆（见 §1.3）：关锚后尾段由**从未被训练过**的 policy rollout 驱动，0.007 是"地板"而非"健康路径"。 |
| K-anchor plan 输出是否系统性错误？ | **有两层问题**：(a) 结构层：`Δψ` 只进入 `dθ_0`，而 `plan[:,0]` 被 `action_mu` 覆盖 → **被执行的后 5 步（tracker 参考尾段）对航向误差 `Δψ` 完全无反馈**；锚选择头的输入 latent 也不含 lane token（plan_head 融合只有 ego/OD/LD/others/nav/signal）。(b) phase3 训练层：锚头被误训（与预注册 §11 相反，见 §3）→ 锚计划系统性左偏（隐含车道横向偏移 mean +0.91 m / |·| 1.98 m，70% 决策 >1 m）；叠加被误训的 specific 专家/router 对 latent 的漂移 → 496/500 out_of_road。 |
| phase3 0.0 的根因 | **phase3 specific_only 微调产生两个独立缺陷，各自足以致崩**：① **被训练的 specific 专家/router**（决定性）——p3+b10锚头 MoE 开=0.013、MoE 关=0.260（与 b10 逐 spec 一致）→ 专家路径把 0.26 打到 0.013；② **锚头漂移**（契约缺口：预注册 §11 要求 phase3 冻结锚头，但冻结清单漏掉 4 个前缀，锚头落入 base LR 组被训练，且 CE/WTA=0 仅 chain L2）——b10+p3锚头=0.0。两缺陷叠加 → 实际 p3=0.0。 |
| obs/数据是否有问题 | **未发现异常**：lane 块符号/量级与 `lane_lat` 完全对账（d_lat≡−lane_lat）；ttc 全部有限、责任槽一致、用 v4/v5 OD 重算逐行一致（corr 0.994）；v5↔v4.1 仅 ~0.5% 行差异（已知非确定性）。**LD 近场缺失是真实的 obs 退化**（有效槽 15.07→7.12/行；近场槽 8.82→1.31/行；v4 中 97.5% 行有近场"其他车道"槽，v5 为 0），是 b10 与 pre-v5 参照差距的候选因素之一，但不是 phase3 崩塌的机制。 |
| 最小修复 | 1) **修 specific 专家训练（决定性）**：重审 phase3 specific_only 配方（380k 全量重 BC + `mild=1.0` + 仅 action/chain、无 CE/WTA/traj/WM、无 mid-training 闭环 guard；Stage B specific 阶段同样退化）→ 加 epoch 级 guard/降 lr/重配损失，修复前不用 phase3 产物；2) **锚头加回冻结清单**（契约修复，1 行级；必要不充分）；3) 结构层：把 `Δψ` 反馈补进尾段（或 tracker 使用锚链 step0）；4) LD 近场恢复作为独立 A/B（次要）。 |

## 1. E1：plan 路径隔离（clean150）

### 1.1 读数

| 配置 | b10（Stage B ep010） | p3e5（phase3 ep005） |
|---|---|---|
| `plan_anchor.enabled=true`（原链） | **0.260**（39/150，`runs/BTC20261005-115035_v7sb_b_ckpt_epoch010_sub150`） | **0.000**（0/150，`runs/BTC20261005-134526_v7sb_p3_ckpt_epoch005_sub150`） |
| `plan_anchor.enabled=false`（本次 E1） | **0.0067**（1/150；144 out_of_road + 4 collision）`runs/BTC20261005-141638_v7diag_b10_noanchor_sub150` | **0.0067**（1/150；140 out_of_road + 6 max_step + 3 collision）`runs/BTC20261005-141922_v7diag_p3e5_noanchor_sub150` |
| `--eval-reference repeat_action`（附加对照：只消费 `action_mu`，不消费 plan） | **0.000**（rc mean 0.147；141 out_of_road + 4 collision + 5 max_step）`runs/BTC20261005-142621_v7diag_b10_repeatact_sub150` | **0.000**（rc mean 0.222；146 out_of_road）`runs/BTC20261005-142650_v7diag_p3e5_repeatact_sub150` |

配对（150 spec 全配对）：b10 anchor 的 39 个成功**没有任何一个**被 noanchor/repeat_action/p3 任意配置复现（onlyA=39, onlyB=0/1）→ 锚计划路径是唯一在工作的路径。

### 1.2 判读（按预设判据）

- a) 关锚后**没有恢复**（b10 0.26→0.007，p3 0.0→0.007）⇒ 按字面判据：**不是 K-anchor plan 路径单独导致的**，指向 obs/数据/共享权重。
- 但 p3 的 0.0 vs 关锚 0.0067（1/150）在 n=150 下差异很小、不构成单独结论；**锚计划被训坏**的更强证据是 §3.3 的权重交换实验（b10+p3 锚头 = 0/150，而 b10 = 39/150）。

### 1.3 混淆说明（必须记录）

- 关锚后 `plan[:,1:]` 来自 policy rollout；而在 anchor 训练体制下（Stage B BC 的 rollout 尾段 = 锚计划），**policy 头的尾段输出从未被监督**（Step B 的 `traj`/`action` 损失只覆盖锚链与 step0 `action_mu`；phase3 的 `action_chain` 损失同样只监督锚链）。因此 noanchor ≈ 0.007 是"未训练尾段"的地板值，不能证明"rollout 路径健康"。
- `repeat_action` 把 `action_mu` 外推 6 步（3 s 等曲率弧），对 dθ 偏差极敏感，0.0 说明 step0 策略在闭环中偏弱/有偏，但不能单独定位 obs vs 权重。作为参照：pre-v5 的 RL 模型（v3sa_u025，PPO+A-hold 训练）repeat_action = 0.273，说明该口径本身可工作。

## 2. plan 输出检查（闭环 dump：anchor vs rollout、lane 变换中间量、锚 id）

探针：`/tmp/opencode/v7_fail_diag/probe_plan_obs.py`（13 个 clean150 spec：5 成功 + 8 失败，覆盖 straight/curve/uturn/roundabout/merge/tollgate；逐决策记录锚/无锚前向、lane/ttc 原始块、lane 变换中间量）。产物 `probe_b10.json` / `probe_p3e5.json`。

### 2.1 lane 帧变换：符号/幅值正确，但 `Δψ` 只在 `dθ_0`

- 代码事实：`net/anchor.py::lane_follow_dtheta`：`follow[0]=Δψ+κ·ds_0`、`follow[i]=κ·ds_i (i≥1)`；`net/model.py::_rollout`：`actions[0]=action_mu`（= `plan[:,0]`），`actions[k≥1]=anchor_plan[:,k]`。
- 读数（b10，n=1181 有效决策）：`corr(action_mu_dθ, Δψ)=0.55`、`corr(锚尾 dθ_1, Δψ)=0.05`、`corr(锚尾 dθ_1, κ)=0.54` → **step0 跟随航向误差，尾段只跟随曲率**；`ctx_valid=0.88`（近场点有效门控），`|Δψ| mean=0.054 rad`、`|κ| mean=0.0034`。
- 开环"计划 vs 车道"隐含横向偏移（6 步积分 `−Σ ds·sin(Δψ+κs−Σdθ)`，左正）：
  - b10 执行链（mu+锚尾）：mean **+0.125 m**、|·| 0.467 m、>1 m 占 10.8%；若用完整锚链（含锚 step0）：|·| 0.394 m、>1 m 占 6.4%。
  - p3 执行链：mean **+0.912 m（左）**、|·| **1.979 m**、**>1 m 占 70.2%**；完整锚链 |·| 1.295 m、>1 m 占 49.6%。
  - 未训练 rollout 尾（b10）：mean +1.385 m、|·| 1.556 m、>1 m 占 60.1% —— 左偏更强。
- 结论：lane 变换本身符号/量级与 obs 文档一致（无恒向符号错），**结构性缺陷是"Δψ 只出现在被丢弃的 step0"**；p3 的尾段左偏被 phase3 放大（§3）。

### 2.2 锚 id 分布 / 软混合 / 幅值

| 指标 | b10 | p3e5 |
|---|---|---|
| argmax 锚直方图（c0..c5） | c0 77.3%、c5 7.5%、c2 5.4%、c3 4.3%、c4 3.8%、c1 1.7% | c0 60.3%、c5 10.9%、c3 9.8%、c1 6.5%、c4 6.3%、c2 6.1% |
| probs 熵 / max prob 均值 | 0.913 / 0.673 | 0.981 / 0.619 |
| 尾段 dθ 均值（steps1..5） | +0.0027/+0.0045/+0.0035/+0.0071/+0.0073 | +0.0093/+0.0083/+0.0101/+0.0147/+0.0142 |
| 尾段 |dθ| 均值（steps1..5） | 0.0165/0.0192/0.0199/0.0200/0.0231 | 0.0298/0.0282/0.0315/0.0303/0.0315 |
| 未训练 rollout 尾段 |dθ| | 0.0435/0.0568/0.0602/0.0557/0.0433 | 0.0415/0.0448/0.0499/0.0600/0.0622 |

- 软混合未硬坍缩（熵≈0.91–0.98），但 **p3 的选择分布从 c0 明显向 c1/c3/c5 迁移，尾段 dθ 幅值/左偏约为 b10 的 1.5–2 倍**。
- 失败轨迹示例（spec 0，直道 κ=0）：b10 全程 d_lat 振荡 −0.5~−0.9 m 但完赛；p3 在 step 10–50 内 d_lat 0→+0.63→+1.08→+1.75（左缘）出界，期间尾段 dθ 持续 +0.02~+0.08（左），而 `Δψ` 为负（车道在右）→ **参考路径把车带向左出界**。

## 3. phase3 0.0 的直接根因：specific 专家/router 被训坏（决定性）+ 锚头被误训（契约缺口）

### 3.1 契约 vs 实现

- 预注册 `docs/v7_program_prereg.md` §11：*"phase3：`freeze=specific_only`（默认）下锚头随 plan head 主干冻结 → anchor_ce/anchor_wta 自动降级为 0；本臂 phase3 不启用锚损失"*。
- 实现事实（离线复现，`pipeline/stages.py::_SPECIFIC_PHASE_FREEZE` + `pipeline/trainer.py::apply_freeze_prefixes`）：
  - 冻结前缀 = `st_gnn./value./encoders./mem_encoder./plan_head.fusion./plan_head.norm./plan_head.ego_next./plan_head.moe.primary./policy.` —— **不含** `plan_head.anchor_head.` / `speed_head.` / `residual_head.` / `anchor_embed`；
  - 因此 `requires_grad=True` 的参数 = 50 个，其中 **13 个是锚头参数**；`build_phase3_optimizer` 把它们放进 **base 组（lr=3e-4×0.25=7.5e-5）**（base 组非空，没有报错）。
  - phase3 日志实证：`frozen=119 · 可训=50 · lr=3.00e-04（base×0.25 · specific×0.5）`（`runs/..._p3/logs/stage_b.log` L10/L22）。
- 参数漂移（b10 → p3e5，相对 L2）：`anchor_head.2.weight` **10.1%**、`anchor_head.0.weight` 8.4%、`speed_head.2.weight` 15.5%、`residual_head.2.weight` **26.6%**、`anchor_embed` 1.8%；同时 `moe.experts.*` 23.7%、`moe.router.*` **35.1%**；`policy./fusion.` 漂移 **0.000**（正确冻结）。
- phase3 没有 anchor CE/WTA（losses 全 0），锚头唯一监督 = `action_chain`（权重 0.2，L2，只监督 `plan[:,1:]`）+ 全量重 BC（380,652 行；anchor 360,572 + dagger 20,080；`mild_weight=1.0`）。

### 3.2 训练侧已观测到左偏

phase3 `metrics.json` 的 val（含 expert anchor 行）：`bc_bias_chain_dtheta_h2..h6 = +0.0089/+0.0080/+0.0075/+0.0105/+0.0099 rad`（系统正偏=左）；`bc_bias_traj_lat_h3..h6 = +0.020/+0.037/+0.037/+0.025 m`（横向左偏）。这与 §2.1 闭环隐含偏移（p3 +0.91 m 左）方向一致。

### 3.3 分解实验（锚头 vs 专家/router 漂移；clean150、anchor 开）

交换 b10/p3 的 13 个锚头参数生成两个混合 ckpt（`/tmp/opencode/v7_fail_diag/hybrids/`）：

| 混合 | succ | rc mean | 终止 | 说明 |
|---|---|---|---|---|
| `b10_p3anchors.pt`（b10 主干 + p3 锚头） | **0.0000**（0/150） | 0.441 | 138 out_of_road + 11 max_step + 1 collision | **仅锚头漂移（b10 专家仍为 0，MoE 无效）就足以把 0.26 打到 0.0** |
| `p3_b10anchors.pt`（p3 主干 + b10 锚头，MoE 开） | **0.0133**（2/150） | 0.355 | 141 out_of_road + 7 max_step + 2 arrive_dest | 仅专家/router 漂移（锚头还原）也基本崩塌 |
| `p3_b10anchors.pt --moe-off`（同上前提，评测关 MoE） | **0.2600**（39/150） | 0.588 | 89 out_of_road + 39 arrive_dest + 16 max_step + 6 collision | **与 b10 anchor 逐 spec 完全一致（150/150 同判定、同 39 成功）** → primary 主干 + 还原锚头 = 原始 b10；崩塌完全来自**被训练的 specific 专家/router** |

判读（三层分解）：
1. **专家/router 是决定性破坏项**：p3 主干 + b10 锚头，MoE 开 = 0.013，MoE 关 = 0.260（与 b10 完全相同）→ phase3 训练的 specific 专家/router 把 0.26 打到 0.013。
2. **锚头漂移是独立第二缺陷**：b10 + p3 锚头（专家为零）= 0.0 → 即使专家没问题，锚头漂移也能单独致崩。两缺陷同时存在 → 实际 p3 = 0.0。
3. **只修一处不够**：只冻结锚头（p3+b10锚头，MoE 开）仍 0.013；只关 MoE（b10+p3锚头）仍 0.0。要回到 0.26 必须**两者同时**修复（冻结锚头 + 修/停 specific 专家训练）。

旁证：Stage B 自己的 specific 阶段（epoch 11–20，专家从零开始训练，CE/WTA 正常开启）也已把 clean150 从 0.26（ep010）降到 0.20（ep015）/0.16（ep020）→ "specific 训练在 v5 链上普遍有害"，与上表一致。

## 4. obs 检查（v4.1 vs v5 同场景，eval500 val 对齐 36,147 行）

脚本 `probe_obs_v4v5.py`，产物 `probe_obs_v4v5.json`。

### 4.1 LD 近场缺失（真实退化，非异常）

| 指标 | v4.1 | v5 |
|---|---|---|
| 有效槽/行 | 15.07 | **7.12** |
| 近场（0<dx<20 m）槽/行 | 8.82 | **1.31**（多为车道末端 clamp 残值） |
| 有任意近场槽的行 | 100% | 60.5% |
| 近场"非当前车道"槽/行（槽位 5–15） | 4.60（97.5% 行） | 0（设计移除） |
| 有效 dx 分位 p50 / p95 | 17.2 / 53.9 m | 40.0 / 90.5 m |

- v5 的当前车道近场由 `lane` 块补偿（near/mid/far valid = 0.883/0.680/0.085），但**其他车道的近场几何全部丢失**，且 LD 有效槽数腰斩（池化特征变粗）。这是 b10（0.254）与 pre-v5 BC 参照 P1-B（0.314）差距的候选因素之一；对 phase3 崩塌无解释力（模型全程只见过 v5 口径）。

### 4.2 lane 块（v5）：符号/量级合理

- `d_lat ≡ −lane_lat`（corr=1.000，平均绝对残差 0）；`heading_err` std 0.0775、`curvature` std 0.0147、`width≡3.5 m`；
- `corr(heading_err, near_dy)=0.74`、`near_dy≈−d_lat`（均值 +0.03 m）、曲率符号与 mid 点横向方向一致（corr 0.40）；`corr(heading_err, d_lat)=−0.06`（全场景混合，属正常）；
- near/mid/far_valid = 0.883/0.680/0.085（far 低是 60 m 采样超车道末端的设计行为）。
- 结论：**lane 块数值合理，无符号/量级异常**。

### 4.3 ttc：无异常，且与 v4 口径一致

- `mask=1.0`、有限率 100%、`valid=0.708`、cap 率 83.9%、`min_ttc_x<3 s` 8.6%、`n_lt3_x` max=8、`resp_index∈[−1,15]`、valid/resp 一致性 100%、`resp_is_path=20.4%`；
- 用 v5 OD 重算 min-TTC：逐行一致（max |diff| 2.4e-7）；用 v4 OD 重算（同场景）vs v5：分位几乎相同、corr **0.994**、<3 s 率 8.63% vs 8.61% → **ttc 无 v4/v5 异常**。

### 4.4 数据

- v5 val 36,159 行 vs v4.1 36,148 行，对齐 36,147；基础字段差异 ≤0.53% 行且集中于已知非确定性 episode（与采集报告一致），signal/route_world 逐位一致。**未发现数据损坏**。

## 5. 判定与最小修复建议

### 5.1 根因判定

1. **phase3 崩塌（0.0）的直接原因：phase3 specific_only 微调产生两个独立缺陷（各自足以致崩），其中被训练的 specific 专家/router 是决定性项。**
   - **专家/router（决定性）**：漂移 23.7%/35.1%，gate 熵 0.95、负载均匀；`p3+b10锚头` MoE 开 = 0.013，**MoE 关 = 0.260 且与 b10 逐 spec 完全一致（150/150）** → 专家路径把 0.26 打到 0.013；其机制是专家残差改变 latent（锚选择分布 c0 77%→60%，锚计划左偏）。
   - **锚头（独立第二缺陷）**：冻结清单缺 4 个前缀 → 13 个锚头参数被训练（base 组 lr×0.25），且 `anchor_ce/wta=0`，唯一监督 = `action_chain` L2；参数漂移（anchor_head.2 10%、residual_head.2 27%、speed_head.2 16%）→ 锚计划系统性左偏/离车道（隐含横向偏移 mean +0.91 m、70% 决策 >1 m）。混合实验 `b10+p3锚头 = 0.0` 证明其单独即可致崩。
   - 旁证：Stage B 自己的 specific 阶段（CE/WTA 正常开启）也已把 clean150 从 0.26（ep010）降到 0.20/0.16（ep015/020）→ **specific 专家训练在 v5 链上普遍有害**，不限于 phase3 的降级损失配方。
2. **结构层缺陷（限制天花板，非本次崩塌主因）**：`Δψ` 只进入 `dθ_0`，而执行契约丢弃锚 step0（`plan[:,0]=action_mu`）→ tracker 参考尾段对航向误差无反馈；锚选择头 latent 不含 lane token。b10 的隐含车道偏移 |·| 0.47 m（10.8% 决策 >1 m）与此一致；b10 的 0.26 vs P1-B 0.314 的差距可能部分来自此。
3. **E1 字面判据**：关锚不恢复 ⇒ 不把"K-anchor plan 路径"单独定为根因；但该实验不能排除锚路径问题（noanchor 尾段未训练，0.007 是地板值）。
4. **obs/数据**：无异常发现；LD 近场缺失是真实退化（有效槽 15.07→7.12、近场槽 8.82→1.31），建议作为独立 A/B，而非本次失败的解释。

### 5.2 对任务给定候选根因的直接回答

| 候选根因 | 判定 | 证据 |
|---|---|---|
| ① K-anchor plan 输出在闭环中系统性错误（lane 帧变换符号/软混合/phase3 降级不匹配） | **部分成立**：lane 变换符号/幅值无错、软混合未坍缩；但 **phase3 降级不匹配成立**（锚头该冻未冻，被 chain L2 训出左偏）；另有结构层 `Δψ` 只进 step0 且被覆盖 | §2.1、§3.1–3.3 |
| ② LD 远场 {20,40,60,80} 丢失近场横向锚定 | **真实退化，但非本次崩塌机制**：近场槽 8.82→1.31/行、其他车道近场 4.6/行→0；模型全程只见过 v5 口径，且 phase3 崩塌可由权重交换实验完全解释 | §4.1 |
| ③ v5 obs/数据（lane/ttc）问题 | **未发现异常**：lane 与 lane_lat 完全对账、ttc 有限/一致/与 v4 OD 重算 corr 0.994；数据仅 ~0.5% 行已知非确定性差异 | §4.2–4.4 |
| （本轮新增）phase3 specific 专家/router 训练有害 | **决定性**：`p3+b10锚头` MoE 开 0.013 / MoE 关 0.260（≡b10）；Stage B specific 阶段同样退化 | §3.3、§1.1 |

### 5.3 最小修复（按性价比）

1. **修 specific 专家训练（决定性）**：当前 phase3 = 380,652 行全量重 BC（anchor 360,572 + dagger 20,080，`mild_weight=1.0`）+ 仅 `action 1.0 / action_chain 0.2`（无 anchor CE/WTA、无 traj/WM、`bias_calib=0`），5 epoch 把 clean150 从 0.26 打到 0.0；且 Stage B 的 specific 阶段（损失更全）同样退化 0.26→0.16 → **问题在 specific 专家训练本身**。建议：先做"Stage B 只训 primary（不进入 specific 阶段）+ anchor"对照，确认 0.26 可保持；再逐步加回 specific 训练并加 **epoch 级闭环子集 guard**（当前 guard 只在训练后筛查，未拦住）与更低 specific lr，并用 `bc_bias_chain_dtheta_h*`、`bc_bias_traj_lat_h*`、`anchor_plan_ade` 做 keep-best 判据。修复前**不要**用 phase3 产物做终评。
2. **修冻结契约（1 行级，必要不充分）**：把 `plan_head.anchor_head.`、`plan_head.speed_head.`、`plan_head.residual_head.`、`plan_head.anchor_embed` 加入 `_SPECIFIC_PHASE_FREEZE`（`pipeline/stages.py`），使实现与预注册 §11 一致；单独修此项后 phase3 仍 ≈ 0.013（专家缺陷未除）。
3. **结构修复（需设计决策）**：让 tracker 参考尾段携带 `Δψ` 反馈（例如把 `Δψ` 分摊到 `dθ_i`，或 tracker 使用完整锚链 step0），并把 lane token 接入 plan head 融合路径，使锚选择/残差能条件化于航向误差。
4. **LD 近场**：恢复 1 档近场（如 offset 5 m 或 {5,20,40,60}）做单变量 A/B；当前证据不支持它是主因。
5. **流程**：phase3 目前没有针对"锚计划离线偏置"的监控（`bc_bias_*` 已有但未设阈值/guard）；建议把 `bc_bias_chain_dtheta_h*`、`bc_bias_traj_lat_h*`、`anchor_plan_ade` 纳入 phase3 keep-best/guard 判据。
6. **不建议"关 plan_anchor 重训"**：b10 关锚 = 0.007（rollout 尾段未训练），锚路径是当前唯一能把 0.26 做出来的机制；崩塌发生在 phase3（specific 训练），关锚只会丢掉唯一有效路径。应先修 specific 专家训练 + 冻结锚头，再评估锚路径结构修复（第 3 条）。

## 6. 证据文件与复现

| 产物 | 路径 |
|---|---|
| E1 配置副本 | `/tmp/opencode/v7_fail_diag/v7_struct_b_v5_eval_noanchor.yaml` |
| E1/E1b/E2/E3 日志与结果 | `/tmp/opencode/v7_fail_diag/e1.log`、`e1b.log`、`e2.log`、`e3.log`；`runs/BTC20261005-14*_v7diag_*` |
| 混合 ckpt 生成 | 交换 13 个锚头参数（`plan_head.anchor_head./speed_head./residual_head./anchor_embed`）；`/tmp/opencode/v7_fail_diag/hybrids/{b10_p3anchors,p3_b10anchors}.pt` |
| plan/obs 探针 | `/tmp/opencode/v7_fail_diag/probe_plan_obs.py`、`probe_b10.json`、`probe_p3e5.json` |
| obs v4/v5 探针 | `/tmp/opencode/v7_fail_diag/probe_obs_v4v5.py`、`probe_obs_v4v5.json` |
| phase3 训练侧 | `runs/BTC20261005-0856_v7struct_v5_p3/stage_b/metrics.json`、`logs/stage_b.log` |

复现命令（GPU 串行）：

```bash
cd /workspace/01_Proj/DRL_PathPlan
# E1（关锚）
tools/venv-python tools/test.py --policy ckpt --ckpt <ckpt> --spec /tmp/opencode/phase3_diag/exp/specs_val_only150.json \
  --out /tmp/opencode/v7_fail_diag/runs --name v7diag_<tag>_noanchor_sub150 --workers 6 --tracker lqr \
  --config /tmp/opencode/v7_fail_diag/v7_struct_b_v5_eval_noanchor.yaml --eval-reference plan --device cuda
# repeat_action 对照（config 用 config/arms/v7_struct_b_v5_eval.yaml，--eval-reference repeat_action）
# 探针
tools/venv-python /tmp/opencode/v7_fail_diag/probe_plan_obs.py --ckpt <ckpt> --tag <tag> --device cuda
tools/venv-python /tmp/opencode/v7_fail_diag/probe_obs_v4v5.py
```

## 7. 异常与备注

- 评测/探针期间 GPU 串行，无并发 GPU 任务；所有输出写 `/tmp`（runs 输出为 `--out /tmp/...`），repo 工作区保持干净（`git status` 空）。
- E1 的 noanchor 配置把锚参数/缓冲作为 unexpected 丢弃（eval 侧过滤）：noanchor 模型 loaded=158/158，ckpt 174 键中 16 个 unexpected（13 参数 + 3 buffer `anchors/anchor_ds/anchor_dth`）；anchor 模型 174/174。
- phase3 冻结缺口是"静默"的：`build_phase3_optimizer` 只要求 specific 组非空；base 组被 13 个锚头参数填满后不报错，日志只显示"base×0.25"，无告警。
- 混合实验的边界：`p3+b10锚头` 的 b10 锚头作用在 p3 latent 上（专家已漂移）属跨模型拼接，不能完全排除"锚头 OOD"效应；但 `--moe-off` 对照（p3 主干 + b10 锚头 + 关专家）与 b10 逐 spec 完全一致（150/150），说明 primary 路径与锚头还原是干净的，差异全部来自专家路径。
- 诊断未覆盖：phase3 dagger 数据（20,080 行窗口 + pure_pursuit 反事实链）的质量未单独审计（本轮只做了 v5 expert 数据与 obs 的检查）；"specific 训练为何有害"的机制（专家残差/latent 漂移的具体方向）留待后续。
