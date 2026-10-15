# v7 程序收尾报告（IL 底座 → RL arm1 → 结构迭代；目标未达）

> 生成：2026-10-05（v7 程序收尾；写作时 HEAD `1ea339d`）。上游：[`v7_program_prereg.md`](v7_program_prereg.md)（§0–§11）、[`experiments.md`](experiments.md) §4（IDM 锚）、[`v6_program_report.md`](v6_program_report.md)（E-β″ 零点 / 配对口径）；执行计划 `.slim/deepwork/v7-beat-idm.md`。
>
> **一句话结论：IL 底座由 0.44（E-β″）提升到 w1 的 0.53（clean500 0.526 / eval500 0.530）；RL arm1 n=4 clean500 Δ vs w1 mean **+8.1pp**、4/4 正（3/4 显著），但 sd 5.6pp **方差闸（§2.2 sd≤5pt）未过**且碰撞普遍上升（0.102–0.182 vs w1 0.066）；最佳产物 s11 eval500 **0.646**（vs IDM −11.0pt）；结构迭代（obs v5 + K-anchor）经 fix-11 补救后 0.328 / 0.324 / 0.178，不具竞争力关闭 ⇒ **超越 IDM（0.756）的目标未达**。**

## 0. 摘要

- **目标（冻结）**：在冻结评测协议（`--spec env/specs/scenarios_eval500.json` / `--tracker lqr` / `--eval-reference plan` / `--max-steps 1000` / 单一 pinned IDM baseline）上，多 seed 汇总配对差 95% CI 下界 > 0 且过**方差闸（run sd ≤5pt 且 min Δ ≥−5pt）**，**显著超过 IDM 规则基线**（eval500 0.756）。
- **P0 锚定（Gate A：需修正+需补证 → 修正集 #1 后解剖通过）**：IDM 现口径复测 eval500 **0.756**（逐条 500/500 一致）/ clean500 **0.742** / tg45 **0.778**；best v6 RL（arm0）eval500 0.622 ⇒ 差距 **13.4pt**，分解为 tollgate **+7.0** / merge **+2.8** / t_intersection **+2.6** / roundabout +1.4 / curve +1.2…（A 侧终止类：off_road 11.2pt gross / collision 8.0pt）。工具 `tools/paired_eval.py` + 预注册入库。
- **P1 IL 底座（Gate B：需修正+需补证 → 修后放行）**：AB-only 三轮（obs v4 / roundtrip 过滤 / 横向权重）均未在闭环获益（P1-B 0.314/0.312；iter3 0.260/0.274）→ 探针 V1+V2 定位 **plan 尾链断点**（μ 改善不传导尾链）→ **DAgger 闭环重标注 cycle**（四窗口 clean500 +11.4~+21.2pp）；Gate B 覆盖 base 为 **w1 e005**（clean500 **0.526** / eval500 **0.530**，vs E-β″ +8.6/+9.0pp，vs P1-B +21.2/+21.8pp）。数值线 0.55–0.60 未达（记"结构性转 RL"）。
- **P2/P3 RL arm1**（w1 + bundle rc=1 + `off_road_edge` −0.5 + KL 锚 0.05→0.02）n=4（seeds 0/11/1/2）：clean500 **mean +8.1pp / sd 5.6pp / min +1.0 / max +14.2，4/4 正（3/4 显著）**；§2.2 方差闸 sd≤5pt **边际未过**（且 n<5）；**碰撞普遍上升**（+3.6~+11.6pp vs w1；s11/s2 超 IDM 锚）⇒ 不得宣称超越。逐 seed：s0 0.536/+1.0/0.512、**s11 0.668/+14.2/0.646**、s1 0.592/+6.6/0.550、s2 0.630/+10.4/0.622。
- **结构迭代（用户定向；非原 Gate）**：obs v5（`0291f3a`）→ A2 接线（`fcf047e`）→ K-anchor 计划头（`f42d648`）→ v5 重训链 **Stage B 0.254 / phase3 崩 0.0** → 根因诊断（specific 专家/router 训练=决定性破坏项 + `_SPECIFIC_PHASE_FREEZE` 漏锚头）→ fix-11 安全配方补救（`55adf90`）：clean500 **0.328** / eval500 **0.324** / tg45 **0.178（结构线首个非零，8/45）**，仍远低于 w1 ⇒ **不具竞争力，路线关闭**。
- **结论**：目标未达（eval500 最好 0.646 = −11.0pt vs IDM；clean500 最好 0.668 = −7.4pt；n=4 mean −17.4pt / −13.6pt）。**最佳产物 = arm1 s11 u150**（clean500 0.668 / eval500 0.646）；**IL 底座 = w1 e005**（0.526/0.530）。资产/tag 见 §4/§6。

## 1. 阶段表（P0–P4）与各门结论

| 阶段 | 内容 | 裁定 | 处置 / 关键证据 |
|---|---|---|---|
| **P0 锚定与差距解剖** | IDM 现口径复测（eval500/clean500/tg45）+ v6 best RL 差距分解 + 配对工具 + 预注册（含 Gate A 修正集 #1） | **Gate A：需修正+需补证 → 修正集 #1 后解剖通过** | 需修正 7 项（方差闸/n≥5、pin 表、IDM spec-seed 变体、Gate B/C 加固、安全闸量化、`expert500val` 禁训、`paired_eval` fail-closed）全部落文；内容 commit `b711e5f`（工具+prereg）、`0454308`+锚 `f5f0dc3`（修正集；611 passed）、`ee763f4`（fix-15：评测非严格恒等 ±0.2pp + IDM 锚复测）；差距报告 `docs/v7_reports/v7_p0_gap_decomposition.md` |
| **P1 IL 底座** | AB-only 三轮（obs v4 / roundtrip 过滤 / 横向权重）+ 探针 V1+V2 + DAgger 闭环重标注 cycle（4 窗口 × 15ep）+ Gate B 覆盖 base | **Gate B：需修正+需补证（低成本）→ 修后放行 P2；数值线 0.55–0.60 未达** | `b99e127`（obs v4 static）、`386b091`（过滤放宽）+`2e13dfa`（`action_dim_weights`；iter3 0.260/0.274）、`4a46c5e`（链标签+标定；630 passed）；cycle 四窗 top1 clean500 0.526/0.428/0.472/0.378（vs P1-B +21.2/+11.4/+15.8/+6.4pp）；报告 `docs/v7_reports/v7_p1_dagger_cycle.md`、`docs/v7_reports/v7_p1_probe_v1v2.md` |
| **P2 RL arm1** | w1 e005 + v5 bundle(rc=1) + `off_road_edge`(−0.5 / scale 1.0) + KL 锚 0.05→0.02；seeds 0/11（后补 1/2） | 单臂；seed0 flat / seed11 strong；u50+u100 双点闸无 early-collapse | 预注册 §9 内容 `2ddfa22` / 锚 `c37acbb`；`config/arms/v7_arm1_offroad.yaml`；报告 `docs/v7_reports/v7_p2_arm1.md` |
| **P3 n=4 分布** | seeds 0/11/1/2 全量 clean500 + tg45/T3 + 每 seed 一次 eval500（keep-best 全量配对采纳） | clean500 **mean +8.1pp / sd 5.6pp / 4/4 正（3/4 显著）**；§2.2 方差闸未过；碰撞上升、安全闸未过 | 报告 `docs/v7_reports/v7_p3_arm1_seeds.md`；s11 洁净复验 `docs/v7_reports/v7_p2_s11_verify.md`；s1/s2 于 pre-v5 worktree（`2f4450e`）复跑保代码可比 |
| **结构迭代（用户定向；非原 Gate）** | Lane A obs v5（`0291f3a`）→ A2 lane/ttc 接线（`fcf047e`）→ B K-anchor 计划头（`f42d648`）→ v5 重训链 → 崩 → 根因诊断 → fix-11 补救（`55adf90`/锚 `1ea339d`） | **失败关闭（不具竞争力）**：补救后 clean500 0.328 / eval500 0.324 / tg45 0.178，仍远低于 w1 0.526/0.530 | `0065264`（重训臂配置）；报告 `docs/v7_reports/v7_struct_fail_diag.md`、`docs/v7_reports/v7_p3fix_retry.md`（`v7_struct_retrain_v5.md` 未入仓，暂留 `/tmp/opencode/KEEP/`，见 `docs/v7_reports/README.md` 未决项）；可行性/接线 `docs/v7_reports/v7_kanchor_feasibility.md`、`docs/v7_reports/v7_struct_a_obs_v5.md`、`docs/v7_reports/v7_struct_a2_wiring_probe.md`、`docs/v7_reports/v7_struct_b_kanchor.md` |
| **P4 收尾** | 终版报告 + tag + 资产清单 | 目标未达如实记录；最佳产物 arm1-s11 归档 | 本报告；tag `v7-close-20261006`（§6） |

- **未跑/降级（如实）**：arm2（bundle+KL）/ arm3（bundle only）已预注册（§10，内容 `a1b6950` / 锚 `2f4450e`；配置 `config/arms/v7_arm2_bundle_kl.yaml`、`v7_arm3_bundle_only.yaml`）但**未运行**——用户定向重排（Lane A/B 优先）后降级为后备；Gate D（≥5 seed 主判据）未启动。

### 1.1 各门要点（详）

- **Gate A（需修正+需补证）**：解剖逐项复算一致（IDM 代码自 `54f8c5e` 至 `b657fea` 逐字节未变、spec/seed 对齐）；修正 = 主判据加方差闸 + n≥5、评测 pin 表（`lqr`/`plan`/`1000` 步/单 baseline）、IDM 多 run 改 spec-seed 变体（同 spec 重复=恒等重复；实测变体 mean 0.729 / sd 2.5pp）、u50 单点闸假阳性 3/4 ⇒ 双点闸 + keep-best、安全闸量化（collision 绝对 10% 不可达——IDM 自身 14.4% ⇒ 相对支路；off-road ≤IDM+2pt；speed_ratio ≥0.9×IDM=0.666）、`expert500val` 禁入训练（preflight fail-closed）；补证 = IDM 现口径复测（0.756 逐条一致 / clean500 0.742；fix-15）+ tollgate 机理 T1/T2/T3。
- **Gate B（需修正+需补证 → 放行）**：P1 clean 侧通过（DAgger 实质有效：四窗口 +11.4~+21.2pp）；**tollgate 侧未通过**（命中来自 w4 的单条 T3，判据分句缺陷 + 多重比较未校正）；数值线 0.55–0.60 未达。修正 = ① P2 base **覆盖为 w1 e005**（clean500 选点纪律；w4 降档案/fallback；记录"cycle 采纳 vs Gate 覆盖"偏差）；② `bias_calib` 默认关闭（0.2 三窗 T3 全 0/9）。补证 = w1 eval500 一次（先冻结 base，标"探索性复评"）：**0.530**（vs E-β″ +9.0pp net45 z5.13；vs P1-B +21.8pp net109 z9.99；vs IDM −22.6pp net−113 z8.85），与 clean500 0.526 高度一致（无选点膨胀）。
- **P2/P3（arm1）**：见 §2.2；KL 锚末值 0.02 已执行（v6 教训），但 n=4 下 sd 仍 5.6pp；u50/u100 双点闸在 s11（u50 net−25）与 s2（u100 net−27）各触发一次，均未双点 ⇒ 无 early-collapse；keep-best 采纳点跨 u050–u200（s0 u100 / s11 u150 / s1 u050 / s2 u200），提示 run 内晚段不稳定仍存在（s0 u200 sub150 坍缩 succ 0.007 / off 0.987）。
- **结构迭代（用户定向）**：详见 §3.3；表示层主判据 2/3 满足（CE 0.588→0.309 ≤0.896 ✓；分配非坍缩 c0 0.751 <0.90 ✓；WTA 0.162→0.105 未达 <50% 即 0.081 ✗）；闭环因 phase3 崩塌与表示天花板失败；fix-11 补救保住了不崩（guard 0.32/0.327/0.26/0.253/0.187，best e002），但无法回到 w1 水平。

## 2. 关键数字表

> 全部数字可由盘上 `runs/*/episodes.csv`（逐 `(id,seed)` 配对；`net=fixed−broken`、`z=|net|/√(fixed+broken)`）与 `runs/*/metrics.json`（overall）复算；配对报告 `tools/paired_eval.py`（json+md）。复算入口见 §6。

### 2.1 锚与 IL 基线

| 对象 | clean500 | eval500 | tg45 | 路径 |
|---|---|---|---|---|
| **IDM（冻结锚）** | **0.742** | **0.756** | **0.778** | `runs/BTC20261002-100413_v7p0_idm_clean500` / `runs/BTC20260927-1839_eval500_baseline` / `runs/BTC20261002-160154_v7p1b_tg45_idm` |
| E-β″（v6 IL 零点） | 0.440 | 0.440 | — | `runs/BTC20261001-195447_v6p3_e3_clean500` / `runs/BTC20261001-200331_v6p3_e3_eval500` |
| P1-B e010（v4 obs 链） | 0.314 | 0.312 | 0.000 | `runs/BTC20261002-162639_v7p1b_screen_epoch010` / `runs/BTC20261002-164209_v7p1b_eval500_sel` |
| **w1 e005（P2 init）** | **0.526** | **0.530**（探索性） | 0.000 | `runs/BTC20261003-045912_v7p1dagger_w1_clean500` / `runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` |
| arm0（v6 RL 最好，单 seed） | — | 0.622 | — | `runs/BTC20261002-0026_p4arm0_eval500` |

- 配对：w1 vs E-β″ clean **+8.6pp**（net 43，z 5.25）/ eval **+9.0pp**（net 45，z 5.13）；w1 vs P1-B **+21.2pp**（net 106，z 10.11）/ **+21.8pp**（net 109，z 9.99）；w1 vs IDM **−21.6pp**（net −108，z 7.96）/ **−22.6pp**（net −113，z 8.85）；arm0 vs IDM eval **−13.4pt**（net −67，z 5.53）。
- P1 DAgger cycle 四窗口 top1（clean500 配对 vs P1-B e010）：w1 **0.526**（+21.2pp，z10.1）/ w2 0.428（+11.4）/ w3 0.472（+15.8）/ w4 0.378（+6.4）；tg45 0/0/0.022/0.067；T3 S1 0/0/0/1（仅 w4 命中判据字面；Gate B 覆盖 base 为 w1）。

### 2.2 RL arm1 n=4（seeds 0/11/1/2；base = w1 e005）

| seed | 采纳 ckpt | clean500 | Δ vs w1（CI95 / z / p） | eval500 | Δ vs w1 eval（CI95 / z） | collision | off_road | tg45 |
|---|---|---|---|---|---|---|---|---|
| 0 | u100 | 0.536 | **+1.0** [−4.2, +6.0] / 0.38 / 0.76 | 0.512 | −1.8 [−7.0, +3.2] / 0.70 | 0.102 | 0.344 | 0.000 |
| 11 | u150 | **0.668** | **+14.2** [+10.0, +18.4] / 6.30 / 1.7e−10 | **0.646** | **+11.6** [+7.6, +15.6] / 5.58 | 0.182 | 0.138 | 0.000 |
| 1 | u050 | 0.592 | **+6.6** [+2.0, +11.2] / 2.84 / 0.0057 | 0.550 | +2.0 [−2.6, +6.6] / 0.86 | 0.162 | 0.208 | 0.000 |
| 2 | u200 | 0.630 | **+10.4** [+6.0, +14.8] / 4.60 / 4.9e−6 | 0.622 | **+9.2** [+5.0, +13.4] / 4.31 | 0.174 | 0.172 | 0.044 |
| **分布** | — | mean 0.607 / **Δ mean +8.1pp / sd 5.6pp** / min +1.0 / max +14.2 | 4/4 正；3/4 显著（z≥1.96） | mean 0.583 | **Δ mean +5.3pp / sd 6.2pp** | 0.155（mean） | 0.216（mean） | 0.011（mean） |

- **§2.2 判读**：sd 5.6pt > 5pt **方差闸未过**（且 n=4 < 5）⇒ 按预注册判"稳定性未达标，不宣称超越"；min Δ +1.0 ≥ −5pt 单侧满足。
- **vs 参照**：每 seed eval500 vs P1-B +20.0 / +33.4 / +23.8 / +31.0pp；vs IDM（0.756）−24.4 / −11.0 / −20.6 / −13.4pp。clean500 vs IDM（0.742）：s0 −20.6 / s11 −7.4 / s1 −15.0 / s2 −11.2pp（s11 最小缺口，z 3.31）。
- 路径：clean500 `runs/BTC20261003-072348_v7p2_s0_u100_clean500` / `runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean` / `runs/BTC20261005-181715_v7p2_s1_u050_clean500` / `runs/BTC20261005-194156_v7p2_s2_u200_clean500`；eval500 `runs/BTC20261003-074819_v7p2_s0_eval500` / `runs/BTC20261005-074755_v7p2_s11_eval500_clean` / `runs/BTC20261005-184357_v7p2_s1_eval500` / `runs/BTC20261005-200952_v7p2_s2_eval500`。

### 2.3 结构线（v5 + K-anchor）

| 项 | 读数 | 路径 / 说明 |
|---|---|---|
| Stage B b10（入口） | clean500 **0.254**（vs w1 net −136，z 11.33）；sub150 0.26 | `runs/BTC20261005-120728_v7sb_b_ckpt_epoch010_clean500` |
| phase3（失败链） | clean500 **0.0** / eval500 0.0（496/500 out_of_road） | `runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500` |
| fix-11 守护（clean150） | epoch1–5 = 0.32 / **0.327** / 0.26 / 0.253 / 0.187（min_success 0.13 未触发） | `runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/guard/guard.json` |
| **补救终评（best e002）** | clean500 **0.328** / eval500 **0.324** / tg45 **0.178**（8/45，coll 0.0） | `runs/BTC20261005-161813_v7p3fix_best_clean500` / `…-163243…eval500` / `…-163040…tg45` |
| 配对（clean500） | vs w1 **−19.8pp**（net −99，z 8.17）；vs P1-B +1.4pp（net 7，z 0.74，ns）；vs IDM −41.4pp（z 12.91） | `tools/paired_eval.py`（p3fix 配对） |
| 配对（eval500） | vs w1 **−20.6pp**（net −103，z 8.33）；vs P1-B +1.2pp（net 6，z 0.65，ns）；vs IDM −43.2pp（z 13.72） | 同上 |
| 配对（tg45） | vs IDM −60.0pp（0.178 vs 0.778，z 5.2；coll 0.0 是唯一亮点） | 同上 |
| 表示层主判据 | CE 0.588→**0.309**（≤0.896 ✓）；分配 c0 **0.751**（<0.90 ✓，C1–C5 各 >0）；WTA 0.162→**0.105**（<50% 未达 ✗）；plan ADE 0.327→0.243 m | `v7_struct_retrain_v5.md` §Stage B（未入仓，暂留 KEEP；见 `docs/v7_reports/README.md` 未决项） |
| 安全/行为 | clean500 coll 0.044 / off 0.520；eval500 coll 0.034 / off 0.534；tg45 off 0.600 | `metrics.json` |

### 2.4 安全闸（§5）与事件率

- **collision**：arm1 vs w1（0.066）**全部上升**：s0 +3.6pp / s11 +11.6pp / s1 +9.6pp / s2 +10.8pp（s11、s2 配对显著）。对 IDM 锚：clean500 0.174（s11 0.182 **超**）；eval500 0.144（s11 0.202、s2 0.172 **超**）⇒ 相对闸（≤IDM +0pt）**未过**。
- **off-road**：arm1 相对 w1（0.398）显著改善（−5.4~−26.0pp），但绝对率 0.138–0.374 仍远超 IDM（clean 0.054 / eval 0.068）⇒ 绝对/相对闸均未过。
- **speed_ratio / rc**：IDM eval500 sr 0.740 / rc 0.882；w1 0.462 / 0.734；arm1 s11 0.477 / 0.860（s1 0.418 / 0.804、s2 0.453 / 0.848、s0 0.403 / 0.754）⇒ **speed_ratio 闸（≥0.666）全部未过**；rc 仅 s11/s2 接近 IDM。
- **tg45**：arm1 0–0.044（IDM 0.778）⇒ tollgate 结构问题未解；结构线补救 0.178 为程序内最高，仍 −60pt。

## 3. 归因与教训

### 3.1 DAgger 有效：断点在"plan 尾链"，闭环重标注绕过了它

- **诊断链**：三轮 AB-only 阴性——① **obs v4 static**（`b99e127`）：演示含变道绕亭（train tollgate 317/455=69.7%，净空中位 1.81 m）、static 特征确被使用（置零改变行为）但只学会纵向减速；变道执行帧 62% 被 roundtrip 过滤；闭环 P1-B **0.314/0.312**（vs E-β″ −12.6/−12.8pp，vs IDM −42.8/−44.4pp）。② **roundtrip 过滤放宽**（`386b091`）：LC 可训练率 33.4%→78.0%（tollgate 33.3%→91.1%），教师成功率不变。③ **横向权重 `action_dim_weights=[1.0,69.4]`**（`2e13dfa`）：TF 探针 μ dθ MAE 0.0341→0.0137（train）但完整 A20+B20（iter3）**0.260/0.274**（vs P1-B −5.4/−3.8pp 显著更差）。
- **断点定位（探针 V1+V2）**：μ/head 改善（dθ MAE −31%、head 横向 −28%）**不传导到尾链**（tail 横向 MAE +4.9%；tail/head 比 8.4→12.2；尾链横向 bias 放大 5–10×；μ dθ 恒定欠转 bias −0.0043 rad/step）；`repeat_action` 口径双 ckpt 地板效应（≤1.3%），不能单独证明。
- **修复（DAgger cycle）**：闭环 IDM 教师重标注 `action[0:6]=t..t+5` 链标签 + bias 标定（`4a46c5e`）；四窗口 clean500 **0.378–0.526（Δ +6.4~+21.2pp vs P1-B）**；采纳 w1 e005（0.526/0.530）。**教训：表示/μ 的离线改善必须以"尾链执行参考"为监督对象，否则不传导闭环。**

### 3.2 RL 增益 = 方差主导 + 碰撞代价（不可采纳）

- **方差**：n=4 clean500 Δ mean +8.1pp / sd 5.6pp；逐 seed 跨 +1.0~+14.2；eval500 Δ sd 6.2pp（0.512–0.646）。sd 5.6pt 超 §2.2 闸（≤5pt）且 n<5 ⇒ 按预注册"稳定性未达标，不宣称超越"；KL 锚 0.05→0.02 未能把方差压到闸内（s0 u200 sub150 坍缩 succ 0.007 / off 0.987；s11 中段 u50 net −25 → u100 +16 翻转；采纳点跨 u050–u200）⇒ **run 内晚段不稳定性仍是第一瓶颈**。
- **碰撞代价**：success 增益伴随碰撞普遍上升（+3.6~+11.6pp vs w1；s11 0.182 / s2 0.174 超 IDM clean500 锚 0.174；s11 eval 0.202 超 IDM eval 锚 0.144）。`off_road_edge` 的 off-road 改善（−5.4~−26.0pp）没有换来安全面净收益 ⇒ 单臂奖励改动不足；安全闸与 success 必须同判（v6 arm8 立项逻辑在 v7 未落地）。
- **目标缺口（诚实）**：最好 run eval500 0.646（−11.0pt）、clean500 0.668（−7.4pt）；n=4 mean −17.4 / −13.6pt。**"超越 IDM"按现证据仍属低概率尾部**（`.slim/deepwork/v7-beat-idm.md` 目标概率条）。

### 3.3 结构线失败链条（K-anchor）

1. **Stage B 天花板**：b10 clean500 0.254 < pre-v5 P1-B 0.314；**obs v5 LD 近场退化**是候选因素——有效槽 15.07→7.12/行、近场槽 8.82→1.31/行、其他车道近场几何全部丢失（`v7_struct_fail_diag.md` §4.1）；lane 块只补偿当前车道。
2. **phase3 崩 0.0 = 两个独立缺陷各自足以致崩**：① **specific 专家/router 被 phase3 训练（决定性）**——`p3+b10锚头 --moe-off` = 0.260 与 b10 逐 spec 一致 150/150，MoE 开 = 0.013；② **锚头被误训（契约缺口）**——`_SPECIFIC_PHASE_FREEZE` 漏 `plan_head.anchor_head./speed_head./residual_head./anchor_embed` 4 前缀，锚头落入 base LR 组（漂移 8.4–26.6%），`b10+p3锚头` = 0.0。只修一处不够（两者同时修复才回到 0.26）。
3. **Δψ 结构问题**：`Δψ` 只进入 `dθ_0`，而 `plan[:,0]` 被 `action_mu` 覆盖 ⇒ **被执行的后 5 步（tracker 参考尾段）对航向误差零反馈**（corr(tail dθ1, Δψ)=0.05 vs corr(action_mu dθ, Δψ)=0.55）；p3 尾段左偏被放大（隐含横向偏移 mean +0.91 m / >1 m 占 70.2%）→ 496/500 出界。锚计划是 b10 唯一工作路径（39 成功全部来自锚路径；关锚/`repeat_action` 均为地板 ~0.007/0.0）。
4. **选择头输入不含 lane token**：plan_head 融合仅 ego/OD/LD/others/nav/signal；失败窗口模式偏移（tollgate 64% / merge 43% 落 C2"左转回正"、JSD 0.32/0.29）未被选择头有效利用。
5. **补救（fix-11）**：锚头冻结契约修复 + `freeze=trunk_only`（只训共享主干，冻结 specific 专家/router/锚头/WM/value）+ 保留锚 CE/WTA 1.0 + epoch 级 clean150 守护（<0.13 即停，fail-closed）→ 不崩、tg45 首次非零（0.178，coll 0.0），但 clean500 0.328/eval500 0.324 仍远低于 w1 ⇒ **关闭路线**（不再投入）。
6. **表示层读数**：CE/分配判据达标、WTA 未达标；锚分布 p3 从 c0 向 c1/c3/c5 迁移且尾段 dθ 幅值 1.5–2×（左偏），说明表示本身不是唯一问题，**训练配方（specific 专家）与结构反馈（Δψ）才是**。

### 3.4 评测纪律与事故（含 dirty 事故）

| # | 事件 | 根因 | 处置 | 影响/旁证 |
|---|---|---|---|---|
| ① | **seed11 终评 void 三连（dirty 事故）**：2026-10-05 07:12–07:19 另一 lane 在本 repo 并发改动未提交源码（`env/obs/{ld,schema,ttc,...}.py`、`net/*`），seed11 的 u125 clean500 / tg45 / T3 / eval500 首帧报 `obs['ttc']` 形状错（n_error=n） | GPU 评测与源码 WIP 并发（评测进程冷启动读到新代码） | 全部标 **void 并从分布剔除**；tree 干净后在 pre-v5 worktree（`2f4450e`）洁净复验 4 项：u150 clean500 **0.668 逐 episode 一致复现**、eval500 **0.646**、tg45 0.0、T3 0/9（`v7_p2_s11_verify.md`） | 教训：**训练/评测期间禁止并发改 env/net 代码**；dirty 期读数必须 worktree 隔离复验后才能采用 |
| ② | `manifest dirty=1` 取证歧义 | `pipeline/run_paths._git` 把空 porcelain 输出映射为 `"unknown"`（计 1 行） | 澄清：**dirty=1 = 干净树**；真实 WIP dirty=15/16 | 历史所有干净 run 均 dirty=1；不影响读数 |
| ③ | P1-B eval500 被触碰两次 | 链 pin `final.pt`（0.330）与 clean500 选定 `ckpt_epoch010`（0.312）各评一次 | 如实记录（选点仅用 clean500；严格口径下属两次评估） | 后续 P2 每 seed 采纳 candidate 仅评一次（§9/§10 纪律） |
| ④ | P2 driver seed0 `write_report` 崩溃 | `_fmt_pair` 把 None 交给格式串（`eval500_pair` 裸 pair struct） | 修复 None 安全格式化；seed0 数据完整保留，补跑 seed11 | 批中止于 seed0 终评后（无 GPU 浪费） |
| ⑤ | s1 首跑 1.9s 失败（`BTC20261005-1724`） | 主树已推进到 v5 代码（encoders.lane/ttc，`type_embed` 5→7），与 pre-v5 w1 ckpt 不兼容 | s1/s2 改在 **pre-v5 worktree**（`2f4450e`）运行以保代码可比（w1/s0/s11 同为 pre-v5 代码期） | s1/s2 读数与 s0/s11 可比；worktree `runs` 软链主树 |
| ⑥ | 结构链 phase3 epoch1 abort（234.1s > 196s 旧基线） | v5 链 phase3 组成变化（锚 360,572 + 窗口 20,080 行）导致 epoch1 装载变慢 | 基线改锚 200s×1.35=270s 后 resume（操作参数修正，不改判据）；原现场保留 | 稳态 169.6–223.7s；无判据变化 |
| ⑦ | DAgger cycle w1 首训被 wall-guard 误杀 | 守卫自进程启动计时（含 ~4 min 装载），469.3s > 217.5s 阈值 | 修复计时起点 + 阈值 145s×1.75=253.75s；数据复用重训，配方/评测口径不变 | 误杀 run 删除；w1 最终读数不受影响 |

## 4. 资产清单

### 4.1 ckpt sha256（均对盘上文件复核）

| 资产 | 路径 | sha256 |
|---|---|---|
| **w1 e005（P1 采纳 / P2 init）** | `runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt` | `fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9` |
| arm1 s0 u100 | `runs/BTC20261003-0641_v7p2_s0_arm1/ckpt_u100.pt` | `aa4b657a328797c48552af1f1cc1da44bb070762166adcdbff823861d85a5227` |
| **arm1 s11 u150（程序最佳 RL）** | `runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt` | `a7cc091fcbda670b25c396a43e18dc39abe089053e50aa292b5fc0f19164ba2e` |
| arm1 s1 u050 | `runs/BTC20261005-1732_v7p2_s1_arm1/ckpt_u050.pt` | `e6d6db418329c082bf64e35d45b8fea420b9bf1ac986df9d84f04cfa32e6d285` |
| arm1 s2 u200 | `runs/BTC20261005-1856_v7p2_s2_arm1/ckpt_u200.pt` | `286e754648e9fc45e191f9406fc288a9645a5080f53a0216ce9778f2309ebe54` |
| P1-B e010（DAgger 起点） | `runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt` | `d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63` |
| 结构 B10（K-anchor 入口） | `runs/BTC20261005-0856_v7struct_v5/stage_b/ckpt_epoch010.pt` | `2f463a2186e41f523328902d26fcdcc1512d7885f6ea44fe18db0dc56ed9d55c` |
| 结构 fix-11 best e002 | `runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch002.pt` | `6372552ff8eef7074f2a1d5188368cb515780ce33ff1fed6881571be5d26bb86` |
| E-β″（v6 零点，参照） | `runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt` | `723db1c27d9da33ed8da17375af718cb421058e000a54555a2e5c667f5a56fa5` |
| 锚字典 K=6 | `config/plan_anchors_k6.json` | `79829ef705285c79a60252e9db6ac58d1b1e48830a4fe4d32327a1fa0f3873bf` |

### 4.2 runs/ 关键路径

- **P0**：`runs/BTC20260927-1839_eval500_baseline`（IDM 0.756）、`runs/BTC20261002-100413_v7p0_idm_clean500`（0.742）、`runs/BTC20261002-160154_v7p1b_tg45_idm`（0.778）。
- **P1**：DAgger 四窗口数据 `datasets/BTC20261002-2329_v7p1dagger_w{1..4}`、训练 `runs/BTC20261002-2329_v7p1dagger_w{1..4}`、冠军 clean500 `runs/BTC20261003-045912_v7p1dagger_w1_clean500`、eval500 探索 `runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory`；P1-B 链 `runs/BTC20261002-0941_v7p1b`。
- **P2/P3**：arm1 `runs/BTC20261003-0641_v7p2_s0_arm1` / `runs/BTC20261005-0601_v7p2_s11_arm1` / `runs/BTC20261005-1732_v7p2_s1_arm1` / `runs/BTC20261005-1856_v7p2_s2_arm1`；s11 洁净复验 `runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean` / `…-074755…eval500_clean` / `…-075959…tg45_clean`；sub150 闸与 keep-best 曲线见各 run 日志。
- **结构**：`runs/BTC20261005-0856_v7struct_v5`（A/B）、`runs/BTC20261005-0856_v7struct_v5_p3`（失败 phase3）、`runs/BTC20261005-0856_v7struct_v5_p3fix`（fix-11）、终评 `runs/BTC20261005-161813_v7p3fix_best_clean500` / `…-163040…tg45` / `…-163243…eval500`。
- **pre-v5 复验 worktree（已移除，2026-10-06）**：原 `/tmp/opencode/v7_pre_v5` @ `2f4450e`；如需复现 s1/s2/s11 与 P4-extra/奖励/§14 臂（同代码期），重建：`git worktree add <dir> 2f4450e`（`runs`/`datasets` 软链主树）。

### 4.3 数据 / 配置 / 工具 / 预注册

- **数据**：`datasets/BTC20261002-0941_expert5k_v4`、`…_v41`、`datasets/BTC20261002-2329_v7p1dagger_w{1..4}`、`datasets/BTC20261005-0814_expert5k_v5` / `…expert500val_v5`、`datasets/BTC20261005-0856_phase3_dagger_v5`（20,080 行）。
- **配置**：`config/arms/v7_arm1_offroad.yaml`、`v7_arm2_bundle_kl.yaml`、`v7_arm3_bundle_only.yaml`（未跑）、`v7_struct_b_v5_{train,model,eval}.yaml`、`config/plan_anchors_k6.json`。
- **工具**：`tools/paired_eval.py` + `tests/test_paired_eval.py`（配对 McNemar/bootstrap CI/方差闸/单 baseline fail-closed）、`tools/dagger_collect.py`（闭环重标注 + `assert_no_eval_val_overlap`）、`tools/fit_plan_anchors.py`。
- **预注册**：`docs/v7_program_prereg.md`（§0–§11：§9 arm1、§10 arm2/3、§11 K-anchor + §11.5 fix-11 修订锚）；`docs/experiments.md` §4（IDM 锚）；`.slim/deepwork/v7-beat-idm.md`（执行计划）。

### 4.4 证据/报告档（已入仓 `docs/v7_reports/`；索引/sha256 见 `docs/v7_reports/README.md`）

- P0：`docs/v7_reports/v7_p0_gap_decomposition.md`、`docs/v7_reports/v7_p0_idm_baseline.md`（入仓）；`v7_p0_gap_tables.md`、`v7_p0_idm_analysis.md`、`v7_p0_strat_{clean500,eval500}.md`（未入仓，已随 2026-10-06 `/tmp` 清理删除）。
- P1：`docs/v7_reports/v7_p1_dagger_cycle.md`、`docs/v7_reports/v7_p1_probe_v1v2.md`、`docs/v7_reports/v7_p1b_failure_diag.md`、`docs/v7_reports/v7_p1_iter2.md`、`docs/v7_reports/v7_p1_iter3.md`（入仓）；`v7_p1b_report.md`、`v7_p1a_obs_static.md`、`v7_p1_tollgate_triage.md`、`v7_p1_dagger_fix.md`（已清理）。
- P2/P3：`docs/v7_reports/v7_p2_arm1.md`、`docs/v7_reports/v7_p3_arm1_seeds.md`、`docs/v7_reports/v7_p2_s11_verify.md`（入仓）；`v7_p2_status.txt`、`v7_p3_status.txt`（已清理）。
- 结构：`docs/v7_reports/v7_kanchor_feasibility.md`、`docs/v7_reports/v7_struct_a_obs_v5.md`、`docs/v7_reports/v7_struct_a2_wiring_probe.md`、`docs/v7_reports/v7_struct_b_kanchor.md`、`docs/v7_reports/v7_struct_fail_diag.md`、`docs/v7_reports/v7_p3fix_retry.md`（入仓）；`v7_struct_retrain_v5.md`（未入仓，暂留 `/tmp/opencode/KEEP/`，见 README 未决项）、`v7_struct_retrain_chain.log`（已清理）。
- 驱动/脚本（不入 repo；已随 2026-10-06 清理删除）：`v7_p2_driver.py`、`v7_p3_driver_pre_v5.py`、`v7_struct_retrain_chain.py`、`v7_p3fix_driver2.py`、`v7_p0_gap_decomp.py`。

## 5. 未决与后续建议

1. **K-anchor 修复版（重开前置）**：① 修 specific 专家/router 训练（决定性破坏项）——重审 phase3/Stage B specific 配方（lr/损失/epoch guard；fix-11 的 `trunk_only`+clean150 守护已提供保底不崩路径）；② 锚头冻结契约已修（`55adf90`，含两 ckpt 逐位断言）；③ **LD 近场恢复**作为独立 A/B（v4.1→v5 有效槽 15.07→7.12、其他车道近场全丢）；④ 保留锚 CE/WTA 监督。**不承诺** tollgate 修复（模式选择偏移 JSD 0.32 仍在）。
2. **Δψ 尾段反馈**：把航向误差注入被执行尾段（或 tracker 消费锚链 step0）——当前尾段对 Δψ 零反馈（corr 0.05），是"锚计划左偏出界"的最小结构修复项。
3. **lane 入选择头**：选择头输入 latent 不含 lane token（融合仅 ego/OD/LD/others/nav/signal）；lane/ttc 已接线（`fcf047e`），应让选择头消费 lane 以对准 tollgate/merge 失败窗口（64%/43% 落 C2）。
4. **critic 修复 + TTS（test-time search / 多候选打分）**：参考 `.slim/deepwork/v7-beat-idm.md` Q3/Q4——8 通道 critic、多候选打分/TTS、goal 增广；当前 PPO 单 env × 256 steps/update 梯度噪声大，arm1 方差主导与 s0 末段坍缩提示优化稳定性是首要杠杆。
5. **多 seed 协议（下次主判据）**：n≥5（建议 8–10）加宽证据 + 方差闸；**success 增益必须与碰撞闸同判**（arm1 碰撞 +3.6~+11.6pp vs w1、s11/s2 超 IDM 锚）；keep-best 禁止 sub150 直采。
6. **未跑项与目标重估**：arm2/arm3（§10 已预注册）可选补做单变量归因；tollgate（P0 +7.0pt）/merge（+2.8pt）结构问题未解；当前最好 eval500 0.646 ⇒ 距 IDM −11.0pt，"超越 0.756"按现证据仍是低概率尾部，任何重开须先声明方差与安全闸预算。

## 6. 现场与复现

- **本报告 §2 关键数字已从盘上产物独立复算**（`episodes.csv` 逐 `(id,seed)`；`net=fixed−broken`、`z=|net|/√(fixed+broken)`；`success=arrive_dest`；绝对值另对照 `metrics.json::overall`）：
  - IDM 0.742/0.756/0.778、E-β″ 0.440/0.440、P1-B 0.314/0.312、w1 0.526/0.530；
  - arm1 四 seed clean500 Δ（+1.0/+14.2/+6.6/+10.4）与 eval500（0.512/0.646/0.550/0.622）逐位复核；mean/sd（+8.1/5.6、+5.3/6.2）复算一致；
  - 结构线 0.328/0.324/0.178 与配对（−19.8/−20.6/−60.0 vs w1/IDM）逐位复核；
  - arm0 vs IDM −13.4pt、w1 vs E-β″ +8.6/+9.0、w1 vs P1-B +21.2/+21.8 复算一致。
- **复算入口**：`runs/*/episodes.csv`（配对）+ `runs/*/metrics.json`（overall）+ `tools/paired_eval.py --baseline <b>/episodes.csv --agent <a>/episodes.csv --out-dir <dir>`；配对报告副本原在 `/tmp/opencode/v7_p2_paired/`、`/tmp/opencode/v7_struct_retrain_paired/`、`/tmp/opencode/v7_finalcheck/`（已清理，可由上述命令重生成）。
- **未跑/未决（如实）**：arm2/arm3 未跑（预注册在案）；Gate D（≥5 seed 主判据）未启动；结构线关闭（补救版保留在盘）；seed11 tg45 0.0 / T3 0/9 为洁净复验值（原 dirty 期 void）；tg45 上 IDM 教师天花板 0.778（评测口径）未被任何 v7 候选接近。
- **tag**：`v7-close-20261006`（指向本报告入库 commit；tag 说明含"目标未达"的诚实陈述与最佳产物清单：w1 e005 / arm1 s11 u150）。

## 7. 版本与 tag

- 本报告随 v7 收尾 commit 入库；`git tag -a v7-close-20261006` 指向该 commit（`git show v7-close-20261006` 查看 tag 说明）。
- 前置 commit 链（收尾时 HEAD）：`1ea339d`（§11 修订锚 fix-11）← `55adf90`（freeze 契约修复 + trunk_only + guard）← `0065264`（结构 B 重训臂配置）← `3ac92d7`/`583efee`（K-anchor 透传/断言）← `489091b`/`f42d648`（结构 B K-anchor）← `326c260`/`fcf047e`（A2 接线）← `aa4c69c`/`0291f3a`（A obs v5）← `2f4450e`/`a1b6950`（arm2/3 预注册）← `c37acbb`/`2ddfa22`（P2 arm1 预注册+奖励）← `4a46c5e`（DAgger 前置）← `2e13dfa`/`386b091`（P1 iter2）← `ee763f4`（fix-15）← `b99e127`（obs v4）← `f5f0dc3`/`0454308`（P0 修正集 #1）← `b711e5f`（P0 工具+预注册）。
- **增补（2026-10-06）**：§8（夜间/今晨结果：碰撞抑制 / 碰撞类型 / tollgate 图 / 奖励臂 A-C / §14）+ tag `v7-close-2-20261006`。

## 8. 增补（2026-10-06；夜间/今晨定向结果：碰撞抑制、碰撞类型、tollgate 可视化、奖励臂 A/B/C、§14 ttc）

> 本节为主报告收尾后的增补（§0–§7 保持 2026-10-05 原文）；新增预注册 §12–§14 见 `docs/v7_program_prereg.md`。所有数字已从盘上产物独立复算：`runs/*/episodes.csv` 逐 `(id,seed)`（success/collision/off_road/speed_ratio/jerk 与 Δpp），配对 p/z 取 `tools/paired_eval.py` 输出 JSON（原 `/tmp/opencode/v7_p4extra_paired/`、`/tmp/opencode/v7_s14_paired/`，已清理、可重生成）；ckpt/config sha256 对盘复核一致。

### 8.1 P4-extra（§12）：s11 + `terminal_values.collision` −22→−32（单变量；seeds 0/11）

Base = s11 u150（`a7cc091f…`）；臂 `config/arms/v7_arm1_collision_suppress.yaml`（`b8697d54…` = arm1 逐位不变 + collision −32）；预注册内容 `eec0abe` / 锚 `ee1ee49`；pre-v5 worktree `2f4450e`。

| seed | 采纳 | clean500 succ / coll | eval500 succ / coll | 判定 |
|---|---|---|---|---|
| 0 | u75 | **0.678 / 0.154** | **0.652 / 0.164** | partial（clean 过；eval coll ✗） |
| 11 | u50 | 0.654 / 0.160 | 0.646 / 0.154 | partial（同上） |
| s11（基线） | u150 | 0.668 / 0.182 | 0.646 / 0.202 | — |
| IDM（锚） | — | 0.742 / 0.174 | 0.756 / 0.144 | — |

- 配对 vs s11：clean coll **−2.8pp**（z 2.06，p 0.054）/ **−2.2pp**（z 1.46，p 0.185）；eval coll **−3.8pp**（z 2.47，p 0.018）/ **−4.8pp**（z 3.05，p 0.0032）；success 全 ns（clean +1.0 / −1.4pp；eval +0.6 / 0.0pp）。
- 判据（§12.3，PASS = 四条全过）：clean500 coll ≤0.174 且 succ ≥0.638 **四条全过**；eval500 仅 collision 未过（0.164 / 0.154 vs 0.144，差 +2.0 / +1.0pp）⇒ **partial 2/2**（不宣称过安全闸）。
- 分布（n=2）：clean Δcoll mean −2.5pp / sd 0.4pt；eval Δcoll mean −4.3pp / sd 0.7pt；u50+u100 双点闸 2/2 未触发、无 void。
- 采纳 ckpt sha256：s0 u75 `4217abe0d67cc231e48b1fe783c3d2fbd87dac5022a091c71d7b1cf4f54dd342`；s11 u50 `711e96a711c1d96b758520dfc76f9dfb78e89eed21106e7a3b0894136bfd387e`。

### 8.2 碰撞类型（Q6）：s11 eval500 101 条碰撞仪器化重放（101/101 复现）

- 重放协议：`env/specs/scenarios_eval500.json` 101 条 collision；pre-v5 `2f4450e`；tracker=lqr / plan / max_steps=1000；termination+steps 与冻结 CSV 逐条一致（`csv_mismatch=[]`）。

| 类别 | n | 占比 |
|---|---|---|
| **自车追尾前车（ego_rear_ends_lead）** | **64** | **63.4%** |
| 撞静态建筑（岗亭/建筑，static_building；17/17 全 tollgate） | 17 | 16.8% |
| 侧碰/cut-in（side_cutin 8 + side 6） | 14 | 13.9% |
| 后车撞自车（rear_by_follower；5/5 对手 IDMPolicy） | 5 | 5.0% |
| 撞路缘/人行道（static_sidewalk） | 1 | 1.0% |

- 按 primary：tollgate 25（17 撞亭 + 8 追尾）、curve 14、straight 14、ramp_out 13、ramp_in 10、split 9、t_intersection 7（4 后车撞）、merge 5、roundabout 2、uturn 1、intersection 1。
- **追尾证据（64 条）**：自车 2s 内 brake_frac>0 = **0 条**（从不刹车）、平均 a_lon −0.56 m/s²（减速过缓）；前车加速（>0.5 m/s²）39 条、2s 急刹（<−2）0 条、2s 历史缺失（新生成）15 条 ⇒ 无"保持距离/提前减速"激励，动作分布上根本没有刹车。
- 结论：**碰撞主因是自车追尾（63.4%）且"几乎不刹车"**；后车撞自车仅 5%。修复方向应是纵向直接机制（近失/间距/刹车动作），而非横向。

### 8.3 tollgate 双面板（Q2，spec 34 / s11 u150）

- 图 ×4（2100px；桌面 app 打开；已入仓 `docs/v7_reports/figures/`）：`tollgate_spec34_A_first_sighting_step0821.png`、`…B_decision_zone_39m_step0869.png`、`…C_gate_entry_step0929.png`、`…D_final_crash_step0961.png`。
- 运行时度量（原 JSON `/tmp/opencode/v7_q6q2/tollgate_viz.json`，已清理、暂留 `/tmp/opencode/KEEP/v7_q6q2/`）：
  - 岗亭 lane_id=1（奇数车道正中，lat≈0，宽 3.5 m）；`$` block 实测限速 **5.6 m/s**；
  - `others.static` 首次 present=1：step 821 / **54.71 m**（早于 IDM 首扫 43.819 m / step 362）；
  - `road_class=tollgate` 首次=1：step 930 / **16.995 m**（进入 `$` block step 929 / 17.257 m 之后——变道决策点 25–39 m 时仍为 0，**信号迟到**）；
  - IDM 首次变道 step 378 / **34.653 m**（成功 arrive_dest @458）；s11 不变道、撞亭时 ego_v **0.45 m/s**（throttle 0.86 / a_lon −1.21，仍为 building 碰撞）。
- 结论：**信号在（static 54.7m）而行为不在**——LD 16 槽只编码车道中心线（岗亭是 BaseBuilding，不在 LD，无专用 tollgate 特征）；static 信号已在却未触发变道 ⇒ tollgate 是"决策/结构"问题，非感知缺失。注：s11 冻结协议 LD offset={5,10,15,20,30}（pre-v5）；{20,40,60,80} 属已关闭的 v5 结构线。

### 8.4 奖励单变量臂 A/B/C（§13；base = s11 u150）

臂 `v7_reward_{A,B,C}.yaml`（`32b57bb1…` / `704de4f2…` / `d817110f…`）；预注册内容 `db98501` / 锚 `23ae105`。

| 臂 | 单变量 | seed | clean succ / coll / off | eval succ / coll | 判定 |
|---|---|---|---|---|---|
| A | `off_road_edge.edge_scale_m` 1.0→2.5 | 0 | 0.646 / 0.158 / 0.192 | 0.614 / 0.180 | fail |
| A | 同上 | 11 | 0.668 / 0.152 / 0.146 | 0.644 / 0.162 | fail |
| B | 追加 `speed_deficit`（weight −0.3） | 0 | 0.658 / 0.138 / 0.144 | 0.638 / 0.142 | 弱 pass（仅替代条款） |
| B | 同上 | 11 | 0.666 / 0.166 / 0.156 | 0.626 / 0.178 | 弱 pass（仅替代条款） |
| C | 追加 `comfort_jerk_win`（−0.1） | 0 | 0.654 / 0.138 / 0.148 | 0.660 / 0.130 | fail |

- **A fail（2/2）**：off_road 反向 **+5.4pp**（z 4.02，p 6.6e−05）/ +0.8pp（判据要求 ≤−3pt）；collision −2.4 / −3.0pp（顺带降低）；success −2.2 / 0.0pp。
- **B 弱 pass（2/2，仅替代条款）**：success 0.658 / 0.666 ≥ s11−3pp（0.638）→ 判据路径②过；但 **speed_ratio Δ −0.053 / −0.013**（路径①要求 ≥+0.05）未达 ⇒"不提速"，机制目标未实现。
- **C fail**：jerk_p95 29.08（**仅 −3.3%**，判据 ≥20% 降幅）；seed11 未跑（§13.4 预算，not-run）。
- 结论：三臂均未产生目标机制改善（A 甚至反向）；B 说明 **speed_ratio 收益项对策略约束力不足**（risk 项主导），速度机制需更直接手段（更强罚 / 进度×速度耦合 / 直接动作机制）。

### 8.5 §14：s11 + `ttc`（TTCLeadPenalty −0.5 / 2.0 / 0.5；seeds 0/11）

臂 `config/arms/v7_arm1_ttc.yaml`（`8d68b3b3…` = arm1 逐位不变 + ttc 稠密近失罚，terminal collision 保持 −22）；预注册内容 `7bf18d1` / 锚 `5db385e`。

| seed | 采纳 | clean500 succ / coll | eval500 succ / coll | 判定 |
|---|---|---|---|---|
| 0 | u75 | 0.648 / 0.194 | 0.620 / 0.194 | **fail**（clean coll +1.2pp 反向；succ −2.0pp） |
| 11 | u25 | **0.672 / 0.152** | 0.648 / **0.156** | **PASS**（clean coll −3.0pp p=0.04；eval −4.6pp） |

- seed11：clean coll 配对 **−3.0pp**（z 2.19，p 0.040）；eval coll 0.156（vs s11 0.202，−4.6pp）；success 持平（clean +0.4 / eval +0.2pp，ns）。
- **臂判定：混合（1 fail / 1 PASS）→ 未达 2/2 PASS**；ttc 是首个直击"追尾"主因的奖励项，但 seed 间不一致（seed0 反向）。
- 采纳 ckpt sha256：s11 u25 `229bbc1e37c6e6fbfbfcdcc96fa6426c6fd0f9f86412d78cecd1c2f088e29b7f`；s0 u75 `0225c2d35f957f3ed960026c59eca21dc25e81dc7b159ed41e7eebebd8e4b9d6`。

### 8.6 增补结论

1. **碰撞抑制 = 可复现的确定性收益**：`terminal collision −32` 在 2/2 seeds 上 clean500 coll −2.2~−2.8pp（≤IDM 0.174）、eval500 −3.8~−4.8pp（显著，p≤0.02），**success 不降**（ns）；距 eval500 IDM 闸（0.144）仍差 1–2pp ⇒ 按预注册记 partial，但已是 v7 内**首个"安全改善 + 性能保持"的可采纳产物**（s0u75 / s11u50；§14 s11u25 为同类单点证据）。
2. **追尾 63.4% 是主因且"几乎不刹车"**（brake_frac=0.00）⇒ 纵向直接机制（刹车 / 间距 / 近失）是碰撞面的第一杠杆；后车撞自车（5.0%）不构成主因。
3. **tollgate：信号在（static 54.7m，早于 IDM 43.8m）而行为不在**（LD 无专用特征、road_class 17.0m 才置位、s11 不变道撞亭）⇒ 属决策/结构缺陷；本夜仅可视化定论，未修复。
4. **奖励臂 A/C 无效、B 弱**（速度机制需更直接手段）；§14 ttc 混合（1/2 PASS）。
5. **主目标（超越 IDM 0.756）仍未达**：本轮成功读数最高为 P4-extra s0 u75（clean **0.678** / eval **0.652**，均略高于 s11 的 0.668/0.646），但其 eval collision 0.164 未过 IDM 闸（0.144），按预注册记 partial——安全与成功仍未同闸通过（距 IDM：clean −6.4pp / eval −10.4pp）。

### 8.7 资产清单增补

- **ckpt sha256（盘上复核）**：

| 资产 | 路径 | sha256 |
|---|---|---|
| **P4-extra s0 u75（安全改善版）** | `runs/BTC20261005-2138_v7p4extra_s0_colls/ckpt_u075.pt` | `4217abe0d67cc231e48b1fe783c3d2fbd87dac5022a091c71d7b1cf4f54dd342` |
| **P4-extra s11 u50（安全改善版）** | `runs/BTC20261005-2243_v7p4extra_s11_colls/ckpt_u050.pt` | `711e96a711c1d96b758520dfc76f9dfb78e89eed21106e7a3b0894136bfd387e` |
| §14 s11 u25（单点 PASS） | `runs/BTC20261006-0711_v7s14_s11_ttc/ckpt_u025.pt` | `229bbc1e37c6e6fbfbfcdcc96fa6426c6fd0f9f86412d78cecd1c2f088e29b7f` |
| §14 s0 u75（fail 参照） | `runs/BTC20261006-0615_v7s14_s0_ttc/ckpt_u075.pt` | `0225c2d35f957f3ed960026c59eca21dc25e81dc7b159ed41e7eebebd8e4b9d6` |
| 奖励 A s0 u25 / s11 u100 | `runs/BTC20261006-0027_v7reward_A_s0/ckpt_u025.pt` / `runs/BTC20261006-0127_v7reward_A_s11/ckpt_u100.pt` | `0806c224…` / `b5b63e78…` |
| 奖励 B s0 u25 / s11 u25 | `runs/BTC20261006-0233_v7reward_B_s0/ckpt_u025.pt` / `runs/BTC20261006-0342_v7reward_B_s11/ckpt_u025.pt` | `f795a3d3…` / `43c896af…` |
| 奖励 C s0 u25 | `runs/BTC20261006-0445_v7reward_C_s0/ckpt_u025.pt` | `b4557630…` |

- **配置**：`config/arms/v7_arm1_collision_suppress.yaml`（`b8697d54…`）、`v7_arm1_ttc.yaml`（`8d68b3b3…`）、`v7_reward_A/B/C.yaml`（`32b57bb1…` / `704de4f2…` / `d817110f…`）。
- **预注册**：`docs/v7_program_prereg.md` §12（内容 `eec0abe` / 锚 `ee1ee49`）、§13（`db98501` / `23ae105`）、§14（`7bf18d1` / `5db385e`）。
- **runs/ 关键路径**：P4-extra `runs/BTC20261005-2138_v7p4extra_s0_colls` / `runs/BTC20261005-2243_v7p4extra_s11_colls`（clean/eval/tg45 子 run 见 §8.1）；§14 `runs/BTC20261006-0615_v7s14_s0_ttc` / `runs/BTC20261006-0711_v7s14_s11_ttc`；奖励 `runs/BTC20261006-{0027,0127,0233,0342,0445}_v7reward_{A_s0,A_s11,B_s0,B_s11,C_s0}`。
- **报告/证据档（已入仓 `docs/v7_reports/`，索引见 `docs/v7_reports/README.md`）**：`v7_p4extra_collision.md`、`v7_q6_collision_types.md`、`v7_q2_tollgate_figure.md`、`v7_p4extra_rewards.md`、`v7_s14_ttc.md`、`v7_morning_brief.md`、`v7_night_watch.log`；JSON `v7_q6q2/collision_replay.json`、`v7_q6q2/tollgate_viz.json`（原 /tmp 档已清理，暂留 `/tmp/opencode/KEEP/v7_q6q2/`）；配对目录 `v7_p4extra_paired/`、`v7_s14_paired/`（已清理；可由 `runs/*/episodes.csv` + `tools/paired_eval.py` 重生成）。
- **图（已入仓）**：`docs/v7_reports/figures/tollgate_spec34_{A_first_sighting_step0821,B_decision_zone_39m_step0869,C_gate_entry_step0929,D_final_crash_step0961}.png`（×4）。

### 8.8 版本与 tag（增补）

- 本节随增补 commit 入库；`git tag -a v7-close-2-20261006` 指向该 commit（`git show v7-close-2-20261006` 查看说明；含夜间结果与"安全改善版"产物清单）。
- 增补期 commit 链：`5db385e`（§14 锚）← `7bf18d1`（§14 ttc 臂）← `23ae105`（§13 锚）← `db98501`（奖励臂 A/B/C）← `ee1ee49`（§12 锚）← `eec0abe`（P4-extra 碰撞抑制臂）← `9635591`（v7 收尾报告 + tag `v7-close-20261006`）。
