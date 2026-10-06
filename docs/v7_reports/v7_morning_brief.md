# v7 晨间 review 包（夜间 23:00–06:10 产出汇总）

> 生成：2026-10-06 06:10；夜间队列：P4-extra（碰撞抑制）→ fix-3 诊断（碰撞类型 + tollgate 图）→ fix-4 奖励臂 A/B/C → fix-5 §14（ttc 臂，进行中）。
> 所有任务 setsid 解耦；时间线见 `/tmp/opencode/v7_night_watch.log`。

## 0. TL;DR

1. **碰撞类型定论**（101 条仪器化重放，101/101 复现）：**自车追尾前车 63.4%**（共性：减速过缓、**brake_frac=0.00 从不刹车**）；后车撞自车仅 **5.0%**；侧碰/cut-in 13.9%；**撞岗亭建筑 16.8%（17/101，全部 tollgate）**。⇒ 你问的二选一：**是"自车没刹住追尾前车"（63.4%），不是"后车撞自车"（5%）**。
2. **tollgate 结论**：LD **无** tollgate 特征（岗亭是 BaseBuilding）；专用建模=`others.static`（首次 present 在 **54.71m**，比 IDM 的 43.8m 还早）+ `road_class`（**17.0m 才置位，太迟**）；IDM 在 34.65m 变道成功，s11 不变道撞亭。**信号在、行为不在**。实测 `$` 限速 = **5.6 m/s**（此前说的 3 m/s 不对，已纠正）。
3. **碰撞抑制臂（P4-extra，s11+collision −32）**：**partial 2/2**——clean500 闸**过**（coll 0.154/0.160 ≤0.174；succ 0.678/0.654）；eval500 碰撞 0.164/0.154 vs IDM 0.144 **差 1–2pp**；**碰撞 −2.2~−4.8pp（eval 显著）且 success 不回吐**。
4. **奖励臂 A/B/C**：A（edge scale 2.5）**fail**（off-road 反向 +5.4/+0.8pp；collision 反降）；B（speed_deficit −0.3）**弱 pass**（success 不劣，但 speed_ratio 未达——仅替代条款）；C（窗口化 jerk）**fail**（jerk 仅 −3.3%）。
5. **§14（进行中）**：s11 + `ttc`（−0.5）稠密近失罚——直击"自车追尾"根因（arm1 bundle 此前不含 ttc）。

## 1. P4-extra（碰撞抑制）`/tmp/opencode/v7_p4extra_collision.md`

| seed | 采纳 | clean succ/coll | eval succ/coll | 判定 |
|---|---|---|---|---|
| 0 | u75 | 0.678 / **0.154** | 0.652 / **0.164** | partial |
| 11 | u50 | 0.654 / **0.160** | 0.646 / **0.154** | partial |
| s11（基线） | u150 | 0.668 / 0.182 | 0.646 / 0.202 | — |
| IDM | — | 0.742 / 0.174 | 0.756 / 0.144 | — |

- 配对：collision vs s11 = clean −2.8pp(p=0.054)/−2.2pp；**eval −3.8pp(p=0.018)/−4.8pp(p=0.0032) 显著**；success 全 ns。
- 安全闸：clean 过、eval 差 1–2pp（不宣称过闸，按预注册记 partial）。

## 2. 碰撞类型（Q6）`/tmp/opencode/v7_q6_collision_types.md`

| 类别 | n | 占比 |
|---|---|---|
| **自车追尾前车** | **64** | **63.4%** |
| 撞静态建筑（岗亭/建筑） | 17 | 16.8% |
| 侧碰/cut-in | 14 | 13.9% |
| 后车撞自车 | 5 | 5.0% |
| 路缘/人行道 | 1 | 1.0% |

- 按几何：tollgate 25（17 撞亭 + 8 追尾）、curve 14（13 追尾）、ramp_out 13（10 追尾）、straight 14（12 追尾）、t_intersection 7（4 后车撞）、merge 5。
- 追尾案例证据：`ego_a_lon_2s ≈ −0.4~−0.9`（减速不足）、`brake_frac=0.00`、部分前车加速中 ⇒ 无"保持距离/提前减速"激励。
- 后车撞自车案例（5 条，全部 t_intersection 等交叉场景，对手全 IDMPolicy）：自车 throttle_min 0.49–1.0、不加速离场。

## 3. tollgate 双面板图（Q2）`/tmp/opencode/v7_q2_tollgate_figure.md`

- **PNG（4 张，2100px 宽；请在桌面 app 打开——preview 需桌面浏览器连接）**：
  - `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_A_first_sighting_step0821.png`（首次 static present）
  - `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_B_decision_zone_39m_step0869.png`（决策区 ~39m）
  - `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_C_gate_entry_step0929.png`（进闸口）
  - `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_D_final_crash_step0961.png`（撞亭）
- 关键度量：岗亭 lane_id=1（奇数车道正中，lat≈0，宽 3.5m）；`$` 限速 **5.6 m/s**；static present 首现 **54.71m**；road_class 首现 **16.995m**；IDM 首扫 43.819m、首变道 **34.653m**（成功）；s11 到亭前 0.45 m/s 仍撞（throttle 0.86、a_lon −1.21）。
- 注：s11 冻结协议（pre-v5 `2f4450e`）LD offset 实际为 **{5,10,15,20,30}**（图中按实际标注）；v5 的 {20,40,60,80} 属已关闭的结构线。

## 4. 奖励臂 A/B/C（§13）`/tmp/opencode/v7_p4extra_rewards.md`

| 臂 | 结果 | 关键读数 |
|---|---|---|
| A `off_road_edge` scale 1.0→2.5 | **fail（2/2）** | success Δ −2.2/0.0pp；**off_road Δ +5.4/+0.8pp（反向）**；collision −2.4/−3.0pp |
| B `speed_deficit` −0.3 | **弱 pass（2/2，仅替代条款）** | success 0.658/0.666（≥0.638）但 **speed_ratio Δ −0.053/−0.013（未达）** |
| C `comfort_jerk_win` −0.1 | **fail（seed0）** | jerk_p95 29.08（**仅 −3.3%**，需 ≥20%）；seed11 未跑（预算） |

- 结论：三臂均未产生目标机制改善（A 甚至反向）；B 的"不提速"说明 **speed_ratio 的收益项对策略的约束力不足**（risk 项主导），需要更直接的机制（如更强罚/进度项改造）。

## 5. 在飞与后续（建议）

- **§14（进行中，~2.5h）**：s11 + `ttc`(−0.5)——预期直击 63.4% 追尾；判据：collision 再降 ≥2pp 且 success ≥ s11−3pp。
- **建议（待你定）**：
  1. 若 §14 有效 ⇒ 与 P4-extra（−32）组合做 2 变量臂（预注册）→ 冲"eval500 碰撞 ≤0.144"的安全闸；
  2. **tollgate 结构修复**（唯一还没做对的 +7pt 项）：static 信号在（54.7m）但无变道行为——建议在 **v5 线修复版**上做（K-anchor freeze 修复 + 自由车道目标 + specific 专家 guard），或先做一个"static 信号 → 变道决策"的专项监督探针；
  3. 速度机制（B 的失败说明需要直接机制）：`speed_deficit` 权重加大（−0.5~−0.8）或改成"进度×速度"耦合项；jerk 窗口化（C）权重加大/换 p95。
- **预算**：v7 100h 约至 10-06 12:00；剩余 ~6h。
