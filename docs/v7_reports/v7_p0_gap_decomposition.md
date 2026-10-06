# v7-P0：差距分解（我方策略 vs IDM 分层剖面）

- 日期：2026-10-02（CPU-only，只读；未改 repo）
- 可复算脚本：`/tmp/opencode/v7_p0_gap_decomp.py`
- 完整数字：`/tmp/opencode/v7_p0_gap_decomp.json`；全部表格：`/tmp/opencode/v7_p0_gap_tables.md`
- 复算：`python3 /tmp/opencode/v7_p0_gap_decomp.py`

---

## 0. 数据与口径（含 caveat）

| 对象 | 来源 | n | success |
|---|---|---|---|
| IDM（PurePursuitIDM，冻结基线） | `runs/BTC20260927-1839_eval500_baseline/episodes.csv` | 500 | 0.756 |
| arm0（RL u200，主对象） | `runs/BTC20261002-0026_p4arm0_eval500/episodes.csv` | 500 | 0.622 |
| arm5（RL u200） | `runs/BTC20261002-0303_p4arm5_eval500/episodes.csv` | 500 | 0.558 |
| 零点 IL（E-β″，v6p3 e3） | `runs/BTC20261001-200331_v6p3_e3_eval500/episodes.csv` | 500 | 0.440 |
| seed11 u200（clean500，**08:44 完成**） | `runs/BTC20261002-083129_v6p4_seed11_u200_clean500/episodes.csv` | 500 | 0.322 |
| （附录）arm0/arm5 clean500 | `runs/BTC20261002-0026_p4arm0_u200_clean500` / `…0303_p4arm5_u200_clean500` | 500 | 0.638 / 0.594 |

口径与校验：
- eval500 = `env/specs/scenarios_eval500.json`；`primary` 与 spec `labels.geometry`、`difficulty` 逐条一致（脚本断言，0 mismatch）；4 类终止与 `success` 一一对应，无 error 行。
- IDM 基线产自 **2026-09-27（git 54f8c5e）**，我方 RL eval 为 10-02（b657fea/5a26411），**存在版本差**；但 spec/seed/tracker(lqr)/limit 相同，且 0.756 与 `docs/experiments.md` §4 冻结数字一致。
- **并行 lane 的 `/tmp/opencode/v7_p0_idm_baseline.md` 截至 08:46 未就绪**；本报告直接使用盘上精确 IDM eval500 产物（优于 §4 聚合分层）。若 lane 复测不同，以复测为准。
- clean500 是 `val − eval500` 留出 500 条（不同 spec、不同 seed），**无匹配 IDM 对照**，仅作对象间比较与稳健性。
- 分层交叉表以两个二维边际给出（几何×难度 success、几何×终止类），三维逐 cell 见 §2 的 top 亏项分解。
- 附录对照：`datasets/BTC20260927-1734_expert500val/report.json`（同 500 场景的专家采集成功率，tollgate=0.644）与 `scenarios_train_5k.json` 几何覆盖（tollgate 455/5000=9.1%）——说明 tollgate 不是数据缺失问题。

---

## 1. 分层交叉表

### 1.1 几何×难度 success（n/成功）

IDM（冻结基线）：

| geometry | easy | medium | hard | 合计 |
|---|---|---|---|---|
| split | 16/16 | 11/14 | 10/15 | 37/45 |
| straight | 14/14 | 16/17 | 4/14 | 34/45 |
| ramp_out | 10/10 | 0/13 | 16/23 | 26/46 |
| ramp_in | 16/16 | 5/15 | 7/15 | 28/46 |
| merge | 18/18 | 11/13 | 11/15 | 40/46 |
| intersection | 18/19 | 12/20 | 2/7 | 32/46 |
| t_intersection | 16/16 | 11/11 | 12/18 | 39/45 |
| curve | 17/17 | 12/14 | 5/15 | 34/46 |
| uturn | 16/17 | 11/13 | 12/15 | 39/45 |
| roundabout | 26/28 | 8/17 | - | 34/45 |
| tollgate | 12/12 | 16/21 | 7/12 | 35/45 |

arm0（RL 主对象）：

| geometry | easy | medium | hard | 合计 |
|---|---|---|---|---|
| split | 16/16 | 14/14 | 11/15 | 41/45 |
| straight | 14/14 | 17/17 | 2/14 | 33/45 |
| ramp_out | 10/10 | 0/13 | 16/23 | 26/46 |
| ramp_in | 16/16 | 5/15 | 4/15 | 25/46 |
| merge | 10/18 | 8/13 | 8/15 | 26/46 |
| intersection | 18/19 | 13/20 | 6/7 | 37/46 |
| t_intersection | 8/16 | 9/11 | 9/18 | 26/45 |
| curve | 16/17 | 11/14 | 1/15 | 28/46 |
| uturn | 16/17 | 12/13 | 14/15 | 42/45 |
| roundabout | 24/28 | 3/17 | - | 27/45 |
| tollgate | **0/12** | **0/21** | **0/12** | **0/45** |

arm5 / IL-Ebpp 的同表见 `v7_p0_gap_tables.md` T1。要点：arm5 与 arm0 同构但更差（tollgate 仍 0/45；t_intersection 18/45；ramp_out 20/46）；IL 的失败面更宽（curve 11/46、roundabout 16/45、uturn 20/45、tollgate 0/45）。

### 1.2 几何×终止类（arm0 vs IDM，计数(rate)）

IDM：

| geometry | n | success | collision | off_road | max_step |
|---|---|---|---|---|---|
| split | 45 | 37(.822) | 7(.156) | 0 | 1(.022) |
| straight | 45 | 34(.756) | 10(.222) | 0 | 1(.022) |
| ramp_out | 46 | 26(.565) | 7(.152) | 13(.283) | 0 |
| ramp_in | 46 | 28(.609) | 5(.109) | 10(.217) | 3(.065) |
| merge | 46 | 40(.870) | 6(.130) | 0 | 0 |
| intersection | 46 | 32(.696) | 12(.261) | 2(.043) | 0 |
| t_intersection | 45 | 39(.867) | 6(.133) | 0 | 0 |
| curve | 46 | 34(.739) | 6(.130) | 0 | 6(.130) |
| uturn | 45 | 39(.867) | 6(.133) | 0 | 0 |
| roundabout | 45 | 34(.756) | 4(.089) | 7(.156) | 0 |
| tollgate | 45 | 35(.778) | 3(.067) | 2(.044) | 5(.111) |

arm0：

| geometry | n | success | collision | off_road | max_step |
|---|---|---|---|---|---|
| split | 45 | 41(.911) | 4(.089) | 0 | 0 |
| straight | 45 | 33(.733) | 12(.267) | 0 | 0 |
| ramp_out | 46 | 26(.565) | 7(.152) | 13(.283) | 0 |
| ramp_in | 46 | 25(.543) | 11(.239) | 10(.217) | 0 |
| merge | 46 | 26(.565) | 3(.065) | **17(.370)** | 0 |
| intersection | 46 | 37(.804) | 0 | 6(.130) | 3(.065) |
| t_intersection | 45 | 26(.578) | 3(.067) | **16(.356)** | 0 |
| curve | 46 | 28(.609) | 13(.283) | 0 | 5(.109) |
| uturn | 45 | 42(.933) | 1(.022) | 2(.044) | 0 |
| roundabout | 45 | 27(.600) | 2(.044) | 8(.178) | 8(.178) |
| tollgate | 45 | **0** | **21(.467)** | **24(.533)** | 0 |

**总体形态**：arm0 的 collision(0.154) 与 IDM(0.144) 接近，**off_road 0.192 vs 0.068 是主要超额失败**；max_step 同为 0.032 但形态完全不同（见 §4）。arm5 的 collision(0.200)/off_road(0.224) 双高；IL 几乎全是 off_road(0.498)。

---

## 2. 亏项归因：success 差距分解到 +pt

口径：`gap = succ_IDM − succ_ours`（pt）；按几何的贡献 = `(A−B)/500×100`，其中 A = {IDM 成 & 我败}（每条 +0.2pt）、B = {IDM 败 & 我成}（每条 −0.2pt）；Σ 恒等于总 gap（脚本断言）。

### 2.1 arm0（gap = **13.4pt**；A=107，B=40）

| geometry | succ_IDM | succ_arm0 | A | B | pt 贡献 |
|---|---|---|---|---|---|
| **tollgate** | .778 | .000 | 35 | 0 | **+7.0** |
| **merge** | .870 | .565 | 17 | 3 | **+2.8** |
| **t_intersection** | .867 | .578 | 18 | 5 | **+2.6** |
| roundabout | .756 | .600 | 8 | 1 | +1.4 |
| curve | .739 | .609 | 8 | 2 | +1.2 |
| ramp_in | .609 | .543 | 5 | 2 | +0.6 |
| straight | .756 | .733 | 2 | 1 | +0.2 |
| ramp_out | .565 | .565 | 4 | 4 | 0.0 |
| split | .822 | .911 | 3 | 7 | −0.8 |
| intersection | .696 | .804 | 4 | 9 | −1.0 |
| uturn | .867 | .933 | 3 | 6 | −0.6 |

- A 按我方终止类：**off_road 56（11.2pt gross）/ collision 40（8.0pt）/ max_step 11（2.2pt）**；B=40（−8.0pt）集中在 split/intersection/uturn（我方已强于 IDM，勿回退）。
- **去掉 tollgate 后 gap 仍 7.0pt**（IDM .754 vs arm0 .683）：merge+t_intersection 就占 5.4pt。
- "IDM 成功条件下我方仍失败"的条件失败率：tollgate **35/35=1.00**、t_intersection 18/39=0.46、merge 17/40=0.43、roundabout 8/34=0.24、curve 8/34=0.24、其余 ≤0.18。
- A 的难度分布意外地平：easy 34 / medium 33 / hard 40 → **不是单纯"难题"**，tollgate 和 t_intersection 的 easy 也在丢分。

Top (geometry|difficulty|我方终止类) 亏项类（arm0）：

| class | 条数 | pt |
|---|---|---|
| tollgate / medium / off_road | 9 | 1.8 |
| t_intersection / easy / off_road | 8 | 1.6 |
| merge / easy / off_road | 8 | 1.6 |
| tollgate / medium / collision | 7 | 1.4 |
| tollgate / easy / off_road | 7 | 1.4 |
| t_intersection / hard / off_road | 5 | 1.0 |
| ramp_in / hard / collision | 5 | 1.0 |
| tollgate / easy / collision | 5 | 1.0 |
| merge / medium / off_road | 4 | 0.8 |
| tollgate / hard / off_road | 4 | 0.8 |

### 2.2 arm5（gap = **19.8pt**；A=128，B=29）

按几何 pt：tollgate **+7.0**、t_intersection **+4.2**、merge **+3.2**、ramp_out +1.2、roundabout +1.0、curve +0.8、ramp_in +0.6、uturn +0.6、intersection +0.6、split +0.4、straight +0.2。
A 按终止类：collision 60 / off_road 62 / max_step 6。Top 类：tollgate|easy|collision 11(2.2pt)、tollgate|medium|collision 10(2.0)、ramp_out|hard|collision 9(1.8)、t_intersection|easy|off_road 8(1.6)、t_intersection|hard|off_road 8(1.6)、merge|easy|off_road 8(1.6)。

### 2.3 IL-Ebpp（gap = **31.6pt**；A=188，B=30）

按几何 pt：tollgate **+7.0**、curve **+4.6**、t_intersection **+4.6**、uturn +3.8、roundabout +3.6、merge +3.4、straight +2.2、intersection +1.6、split +0.4、ramp_out +0.2、ramp_in +0.2。
A 按终止类：off_road 168 / collision 17 / max_step 3 → IL 的亏项几乎全是**横向出界**。

### 2.4 IL → arm0 的改进（RL 修复了什么）

IL 败而 arm0 成 **101 条**（+20.2pt），IL 成而 arm0 败 10 条（−2.0pt），净 **+18.2pt**。修复集中在 off_road 类：uturn 22、curve 17、intersection 13、t_intersection 12、roundabout 11、straight 11、split 7。**tollgate 一条未修复（IL 0/45 → arm0 0/45）**——RL 配方对绝大多数几何有效，唯独 tollgate 完全无效。

---

## 3. tollgate 钻取（最大单项，+7.0pt）

| 对象 | n | success | collision | off_road | 失败 rc 中位 | speed_ratio 均值 |
|---|---|---|---|---|---|---|
| IDM | 45 | .778 | .067 | .044 | .689 | .737 |
| arm0 | 45 | .000 | .467 | .533 | .834 | .480 |
| arm5 | 45 | .000 | .667 | .333 | .876 | .534 |
| IL-Ebpp | 45 | .000 | .222 | .778 | .579 | .562 |
| seed11-u200（clean500） | 45 | .067 | .133 | .778 | .588 | .406 |

按 blocks 形态（`$`=tollgate 块；失败 rc 中位）：

| blocks | 对象 | n | success | collision | off_road | 失败 rc 中位 |
|---|---|---|---|---|---|---|
| CS$ | IDM / arm0 / IL | 20 | .750 / **.000** / .000 | .150/.550/.050 | .100/.450/.950 | .848 / .888 / **.212** |
| S$S | IDM / arm0 / IL | 13 | .615 / **.000** / .000 | 0/.385/.154 | 0/.615/.846 | .550 / .552 / .544 |
| SS$ | IDM / arm0 / IL | 12 | **1.000** / **.000** / **.000** | 0/.417/.583 | 0/.583/.417 | — / .840 / .876 |

- **SS$（纯直道+tollgate，IDM 12/12）我方 0/12，失败 rc 0.84–0.88 → 失败就发生在最后一块（闸口本身）**；S$S 失败 rc≈0.55（闸口块中部）；arm5 在 SS$/CS$ 以碰撞为主（.92/.65）→ 与历史取证"撞岗亭"一致。
- IL 在 CS$ 的失败 rc≈0.21 是在 curve 段（另一机制），但 S$S/SS$ 同样在闸口块失败。
- 数据侧排除"没学过"：train_5k 覆盖 tollgate 455/5000（9.1%）；同 500 场景专家采集成功率 0.644（29/45）。**有数据、有专家信号，但闭环学不到/执行不出。**

---

## 4. max_step 钻取（我方 vs IDM 是两种病）

| 对象 | n | rc 中位 | max_rc==rc | compound | speed_ratio | mean_speed | crawl_s 中位 | 几何 top3 |
|---|---|---|---|---|---|---|---|---|
| IDM | 16 | .574 | 1.00 | .125 | .216 | 1.998 | **72.1** | curve 6 / tollgate 5 / ramp_in 3 |
| arm0 | 16 | .826 | 1.00 | .688 | .470 | 3.957 | **0.0** | roundabout 8 / curve 5 / intersection 3 |
| arm5 | 9 | .897 | 1.00 | .778 | .536 | 4.448 | 0.0 | roundabout 5 / curve 2 / intersection 2 |
| IL | 6 | .908 | 1.00 | 1.000 | .426 | 3.577 | 5.5 | roundabout 3 / intersection 2 / straight 1 |
| seed11-clean500 | 68 | .874 | 1.00 | .794 | .365 | 3.236 | 0.0 | ramp_in 14 / t_intersection 13 / straight 9 |

- IDM 的 max_step = **堵死**（crawl 72s、speed_ratio 0.22）；我方的 max_step = **一直在动但到不了终点**（crawl≈0、rc 0.83–0.91、max_rc==rc、69% 是 compound 场景）。
- arm0 的 16 条里 **IDM 有 11 条成功、5 条碰撞**，如 roundabout id 14/205/646/724/830/992、curve 187/716/784/816/991、intersection 629 —— 都是"IDM 能完成、我方绕圈/错过出口"的样本。典型 rc：.99/.97/.99（几乎到终点仍 max_step）。
- seed11 把这一病理放大到 68/500（13.6%）→ 该失败模式与训练稳定性强相关（arm0 在 u100–u150 达 0.11–0.23，u175/u200 回落到 0.033/0.013）。

---

## 5. 行为剖面：速度、压线、控制

| 对象 | speed_ratio 均值 | mean_speed | 成功 final_speed | 成功 steps | jerk_p95 | a_lat_p95 | collision min_ttc |
|---|---|---|---|---|---|---|---|
| IDM | .788 | 7.61 | 10.30 | 364 | 5.41 | 1.20 | 2.98 |
| arm0 | .433 | 4.14 | 4.53 | 649 | 25.17 | 1.90 | 2.08 |
| arm5 | .478 | 4.54 | 3.80 | 609 | 34.14 | 2.78 | 2.22 |
| IL | .469 | 4.38 | 3.68 | 564 | 23.76 | 2.03 | 3.41 |

- **我方全程只有限速的 43–48%**（IDM 79%），逐几何一致（arm0/IDM 比值 0.51–0.65，tollgate 0.65 最高），成功 episode 步数约 1.8×。这是全局特征，**不能解释类间差异**，但会放大 max_step（时间预算）与合流难度。
- 失败前速度并不低：arm0 off_road 失败 episode 的 speed_ratio_p95 0.74–0.86，高于同几何成功者的 0.52–0.72（见 T5）→ 失败瞬间反而"相对自己更快"。
- **压线（solid_line_crossing）是与亏项高度对齐的车道纪律信号**：

| geometry | arm0 | IDM | Δ |
|---|---|---|---|
| tollgate | .533 | .044 | **+.489** |
| merge | .370 | .000 | **+.370** |
| t_intersection | .356 | .000 | **+.356** |
| intersection | .130 | .043 | +.087 |
| roundabout | .178 | .156 | +.022 |
| ramp_out / ramp_in | .283/.217 | .283/.217 | .000 |
| split/straight/curve | .000 | .000 | .000 |

- 曲线失败是另一机制：arm0 curve **0 off_road、13 collision（.283 vs IDM .130）**，碰撞 TTC 中位 **1.54s vs IDM 3.32s** → 反应晚/让行差；同时 jerk_p95 25 vs 5.4、a_lat_p95 1.9 vs 1.2，控制抖动明显。

---

## 6. 种子稳健性（重要操作事实）

clean500（500 条同 spec，无 IDM 对照）：

| run | success | off_road | max_step |
|---|---|---|---|
| arm0-clean500 | .638 | .164 | .030 |
| arm5-clean500 | .594 | .216 | .008 |
| seed11-u200-clean500 | **.322** | .468 | **.136** |

sub150（同一 150 条）u200 重评：arm0 .627、arm5 .587、arm1 .640、arm2 .627、**arm3 .233**、arm4 .447、**seed11 .300**。
→ **同配方不同种子 success 0.23–0.64**；单种子结论不可靠，任何 P1/P2 杠杆需多种子验证；max_step 病理（seed11 .136）很可能是训练稳定性/局部解问题。

---

## 7. root cause 假设清单（可验证）

| # | 假设 | 支持证据 | 验证方法（CPU） | 预期 pt |
|---|---|---|---|---|
| H1 | **tollgate 过闸行为缺失**（撞岗亭/在闸口漂出）；plan 参考在 `$` 块无绝对锚点 | SS$ IDM 12/12 vs 我方 0/12，失败 rc .84–.88（最后一块）；arm5 .92 碰撞；历史取证 §8"2/4 撞岗亭、专家有专门 tollgate 扫描逻辑"；数据/专家信号都在 | 45 条 tollgate 用专家路径参考重放（forensics E1 式）；`--tracker exact` 重评；失败时刻可视化 vs 岗亭位置 | ≤ +7.0 |
| H2 | **merge / t_intersection 车道保持失败**（无绝对车道锚点、横偏不回收 → 压线/漂出） | merge off_road 17 条 rc 中位 .78（合流点附近）、t_intersection 16 条 rc .49；IDM 同几何 off_road=0；压线 Δ +.37/+.36；forensics 结论 1/2 | 同 H1 的专家路径参考重放；加横向误差反馈/车道中心锚点后重评该两几何 | ≤ +5.4 |
| H3 | **终点/出口路径选择失败**（绕圈/错过出口，非堵死） | max_step rc 中位 .83、crawl 0、仍在动、max_rc==rc；IDM 在 10/16 条上成功；roundabout/curve/intersection 为主 | rollout 可视化 16 条；记录终止位置/heading vs 出口；终点/导航奖励消融 | ≤ +2.2 |
| H4 | **全局速度过低**（.43 vs .79）放大 max_step 与合流难度 | 逐几何一致；成功 steps 649 vs 364；u150→u200 速度与 success 同升 | 提高速度目标的对照（速度奖励/参考），看 success–collision 权衡 | 间接 |
| H5 | **曲线碰撞：反应晚 + 控制抖动** | curve collision .283 vs .130、TTC 1.54 vs 3.32；jerk_p95 25 vs 5.4 | 平滑正则/LQR 调参后在 curve 子集重评；检查 cut_in/前车 TTC 分布 | ≤ +1.2 |
| H6 | **训练不稳定/种子方差**是当前第一不确定源 | u200 同 150 条 .233–.640；seed11 max_step .136 | 多种子重复 u200；max_step 与优化阶段/early-stop 关系 | 影响全部 |

排除项：数据覆盖（tollgate 9.1% train、专家成功 64%）；单纯"难"（A 难度分布 flat）；速度作为类间差异主因（逐几何一致）。

---

## 8. 给 P1/P2 的杠杆建议（按 pt 排序）

1. **P1：tollgate 专项**（≤+7.0pt，52% 的 arm0 gap）：优先查闭环 plan/执行侧（expert-path 参考重放 + exact tracker 两个 10 分钟实验即可分诊是"plan 缺闸口行为"还是"策略没学会"）；若 plan 侧，补"过闸"参考/行为（专家有专门扫描逻辑）。
2. **P1：merge + t_intersection 车道纪律**（合计 ≤+5.4pt）：绝对车道锚点 + 横向误差回收；压线率是现成回归指标。
3. **P2：max_step/终点行为**（≤+2.2pt，且能降种子方差）：rollout 诊断 16 条（roundabout 出口为主）。
4. **P2：曲线碰撞/平滑**（≤+1.2pt）：TTC 与 jerk 是双指标。
5. **横向使能：速度与训练稳定性**——速度对类间差异解释力弱，但低速度+绕圈会互相放大；任何杠杆需 ≥3 seeds 验证（H6）。

**建议的下一步最小实验**（均 CPU，只读+输出到 /tmp）：(a) arm0 u200 在 45 条 tollgate 上 `--tracker exact` 重评；(b) merge/t_intersection/tollgate 子集用专家路径参考的 LQR 重放（复用 `tools/diagnostics/forensics_closed_loop.py`）；(c) arm0 的 16 条 max_step rollout 可视化。

---

## 9. 局限

- IDM 基线为 09-27 旧 commit 产物（见 §0）；clean500/seed11 无匹配 IDM。
- 逐 episode CSV 只有终止时聚合量，无逐 step 轨迹；"绕圈/错过出口"与"失败点位置"由 rc/速度/crawl 推断，需可视化确认。
- `speed_limit_violation` 在低限速块（uturn/tollgate）几乎恒 True，**不可用作判别指标**；本报告改用 solid_line_crossing。
- 三维交叉表以二维边际 + top cell 呈现；完整 11×3×4 数字可由脚本从 JSON 复算。
