# Stage C v4 / v4.1 报告（2026-10-01 夜）

> 一句话结论：**9 个训练臂（v4 7 + v4.1 2）全部未达"不损伤"（clean500 net ≥ −10）**；但本夜建立了完整的因果地图——**损伤 = policy+experts 的联合漂移，属优化/目标层面**（训练分布上也崩）、**非 critic**（修复后仍坏）、**非泛化/过拟合**（in-dist 也崩）、**非数据量单因**（500 池只是减伤）；**pool 是唯一"防塌"杠杆，真锚/keep-best 是减伤工具**；两个典型失效模式：**末段崩解**与**爬行盆地**。

## 0. 代码变更（v4 修复栈；全部已提交并验证）
| # | 内容 | commit | 验证 |
|---|---|---|---|
| ① | `--value-lr-scale`（value 组独立 lr，默认 1.0） | `fdfa277` | 组序逐位一致 |
| ② | `target_kl` 守门接线（超阈跳 epoch 余量） | `26b0a4a` | 测试 + 日志触发 |
| ③ | 分参数组梯度/更新范数探针（policy/value/experts/other） | `fb8a358` | GPU 冒烟 |
| ④ | KL 锚梯度探针（`anchor_grad_norm/*`） | `6563cbc` | 实锤"旧锚梯度=0" |
| ⑤ | episode 数据探针（终止/步数/回报按原因） | `d0ca165` | 冒烟 |
| ⑥ | **奖励 ctx 修复**：限速源对齐 KPI（a）+ 终局帧掩码回退（b） | `6ab70b3` | 指纹记录 |
| ⑦ | ctx 新键注入（`lead_gap_m`/`lead_speed_mps`/`lane_half_width_m`） | `3268cc7` | 真机 6/6 非 null |
| ⑧ | no-op 探针 + 跨版本指纹 | `cf74b1c` | 逐位一致 |
| ⑨ | **KL 锚修复**：mu/raw logstd 进梯度（coef=0 逐位不变） | `3590d30` | CPU/GPU 指纹 + 锚梯度非零 |
| — | v4 预注册 | `04d9be8` | — |
| — | v4.1 预注册 | `674d502` | — |

## 1. v4 批（预注册 `04d9be8`；7 臂；子集150 配对 net，终评 clean500 全量）

| 臂 | 变量 | u25 | u50 | u75 | u100 | 判定 / 终评 |
|---|---|---|---|---|---|---|
| V0 | critic bundle（warmup10/value-lr×5/kl 0.05） | −62 | −22 ✗ | −8 | −34 | early-collapse |
| V1 | −bundle | −66 | −20 ✗ | −28 | −40 | early-collapse |
| V-anchor | ＋真锚 0.01 | −32 | −27 ✗ | −26 | **−12** | early-collapse（破线后恢复） |
| **V-data** | ＋500 池 | −20 | **−8 ✓** | −11 | −9 | **terminal:failed**：clean **−26** / eval **−23**（最稳） |
| V-lr | `--lr 1e-4` | −13 | −67 ✗ | −67 | −67 | **爬行塌缩**（succ 0；超时型） |
| V-ttc | ＋ttc −0.5 | −62 | −16 ✓ | −59 | −58 | **terminal:failed**：clean **−187**（假过→崩） |
| V-boundary | ＋lane_boundary −0.2 | −31 | −67 ✗ | −67 | −67 | early-collapse（drift 1.31 最大） |

要点：① pool 是唯一防塌；② 真锚 dampen 早期坑（−32 vs −62/−66）且破线后逐点恢复（端点 −12）；③ 闸弱筛选（V0 漏报、V-ttc 误报）；④ 末段崩解三例（V0/V-ttc/v41pa）；⑤ 爬行盆地实锤（V-lr）。

## 2. v4.1 批（预注册 `674d502`；2 臂；V-data 底座 × 单杠杆）
| 臂 | 变量 | u25 | u50 | u75 | u100 | 判定 / 终评 |
|---|---|---|---|---|---|---|
| v41pa | ＋真锚 0.01 | −24 | −17 ✓ | **−13** | −47 | terminal:failed：clean **−147**（末段崩） |
| v41pk | ＋target-kl 0.03 | −25 | **−29 ✗** | — | — | early-collapse（u50 即破线） |

⇒ "pool + 单杠杆"仍未触及不损伤；收紧 KL 守门无效。

## 3. 诊断证据链

### 3.1 模块回滚归因（`/tmp/opencode/rl_v4_rollback.md`）
| 臂 | 完整 | 单回 policy | 单回 experts | 双回 | 全回（sanity） |
|---|---|---|---|---|---|
| P0 | −67 | −67 | −67 | **−1** | **0** |
| A1 | −21 | **−67（更差）** | −65 | **−3** | 0 |
| A3 | −6 | −67 | −65 | **−2** | 0 |

⇒ **损伤 = policy+experts 联合/共适应**（单回滚造成"初始 policy × 漂移 experts"错配）；sanity 150/150 逐行=零点。

### 3.2 in-distribution（`/tmp/opencode/rl_v4_indist.md`）
| 臂 | 训练分布 net | 验证 net |
|---|---|---|
| V0 | **−21** | −34 |
| V-ttc | **−48** | −58 |
| V-data | **+1** | −9 |
| v41pa | **−30** | −47 |

⇒ **主因=优化/目标问题**（训练分布上也崩，幅度约 6–8 成），分布差距次要；V-data 在两分布一致（+1/−9）。

### 3.3 探针（u60–100 逐 update；4 臂对比）
- 无单一指纹可区分"崩/稳"：value 梯度爆炸与 EV 负尖峰在 v41pa（崩）与 v-data（稳）**都出现**（v-data 更猛：value grad 431、EV −0.64）；V0 EV 全程 +0.5 仍末段崩；V-ttc 训练 returns 末段升（14.6）而验证崩（−59）⇒ "训练好/验证坏"分离。

### 3.4 keep-best 全量（`/tmp/opencode/rl_v4_keepbest.md`）
| 候选 | clean500 net |
|---|---|
| V0 u75（子集 −8） | **−27** |
| v41pa u75（子集 −13） | **−52** |
| V-anchor u100（子集 −12） | **−38** |

⇒ keep-best 相对终点大幅减损（v41pa：−147 → −52），但**均未跨过 −10**；**子集150 最优点对全量系统性偏乐观（2–4×）**。

### 3.5 奖励可视化重建（`runs/reward_viz/`，207MB）
- v4 修复的量化（旧→新）：id179 +0.298（终局 speed_ratio 恢复）；id31 **−2.0**（CaRL 修复：同帧 solid_line 不再被抹）；id86 +0.462（限速源对齐 +0.190 + 终局恢复）；id680 +0.229（限速源对齐）。
- would-be 量化（默认关项）：id179 `lane_center` ≈ **−10.7**；id31 `lane_boundary` 10/69 策略步；id680 碰撞前 `ttc`=0（前车 29m）。
- 失败前 5s 零惩罚触发（复现）；骑线无惩罚（397/555 帧 |d_lat|>1m）。

## 4. 结论（按证据强度）
- **确凿**：① 修复栈下 9/9 臂未达不损伤（单 seed 限定）；② 损伤是 policy+experts 联合漂移（回滚+sanity）；③ 训练分布上也崩 ⇒ 优化/目标层（in-dist）；④ pool 是唯一防塌杠杆（V-data 稳定；池 vs 单变量对比最大）；⑤ 真锚首次真正生效（⑨ 修复；dampen 早期坑、破线后恢复）；⑥ 奖励修复的实际增量已量化（viz）。
- **高置信**：⑦ critic 修复不阻损伤（EV 0.31 vs 0.13，闭环同样坏）；⑧ 末段崩解是复现型失效模式（V0/V-ttc/v41pa）；⑨ 爬行盆地真实（V-lr；D3 经济学：超时回报≈成功 75%）；⑩ 闸弱筛选（漏报+误报）⇒ 全轨迹+keep-best 必备。
- **待解**：⑪ 末段崩解的确切触发机制（探针无指纹）；⑫ 为什么 500 池能防塌（覆盖/多样性/难度分布）；⑬ A2/A3 类"同轨-分岔"异常的最终定位（锚修复后该问题部分缓解，但未复测）。

## 5. 下一步设计建议（v5 候选，按优先级）
1. **奖励对齐**（证据最足）：`lane_center`（或门控版）/`ttc`/`lane_boundary` 的预注册臂（would-be 已量化）；**terminal 结构**（爬行盆地：提高 max_step 罚或最低速成形——超时=75% 成功回报是结构性问题）；量纲/权重（speed_ratio 占 83%、到达仅 ~1%）。
2. **更新约束/防末段崩**：keep-best 采纳协议（最优 ckpt 保留）；外科更新（只训 policy 或 experts 子空间/LoRA 式）；lr/trust-region 组合（当前单一 lr 3e-4 与 1e-4 都失败）；**末段专项**（崩解窗口的 lr 衰减/早停）。
3. **输入/梯度通道**：plan-as-action (C) 或至少 P0-3（nav 冻结）修复 + plan head 池化审计（D2 报告 §3）。
4. **信用分配**：group baseline（critic-free；`grouped_discounted` stub 已备）——但注意 ICC 低、K 需 ≥8/32。
5. **数据侧**：500 池 + 覆盖（dagger_r1 97.3%）；块轮换 K=4–8（组模式前置）。
6. **协议**：闸改为"全轨迹 + 分层子集"；seed 复现门槛。

## 6. 现场与复现
- 结果档：`/tmp/opencode/rl_v4_results.{md,json}`、`rl_v41_results.{md,json}`；诊断：`rl_v4_rollback.md`、`rl_v4_indist.md`、`rl_v4_posthoc{,2}.md`、`rl_v4_keepbest.md`、`rl_v4_analysis_{data,net,reward_kpi}.md`。
- 运行：`runs/BTC20261001-*`（v4/v4.1/diag/indist/keepbest）；可视化 `runs/reward_viz/`。
- 代码：`04d9be8`（v4 冻结）→ `674d502`（v4.1 冻结）→ 本报告。
- 纪律：GPU 串行（臂间可比性；诊断评测允许并行并标注）；单 seed 仅方向性；v3 数值不可直接比。
