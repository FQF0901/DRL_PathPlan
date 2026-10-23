# arch_lock 与 chatgpt5p6sol_review 复核结论

> 复核日期：2026-10-23
> 复核对象：`docs/arch_lock.md`（合同正文及当时的【】状态批注）与 `docs/chatgpt5p6sol_review.md`
> 代码基线：HEAD `b0738b4`；合同正文与 `94d271e` 的 `docs/arch_lock.md` 相同
> 方式：只读代码/artifact 审计，加两个只读 oracle lane；没有修改合同、review 或代码。
> 原则：不把“review 写得很长”当成正确，也不把“当前代码能跑”当成合同满足。以下把 **事实（代码/artifact 可复现）**、**推断（由代码路径推导但未复现）** 和 **解释（合同语义有歧义）** 分开写。

---

## 0. 结论摘要

1. **`arch_lock.md` 的规范正文总体是合理的设计合同，不是“自相矛盾到无法实施”的文档。**
   阶段顺序、0.5 s 决策/3 s 想象、4D 动作、稳定 ID/坐标补偿、动态/静态 WM 分工、Val/Test 隔离、checkpoint 选择，这些核心约束是一致的，也都是正确的工程方向。

2. **但把【】批注当成当前状态时，`arch_lock.md` 作为一份文档是不一致的。**
   批注停留在 `639e958`（82 tests、58 tracked 等），与 HEAD（100 tests、71 tracked、D1/D2 已存在、cost 已进 policy、Test500 已标 `split=test`）大面积冲突。这个问题的根因是“规范正文 + 手工状态批注”混在一个文件里，而不是合同逻辑本身。

3. **合同正文有多处规范冲突/歧义，不能只说成“D1 target 一处”。**
   最需要修的是 D1/D2 Target Encoder：`§10.3` 写的是“D1/D2 冻结 Encoder 时 target 固定，EMA 为 no-op”；`§13 D1` 写的是“target 来自 EMA Target Encoder”；`§22.11` 又写成硬约束。当前代码使用的是同一个 planner encoder 直接算 target，不是独立的 frozen target copy。除此之外还有 A2 阶段归属、0.5 s vs 3 s LQR、latent delta 可选 vs 硬约束、轨迹缺 heading、目标 vs 验收、reward/cost 标量化等正文冲突；详见 1.2。这个不是“review 凭空挑错”，但也不是“合同全错”。

4. **`chatgpt5p6sol_review.md` 的核心 P0/P1 判断大部分成立，但不是每一条都成立。**
   最有价值的发现是 `TemporalPool` 的 `[B,T,N,H]` reshape 错误：独立复现显示，改动其他实体会改变实体 0 的输出（最大差 2.88），正确 `permute` 后差为 0；测试只查 shape。由此，当前 r3b、history 1/2/4/6、WM D1/D2 都不能作为“合同所述 per-entity temporal fusion”的证据。
   但 review 也有几处 **事实错误或过度推断**：cost critic 并没有共享 trunk 的梯度（代码是 `fused.detach()`）；selector 已经明确拒绝 Test500；PPO warm-start 在 f523 是 strict；`approx_kl` 在 f523 记录的是 last epoch 而非 first；`TemporalPool` 的“输出应等于实体常量”测试被 LayerNorm 破坏；status 的 v6/245,792 等引用来自旧快照。

5. **当前正确的工程结论不是“继续 Stage E/F”，而是：先修 TemporalPool、补语义测试、重训/重测；同时修 DAgger/BC 的 label 语义、WM target 语义、评测口径和证据链。**
   当前 exact/Val500 数字可以作为“带 bug 的实现诊断”，不能作为合同架构的 gate；D1/D2 只能是 smoke；PPO v8 只能算 wiring canary。

两个独立 oracle lane 的结论也一致到这个程度：合同 lane 的裁决是 **“局部自相矛盾；加 precedence/scope 后降为 needs-clarification”**；事实核查 lane 的裁决是 **“TemporalPool P0 正确，review 大部分 artifact/指标数字可复现，但 §5.4 的 blanket terminal_failed 与 gate() 断言错误”**。下面的结论以代码/artifact 为准，不把任何 lane 的措辞当证据。

---

## 1. `arch_lock.md` 审计

### 1.1 规范正文：哪些是合理且内部一致的

以下部分我判断是合理的，且没有发现规范层面的自相矛盾：

- **时间合同**：`env/control_dt=0.1 s`、`policy_dt=0.5 s`、`rollout_steps=6`、`rollout_horizon=3.0 s`；真实环境每次执行一个 0.5 s 段，imagined 6 步只做 preview。见 `docs/arch_lock.md:64-77`。这解决了“6 步轨迹到底是不是真实执行”的因果歧义。
- **动作语义**：主动作固定为 `(dx,dy,dpsi,v_next)`，PSTL 精确执行端点/终端速度，BC/DAgger/PPO/WM 不允许各自换动作语义。见 `:479-545`。这是正确的单一真源方向。
- **数据/坐标**：OD/LD 先做 ego-motion compensation，再按 stable ID 对齐；content latent 与 geometry anchor 分开；禁止把任意 latent 当二维坐标旋转。见 `:269-363`。这是合理的，尤其能避免“latent 直接做 SE(2)”的错误。
- **静态/动态 WM 分工**：只有 dynamic OD/RoadState 进入 learned transition；Static LD 做 geometry reprojection + shared encoder re-encode；Nav 按 route progress 更新；Ego 由 ego dynamics 推进。见 `:609-623`、`:708-738`。这与“世界模型不应从零幻觉地图/Nav/Ego”一致。
- **评测隔离**：Val500 选 checkpoint，Test500 只用于最终里程碑；checkpoint 选择是安全门槛 → success → completion → comfort/efficiency，不能按 open-loop loss。见 `:1211-1281`。这是合理的治理方向。
- **硬约束**：禁止 lane-level oracle、禁止用一个加权总分掩盖碰撞、WM 必须 ego-action conditioned、Future GT 必须 stop-gradient。见 `:1836-1855`。方向正确。
- **阶段门槛**：D1 先单步、D2 再多步，WM fidelity 不过不得进入 imagined RL；Stage E/F 才做 3 s Plan+WM+LQR。见 `:1101-1191`。作为分阶段合同是合理的。

### 1.2 规范正文的冲突/歧义（按严重度）

合同不是“一句没错”的圣经；按正文条款互相约束后，至少有 14 类问题，其中 8–9 个是直接冲突或高影响歧义。它们多数可以通过**作用域/例外条款**修好，不是必须推翻重写。

| ID | 严重度 | 冲突/歧义 | 证据 | 建议 |
|---|---|---|---|---|
| C1 | 高 | **D1/D2 Target Encoder 语义冲突**：`§10.3` 说 D1/D2 target 固定、EMA no-op；`§13 D1` 说 target 来自 EMA Target Encoder；`§22.11` 又把它写成硬约束 | `docs/arch_lock.md:834-843`、`:1103`、`:1848` | 统一为“D1/D2 用独立 frozen target copy（m=1/no-update）；Stage E 才启用 EMA”；禁止直接把 online encoder 当 target。当前代码正是后者。 |
| C2 | 高 | **content latent delta 是可选基线还是硬约束**：`§4.2` 说“可选 content residual，基线可先关闭”；`§22.6/7` 说 latent delta 必须与绝对 latent、时间、mask 一起使用 | `:356`、`:360`、`:1843-1844` | 把“可选”限制为 A2 前/诊断 ablation；从 A2 开始强制 latent delta，或把 §22 降级为最终系统要求，并写 precedence。 |
| C3 | 高 | **阶段定义冲突**：`§1.2` 把第一阶段（无 WM）定义为 BC/DAgger/PPO；`§1.3` 把 Real Memory Bank 放进第二阶段；`§13` 的 A2（Real Memory Bank）在 B/C 之前 | `:28-45`、`:1033-1082` | 把 Real Memory Bank 归入第一阶段 A2，或明确“phase 1 = A1-A2/B/C，phase 2 = D/E/F”。 |
| C4 | 高 | **LQR 参考长度冲突**：`§6.3`/`§7.1` 要求完整 2–3 s / 3 s，不得只传 0.5 s；`§13 A1-C` 允许 0.5 s canary，`§22.4` 例外；且“2–3 s”与“完整 3 s”不一致 | `:543`、`:587`、`:1013-1016`、`:1841` | 明确两种模式：formal LQR = 3 s/6 点；canary LQR = 0.5 s/5×0.1 s，仅 gross-failure。把“2–3 s”改成 3 s，并给 terminal segment 截断例外。 |
| C5 | 中高 | **轨迹表示缺 heading**：`§6.2` 定义 `τ={(x_i,y_i,v_i)}`，`§6.3` 又要求用 heading 作为切线；插值器无法从 `x,y,v` 得到 heading | `:522`、`:539` | 定义 `τ={(x_i,y_i,ψ_i,v_i)}` 加 t0 knot；明确 current ego → t0 ego → LQR/world 的 frame。 |
| C6 | 中高 | **D1 的 ego 监督 vs `§7.3` 的 ego dynamics**：D1 说对 ego 分模态监督，`§7.3` 说 ego 由显式 dynamics 推进，不是 learned transition | `:609-623`、`:1110-1111` | 明确 ego 只做“dynamics 推进 + frozen encoder 一致性”，不作为自由 learned head；WM 只学 dynamic OD/RoadState residual。 |
| C7 | 中高 | **最终目标 vs 验收**：`§1.1` 要求最终 controller-realized 指标超过 IDM+PP；`§13 F` 用“争取超过”，`§24` 写“接近或超过” | `:26`、`:1185`、`:1900` | 定义 primary endpoint 和非劣效 margin；“接近”最多算 partial，不应作为最终 pass。 |
| C8 | 中高 | **reward/cost 分离 vs 参考表**：正文要求 reward 和 cost 分开、不能用加权总分掩盖碰撞；参考表把 collision/off-road 惩罚放进同一个 scalar 量级表 | `:923-959` | 把表拆成 reward/cost 两张；明确是否双计；给出 cost limit/Lagrangian/terminal penalty 的机制，scalar 只作诊断。 |
| C9 | 中高 | **DAgger 混合系数语义不清**：`§11.3` 一边说“在 Student 访问状态查询 expert”，一边给 `1.0→0.7→...`，又说“不按固定轮次机械下降” | `:885-895` | 分开 `beta_t`（rollout 中 expert 执行概率）与 `w_expert_t`（expert loss/data 权重）；否则 1.0 时状态是 expert-visited，不是 student-visited。 |
| C10 | 中 | **`§22` 的硬约束被写成所有阶段通用**：例如 EMA target、3 s LQR、每个 checkpoint 含 optimizer/EMA；但 A1-C/D1 有阶段例外 | `:1836-1855`、`:1684-1695` | 增加 precedence 条款：§22 适用于最终 deployed system；阶段例外必须逐条列出，不能靠批注解释。 |
| C11 | 中 | **WM deployable 预算不清**：`§8.1` 允许 0.8–1.2M；`§8.2` 的模块范围加上 WM 180–260k，且没有明确 WM 是否计入 deployable；当前 planner 1,053,142 + WM 167,952 = 1,221,094 | `:642-648`、`:662` | 明确 deployable = planner + WM（不含 EMA target）；当前已超 1.2M，必须缩/重平衡或正式记录例外。 |
| C12 | 中 | **Test500/Val500 的 Final claim**：`§1.1` 说冻结验证集，`§14.1` 说 Val500 选模/Test500 最终，`§22.16` 说最终必须 Test500 | `:26`、`:1215-1216`、`:1853` | final claim 统一为 Test500；Val500 只用于 selection/diagnostic；给 Test500 预注册“milestone”规则，避免多用。 |
| C13 | 中 | **目标/指标定义缺失**：success/collision/off-road/wrong-route/stuck/timeout/comfort 均无操作定义；`approach`/`clearly`/`multiple seeds`/`hard threshold` 无阈值 | `§14.2`、`§13 B/D1/D2/E/F`、`§14.4` | 先定义指标分母、stuck 速度/时长、wrong-route 是否 RC-drop、comfort 是 plan 还是 realized；再谈 gate。 |
| C14 | 中低 | **birth/death/source 的 Bank schema 不完整**：`§4.3`/`§4.2` 要求 birth/death/source mask，`§5.1` 只列 `valid_mask` | `:356-370`、`:415-437` | 把 `birth/death/source/depth` 写进 Real/Imagine Bank schema；否则 Imagine Bank 不能表达 FIFO 来源。 |

其中 **C1 是当前最相关的真正规范冲突**：当前实现既不是“独立 frozen target copy”，也不是“EMA updated target”，而是同一个 online planner 直接算 target。`train_wm.py` 的 `target_encoder` 只是字符串，`target_drift=0.0` 是硬编码。这一点必须修合同 + 修代码，不能只靠 status 解释。

`C4` 要特别说明：`§13 A1` 的 0.5 s canary 是**阶段作用域**，不是把 `§6.3` 的 3 s formal LQR 作废。当前 `implementation_status` 把 34%/52% timeout 当作“LQR gate 未闭合”是**状态文字错误**；合同正文不要求 0.5 s 的 success 门槛。正确做法是 report NaN/Inf、tracking/saturation，并把 formal 3 s 留到 Stage F。

`C7` 的 `§1.1` 与 `§24` 也不能靠“争取”含糊过去：最终验收必须定义 primary endpoint 和 non-inferiority margin；“接近”不应自动算 pass。

### 1.2.1 `arch_lock` 旧【】批注（文档层冲突，高）

`arch_lock.md` 的开头批注仍以 `639e958` 为基线，内容与 HEAD `b0738b4` 大面积不一致：

- `:7` 仍写复核 `639e958`；实际 HEAD 是 `b0738b4`。
- `:10` 仍写 `82 passed, 1 skipped`、58 tracked；当前是 100 passed/1 skipped、71 tracked。
- `:13` 仍写“P4 不动”；当前已有 D1/D2 smoke（虽然越级）。
- `:11` 仍写“PPO cost_adv 不进 policy”；当前 policy loss 已有 cost 项。
- `:1216`/`:1853` 仍写“Test500 split=val”；当前 manifest/spec 已是 `split=test`。
- `:10`/`:11` 的 short-path κ、PPO 4 transitions 等也与 HEAD 不一致。

这些不是规范正文的逻辑矛盾，但**作为一份“合同 + 状态”混合文档，它当前不能被视为事实**。根因是手工批注没有 CI/日期/commit gate。建议把【】状态移到 `implementation_status.md`，或在每条批注标“baseline commit + superseded”。

### 1.3 合同是否“合理”：部分要求过强或过虚

**合理但需要量化：**

- **0.5 s LQR canary**：低门槛的方向是对的；但“tracking/trackability 有限”没有数值阈值，容易在 review 中被解释成性能 gate。
- **D1/D2 “明显超过简单基线”**：没有定义“明显”（相对百分比、置信区间、分模态、depth 曲线）。D1 当前 0.616 vs 1.051 是 aggregate，且约 95% 的改善来自 static/ego/nav；不能仅凭 aggregate 说“WM 可用于 policy”。
- **Stage B “接近 Expert”**：没有“接近”的阈值；Expert 73.0% 是特权参考，不是理论上界。
- **Stage F “超过 IDM+Pure Pursuit”**：合同正文用“争取”，但最终验收又写成“接近或超过”；应该区分“目标”与“硬门槛”，否则会拿不可达目标阻塞所有工作。

**过于强/需要解释：**

- **100 万参数 + WM + planner + EMA target**：在当前 planner 已经 1.053M 的情况下，WM 167k 就超过 1.2M；合同没有说明是否允许 WM 训练时共享/部署时剪枝。建议明确“deployable planner+WM”以及 target 只在训练时存在。
- **每个新增模块必须同 seed ablation**：这是好原则，但成本高；应区分“gate 必需 ablation”和“探索性 ablation”。
- **“所有环境语义必须经 Encoder”**：`q_od` 的 TTC/gap 规则只做 auxiliary supervision，不直接进 plan；这在合同里是灰区，应该明确“aux 监督可以规则生成，但不得作为 plan 输入”。
- **“Test500 only final”**：合理；但 eval 的 `--allow-final-test` 是一个可绕过的开关，gate runner 应禁止它，而不是只靠约定。

**技术上不现实的表述：**

- **“100% success / 超过 Expert”不是合同硬要求**；合同自己说“争取”，但状态文档把它写成“100% 不可实现”。应改成“privileged expert reference = 73.0%，不是 theorem，也不是必须达到的 ceiling”。

### 1.4 合同审计结论

**规范正文：基本合理，但存在 C1–C14 的冲突/歧义；其中 C1（target）、C2（latent delta 作用域）、C4（LQR formal vs canary）、C11（WM 预算）、C12（final split）最需要 owner 决策。**
**文档整体：前后不一致，主要因为旧【】批注；需要把状态批注拆出去或标记失效。**
**更准确的说法是：合同“局部自相矛盾、整体可实施”；加一个 precedence/scope 条款后可降为 needs clarification，而不是 redesign。**

---

## 2. `chatgpt5p6sol_review.md` 逐项判断

### 2.1 接受（有代码/artifact/复现证据）

| # | review 结论 | 判断 | 证据/说明 |
|---|---|---|---|
| A1 | `TemporalPool` 的 `[B,T,N,H]` reshape 错位，OD/LD 的 per-entity 时序被混在一起 | **接受（P0）；补充 review 漏掉的更严重后果** | `net/model.py:104-118` 缺少 `permute(0,2,1,3)`；独立复现：固定实体 0，改动其他实体，实体 0 输出变化 max 2.88；正确 permute 后变化 0。**B=1,N=16 必现**（review 表格里的“任意 B>1 或 N>1”不准确）。每个 slot 实际拿到 raster `[T,N]` 的连续 6 个元素，通常混合 1–2 帧 × 6 个实体；`q=x[:,-1:]` 对 13/16 个 slot 不是“当前帧”。 |
| A2 | 因此 r3b、history 1/2/4/6、WM D1/D2 不能作为合同架构证据 | **接受（P0）；但保留诊断价值** | OD/LD 都走 `TemporalPool`；`encode_entities` 的 `od/ld` 也被 WM 当 current/target；r3b 是 bug 下训练的 checkpoint。数值仍可作为“带 bug 实现”的闭环诊断，但不能证明 per-entity 合同。review 未指出的额外后果：`--history-frames k<6` 时 13/16 OD slots 会被错误 mask 成全零，但 `od_valid` 仍把它们当 valid 送进 plan head；aux heads（od_gap/ttc/urgent、lane_role）也在错位的实体监督上训练。 |
| A3 | `runs/val500_clean_r3_p0fixed` 的 500/500 `realized_action` 缺失、config 无 manifest；不能与 r3b 同表称同口径 | **接受（P0 证据链）** | `metrics.json` 的 episode 没有 `realized_action` 字段（不是 dense `null`，而是 key 缺失）；config 无 `split_manifest`。r3b 的 500 条有 `realized_action` 且有 manifest。 |
| A4 | DAgger `label_valid` 要求 `n_substeps==5`，把学生终止时仍有效的完整 expert lift 标签判 0 | **接受（P1，影响被 review 低估）** | `dataset/collect_dagger.py:402-407`；实测 `datasets/dagger_r1/r2/r3_hard500` 的 failed-terminal rows 中 **79%/80%/87%** 的 `label_valid=0`；`n_substeps` 是学生执行步数，不是 lift 标签的物理时长。DAgger 最需要的终止恢复样本被删除。 |
| A5 | BC `terminal_failed` 的 0.2 权重在 `label_valid=0` 后被乘成 0 | **部分接受；不能写成“当前 expert 数据普遍失效”** | `pipeline/train_bc.py:425-426` 的乘法顺序确实如此。但 `datasets/train_bc_2k` 与 terminalfix2 的 430/145 个 failed-terminal rows 全部 `n_substeps==5`，0 条被 `label_valid=0` 清零；只有 termfix3 的 145 行中 81 行（56%）如此。且 `7e9c14e` 后 `terminal_failed` 被 `source==0` 限制，DAgger 的 0.2 路径本来就不适用。正确修法是拆 `segment_complete`/`expert_label_valid`/`terminal_outcome`，不是只交换乘法顺序。 |
| A6 | `expert_plan_valid` 在 collect 与 dagger 语义不同 | **接受（P1）** | `collect.py` 用 episode success 广播；`collect_dagger.py` 用 label_valid；作为“expert failed”分析列不可比。`source` 又把 DAgger 的 student outcome 与 expert label 分开，不能只看同名列。 |
| A7 | short-path κ 门漏洞、速度 overshoot | **接受（P1）** | `net/action_space.py`；复现 `[0.1,0.05,0,0], v0=0` 得 `feasible=True, max|κ|=24, curvature_checked=False`；`[0.6,0,0,1], v0=1` 得 max speed 1.288 > max(v0,v1)=1。 |
| A8 | exact 在 `env.step` 后 `tracker.apply`，`info` 来自 teleport 前物理步；tracking_xy 是 round-trip | **接受机制；影响程度保留** | `pipeline/eval.py:372-384`；代码顺序是事实；具体 collision/off-road 漏检率未做受控复现，review 的“可能滞后”属推断，不能当成已量化 P0。tracking_xy 约 2.5e-6 m 是 teleport round-trip，不是控制能力。 |
| A9 | 场景分桶只用 `spec.geometry[0]`，而主标签是 `spec.labels["geometry"]` | **接受（P1，已复现）** | 独立检查 Val500：468/500 不一致；例如 intersection/ramp/tollgate 被记成 straight/curve。`pipeline/eval.py:484`。 |
| A10 | timeout/stuck/comfort/feasible 指标语义有偏差 | **部分接受** | timeout 在 trunc 路径代码上确实有洞，但当前 r3b 500 条没有 trunc（arrive 338 + collision 78 + off-road 37 + timeout 47 = 500），对当前数字无实际低估；stuck 与 timeout 重叠 22/23；comfort 是 plan-side，且 `comfort_violation_rate=0.996` 是 99.6% **violation**；feasible 是 episode-level bool。 |
| A11 | WM target 用同一 frozen planner encoder，不是独立 EMA target；target 是 6 帧 temporal pooled；static LD/Nav/Ego 仍 learned transition | **接受代码事实，合同解释见 C1/C6** | `train_wm.py` 加载同一 encoder；`target_drift=0.0` 硬编码；`net/world_model.py` 对 ld/nav/ego 都有 head；`_align_by_id` 做了 ID 对齐，但内容本身来自错误 temporal pool；D1 不是合同定义的 static/dynamic 路由。 |
| A12 | D1 的 200 个 WM val scenes 中 90 在 planner train、110 在 planner val、0 unseen | **接受（已核 artifact）** | `runs/wm_d1_final/metrics.json` 的 `val_scene_seeds` 与 `runs/bc_2k_dagger_r3b/best.pt` 的 `train_seeds/val_seeds` 交集；0 unseen。held-out 只对 WM pair 训练成立，不对 encoder 成立。 |
| A13 | D1 val loss 与 persistence 的估计量不完全一致（batch mean vs full sample） | **接受（方法学问题）** | `train_wm.py` 的 val 是每 batch 均值再平均；persistence 是全集均值；末 batch 不整除时会有偏差。不是最大问题，但应统一。 |
| A14 | D2 的 persistence 是 t0 重复，不是 constant-velocity；masks 固定 t0 | **接受（P1）** | `probe_wm.py` 的 persistence 是初始 state 重复；没有 constant-velocity；birth/death mask 不随 depth 变。 |
| A15 | PPO 的 cost/trust-region、split 强制、fallback mask、cost 未裁剪等问题 | **接受主要项** | `train_ppo.py` 不用 split manifest；cost 项未 clip；fallback 被 policy mask；v8 实际 post-update KL .189/.639（不是 clean）。 |
| A16 | checkpoint/YAML/provenance/测试 hermeticity、duplicate `resolve_action` 等工程债 | **接受大部分** | 无 yaml loader；checkpoint 缺 optimizer/RNG/split hash/encoder version；`env/specs` gitignored；`resolve_action`/`safe_brake`/`_batch_from_history` 多份；测试以 shape/finite 为主。 |
| A17 | `arch_lock` 的旧批注与 HEAD 不一致 | **接受（文档 P0）** | `:7`/`:10`/`:11`/`:13`/`:1216` 等；详见 1.2.1。 |

### 2.2 部分接受 / 需要解释

| # | review 结论 | 判断 | 解释 |
|---|---|---|---|
| P1 | `TemporalPool` 的“每个实体恒定 c，输出应等于 c”最小测试 | **反驳具体证明；接受结构结论** | review 的 17.24 数值可以复现，但它的“期望输出等于 c”不成立：`TemporalPool` 末尾有 `LayerNorm(out+q)`；即使正确 permute，输出也不会等于 c。这个数值只能说明“错误 reshape 与参考实现不同”，不能单独证明“跨实体污染”；正确测试应是：改变其他实体不改变当前实体输出（isolation/equivariance）。该测试已确认 bug。 |
| P2 | `target` 必须是独立 EMA Target Encoder，D1/D2 当前不满足 | **部分接受** | 代码确实用的是同一个 online planner，不是独立 target copy；但 §10.3 明确允许 D1/D2 固定 target、EMA no-op。应接受“不是独立 target copy”，不应接受“D1 必须现在做 EMA 更新”。合同需要 C1 的澄清。 |
| P3 | `terminal_failed` 的 0.2 应该在 `label_valid` 之后仍生效 | **部分反驳** | `terminal_failed` 的 partial terminal 本身不是有效 0.5 s plan label；直接 0.2 权重会训练不完整动作。真正的问题是字段语义混用（plan label vs outcome/value），而不是乘序。 |
| P4 | `cost critic` 的 level loss 通过 shared trunk 改写 policy | **反驳（代码事实）** | f523 的 `net/model.py:396-398` 是 `self.cost_value(fused.detach())`；cost head 的 MSE 不会回传到 trunk。v8 KL 仍然大，但根因不是“cost critic 的 level loss 共享 trunk”；更可能是 `cost_adv` 进入 policy、4 个 inner epochs、lr/clip 的组合。需要 `cost_coef=0` 对照实验，而不是凭 loss 大小断言。 |
| P5 | PPO 的 `approx_kl` / `clip_frac` 是 pre-update | **反驳（当前 f523 代码）** | f523 的 `train_ppo.py:1219` 记录 `last["approx_kl"]`，`first_epoch` 单独保留；v8 contract 顶层是 `.188/.639`，不是 `1e-6`。review 可能混用了 v7 或旧代码。 |
| P6 | PPO warm-start 仍用非 strict eval loader | **反驳（当前 f523）** | `train_ppo.py:1450` 是 `load_model(..., strict=True)`；review 的这条引用不成立。 |
| P7 | selector 允许 Test500 | **反驳（当前 f523）** | `select_checkpoint.py` 显式 `if args.split == "test": parser.error(...)`；eval 需要显式 `--allow-final-test`，不是无门槛。review 这条与代码不符。 |
| P8 | `r3b` 的所有性能数字都“无效” | **部分反驳** | 作为合同架构证据无效（TemporalPool bug）；作为“这个 buggy checkpoint 的闭环行为”仍是可观察诊断。不能把“不能证明合同”偷换成“所有数字都没有信息”。 |
| P9 | status 的 `245,792`、v6、LQR 等引用 | **反驳为陈旧引用** | f523 的 status 已是 167,952、v8；review 有些行号/数值是旧快照。review 需要与当前 HEAD 对齐，否则会制造“假问题”。 |
| P10 | `exact` info 是 P0 | **接受机制，降低定性** | 代码顺序确实可能让 `info` 对应 teleport 前状态；但未做 controlled replay，影响是推断。应列为 P1 证据/指标问题，不把未量化的 P0 当成事实。 |

### 2.2.1 review 漏掉或低估的事实（补足）

| # | 事实 | 证据 | 影响 |
|---|---|---|---|
| M1 | `TemporalPool` 的错误不仅混实体，`--history-frames k<6` 时还会把 13/16 个 OD slot 错误地变成全 zero，同时 `od_valid` 仍把 zero 当 valid 送进 plan | `net/model.py:107-118`、`:370-374`；k=1 时 13/16 slots 全 masked | history 16/38/52/68% 曲线混合了“历史减少”和“零 token 注入”，不能作为 history 因果证据 |
| M2 | `od_gap/od_ttc/od_urgent/lane_role` 辅助头也按错位的 slot 训练，和 `batch["od"][:, -1]`/lane 标签错配 | `net/model.py:382-385`；`pipeline/train_bc.py:644-686` | 第二条独立污染通道；即使修 plan path，aux loss 仍可能继续伤害 encoder |
| M3 | `select_checkpoint` 的 50 scene 是 `all_specs[:limit]` 前 50 个，不是随机/固定 seed 子集 | `pipeline/select_checkpoint.py:49` | 每次候选都用同一前缀，存在系统性选择偏差；“50 scene r3b +10pp”不可当稳定结论 |
| M4 | r3b vs r3 的 +1.4 pp 经配对 McNemar 为 41/34，p≈0.49 | 500 个共同 seed 的 artifact | 比 CI 更直接地说明 success 差异是噪声；不能 promotion |
| M5 | r3b 的 `infeasible_count_mean≈fallback_count_mean`（2.642 vs 2.632） | `runs/val500_clean_r3b_p0fixed/metrics.json` | `feasible` 近乎“gate 是否接受原始 sample”，不是 outcome；不能当独立安全指标 |
| M6 | `timeout` 在 trunc 路径代码上确实有洞，但当前 r3b 500 条没有 trunc：arrive 338 + collision 78 + off-road 37 + timeout 47 = 500 | `pipeline/eval.py:440-478`；artifact 计数 | 代码债存在，但对当前 r3b 数字没有实际低估；应降为 P1/P2 |
| M7 | `comfort_violation_rate=0.996` 是 **violation** rate，不是 comfort 99.6% | `pipeline/eval.py:339`；r3b metrics | 状态/审阅文字若写“comfort 99.6%”会反义；应写 99.6% violation |
| M8 | `stuck=0` for Expert 是 `collect.py` 硬编码，不是实测 | `dataset/collect.py:576` | 不能把 Expert 的 stuck 当作真实安全率 |
| M9 | `r3b` 的 500 条有 `realized_action`/manifest；r3 的 500 条没有字段；r3b−legacy 的 CI 也不是 review 引用的精确值（重算约 [−0.8,+6.0] vs review [−1.2,+5.8]） | artifact + paired bootstrap | 方向不变，但数字引用应统一 |
| M10 | `gate()` 的 `v_next<0` 在现有 `CLEAN_BOUNDS.low[3]=0` 下不可达，且 PPO/DAgger 都捕获 `ValueError`；review 的“fallback 无统一捕获”不成立 | `net/action_space.py:211-215`、`train_ppo.py:257-261`、`collect_dagger.py:310-313` | 不是 P1；最多是 dead-code robustness |
| M11 | `select_checkpoint` 的 `selected.pt` 旧文件在 no-eligible 分支不会删除 | `select_checkpoint.py:71-84` | 下游若只看文件可能误用旧 selected；应 unlink/写 null sentinel |

### 2.3 明确反驳

1. **“TemporalPool 输出应等于实体常量 c”的数值证明不成立。**
   `LayerNorm` 会改变常量向量；review 的 17.24 “误差”不能作为根因证据。但 review 的 **结构判断（reshape 错位）成立**，且更严重的证据是实体隔离测试失败。

2. **“cost critic 通过 shared trunk 主导 policy”不成立。**
   `cost_value(fused.detach())` 明确切断了 level loss 到 trunk；v8 的高 KL 需要别的解释。把 `cost_loss` 的数值大当成“cost critic 主导”是因果跳跃。

3. **“selector 允许 Test500”不成立。**
   当前 selector 拒绝 test；eval 有显式 final-test flag。治理 loophole 可以批评，但不能说成“无保护”。

4. **“PPO warm-start 非 strict”不成立（f523）。**
   f523 的 PPO 初始化是 strict=True；review 的引用是旧代码/旧行。

5. **“PPO 记录的是 pre-update KL”不成立（f523）。**
   当前代码记录 last epoch；v8 contract 顶层就是 `.188/.639`。review 需要区分 v6/v7/v8。

6. **“DAgger 从不写 terminal_failed”在当前代码不成立。**
   `collect_dagger.py:648` 已经写入；旧 shard 全 0 是数据生成时间早于代码修复。应说“旧 DAgger shard 未覆盖新语义”，不能说“当前收集器永远不写”。

7. **“所有数字都没有意义”过度。**
   TemporalPool 使“合同架构证据”失效；它不否定 exact 闭环的实际观测值。正确表述是“诊断价值保留、gate 价值撤销”。

### 2.4 review 自身的事实/引用问题

`chatgpt5p6sol_review.md` 有不少结论基于 f523，但引用了更早的 status/code 片段：

- 它说 status 引用 v6/245,792；f523 的 status 已是 v8/167,952。
- 它说 PPO `approx_kl` 是 pre-update；f523 已是 last，v8 contract 顶层 .188。
- 它说 PPO warm-start 非 strict；f523 是 strict。
- 它说 selector 允许 test；f523 明确拒绝。
- 它说 cost critic shared trunk；f523 是 detach。
- 它的 `TemporalPool` 常量测试不符合 LayerNorm 语义。
- 它把 `r3b` artifact 有 realized/manifest 与 r3 无 realized/manifest 的差异说对了，但“500/500 null”在 r3 实际是字段缺失（JSON 没有该 key），严格说不是 null；不影响结论。

**根因不是 review 作者“乱写”，而是 review 在多个代码/文档版本之间跳转，且没有把“f523 当前代码”和“旧 artifact/旧 status”分开。** 这也说明为什么必须做 provenance：否则 review 也会被旧 artifact 误导。

---

## 3. 仍然成立的关键事实

### 3.1 TemporalPool P0：为什么它是 P0，而不是“小 bug”

`TemporalPool.forward` 的输入是 `[B,T,N,H]`，当前写：

```python
x = tokens.reshape(B * N, T, Hd)
mask = ~(valid.reshape(B * N, T) > 0.5)
```

正确的 per-entity 展开必须是：

```python
x = tokens.permute(0, 2, 1, 3).reshape(B * N, T, Hd)
mask = ~(valid.permute(0, 2, 1).reshape(B * N, T) > 0.5)
```

当前 `reshape` 把 `[b,t,n]` 按 flat memory 切块，`B*N` 行中每个“实体行”拿到的是不同 `t/n` 的组合，不是同一实体的 6 帧。独立复现：

- 固定实体 0，随机改实体 1..15：实体 0 输出最大变化 **2.88**；
- 用正确 `permute` 的参考实现：最大变化 **0.0**；
- 这说明不是“LayerNorm 造成的常量误差”，而是真实跨实体污染。

`test/test_model_shapes.py` 只检查 shape/finite；没有“改动其他实体，当前实体不变”的语义测试，所以该 bug 从 `c8c4ce2` 一直存活。r3b、history、WM 都是在这套错误时序上训练/评估的。

**影响边界要诚实：**

- **不是**说 exact 闭环结果完全没有物理意义；它仍然是某个网络的闭环表现。
- **是**说它不能证明 `arch_lock` 的“per-entity temporal fusion”、history 因果收益、WM 的 per-entity target。
- 修复后必须重训，不能只修代码继续用旧 checkpoint。

### 3.2 DAgger/BC 的 label 语义

- DAgger `label_valid` 把 `n_substeps==5` 当作标签有效条件（`collect_dagger.py:402-407`）。学生提前碰撞/到点，`n_substeps<5`，但 expert lift 仍是对学生状态的一个 0.5 s 反事实动作；这个动作不是“学生没有执行完”就无效。这里混淆了 **student execution completeness** 与 **expert label validity**。实测 `datasets/dagger_r1/r2/r3_hard500` 的 failed-terminal rows 中 79%/80%/87% 被置 `label_valid=0`，这不是“少量样本”。
- BC 的 partial terminal 被设为 `label_valid=0`（`collect.py:524-527`），随后 `train_bc` 的 `w_failed_expert` 被 `label_valid` 清零（`train_bc.py:425-426`）。这不是“0.2 降权”，而是“plan loss 完全不用这个 terminal row”；value/risk 可能仍用它。**但必须区分语料：`datasets/train_bc_2k` 的 430 个 failed-terminal rows 全部 `n_substeps==5`，terminalfix2 的 145 个也全部 5；所以旧语料里 0.2 实际生效，不能说“所有 expert 数据都失效”。只有 termfix3 的 145 行里 81 行（56%）是 partial。** 如果合同要的是 failed expert segment 0.2 权重，必须另设 `terminal_plan_valid` 或显式 `segment_complete`，不能靠乘序。
- `expert_plan_valid` 在 BC 是 episode success 广播，在 DAgger 是 label_valid；两个数据的同名列不可直接做失败分析。`7e9c14e` 后 `terminal_failed` 又被 `source==0` 限制，DAgger 的 student outcome 不会套 failed-expert 权重——这个修复方向正确，但没被 review 反映。
- `privileged_expert_sample` 仍基本恒 1；`multi_solution` 字段不存在；`failure-prioritized buffer` 没有实现。Stage B 的“接近 Expert”目前没有可审计的数据语义。

### 3.3 WM 证据

- D1 当前 `runs/wm_d1_final`：167,952 params，0.616 vs persistence 1.051；数值本身可复算。
- 但：
  - target 与 current 来自同一个 frozen planner encoder，不是独立 target copy；
  - target 是 6 帧 temporal pooled，current/target 共享 5/6 历史；
  - static LD/Nav/Ego 都有 learned heads，不是 §7.3 的 reproject/re-encode/dynamics；
  - 200 个 WM val scenes 中 90 在 planner train、110 在 planner val、0 unseen；
  - D1 只用 Expert/DAgger r1/r2，没有 PPO 实际执行 motion；
  - D1 数据/场景 split 与 loss 估计量没有统一 provenance；
  - D2 是 recorded-action 自回归 probe，不是 1/2/3/6 训练课程。

因此 **D1 只能叫“aggregate persistence smoke”，不能叫 D1 gate**；D2 只能叫 diagnostic。这个结论与 review 的主判断一致。

### 3.4 PPO

- v8 的 per-transition log-prob diff = 0；batch diagnostic 约 1e-5，严格 gate 用 per-transition；这是正确的方向。
- 但 v8 的实际 post-update `approx_kl` 在 update 1 是 .189、clip .639；update 5 是 .015/.212。不是 clean P3。
- 问题不是 review 说的“cost critic level loss 共享 trunk”（代码已 detach）；更可能是 cost advantage 直接进入 policy、4 个 inner epochs、lr/clip 的组合。
- `train_ppo` 没有 split manifest；cost 项未裁剪；fallback 被 mask 出 policy；BC/DAgger auxiliary 未保留；没有 reward/cost hacking audit。Stage C 只能算 wiring canary。
- **不要用 “policy_loss + cost_loss 数值大”直接推断 trunk 梯度；必须做 `cost_coef=0` 的同 seed 对照。**

### 3.5 评测口径

- `pipeline/eval.py:484` 用 `spec.geometry[0]` 分桶，而 Val500 的 `labels["geometry"]` 是更细的真实类别；独立核对 468/500 不一致。分桶表不能用于“路口/匝道/收费站”结论。
- `env.step` 然后 `tracker.apply`，`info` 来自 teleport 前物理步；这会影响 collision/off-road/arrival 的解释，但需要 controlled replay 量化，不应直接当成已证明的 P0。
- exact 的 `tracking_error_xy` 是 realized vs reference 的 round-trip，不是控制跟踪误差；r3b 的 2.5e-6 m 不能当作“exact 控制精度”。
- `timeout` 在 trunc 路径代码上确实有洞，但当前 r3b 500 条没有 trunc：arrive 338 + collision 78 + off-road 37 + timeout 47 = 500；因此代码债要修，但对这组数字没有实际低估。`stuck` 与 timeout 重叠 22/23。`comfort_violation_rate=0.996` 是 **violation** rate（99.6% 违反），不是 comfort 99.6%。`feasible` 是 episode-level bool，且 r3b 的 `infeasible_count_mean≈fallback_count_mean`（2.642 vs 2.632），更像 “gate 是否重写 sample”，不是 outcome。评测文档必须拆开这些指标。
- Test500 当前是 `split=test`；selector 拒绝 test；eval 需要显式 `--allow-final-test`。合同正文的“Test500 只最终”是合理的，旧批注的 `split=val` 是过期信息。

### 3.6 证据链

- `runs/`、`datasets/`、`env/specs/` 均被 gitignore；clean clone 不能复现。
- `runs/val500_clean_r3_p0fixed`/`lqr50_clean_r3_p0fixed` 没有 `realized_action`、没有 manifest，不能称 `p0fixed` 同口径。
- `runs/val500_clean_r3b_p0fixed` 有 realized/manifest，可以作为 r3b 的闭环诊断，但它仍带 TemporalPool bug。
- `runs/expert_val500/metrics.json` 的 73.0/14.4/6.6/6.0 是 episode-level；`/tmp/opencode` 的 npz 是证据来源，clean clone 不可见。
- checkpoint 缺 optimizer/RNG/stage/global step/split hash/encoder version；`best.pt` 仍按 open-loop val loss 选择，不是闭环 selector。

---

## 4. 对 `arch_lock.md` 的修改建议（不建议我直接改合同）

以下是建议 owner 考虑的最小澄清，不是让合同向当前代码妥协：

1. **Target Encoder**
   - D1/D2：明确为“从 online planner 初始化的独立 frozen target copy；`m=1` 或 no-update；stop-grad；target drift 记录为 0”。禁止直接把 online encoder 当 target。
   - Stage E：才启用 EMA 更新；`§22.11` 的 EMA 要求加一句“D1/D2 例外为 m=1/no-update”。
2. **Static/Dynamic**
   - 明确 `LD/Nav/Ego` 不允许普通 learned transition；LD 只允许 residual + reproject；Nav 只按 route progress/re-encode；Ego 由显式 dynamics 推进。
   - D1 的“ego supervision”改为“ego dynamics 的监督”，不是 learned ego prediction。
3. **LQR**
   - `§13 A1` 的 0.5 s canary 增加数值化最低门槛：NaN/Inf 为零、tracking finite、无持续饱和；success/timeout 仅报告，不作 gate。
   - 保留 Stage F 的 3 s 正式执行；不要把 0.5 s 的 success 当成 Part F。
4. **Budget**
   - 明确 deployable = planner + WM（不含 EMA target）；当前 1,221,094 已超过 1.2M，必须要么缩 planner、要么缩 WM、要么正式记录例外。
5. **Stage**
   - 明确 D1/D2 only after C，P4 after P3；如果 owner 要 out-of-order，写成“deviation/waiver”，不能写成 gate pass。
6. **Evidence**
   - 每个 gate artifact 记录 git commit、dirty、checkpoint SHA、split manifest SHA、controller、limit、seed、code version；`runs` 无法入库时，至少存 checksum 与生成命令。
7. **Wording**
   - “Expert 73.0% reference”不是“upper bound theorem”；“D1 aggregate beats persistence”不是“WM usable for policy”；“PPO wiring canary”不是“P3 pass”。

---

## 5. 推荐的下一步（按优先级）

1. **修 `TemporalPool`（P0）**
   - `tokens.permute(0,2,1,3).reshape(B*N,T,H)`；`valid.permute(0,2,1).reshape(B*N,T)`；
   - 加 4 个语义测试：实体隔离、B>1/N>1、mask 单帧、hist_valid；
   - 重训 r3b/r3/history/WM；旧 checkpoint 不能继续作为合同证据。

2. **重建证据链**
   - 标记 `runs/val500_clean_r3_p0fixed`、`lqr50_clean_r3_p0fixed` 为过期；
   - 给 r3b 加 TemporalPool 修复前后的对照，而不是把旧 67.6 当最终；
   - 每个 artifact 记录 SHA/commit/split/limit/controller。

3. **修 DAgger/BC**
   - 拆 `segment_complete`、`expert_label_valid`、`terminal_outcome`；
   - 学生提前终止时，完整 expert lift 仍可作 plan label；
   - partial terminal 用 outcome/value 监督，不强行当 0.5 s plan label；
   - 重采 terminalfix 数据，并把 source/privileged/multi_solution 写入。

4. **修 WM target**
   - 独立 frozen target copy；static LD/Nav/Ego 走 reproject/dynamics；
   - D1 数据加入 PPO executed motion、恢复扰动；
   - 先修 TemporalPool 再重跑 D1/D2；D1 aggregate 通过不等于 D1 gate。

5. **修 PPO**
   - 加 target-KL/early stop 或降低 inner epochs/lr；
   - `cost_coef=0` vs 0.1 同 seed 对照，确认高 KL 是否来自 cost advantage；
   - cost surrogate clip、fallback policy gradient、split manifest、BC/DAgger aux。

6. **修评测**
   - `spec.labels["geometry"]` 分桶；timeout/trunc、stuck、comfort、feasible 分列；
   - exact info 顺序做 controlled replay；tracking error 明确 round-trip vs control；
   - selector 加 LQR shadow、comfort/efficiency、SHA；禁止 test 选模。

7. **文档**
   - 把 `arch_lock` 的【】批注移出正文或标记 baseline；
   - 保留“合同正文”与“当前状态”两个文件；
   - 本文件（argue）只保留当前结论，不再混用旧版本。

---

## 6. 最终裁决

- **`arch_lock.md` 规范正文：基本合理，但存在 C1–C14 的冲突/歧义；其中 C1（target）、C2（latent delta 作用域）、C4（LQR formal vs canary）、C11（WM 预算）、C12（final split）最需要 owner 决策。** 它不是“前后矛盾到没有价值”，但作为包含旧状态批注的文档，当前不一致。
- **`chatgpt5p6sol_review.md`：核心 P0/P1 判断大体正确，尤其 TemporalPool、r3 artifact、DAgger label、WM static/target、eval bucket、PPO trust region；但存在 stale refs 和几处事实错误。** 不能无条件接受它的所有说法，尤其不能接受 cost critic 共享 trunk、selector 允许 Test500、PPO pre-update KL、TemporalPool 常量测试这些说法。
- **当前实现：r3b 67.6%、history、D1/D2 都只能作为带 TemporalPool bug 的诊断；不是合同 gate。** 先修 P0，再谈 Stage E/F。
- **不要把“review 很长”当权威，也不要把“当前能跑”当合同满足。** 事实链是：reshape 错位 → per-entity 时序失真 → checkpoint/历史/WM 证据不可用；这是当前最重要的根因。
