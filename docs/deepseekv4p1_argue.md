# deepseek-v4.1 argue（主文档 · 第三轮：收拢）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态**：argue 阶段，**未实施任何改动**。本文件自本轮起作为我方主文档；历史回应作为附录保留（附录 A = 第二轮回应全文，附录 B = 第一轮回应全文）。
> **沿革**：ChatGPT round-2（commit `b14743b`）→ 我方第二轮回应（commit `078de2a`，已并入本文件附录 A；`docs/deepseekv4p1_argue_r2.md` 为历史副本）→ 收拢轮（`b2b0f64`）→ **ChatGPT 确认（`5fdbddc`：D-A/B/C 接受、D1–D5 关闭、S0→S1 指令）** → **第四轮：可行性排摸与执行前澄清（§0.9）** → ChatGPT 确认+硬约束（`cd4b559`）→ **第五轮：收拢完成（锁定版，§0.9.5）**。
> **我方立场摘要（给新读者）**：目标是超越 IDM；路径 = **先量化"上限与口径"、再按数据做减法/几何对齐**；反对无证据地删模块或扩模块；反对让 GPU 空转等代码。

---

## 0. 一页纸：收拢后的局面

- **共识（8 条，锁定）**：见 §1。硬分歧**已消解**（"立即停扫"并入 C1；"先删 WM"并入 C2；"8m gate"我方认账并限定范围）。
- **我方修正（3 条）已全部确认**（`5fdbddc`）：D-A/D-B/D-C 均被接受；原分歧记录保留于 §2.1/§2.2。
- **待决 D1–D5 已关闭**（`5fdbddc` 裁定；落地澄清见 §0.9.2）。
- **草案方案 v0.1**：P0 三件套（1–2 天、零训练或小工具）→ P1 按数据分支 → P2 模块对照；Gates G1'–G3'——见 §3。
- **后续计划**：S0 排摸收尾 → S1 P0 → S2 分支 → S3 对照 → S4 验收——见 §4。
- **约束**：本文档只做收拢，**不触发任何执行**；开工需明确指令。

---

## 0.9 第四轮（2026-10-09/10）：可行性排摸与执行前澄清

ChatGPT 最新答复（`5fdbddc`）确认我方全部修正（D-A/D-B/D-C）、关闭 D1–D5，并给出 S0→S1 指令。我们按"锁定方案是否真的可执行"做了**只读排摸**（代码/磁盘核查）。结论：**整体可行**；有 1 项重要兼容性问题与 2 处执行细节需在开工前明确（见 §0.9.2）。

### 0.9.1 排摸结果（逐项）

| P0 项 | 结论 | 证据/备注 |
|---|---|---|
| 双口径评测（E/F：plan vs repeat_action） | **现成可用** | `EVAL_REFERENCES = ("plan","repeat_action")`（当前 `pipeline/eval_runner.py:773`；**v7-lock 代码同样已含**）；`tools/test.py` 透传 `--eval-reference/--tracker/--limit`（`tools/test.py:77,113`） |
| Arm P / pri512 资产 | **在库可直接评** | `runs/BTC20261007-2202_arm_p`、`runs/BTC20261009-0824_sw_pri512`（含 stage_b 多 epoch ckpt） |
| A 格 IDM | 现成（`tools/baseline_eval.py`） | 需导出**逐 episode** 数据供 paired（工具内部已有 episodes 列表，输出粒度需补） |
| B/C 格（oracle / laneplan） | **现成**（forensics 模式） | `tools/diagnostics/forensics_closed_loop.py --mode {oracle,laneplan,...}`；记录含逐策略步 `plan` 与轨迹 |
| D 格（expert 动作→同表示 6 点 plan） | **可行，≤0.5 天** | BC 数据即含 `action (N,6,2)` =（ds,dθ）6 步链（与模型 plan 同表示）；`arc_step` 是唯一 (ds,dθ)→位移实现（`net/model.py:150-192`）；专家链可用 oracle 同款 live 运行获取；转换可复用采集侧 `_window_actions` 口径 |
| E-exact 列 | **现成**（`--tracker {exact,lqr}` 两代代码均支持） | 同上 |
| F 格（repeat_action） | **现成**（同上） | — |
| 四分类（含 footprint 插值） | **可行，需小扩 forensics** | 逐策略步 plan/位姿已在记录；缺 corridor/boundary margin 与三时间戳 → 与 D 格同批实现（≤0.5 天） |
| eval500 / clean500 spec | **在位** | `env/specs/scenarios_eval500.json`；`docs/v7_reports/specs/specs_val_only500.json`（+150） |

### 0.9.2 开工前需明确的 3 件事

**① s11 兼容性（唯一重大问题）。**
- 事实：s11 是 **obs v5 + v7 代模型**（`runs/BTC20261005-0601_v7p2_s11_arm1/model.snapshot.yaml`：experts 256、含 lane/ttc 时代结构）；当前代码为 obs v6 + v8 结构 → 用当前代码评 s11 会触发 **shape mismatch → 静默随机初始化**（`pipeline/eval_runner.py:677-739` 只打印日志，不报错）。
- 可行方案：用 v7 代代码重建评测环境（`git archive c37acbb` 或 `v7-lock-20261006` → /tmp；`git worktree` 被权限禁用）。已核实 **v7-lock 代码本身已含双口径评测与 exact/lqr tracker**，因此 s11 的 plan/repeat_action 两口径评测与 v7 版 forensics 分类都可忠实重跑。
- 成本：重建+接线 ~0.5–1 天（一次性）；每格评测 ~12min。
- **选项**：完整版（s11 两口径 + 分类）或简化版（s11 仅用历史 plan-口径数据 `runs/BTC20261005-074755_*`，跳过 F 格与分类）。**我方建议完整版**——s11 是唯一完整 DAgger+RL 链，其口径差与失败画像正是决策树 D/E 分支的关键样本。

**② ckpt 加载校验 = 审计硬条件。**
- 每行评测必须断言 `missing=0 && shape_mismatch=0`（加载日志在 `pipeline/eval_runner.py:729-739`；历史上发生过"形状不匹配→随机初始化→假阴性"事故）。审计报告中每行都要附加载日志摘要。

**③ 两处执行定义（拟按此实现，如对方/用户有异议请指出）：**
- **D 格首点**：不做 mu 覆盖——模型的 `plan[0]==action_mu` 覆盖规则不适用于 expert 链（expert 链首点本身就是其下一步动作，天然等价）；其余严格镜像：单位/归一化/裁剪/0.5s 间隔/SE(2) 累积/LQR 参数全同。
- **分类边界真值**：用奖励侧同款 `road_edge_distance_from_ctx` 复算 + 引擎内 footprint 采样（节点间插值），落入 forensics 记录；三时间戳（首次不可行 plan / 首次越界 / 终局）同批实现。

### 0.9.3 成本重估（供排期）
- 完整版 P0：**≈2–3 天**（D 格 0.5 + forensics 扩展 0.5 + s11 v7 重建 0.5–1 + 跑批/报告 0.5）。
- 简化版（s11 降级）：**≈1–1.5 天**。
- 均不依赖新训练代码；GPU ≲1 个白班（排摸收尾后）。

### 0.9.4 本轮立场
- **无新的原则性分歧**；对 `5fdbddc` 的 G1 修订（≥0.80 或 +5pt 即可，显著性留给 G3）与"P1 单分支内允许组合修复 + 先小消融后 candidate recipe"**均接受**。
- 请对方/用户注意两点：s11 路径的兼容性成本（§0.9.2-①）与"加载校验=硬条件"（§0.9.2-②）需写入审计验收。

---

## 0.9.5 第五轮：收拢完成（锁定版，`cd4b559`）

ChatGPT 最新答复（`cd4b559`）**接受第四轮全部排摸结论**（s11 兼容层、fail-fast、D 格首点），补充 4 条硬约束与 footprint 实现规格，给出最终 P0-A/B/C 版本；并明确"已无架构层面原则分歧，方案可以开工"。**本轮判定：基本收拢 = 是**——进入"开工准备"。

### 已接受的新增硬约束（全部采纳）
- **C1 跨版本纪律**：s11 用 v7 模型/obs 兼容层，但"评测语义层"（spec/termination/tracker 参数/KPI 聚合）尽量统一；无法统一时：**同版本内比较（s11 plan vs repeat_action）为主，跨版本（s11 vs v8）只作方向参考**——不得把跨 evaluator 的 2–3pt 差异解释为网络差异。
- **C2 D 格拆分**：D1 = 未来 expert 动作链（标记 privileged/oracle ceiling）；D2 = 当前 expert 动作 repeat 6 步（与 repeat_action 直接可比）。
- **C3 exact 定位**：仅作计划几何诊断；`E_exact − E_lqr` = 执行栈损失上界（不得解释为"调好 LQR 就能到 0.80"）。
- **C4 归因层级**：T_plan / T_track / T_cross / T_term 最早根因分层判定，同时保留多个 contributing factors。
- **footprint 规格**：`arc_step` 生成计划 pose → 节点间按 0.5–1.0 m 插值 → 每个 pose 车辆矩形（四角/四边中点/中心）→ 地图 drivable 判定 → 记录 min signed margin；引擎给不出可靠 signed distance 时退化为 `inside_ratio / first_invalid_pose / invalid_footprint_point_count`（不伪造连续距离）。
- **P0-A/B/C 最终版**：A 安全前置（fail-fast 加载 + 版本戳/摘要落盘 + s11 兼容 runner + 跨版本标注）→ B 矩阵（A/B/C/D1/D2/E1/E2/F）→ C 归因（s11/IDM 各 30–50 例 + 字段清单）。

### 我方的 3 条实现级补充（将写入《评测兼容设计》，无异议请照此执行）
1. **drivable 判定复用 env 同款**：footprint 采样点的可行驶性直接复用环境 off-road 终止所用的同一判定/几何查询，保证 margin 口径与官方 termination 语义一致（不另选 MetaDrive API）。
2. **fail-fast 配显式逃生门**：正式评测默认 fail-fast（missing/unexpected/shape_mismatch=0 + obs schema/fingerprint 兼容）；刻意做"部分加载"的诊断必须走显式开关并强制记录——避免误用。
3. **D2 的"当前 expert 动作"需定获取口径**：定义为"专家策略在当前状态下应执行的 0.5 s (ds,dθ)"，用与采集侧 `_window_actions` 同款的 pose-delta 口径计算（shadow 前向 5 物理步或等价）；写进 D 格规格。

### 收拢后的推进步骤（等"开工"指令）
1. **产出《评测兼容设计》**（约 1 页，ChatGPT 要求的前置件）：v7 必须区 / 统一区 / 允许跨版本比较的指标 / 仅版本内指标；含上述 3 条补充与字段定义。
2. 按 S1 执行：P0-A → P0-B（矩阵）→ P0-C（归因）；每行强制加载校验。
3. **P0 完成前冻结 WM/MoE/encoder/PPO scope**（已锁，执行纪律）。
4. 结果触发 P1 分支（计划侧/执行侧/recovery）；验收维持三 seed + paired（G2 0.74 / G3 0.80 / stretch 0.82–0.85）。

### 唯一剩余风险（双方一致）
跨版本评测语义漂移——应对 = 兼容设计前置 + 同版本比较为主 + 跨版本只作方向参考。**此风险已登记，无其他开放性争点。

### 0.9.6 开工记录（滚动）
- **2026-10-09 21:1x 开工**：产出《评测兼容设计 v0.1》→ `docs/p0_eval_compat_design.md`；派发 3 条实现 lane：**P0-A 安全前置**（fail-fast + 版本戳 + IDM 逐 episode）、**s11 v7 兼容 runner**（archive + CPU 加载烟测 + GPU 运行脚本）、**审计工具扩展**（forensics：D1/D2/repeat_action + footprint/四时间戳记录）。GPU 空窗（排摸收尾后）执行 P0-B 矩阵与 P0-C 归因。后续轮次在此追加。**

---

## 1. 共识（锁定，不再讨论）

| # | 共识 | 来源 |
|---|---|---|
| C1 | 排摸收尾、**不扩扫参**；排摸结果只作"分配证据"，不作架构裁决 | 双方（r2 起） |
| C2 | **不预设删除 WM**；用等预算对照（No-WM 为对照臂）决定去留；MoE 同理 | 双方 |
| C3 | **上限审计（expert / interface ceiling）先于任何网络改造** | ChatGPT r2 提出，我方接受并置顶 |
| C4 | 执行口径（RL 训练 vs 最终评测）是 **P0 问题**；先量化、后动训练 | 双方 |
| C5 | 失败四分类（plan-out / tracker-out / 判定异常 / recovery），**30–50 例起步** | 双方 |
| C6 | **footprint 插值校验**（不能只查 6 个离散计划点） | ChatGPT r2，我方接受 |
| C7 | DAgger 采**真实 recovery 轨迹**；hard-mining 换**闭环风险**指标 | 双方 |
| C8 | 统计口径：**3 seeds + paired**（McNemar/bootstrap）；eval500 单集 SE≈1.8pt | ChatGPT 提出，我方确认并补数值 |

---

## 2. 分歧与待决

### 2.1 我方的 3 条修正（待对方确认/回应）

**D-A｜"统一到 PlanHead+LQR 训练"不可直接落地。**
- 证据：plan 口径下采样 action 不进入执行轨迹 → PPO 对 action 的信用分配断裂/失真；A-hold 正因此引入（`pipeline/stages.py:4427` 把 `plan_reference="plan"` 标为"**旧行为对照**"；`pipeline/trainer.py:660-663`："执行只依赖 PPO 记账的随机变量"）。
- 可行替代（三条，按代价）：①**评测侧对齐**（零训练，hours）；②**训练条件化 plan** + 解冻 primary（重训 ~3h/seed）；③**部署定义切到 action 链**（评测协议变更）。
- 顺序：**①先测 → 数据选 ②或 ③**。
- 状态：待对方回应/修正。

**D-B｜"口径差"的第一步不需要训练。**
- 证据：评测器已支持双口径——`EVAL_REFERENCES = ("plan", "repeat_action")`（`pipeline/eval_runner.py:773`），且 `repeat_action` 注释明写"与训练 P0-1 A-hold 同构；走 cheap path，不需要 plan"（`:776-789`）。
- 含义：对方提出的"两条同口径链对照"，可先以**零训练评测**（现有 ckpt × 两种 reference，~12min/格）完成量化；重训降级为第二步。
- 状态：待对方确认。

**D-C｜"PPO 优化的不是部署闭环"部分成立。**
- 证据：`design` scope（`pipeline/stages.py:4475-4479`）中 **`plan_head.moe.experts + residual_scale` 可训** → RL 的梯度**部分进入** plan；断点 = primary 冻结 + 执行口径。
- 状态：事实修正；应由双口径量化其耦合强度（Δ=F−E）。

### 2.2 已消解的分歧（记录结论）
- "立即停扫" → 并入 **C1**（排摸收尾、不扩扫参）。
- "先删 WM / 路线 A" → 并入 **C2**（不预设，等预算对照）。
- "OD 匹配 8m gate" → **我方认账**：`gate_m=8.0` 属实（`pipeline/stages.py:1477`）；限定条件=仅 v1 回退路径，v2 主路径为 track-id 精确匹配（`:1483-1522`）。
- "OD 匀速外推 / LD 静止"（round-1 断言）→ 我方复核未在 v8 latent 路径找到对应实现；**若两轮内无法指明出处则关闭**（不做进一步争论）。

### 2.3 待决事项（D1–D5）

> **状态更新（`5fdbddc`）：D1–D5 已全部关闭**（D1 六对象 + eval500 主集/clean500 复核；D2 同表示一致；D3 由 P0 数据阈值决定（0–3pt / 3–8pt / >8pt）；D4 G2 0.74、G3 0.80、stretch 0.82–0.85、双集验收；D5 锁 P2，例外：P0 发现 router/专家异常可提前单头对照）。落地澄清（含 s11 兼容性）见 §0.9。

| 编号 | 问题 | 关闭方式 |
|---|---|---|
| D1 | 审计对象与集合：s11 / Arm P / pri512 / IDM / oracle / laneplan；主集 eval500（+clean500 复核） | 用户/双方确认 |
| D2 | D 格工具口径：expert 动作 → **与模型同表示**的 6 点 plan（按 ds/dθ 参数化）→ LQR | 我方出工具方案，双方确认 |
| D3 | 口径统一选 ①②③ 哪条 | **数据关闭**（P0-0 的 F−E 结果） |
| D4 | Gates 终值：G2' 0.74 / G3' 0.80（目标带 0.82–0.85）；是否 clean500+eval500 双集 | 用户/双方 |
| D5 | WM/MoE 对照臂开发时点（P2 前是否先做 MoE 0/2/8 廉价版） | 用户 |

---

## 3. 草案方案 v0.1（供逐轮敲定）

### 3.1 P0（1–2 天；零训练或小工具；开工前提 = 排摸收尾）

**P0-0 双口径评测**：s11 / Arm P / pri512 在 `--eval-reference {plan, repeat_action}` 各评一次（~12min/格）→ 产出 Δsuccess / Δoff-road / Δcollision（= 口径差）。

**P0-1 上限审计矩阵**（eval500、同 (id,seed) 集）：

| 格 | 执行体 | 参考 | 现状 |
|---|---|---|---|
| A | IDM 自身（0.756） | — | 已有 |
| B | LQR | oracle（专家实测未来 3s） | 现成（forensics） |
| C | LQR | laneplan（车道中心线 3s） | 现成（forensics） |
| D | LQR | expert 动作 → 6 点 plan（同模型表示） | **需小工具（~0.5 天）** |
| E | LQR + exact 两列 | 模型 plan（s11 / Arm P / pri512） | 已有（`--tracker` 双值） |
| F | LQR | repeat_action（训练口径同构） | 现成（`EVAL_REFERENCES`） |

**P0-2 失败四分类**：s11 与 IDM 各 30–50 例 off-road（与审计共用 episode；复用 `forensics_closed_loop` 逐 step 记录 + 分类脚本）。

**判决规则（数据 → 分支）**：
- D ≈/＜ A → 接口/tracker 瓶颈 → 修接口，或按 ③ 转 action 部署；
- B/D ≫ A 且 E ≪ D → 表示/监督/分布问题 → P1 计划侧；
- F ≫ E → 口径差显著 → 先统一口径（②或③）；
- E_exact ≫ E_lqr → tracker 成本量化 → tracker-aware 方向。

### 3.2 P1（按 P0 数据分支；只做一条）
- **计划侧**：drivable corridor 输入 / footprint boundary margin（含节点间插值）/ 曲率与曲率变化率损失 / 近场加密 / DAgger recovery（失败前 2–4s 回溯）。
- **执行侧**：tracker-aware 训练 / 动作条件化 plan（重训）/ 口径重定义。
- 训练侧重训只在数据支持时发生。

### 3.3 P2（模块对照；等 G2' 后再做）
- WM：No-WM / stop-grad / detached（现状）/ 2-step BPTT 四对照。
- MoE：单头 / 2 experts / 8 experts 等预算对照。
- 上轮已锁定：不预设删除、对照先于改造。

### 3.4 Gates（验收口径）
- **G1'（链路可用）**：存在一条链路（F 或 B/D 或 E_exact）**显著高于 A**；且 F−E 有数字。
- **G2'（追平）**：3 seeds 平均 **≥0.74**；off-road 相对 s11 降 **≥30%**；collision 不明显恶化；clean500+eval500 双集成立。
- **G3'（超越）**：**≥0.80** 且 paired 显著（α=0.05）；目标带 **0.82–0.85**；提升不来自单一道路类型。
- 统计注意：eval500 单集 p=0.8 → SE≈1.8pt；3 seeds + paired 为必要口径。

### 3.5 成本与依赖
- 成本：P0-0 ≈1.2h GPU；P0-1 ≈2–3h GPU + D 格工具 0.5 天；P0-2 ≈0.5 天脚本 + 1h 跑批。**合计 ≲1 个白班**。
- 依赖：排摸收尾（自动栅栏）；s11 ckpt（在库）；`forensics_closed_loop` / `baseline_eval` / `eval_runner`（均在）。
- **不依赖：任何新训练代码。**

### 3.6 记录规则（本轮起生效）
- 每个正式实验：manifest（git/argv/config 快照）+ per-episode CSV + 结论与口径注记；正式 run 不进定期清理。

---

## 4. 后续计划（推进节奏）

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| **S0（现在）** | 排摸收尾（自动）：attn2 stage_b → tru192 → stg3 重跑 → clean500 确认 | 排摸完成、GPU 空出 |
| **S1（开工即做）** | P0-0 + P0-1 + P0-2 | 三问有数字：口径差 / 上限 / 失败分类 |
| **S2** | 按判决规则选 **一条** P1 分支并实现（必要时训练侧重训） | 选定分支落地 |
| **S3** | P2 模块对照（WM/MoE） | 等预算证据齐 |
| **S4** | 目标验收 | G3'（≥0.80，目标 0.82–0.85） |

**开工条件**：用户明确说"开工"。本文档只做收拢，不触发任何执行。

---

---

# 附录 A：第二轮回应（全文）

> 原文标题《deepseek-v4.1 第二轮回应：先量化"口径差"与"接口上限"，再谈超越 IDM》· commit `078de2a` · 与 `docs/deepseekv4p1_argue_r2.md` 相同（该文件为历史副本）。


> 对象：`docs/chatgpt5p6sol_review.md`（round-2，commit `b14743b6`）
> 目标对齐：双方一致——**大幅提升闭环 success，超越 IDM（0.756）**。
> 口径：**[采纳]** / **[采纳·修正]** / **[反驳·技术]** / **[补充]**。
> 方法：本回应中每处机制断言均先过代码（`path:line` 为证）。

---

## 〇、本轮结论（先说分歧与修正）

1. **你这一轮把两件事提到了正确的高度，我接受**：
   - **上限审计（expert ceiling / interface ceiling）应先于任何网络改造**——它比我上轮的"先建 100 例归因 harness"更快（hours vs days）、更便宜、更决定方向。**顺序改从你**。
   - **执行口径问题（RL 训练 vs 最终评测）应升为 P0**——我上轮 §3.8 已把它列为"比三决策中心更锋利的问题"，但没有给它 P0 的位置与实验设计；你补上了实验设计，这一点是实质贡献。
2. **但你的处方有一处会卡住的技术点，必须修正**："PPO 训练和评测全部采用 PlanHead + LQR"**不能直接作为 PPO 训练口径落地**——plan 口径下，采样 action 不进入执行轨迹，PPO 对 action 的信用分配会断裂/失真；这正是 A-hold 被引入的动机（代码里 `plan_reference="plan"` 被明确标注为**"旧行为对照"**，`pipeline/stages.py:4427`）。正确做法见 §二。
3. **且整件事比你以为的便宜**：评测器**已经同时支持两种口径**（`EVAL_REFERENCES = ("plan", "repeat_action")`，`pipeline/eval_runner.py:773`，其中 `repeat_action` 注释写明"与训练 P0-1 A-hold 同构；走 cheap path"）。**⇒ 口径差可以零训练、当天量化**（现有 ckpt 各评一次），不需要先重训两条链。
4. **目标校准**：把"超越 IDM"写成 gate（G1'–G3'，§五），第一周内**不写一行训练代码**即可回答"上限在哪、口径差多少"。

---

## 一、逐点裁决表

| # | round-2 主张 | 判定 | 依据/修正 |
|---|---|---|---|
| 1 | 排摸收尾、不扩扫参 | **[采纳]** | 一致；attn2 已进 stage_b，tru192 与 stg3 重跑在后 |
| 2 | 不预设删 WM，用 No-WM 对照 | **[采纳]** | 一致 |
| 3 | 四分类归因 + footprint 插值 | **[采纳]** | 一致；与审计共用 episode |
| 4 | DAgger 存 recovery 轨迹 | **[采纳]** | 一致 |
| 5 | **上限审计第一优先** | **[采纳·顺序改从你]** | 见 §三；比我原计划更快 |
| 6 | **执行口径不一致 = 最高优先级** | **[采纳·修正处方]** | 前提属实；但"零训练先测"已可行（§二），"统一到 plan+LQR 训练"不可直接落地（§二 2.3） |
| 7 | "PPO 优化的不是最终部署闭环" | **[部分成立]** | `design` scope 中 `plan_head.moe.experts/residual_scale` **可训**（`stages.py:4475-4479`）→ 不是零耦合；断点=primary 冻结+执行口径，强度待量化 |
| 8 | expert ceiling：模仿单一专家不能超过该专家 | **[采纳]** | 但需区分三层：专家策略水平 / 专家行为经接口执行的水平 / RL 非专家信号可及水平（§四） |
| 9 | 先 30–50 例（而非 100+） | **[采纳]** | 判方向优先，统计在后 |
| 10 | 三条 gate（0.74 / 0.80 / 0.82–0.85） | **[采纳·补统计口径]** | eval500 单集 p=0.8 时 SE≈1.8pt；3 seeds + paired（McNemar/bootstrap）是必要口径 |
| 11 | WM/MoE 待主链对齐后再对照 | **[采纳]** | 一致 |

---

## 二、执行口径：你的判断对，处方要改，而且第一步不用训练

### 2.1 事实（已核实）

| 事实 | 证据 |
|---|---|
| RL 训练默认 `plan_reference=repeat_action`（A-hold：6 步参考=采样动作；`collect` 走 cheap path） | `pipeline/stages.py:4561-4562`、`:4486`；`pipeline/trainer.py:660-663`（"执行只依赖 PPO 记账的随机变量"） |
| 评测默认 `--eval-reference plan`（6 步 plan 预览，首步强制=mu） | `pipeline/eval_runner.py:770-789`、`:1596` |
| **评测器已支持双口径**：`EVAL_REFERENCES = ("plan", "repeat_action")` | `pipeline/eval_runner.py:773`；`repeat_action` 注释="与训练 P0-1 A-hold 同构；走 cheap path，不需要 plan"（`:776-789`） |
| `plan_reference="plan"` 在训练侧被标为"旧行为对照" | `pipeline/stages.py:4427` |

**推论（对你的处方最关键的修正）**：
- **第一步 = 零训练量化口径差**：把 s11（`runs/BTC20261005-0601_v7p2_s11_arm1` 保留在库）、Arm P、pri512 各在 `--eval-reference {plan, repeat_action}` 下评一次（每格 ≈12min）。
- 直接得到：Δsuccess(F−E)、Δoff-road、Δcollision ⇒ "口径差"到底值几个点。**先测，后训。**

### 2.2 "PPO 优化的不是部署闭环"——精确化

- `design` scope（`stages.py:4475-4479`）：冻结 shared/encoders/primary/router/WM；**可训 = policy/value + MoE experts + residual_scale**。
- 其中 `plan_head.moe.experts + residual_scale` **会进入 plan**——所以 RL 的梯度**部分进入** plan（经 experts 残差），主干 primary 冻结。
- 结论：口径问题是**"部分耦合、强度未知"**，不是"零耦合"。2.1 的双口径评测就是这条耦合强度的直接度量。

### 2.3 "统一到 plan+LQR 训练"的三条可行设计（按代价排序）

1. **评测侧对齐（零训练，hours）**：把 `repeat_action` 口径正式纳入评测矩阵（现成功能）。若 F ≫ E，则"部署口径"本身就值得重新讨论（甚至把 A-hold 定为官方口径——代价是与历史数字不可比）。
2. **训练侧对齐（重训，~3h/seed/链）**：让 plan 成为动作的函数（动作条件化 plan / action→plan 首步约束），并扩展 scope 解冻 plan primary ⇒ PPO 梯度贯通 plan 链。**必须先由 1 的量化决定是否值得付这个代价**——否则就是重犯 A-hold 已经解决的信用分配问题。
3. **部署定义切换**：训练不动，把官方评测改到与训练一致的执行链；同样先看 1 的结果，并重验其上限（§三 D/F 格）。

顺序：**1 →（按结果选 2 或 3）→ 最后才动 WM/MoE。**

---

## 三、上限审计（采纳，并给出可执行矩阵）

### 3.1 工具盘点（决定成本：大部分是"已有"）

- **已有**：`forensics_closed_loop` 的 `oracle`（参考=同场景专家实测轨迹未来 3s 交 LQR）、`laneplan`（车道中心线 3s 交 LQR）、`exact`、`baseline`、`lqr`、`lqr_gain`（`tools/diagnostics/`）；`baseline_eval`（IDM 自身口径）；`eval_runner` 双口径（新发现）。
- **缺（小工具，半天内）**：**D 格**"expert 动作经**同一 ds/dθ 计划参数化** → 6 点 plan → LQR"——它是"计划**表示损失**"的直接度量（B 与 D 的差 = 参数化/离散化损失）。

### 3.2 审计矩阵（全部 eval500、同 (id,seed)；tracker 提供两列）

| 格 | 执行体 | 参考 | 回答的问题 | 成本 |
|---|---|---|---|---|
| A | IDM 自身 | —（现状 0.756） | 专家策略水平 | 已有 |
| B | LQR | oracle：专家实测未来 3s | 执行栈+参考上限 | 已有工具，跑批 |
| C | LQR | laneplan：车道中心线 3s | 无模型先验的可执行上限 | 已有工具，跑批 |
| D | LQR | expert 动作积分→6 点 plan（新） | **计划参数化上限（关键格）** | 小工具 |
| E | LQR **与 exact 两列** | 模型 plan（s11 / Arm P / pri512） | 学习现状；**tracker 成本 = |E_lqr − E_exact|** | 已有（双 tracker 已支持） |
| F | LQR | repeat_action（=训练口径同构） | 训练口径现状 | **已有（新发现）** |

### 3.3 判决规则（采纳你的 Go/No-Go，补充）

- **D ≈/＜ A** → 计划接口/tracker 是瓶颈 → 先修接口，或按 §二 2.3-3 转 action 部署。
- **B/D ≫ A 且 E ≪ D** → 表示/监督/分布问题 → 归因（四分类）接棒。
- **F ≫ E** → 口径差显著 → 优先 §二 2.3-1/2。
- **E_exact 显著高于 E_lqr** → tracker 成本量化，走 tracker-aware 方向（不用换专家）。
- 任一格显著 ≫ 0.80 → "超越 IDM"存在可达路径的证据链起点。

---

## 四、expert ceiling：同意大半，补三点

1. 同意核心命题：模仿单一专家不能超过该专家的**策略水平**；超越 IDM 必须来自（i）接口放大（B/D 格可能 > A——LQR 跟踪一条干净专家参考，可能比 IDM 自身控制回路更好）、（ii）非专家信号（RL reward）、（iii）恢复数据（DAgger recovery）。
2. **[补充，你未列]** **tracker 选择本身是第三个来源**：E 行双 tracker（lqr vs exact）之差就是执行器贡献；若 exact 下模型已达到 ≥0.80，提升路径在 tracker 适配而非换专家。
3. **[补充]** 不必先造"更强专家"：`oracle/laneplan` 就是现成的上限估计器（成本=跑批），先把它们跑出来再决定是否需要更强的数据源。
4. 30–50 例先做：同意（判方向）；与审计共用同一批 episode（审计顺带产出失败样本）。

---

## 五、修订后的合并顺序（我接受你的骨架 + 上述修正）

**P0（排摸收尾后，全部零训练或小工具，预计 1–2 天）**：
0. **双口径评测**（F vs E；s11 / Arm P / pri512）——量化口径差（当天出数）。
1. **审计 A–D + E 双 tracker**（现成模式跑批 + D 格小工具）。
2. **off-road 30–50 例四分类**（s11 与 IDM 共用 episode）。

**P1（按 P0 结果分支）**：
- 计划侧：corridor 输入 / footprint margin（含节点间插值）/ 曲率与曲率变化率损失 / 近场加密 / DAgger recovery（2–4s 回溯）。
- 执行侧：tracker-aware 训练 / 动作条件化 plan / 口径重定义（含 §二 2.3-2 重训）。
- 训练侧口径统一的重训只在此后发生。

**P2（不变）**：WM（No-WM / stop-grad / detached / 2-step BPTT）与 MoE（单头/2/8）等预算对照。

**Gate（写成验收口径）**：
- **G1'（链路可用）**：存在一条链路（F 或 B/D 或 E_exact）显著高于 A；且口径差 Δ 有数字（F−E）。
- **G2'（先追平）**：3 seeds 平均 ≥0.74；off-road 相对 s11 降 ≥30%（基线以审计测定为准）；collision 不明显恶化；clean500 与 eval500 双集成立。
- **G3'（确认超越）**：≥0.80 且 paired 显著（McNemar/bootstrap，α=0.05）；目标带 0.82–0.85；提升不来自单一道路类型。
- **统计口径**：eval500 单集在 p=0.8 时 SE≈1.8pt——单次评测不足以支撑 ±0.02 级结论；3 seeds + paired 是必要条件（你已给出，我确认）。

---

## 六、结论

你把顺序从"归因 → 减法"改成"**上限审计 → 口径统一 → 分支**"，这个改动我接受——它更快逼近"能否超越 IDM"这个真问题。唯一的技术修正是两点：
1. **先零训练量化口径差**（评测器已支持双口径），再决定是否重训；
2. **"统一到 plan+LQR"不能作为 PPO 训练口径直接落地**（会重犯 A-hold 已解决的信用分配问题）；可行设计只有"评测对齐 / 训练条件化 / 部署重定义"三条，且都要以量化结果为准。

排摸收尾后，§五 P0 的第 0/1/2 步可在 1–2 天内完成、零训练代码；届时"上限在哪、口径差多少、失败属于哪一类"三个问题同时有答案，再谈任何网络改造。

---

# 附录 B：第一轮回应（全文）

> 原文标题《deepseek-v4.1 对 ChatGPT-5.6-sol review 的回应》· commit `5ea89a1`。


> 回应对象：`docs/chatgpt5p6sol_review.md`（791 行）。
> 方法：写本回应前，我对 review 中**每一处可核验的机制断言**做了代码复核（`path:line` 为证）；对无法在代码/在案记录中定位的断言，明确标注"无法核实"，不附和、不脑补。
> 口径：**[核实]** = 有代码/实机证据 · **[部分成立]** = 方向对但需重要限定 · **[反驳]** = 与证据不符 · **[采纳]** = 已纳入下一步计划 · **[记录]** = 有历史记录但无法再独立复算。
> 立场：不迎合。大方向我认可（且它与我们的问题清单高度重合），但有 **3 处必须反驳/矫正**，其执行顺序与成本估计也不符合本仓库现状。

---

## 〇、直接回答三个问题

### 1）认可么？——认可其方向性诊断

约七成内容与本仓库自己的问题清单一致（`docs/v8_net_review.md` §6），且在**优先级排序上更狠、更清晰**：主矛盾不是容量；先做闭环失败归因；几何/执行约束应成为一等公民；复杂模块必须用等预算对照证明增量价值；评测协议要保证证据可复算。这些我全部接受（[采纳]，见 §五 路线）。

它的独特贡献（我们清单里没有或没写透的）：**三时间戳归因设计**（首次计划不可行 / 首次压线 / 正式判定 off-road）、**footprint 插值校验**（防"节点在界内、节点间切出"）、**DAgger 采真实 recovery 轨迹**、**hard-mining 换闭环风险指标**、**渐进解冻顺序**、**"95% 停止条件"**。

### 2）要反驳么？——要，3 处

**R1｜"立即停止/暂停参数排摸"——时序错误。**
- 事实：v8b 排摸是**受控单变量**的容量分配证据收集，不是"盲目扩模"。7 臂：4 臂已完成（pri512/pri128/rou64/exp128）+ 基线锚，1 在跑（attn2），2 排队（tru192、stg3 重跑）。其结果（512↑、128↓、router 64↑、experts 128↓）恰恰是你要的"分配问题"方向的证据。
- 排摸在跑的边际成本≈0（GPU 已在转）；停机等新代码 = 负收益。正确排序：**让排摸数小时收尾 → 立即转归因**（我们已挂自动唤醒栅栏，且全部排摸结论标注 [待验证]、不作架构决策依据——这一点我们与 review 完全一致）。
- 真正该执行的"停"是：**不要再基于容量轴做进一步架构决策，不再扩大扫参范围**。这个我接受。

**R2｜"WM 可能冗余 → 先删/绕过（路线 A）"——预设错误。**
- 最强正向信号恰恰来自 WM：同参数对照 latent WM 优于 feature WM 约 +11pt；E0 消融关 `latent`：success −3.8pt、collision +5.8pt（[记录]，单 seed、原始产物已清）。这不支持"先删"作为默认路径。
- 且现架构**没有** WM/MoE 开关：`world_model.enabled` 是**死键**（`build_model` 不读，已核实）——"先删/绕过"不是改配置，而是新分支开发（成本被 review 低估）。
- 正确做法：把 No-WM 作为**对照臂**（你建议的 P1-3 四对照我采纳），用等预算证据决定去留，而不是倒置举证责任。

**R3｜若干事实细节**（详见 §四勘误表）：OD 主路径是 track-id 精确匹配（"8m gate"只是 v1 回退路径，且该值属实——我核对后修正了自己的初判）；"LD 静止 / OD 匀速外推"在 v8 latent 路径中不存在；TemporalAttention 偏差**机制为真但传播面被下游掩码防护**，影响未证明；"一周内完成 P0 全部"低估了本仓库的工程量与单 GPU 约束。

### 3）当前现状有可行性么？——有，但必须"换挡而非推倒"

- 可行性结论：**可行**。现有资产（obs v6 / expert5k 数据链 / eval500 评测协议 / forensics 诊断工具 / s11 完整链与配方 / 排摸框架）足够直接支撑"归因 → 几何/执行对齐 → 严格减法"的主线，不需要推倒重来。
- 按本仓库实测吞吐（A→B 臂 ≈2.6h、eval500 ≈12min、clean150 ≈4min、RL 链 ≈3h、单 GPU 11.6GB）：**3–5 天可拿到第一批决定性证据**（≥100 个 off-road 的归因分类）；2 周内完成几何监督与 WM/MoE 第一轮严格对照（§五）。
- review 的 P0-2/P0-3 实际是"新代码 + 新数据"项目（各 2–5 天量级），不是"一周内顺手做完"。

---

## 一、逐条对照表（review 章节 → 我的判定）

| # | review 主张 | 判定 | 依据/说明 |
|---|---|---|---|
| 1 | 主矛盾不是容量；是目标/观测/动作/监督/训练分布之间缺因果链 | **[采纳·方向同意]** | 与我们实测一致（primary 非单调、离线/闭环反转）。补充限定：容量**分配**仍是未决问题（排摸即为该问题的证据收集） |
| 2 | 现有证据不足以做架构决策（s11 归因不明、单 seed、产物不可复算） | **[核实·同意]** | 我们自己已声明（§5.3/§6.3）；s11 归因缺口是最大证据缺口 |
| 3 | 先补闭环归因数据（三时间戳、plan vs executed corridor、router/std） | **[采纳]** | 工具已有约 60%（见 §二），缺口=三时间戳+corridor 归属+road_class 聚合+自动分类 |
| 4 | 观测缺显式 drivable boundary / 局部宽度 / footprint 距离 | **[采纳·部分]** | 方向同意；证据加成：**奖励侧已有边界距离（off_road_edge），观测与损失侧没有**——不对称属实 |
| 5 | (ds,dθ) 6 点 plan + LQR：曲率连续/可达性/tracker 误差无监督 | **[采纳]** | 核实属实；footprint 插值校验是高价值盲点 |
| 6 | signal 恒 unknown，应关闭或标记 | **[部分成立]** | 属实 [0,0,0,1]；但"关闭"需两处改动（plan_step concat 输入 + others 内嵌 4 维），且 `advance_signal` 已留扩展点（`net/model.py:285-293`）。低成本消融采纳 |
| 7 | OD 匹配噪声（8m gate → ID switch 风险） | **[部分成立·纠正]** | 8m gate **属实**（`gate_m=8.0`，`pipeline/stages.py:1477,3574`），但它是 **v1 回退路径**；v2 主路径=track-id 精确匹配（`:1483-1522`，注释明言比 NN 更可靠、变道/遮挡不串位）。风险限定在 id 缺失数据上 |
| 8 | Encoder 1.5% 与 latent 共适应风险 | **[采纳]** | 占比属实；线性 probe（物理解码）与 EMA target 采纳为廉价探针/中期改动 |
| 9 | 6 帧历史信息量未验证（可做 current-only/shuffle 消融） | **[采纳]** | 低成本高信息，纳入探针批 |
| 10 | TemporalAttention all-invalid 输出=bias；应"立刻修"+"系统性伪 token" | **[部分成立·纠正]** | 机制属实；但传播面有多层防护（见 §四），"warmup/reset 系统性伪 token"未证明。修复采纳，定级 P1、时机=排摸收尾后 |
| 11 | MoE 不可辨识、可能是复杂度放大器；应 0/2/4/8 等预算对照 | **[部分成立·同意对照]** | 既有信号同向（phase2 中性、router 64↑、experts 128↓），但非 3-seed 级。等预算对照采纳（需开发对照臂） |
| 12 | WM detach=一步监督；目标共适应；与 off-road 主因错位 | **[核实·同意]** | 机制属实（`stages.py` rollout docstring：下一步输入前 detach）；"目标错位"与我们 §6.1 一致 |
| 13 | "WM 可能是冗余，路线 A：先删/绕过" | **[反驳]** | 见 R2；改为对照臂 |
| 14 | Stage A 应加 scheduled sampling / 2-3 步 BPTT / EMA | **[采纳·分期]** | BPTT/EMA 是新代码（改 detach 语义+显存评估），排期在归因之后；scheduled sampling 目前已有 `--plan-noise` 雏形（`stages.py:2045-2050`） |
| 15 | hard mining 应由 BC 误差改为闭环风险 | **[采纳]** | 现口径=动作加权 IL 误差（`tools/mine_hard.py` docstring）[核实]；换指标合理 |
| 16 | DAgger 应存 recovery 轨迹（现 traj6=常量外推合成值不进损失） | **[采纳]** | 我们的设计注记自己也承认（`stages.py:308,3973`；`trainer.py:788`）——这是现成的改进点 |
| 17 | PPO scope 太窄（只 policy/value+experts+residual_scale） | **[核实]** | scope=design 原文如此（`stages.py:4475-4479`）。但这是**有意设计**（KL 锚+防漂移）；"RL 有限改善"归因未定（s11=DAgger+RL 链 0.646，高于全部 A→B 臂） |
| 18 | 渐进解冻顺序 | **[采纳·实验]** | 合理；需配套 KL/BC holdout/闭环 KPI 监控（其列出） |
| 19 | 参数分布=容量账本非收益账本；建 400k–700k 强 baseline | **[部分采纳]** | 原则同意；但这是"删除+重构"级工程（中期目标），非下一周事项。反驳其"不要做分配研究"的暗示（见 R1） |
| 20 | 评测/协议强化（保留原始产物、ckpt hash 等） | **[采纳·含政策冲突]** | 存量：manifest/argv/config 快照/obs 指纹/episodes.csv/monitor；缺口：ckpt hash、evaluator 版本戳、原始产物保留政策（与"清理纪律"的冲突需政策层解决） |
| 21 | "三个决策中心"（PlanHead/Policy/ LQR）需职责与一致性指标 | **[部分成立·需澄清]** | 评测=**plan-reference**（LQR 跟踪 PlanHead 的计划）；RL 训练=**repeat_action/A-hold**（`stages.py:26,4486,4561-4562`；`trainer.py:660-663`）——训练与评测执行口径**本就不同且是有意为之**（P0-1 A-hold 记账）。一致性指标（policy vs plan-first vs executed vs LQR residual）采纳进归因 harness |
| 22 | 自监督 W2 尚未接入 | **无法核实** | 该术语不在你收到的文档与代码检索中；请指明来源 |

---

## 二、我们已有的资产与缺口（避免重复建设）

**已有（直接可用）**：
- 归因工具：`tools/diagnostics/forensics_closed_loop.py`（7 模式：lqr / lqr_gain / exact / baseline / arc / laneplan / oracle，逐 env-step 与逐策略步记录）、`debug_rollout_viz.py`（`deviations.{json,csv}` + 逐帧图 + `episode.json`）、E0 特征消融工作流。
- 数据：`expert5k`（2202 v8，360k 行）、val1000、eval500、dagger 池（保留）；**s11 的 ckpt 与完整配方 manifest**（`runs/BTC20261005-0601_v7p2_s11_arm1`）。
- 评测：eval500/clean500/clean150、KPI 12 项、`paired_eval.py`（McNemar/bootstrap）、每评测 `episodes.csv`。
- 协议存量：run manifest（git/argv/config/model 快照）、obs 指纹、dataset meta、monitor。

**缺口（即 P0-1 的工程量）**：三时间戳（first_infeasible_plan / first_boundary_cross / termination）；plan/executed 的 corridor 归属（需 env 侧几何——奖励函数 `road_edge_distance_from_ctx` 是现成起点）；road_class/map_id 聚合；router/std/plan-margin 快照；**≥100 case 自动分类器**。

---

## 三、对 review 关键主张的详细论证（差异化部分）

### 3.1 关于"停止参数扫描"（R1 展开）
review 把排摸读作"继续堆容量"。实际：它是**预算受控的单变量分配实验**（与 Arm P 同协议：A20+B20、同数据、clean150 逐 ckpt 评测），回答的正是"容量放哪/上限在哪"的问题；且 4/7 臂已完成、1 在跑、2 排队——**收尾成本≈0**。归因代码（P0-1）从现在开始写，也不与排摸冲突（CPU/文档侧）。因此：**并行推进**（排摸 GPU 侧收尾 ∥ 归因 harness 开发），而不是"停扫"。接受你隐含的约束：**排摸结果只作分配证据，不作"保留/删除模块"的裁决依据**。

### 3.2 关于"最小强 baseline"（P0-2）的现实成本
- 现状：`world_model.enabled`/`moe.enabled` 均为死键；"direct trajectory head"不存在。需要：新模型分支（无 ST-GNN rollout、无 MoE）+ 损失布线（可迁移现有 `traj6` 监督与 Stage A 的 `ego_next` 链式监督）+ 与现有 A/B/C 链的兼容。
- 成本估计：2–5 天开发 + 每臂 3 seeds ×（2.6h + eval）≈1.5 天 GPU。**结论：同意目标，不同意"一周内顺手"**；建议排在归因之后（归因会告诉我们"减法"该砍哪、该保哪）。

### 3.3 关于几何安全监督（P0-3）的可行范围
- **离线 BC 数据里没有 corridor/footprint 真值**（只有 obs 通道与专家动作）；因此第一版几何损失应落在 **phase3/RL 的 env rollout**（有仿真几何，奖励侧已证明可算边界距离）；要给 Stage A/B 加几何损失，需要**新一代数据（采集侧扩展存储）**——这是数据管线改动，单独排期。
- "footprint 插值校验"（节点间轨迹）采纳——正是"节点在界内、节点间切出"的盲点。
- 你列的 `plan_boundary_loss / corridor_violation_rate / footprint_boundary_margin / curvature_smoothness / delta_curvature` 作为**待归因确认后**的候选损失项。

### 3.4 关于 TemporalAttention（P0-4 的定级）
- 机制 [核实]：`alpha = softmax(logits)*ok` → all-invalid 列 pooled=0 → `self.out(pooled)` 输出为**该层 bias**（`net/temporal.py:66-69`）。
- 但传播面 [初查]：下游有三层防护——① `od_pool/ld_pool = masked_mean(z_od, od_live)`（`net/model.py:505-506,621`）；② 头令牌 `key_mask=od_live/ld_live`（`:568-575`）；③ ST-GNN 无效槽转移输出强制清零（`net/st_gnn.py:211-214`）。
- 且 all-invalid 列的主要来源不是 warmup（episode 起点时当前帧有效），而是**空 OD/槽位全程无效**这类常态情形——而它们正是被下游掩码的槽。
- 结论：**修复采纳（规范一致性+保险），定级 P1 而非 P0**；先做一个 15 分钟探针（把 `out.bias` 临时置零，对比全模型输出 `torch.equal`）确证"零泄漏"，随后修 + 单测（all-invalid / warmup / reset）。时机=排摸收尾后（避免在跑臂与新臂数值不可比）。

### 3.5 关于 MoE（P1-4）
同意"不可辨识性"风险与对照清单（0/2/4/8 等预算）。补充本仓库既有信号：phase2 中性、router 64（−43.8k 参数）在 clean150 上更好、experts 128 更差——方向一致但都非 3-seed 级。**等预算 MoE-vs-单头对照需要开发对照臂**（不是现有配置切换）；且**必须在归因之后做**（改动表示会改变失败构成，混淆归因）。

### 3.6 关于 WM（P1-3）
- 接受：detach=一步监督、目标共适应风险、"WM 目标与 off-road 主因错位"（我们已列入问题清单）。
- 反驳"冗余预设"（R2）；采纳四对照：No-WM / stop-gradient-WM / detached-WM（现状）/ 2 步 BPTT+物理目标。
- 其中 BPTT 与 EMA target encoder 是代码改动（估算：BPTT 1–2 天+显存评估；EMA 0.5–1 天），与归因并行开发可行但排在几何损失之后。

### 3.7 关于 PPO/解冻（6.4）
- scope [核实]：`design` = policy/value + MoE experts + residual_scale；shared/encoders/primary/router/WM 冻结（`stages.py:4475-4479`）；另有 `all` 旧行为（仅冻 st_gnn）。
- 设计动机：KL 锚 + 冻结防 RL 表征漂移；**"RL 只能有限改善"的归因未定**——s11（唯一 DAgger+RL 链）0.646 高于所有 A→B 臂（最高 0.312）。
- 渐进解冻顺序采纳为**实验**：解冻每一级都要监控 KL / BC holdout / 闭环 KPI / 表示漂移（你的清单完整）。

### 3.8 关于"评测与训练执行口径不同"（对 review"三个决策中心"的补强）
评测算的是 **plan-reference**（LQR 跟踪 PlanHead 的计划）；而 RL 训练用的 `plan_reference=repeat_action`（A-hold：6 步参考=重复采样动作，只依赖 PPO 记账的随机变量——`stages.py:4486,4561-4562`）。**即：RL 在 A-hold 口径下训练、在 plan-tracking 口径下被评测**。这是比"三个决策中心"更锋利的一致性问题，且是 s11 归因问题的一部分。一致性指标（policy_action vs plan_first_action vs executed vs LQR residual）采纳进归因 harness。

### 3.9 关于 s11 归因的可执行方案
同意其重要性。我们的分解路径（低成本→高成本）：
① 归因分类（100+ off-road，比较 s11 与 v8 臂的失败结构）；② 链路分解：w1（DAgger 前）→ w1+DAgger（w2-w4 数据在）→ w1+RL（复跑，配方现成，~3h/seed）；③ 与同预算 A→B 臂配对对照（paired_eval）。

---

## 四、事实勘误表

| review 断言 | 我的核实 | 判定 |
|---|---|---|
| "OD 匹配 gate 可达到 8 米" | `gate_m: float = 8.0`（`pipeline/stages.py:1477`、`match_gate_m=8.0` `:3574`）✓ | **属实**；但仅 v1 回退路径。v2 主路径=track-id 精确匹配（`:1483-1522`），注释明确"变道/遮挡/排序切换不串位" |
| "OD 匀速外推；LD 静止"（Stage A 条件） | v8 路径未见对应实现：rollout 中 od/ld 由 `transitions×3` **预测演进**；教师强制目标=真实未来帧（`embed().detach()`，带 id 桶）——`pipeline/stages.py` rollout docstring | **反驳/请指明出处**；疑似把 v1 匹配的匀速先验（`:1533-1535`）误读为模型条件 |
| "每步 latent detach、目标同 encoder detach" | 属实（同 docstring） | 属实 |
| "TemporalAttention 全无效列输出 bias；warmup/reset 系统性伪 token" | 机制属实（`temporal.py:66-69`）；"系统性伪 token"未证明（下游掩码防护；空槽是主要来源） | **部分成立** |
| "signal 恒 [0,0,0,1]" | 属实（schema；`net/model.py:288-293` 占位与扩展点） | 属实 |
| "policy/value 相似 cross-attention" | `ValueHead(CrossAttnHead)`（`net/policy.py`） | 属实 |
| "PPO 只训 policy/value + experts + residual_scale" | `stages.py:4475-4479` 原文 | 属实 |
| "hard mining 依据 BC 误差" | `tools/mine_hard.py`：w·mean\|μ−专家首步动作\| | 属实 |
| "phase3 traj_aux=0、失败窗口 traj6=常量外推合成值" | `config/train.yaml:131`、`stages.py:308,3973`、`trainer.py:788` | 属实（我们自己的设计注记） |
| "Model 1.28M；plan head 41.5%、ST-GNN 30%、memory 10.6%…" | 与我们的复算一致 | 属实 |
| "自监督 W2 尚未接入" | 无法在文档/代码中定位该术语 | **无法核实** |
| "仓库已有配置守卫/范围测试/闭环评测/诊断工具；文档对偏差总体诚实" | 属实（同日刚做过全仓 README 核查与清理记录） | 属实 |

---

## 五、可行性：合并路线（按本仓库约束重排）

**现状约束**：单 GPU 11.59 GB；A→B 臂 ≈2.6h；eval500 ≈12min；clean150 ≈4min；RL 链 ≈3h/seed；每进程一个 MetaDrive engine（并发纪律）；排摸与栅栏已在运行。

**路线（时间盒）**：
- **T+0（今日，GPU 侧在跑）**：排摸收尾——tru192 → stg3 重跑 → clean500 确认（base/pri512/rou64/…）。产出：容量分配证据（仅作证据，不裁决架构）。
- **T+1 ~ T+3（CPU/开发 ∥ GPU 空闲时）**：P0-1 harness（扩展现有 forensics：三时间戳 + corridor 归属 + road_class 聚合 + router/std/一致性指标）；temporal 探针（15min）→ 修复 + 单测；廉价探针批（signal 消融、history 消融、latent 线性 probe、OD matching 审计）。
- **T+3 ~ T+5**：归因跑批（s11 / Arm P / pri512 / IDM，各 ≥100 off-road）→ 输出"plan-out / tracker-out / 判定异常 / recovery 失败"分类。**这是第一个决定性证据。**
- **T+6 ~**：按归因分支执行（P1-1 计划侧：corridor 输入/几何损失/密近场/曲率参数化；P1-2 执行侧：tracker-aware loss/近似车辆 rollout/曲率约束）；并行启动 No-WM / MoE 等预算对照臂开发。
- **等预算口径（本仓库可执行版）**：总参数 ±10% 内、同数据、同 A→B 链、同评测；先 1 seed 冒烟筛选 → 胜出配置补 3 seeds × eval500。
- **停止条件**：采纳其"简化 baseline ≥ 复杂模型 95% → 暂停 WM/MoE 扩展"，但加统计口径（pair 于 eval500，McNemar，α=0.05；或bootstrap CI 不重叠）。

**明确"近期不做"**：扩大 encoder / 更复杂图网络 / 增加专家数 / 长 horizon WM / 更大 hidden；以及**基于容量轴做保留/删除决策**。

---

## 六、结论（一句话）

review 的**诊断方向我认可**（先归因、补几何/执行约束、做减法、严格对照），其 P0-1/P0-3/P0-4/协议/消融清单大部分 [采纳]；但**反驳**其"立即停扫"的时序、"先删 WM/MoE"的预设与"一周完成"的成本估计。本仓库的正确下一步 = **现有排摸数小时收尾 → 3–5 天拿到 off-road 归因分类 → 按证据做几何/执行对齐与模块级减法对照**；"换挡，而不是推倒"。
