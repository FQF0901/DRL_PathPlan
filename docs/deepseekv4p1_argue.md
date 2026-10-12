# deepseek-v4.1 argue / 主文档（Round-20 收官 · 转数据侧 · round-21 输入）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-12 深夜）**：P1-v2 条件轨已关闭（Round-12A NO-GO）；Round-15A/A3/Round-18A 递进（§5.3/5.5/5.8）；**Round-20 完成：M1 无净胜、M2 崩 ⇒ 预注册触发——停 LD 架构精修、转数据侧（§5.10）**；请求 round-21 立项数据侧（最小语义特征+重采+M3）；RL 冻结。
> **沿革**：round-2 收拢 → P0 审计 + 排摸（§2.1/2.2）→ P1 计划 round-3–8 锁板（§2.4）→ P1 执行（§2.3）→ round-9/10 P1-v2 定板与共识（§3）→ round-12/13/14 讨论（§4.1）→ **Round-12A 诊断与关闭（§4）** → round-15/16 讨论（§5.1–5.2）→ Round-15A 全批次完成（§5.3）→ round-17：A3 执行/判负（§5.4–5.5）→ round-18/19：probe-first + LD 拓扑审计锁定（§5.6–5.7）→ Round-18A 完成（§5.8）→ round-20：M0/M1/M2 消融 + 因果探针（§5.9）→ **Round-20 结果：停 LD 架构、转数据侧（§5.10）**。
> **本版说明**：第六轮精简（342 → 261 行）：§3/§4 执行细节与 §5.3/§5.5/§5.9 压缩、§0/§6/§7 刷新；改前 342 行 = `git show e638b76:docs/deepseekv4p1_argue.md`。历版全文见 §7。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **主线结果（P1-v2 → Round-20 收官）**：

| 步 | 结果 | 判定 |
|---|---|---|
| Step 1 可辨识性探针 | 正式 Gate 过（F1 0.934）；公平 progress 匹配 **0.682/0.731**（`no_increment`）；F2m 自车特征 **0.804/0.843** | 条件重开（特征级） |
| Step 2 recovery_v2 采集 | 2,693 行；**六硬门全 PASS**；v1 根因数值实证（k≥1 vs Pass-1 中位差 33.2） | PASS → Step 3 |
| Step 3 gated residual pilot | clean150 gated **0.3067 vs 基线 0.3400（−3.33pp）**；其余门全过 | **FAIL（停止）** |
| **Round-12A 诊断** | 15 配置全未过九条件；**gate 无选择性**；后端 6/40 集翻转 | **NO-GO → 主线关闭** |
| **Round-15A 零训练批次** | A1 共变量偏移（6.4×/16.5×）；A2 硬上限未触发、repeat 最优 **0.778**；C1 FAIL；P1-3 FAIL | A3/A4 条件满足（§5.3） |
| **Round-17 A3** | 确定性尾全线 succ **0**；同 ckpt chain **0.3533**（隔离=动作可持性）；远期链监督无有害证据；A1 复跑 learner 无改善 | **FAIL → round-18（§5.5）** |
| **Round-18A+20（LD 结构）** | 伪边输出层惰性（置换不变；M1 +2.67pp 无净胜）；几何响应仅 oracle **2–4%**；M2 set-attn **崩（0.0）** | **停 LD 架构 → 转数据侧（§5.10）** |

- **旁线**：D（dθ-only 0.728 / +2.2pp，V1=组合效应确认）；E（常态 μ 无系统性 bias、③ 升级证据弱）。均不改训练 target。
- **终局**：条件轨已关闭（gate 无选择性）；A3 判负（**动作可持性**）；Round-18A/20 穷举 LD 架构侧（伪边输出层惰性、M1 无净胜、M2 崩）⇒ **停 LD 架构、转数据侧**——待 round-21 立项（§5.10）。
- **约束**（持续有效）：plan 部署/评测协议；单 GPU；fail-fast；task-session 解耦 + 20 min 节拍；conda/python 只读；正式 run 不清理。**冻结维持**：P1-4、K-anchor、GRPO、WM/MoE、容量、recovery 扩采、几何反传；A4/可持性/数据侧重采待 round-21；禁制详见 §5.1/§5.4/§5.10。

---

## 1. 跨阶段锁定原则与验收（不再讨论）

| # | 共识（C1–C8） |
|---|---|
| C1 | 排摸收尾、**不扩扫参**；排摸结果只作"分配证据"，不作架构裁决 |
| C2 | **不预设删除 WM**；等预算对照（No-WM 为对照臂）决定去留；MoE 同理 |
| C3 | **上限审计（expert / interface ceiling）先于任何网络改造** |
| C4 | 执行口径（RL 训练 vs 最终评测）是 P0 问题；先量化、后动训练 |
| C5 | 失败四分类（plan-out / tracker-out / 判定异常 / recovery），30–50 例起步 |
| C6 | **footprint 插值校验**（不能只查 6 个离散计划点） |
| C7 | DAgger 采**真实 recovery 轨迹**；hard-mining 换**闭环风险**指标 |
| C8 | 统计口径：**3 seeds + paired**（McNemar/bootstrap）；eval500 单集 SE≈1.8pt |

- **执行口径**：**plan 为部署与评测协议**；`repeat_action` 仅作诊断（§2.1-Q2）。
- **评测纪律**：fail-fast 加载校验（missing/unexpected/shape_mismatch=0）为硬条件；跨版本（s11）只作方向参考；footprint 规格 = `arc_step` 位姿 + 0.5–1 m 插值 + 同款 drivable 判定（引擎无可靠 signed distance 时退化 `inside_ratio/first_invalid_pose`，**不伪造连续距离**）；`ld_line_margin_loss` 命名；引擎 run-to-run 随机性地板 0.2–1%（判读噪声 ±0.2–0.4pp）；正式实验记录 = manifest/CSV/decision.md，不进定期清理。
- **Gates**：
  - **G1 拆分**：G1-diagnosis ✅（主要损失层=计划/表示侧）；**G1-ceiling ❌**——尚无任何已验证链 ≥0.756，"超越路径"须由 **配速修复 / 更强教师 / RL 超教师** 三者之一建立。
  - **G2 追平**：3-seed mean ≥0.74；off-road：eval500 ≤0.098、clean500 ≤0.097（s11 目标锚；正式 paired 主比较 = vs P1 起点模型）；collision ≤ IDM（eval500 ≤0.144、clean500 ≤0.174）；双集成立。
  - **G3 超越**：mean ≥0.80、worst ≥0.77、paired bootstrap LB>0、McNemar p<0.05、collision ≤0.144/≤0.174、提升非单一路型；声明 0.82–0.85 需 5 seeds。

---

## 2. 前序结果压缩存档（P0 → P1）

### 2.1 P0 审计结论（三问三答，2026-10-10 定稿）

- **Q1 上限**：**d2=0.706 > oracle 0.608 < IDM 0.756** ⇒ 接口可承载 ≈0.71–0.76，差距在"学到的 plan/动作质量"。
- **Q2 口径差**：v8 模型在 `repeat_action` 归零；s11 plan 0.646 > repeat 0.418 ⇒ **保持 plan 协议**（A-hold 只是 RL 记账口径，不得用于部署/评测）。
- **Q3 失败分类**：off-road/压线为主（76–78%）；**>50% 带 plan-infeasibility 因子**（T_plan 中位 120/295 步，随驾驶漂移恶化）⇒ **P1 = 计划侧**。
- 关键行（succ/off/coll）：A IDM **0.756**/0.068/0.144；B oracle 0.608/0.134/0.058；C laneplan 0.442/0.232/0.192；**D2 0.706**/0.038/0.020；v8 模型行 0.22–0.33。全表见 git 历史 §5.1 + `runs/forensics/`。
- **timeout 分析**：d2 的 timeout rc 中位 0.849（55% ≥0.8）vs IDM 0.597 ⇒ 执行链"跑得慢"，**d2=0.706 可能低估接口上限**。
- **防线记录**：fail-fast 两次拦截静默错评（stg3 `spatial` 漏传 → `08582f2`；laneplan 参考构造 bug → `5e6e1b3`，0.002→0.442）。

### 2.2 排摸收官（容量分配证据）

- 单变量赢家：**pri512**（eval500 0.332；clean500 0.334@e020）、**rou64**（clean500 0.370@e015）；attn2 温和正向；组合臂 C-N1（pri512+rou64）**负交互被拒**；最优 epoch 依配置（base 峰 e005、pri512/rou64 峰 e015–e020）。仅作分配证据（C1）。

### 2.3 P1 前序结果要点

- **P1-0**：起点 **pri512@e020**（rou64 +1.40pp、p=0.52 不显著→"不稳定"取 pri512）；配速 25 行：V3（纯速）平坦 <0.75；**V1（ds 拉伸）0.762/0.776/0.778 @1.20/1.25/1.30**（timeout 0.236→0.086；vs s100 配对 z=4.10；vs IDM s120–135 打平）；等价口径 PASS（flags 全等 + |Δrc|≤1e-3；499/500 逐位一致）。
- **P1-1**：17,256 可用行/1,493 eps，数据门 13/13；**勘误**：k≥1 行**反事实错位 9,377 行**（obs=Pass-1 学生态 × 标签=Pass-2 专家帧）；`takeover_only` 过滤后对齐可用 2,317 行。
- **P1-2（四轴全败，诊断闭合）**：混采 1–2 epoch 崩（succ→0.01–0.14，离线 val 不可见）；仅配方（A2）0.300/0.307 稳。两层根因：①反事实错位（已修）；②**绝对目标恢复监督 × 失败态/正常低速态在 obs 不可分 × 高倍过采样**（受控探针：偏移集中 v∈[2,4)、Δds +0.44~0.57、~40×/epoch）。详见 `docs/p1_2_diagnosis.md`。
- **P1-3**：三损失实现就绪（默认关、既有路径逐位不变）；**cos(grad_traj, grad_geometry)=−0.73**；曲率阈值须按分布校准；**待几何 FP 门**（未训练）。

### 2.4 P1 执行手册 v1 锁定要点（存档）

- **范围**：P1-0A/B/C → P1-1 → P1-2 → P1-3；条件 **P1-4A**（启动条件 = 监督模型 eval500 ≥0.60 + probes 全 PASS + sampled action 贯通 plan/tracker reference；默认 base = P1-3 胜者，否则 P1-2 胜者）。
- **不承诺**：P1-4B / K-anchor 重启 / GRPO / Best-of-G / WM-MoE 对照 / 容量扫描。
- **手册 3 个开放实验问题**：① V3 配速后是否 ≥0.80（已答：<0.75）；② rou64@e015 vs pri512@e020（已答：不显著取 pri512）；③ recovery+几何能否把 ~0.37 推到 ≥0.60（**未达成**：P1-2 崩、P1-v2 pilot 未过门）。
- 细则（数据门终值 10k/500ep/80%、产物 11 项、canary 两段式挂 4B 前置等）原文在 git 历史 §7.10；契约落点 `docs/p1_implementation_contract.md`、`docs/p1_probe_spec.md`。

---

## 3. P1-v2 执行纪要（Step 1–3；压缩存档）

- **锁板**：可辨识性 probe → recovery_v2 同状态采集 → gated residual pilot → 几何 FP 门 → 独立小消融 → candidate recipe；intervention 标签 = 同状态"baseline 将失败 ∧ expert 可恢复"、行级同刻同态契约；每步 Go/No-Go、不自动晋级；几何 proxy 未过 FP 门不得反传。无条件轨降级为诊断旁线（不改训练 target；dθ-only 反事实 + bias ③ 映射既有 calibration）。**冻结**：recovery_v1 训练、P1-4、K-anchor、GRPO、WM/MoE、容量。

**Step 1 可辨识性探针（1a→1b→1c）**：1a 正式 Gate 过（F1 ROC-AUC 0.934；tollgate 0.987、merge 0.965）但 **time-only 0.940 ≥ F1** ⇒ conditional_pass；**1b 决定性对照**（progress 匹配、时间不参与；Δrc 中位 0.0054）F1 **0.682/0.731** ⇒ 预注册判据 **`no_increment`**（1a 系进度/时间混淆）；**1c 特征重测**：**F2m（自车可算：lane_lat + plan footprint·inside_ratio + 历史；无锚点 oracle 列）= 0.804/0.843 ≥ 0.80 ⇒ 路径重开**（F2/F3=0.996 属定义性 oracle 循环，不作证据）。→ **Step 2 GO**；gate 禁锚点 oracle 列。

**Step 2 recovery_v2 同状态采集（PASS）**：**2,693 行/110 eps**（labeled intervention 836 = correction 660/prevention 400/late 232；expert recovery 89.2%）；**六硬门全 PASS**（同状态 100%〔max_plan_diff 6.7e-06〕、确定性重放 1,216 行含 471 k≥1 全 0 失配、split 88/22 无泄漏、action-chain 一致 0 失败、control 覆盖比 1.08）；**v1 根因正面实证**：k≥1 行存 Pass-2 同状态 obs（vs Pass-1 中位差 **33.2**）。probe_v2_check：F2m 0.87–0.92、净增量 +0.33~+0.39。注记：mode 邻步切换率 0.611（仅分层，**不得作训练目标**）；难度偏 easy；CPU/CUDA 结局分歧（标签自洽于 CPU）。

**Step 3 gated residual pilot（FAIL · 按预注册停止）**：架构 `final = base + gate(obs, base_plan) × residual`；base=pri512@e020 冻结；gate=0 逐位==base（单测 + 闭环复验）；离线全过（正常行激活 3.99%<5%、recall 61.1%≥60%、gate AUC 0.870）。**clean150 唯一失败项**：gate-off 逐位复现 0.3400；gated **0.3067（−3.33pp）**；配对 13 成功→失败 vs 8 失败→成功（净 −5）；timeout 23→30。机制：闭环 gate 激活率 **16.1% 步 vs 离线 3.99%** ⇒ **部署分布迁移下过触发**（根因：F2m footprint 特征需活 env ⇒ gate=0 监督只能取 v2 control）。零生产改动；`gated_residual_report.json` verdict=FAIL。

**诊断旁线（不涉训练 target）**：**E** 常态 μ 无系统性 bias（μ_ds −0.0015 t=−0.71、μ_dθ −0.00011；唯一系统性=尾链 dθ −0.00135 但 spread 17×）；大偏差全在 w=0 过滤行；失败态对照偏差巨大（状态条件性=recovery 监督域）。**D** dθ-only dth0.85 succ **0.728**（+2.2pp）、steer 饱和 0.132→**0.043** ⇒ **V1（+7pp）= "更长步+更平曲率"组合效应**（维持 calibration oracle 定位）。Δκ/progress 监控未实施（归 P1-3 项）。

---

## 4. Round-12A 零训练诊断与主线关闭（2026-10-11 晚；压缩存档）

**授权链（round-12→14）**：round-12（`017e425`）先零训练复盘（gate 误报/残差幅度/时序抖动三机制分解）、同架构仅一训；round-13（`cb34622`）补三约束（CPU/CUDA 全链一致性、common-prefix 首分歧分解、episode 级破坏性误触发 <5%）、扫描补 2×2×2 交叉；round-14（`ea29866`）**12A LOCKED**、统一 CUDA 口径、九条件（含"非孤立有效区域"）、**允许"无有效区域"=判负关闭**。我方并入：闭环重跑+逐步遥测（旧产物无逐步字段）、几何余量改善预注册口径、"12B 同状态双 rollout"、"残差有效"表述纠偏。

**执行（零训练）**：Stage1 逐 step 遥测（gate_prob/residual/base·final plan/tracker_ref/rc/speed/pose + `--residual-scale`/`--debounce`；产物 trace.npz+JSONL）；Stage2 逐 episode 复现（gate-off 0.3400/gated 0.3067；13 翻负/8 翻正集合一致）；Stage3 CPU/CUDA 全链探针（2,538 状态：gate 概率 Δp95 2.5e-4、各阈值跨界率 ≤0.12%；**e2e 40 eps 6 集翻转**、材料分离中位 40 env steps）；Stage4 **15 配置扫描**（threshold×5→scale×4→debounce×3 + 2×2×2 交叉，单轮 ~3.7 min）。

**终判（九条件与三问）**：15 配置全未过——① "保留 ≥6/8 历史翻正"全网格无人达标（最好 5/8）；② CPU/CUDA 无实质分叉全 False（6/40 集翻转）。三问：① **gate 误报且无选择性**（闭环激活 16.1% vs 离线 3.99%；hist_8 17.07% ≈ hist_13 15.87%；保守配置下反选择）；② 残差 1.0 偏强（"救余量、伤进度"：timeout 23→30、均速 3.39→3.27）；③ 连续触发 + 后端离散化放大（激活段均值 4.43 步、最长 55 连续步；2,336 个分叉后激活）。正面事实（记录不改判）：t050_s050_d2 aggregate 反超（+1.33pp）但机制不可信（门无选择性、单 seed）。全表见 `249c51c` 版 §4。

**口径教训（随档）**：① "正常态激活 <5%"须定义**闭环**度量（3.99%→16.1% 不可迁移）；② destructive-FP 口径结构性近空（复活条件轨须改"post-divergence 净变化"）；③ common-prefix 仅适合同设备对照；④ 条件间张力：翻正集由激进配置定义 ⇒ 保守配置天然丢失；⑤ **跨后端 ~15% 集级差 ⇒ 条件监督/采集/校准统一 CUDA 口径**；⑥ 工程坑：npz 重复解压 SIGKILL、footprint 判定需活 env、P@10 须带 n_pos/n_neg、mode 邻步切换 0.611；⑦ gate **选择性**才是瓶颈（失败态模型-专家偏离巨大）。

**判定**：**NO-GO（无有效区域）→ 按预注册关闭 gated residual 主线**——不补 hard negatives、不扩 gate、不调采样；产物保留 `runs/p1_v2/round12a/`；零生产代码改动 → 转 round-15（§5）。

---

## 5. round-15→20 执行批次（15A/A3/18A/20；结果导向）

### 5.1 授权链摘要（round-15/16；已执行）

- **round-15（`b2f96de`）**：同意关闭 gated residual；**反对 RL 优先**（0.33 级上 RL 会优化错误流形）；**根因重估："教师行为与训练目标不等价"**——系统靠"观察→重规划→修正"（D2 0.706），训练却拟合 6 步/3s 远期链（D1 开环≈0、模型 0.33）；三线：**A 基础监督重构**（门 clean150 ≥0.50 → eval500 → 0.60–0.65 才谈 RL）/ **B** V1 机制分解（仅稳定 signed bias 才启 bias_calib）/ **C** IDM+PP 安全残差（仅设计 + 离线 oracle；≥0.82 门）；CPU/CUDA 定性=评测方差/复现风险（统一 CUDA 口径）。
- **round-16（`24bfaba`）**：接受我方全部澄清（D1≈0=开环；D2 差距三成分；DAgger 归 Track A）；**批准 A1/A2 零训练 + B/C 只读 + P1-3 只读 + CUDA 口径统一**；**A3/A4 条件批准**；禁止清单（gated residual 复活 / recovery 扩采 / tail decoder / PPO·GRPO / K-anchor / WM·MoE / 容量 / 几何反传）。
- 我方细化并入执行：先编配速旧账、家族含"动作变换"轴、Go 条件联合解读（硬上限仅由 succ 明显低于 IDM 触发）。
- 预注册门：**DAgger 触发 = teacher 低 ∧ learner 高**；**A2 Go = succ ≥0.72 ∧ off ≤0.08 ∧ coll ≤IDM (0.144)**。

### 5.2 round-15/16 补充存档（Track 定义 / RL 门 / 我方要点；均已执行）

- **Track A（已执行为 A3）**：obs/history → 当前 expert action → 固定确定性尾 → arc_step → LQR → 0.5s 重规划；训练 `L=L_cur+λ_smooth·L_smooth+λ_curv·L_Δcurv`（禁未来链监督项）；门 clean150 ≥0.50 → 完整 eval500 → 0.60–0.65 才谈 RL。
- **Track B**：ds/dθ/curvature 分层 signed bias（仅跨路型稳定才启 `phase3_bias_calibration`）；**Track C**：IDM+PP 安全残差（仅设计 + 离线 oracle；≥0.82 门）；**P1-3**：FP 门只读（FAIL 后 `ld_margin_weight=0`）。
- **RL 启动门（6 条）**：监督/混合基线 eval500 ≥0.60–0.65、当前动作确实决定执行、训练/评测同 PlanHead+LQR 口径、clipping<5%、LQR 饱和可控、value 能识风险。
- **我方要点（存档）**：接受 + 认账（RL 排序：先修监督、RL 后置）；三条澄清（D1≈0=开环 / D2 差距三成分〔特权判断·参考尾形状·learner 偏移〕/ Track C 战略前提）；四条建议（A1/A2 先出数、尾族 oracle 预筛、落 phase3 `chain_step_weights` 机制、DAgger 归 Track A）；Go 条件联合解读（硬上限仅由 succ 明显低于 IDM 触发）。

### 5.3 Round-15A 结果（全批次；2026-10-12 00:51；压缩存档）

**A1 · 当前动作分层审计**：teacher-forced（26,767 行）MAE_ds **0.064 m**/MAE_dθ **0.0060 rad**、sign 错 3.1%；learner（836）**0.406 m/0.0986 rad（6.4×/16.5×）**、sign 错 **21.7%** ⇒ 专家态准、自身态显著退化=**共变量偏移**（链/当前比 2.79）。Track B：唯一跨路型稳定 signed bias = **learner 态 ds（−0.33…−0.51 m/step，4/4 路型 |t|≥3）**；teacher 侧无 ⇒ bias_calib 若启只对 ds+learner 态。

**A2 · 确定性尾族 oracle（10 新 + 22 旧）**

| 族 | 最佳 | 结论 |
|---|---|---|
| repeat（旧账） | **V1 s130 = 0.778**（s125 0.776） | V1 ds×s 0.728 ≫ V2 0.676 ≈ V3 0.69 ⇒ 增益 = **"拉伸 ds 使参考曲率相对速度下降"（几何×速度耦合）** |
| decay 新族 | 0.726（仅 1 组过 Go） | 全不敌 repeat；**geo-decay 灾难**（0.000–0.042）；V1×decay 交叉 0.282 亦毁 |
| 复算 | v1/v2_s110 **逐位一致** | 因果核验通过 |

- **q1 天花板**：正确当前动作+简单尾 = **0.72–0.78（≈IDM 上方）**、硬上限未触发；**q2** 饱和最低 dth_s085（0.043）；**q4** 执行参考**无需学习六步链**即可达该上限带。

**Track C / P1-3**：C1 union **0.790 < 0.82 FAIL（不实现）**（救援 17/122；"union≠严格上界"已注记）、C2 动态净增量≈0；**P1-3 FP 门 FAIL（5 项）** ⇒ `ld_margin_weight` 维持 0、几何反传维持禁止（根因=LD 通道语义，需改损失定义）。

**综合（去向 round-17）**：① 共变量偏移已证（A1）；② 执行参考无需远期链、**V1 式 ds 拉伸是有效自由度**（A2，已证）；③ "远期链=历史主损"高度可信但未证——**A3 同预算对照为裁决工具**。A3/A4 放行条件满足；**A3 设计**=当前动作 BC+repeat 尾（无 encoder/WM/MoE 改动、门 clean150 ≥0.50→eval500→0.60–0.65 才谈 RL）；**A4**=learner 态 DAgger（10–20% 混合、保 expert5k）。**天花板提示**：特权动作+最佳尾 ≈0.778 ≈ IDM+2pp——G3 仍须超越教师（RL 后置）或更优执行。

### 5.4 round-17 往来残档（`732ba96`；已并入执行）

- 无异议、开工：4 条落法（训练/尾解耦〔禁未来链监督〕、隔离〔单 head/纯函数/LQR 逐位〕、对照臂〔同预算六步链监督〕、A4 契约〔learner 态采集 90:10–80:20〕）+ 验收全部并入执行——**结果见 §5.5**。

### 5.5 Round-17 A3 结果（2026-10-12 08:49 报告）：**FAIL（决定性负结果）**

**结果表（clean150；基线 pri512 0.3400/0.4933/0.0333）**

| 臂 / 执行 | succ | off | coll | tmo | steer_sat |
|---|---|---|---|---|---|
| A3 臂（当前动作监督）× R / V115 / V125 | **0.000** | **1.000** | — | 0 | 0.68–0.70 |
| A3 臂 × chain（同 ckpt 隔离） | **0.3533** | 0.4400 | 0.0200 | 0.187 | 0.768 |
| 对照臂（六步链监督）× R / V115 / V125 | **0.000** | **1.000** | — | 0 | 0.66–0.68 |
| 对照臂 × chain（原生） | 0.3267 | 0.5533 | 0.0333 | 0.093 | 0.796 |
| oracle 动作 × R / V115 / V125 | **0.687 / 0.747 / 0.780** | 0.02–0.047 | — | — | 0.09–0.14 |

- **门**（A3 最优尾 R）：succ ✗ / off ✗ / learner-误差 ✗（ds **−6.0% 恶化**）/ steer-饱和 ✗（5.1–7.5× oracle）；coll / tmo / clipping / teacher-MAE / plan-逐位 ✓ ⇒ **判负；A4 按锁定不启动**（仅记录契约）。
- **同 ckpt 隔离（关键）**：A3 ckpt——chain 模式 **0.3533** vs 确定性尾 **0.0000** ⇒ 瓶颈 = **动作可持性（holdability）**，非尾形、非监督配方。
- **对照裁决**：A3+chain 0.3533 ≈ 对照+chain 0.3267 ≈ 基线 0.3400 ⇒ **远期六步链监督"无害亦无明显收益"**——round-15 的"远期链=历史主损"假说**未获实证**（方向受挫）。
- **A1 复跑**：A3 ckpt teacher ds 0.0553（↓13%）但 learner 态无改善（ds 0.4308，**+6%**）⇒ 共变量偏移缺口**未被"专家数据上的当前动作 BC"触及**（强化 DAgger 结论）。
- **机制（trace）**：模型 mu 的 dθ 在直道态即抖 ±0.09–0.23 rad（≈ learner 态 dθ MAE 0.097）；repeat 3s → LQR 预瞄成紧弯 → steer 饱和 0.69（oracle 0.14）→ 10–20 决策步内甩出。**学习链在 chain 模式下充当"执行噪声缓冲"**（尾部衰减/平滑使执行可行）——原设计的学习尾在执行稳健性上有实质功能。
- **工具可信度**：等价核验 **100/100 flags/终止全等 + |Δrc|=5e-5 + plan0/ref 失配 0**；oracle 尾族 0.687/0.747/0.780 与 A2 eval500 旧账一致 ⇒ **负结果非工具伪影**。
- **新认识与去向**：① **动作可持性是被忽视的独立属性**（专家天生 dθ≈0；模型不具备）；② 六步链或主要承担**执行噪声缓冲**；③ learner 态数据缺口仍是第一瓶颈（A1 复跑确认）。round-18 候选 a–d（可持性约束重跑 / 重议 A4 / 接受 chain 缓冲 / obs 改造）→ 由 round-18 裁决 probe-first（§5.6）；生产改动提交 `f8169fd`（默认关 + 7 单测）。

### 5.6 round-18 往来残档（`95f23b0`；已并入执行）

- 接受 probe-first：两层根因框架（primary 动力学 shortcut + learner 态放大）、lanekeep 归 primary、chain 保留为执行缓冲并重定义——全部接受。
- 2 处事实核查：LD 7 维**已隐式含** lane-relative 几何（缺的是关联/选择/监督）；`ld_hist` 是输入（"只作输入"≠不进 memory）。
- 4 条落法与 3 验收补丁——**已并入 §5.8 执行与结果**。

### 5.7 round-19 往来残档（`1ea85dc`；已并入执行）

- **七项代码论断逐条核验=全部成立**（file:line 摘要）：`model.py:505-507`（PlanHead 只看 ld_pool 均值）、`spatial.py:103-106`（LD-LD=槽位链 ⇒ 伪边）、`mem.py:699-705`（LD 仅当前帧）、`model.py:627-632`（K-anchor lane_ctx=0）、`ld.py:209-260/137-170`（环优先排序 / ego.lane 优先）、注释漂移——**量化结果见 §5.8**。
- 接受 0A/0B/0C 前置与全部修正（全语义镜像 / 输出变化非充分 / Probe B 分层 / primary_only 严格旁路）；4 条实施路线（诊断重放 / 一致镜像近似 / 位级纪律 / 同帧配对）——**已执行（§5.8）**。

### 5.8 Round-18A 结果（2026-10-12 19:2x）：**拓扑/输入结构性缺陷 + primary 几何响应 ~50× 过弱（动力学主导）**〔round-20 复核见 §5.10〕

**0A·槽位/拓扑（186 eps / 14,461 决策步）**：相邻槽同 lane 仅 **23.5%** ⇒ 槽位链 **伪边率 76.5%**（215,067 边；hard 0.766 vs other 0.762）；跨帧同槽身份保持 91.8%（抖动 8.2%）；`ego.lane≠nav current[0]` 99.3%（定义注记：nav 参考按 lane id 排序、非 ego 专属）。

**0B·target-lane 覆盖**：ego lane 在场 **95.07%**（缺失 4.93%，hard 5.07%）；nav current/next 95.05/97.07%；槽近恒满（15.87/16）⇒ 缺失时槽位被其他车道填满（无降级信号）；曲率非零仅 21.7%；最近点 5.0m。

**0C·lane-assignment 断点**：**失败集 0.5146 vs 成功集 0.0363（14×）**；失败前 20 决策步内断点 **127/127**；断点处 mu_dθ 3.1×、LD jump 8.5×。注：非因果（失败簇集于拓扑转换处，需与成功集同位对比才可作因果论——本轮给比值）。

**Part 1·四 probe（n=500；双 ckpt×双模式）**：
| 反事实 | Δmu_dθ | latent 相对 | 判读 |
|---|---|---|---|
| zero_ld | **0.0153** | 0.32 | LD 必要性 ✓ |
| straighten | 0.0005 | 0.007 | 极弱 |
| **translate ±1m / rotate ±10°** | **±0.002 / ±0.0026（≈0.1–0.15°）** | 0.02–0.03 | **符号正确、单调，但比合理量级小 ~50×** |
| remove_prev_action | 0.0024 | 0.139 | 弱 |
| **zero_history** | **0.0134（几何响应 5–7×）** | 0.71 | **转向由动力学/历史主导** |
| 镜像（tensor 级） | 等变率 0–0.2% | — | 不闭合（需 env 级+OD 身份定论） |
| primary_only vs full | ≈同（0.0196 vs 0.0153） | — | experts 关不降级 ✓ |
- B 解码：curvature 可解（R² 0.52/0.58）✓；lat/heading/边界距 R² 负（跨 episode 标签方差主导，需 episode 内归一化/分类化重做）。

**分支判定（report.json）**：`input_topology`=缺陷（伪边 76.5%+缺 lane 4.9%+抖动 8.2%）⇒ **最小语义特征路线**（fingerprint 变更 ⇒ 需重采）；`primary_response`=弱 ⇒ 同时支持**显式池化（target/adjacent/boundary）**设计。

**我方综合（供 round-20）**：
- **修复阶梯（先模型侧、后数据侧）**：**a. 模型侧（无重采）**：关闭/按语义重建 LD-LD 邻接（`spatial.py:103-106`）+ 显式池化替换全槽均值（`model.py:505-507`）+（可选）几何响应辅助监督——数据可复用、只需重训；**b. 数据侧（需重采）**：最小语义特征（target_lane flag/lane_id/offset idx/route assoc/侧偏/航向差/曲率）——fingerprint 变更、全量重采。0B 显覆盖非主缺口 ⇒ 先 a 后 b（执行见 §5.9/§5.10：**a 已穷举关闭，b 为现行路线**）。
- **坑（必带）**：① **模型消费 LD 走 `ld_hist[:, -1]`**——任何 LD 干预必须同时改该通道（否则假 0）；② 镜像不闭合=现象、结论待 env 级；③ 网格符号度量为口径 bug（真实符号由反事实均值证正确）；④ B 解码仅 curvature 可用；⑤ `a3/primary_only` 组合缺（3/4，模式同 pri512/primary_only）；⑥ 0C 非因果。
- **纪律**：零训练 ✓；net/pipeline 零改动 ✓（primary_only 复用 `set_moe(enabled=False)`）；`ld.py` 仅默认关旁路（fingerprint 不变、2 单测过）✓；已提交（见 git log）。

### 5.9 round-20 往来残档（`37ee932`；已并入执行）

- 裁决=**接受三项收紧**（伪边=架构缺陷证据非闭环因果、几何响应改相对口径、zero_history 强 OOD 需温和对照）+ 批准 M0/M1/M2 等预算消融 + 20A/20B 因果探针 + 验收门（**M2<0.45 → 转数据侧**）；我方 3 条操作性补充（oracle 对照基准、LD 干预六帧一致、M2 绕过 spatial MP）——**已执行，结果见 §5.10**。

### 5.10 Round-20 结果（2026-10-12 20:3x）：**M2 触发预注册停止——停 LD 架构精修、转数据侧**

- **20A 置换探针（关键否定）**：模型对 LD 槽序置换**本就近似不变**（|Δmu_dθ|≈2–5e-06、inv@1e-4=1.0，四组合）⇒ **76.5% 伪边在输出层"惰性"**——伪边假设在输出层不成立；M1（关链）无下降空间。
- **20B matched-case（125 对）**：失败 switch rate **0.511 vs 成功 0.037（14×）**；但**切换处跳变幅度不判别**（0.0058 vs 0.0068）；判别量=**全程动作基噪声（0.00266 vs 0.00052，5×）**；断点=共现/暴露因子、非唯一领先指标（断点→终点 21.0 步）。
- **三臂 clean150（同批 M0）**：

| 臂 | succ | off | coll | tmo | steer_sat |
|---|---|---|---|---|---|
| M0 | 0.3267 | 0.5533 | 0.0333 | 0.0933 | 0.796 |
| **M1（no_chain）** | **0.3533（+2.67pp）** | 0.5200（−3.33pp） | **0.0067（−2.67pp）** | 0.1200（+2.67pp） | 0.786 |
| **M2（set_attn）** | **0.0000（−32.67pp）** | 0.9933 | 0.0133 | 0.0067 | 0.764 |

- 门：**M1 不过门（+2.67pp<+5pp、−3.33pp>−10pp）但无害且 coll 显著降**；**M2 崩（0.0）** ⇒ **M2<0.45 → 停 LD 架构精修、转数据侧（预注册触发）**；eval500 按规则跳过。
- **结构门（n=200）**：置换敏感度**基线已触底（无下降空间）**；**几何响应仅 oracle 的 2–4%（≈50× 过弱再确认）**；方向正确率各臂 ~90%（M2 未改善）——三项均未过。
- **坑（必带）**：① M2 载入必须"先 `set_ld_mode` 再 load"（fail-fast 首跑踩中；修后得真实崩塌数）；② oracle=解析近似（env 级 expert 重放仍缺口；静止态限定 v≥2）；③ **离线 loss 不预示闭环保崩**（M2 val 仅差 ~9%、闭环 0→崩——离线不可作架构选择依据）；④ 终版结构探针未含 A3（仅 20 样冒烟）；⑤ 峰值 RSS 未埋点（缺口）。
- **综合（Round-18A+20 收敛）**：瓶颈=**几何响应量级（~50× 过弱）+ 动作基噪声（5×）**——均指向**数据/监督侧**而非 LD 接线；架构侧 LD 修复已被穷举（伪边惰性、M1 无净胜、M2 反而崩）。
- **纪律/提交**：默认关+惰性构造（既有 ckpt 键集不变）✓；5 新测试 + 组合 **99 passed** ✓；M3/数据侧重采/A4 均未启动；生产改动（`net/spatial.py`/`net/model.py`/`pipeline/stages.py` + 测试）提交见 git log。
- **下一步（供 round-21）**：数据侧路线立项——**最小语义特征**（target_lane flag/lane_id/offset idx/route assoc/侧偏/航向差/曲率；obs fingerprint 变更 ⇒ **需重采**）+（M3）几何响应辅助监督（标签须先过可靠性）→ 重训 → 结构门复测 → 闭环保。**请求 round-21**：确认转数据侧范围与重采规模；M1 小正向（coll 降）是否纳入后续配方；环境级 expert 对照列入数据侧计划。

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`（P1-v2 决策 + Round-12A 结果）、`docs/recovery_v2_contract.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`
- **报告**：`docs/p1_0_report.md`（P1-0 FINAL）、`docs/p1_2_diagnosis.md`（P1-2 FINAL）、`docs/recovery_v1_dataset_card.md`
- **数据**：`datasets/recovery_v1/`、`datasets/recovery_v2/`（samples / manifest / split / data_gate_report / _verify）
- **产物**：`runs/p1_v2/round12a/`（trace/探针/15 配置/归因/九条件）、`runs/p1_v2/round15a/`（A1 审计 / A2 尾族 oracle / Track C / P1-3 FP 门）、`runs/p1_v2/round17/`（两臂 ckpt / 三尾 clean150 / a3_report）、`runs/p1_v2/round18a/`（probes/审计/断点）、`runs/p1_v2/round20/`（20A 置换 / 20B matched-case / M0-M2 / 结构门）、`runs/p1_v2/`（intervention_probe*、probe_robustness、probe_v2_check、bias_diagnostic、gated_residual_report、gated_residual_pilot/）、`runs/p1_0/`（d2_pacing_matrix、dth_sweep_summary、start_model_decision）、`runs/p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`（全链流水）、`.slim/deepwork/p1_v2/`（probe / collector / gated_residual / round12a / round18a / round20 脚本 + logs）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮）
- **关键提交**：`37000ae`/`a371cc5`（过滤）、`e953f10`（P1-3）、`c20fa00`（P1-2 诊断）、`28acbcd`（dθ-only）、`1800e69`/`a42ac79`/`27ced1f`/`7b83730`（P1-v2 步）、`061bbac`/`663fedc`（argue §7/§8）、`5962848`（12A 开工）、`249c51c`（12A NO-GO）、`3b5124f`（尾族工具）、`dab85f8`（§5.3 批次结果）、`74f522f`（第三轮精简，位于 `Planner_RL_copy`）、`d70ec1b`（§5.4）、`f8169fd`（A3 配方开关）、`96df7cb`+`c71f389`（Round-18A）、`da83ce1`（LD 消融开关）、`e638b76`（§5.10）

---

## 7. 存档说明（历次精简 / 去哪找）

- **第六轮精简（本版；342 → 261 行）**：§3/§4 执行细节压缩，§5.3/§5.5/§5.9 收缩，§0 补 Round-18A+20 行，§6/§7 刷新；改前版本 = `git show e638b76:docs/deepseekv4p1_argue.md`（342 行版）。
- **第五轮精简**：§5.4/5.6/5.7 往来压为残档；改前 355 行 = `git show 96df7cb:docs/deepseekv4p1_argue.md`。
- **第四轮精简（299 → 282 行）**：§5.2–5.4 往来压为补充存档、§5 编号顺延（5.5/5.6/5.7→5.3/5.4/5.5，后经轮次自然演化为现编号）；改前 299 行 = `git show 58ac4c3:docs/deepseekv4p1_argue.md`。
- **第三轮精简（原提交 `74f522f`，位于 `Planner_RL_copy`；已合入本分支）**：round-15/16 往来压缩为 §5.1 授权摘要、§5.2 补充存档；§0/§6/§7 刷新。
- **第二轮精简（262 → ~210 行）移除**：round-12/13/14 往来全文（原 §7/§8）、round-12 请求（原 §4）、12A 原 §9（并入 §4）；全文见 `git show 249c51c:docs/deepseekv4p1_argue.md`（262 行版）。
- **第一轮精简（1114 → 166 行）移除**：§0.9 收拢细节、§1–§4 早期草案、§5/§6 详表、§7.1–7.11 回合全文、§9–§11 原文、附录 A/B；全文见 `git show 7b83730:docs/deepseekv4p1_argue.md`（1114 行版）。
- **压缩时间线**：
  - 2026-10-09：round-1/2 收拢；《评测兼容设计》落 `docs/p0_eval_compat_design.md`；
  - 2026-10-09/10：P0 三件套 → P0-B 矩阵 + P0-C 归因 → 审计定稿（§2.1）；
  - 2026-10-10：排摸收官（§2.2）；round-3–8 与 P1 锁板（§2.4）；
  - 2026-10-10→11：P1 执行——P1-0/1/2/3（§2.3）；
  - 2026-10-11：round-9/10 定板（§3.1）→ P1-v2 三步（§3.2–3.4）+ 旁线 D/E（§3.5）；
  - 2026-10-11 晚：round-12/13/14 讨论（§4.1）→ **Round-12A 执行与 NO-GO（§4.2–4.5）**；
  - 2026-10-11 深夜–10-12：round-15/16/17 讨论与执行（§5.1–5.7）：Round-15A 全批次 → **A3 执行/判负 → round-18**；
  - 2026-10-12：round-18/19/20（§5.6–5.10）：Round-18A probe+审计 → round-20 M0-M2 消融+因果探针 → **停 LD 架构、转数据侧（round-21 输入）**。
