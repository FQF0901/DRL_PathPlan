# deepseek-v4.1 argue / 主文档（R21-3 完成 · 首现强正向 · round-24 待裁）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-13 凌晨）**：P1-v2 条件轨已关闭；Round-15A/A3/18A/20 递进（§5.3/5.5/5.8/5.10）；round-21 复核（§5.11）；round-22 Outcome D（§5.13）；R21 0..3 递进（§5.14–5.16）；**round-24 批准 R21-3D（代码语义一致性 + 独立集复测 + 小规模闭环 canary）——已开工；正式采集条件后置（§5.17）**；RL 冻结。
> **沿革**：round-2 收拢 → P0 审计 + 排摸（§2.1/2.2）→ P1 计划 round-3–8 锁板（§2.4）→ P1 执行（§2.3）→ round-9/10 P1-v2 定板与共识（§3）→ round-12/13/14 讨论（§4.1）→ **Round-12A 诊断与关闭（§4）** → round-15/16 讨论（§5.1–5.2）→ Round-15A 全批次完成（§5.3）→ round-17：A3 执行/判负（§5.4–5.5）→ round-18/19：probe-first + LD 拓扑审计锁定（§5.6–5.7）→ Round-18A 完成（§5.8）→ round-20：M0/M1/M2 消融 + 因果探针（§5.9）→ Round-20 结果（§5.10）→ round-21 复核（§5.11）→ round-22 四臂 Outcome D（§5.13）→ R21-0..2 门全过（§5.14）→ round-23：R21-3A..3C 批准开工（§5.15）→ 3A/3B 门过、3C 表示 3/3+action 4/5（§5.16）→ **round-24：批准 R21-3D（§5.17）**。
> **本版说明**：第八轮精简（300 → ~270 行）：§5.15/5.13/5.10/5.8/5.1–5.2 收缩、§0 补 R21-3 行、§6 更新；**当前状态=R21-3A..3C 首现强正向（§5.16）、round-24 待裁**；改前 300 行 = `git show 11f0000:docs/deepseekv4p1_argue.md`。历版全文见 §7。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **主线结果（P1-v2 → R21-3）**：

| 步 | 结果 | 判定 |
|---|---|---|
| Step 1 可辨识性探针 | 正式 Gate 过（F1 0.934）；公平 progress 匹配 **0.682/0.731**（`no_increment`）；F2m 自车特征 **0.804/0.843** | 条件重开（特征级） |
| Step 2 recovery_v2 采集 | 2,693 行；**六硬门全 PASS**；v1 根因数值实证（k≥1 vs Pass-1 中位差 33.2） | PASS → Step 3 |
| Step 3 gated residual pilot | clean150 gated **0.3067 vs 基线 0.3400（−3.33pp）**；其余门全过 | **FAIL（停止）** |
| **Round-12A 诊断** | 15 配置全未过九条件；**gate 无选择性**；后端 6/40 集翻转 | **NO-GO → 主线关闭** |
| **Round-15A 零训练批次** | A1 共变量偏移（6.4×/16.5×）；A2 硬上限未触发、repeat 最优 **0.778**；C1 FAIL；P1-3 FAIL | A3/A4 条件满足（§5.3） |
| **Round-17 A3** | 确定性尾全线 succ **0**；同 ckpt chain **0.3533**（隔离=动作可持性）；远期链监督无有害证据；A1 复跑 learner 无改善 | **FAIL → round-18（§5.5）** |
| **Round-18A+20（LD 结构）** | 伪边输出层惰性（置换不变）；M1 +2.67pp 无净胜；几何响应仅 oracle **2–4%**；M2 实为"未接 PlanHead"的接线缺口（§5.11） | → 修正版四臂（§5.13） |
| **Round-22 + R21-0..2** | 四臂 Outcome D（α≈1.3%、近均匀、几何响应未达 5×；e002 回落）；R21 三阶段门全过（措辞经 §5.15 更正） | → R21-3 接线+复测（§5.15–5.16） |
| **R21-3A..3C** | 接线+契约 **925 passed**；canary 门过；**3C 表示 3/3 + action 4/5（sign 99%、lane 4.31×、hist/lane 0.243；曲率 20.1% 近门）** | **首现强正向 → 待 round-24 裁决正式采集/S0/S1（§5.16）** |

- **旁线**：D（dθ-only 0.728 / +2.2pp，V1=组合效应确认）；E（常态 μ 无系统性 bias、③ 升级证据弱）。均不改训练 target。
- **终局**：条件轨已关闭；A3 判负（**动作可持性**）；round-22 四臂 Outcome D ⇒ LD 架构排摸结束；**R21-0..2 全过（定位 pooling 丢失为主 + 监督不足 + history shortcut）⇒ 下一步=显式接线 + 复测（R21-3 待批，§5.14）**。
- **约束**（持续有效）：plan 部署/评测协议；单 GPU；fail-fast；task-session 解耦 + 20 min 节拍；conda/python 只读；正式 run 不清理。**冻结维持**：P1-4、K-anchor、GRPO、WM/MoE、容量、recovery 扩采、几何反传；A4=冻结（条件见 §5.12）；禁制详见 §5.1/§5.4/§5.10/§5.13。

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

### 5.1 round-15/16 授权链残档（`b2f96de`/`24bfaba`；已执行）

- **round-15**：关闭 gated residual；**反对 RL 优先**；根因重估="教师行为与训练目标不等价"（三线：A 基础监督重构〔门 clean150 ≥0.50 → eval500 → 0.60–0.65 才谈 RL〕/ B V1 机制分解 / C IDM+PP 安全残差〔≥0.82 门〕；CPU/CUDA=评测方差 ⇒ 统一 CUDA 口径）。
- **round-16**：批准 A1/A2 零训练 + B/C 只读 + P1-3 只读；A3/A4 条件批准；禁制（gated residual 复活 / recovery 扩采 / tail decoder / PPO·GRPO / K-anchor / WM·MoE / 容量 / 几何反传）。预注册门：**DAgger = teacher 低 ∧ learner 高**；**A2 Go = succ ≥0.72 ∧ off ≤0.08 ∧ coll ≤IDM**。
- **Track/RL 门（存档）**：A=当前动作 BC+确定性尾（已执行为 A3）；B=分层 signed bias（稳定才启）；C=IDM+PP（C1 FAIL 已关）；**RL 启动门 6 条**（基线 eval500 ≥0.60–0.65、动作决定执行、同 PlanHead+LQR 口径、clipping<5%、LQR 饱和可控、value 识风险）。

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
- **对照裁决**：A3+chain 0.3533 ≈ 对照+chain 0.3267 ≈ 基线 0.3400 ⇒ **远期六步链监督"无害亦无明显收益"**（round-15 假说未获实证）；等价核验 100/100 + |Δrc|=5e-5、oracle 尾族与 A2 旧账一致 ⇒ **负结果非工具伪影**。
- **A1 复跑**：A3 ckpt teacher ds 0.0553（↓13%）但 learner 态无改善（ds 0.4308，**+6%**）⇒ 共变量偏移缺口**未被"专家数据上的当前动作 BC"触及**（强化 DAgger 结论）。
- **机制（trace）**：模型 mu 的 dθ 在直道态即抖 ±0.09–0.23 rad（≈ learner 态 dθ MAE 0.097）；repeat 3s → LQR 预瞄成紧弯 → steer 饱和 0.69（oracle 0.14）→ 10–20 决策步内甩出。**学习链在 chain 模式下充当"执行噪声缓冲"**（尾部衰减/平滑使执行可行）——原设计的学习尾在执行稳健性上有实质功能。
- **新认识与去向**：① **动作可持性是被忽视的独立属性**（专家天生 dθ≈0；模型不具备）；② 六步链或主要承担**执行噪声缓冲**；③ learner 态数据缺口仍是第一瓶颈（A1 复跑确认）。round-18 候选 a–d（可持性约束重跑 / 重议 A4 / 接受 chain 缓冲 / obs 改造）→ 由 round-18 裁决 probe-first（§5.6）；生产改动提交 `f8169fd`（默认关 + 7 单测）。

### 5.6 round-18 往来残档（`95f23b0`；已并入执行）

- 接受 probe-first：两层根因框架（primary 动力学 shortcut + learner 态放大）、lanekeep 归 primary、chain 保留为执行缓冲并重定义——全部接受。
- 2 处事实核查：LD 7 维**已隐式含** lane-relative 几何（缺的是关联/选择/监督）；`ld_hist` 是输入（"只作输入"≠不进 memory）。
- 4 条落法与 3 验收补丁——**已并入 §5.8 执行与结果**。

### 5.7 round-19 往来残档（`1ea85dc`；已并入执行）

- **七项代码论断逐条核验=全部成立**（file:line 摘要）：`model.py:505-507`（PlanHead 只看 ld_pool 均值）、`spatial.py:103-106`（LD-LD=槽位链 ⇒ 伪边）、`mem.py:699-705`（LD 仅当前帧）、`model.py:627-632`（K-anchor lane_ctx=0）、`ld.py:209-260/137-170`（环优先排序 / ego.lane 优先）、注释漂移——**量化结果见 §5.8**。
- 接受 0A/0B/0C 前置与全部修正（全语义镜像 / 输出变化非充分 / Probe B 分层 / primary_only 严格旁路）；4 条实施路线（诊断重放 / 一致镜像近似 / 位级纪律 / 同帧配对）——**已执行（§5.8）**。

### 5.8 Round-18A 结果（2026-10-12 19:2x）：**拓扑/输入结构性缺陷 + primary 几何响应 ~50× 过弱（动力学主导）**〔round-20 复核见 §5.10〕

- **0A/0B/0C**：槽位链**伪边率 76.5%**（相邻槽同 lane 仅 23.5%）；ego lane 在场 95.07%（缺失 4.93%、槽近恒满⇒无降级信号）；**lane-assignment 断点：失败 0.5146 vs 成功 0.0363（14×）**、失败前 20 步内断点 127/127（**非因果**，共现/暴露因子）。
- **四 probe（n=500）**：zero_ld Δmu_dθ **0.0153**（LD 必要性 ✓）；**translate ±1m / rotate ±10° = ±0.002/±0.0026（符号正确、单调，但 ~50× 过弱）**；zero_history **0.0134（几何的 5–7× ⇒ 动力学主导）**；straighten/remove_prev/镜像极弱（镜像不闭合）；primary_only≈full（experts 关不降级 ✓）。B 解码仅 curvature 可解（R² 0.52/0.58）。
- **分支判定**：`input_topology`=缺陷 ⇒ **最小语义特征路线**（fingerprint 变更、需重采）；`primary_response`=弱 ⇒ 显式池化设计。**去向（已全部执行）**：修复阶梯 a（模型侧）由 round-20/22 穷举关闭；b（数据侧）→ R21 落地（§5.14/5.16）。**坑（长期）**：LD 消费走 `ld_hist[:,-1]` ⇒ 干预须六帧一致；0C 非因果。**纪律**：零训练、net/pipeline 零改动、`ld.py` 默认关旁路（已提交）。

### 5.9 round-20 往来残档（`37ee932`；已并入执行）

- 裁决=**接受三项收紧**（伪边=架构缺陷证据非闭环因果、几何响应改相对口径、zero_history 强 OOD 需温和对照）+ 批准 M0/M1/M2 等预算消融 + 20A/20B 因果探针 + 验收门（**M2<0.45 → 转数据侧**）；我方 3 条操作性补充（oracle 对照基准、LD 干预六帧一致、M2 绕过 spatial MP）——**已执行，结果见 §5.10**。

### 5.10 Round-20 结果（2026-10-12 20:3x）：**M2 触发预注册停止——停 LD 架构精修、转数据侧**

- **20A 置换探针**：模型对 LD 槽序置换**本就近似不变**（|Δmu_dθ|≈2–5e-06）⇒ 76.5% 伪边在输出层"惰性"、伪边假设输出层不成立。**20B matched-case（125 对）**：失败 switch rate **0.511 vs 0.037（14×）**；切换处跳幅**不判别**（0.0058/0.0068）；判别量=**全程动作基噪声（5×）**；断点=共现/暴露因子（断点→终点 21 步）。
- **三臂 clean150（同批 M0=0.3267）**：M1 **0.3533（+2.67pp；coll −2.67pp）**不过门但无害；M2 **0.0000（崩）** ⇒ **M2<0.45 → 预注册触发转数据侧**；eval500 跳过。结构门：置换敏感度触底、几何响应仅 oracle 2–4%、方向率 ~90%——均未过。
- **坑/综合**：M2 载入须"先 set 再 load"；oracle=解析近似；**离线 loss 不预示闭环保崩**；瓶颈=几何响应量级（~50× 弱）+ 动作基噪声 ⇒ 指向数据/监督侧。**更正（§5.11）**：M2 实为接线缺口（attended 只进 policy/value），"穷举/停 LD 架构"撤回。纪律：默认关+惰性构造；99 passed；M3/重采/A4 未启动。

### 5.11 round-21 代码复核残档（`ed9637d` 前轮；已并入执行）

- **核心指控成立**：`plan_step`（`net/model.py:534-536`）恒 `masked_mean`、net 内 `plan_head(` 唯一调用点 ⇒ PlanHead 全路径（t0/rollout）恒用均值——M2 的 attended 读出只进 `_head_tokens`→policy/value；`set_attn` 一次改 3 变量（关链+LD 全排除 MP+随机初始化）、无 gate/初始化、全灭反向 NaN 风险；5 单测无"PlanHead 消费"契约 ⇒ **§5.10"穷举/停 LD 架构"撤回**。
- **立场（已全案批准）**：先 **M2a/M2b** 修正阶梯（统一 `aggregate_ld`、α=0 逐位、4 契约测试、clean150 门）；R21 语义重建为 **fail 分支预备案**（禁 raw lane_id、拆分 ego/route/legal 标志、route 优先·ego 兜底）。**教训：此后"替换 X"类断言必须配消费测试。**——执行结果见 §5.13/§5.14。

### 5.12 round-22 裁决残档（`ed9637d`；已并入执行）

- 批准 **M0/M1/M2a/M2b 四臂 + Gate A/B + clean150/eval500 门**（限 1 session；无第五臂/M2c/超参；失败自动转 R21）；我方注记：same-lane-as-ego 代理、M0/M1 重训保同批、α 梯度时序、先 set_ld_mode 再 load。——**结果见 §5.13**。

### 5.13 Round-22 结果（2026-10-12 23:0x）：**Outcome D——M2a/M2b 未过开发门 → 按预注册转 R21 语义重建**

- **Gate A 全过（9/9）**：α=0 逐位=baseline；`_masked_mean` 恰半、聚合器 7 次/前向；全灭精确 0/fwd-bwd 有限；α 先行→attention 实更新；链开关隔离；严格加载；消费契约过。**零代码状态差**：M0/M1 ckpt 与 round-20 **168/168 键 max|Δ|=0.0**。
- **Gate B/闭环（量级不足）**：α 收敛 **0.0132/0.0129（≈1.2%）**、entropy 比 0.997/0.992、same-lane ~1.06–1.08×、translate/rotate 0.96–1.15×（需 5×）——未过；**clean150 M2a 0.28 / M2b 0.3267（tmo +8.67pp=off→tmo）且 e002 回落** ⇒ 门未过（eval500 未跑）。判读=**量级问题非接线**（条件化读出在当前 LD 语义下不被支付）；坑：policy `mu` 零初始化、α 梯度时序、超参调整未批。

### 5.14 R21 语义重建结果（R21-0→R21-2，2026-10-12 24:0x）：**三阶段门全过；主因定位经 §5.15 更正为"待 R21-3 裁决"**

- **R21-0（门过）**：唯一纯函数 target-lane 选择（route 优先/ego 兜底；4,632 探针 0 切换）+ 12 维通道 + LD 逐槽补充（role/group/route/offset_s × 6 帧）；schema v7/指纹 v7、旧数据 fail-fast；35 新测、**909 passed**；env 审计全门 true（valid 100%、lat 5e-8m、roundtrip 3.6e-14）；可行性 216 case 99.4%。
- **R21-1（门过）**：smoke 681（重放 501/501 exact）→ pilot **9,515 行**（A 40.4%/B 42.7%/C 16.9%）；质量门全过（同状态 1,013/1,013、valid 100%、expert 恢复 92.6%、跳变 0%）。
- **R21-2（零训练）**：raw 全过（R²≈1）而模型侧全灭（ld_pool lat 0.191、fused −0.046、mu 0.001）；counterfactual sign 73%、curv 8.4% expert、lane/prev 0.36、**history_shuffle ≫ lane（15×）**。→ **唯一行动=lane_semantic 显式接线 + 复测（R21-3）**；措辞更正见 §5.15。
- 产物：`runs/p1_v2/round21/`、`datasets/semantic_v1_pilot/`；零主模型训练；请求 round-23 已由 §5.15/5.16 覆盖。

### 5.15 round-23 裁决残档（`085f4f3`；已并入执行）

- **三项代码指控核验成立**（net 零消费 / 12 维未归一化 / topology 与 margin 语义）⇒ 批准 **R21-3A..3C**（固定归一化+改名+target token+分组池化+统一消费；canary；复测门）+ S0/S1 两臂（弃 S2）+ 正式采集条件后置；§5.14 措辞更正（encoder 清白 → 主因待裁决）。注记：并行采集矛盾按保守（等门）、冻结栈残余区分、改名保 pilot、六帧铁律。——**已执行，结果见 §5.16。**

### 5.16 R21-3A..3C 结果（2026-10-13 01:0x）：**接线+契约+Canary 全过；3C 表示 3/3 过、action 4/5（曲率 20.1% 近门）——首现强正向；门后停止待裁决（round-24）**

- **3A（门过）**：三项修正落地——归一化单一出处（env 常量→schema→net buffer→快照 + 逐位单测）、`distance_to_target_lane_end` 改名**值逐位不变**（71 行×39 键 max diff 0.0；pilot schema 同步 + `schema_sync.json` 白名单）、margin 改名 same-direction road-surface margin（禁监督真值）；接线=`target_lane_token`（mask/valid=0⇒严格 0）+ LD 逐槽增广（role/route/offset/group-equality）+ **显式分组池化** + `build_geometry_context` 统一消费（t0+rollout 1–6+policy/value；禁均值回退+代码断言）；16 新测、全套 **925 passed**；默认关 baseline 逐位不变。
- **3B canary（门过）**：冻结 encoder/MemBank/ST-GNN/MoE/router/policy/value，只训 299,909 参数（+PlanHead fusion 低 LR）；2 epoch/15.1s；aux 收敛（lat 0.968/curv 0.983/change 1.000；heading aux 弱为已知量级效应，表示门以 target token probe 为准）。
- **3C 复测（关键突破）**：target token 解码 **1.000/0.998/1.000**（lat/heading/near-curv）；action 反事实 **sign 99.4%/99.2%**（旧 73%）、**lane 响应 4.31× 旧**（门 3×）、hist/lane **0.243**（门 ≤2；旧口径 36.5）、mirror 98.2/96.5%；**曲率 20.1% expert（门 25%，唯一近门）**；OOD 历史（逆序/shuffle）ratio 3.74 为非门诊断；解码器分析=expert dθ 可从 context/token 线性解码（R² 0.33/0.36）⇒ 残余=**冻结解码器表达 + 短适配预算**，非语义缺失。
- **判读**：语义接线把"几何响应弱 + history 主导"两症候同时翻面（方向正确率 ~99%、history 相对影响骤降）；**首现强正向信号**；唯一未达=曲率量级（近门），机制上归因于 canary 冻结栈（policy/value 未放开、15s 预算）。
- **请求（round-24）**：按预注册裁决——① 视 3C 为**门过**（表示 3/3 + sign/lane/hist 全过、curv 近门属冻结栈伪影）→ 批准**正式 v7 主采集（20–30 万 expert + 3–5 万扰动）→ S0/S1 两臂**；或 ② 先补一轮 canary v2（PlanHead fusion 解冻至正常 LR/加 epoch）清 25% 门再采。**我方倾向 ①**（方向性已证；S1 全适配将直接复检曲率指标）。门表/证据：`runs/p1_v2/round23/canary_report.md` + `r21_3c_probe.json`。
- 产物：`runs/p1_v2/round23/`（3A 审计 / 3C 探针 / canary / 报告）；零正式采集、零全模型训练、DAgger/MoE/RL 未动。

### 5.17 round-24 裁决：**批准 R21-3D（代码语义一致性 + 独立集复测 + 小规模闭环 canary）；正式采集仍后置——已开工**

- **核验（三项代码指控全部成立）**：① `model.py:414` `include_ld_chain = mode in ("baseline","attn","semantic")` ⇒ **semantic 仍带 LD-LD 伪边链**，且 z_ld 已过 ST-GNN、可能已混入跨 lane 伪消息；② rollout `:1004-1014` 每步 `build_geometry_context(z_ld_k, z_ego_k, semantic=t0)` ⇒ **lat/heading/margin/distance 为 t0 静态值重复注入 6 步**（过度纠偏风险）；③ target-lane"稳定/有效"≠"正确 route lane"（缺复杂拓扑真值审计）。
- **接受**：三项修正方向（**semantic_chain vs semantic_no_chain** 单变量对照、倾向前者默认 **no_chain**；rollout 口径对照 **full-static / t0-only / static-role-only**；**target-lane correctness 审计** 500–1000 复杂状态、门 99%/95%/<1%）；**曲率措辞更正**（不当称"冻结栈伪影"——改"方向接地已强证、**曲率幅值未闭合**"，且与 risk ② 相关）；**pilot 局限**（B traffic=0/reset warmup、单 ckpt、无高速、9.5k；3C 阳性或部分来自"干净受控分布"）；**exposure 比例**（normal 50–60% / pert 20–25% / learner 15–20% / 复杂 5–10%，sampler 控制）替代"30 万+5 万"直采。
- **R21-3D 门（已锁定）**：**3D-1** 三项一致性（推荐默认+证据；后段不过度纠偏；target 正确率门）→ **3D-2** 独立集复测（全新 episode/连续 history/含 traffic/未见 spec；sign ≥95%、mirror ≥95%、lane ≥3×、hist/lane ≤1、curv ≥20%）→ **3D-3** 小 canary（45–75k 行、连续真实 history+traffic 子集+速度带+对称扰动；S0/S1 Primary-only、learned chain、MoE/WM 关；clean150：succ ≥+5pp ∧ **≥0.40**、off ≥−5pp、coll ≤+2pp、tmo ≤+3pp、饱和 −20%、反事实过；分组闭环+全监控清单；只从普通路提升而复杂路恶化 ⇒ 停）→ **3D-4 正式采集=条件批准（过门后另请；150–250k/50–80k/30–50k/20–30k）**。
- 未达门处置（预注册）：先定位 **chain / rollout semantic / target selection**，不得转 DAgger/MoE/RL。
- **开工**：R21-3D lane 已派发（复用 R21 session；产物 `runs/p1_v2/round24/`；零 20 万级采集、零 DAgger/MoE/RL）。

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`（P1-v2 决策 + Round-12A 结果）、`docs/recovery_v2_contract.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`
- **报告**：`docs/p1_0_report.md`（P1-0 FINAL）、`docs/p1_2_diagnosis.md`（P1-2 FINAL）、`docs/recovery_v1_dataset_card.md`、`docs/r21_lane_semantic_contract.md`、`docs/r21_target_lane_selection.md`、`docs/semantic_v1_dataset_card.md`、`docs/r21_3a_semantic_wiring.md`
- **数据**：`datasets/recovery_v1/`、`datasets/recovery_v2/`（samples / manifest / split / data_gate_report / _verify）、`datasets/semantic_v1_pilot/`（v7 语义 pilot 9,515 行）
- **产物**：`runs/p1_v2/round12a/`（trace/探针/15 配置/归因/九条件）、`runs/p1_v2/round15a/`（A1 审计 / A2 尾族 oracle / Track C / P1-3 FP 门）、`runs/p1_v2/round17/`（两臂 ckpt / 三尾 clean150 / a3_report）、`runs/p1_v2/round18a/`（probes/审计/断点）、`runs/p1_v2/round20/`（20A 置换 / 20B matched-case / M0-M2 / 结构门）、`runs/p1_v2/round22/`（四臂 / LdAggregate / 契约 / report）、`runs/p1_v2/round21/`（契约审计 / 可行性 / R21-2 表示 probe）、`runs/p1_v2/round23/`（3A 审计 / 3C 探针 / canary）、`runs/p1_v2/`（intervention_probe*、probe_robustness、probe_v2_check、bias_diagnostic、gated_residual_report、gated_residual_pilot/）、`runs/p1_0/`（d2_pacing_matrix、dth_sweep_summary、start_model_decision）、`runs/p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`（全链流水）、`.slim/deepwork/p1_v2/`（probe / collector / gated_residual / round12a / round18a / round20 / round21 / round23 脚本 + logs）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮）
- **关键提交**：`37000ae`/`a371cc5`（过滤）、`e953f10`（P1-3）、`c20fa00`（P1-2 诊断）、`28acbcd`（dθ-only）、`1800e69`/`a42ac79`/`27ced1f`/`7b83730`（P1-v2 步）、`061bbac`/`663fedc`（argue §7/§8）、`5962848`（12A 开工）、`249c51c`（12A NO-GO）、`3b5124f`（尾族工具）、`dab85f8`（§5.3 批次结果）、`74f522f`（第三轮精简，位于 `Planner_RL_copy`）、`d70ec1b`（§5.4）、`f8169fd`（A3 配方开关）、`96df7cb`+`c71f389`（Round-18A）、`da83ce1`（LD 消融开关）、`e638b76`（§5.10）、`02ef947`+`a1986f3`（第六/七轮精简）、`69c0545`+`7fba3f5`（§5.11/§5.15）、`5e6e3e2`（LD 聚合）、`0f73b6d`+`51712bd`（R21 语义）、`11f0000`（§5.16）

---

## 7. 存档说明（历次精简 / 去哪找）

- **第八轮精简（本版；300 → ~270 行）**：§5.15/5.13/5.10/5.8/5.1–5.2 收缩、§0 补 R21-3 行、§6 更新；改前 300 行 = `git show 11f0000:docs/deepseekv4p1_argue.md`。
- **历次精简留档**：第七轮（305→283，`a1986f3`）；第六轮（342→261，`e638b76`）；第五轮（355→310，`96df7cb`）；第四轮（299→282，`58ac4c3`）；第三轮（`74f522f`，`Planner_RL_copy`）；第二轮（262→~210，`249c51c`）；第一轮（1114→166，`7b83730`）。
- **压缩时间线**：
  - 2026-10-09：round-1/2 收拢；《评测兼容设计》落 `docs/p0_eval_compat_design.md`；
  - 2026-10-09/10：P0 三件套 → P0-B 矩阵 + P0-C 归因 → 审计定稿（§2.1）；
  - 2026-10-10：排摸收官（§2.2）；round-3–8 与 P1 锁板（§2.4）；
  - 2026-10-10→11：P1 执行——P1-0/1/2/3（§2.3）；
  - 2026-10-11：round-9/10 定板（§3.1）→ P1-v2 三步（§3.2–3.4）+ 旁线 D/E（§3.5）；
  - 2026-10-11 晚：round-12/13/14 讨论（§4.1）→ **Round-12A 执行与 NO-GO（§4.2–4.5）**；
  - 2026-10-11 深夜–10-12：round-15/16/17 讨论与执行（§5.1–5.7）：Round-15A 全批次 → **A3 执行/判负 → round-18**；
  - 2026-10-12：round-18/19/20（§5.6–5.10）：Round-18A probe+审计 → round-20 M0-M2 消融+因果探针 → **停 LD 架构、转数据侧（round-21 输入）**；
  - 2026-10-12 深夜–10-13 凌晨：round-21 复核 → round-22 四臂（Outcome D）→ **R21-0..2 三阶段门全过**（§5.11–5.14）；
  - 2026-10-13 凌晨：round-23 → R21-3A..3C（§5.15–5.16）：3A/3B 门过、**3C 表示 3/3 + action 4/5（首现强正向）**。
