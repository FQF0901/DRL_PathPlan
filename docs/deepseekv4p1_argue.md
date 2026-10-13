# deepseek-v4.1 argue / 主文档（R28：28A 归因 ✅ · K1 假设否定+S2 崩塌 · round-29 待裁）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-13 傍晚）**：P1-v2 条件轨已关闭；Round-15A/A3/18A/20 递进；round-21/22 复核与四臂（§5.11–5.13）；R21 0..3D（§5.14–5.18）；round-25/26/27 R25–R27（§5.19–5.24）；**round-28 R28：28A 归因 ✅（主轴=F3 速度参考；F1 κ-ds 耦合 / F4 LQR 尾段命中）；K1 核心假设被否定（ds≡legacy 未消除 tmo）+ S2 崩塌 ⇒ 预注册停止（§5.26）；待 round-29（lateral/tracking 线；S2 解冻模式待审）**；正式采集/DAgger/MoE/RL 仍冻结。
> **沿革**：round-2 收拢 → P0 审计 + 排摸（§2.1/2.2）→ P1 计划 round-3–8 锁板（§2.4）→ P1 执行（§2.3）→ round-9/10 P1-v2 定板与共识（§3）→ round-12/13/14 讨论（§4.1）→ **Round-12A 诊断与关闭（§4）** → round-15/16 讨论（§5.1–5.2）→ Round-15A 全批次完成（§5.3）→ round-17：A3 执行/判负（§5.4–5.5）→ round-18/19：probe-first + LD 拓扑审计锁定（§5.6–5.7）→ Round-18A 完成（§5.8）→ round-20：M0/M1/M2 消融 + 因果探针（§5.9）→ Round-20 结果（§5.10）→ round-21 复核（§5.11）→ round-22 四臂 Outcome D（§5.13）→ R21-0..2 门全过（§5.14）→ round-23：R21-3A..3C 批准开工（§5.15）→ 3A/3B 门过、3C 表示 3/3+action 4/5（§5.16）→ round-24：R21-3D 执行（3D-1/2 过、3D-3 未过门+定位）（§5.17–5.18）→ round-25：R25 执行（残差+解冻+动作监督；P1 最优、门未全过；标签 bug 修复）（§5.19–5.20）→ round-26：R26 执行（Q1 改进全面生效、teacher/sat 未闭合+停止）（§5.21–5.22）→ round-27：R27 执行（27A/27B 过、R1 修饱和但 tmo/teacher/plan-Δκ 未闭合+停止）（§5.23–5.24）→ round-28：R28 执行（28A 归因 ✅、K1 假设否定+S2 崩塌+停止）（§5.25–5.26）。
> **本版说明**：第十三轮精简（289 → ~285 行）：§5.25 并入残档、§5.24/5.22/5.26 收缩、§0 补 R28 行并刷新终局、§6/§7 更新；**当前状态=R28：28A 归因 ✅（主轴 F3）、K1 假设否定（ds≡legacy 未消除 tmo）+S2 崩塌 ⇒ 停止（§5.26）、round-29 待裁**；改前 289 行 = `git show c5a44df:docs/deepseekv4p1_argue.md`。历版全文见 §7。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **主线结果（P1-v2 → R21-3D）**：

| 步 | 结果 | 判定 |
|---|---|---|
| Steps 1–3（P1-v2；后由 12A 关闭） | 1a 过（F1 0.934）→ 1b **`no_increment`**（F1 0.682）→ 1c F2m **0.804/0.843 重开**；recovery_v2 2,693 行、六门 PASS（v1 根因 33.2）；gated pilot **−3.33pp FAIL** | 条件轨（12A 关闭） |
| **Round-12A 诊断** | 15 配置全未过九条件；**gate 无选择性**；后端 6/40 集翻转 | **NO-GO → 主线关闭** |
| **Round-15A 零训练批次** | A1 共变量偏移（6.4×/16.5×）；A2 硬上限未触发、repeat 最优 **0.778**；C1 FAIL；P1-3 FAIL | A3/A4 条件满足（§5.3） |
| **Round-17 A3** | 确定性尾全线 succ **0**；同 ckpt chain **0.3533**（隔离=动作可持性）；远期链监督无有害证据；A1 复跑 learner 无改善 | **FAIL → round-18（§5.5）** |
| **Round-18A+20（LD 结构）** | 伪边输出层惰性（置换不变）；M1 +2.67pp 无净胜；几何响应仅 oracle **2–4%**；M2 实为"未接 PlanHead"的接线缺口（§5.11） | → 修正版四臂（§5.13） |
| **Round-22 + R21-0..2** | 四臂 Outcome D（α≈1.3%、近均匀、几何响应未达 5×；e002 回落）；R21 三阶段门全过（措辞经 §5.15 更正） | → R21-3 接线+复测（§5.15–5.16） |
| **R21-3A..3C** | 接线+契约 **925 passed**；canary 门过；3C 表示 3/3 + action 4/5（曲率 20.1% 近门） | → R21-3D（§5.17–5.18） |
| **R21-3D** | **3D-1/2 门过**（独立集 **5/5**：曲率 **25.0%**、lane 6.17×、target 1.00/1.00）；**3D-3 小 canary 未过门（S0 −8.7pp / S1 −8.0pp）** | → R25（§5.19–5.20） |
| **Round-25（R25）** | Gate A 全过（β=0 逐位）；**P1 首现闭环超 legacy：succ 0.400（+7.3pp）/ off 0.347（−20.7pp）**；B/C/D 未全过（mirror-lat 0.891、teacher +57%、tmo +14pp、sat 0.894）；**修复 R21-1 遗留 B 行标签错配（corr −0.156→+0.606）** | → R26（§5.21–5.22） |
| **Round-26（R26）** | 26A 数据 ✅（v1 暴露 0）；**Q1 全面优于 Q0：pert −53.6% / learner −51.5%、Δp99× 1.017、mirror ✓、calib50 0.50 vs 0.40**；硬门未全过（**teacher +43.7%、sat +14.5pp**） | → R27（§5.23–5.24） |
| **Round-27（R27）** | **27A/27B ✅、R0 逐位复现 Q1**；R1 目标轴见效（**sat 0.949→0.710 ✓**、learner −72.2%、N0 ↓40%）但 **tmo +26pp、teacher +59.6%、plan Δκ 2.1–2.3× ✗** | → R28（§5.25–5.26） |
| **Round-28（R28）** | **28A 归因 ✅**（主轴 F3；F1 κ-ds 耦合/F4 LQR 尾段命中；覆盖 1.0）；**K1 假设否定（ds≡legacy 未消除 tmo 0.28）+S2 崩塌**；K1 S1 强（κ MAE −71.6%、sat −12.5pp、calib 0.52/off 0.12） | **停止+定位：转 lateral/tracking；S2 解冻模式待审 → 待 round-29（§5.26）** |

- **旁线（D/E）**：dθ-only 0.728（+2.2pp，V1=组合效应确认）；常态 μ 无系统性 bias。均不改训练 target。**终局**：条件轨已关闭；A3 判负（**动作可持性**）；LD 架构排摸结束（Outcome D）；R21→R26 语义链验证（§5.14–5.22）→ R27（§5.23–5.24）→ **R28：28A 归因 ✅（主轴 F3 速度参考；F1 κ-ds 耦合、F4 LQR 尾段命中）；K1 假设否定（ds≡legacy 未消除 tmo）+ S2 崩塌 ⇒ 停止+定位（转 lateral/tracking；S2 解冻模式待审），待 round-29（§5.26）**。**收敛一句话：lane semantic 前三层已正证；未闭合 = normal-state preservation + plan-level trackability，其耦合层已被 28A 解构（F3/F1/F4）。**
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
- **Q2/Q3**：保持 plan 协议（`repeat_action` 归零、A-hold 仅 RL 记账口径）；失败以 off-road/压线为主（76–78%）、**>50% 带 plan-infeasibility 因子** ⇒ **P1 = 计划侧**。
- 关键行（succ/off/coll）：A IDM **0.756**/0.068/0.144；B oracle 0.608/0.134/0.058；C laneplan 0.442/0.232/0.192；**D2 0.706**/0.038/0.020；v8 模型行 0.22–0.33。全表见 git 历史 §5.1 + `runs/forensics/`。
- **timeout/防线**：d2 timeout rc 中位 0.849（55% ≥0.8）⇒ 执行链"跑得慢"、d2 可能低估上限；fail-fast 两次拦截静默错评（`08582f2`/`5e6e1b3`）。

### 2.2 排摸收官（容量分配证据）

- 单变量赢家：**pri512**（eval500 0.332；clean500 0.334@e020）、**rou64**（clean500 0.370@e015）；attn2 温和正向；组合臂 C-N1（pri512+rou64）**负交互被拒**；最优 epoch 依配置（base 峰 e005、pri512/rou64 峰 e015–e020）。仅作分配证据（C1）。

### 2.3 P1 前序结果要点

- **P1-0**：起点 **pri512@e020**（rou64 +1.40pp、p=0.52 不显著→"不稳定"取 pri512）；配速 25 行：V3（纯速）平坦 <0.75；**V1（ds 拉伸）0.762/0.776/0.778 @1.20/1.25/1.30**（timeout 0.236→0.086；vs s100 配对 z=4.10；vs IDM s120–135 打平）；等价口径 PASS（flags 全等 + |Δrc|≤1e-3；499/500 逐位一致）。
- **P1-1**：17,256 可用行/1,493 eps，数据门 13/13；**勘误**：k≥1 行**反事实错位 9,377 行**（obs=Pass-1 学生态 × 标签=Pass-2 专家帧）；`takeover_only` 过滤后对齐可用 2,317 行。
- **P1-2（四轴全败，诊断闭合）**：混采 1–2 epoch 崩（succ→0.01–0.14，离线 val 不可见）；仅配方（A2）0.300/0.307 稳。两层根因：①反事实错位（已修）；②**绝对目标恢复监督 × 失败态/正常低速态在 obs 不可分 × 高倍过采样**（受控探针：偏移集中 v∈[2,4)、Δds +0.44~0.57、~40×/epoch）。详见 `docs/p1_2_diagnosis.md`。
- **P1-3**：三损失实现就绪（默认关、既有路径逐位不变）；**cos(grad_traj, grad_geometry)=−0.73**；曲率阈值须按分布校准；**待几何 FP 门**（未训练）。

### 2.4 P1 执行手册 v1 锁定要点（存档）

- **范围**：P1-0A/B/C → P1-1 → P1-2 → P1-3；条件 **P1-4A**（启动=监督模型 eval500 ≥0.60 + probes 全 PASS + sampled action 贯通）；不承诺 P1-4B / K-anchor 重启 / GRPO / Best-of-G / WM-MoE 对照 / 容量扫描。
- **手册 3 问**：① V3 配速 ≥0.80？（答：<0.75）；② rou64@e015 vs pri512@e020？（答：不显著取 pri512）；③ recovery+几何把 ~0.37 推到 ≥0.60？（**未达成**）。细则原文 git 历史 §7.10；契约 `docs/p1_implementation_contract.md`、`docs/p1_probe_spec.md`。

---

## 3. P1-v2 执行纪要（Step 1–3；压缩存档）

- **锁板**：可辨识性 probe → recovery_v2 同状态采集 → gated residual pilot → 几何 FP 门 → 独立小消融 → candidate recipe；intervention 标签 = 同状态"baseline 将失败 ∧ expert 可恢复"、行级同刻同态契约；每步 Go/No-Go、不自动晋级；几何 proxy 未过 FP 门不得反传。无条件轨降级为诊断旁线（不改训练 target；dθ-only 反事实 + bias ③ 映射既有 calibration）。**冻结**：recovery_v1 训练、P1-4、K-anchor、GRPO、WM/MoE、容量。

**Step 1/2（GO/PASS）**：1a Gate 过（F1 0.934）但 time-only 0.940 ≥ F1 ⇒ conditional_pass；**1b progress 匹配 F1 0.682/0.731 ⇒ `no_increment`**；**1c F2m=0.804/0.843 ⇒ 路径重开**（禁锚点列）。**Step 2**：**2,693 行/110 eps**（intervention 836；recovery 89.2%）；六硬门全 PASS（同状态 100%、重放 1,216 行 0 失配、split 无泄漏、覆盖比 1.08）；**v1 根因实证：Pass-2 同状态 obs vs Pass-1 中位差 33.2**；probe_v2_check F2m 0.87–0.92。注记：mode 邻步切换 0.611（仅分层，**不得作训练目标**）；CPU/CUDA 分歧（标签自洽 CPU）。

**Step 3（FAIL）+ 旁线**：gated pilot：gate=0 逐位==base；离线过（激活 3.99%、recall 61.1%、AUC 0.870）；**clean150 0.3067（−3.33pp）**（配对 13 负/8 正）；机制=闭环激活 **16.1% vs 离线 3.99%** ⇒ 部署迁移过触发（根因：F2m footprint 需活 env）。旁线：**E** 常态 μ 无系统性 bias（唯一系统性=尾链 dθ −0.00135、spread 17×）；**D** dθ-only 0.728（+2.2pp）、饱和 0.132→**0.043** ⇒ V1="更长步+更平曲率"组合效应。零生产改动；`gated_residual_report.json` verdict=FAIL。

---

## 4. Round-12A 零训练诊断与主线关闭（2026-10-11 晚；压缩存档）

**授权链（round-12→14）**：round-12（`017e425`）先零训练复盘（gate 误报/残差幅度/时序抖动三机制分解）、同架构仅一训；round-13（`cb34622`）补三约束（CPU/CUDA 全链一致性、common-prefix 首分歧分解、episode 级破坏性误触发 <5%）、扫描补 2×2×2 交叉；round-14（`ea29866`）**12A LOCKED**、统一 CUDA 口径、九条件（含"非孤立有效区域"）、**允许"无有效区域"=判负关闭**。我方并入：闭环重跑+逐步遥测（旧产物无逐步字段）、几何余量改善预注册口径、"12B 同状态双 rollout"、"残差有效"表述纠偏。

**执行（零训练）**：Stage1 逐 step 遥测（gate_prob/residual/base·final plan/tracker_ref/rc/speed/pose + `--residual-scale`/`--debounce`；产物 trace.npz+JSONL）；Stage2 逐 episode 复现（gate-off 0.3400/gated 0.3067；13 翻负/8 翻正集合一致）；Stage3 CPU/CUDA 全链探针（2,538 状态：gate 概率 Δp95 2.5e-4、各阈值跨界率 ≤0.12%；**e2e 40 eps 6 集翻转**、材料分离中位 40 env steps）；Stage4 **15 配置扫描**（threshold×5→scale×4→debounce×3 + 2×2×2 交叉，单轮 ~3.7 min）。

**终判/教训**：15 配置全未过——① "≥6/8 历史翻正"无人达标（最好 5/8）；② CPU/CUDA 无实质分叉全 False（6/40 集翻转）。三问：① **gate 误报且无选择性**（闭环 16.1% vs 离线 3.99%；hist_8 ≈ hist_13；保守配置反选择）；② 残差 1.0 偏强（"救余量、伤进度"）；③ 连续触发 + 后端离散化放大（最长 55 步；2,336 个分叉后激活）。t050_s050_d2 aggregate 反超（+1.33pp）但机制不可信（记录不改判）。教训：闭环度量不可迁移（3.99%→16.1%）；destructive-FP 口径近空；common-prefix 仅同设备；**跨后端 ~15% ⇒ 统一 CUDA**；工程坑（npz 重复解压 SIGKILL、footprint 需活 env、P@10 带 n_pos/n_neg、mode 0.611）；gate **选择性**才是瓶颈。全表见 `249c51c` 版 §4。**判定=NO-GO（无有效区域）→ 关闭 gated residual 主线**（不补 hard negatives/不扩 gate/不调采样；产物 `runs/p1_v2/round12a/`；零生产改动 → 转 round-15）。

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
- 复算 v1/v2_s110 **逐位一致**（因果核验 ✓）；**q1 天花板**：正确当前动作+简单尾 **0.72–0.78（≈IDM 上方）**、硬上限未触发；**q2** 饱和最低 dth_s085（0.043）；**q4** 执行参考无需六步链。

**Track C / 综合（去向 round-17）**：C1 union **0.790 < 0.82 FAIL（不实现）**、C2 动态净增量≈0；**P1-3 FP 门 FAIL（5 项）** ⇒ `ld_margin_weight` 维持 0、几何反传禁止（根因=LD 通道语义，需改损失定义）。① 共变量偏移已证（A1）；② 执行参考无需远期链、**V1 式 ds 拉伸是有效自由度**（A2）；③ "远期链=历史主损"高度可信但未证——**A3 同预算对照为裁决工具**。A3/A4 放行条件满足；**A3 设计**=当前动作 BC+repeat 尾（门 clean150 ≥0.50→eval500→0.60–0.65 才谈 RL）；**A4**=learner 态 DAgger。**天花板提示**：特权动作+最佳尾 ≈0.778 ≈ IDM+2pp——G3 仍须超越教师（RL 后置）或更优执行。

### 5.4 round-17 往来残档（`732ba96`；已并入执行）

- 无异议、开工：4 条落法（训练/尾解耦〔禁未来链监督〕、隔离〔单 head/纯函数/LQR 逐位〕、对照臂〔同预算六步链监督〕、A4 契约〔learner 态采集 90:10–80:20〕）+ 验收全部并入执行——**结果见 §5.5**。

### 5.5 Round-17 A3 结果（2026-10-12 08:49 报告）：**FAIL（决定性负结果）**

**结果表（clean150；基线 pri512 0.3400/0.4933/0.0333）**

| 臂 / 执行 | succ | off | coll | tmo | steer_sat |
|---|---|---|---|---|---|
| A3 臂 / 对照臂 × R / V115 / V125（确定性尾） | **均 0.000** | **1.000** | — | 0 | 0.66–0.70 |
| A3 臂 × chain（同 ckpt 隔离） | **0.3533** | 0.4400 | 0.0200 | 0.187 | 0.768 |
| 对照臂 × chain（原生） | 0.3267 | 0.5533 | 0.0333 | 0.093 | 0.796 |
| oracle 动作 × R / V115 / V125 | **0.687 / 0.747 / 0.780** | 0.02–0.047 | — | — | 0.09–0.14 |

- **门**（A3 最优尾 R）：succ ✗ / off ✗ / learner-误差 ✗（ds **−6.0% 恶化**）/ steer-饱和 ✗（5.1–7.5× oracle）；coll / tmo / clipping / teacher-MAE / plan-逐位 ✓ ⇒ **判负；A4 按锁定不启动**（仅记录契约）。
- **同 ckpt 隔离（关键）**：A3 ckpt——chain 模式 **0.3533** vs 确定性尾 **0.0000** ⇒ 瓶颈 = **动作可持性（holdability）**，非尾形、非监督配方。
- **对照裁决 + A1 复跑**：A3+chain 0.3533 ≈ 对照+chain 0.3267 ≈ 基线 0.3400 ⇒ **远期六步链监督"无害亦无明显收益"**（假说未获实证；等价核验/尾族旧账一致 ⇒ 非工具伪影）；A1 复跑 teacher ds ↓13% 但 learner 态 **+6%** ⇒ 共变量偏移缺口未被"专家数据上当前动作 BC"触及。
- **机制（trace）+新认识**：直道态 dθ 即抖 ±0.09–0.23 rad → repeat 3s 使 LQR 预瞄成紧弯 → 饱和 0.69 → 10–20 步内甩出；**学习链在 chain 模式下充当"执行噪声缓冲"**。① **动作可持性是独立属性**（专家 dθ≈0；模型不具备）；② learner 态缺口仍是第一瓶颈；round-18 候选 a–d → probe-first（§5.6）；生产改动 `f8169fd`。

### 5.6 round-18 往来残档（`95f23b0`；已并入执行）

- 接受 probe-first：两层根因框架（primary 动力学 shortcut + learner 态放大）、lanekeep 归 primary、chain 保留为执行缓冲并重定义——全部接受。
- 2 处事实核查：LD 7 维**已隐式含** lane-relative 几何（缺的是关联/选择/监督）；`ld_hist` 是输入（"只作输入"≠不进 memory）。
- 4 条落法与 3 验收补丁——**已并入 §5.8 执行与结果**。

### 5.7 round-19 往来残档（`1ea85dc`；已并入执行）

- **七项代码论断逐条核验=全部成立**（file:line 摘要）：`model.py:505-507`（PlanHead 只看 ld_pool 均值）、`spatial.py:103-106`（LD-LD=槽位链 ⇒ 伪边）、`mem.py:699-705`（LD 仅当前帧）、`model.py:627-632`（K-anchor lane_ctx=0）、`ld.py:209-260/137-170`（环优先排序 / ego.lane 优先）、注释漂移——**量化结果见 §5.8**。
- 接受 0A/0B/0C 前置与全部修正（全语义镜像 / 输出变化非充分 / Probe B 分层 / primary_only 严格旁路）；4 条实施路线（诊断重放 / 一致镜像近似 / 位级纪律 / 同帧配对）——**已执行（§5.8）**。

### 5.8 Round-18A 结果（2026-10-12 19:2x）：**拓扑/输入结构性缺陷 + primary 几何响应 ~50× 过弱（动力学主导）**〔round-20 复核见 §5.10〕

- **0A/0B/0C**：伪边率 **76.5%**（相邻槽同 lane 仅 23.5%）；ego lane 在场 95.07%；**断点：失败 0.5146 vs 成功 0.0363（14×）**、失败前 20 步 127/127（**非因果**，共现/暴露因子）。
- **四 probe（n=500）+去向**：zero_ld **0.0153**；translate/rotate **±0.002/±0.0026（符号对、单调、但 ~50× 过弱）**；zero_history **0.0134（几何 5–7× ⇒ 动力学主导）**；镜像不闭合；primary_only≈full。⇒ `input_topology` 缺陷 ⇒ **最小语义特征路线**；`primary_response` 弱 ⇒ 显式池化。**已全部执行**（a 由 round-20/22 关闭；b → R21）。**坑（长期）**：LD 消费走 `ld_hist[:,-1]` ⇒ 干预须六帧一致。零训练、`ld.py` 默认关（已提交）。

### 5.9 round-20 往来残档（`37ee932`；已并入执行）

- 裁决=**接受三项收紧**（伪边=架构缺陷证据非闭环因果、几何响应改相对口径、zero_history 强 OOD 需温和对照）+ 批准 M0/M1/M2 等预算消融 + 20A/20B 因果探针 + 验收门（**M2<0.45 → 转数据侧**）；我方 3 条操作性补充（oracle 对照基准、LD 干预六帧一致、M2 绕过 spatial MP）——**已执行，结果见 §5.10**。

### 5.10 Round-20 结果（2026-10-12 20:3x）：**M2 触发预注册停止——停 LD 架构精修、转数据侧**

- **结果（同批 M0=0.3267）**：20A——模型对 LD 槽序置换**本就近似不变**（|Δmu_dθ|≈2–5e-06）⇒ 76.5% 伪边输出层"惰性"；20B——失败 switch rate **0.511 vs 0.037（14×）**、切换处跳幅不判别、判别量=**动作基噪声（5×）**；三臂：M1 **0.3533（+2.67pp；coll −2.67pp）** 不过门但无害、M2 **0.0000（崩）** ⇒ 预注册触发转数据侧（eval500 跳过；结构门未过）。坑：先 set 再 load；oracle=解析近似；**离线 loss 不预示闭环保崩**。**更正（§5.11）**：M2 实为接线缺口，"穷举"撤回；99 passed；M3/重采/A4 未启动。

### 5.11 round-21 代码复核残档（`ed9637d` 前轮；已并入执行）

- **核心指控成立**：`plan_step`（`net/model.py:534-536`）恒 `masked_mean`、net 内 `plan_head(` 唯一调用点 ⇒ PlanHead 全路径（t0/rollout）恒用均值——M2 的 attended 读出只进 `_head_tokens`→policy/value；`set_attn` 一次改 3 变量（关链+LD 全排除 MP+随机初始化）、无 gate/初始化、全灭反向 NaN 风险；5 单测无"PlanHead 消费"契约 ⇒ **§5.10"穷举/停 LD 架构"撤回**。
- **立场（已全案批准）**：先 **M2a/M2b** 修正阶梯（统一 `aggregate_ld`、α=0 逐位、4 契约测试、clean150 门）；R21 语义重建为 **fail 分支预备案**（禁 raw lane_id、拆分 ego/route/legal 标志、route 优先·ego 兜底）。**教训：此后"替换 X"类断言必须配消费测试。**——执行结果见 §5.13/§5.14。

### 5.12 round-22 裁决残档（`ed9637d`；已并入执行）

- 批准 **M0/M1/M2a/M2b 四臂 + Gate A/B + clean150/eval500 门**（限 1 session；无第五臂/M2c/超参；失败自动转 R21）；我方注记：same-lane-as-ego 代理、M0/M1 重训保同批、α 梯度时序、先 set_ld_mode 再 load。——**结果见 §5.13**。

### 5.13 Round-22 结果（2026-10-12 23:0x）：**Outcome D——M2a/M2b 未过开发门 → 按预注册转 R21 语义重建**

- **Gate A 全过（9/9）**：α=0 逐位、`_masked_mean` 恰半、α 先行→实更新、链隔离、严格加载、消费契约过；M0/M1 ckpt 与 round-20 **168/168 max|Δ|=0.0**。**Gate B/闭环未过（量级不足）**：α 0.0132/0.0129（≈1.2%）、translate/rotate 0.96–1.15×；clean150 M2a 0.28 / M2b 0.3267（tmo +8.67pp）且 e002 回落 ⇒ **量级问题非接线**（条件化读出不被支付）。坑：`mu` 零初始化、α 时序。

### 5.14 R21 语义重建结果（R21-0→R21-2，2026-10-12 24:0x）：**三阶段门全过；主因定位经 §5.15 更正为"待 R21-3 裁决"**

- **R21-0/1（门过）**：唯一纯函数 target-lane 选择（route 优先/ego 兜底；4,632 探针 0 切换）+ 12 维通道 + LD 逐槽补充；schema v7、旧数据 fail-fast；35 新测、909 passed；env 审计全门 true；可行性 99.4%。pilot **9,515 行**（A 40.4%/B 42.7%/C 16.9%；同状态 1,013/1,013、expert 恢复 92.6%）。
- **R21-2（零训练）**：raw 全过（R²≈1）而模型侧全灭（ld_pool lat 0.191、fused −0.046、mu 0.001）；sign 73%、curv 8.4%、lane/prev 0.36、**history_shuffle ≫ lane（15×）** ⇒ **唯一行动=lane_semantic 显式接线 + 复测**（请求由 §5.15/5.16 覆盖）。产物 `runs/p1_v2/round21/`、`datasets/semantic_v1_pilot/`；零主模型训练。

### 5.15 round-23 裁决残档（`085f4f3`；已并入执行）

- **三项代码指控核验成立**（net 零消费 / 12 维未归一化 / topology 与 margin 语义）⇒ 批准 **R21-3A..3C**（固定归一化+改名+target token+分组池化+统一消费；canary；复测门）+ S0/S1 两臂（弃 S2）+ 正式采集条件后置；§5.14 措辞更正（encoder 清白 → 主因待裁决）。注记：并行采集矛盾按保守（等门）、冻结栈残余区分、改名保 pilot、六帧铁律。——**已执行，结果见 §5.16。**

### 5.16 R21-3A..3C 结果（2026-10-13 01:0x）：**接线+契约+Canary 全过；3C 表示 3/3 过、action 4/5（曲率 20.1% 近门）——首现强正向；门后停止待裁决（round-24）**

- **3A/3B/3C（门过；3C 首现强正向）**：三项修正（归一化单一出处、改名值逐位不变、margin 降级）+ 接线（target_lane_token 严格零 + 逐槽增广 + 分组池化 + `build_geometry_context` 统一消费）；16 新测、**925 passed**；3C：token 解码 1.000/0.998/1.000、**sign 99.4/99.2%**（旧 73%）、lane **4.31×**、hist/lane 0.243、**曲率 20.1%（近门）**；→ 催生 3D；产物 `runs/p1_v2/round23/`。

### 5.17 round-24 裁决：**批准 R21-3D（代码语义一致性 + 独立集复测 + 小规模闭环 canary）；正式采集仍后置——已开工**

- **三项代码指控核验成立**（semantic 仍带链 / rollout 重复 t0 动态语义 / target 正确性缺审计）⇒ 批准 **R21-3D**（3D-1 一致性 / 3D-2 独立复测 / 3D-3 小 canary；正式采集条件后置）；曲率措辞更正（"冻结栈伪影"→"方向强证、幅值未闭合"）；exposure 比例替代直采；未达门只定位、不转 DAgger/MoE/RL。——**已执行，结果见 §5.18。**

### 5.18 R21-3D 结果（2026-10-13 10:1x）：**3D-1/2 门过（独立集 5/5，含曲率 25.0%）；3D-3 小 canary 未过门（S0 −8.7pp/S1 −8.0pp）→ 按预注册停止并定位**〔追注：3D-3/B 行标签错配污染，见 §5.20〕

- **3D-1/2（门过）**：chain 更优（lane 4.65× vs 2.47×；冻结栈混淆在案）⇒ 保留链；**static_role_only 最优**（lane 6.75×、curv 29.9%、late_share 0.149）⇒ 默认；target correctness 983 状态 **1.00/1.00/严重 0.00**；独立集 8,953 行复测 **sign 99.6/99.8%、mirror 98.4/99.6%、lane 6.17×、hist/lane 0.227、curv 25.0%**。
- **3D-3（未过门 → 停止）**：3d3 53,765 行；S0 0.240 / S1 0.247（−8~9pp）、sat 0.21；独立集 sign 0.56/0.54；定位=**替换式初始化代价 + 语义未被 action 采用**（三嫌疑 chain/rollout/target 全清除）→ 直接催生 R25（§5.19–5.20）；产物 `runs/p1_v2/round24/`，零 20 万级采集。

### 5.19 round-25 裁决残档（`8b35e84`；已并入执行）

- **两项指控核验成立**（替换式接线分布冲击 / `_head_tokens:796-798` target token mask 恒 1）⇒ 批准 **R25**：残差化（β=0 逐位契约）+ mask 修复 + 分阶段解冻 + 动作级监督（加权回归/方向 margin/paired）+ 联合 checkpoint selection（semantic_val/calibration50/clean150）；三臂 P0/P1/P2；门 A–E；失败即停（不堆 loss/数据、不转 DAgger/MoE/RL）。注记：旧契约语义反转、baseline_dθ 符号校验、β 轨迹监控、static curvature 近似。——**已执行，结果见 §5.20。**

### 5.20 Round-25 结果（2026-10-13 12:3x）：**残差平滑接入成功安装"采纳"；P1 首现闭环超 legacy（succ +7.3pp、off −20.7pp）但门未全过 → 按预注册停止+定位；另发现并修复 R21-1 遗留 B 行标签状态错配**

- **25A/Gate A（全过）**：`geometry = legacy + β·Δ`、`target_token_eff = β_token·token`（β 初值 0 进账本）；target token 折叠进 summary 槽（T 恒 39）；β=0 全链 max|Δ|=**0.0** + 残差路径/β 先行梯度/ledger 全过；**937 passed**。
- **重大数据 bug（R21-1 遗留）**：B 行 obs=置位态 × 标签=恢复窗口后（3d3 corr **−0.156**）；**修复=置位后立即查询**（10,803 行、99.96%、corr **+0.606**）；**追注：3B/3C、3D-3 S0/S1 与首跑 P0/P1 均受污染**。
- **三臂+门**：P1 βc +0.10~0.12、sign ~1.0、lane 106–184×、hist ~0.02 vs **P0 不采纳**（lane×0.13–0.6、hist 2.3–9.4）⇒ **动作级监督=采纳必要条件**；B P1 4/5（mirror-lat 0.891✗）；C P1 3/4（teacher +57%✗）；D **succ 0.400（+7.3pp✓）/ off 0.347（−20.7pp✓）** 但 tmo +14pp✗、sat 0.894✗；连续两 epoch 均 False。
- **定位/产物**：量级过强不对称（|Δ|/m 2–4× expert）+ **lat×heading 交互未覆盖**（mirror 大 |heading| 行 0.51–0.73）⇒ 软锚 + 全镜像（R26）；P2 劣于 P1 ⇒ Primary 冻结；`runs/p1_v2/round25/`。

### 5.21 round-26 裁决残档（`1427ed3`；已并入执行）

- **核验成立**（残差 β=0 逐位；target token 折叠式 T 恒 39）⇒ 批准 **R26 四项同改**：① B 标签 v1 全链禁用 + v2 fail-fast；② 监督=相对 legacy 的修正量；③ 分组幅度软锚（P95/P99）；④ 5×5 全镜像 paired。Q0=P1 v2 重跑（基准）/ Q1=Q0+②③④（改进候选）；Primary 冻结；硬门筛选→calibration50；门 A–E（D：succ ≥0.42 且 +7pp）。注记：L_delta 绑定语义 delta + 基底冻结 + 正常态零修正项；独立集 B 行升 v2；β 分组记录。——**已执行，结果见 §5.22。**

### 5.22 Round-26 结果（2026-10-13 14:0x）：**Q1 全面优于 Q0（26B/C/D 均生效）；硬门仍未全过（teacher +40~44%、sat +14.5pp）→ 按预注册停止+定位**

- **26A/26B/C/D（门过/落地）**：标签 v2 全链 fail-fast（**v1 暴露 0**、same-state 99.96%、**ep1091 4 行剔除**；corr **+0.606**）；delta 监督 `mu_on − mu_off.detach()` vs `expert−legacy`（基底冻结、4 单测）；`L_amp`；**5×5 全字段镜像 paired**（对合自测 12 字段）；幅度限额 P95/P99=**0.199/0.329 rad**。
- **关键结果**：**Q1 e1 硬门 6/7（仅 teacher +39.6% ✗）**；连续两 epoch 全过=无 ⇒ **预注册停止**（D/E 未进入）。**Gate C：Q1 4/6**——pert **−53.6%**、learner **−51.5%**、sign err −66.3%、**Δp99× 1.017 ✓**；未过=**teacher +43.7% ✗、sat +14.5pp ✗**。**calib50：Q1 = 0.50**（legacy 0.40；MoE-off 0.52 vs 0.26）。幅度对齐 0.076–0.127 vs expert 0.065–0.106（Q0 超 2×）；镜像残差 0.014–0.040。
- **未闭合+坑+产物**：① teacher 保持性（+40~44%）；② steer 饱和（0.949 vs 0.804）。坑：eval/gate 默认 MoE=on 与训练口径不一致（`--moe-off` 诊断，结论不变）；独立集分母索引（已修）；semantic_off plan 基底须用 z_ld（已修）；`runs/p1_v2/round26/`；待 round-27。

### 5.23 round-27 裁决残档（`268151e`；已并入执行）

- **核验成立** ⇒ 批准 **R27**：27A 归因（饱和六机制分类、≥90% 可归类、**无归因不得设计新饱和 loss**；**禁全局 teacher 零修正强压**）、27B N0/N1/C 分解（trainer/dataset 层）、R0=Q1 复现、R1=+条件 preservation（仅 N0）+归因驱动 trackability（Δκ0 软锚/尾段 Δκ/条件时序平滑）；门 A–E（E：clean150 ≥0.43 且 +8pp）。注记：27A 含 Q1 clean150；NPZ 入口拒 B 行 v1；forward 只读监控；dropout 诊断可选隔离。——**已执行，结果见 §5.24。**

### 5.24 Round-27 结果（2026-10-13 15:5x）：**27A/27B 门过、R0 逐位复现 Q1；R1 修好饱和（sat −9.4pp ✓）但换取 tmo +26pp 与 teacher +59.6%、plan Δκ 2.1–2.3× ⇒ 按预注册停止+定位**

- **27A（Gate A ✅，可归类 100%）**：Q1 sat **0.952** vs legacy 0.816；**73.7% 为"场景必要"**；新增 5,610 步驱动 = **trackability 36.3% / 重规划抖动 34.9% / 首步 27.8%**（尾段 1.0%、Δκ 0%、未归类 0）；calib50 同构；按路型 Q1 0.92–0.97。
- **27B（✅）+ R0**：N0/N1/C 定义锁定（5 单测；比例 **14.3/30.0/55.7%**；教师行 21.7/45.3/33.0）；`load_samples_v2` **硬拒 B 行 v1** + 13 件历史产物 errata；forward 只读监控（plan κ/Δκ + semantic decomposition）。**R0 = Q1 逐位复现**（4 epoch + calib50 全同）⇒ 同批基准成立。
- **R1（+条件 preservation +归因驱动 trackability）**：**目标轴见效**——**sat 0.949→0.710（低于 legacy 0.804；Gate C sat ✅）**、learner −72.2%（R0 −51.5%）、sign err −79.5%、N0 P95 0.033→0.021–0.025（↓40%）；**但换取**：**tmo 0.12→0.34（+26pp ✗✗）**、teacher +43.7%→**+59.6% ✗**、**plan Δκ 2.33×→2.12×（新 Gate C 项，两臂均 ✗；expert P95 0.0169 vs R0 0.0393 / R1 0.0359）**。⇒ 无连续两 epoch 全过 ⇒ **停止**（D/E、eval500 未跑）。
- **定位/产物**：修复"转向过猛"成功，但代价=**进度/纵向驱动被削弱（tmo 爆表）** + 正常态仍被扰（teacher）——即 **ds 进度与转向权限未分离**；lane 建议（未执行）：plan Δκ 步 1–5 全域 expert 分位约束、progress 与 steering 解耦、N1 纳入保持；`runs/p1_v2/round27/`；待 round-28。

### 5.25 round-28 裁决残档（`2acca30`；已并入执行）

- **核验成立、答案认账（根因升级=横纵耦合）** ⇒ 批准 **R28**：28A 零训练耦合审计（物理量换算+遥测补齐；**F1 固定曲率/F2 固定 ds/F3 独立 v_ref 覆写/F4 尾段**；≥90% 归因、逐 episode paired）+ 28B **K0=Q1 复现 / K1=仅横向 κ residual（ds≡legacy、dθ=κ·ds、外部接口不变）**；N0/N1/C 增 `res_kappa + legacy_rollout_safe`；门 A–E（C：ds 逐位=legacy；D：succ ≥0.43 且 +8pp）；K1 败 ⇒ 不加深纵向头。——**已执行，结果见 §5.26。**

### 5.26 Round-28 结果（2026-10-13 17:5x）：**28A 归因 ✅（主轴 F3 速度参考；F1 κ-ds 耦合、F4 LQR 尾段命中）；K1 核心假设被否定（ds≡legacy 未消除 tmo）+ S2 崩塌 ⇒ 预注册停止**

- **28A（Gate A ✅，覆盖率 1.0）**：物理量（Q1 vs legacy）——throttle 饱和 **0.50 vs 0.19**、|e_y| 0.99 vs 0.51、raw steer 2.88 vs 1.40；**chord 效应仅 0.005 m/s**（参考速度差异来自 plan ds 本身）。四族反事实（paired 50）：**F3（几何不动、v_ref×1.15）tmo 12%→8%、off 不变、succ 52% ✓**；**F1（κ 不变同缩）tmo→0 但 off 30%→52% ✓（κ-ds 强耦合）**；**F4（只改尾段）raw steer +0.18、tmo 12%→2% ✓（LQR preview/chain 主耦合源）**；F2 未命中。13 问题 episode 全被 ≥1 族改善（覆盖 1.0）。
- **28B**：**K0=Q1 逐位复现**；**K1 S1 e1/e2 强**（teacher **+1.1%**、learner κ MAE **−71.6%**、lane sign 1.000、**sat −12.5pp vs legacy**、plan Δκ 3.90×→2.78×、calib50 **0.52/off 0.12/rc 0.863**），**S2 解冻末层即崩塌**（e3/e4 teacher 0.28、calib 0.0、off 1.0）。
- **门**：B 4/5（plan Δκ 2.78× ✗、κ mirror **0.70** ✗〔经 common-ds 复核=真横向不对称〕）；C 3/4（**ds 逐位 ✓ max|Δ|=0、teacher ds MAE 逐位 ✓、v_ref 仅 chord ✓；tmo +20pp ✗**）⇒ 无过门 epoch ⇒ 停止（无 clean150/K2）。
- **关键否定结论 + 产物**：**ds≡legacy 并未消除 tmo**（K1 0.28 vs K0 0.24 vs legacy 0.08）⇒ tmo 非纵向 residual 造成，转 lateral/tracking；**S2 解冻崩塌第三次**（R1/K1）待审；`runs/p1_v2/round28/`；待 round-29。

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`（P1-v2 决策 + Round-12A 结果）、`docs/recovery_v2_contract.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`
- **报告**：`docs/p1_0_report.md`（P1-0 FINAL）、`docs/p1_2_diagnosis.md`（P1-2 FINAL）、`docs/recovery_v1_dataset_card.md`、`docs/r21_lane_semantic_contract.md`、`docs/r21_target_lane_selection.md`、`docs/semantic_v1_dataset_card.md`、`docs/r21_3a_semantic_wiring.md`
- **数据**：`datasets/recovery_v1/`、`datasets/recovery_v2/`（samples / manifest / split / data_gate_report / _verify）、`datasets/semantic_v1_pilot/`（9,515 行）、`datasets/semantic_v1_independent/`（8,953）、`datasets/semantic_v1_3d3/`（53,765）
- **产物**：`runs/p1_v2/round12a/`（trace/探针/15 配置/归因/九条件）、`runs/p1_v2/round15a/`（A1 审计 / A2 尾族 oracle / Track C / P1-3 FP 门）、`runs/p1_v2/round17/`（两臂 ckpt / 三尾 clean150 / a3_report）、`runs/p1_v2/round18a/`（probes/审计/断点）、`runs/p1_v2/round20/`（20A 置换 / 20B matched-case / M0-M2 / 结构门）、`runs/p1_v2/round22/`（四臂 / LdAggregate / 契约 / report）、`runs/p1_v2/round21/`（契约审计 / 可行性 / R21-2 表示 probe）、`runs/p1_v2/round23/`（3A 审计 / 3C 探针 / canary）、`runs/p1_v2/round24/`（3D 报告 / 5 探针 / rollout diag / clean150）、`runs/p1_v2/round25/`（残差 / 三臂 / 门表 / selection）、`runs/p1_v2/round26/`（Q0/Q1 / 契约 / calib / gate C）、`runs/p1_v2/round27/`（归因/分解/R0-R1/门/遥测）、`runs/p1_v2/round28/`（耦合审计/反事实×6/K0-K1/门）、`runs/p1_v2/`（intervention_probe*、probe_robustness、probe_v2_check、bias_diagnostic、gated_residual_report、gated_residual_pilot/）、`runs/p1_0/`（d2_pacing_matrix、dth_sweep_summary、start_model_decision）、`runs/p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`（全链流水）、`.slim/deepwork/p1_v2/`（probe / collector / gated_residual / round12a / round18a / round20 / round21 / round23 / round24 / round25 / round26 / round27 / round28 脚本 + logs）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮）
- **关键提交**：`37000ae`/`a371cc5`（过滤）、`e953f10`（P1-3）、`c20fa00`（P1-2 诊断）、`28acbcd`（dθ-only）、`1800e69`/`a42ac79`/`27ced1f`/`7b83730`（P1-v2 步）、`061bbac`/`663fedc`（argue §7/§8）、`5962848`（12A 开工）、`249c51c`（12A NO-GO）、`3b5124f`（尾族工具）、`dab85f8`（§5.3 批次结果）、`74f522f`（第三轮精简，位于 `Planner_RL_copy`）、`d70ec1b`（§5.4）、`f8169fd`（A3 配方开关）、`96df7cb`+`c71f389`（Round-18A）、`da83ce1`（LD 消融开关）、`e638b76`（§5.10）、`02ef947`+`a1986f3`（第六/七轮精简）、`69c0545`+`7fba3f5`（§5.11/§5.15）、`5e6e3e2`（LD 聚合）、`0f73b6d`+`51712bd`（R21 语义）、`11f0000`（§5.16）、`ee2651b`（第八轮精简+推送）、`bf32e23`（round-24 答复）、`eaff4d6`（§5.17）、`3e3df68`（3D 模式）、`f788fdd`（§5.18）、`8b35e84`（round-25 答复）、`19c18b1`（§5.19）、`4ec66ee`（R25 残差）、`6e4fc79`（§5.20）、`f41b06a`（§5.21）、`416e849`（R26 semantic_off）、`9c6a3a2`（§5.22）、`c2da011`（§5.23）、`d5dee2d`（R27 分解/守卫）、`b9e70f4`（§5.24）、`423ae7c`（§5.25）、`8933049`（R28 审计基建）、`c5a44df`（§5.26）

---

## 7. 存档说明（历次精简 / 去哪找）

- **第十三轮精简（本版；289 → ~285 行）**：§5.25 并入残档、§5.24/5.22/5.26 收缩、§0 补 R28 行并刷新终局、§6/§7 更新；改前 289 行 = `git show c5a44df:docs/deepseekv4p1_argue.md`。
- **历次精简留档**：第十二轮（280→270，`e5ae48e`）；第十一轮（269→261，`6e2e2cd`）；第十轮（267→252，`52ce959`）；第九轮（269→248，`48bb8be`）；第八轮（300→252，`ee2651b`）；第七轮（305→283，`a1986f3`）；第六轮（342→261，`e638b76`）；第五轮（355→310，`96df7cb`）；第四轮（299→282，`58ac4c3`）；第三轮（`74f522f`）；第二轮（262→~210，`249c51c`）；第一轮（1114→166，`7b83730`）。
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
  - 2026-10-13 凌晨：round-23 → R21-3A..3C（§5.15–5.16）：3A/3B 门过、**3C 表示 3/3 + action 4/5（首现强正向）**；
  - 2026-10-13 上午：round-24 → R21-3D（§5.17–5.18）：3D-1/2 门过（独立集 5/5）、**3D-3 小 canary 未过门 → 停止+定位（待 round-25）**；
  - 2026-10-13 中午：round-25 → R25（§5.19–5.20）：残差+解冻+动作监督；**P1 首现闭环超 legacy（succ +7.3pp）但门未全过+定位；B 行标签 bug 修复（待 round-26）**；
  - 2026-10-13 下午：round-26 → R26（§5.21–5.22）：数据 v2 契约 + Q1（delta/软锚/全镜像）；**Q1 全面生效（calib50 0.50 vs 0.40、Δp99 ≤1.25、mirror ✓）但 teacher/sat 未闭合 → 停止+定位（待 round-27）**；
  - 2026-10-13 下午·二：round-27 → R27（§5.23–5.24）：27A 归因 + 27B 分解 + R0/R1；**R1 修好饱和（0.710<legacy 0.804）但 tmo +26pp / teacher +59.6% / plan Δκ 2.1–2.3× → 停止+定位（待 round-28）**；
  - 2026-10-13 傍晚：round-28 → R28（§5.25–5.26）：28A 耦合归因 ✅（主轴 F3；F1/F4 命中）+ K0/K1；**K1 假设否定（ds≡legacy 未消除 tmo）+ S2 崩塌 → 停止+定位（转 lateral/tracking，待 round-29）**。
