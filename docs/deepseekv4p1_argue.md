# deepseek-v4.1 argue / 主文档（Round-17 A3 判负 · round-18 输入）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-12 晚）**：P1-v2 条件轨已关闭（Round-12A NO-GO）；Round-15A/A3/Round-18A 递进（§5.3/5.5/5.8）；**round-20 批准模型侧修复——M0/M1/M2 等预算消融 + 20A/20B 因果探针已开工（§5.9）**；数据侧重采/DAgger 条件后置；RL 冻结。
> **沿革**：round-2 收拢 → P0 审计 + 排摸（§2.1/2.2）→ P1 计划 round-3–8 锁板（§2.4）→ P1 执行（§2.3）→ round-9/10 P1-v2 定板与共识（§3）→ round-12/13/14 讨论（§4.1）→ **Round-12A 诊断与关闭（§4）** → round-15/16 讨论（§5.1–5.2）→ Round-15A 全批次完成（§5.3）→ round-17：A3 执行/判负（§5.4–5.5）→ round-18/19：probe-first + LD 拓扑审计锁定（§5.6–5.7）→ Round-18A 完成（§5.8）→ **round-20：M0/M1/M2 消融 + 因果探针（§5.9）**。
> **本版说明**：第五轮精简（355 → ~310 行）：§5.4/5.6/5.7 往来压为残档（结果各自见后节）；上版 355 行 = `git show 96df7cb:docs/deepseekv4p1_argue.md`。历版全文见 §7。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **主线结果（P1-v2 + Round-15A + Round-17）**：

| 步 | 结果 | 判定 |
|---|---|---|
| Step 1 可辨识性探针 | 正式 Gate 过（F1 0.934）；公平 progress 匹配 **0.682/0.731**（`no_increment`）；F2m 自车特征 **0.804/0.843** | 条件重开（特征级） |
| Step 2 recovery_v2 采集 | 2,693 行；**六硬门全 PASS**；v1 根因数值实证（k≥1 vs Pass-1 中位差 33.2） | PASS → Step 3 |
| Step 3 gated residual pilot | clean150 gated **0.3067 vs 基线 0.3400（−3.33pp）**；其余门全过 | **FAIL（停止）** |
| **Round-12A 诊断** | 15 配置全未过九条件；**gate 无选择性**；后端 6/40 集翻转 | **NO-GO → 主线关闭** |
| **Round-15A 零训练批次** | A1 共变量偏移（6.4×/16.5×）；A2 硬上限未触发、repeat 最优 **0.778**；C1 FAIL；P1-3 FAIL | A3/A4 条件满足（§5.3） |
| **Round-17 A3** | 确定性尾全线 succ **0**；同 ckpt chain **0.3533**（隔离=动作可持性）；远期链监督无有害证据；A1 复跑 learner 无改善 | **FAIL → round-18（§5.5）** |

- **旁线**：D（dθ-only 0.728 / +2.2pp，V1=组合效应确认）；E（常态 μ 无系统性 bias、③ 升级证据弱）。均不改训练 target。
- **终局**：条件轨已关闭（gate 无选择性）；Round-15A → A3 判负（**动作可持性**、远期链监督无有害证据）⇒ **round-18 锁定 probe-first**（Round-18A 四反事实 probe + 7 验收；A4/特征/重跑全后置——§5.6）。
- **约束**（持续有效）：plan 部署/评测协议；单 GPU；fail-fast；task-session 解耦 + 20 min 节拍；conda/python 只读；正式 run 不清理。**冻结维持**：P1-4、K-anchor、GRPO、WM/MoE、容量、recovery 扩采、几何反传；A4/可持性实验待 round-18；禁制详见 §5.1/§5.4。

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

## 3. P1-v2 执行纪要（Step 1–3）

### 3.1 锁板与执行序

- **P1-v2 主线（LOCKED）**：可辨识性 probe → recovery_v2 同状态采集 → gated residual pilot → 几何 FP 门 → 独立小消融 → candidate recipe。
- **原则**：intervention 标签 = 同状态"baseline 将失败 ∧ expert 可恢复"；**行级同状态契约**（obs / baseline plan / expert 标签同刻同态）；natural 占比纪律（2–5% 口径留档四元）；几何 proxy 未过 FP 门不得反传；**每步 Go/No-Go、不自动晋级**。
- **我方两条修正（round-10 §11.2，已执行）**：① 探针负样本 episode/time 匹配（Step 1b 的 progress 匹配即其落实）；② 无条件轨分解补 **④ dθ-only** 反事实；bias ③ 映射既有 `phase3_bias_calibration`。**无条件轨降级为诊断旁线**（不改训练 target）。
- **冻结**：recovery_v1 训练、P1-4、K-anchor、GRPO、WM/MoE、容量。

### 3.2 Step 1：可辨识性探针（1a → 1b → 1c）

- 数据：学生 pri512@e020 CPU 确定性重放（60 eps = 40 败 + 20 胜，5,915 帧）；正=166（v1 对齐行）；matched 负=713（同 episode、plan-valid、早段）；跨池负 36k。
- **1a 正式 Gate（过）**：F1 ROC-AUC **0.934/0.933**（LR/MLP）、P@10 **0.943/0.909**；切片 2–4 m/s 0.899、tollgate 0.987、merge 0.965、roundabout 0.899；跨池 0.981/0.984。
- **防泄漏自检**：time-only（仅 episode 时间）**0.940 ≥ F1**；时间邻近负样本 0.906 vs 0.909；2–4 m/s 0.885 vs 0.895 ⇒（1a）判 **conditional_pass**。
- **1b 决定性对照**（progress 匹配：rc±ε + 速度带 + 路型、跨 episode、时间不参与；Δrc 中位 0.0054）：F1 **0.682/0.731**、净增量 +0.177；time-only 0.490≈chance（匹配生效）；弱档 **prevention 0.633/0.712**；晚段健康负样本 2,781 条/51 集（免重放挖出）。→ 预注册判据（<0.80）**`no_increment`**：1a 的 0.934 系进度/时间混淆，F1 单独不成立。
- **1c 特征重测**（按带结构"0.65–0.80 → 加 margin/plan 特征重测"）：**F2m（自车可算：lane_lat + 自身 plan footprint·inside_ratio + 历史；无锚点 oracle 列）= 0.804/0.843 ≥ 0.80** ⇒ 路径重开。注：F2/F3=0.996 属定义性 oracle 循环（含 d_* 锚点列），不作证据。
- 判定：**Step 2 GO**；gate 设计须含 plan/margin 派生特征、**禁锚点 oracle 列**；pilot 自身 Gate 为混淆最终仲裁。

### 3.3 Step 2：recovery_v2 同状态采集（PASS）

- 规模：**2,693 行 / 110 eps**（takeover 1,292 + control 1,401；labeled intervention **836** = correction 660 / prevention 400 / late 232；232 windows；expert recovery **89.2%**）。
- **六硬门全 PASS**：① 同状态 100%（全行离线重算 max_plan_diff 6.7e-06、0 失败 + 1,216 行抽样重放 0 失配）；② 确定性重放 ≥1,000（1,216 行，含 471 k≥1）；③ split 88/22 无泄漏；④ recovery 89.2% ≥80%；⑤ action-chain ↔ recovery traj 一致 0 失败；⑥ control 覆盖（比 1.08、晚段 75.2%、rc≥0.5 36%）。
- **v1 根因正面实证**：k≥1 行存 **Pass-2 同状态 obs**（vs Pass-B 重放逐位 0.0；vs Pass-1 同 step 中位差 **33.2** ≙ v1 错位做法）。
- 附 **probe_v2_check**（干净特征集、v2 真实数据）：F2m **0.87–0.92（natural）/ 0.867–0.895（progress 1:1）**、净增量 +0.33~+0.39；F1 fair（1:1）0.83/0.84。
- 注记：mode 邻步切换率 0.611（仅分层用，**不得作训练目标**）；难度偏 easy（56%）；CPU/CUDA 结局分歧（标签自洽于 CPU）。

### 3.4 Step 3：gated residual pilot（FAIL · 按预注册停止）

- 架构：`final = base + gate(obs, base_plan) × residual(obs, base_plan)`；**base = pri512@e020 冻结锚定**；gate/residual 小头（50,125 参数）；gate 输入 = F2m 类特征（单测断言不含 oracle 列）；正常行 gate 监督 = 0；residual = expert_plan − base_plan（仅 intervention 有效行）；梯度分记；**gate=0 时 final==base 逐位**（单测 + 闭环复验）。
- **Stage A/B1/B2 全过**：单测 5/5；gate AUC 0.870（二 epoch 不塌）；离线：正常行激活 **3.99%<5%**、recall **61.1%≥60%**、MAE/ADE 无退化（intervention 行还改善）；数据纪律记录（raw 31.3% / exposure 3% / eir 0.096，无 >10×）。
- **clean150 闭环（唯一失败项）**：gate-off 逐位复现基线 **0.3400/0.4933/0.0333**；gated（阈值 0.05）**0.3067（−3.33pp；门=≥−2pp）** / 0.4800（−1.33pp） / 0.0267（−0.67pp）；配对 **13 成功→失败 vs 8 失败→成功**（净 −5）；timeout 23→30；均速 3.39→3.27。
- **机制（已验证）**：闭环 gate 激活率 **16.1% 步**（p95 47%、142/150 集）vs 离线 3.99% ⇒ **部署分布迁移下 gate 过触发**，"减速+回正"残差（ds 中位 −1.06）施于正常态。根因数据面：**F2m 的 footprint 特征需活 env** ⇒ gate=0 监督只能取 v2 control（覆盖窄、与 clean150 不同源）。
- **协议合规**：未调参 / 未加复杂度 / 未做过采样调整；零生产代码改动；`runs/p1_v2/gated_residual_report.json` verdict=FAIL。

### 3.5 诊断旁线（不涉训练 target）

- **E · bias 离线诊断**（`runs/p1_v2/bias_diagnostic.json`）：trainer 口径交叉核验 max|Δ|≈1.7e-17；规范口径（w>0，n=4417）：μ_ds −0.0015（t=−0.71）、μ_dθ −0.00011（t=−0.40）⇒ **常态 μ 无系统性 bias**（v7 时代"欠转"未复现）；唯一系统性=尾链 dθ −0.00135（t=−6.4）但 spread 17×（stable=false）；**大偏差全在 w=0 过滤行**（μ_ds +1.10）；失败态对照偏差巨大（μ_dθ −0.083、lat −1.41 m ⇒ 状态条件性，正是 recovery 监督域）。
- **D · dθ-only 反事实**（`runs/p1_0/pacing/dth_sweep_summary.json`）：dth0.85 succ **0.728**（+2.2pp vs 基线 0.706）、timeout 0.236→0.210、steer 饱和 0.132→**0.043**；dth0.90 0.696（≈基线）。结合 ③ 证据弱 ⇒ **V1（+7pp）确认为"更长步+更平曲率"的组合效应**，维持 calibration oracle 定位（不改训练 target）。
- Δκ/progress 监控：未实施（保留于 P1-3 项）。

---

## 4. Round-12A 零训练诊断与主线关闭（2026-10-11 晚）

### 4.1 argue 链摘要（round-12 → 14；均已并入执行）

1. **round-12（`017e425`）**：先零训练复盘（三机制分解：gate 误报 / 残差幅度 / 时序抖动），不得以调参直接下结论；① hard negatives 条件批准；② 阈值=部署参数（校准集选择、独立验证）；③ 特征离线化暂缓；**同架构仅一训、修复仍失败即关闭主线**。
2. **round-13（`cb34622`）**：补三约束——CPU/CUDA 后端一致性探针（端到端全链）、common-prefix（首分歧）分解、episode 级"破坏性误触发 <5%"；扫描补 **2×2×2 交叉**；12B 标签须**同状态双 rollout**；"残差有效"表述纠偏（我方认账）。
3. **round-14（`ea29866`）**：**12A LOCKED、批准开工**；后端差异处置=**统一 CUDA 口径**（基线 0.3400 比较性不得破坏）；九条件（含"非孤立有效区域"与 CPU/CUDA 无实质分叉）；**允许"无有效区域"=判负 → 关闭主线**。
4. **我方并入的执行级修正**（§7.2/§8.2）：① "离线回放"落为**闭环重跑 + 逐步遥测**（旧产物无逐步字段，事实性缺口）；② "几何余量改善"预注册口径（H=6 窗口）；③ 后端分叉预案（CUDA 口径重抽取 vs 评测统一）。均已在 12A 落地。

### 4.2 执行与复现（零训练）

- **遥测**（Stage1）：逐 step 记录 gate_prob / activated / streak / residual（raw·scaled）/ base·final plan / tracker_ref / lane_lat / footprint / rc / speed / pose；新旋钮 `--residual-scale`、`--debounce`（N 连续步 > 阈值才激活）；产物 trace.npz + JSONL + manifest；5 eps 冒烟与旧版动作序列一致。
- **复现**（Stage2）：gate-off **0.3400/0.4933/0.0333**、gated **0.3067/0.4800/0.0267** 逐 episode 复现（13 翻负 / 8 翻正集合一致）。
- **CPU/CUDA**（Stage3）：全链探针（2,538 状态）——gate 概率 Δ p95 2.5e-4、**各阈值跨界率 ≤0.12%**；base plan Δ 重尾（p95 0.137 / max 1.07）；端到端 40 eps：CUDA 16/19/1 vs CPU 14/21/1（**6 集翻转**）、材料分离（|Δpos|>0.5 m）中位 40 env steps。
- **扫描**（Stage4）：15 配置（threshold ×5 → scale ×4 → debounce ×3 → 2×2×2 交叉），单轮 ~3.7 min。

### 4.3 九条件与三问（终判依据）

| tag | succ | off | coll | timeout | destr | ≥6/8 | ≥7/13 | 非孤立 | CPU/CUDA |
|---|---|---|---|---|---|---|---|---|---|
| t005_s100_d1（原） | .3067 | .4800 | .0267 | .2000 | ✓ | ✓8/8 | ✗0/13 | – | ✗ |
| t010_s075_d1 | .3333 | **.5067✗** | .0133 | .1467 | ✓ | 5/8✗（最好） | 9/13✓ | – | ✗ |
| t050_s050_d1 | .3467 | .4800 | .0133 | .1667 | ✓ | 2/8✗ | 11/13✓ | ✓ | ✗ |
| **t050_s050_d2** | **.3533** | **.4667** | .0200 | .1733 | ✓ | 1/8✗ | 12/13✓ | ✓ | ✗ |

- **绑定失败面**：①"保留 ≥6/8 历史翻正"——**全网格无人达标**（最好 5/8）；② CPU/CUDA 无实质分叉——全配置 False（6/40 集翻转）。
- **三问答案（证据）**：
  1. **gate 误报且无选择性**：闭环激活 16.1% 决策步 vs 离线 3.99%；**hist_8（被修好）17.07% ≈ hist_13（被弄坏）15.87% ≈ 其他 16.02%**；保守配置下**反选择**（t050_s050_d2：1.2% vs 4.7%；t010_s050_d1：7.8% vs 12.6%）。
  2. **residual 1.0 强度偏强**：@t050，s050 > s100（succ .3467 vs .3333、off .4800 vs .5067）；方向"**救余量、伤进度**"（timeout 23→30、均速 3.39→3.27；目标 ds 中位 −1.06=减速）。
  3. **连续触发 + 后端离散化放大**：激活段均值 4.43 步、**最长 55 连续决策步（27.5s）**；**2,336 个分叉后激活**（138/150 集）；后端：单步 gate 跨界 ≤0.12%，但 e2e **6/40 集结局改变**、材料分离中位 40 env steps ⇒ 阈值离散化把 ~1e-3 前向差放大成材料性分叉。
- **正面事实（记录，不改判）**：t050_s050_d2 的 **aggregate 反超基线**（+1.33pp succ / −2.67pp off / timeout +2pp 在门内）；但机制不可信（门无选择性、单 seed、净 +2 集），且按锁定条件判负。

### 4.4 口径修正与教训（全轮汇总，随档）

1. **"正常态激活 <5%"必须定义为闭环度量**（离线 3.99% → 闭环 16.1%，不可迁移）。
2. **destructive-FP 预注册口径结构性近空**：首激活即分叉因 → 每集 prefix 内仅 1 个窗口（13/13 判"改善"），其余 26–37 个激活在分叉后——若复活条件轨，改"post-divergence 净变化 / 段均时长×进度损失"。
3. **common-prefix 仅适合同设备对照**（CPU/CUDA 从第 1 步起 ~1e-3 差异）；跨设备用"材料分离步 + 结局计数"。
4. **条件间张力**："翻正集"由激进配置（t005）定义 ⇒ 保守配置天然丢失；若以净收益为目标应改条件表述（本轮按锁定规则判负，仅记录）。
5. **跨后端 ~15% 集级差**（CPU 采集 vs CUDA 部署；单步 ~1e-3 → 长时程放大）⇒ **后续任何条件监督 / 采集 / 校准统一 CUDA 口径或显式量化**。
6. **工程坑**（已修/已记）：npz 逐索引重复解压=内存爆炸（SIGKILL 根因）；专家相位 `prev_policy_action` 语义对齐；footprint 判定必须活 env；P@10 受负比率强影响（须带 n_pos/n_neg）；mode 标签邻步切换率 0.611（仅分层）。
7. **语境**：失败态上模型-专家偏离巨大（E：状态条件性）——与残差设计前提自洽，但 gate 的**选择性**才是瓶颈。

### 4.5 判定与行动

- **NO-GO（无有效区域）→ 按预注册关闭 gated residual 主线**：不补 hard negatives、不扩 gate、不调采样比例；全部产物保留于 `runs/p1_v2/round12a/`（遥测、探针、15 配置扫描、归因报告、九条件表）；零生产代码改动。
- **转 round-15（§5）**。

---

## 5. round-15→17 执行批次（Round-15A 与 A3；结果导向）

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

### 5.3 Round-15A 结果（全批次完成，2026-10-12 00:51）

**A1 · 当前动作分层审计（判定树 → ① 共变量偏移 → DAgger）**
- teacher-forced（canonical w>0，26,767 行）：MAE_ds **0.064 m** / MAE_dθ **0.0060 rad** / sign 错 3.1% / κ P90 0.0048；
- learner（intervention 836）：**0.406 m / 0.0986 rad（6.4×/16.5×）** / sign 错 **21.7%** / κ P90 0.111；
- ⇒ 模型在专家态准、自身态显著退化 = 共变量偏移；链/当前比 **2.79**（尾更差，但当前动作本身已差 6–16×）。
- Track B：唯一跨路型稳定 signed bias = **learner 态 ds（−0.33…−0.51 m/step，4/4 路型 |t|≥3）**；teacher 侧无、dθ/κ 不 systematic ⇒ bias_calib 若启只对 **ds+learner 态**。锚点逐位对账通过。

**A2 · 确定性尾族 oracle（10 新 run + 22 旧账；输入=仅当前专家动作）**

| 族 | 最佳 | 结论 |
|---|---|---|
| repeat（旧账 V1/V2/V3） | **V1 s130 = 0.778**（s125 0.776） | V1 ds×s 0.728（s110）≫ V2 0.676 ≈ V3 0.69 ⇒ 增益 = **"拉伸 ds 使参考曲率相对速度下降"（几何×速度耦合）** |
| decay 新族（lin/geo/θ-decay + v_ref/交叉） | 0.726（仅 dthetadecay_geo_v110 过 Go） | 全部不敌 repeat；**geo-decay 灾难**（0.000–0.042，LQR 参考被"饿死"）；V1×decay 交叉=0.282 亦毁 |
| 确定性复算 | v1/v2_s110_recheck **逐位一致** | 因果核验通过（v1_s100 旧账重算 ids/metrics_match=True） |

- **q1 天花板**：正确当前动作+简单尾 = **0.72–0.78（≈IDM 0.756 上方）**；**硬上限未触发**。
- **q2 LQR 稳定**：饱和最低 = dth_s085（0.043）→ dthetadecay_geo_v110/v3_s090（0.073）；V1 族饱和偏高（0.088–0.165）。
- **q4**：执行参考**无需学习六步链**即可达该上限带；学习链的价值仅在条件化策略（须先超过尾族基线）。

**Track C（上限分析）**：C1 union **0.790 < 0.82 → FAIL（不实现）**（救援 17/122；Wilson 上界 0.823；"union≠严格上界"已注记）；C2 早窗可预测性基本被静态信息解释（动态净增量≈0）、关键仲裁目标无信号。

**P1-3 FP 门**：**FAIL（5 项）** → `ld_margin_weight` 维持 0.0、几何反传维持禁止；根因 = LD 通道语义（需改**损失定义**、非调阈值）。

**综合结论（供 round-17）**
1. **失效假说（③ 待 A3 对照裁决）**：① 当前动作在自身态退化（共变量偏移，A1，已证）；② 执行参考无需学习远期链——**repeat 尾足够、V1 式 ds 拉伸是有效自由度**（A2，已证）；③"历史主要损失之一 = 让模型承担远期六步链拟合"= **高度可信但未证**——A1 的首步退化本身或已主导；**A3 同 encoder/数据/预算对照（旧链 vs 当前动作+尾）为裁决工具**（round-17 已并入批准范围）。
2. **A3/A4 放行条件（round-16 预注册）已满足**：A1 触发 DAgger 条件（teacher 低 ∧ learner 高）；A2 硬上限未触发、尾族可行、Go 通过者存在。
3. **A3 设计（按 round-16 骨架）**：当前动作 BC（步 0 主导、尾只担连续性/平滑）+ **repeat 尾**（V1 式 ds 拉伸列为动作变换候选轴）；无 encoder/WM/MoE 改动、无 recovery gate、无 geometry loss；1 seed；门：**clean150 ≥0.50 → eval500 → 0.60–0.65 才谈 RL**。
4. **A4 设计**：learner 态当前动作 DAgger（10–20% 混合、保 expert5k 防遗忘；采集格式照 round-16 锁定）。
5. **天花板提示（如实）**：特权动作+最佳尾 ≈0.778 ≈ IDM+2pp——**G3（0.80+）仍须超越教师**（RL 后置）或更优执行；A3/A4 的目标是让学习系统先逼近该特权带。

**请求（round-17）**：批准 **A3 开工（+A4 触发条件同理满足）**；确认 A2 四问判读；确认 decay 尾族关闭、`ld_margin_weight` 维持 0。

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
- **新认识（供 round-18）**：① **"动作可持性"是被忽视的独立属性**（专家天生 dθ≈0、可保持；模型不具备）；② 六步链监督可能主要承担**执行噪声缓冲**而非"远期预测"；③ learner 态数据缺口仍是第一瓶颈（A1 复跑再次确认）。
- **round-18 候选（未实施）**：**a.** 在允许损失范围内设计**可持性约束**（mu dθ 稳定性 / 与参考曲率一致）后**复用尾侧重跑 A3**（小成本）；**b. 重议 A4**——本轮门"A3<0.50⇒停"与"失败因 = learner 态缺口（恰为 A4 目标）"构成循环，请求按新证据重裁（契约已备）；**c.** 接受 chain 模式执行（其噪声缓冲已有实证价值），资源集中到 learner 态数据/监督；**d.** obs 表示改造（12A 已证现表示不可辨识）。
- **请求（round-18）**：裁决 a–d 优先级；确认 A3 判负与 A4 暂缓；确认生产改动提交（默认关 + 7 单测，`f8169fd`）。

### 5.6 round-18 往来残档（`95f23b0`；已并入执行）

- 接受 probe-first：两层根因框架（primary 动力学 shortcut + learner 态放大）、lanekeep 归 primary、chain 保留为执行缓冲并重定义——全部接受。
- 2 处事实核查：LD 7 维**已隐式含** lane-relative 几何（缺的是关联/选择/监督）；`ld_hist` 是输入（"只作输入"≠不进 memory）。
- 4 条落法与 3 验收补丁——**已并入 §5.8 执行与结果**。

### 5.7 round-19 往来残档（`1ea85dc`；已并入执行）

- **七项代码论断逐条核验=全部成立**（file:line 摘要）：`model.py:505-507`（PlanHead 只看 ld_pool 均值）、`spatial.py:103-106`（LD-LD=槽位链 ⇒ 伪边）、`mem.py:699-705`（LD 仅当前帧）、`model.py:627-632`（K-anchor lane_ctx=0）、`ld.py:209-260/137-170`（环优先排序 / ego.lane 优先）、注释漂移——**量化结果见 §5.8**。
- 接受 0A/0B/0C 前置与全部修正（全语义镜像 / 输出变化非充分 / Probe B 分层 / primary_only 严格旁路）；4 条实施路线（诊断重放 / 一致镜像近似 / 位级纪律 / 同帧配对）——**已执行（§5.8）**。

### 5.8 Round-18A 结果（2026-10-12 19:2x）：**拓扑/输入结构性缺陷 + primary 几何响应 ~50× 过弱（动力学主导）**

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
- **修复阶梯（先模型侧、后数据侧）**：**a. 模型侧（无重采）**：关闭/按语义重建 LD-LD 邻接（`spatial.py:103-106`）+ 显式池化替换全槽均值（`model.py:505-507`）+（可选）几何响应辅助监督——数据可复用、只需重训；**b. 数据侧（需重采）**：最小语义特征（target_lane flag/lane_id/offset idx/route assoc/侧偏/航向差/曲率）——fingerprint 变更、全量重采。0B 显示覆盖并非主缺口 ⇒ **先做 a、以 a 后复测决定是否上 b**。
- **坑（必带）**：① **模型消费 LD 走 `ld_hist[:, -1]`**——任何 LD 干预必须同时改该通道（否则假 0）；② 镜像不闭合=现象、结论待 env 级；③ 网格符号度量为口径 bug（真实符号由反事实均值证正确）；④ B 解码仅 curvature 可用；⑤ `a3/primary_only` 组合缺（3/4，模式同 pri512/primary_only）；⑥ 0C 非因果。
- **纪律**：零训练 ✓；net/pipeline 零改动 ✓（primary_only 复用 `set_moe(enabled=False)`）；`ld.py` 仅默认关旁路（fingerprint 不变、2 单测过）✓；已提交（见 git log）。

### 5.9 对 round-20 的回应（`37ee932`）：**无分歧、接受全部收紧与裁决；已开工**

**接受（要点）**：① 伪边 76.5% 是**架构缺陷证据、非闭环因果证据**——须以 M0/M1/M2 等预算消融裁决（认账）；② "~50×"改为**相对口径**（vs expert/oracle 所需纠偏敏感度，不给绝对倍数）（认账）；③ zero_history 为强 OOD 干预——改用温和对照（当前帧×6/逆序/shuffle/末 2 帧/LD-固定换 history/history-固定换 LD），关键对照="真六帧 vs 当前帧×6"；④ **M0 基线复训 / M1 仅关 LD-LD 链 / M2 关链+ego·nav 条件 LD set attention / M3 暂缓**——逐臂独立、同预算；⑤ 不立即做 target/adjacent/boundary 启发式分组（缺可靠 lane 身份，防新错配）；⑥ 数据侧重采 4 个触发条件；⑦ **20A 置换探针 + 20B matched-case**（因果闭环）；⑧ 验收门：结构门 + clean150 vs 同批 M0（+5pp/−10pp/coll+2pp/tmo+3pp/饱和降）+ eval500（0.45 有价值/0.55 进 DAgger/0.60 重评 RL）；**M2<0.45 → 停 LD 架构精修、转数据侧**（预注册）；⑨ A4/MoE/WM/禁制清单照锁。

**我方操作性补充（3 条，已并入施工）**：① **几何响应基准 = 同状态 expert/oracle 响应对照**——在重放环境置偏移并记录 expert mu_dθ 作基线（env 级为主、tensor 级辅助）；② **全部 LD 干预（含置换）须六帧一致**处理 `ld`/`ld_hist`/mask（防漏改 `ld_hist[:, -1]` 与跨帧错配）；③ **M2 落地=LD 完全绕过 spatial MP**（独立 token 编码 + ego/nav 条件 cross-attention）——比"仅关 LD-LD 边"更彻底隔离污染（与其"独立 token encoder"一致）。

**开工**：已派发 Round-20 lane（20A 置换探针 → 20B matched-case → M0 复训基线 → M1 关链 → M2 set-attention（含实现+训练）→ 结构门表 + clean150/eval500 门 → 判定；零新数据；产物 `runs/p1_v2/round20/`）。

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`（P1-v2 决策 + Round-12A 结果）、`docs/recovery_v2_contract.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`
- **报告**：`docs/p1_0_report.md`（P1-0 FINAL）、`docs/p1_2_diagnosis.md`（P1-2 FINAL）、`docs/recovery_v1_dataset_card.md`
- **数据**：`datasets/recovery_v1/`、`datasets/recovery_v2/`（samples / manifest / split / data_gate_report / _verify）
- **产物**：`runs/p1_v2/round12a/`（trace/探针/15 配置/归因/九条件）、`runs/p1_v2/round15a/`（A1 审计 / A2 尾族 oracle / Track C / P1-3 FP 门）、`runs/p1_v2/round17/`（两臂 ckpt / 三尾 clean150 / a3_report）、`runs/p1_v2/`（intervention_probe*、probe_robustness、probe_v2_check、bias_diagnostic、gated_residual_report、gated_residual_pilot/）、`runs/p1_0/`（d2_pacing_matrix、dth_sweep_summary、start_model_decision）、`runs/p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`（全链流水）、`.slim/deepwork/p1_v2/`（probe / collector / gated_residual / round12a 脚本 + logs）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮）
- **关键提交**：`37000ae`/`a371cc5`（过滤）、`e953f10`（P1-3）、`c20fa00`（P1-2 诊断）、`28acbcd`（dθ-only）、`1800e69`/`a42ac79`/`27ced1f`/`7b83730`（P1-v2 步）、`061bbac`/`663fedc`（argue §7/§8）、`5962848`（12A 开工）、`249c51c`（12A NO-GO）、`3b5124f`（尾族工具）、`dab85f8`（§5.3 批次结果）、`74f522f`（第三轮精简，位于 `Planner_RL_copy`）、`d70ec1b`（§5.4）、`f8169fd`（A3 配方开关）

---

## 7. 存档说明（历次精简 / 去哪找）

- **第五轮精简（本版）**：§5.4/5.6/5.7 往来压为残档；改前版本 = `git show 96df7cb:docs/deepseekv4p1_argue.md`（355 行版）。
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
  - 2026-10-11 深夜–10-12：round-15/16/17 讨论与执行（§5.1–5.7）：Round-15A 全批次 → **A3 执行/判负 → round-18**。
