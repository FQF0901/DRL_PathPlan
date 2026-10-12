# deepseek-v4.1 argue / 主文档（Round-17 A3 判负 · round-18 输入）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-12 上午）**：P1-v2 条件轨已关闭（Round-12A NO-GO）；Round-15A 全批次完成（§5.3）；round-17 A3 判负（§5.5）；**round-18 锁定 probe-first——Round-18A 四反事实 probe（lane-grounding + shortcut；零训练级），A4/特征改造/A3 重跑全后置（§5.6）**；RL 冻结。
> **沿革**：round-2 收拢 → P0 审计 + 排摸（§2.1/2.2）→ P1 计划 round-3–8 锁板（§2.4）→ P1 执行（§2.3）→ round-9/10 P1-v2 定板与共识（§3）→ round-12/13/14 讨论（§4.1）→ **Round-12A 诊断与关闭（§4）** → round-15/16 讨论（§5.1–5.2）→ Round-15A 全批次完成（§5.3）→ round-17：A3 执行/判负（§5.4–5.5）→ round-18：probe-first 锁定（§5.6）。
> **本版说明**：第四轮精简（299 → ~285 行）：§5.2–5.4 往来压为补充存档、§5 标题对齐范围；此前已合入第三轮精简（原 `74f522f`@`Planner_RL_copy`）。历版全文见 §7。

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

### 5.4 对 round-17 的回应（`732ba96`）：**无异议、开工**——附 4 条落法澄清

- **接受其反驳（认账）**：§5.3-③"远期六步链拟合是历史主要损失之一"**降为待证假说**——A1 表明首步共变量偏移本身或已主导闭环退化；裁决工具 = **A3 同 encoder/数据/预算对照**（旧链监督 vs 当前动作+fixed tail），已并入批准范围。
- **落法 1（训练与尾选择解耦）**：训练一次（`L = L_current_action + λ_smooth·L_tail_smooth + λ_curv·L_delta_curvature`；**禁止**未来 5 步 regression / 未来专家态 traj target / 合成 traj6 / recovery traj loss；曲率项只做平滑、不得启用 `ld_line_margin_loss`）；推理期扫 **A3-R(1.00) / A3-V(1.15, 1.25)**（clean150 选向、**禁止扩扫**）；正则项按"构造尾链"计算时固定规范尾（或改对相邻动作平滑）并文档声明。
- **落法 2（隔离与一致性）**：当前动作由**单一 head** 输出；WM/MoE 不参与尾部；tail=**纯函数**；**LQR reference 与导出 plan 逐位一致**（复用既有等价口径：单步逐位 + 整集 flags/终止全等 + |Δrc|≤1e-3）。
- **落法 3（对照臂）**：标准六步链监督（1.00/0.80/0.60/0.40/0.25/0.15 权重），同 encoder/数据/预算；建议两臂从同一初始 ckpt（pri512@e020）匹配微调，留档理由。
- **落法 4（A4 仅条件、契约照锁）**：数据从 **A3 自身闭环状态**采集（CUDA 口径）；行 = learner obs/history、learner 预测动作、expert 当前动作、road class、speed/curvature、episode outcome；**不采**六步链 / gate / 绝对 traj / Pass-2 plan；混合 expert5k:A4 = 90:10 起、≤80:20。
- **验收照锁**（A3 clean150：succ ≥0.50、off ≤0.35、coll ≤0.05、timeout ≤0.15、clipping<5%、steer 饱和不高于 A2 同尾族明显水平、teacher MAE 不恶化、learner 动作误差 ↓≥25%、plan/LQR 逐位一致；**<0.50 不做 A4**；A4 门=+5pp succ 等；eval500 ≥0.60 → 才谈 RL）。
- **开工**：已派发 lane——A3-R/V + 对照臂（两臂训练）× 三尾 clean150 + 完整 probes + A1 审计复跑。

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

### 5.6 对 round-18 的分析与回应（`95f23b0`）：**接受 probe-first；附 2 处事实核查 + 4 条落法**

**接受（新框架与裁决）**
1. **两层根因框架接受**：① primary 可能主学"expert 态动力学延续 shortcut"（prev_action+速度/航向趋势→下一动作；memory 可读 ego 历史为 shortcut 提供条件），未稳握 ego-lane 关系；② 部署入 learner 态后 covariate shift 放大该缺陷 → dθ 符号/幅度失控。与 A3 同 ckpt 隔离、A1 learner 崩坏、12A"gate 无选择性"全部相容。
2. **lanekeep 归 primary / specific experts 只做条件残差 / router 需语义监督**——接受（"正常直道 lanekeep 依赖 expert residual"=职责倒置的连接成立）。
3. **候选裁决 a–d 全部接受**：a 先 probe（认账"可持性正则可把 dθ 压向 0、掩盖 lane 缺失"的风险）；b A4 晚于 probe；c **chain 保留为执行噪声缓冲并重定义**（"可跟踪平滑 reference"而非精确预测未来）；d 先诊断、后最小显式特征。
4. **Round-18A（四 probe + 7 条验收）**——接受为下一步；零训练、纯前向级成本。

**两处事实核查（以 repo 为准）**
1. **LD 结构**：`ld (16,7)`＝`[dx, dy, heading_rel, curvature, speed_limit, left/right_line_type]`、offset `{5,10,15,20,30}m` ✓；**其 7 维已隐式含 lane-relative 几何**（每槽 dx/dy/heading_rel/curvature 可重建侧偏/航向差/曲率）——缺的更多是**显式关联与选择**（哪条是目标中心线、统一口径）与**监督约束**，而非几何"不存在"。此细化令 Probe B 的两分法（latent 可解码但 action 不响应 → head/监督）更关键。
2. **"LD 未进主要时序"表述需修正**：`net/mem.py` 明列 `ld_hist (B,6,16,7)` 为输入（标注"只作输入"＝不作 WM 预测目标）——准确表述 = **LD 历史是输入、但不参与 WM 预测目标**（融合深度待 probe/静态核查）。

**落法补充（4 条）**
1. **Probe 实施规范**：统一固定状态集（expert5k val ≥5k 行 + learner 源 ≥5k 行；可另采 A3 策略若干 episode 状态）；LD 反事实须联动 slot 语义与 mask（镜像/平移需处理 current-lane-first 排序）；**Probe D 可归并进 Probe A 的平移/旋转网格**（同实现 + 符号/单调性统计）；**两 ckpt（pri512 与 A3 臂）同跑对照**（微调是否改变 grounding）。
2. **决策映射预注册（含"部分带"）**：全过（含 #5 LD 敏感）→ A4 + chain 缓冲；**#5 失败 → 表示/特征优先**；**部分带**（大误差符号 ≥95% 但小误差失败、或 LD 敏感但弱）→ 最小显式特征 + primary-only 监督、后 DAgger——按误差分箱（±0.25/±0.5/±1 m；±2.5°/±5°/±10°）报告，防一刀切误裁。
3. **"最小显式特征"两条实施建议**：a) 先核验这些量可否由现存 obs 直接算（能：侧偏/航向差可自 pose+LD 推导）——改造可走"显式通道"或**"latent 辅助监督"**（把 Probe B 标签作辅助损失、不改观测接口，中间成本档）；b) 新监督信号必须用 probe 验证过的语义（防重蹈 `ld_line_margin` 代理 FP）。
4. **chain 重定义清单排序**：接受方向；但清单中 lane direction / boundary·corridor safety 依赖"可信 lane 语义/可行域信号"——其实现应**后于** probe/特征路线（ld_margin 家族已 FAIL 先例）。

**验收 7 条接受 + 3 条补丁**：① 符号/单调性按误差分箱报告；② #7（experts 关）需小适配（router 强制 primary 的推理开关）；③ 同一 probe 集在两 ckpt 上跑（含 A3 臂）。

**请求（round-19）**：确认 Round-18A 开工范围（四 probe + 决策映射 + 补丁；零训练）；确认 A4 / 特征改造 / A3 重跑均后置。

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

- **第四轮精简（本版）**：§5.2–5.4（round-15/16 往来）压为 §5.2 补充存档、§5 标题对齐；改前版本 = `git show 58ac4c3:docs/deepseekv4p1_argue.md`（299 行版）。
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
