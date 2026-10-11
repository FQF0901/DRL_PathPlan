# deepseek-v4.1 argue / 主文档（P1-v2 执行 · round-12 输入）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-11 晚）**：**P1-v2 三步执行完毕**——Step 1 探针（`no_increment` + F2m 特征重测过线）；Step 2 recovery_v2 同状态采集（六硬门全 PASS）；Step 3 gated residual pilot（**clean150 主门 FAIL −3.33pp，按预注册停止**）；**转 round-12 决策（§4）**。
> **沿革**：round-2 收拢（`b2b0f64`/`5fdbddc`）→ P0 执行+审计+排摸（§2.1/2.2）→ P1 计划 round-3–8 锁板（执行手册 v1，§2.4）→ P1 执行：P1-0/1/2/3（§2.3）→ round-9/10 P1-v2 定板与共识（§3.1）→ **P1-v2 执行（§3）**。
> **本版说明**：应要求精简（**1114 → 166 行**）。原 §0.9 收拢细节、§1–§4 早期草案、§5/§6 详表、§7 回合全文、§9–§11 原文、附录 A/B 已移除——全文索引见 §6 存档说明（git）。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **P1-v2 三步结果**：

| 步 | 结果 | 判定 |
|---|---|---|
| Step 1 可辨识性探针 | 正式 Gate 过（F1 0.934）；公平 progress 匹配 **0.682/0.731**（`no_increment`）；F2m 自车特征 **0.804/0.843** | 条件重开（Attn：特征级） |
| Step 2 recovery_v2 采集 | 2,693 行；**六硬门全 PASS**；v1 根因数值实证（k≥1 vs Pass-1 中位差 33.2） | PASS → Step 3 |
| Step 3 gated residual pilot | clean150 gated **0.3067 vs 基线 0.3400（−3.33pp）**；其余门全过 | **FAIL（停止）** |

- **旁线**：D（dθ-only 0.728 / +2.2pp，V1=组合效应确认）；E（常态 μ 无系统性 bias、③ 升级证据弱）。均不改训练 target。
- **当前问题主线**：闭环 gate 分布迁移过触发（16.1% vs 离线 3.99%）——机制已隔离，属**gate 数据覆盖/标定**问题，残差本身有效（8 例翻正、off/coll 降）。
- **约束**（持续有效）：plan 部署/评测协议；单 GPU；fail-fast；task-session 解耦 + 20 min 节拍监控；conda/python 只读；正式 run 不清理。

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

## 3. P1-v2 执行纪要（round-10/11 → 2026-10-11 晚）

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

### 3.6 问题清单与教训（本轮新增）

1. **离线↔闭环分布迁移**：gated 监督的新失效模式（离线操作点 3.99% → 闭环 16.1%）；"正常态激活 <5%"应定义为**闭环度量**。
2. **gate=0 监督源结构缺陷**：依赖活 env 的特征（footprint/inside_ratio）会限制控制态来源 → 补部署分布采集，或特征离线化。
3. **残差有效但被 gate 拖垮**：8 例翻正、off/coll 降；瓶颈隔离在 **gate 精度**（不否定残差设计）。
4. **数据语义**：CPU/CUDA 结局分歧（10/60 翻转）——数据自洽于 CPU；跨口径解读需谨慎。
5. **工程坑**（已修/已记）：npz 逐索引重复解压=内存爆炸（SIGKILL 根因）；专家相位 `prev_policy_action` 语义对齐；footprint 判定必须活 env；P@10 受负比率强影响（必须带 n_pos/n_neg）。
6. **语境**：失败态上模型-专家偏离巨大（E：状态条件性、residual=expert−base 大）——与 gated 残差设计前提自洽。

---

## 4. round-12 决策请求

- **现状定性**：P1-v2 在"当前形态"未过 pilot 主门；**失败机制已隔离**（gate 分布迁移过触发）；残差正效应存在但被掩盖；探针 / collector / 同状态基础设施全部就绪且经硬门验证。
- **候选方向（均未实施；按我方建议优先级）**：
  1. **gate=0 监督覆盖修复**：在部署同源分布（val_only150 等 spec）上采集控制行 + 失败起点行，**活 env 全特征**（直指已验证根因，采集成本 ~1 小时级）。
  2. **闭环操作点标定**：把"激活 <5%"改为**闭环度量**；用短闭环预跑标定阈值/分位数。
  3. **特征离线可计算化**：footprint/margin 由 obs+plan+地图离线推导（解除活 env 依赖，扩展数据源到 expert5k 等）。
  4. **反事实复盘**：8 翻正 vs 13 翻负的逐集激活模式（量化"gate 精度 ↔ 残差增益"，预估修复收益）。
- **请求裁决**：①–④ 的取舍/组合；是否批准"**数据覆盖修复 + 同架构重跑 pilot**"；P1-3 几何 FP 门是否平行推进；冻结清单是否维持。
- **提醒（不粉饰）**：追平/超越 IDM 的正面证据目前仍主要在 **无条件轨（V1 calibration oracle）**与 **RL 超教师（4B 门保持）**；P1-v2 条件轨遭遇的是**数据覆盖/标定级**问题（非架构级）——若一轮修复后仍不收敛，应认真评估切换主线。

---

## 5. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`（P1-v2 决策与三步结果）、`docs/recovery_v2_contract.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`
- **报告**：`docs/p1_0_report.md`（P1-0 FINAL）、`docs/p1_2_diagnosis.md`（P1-2 FINAL）、`docs/recovery_v1_dataset_card.md`
- **数据**：`datasets/recovery_v1/`、`datasets/recovery_v2/`（samples / manifest / split / data_gate_report / _verify）
- **产物**：`runs/p1_v2/`（intervention_probe*、probe_robustness、probe_v2_check、bias_diagnostic、gated_residual_report、gated_residual_pilot/）、`runs/p1_0/`（d2_pacing_matrix、dth_sweep_summary、start_model_decision）、`runs/p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`（全链流水）、`.slim/deepwork/p1_v2/`（probe / collector / gated_residual 脚本 + logs）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮）
- **关键提交**：`37000ae`（对齐过滤）、`a371cc5`（类别过滤）、`e953f10`（P1-3 损失）、`c20fa00`（P1-2 诊断）、`28acbcd`（dθ-only）、`1800e69`/`a42ac79`/`27ced1f`/`7b83730`（P1-v2 各步决策）

---

## 6. 存档说明（本版移除了什么 / 去哪找）

- **本版移除**：§0.9/0.9.5/0.9.6（收拢期细节与滚动开工记录）、§1–§4（早期共识草案与分歧清单，已全部消解）、§5/§6 详表（矩阵全表/归因细节）、§7.1–§7.11（round-3–8 讨论全文）、§9–§11 原文（P1 执行进展、round-9/10 回应）、附录 A/B（round-1/2 回应全文）。
- **全文位置**：`git show 7b83730:docs/deepseekv4p1_argue.md`（1114 行版）；更早各版见对应提交（`408b2a1`/`d5e85ff`/`9f611cf`…）。
- **压缩时间线**（一句话历史）：
  - 2026-10-09：round-1/2 收拢（`b2b0f64`/`5fdbddc`）；《评测兼容设计》落 `docs/p0_eval_compat_design.md`；
  - 2026-10-09/10：P0 三件套（fail-fast / 兼容 runner / 审计工具）→ P0-B 矩阵 + P0-C 归因 → 审计定稿（§2.1）；
  - 2026-10-10：排摸收官（§2.2）；round-3–8 讨论与 P1 计划锁板（执行手册 v1，§2.4）；
  - 2026-10-10→11：P1 执行——P1-0 起点/配速（V1 发现）、P1-1 采集、P1-2 崩塌诊断、P1-3 实现（§2.3）；
  - 2026-10-11：round-9/10 定板（§3.1）→ P1-v2 三步执行（§3.2–3.4）+ 旁线 D/E（§3.5）；
  - 2026-10-11 晚：本文件精简；**round-12 决策（§4）**。
