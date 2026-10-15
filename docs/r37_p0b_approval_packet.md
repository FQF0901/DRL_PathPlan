# R37 → CG 审批包：P0-B（结构化 Policy 蒸馏 IDM+PP）与配套决策项

- 日期：2026-10-15 ｜ 分支 `Planner_RL` ｜ 提出方：执行侧（DeepSeek V4.1）
- **2026-10-15 跨邮补注**：本包发出后收到 GPT R38 方案（`docs/chatgpt5p6sol_review.md` `2d670cd`："全链路瓶颈裁决 + 代码减枝"，明确"本阶段不先训练"）。**D1 重定位为"裁决后放行"；D2 改为"按 Q4 `path_*` 分类裁决"（neutral 填充降级为诊断补丁）；D3 维持"由替换实验裁决"**；D4/D5/D6/D7 不变。细则见台账 §5.68。
- 依据文档：`docs/round37_report.md`（阶段 1 全量报告）、`docs/deepseekv4p1_argue.md` §0/§5.62–§5.66（决策台账）、`docs/repo_handbook.md`（仓库手册）
- 读数口径：全部闭环读数 = **calib50 开发集**（n=50，同协议、同 seed 集；±1 集 ≈ ±2pp，只作筛选）
- 纪律声明：**本包仅申请；未获批准前不训练、不改 WM 主体、不改 LQR**；生产改动一律默认关 + 逐位验证

---

## 0. 摘要与申请事项

**一句话**：R37 阶段 1 已用 7 条零训练 lane 把"结构化 Policy 为什么闭环差"定位到三层根因；零训练通道已穷尽（唯一有效者=接口修复 +8pp，非完整解）；**申请放行 P0-B：以 IDM+PP 规则专家为教师，蒸馏训练结构化 Policy（含 eval 装配一致化），随后评测回灌 Calib50 门槛链**。

| # | 申请/决策项 | 建议 | 影响面 |
|---|---|---|---|
| **D1** | **P0-B 训练放行**（教师=IDM+PP；结构化头+fusion.2+可选小适配器；eval 装配下训练） | **批准**（附 §4 条件） | 闭环性能主线；唯一剩余杠杆 |
| D2 | 接口修复固化形态（`path_dynamic_ttc` 缺口） | 先按 **eval 侧 flag（neutral）定稿为评测协议组件**；并行开小 lane 做 builder 真值回填 | 训练/评测装配一致性（不定稿会二次分叉） |
| D3 | C3（WM residual/想象步动力学）时机 | **排在 P0-B 首批评测之后**（C2≠C3 开关；由 5 条联合条件触发） | 坐标/WM 线 |
| D4 | 基准口径登记（本协议 C0 与历史 kappa-compose 分离；IDM 0.756 集来源核验） | 登记进台账 §0，禁止混比 | 所有后续数字可比性 |
| D5 | 新增考核项："起步-恢复"切片（v≈0/低速注入 → N 步恢复至场景限速量级） | 批准立项（工具 ~0.5 天） | 防止"spawn=6.0 m/s 掩盖起步缺陷"复发 |
| D6 | 训练放开清单（见 §2.4） | 批准"必须 18 键 + 可选小适配器"；**不放** WM 主干/编码器主体/Mem/旧 PolicyHead/MoE/ValueHead | 训练容量与风险边界 |
| D7 | **风险接受**：M8 试点被取消 ⇒ P0-B 前提（IDM 标签含加速信号、H2/I1 抽取直通）**未验证** | 接受，但**以 §4 的"第一步 fail-fast smoke"为放行条件** | 见 §4.1 |

---

## 1. 为什么要 P0-B（证据链，全部已核验）

### 1.1 三层根因（互相放大；详见 report §4）
1. **结构层（纵向保速自锁）**：`ds = DS_HIGH·σ(ds_raw + logit(clamp(v·dt)/DS_HIGH))`，纵向末层零初始化 ⇒ 零残差=**保持当前速度**；而 **LQR 参考速度 = plan 弦长/dt**（`env/tracking.py:498–504,537`）⇒ plan 几何即速度指令 ⇒ `v_ref≈v` ⇒ 油门≈0 ⇒ **掉速不可恢复**（自锁 ≈1.44 m/s）。实测 C1：`v[0..9]=6.0→4.09→2.74→1.86→1.49→1.21…`、throttle≈0、crawl 中位 52 s、timeout 20/50。对照：旧头（C0）plan 递增（2.69→3.73）⇒ v_ref 6.9–7.4 ⇒ 主动拉回速度。
2. **接口层（TTC 毒化）**：评测侧 `path_dynamic_ttc` 缺键→0（训练语义"无冲突"=30.6 s）⇒ 头读到"立即碰撞"⇒ 持续下小 ds；单去该键复现装配效应 **77%**；中性填充（17.7 s）零训练 **+8pp**（0.340→0.420）。
3. **横向层（@speed 质量）**：加速后 plan 曲率质量/抖动不足（|κ|p90 0.023→0.081、κ 抖动 ×4.7、|err_ψ| 0.017→0.241）⇒ 蛇形/越界（A1 off-road 0.44；cut-in 例 |err_y| 峰值 0.92 m vs C1 0.13 m）。
- **离线健康性**：头在训练分布内没坏（bias +0.056/MAE 0.149/r 0.983）⇒ 坍缩是**闭环（反馈）特有**，**不是坐标层/WM 的问题**（与 GPT R37 原假设不同：C1/C2/C3 四臂中的 C2 已验证 warp 机制生效但不修爬行）。

### 1.2 零训练通道已穷尽（result matrix，calib50 dev）
| 臂 | succ | off | coll | tmo | 结论 |
|---|---|---|---|---|---|
| **C0**（旧 Policy+旧 WM） | **0.580** | 0.180 | 0.120 | 0.140 | 本协议基线 |
| C1（结构化+旧 WM） | 0.340 | 0.120 | 0.140 | 0.400 | 爬行 |
| **C1_fill**（TTC 中性 17.7 s） | **0.420** | 0.120 | 0.100 | 0.360 | ✅ 零训练最优（+8pp，非完整解） |
| C1_true（三键真值） | 0.400 | **0.180**✗ | 0.080 | 0.340 | 更强提速但横向跟不上 |
| C2（+analytic warp） | 0.320 | 0.120 | 0.080 | 0.480 | 机制生效（coll −0.06），不修爬行 |
| A1/A2/A3（先验探针） | 0.46/0.38/0.34 | 0.44/0.28/0.24 ✗ | — | — | Hard 全恶化 |
| F3（v_ref×2） | 0.300 | 0.240✗ | 0.480✗ | 0.020 | 证伪"仅提参考速度" |

⇒ **规则性结论**：剩余缺口只能由**训练层**解决（纵向目标速度 + 横向 @speed 质量，须协同）。

### 1.3 教师选择：为什么是 IDM+PP
- 目标锚 = **IDM 0.756**（calib50 口径；ILI 上限审计最高可执行 d2=0.706）；
- DP-6 的 recorded 专家标签是**保速**的（65% 行贴 8.33 m/s 上限、各速段残差≈0、v<1 行链斜率弱 +0.31/6 步）⇒ 教会了"保持速度"，教不出"加速权限"；
- **IDM 是速度目标策略**（`v_des` + 加速指令），其 (ds,dθ) 标签自带"目标速度/加速"与"曲率-速度协同"；`tools/dagger_collect.py` 已就绪（学生状态 roll-in + 规则专家空问标签 + 解析逆映射 + **train-spec 硬隔离**）。

---

## 2. 技术方案（P0-B）

### 2.1 数据链（三件套）
1. **采集**：`tools/dagger_collect.py --ckpt runs/p1_v2/round28/K1/ckpt_epoch002.pt --specs env/specs/scenarios_train_slice200.json（建议起步）→ 后续 5k --labeler idm --tracker lqr --workers 4 --out datasets/BTC<ts>_dagger_idm`；roll-in 起步用 K1e2（工作驱动、状态分布正常），二期切结构化头学生做 DAgger 迭代；隔离校验由工具强制执行（与 eval500/val 池交集为空）。
2. **抽取**：H2 标签（`round36/h2_data_spec.py::label_row`）→ I1 抽取（`round36/i1_extract_dataset.py`）→ DP-6 同构分片。**已知改造点**：I1 的 `NPZ_PATH` 硬编码 expert5k_v8 ⇒ 需参数化（或等价包装）；`long_exog` 三键（static/dyn/TTC）**必须按训练侧同公式写入**（真值；M7 已给公式与核验方法）。
3. **训练**：`round36/j1_train.py`（allowlist/损失见 §2.4）+ **在 eval 装配（obs 指纹统一）下训练**（M3：`eval_on−train` 残余 Δds +0.138、融合 β·δ 2.72× 正交 ⇒ 否则差集残留）。

### 2.2 首期配置建议（待 lane 细化）
- allowlist：`path_head.`（16 张量）+ `plan_head.fusion.2.`（2）；**可选** head 前小适配器（若离线残差显示输入位移显著）。
- 损失：主 ds/κ（对专家当前步）×1.0；aux profile ×0.5；`--plan-res-coef 0.5`（terminal weight 3.0）；`--terminal-coef 3`；change 行 ×2。
- 优化：bs 64，lr 1e-3（fusion 分组另设 scale），1–2 ep 首期（T1/T2 经验：1 ep 早停 vs 2 ep 收敛/repeat 权衡）。
- **验收门（沿用 GPT 协议）**：Calib50 dev（succ ≥0.62 / off ≤0.10 / coll ≤0.08 / tmo ≤0.10 / rc 不降）→ Clean150 晋级 → Eval500 正式（3 seeds / paired bootstrap / McNemar vs IDM）。
- **新增离线门（官方口径补充）**：`ds(v)` 桶级保真 + **低 v 正残差占比**（防止再次学到"保速"）+ κ@speed 抖动上界——把 stage-1 的两个盲区纳入门。

### 2.3 预算与时间（粗估，含不确定标注）
- 采集：12 ep 试点 ≈ 10–15 min（4 workers，已由 M8 spec 验证可生成）；500–2000 ep ≈ 1–4 h（6 workers）。
- 抽取：分钟–小时级（IO 密集，脚本自管 workers）。
- 训练：R36 实测单 run（2000 步采样）分钟级；**完整 epoch（≈265k 行可用样本）需 lane 内实测**，粗估 0.5–2 h/epoch（单 GPU）。
- 评测：calib50 每臂 1.5–3 min；Clean150 约 ×3；Eval500 正式（3 seeds）≈ 1–2 h。
- 合计：**半天–2 天内可完成 P0-B 首批（含 Calib50 裁决）**（不含 Eval500 正式）。

### 2.4 训练放开清单（D6 的技术含义）
| 放开 | 冻结（本阶段） |
|---|---|
| `path_head.*`（横向/纵向/几何编码器） | WM 主干（`st_gnn.`）、`encoders.` 主体、`mem_encoder.` |
| `plan_head.fusion.2.` | 旧 `policy.`（structured 下不前向）、`value.`、`plan_head.moe.` |
| （可选）head 前小适配器（新增小模块） | 一切 C3 目标（residual/physical heads）留到 D3 |
| κ 限幅表（非参数常量）在闭环执行数据上重标定 | LQR、环境、评测协议 |

---

## 3. 风险与缓解

| # | 风险 | 等级 | 缓解（已内置到方案） |
|---|---|---|---|
| **R1** | **P0-B 前提未验证**（M8 取消：IDM 标签是否含加速信号、H2/I1 抽取是否直通未知） | **高** | **放行条件**：训练 lane 的**第一步 = 30–45 min fail-fast smoke**——①小规模采集（12 ep）后立刻统计 IDM 标签的 `ds(v)`/残差（要求低/中速桶正残差显著 > DP-6 基线）；②I1 抽取直通产出 DP-6 同构分片（字段/单位/坐标核验）。**任一不达标 → 停下回报，不进入训练**（避免白烧训练预算） |
| R2 | 训练/评测装配二次分叉 | 中 | 统一为"eval 装配 + 真值 long_exog 三键"；obs 指纹守卫硬失败（`stages.py`）；D2 定稿 |
| R3 | 蒸馏到 IDM 后"天花板=IDM"（0.756），不足以超越 | 中 | 阶段目标=先追平 C0/IDM；超越靠后续 C3（想象步动力学）+ 结构化表示红利；门链分级不混比 |
| R4 | 横向 @speed 质量仍不足（A1/M7 现象） | 中 | 训练目标含 κ 项 + κ 限幅表闭环重标定 + 离线门新增 κ@speed 上界；Calib50 若 off 仍 >0.10 则进入"横向专项"分支 |
| R5 | 数据隔离/泄漏 | 低 | `dagger_collect` 已硬校验 train 池与 eval500/val 交集为空（工具内 `SystemExit`） |
| R6 | 预算超支（全量 5k 采集 + Eval500） | 低 | 分级：slice200 → 5k；Calib50 快筛 → Clean150 → Eval500 |

---

## 4. 放行条件（建议写入批准意见）

1. **第一步必须 fail-fast**：P0-B 数据链 smoke（§3-R1）通过后，方可进入训练；不通过则回报并重议教师/数据。
2. **训练在 eval 装配下进行**（obs 指纹统一 + long_exog 三键真值；接口定稿见 D2）。
3. **每步评测留痕**：Calib50 全臂 paired 表 + 离线新门（低 v 正残差/κ@speed）；未过 Calib50 门不跑 Clean150/Eval500。
4. **纪律不变**：默认关 + 逐位验证；`.slim`/`runs` 不入库；生产改动随 966 tests + 契约门。

---

## 5. 附：口径登记与术语

- **口径**：本协议 C0 = 0.580/0.180/0.120（rc 0.872）与历史 kappa-compose 0.56/0.14/0.08 **不可直比**；历史锚点（legacy 0.40/Q1 0.50/K1 0.52/T2 0.58/统一基线 0.56/**IDM 0.756**）均为 calib50 口径。
- **术语**：`ds`=每决策步（0.5 s）推进距离；`v_ref`=LQR 参考速度=plan 弦长/dt；`neutral/true`=接口填充两档；`C0/C1/C2/C3`=GPT R37 四臂；`A1–A3/F3`=零训练诊断臂。
- **关键文件**：`docs/round37_report.md`（全量报告）、`.slim/deepwork/p1_v2/round37/{m2_c0c1,m3_report,m4_report,m5_report,m6_report,m7_report}.md`、`runs/round37/*`、`tools/dagger_collect.py`、`.slim/deepwork/p1_v2/a3/eval_a3.py`、`.slim/deepwork/p1_v2/round36/{j1_train,j1_eval,i1_extract_dataset,h2_data_spec}.py`
