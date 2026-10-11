# P1-v2 决策记录（P1-v2: LOCKED）

> 达成：2026-10-11（round-9/10/11 双方共识）。依据：`docs/deepseekv4p1_argue.md` §10/§11、
> `docs/p1_2_diagnosis.md`、`docs/chatgpt5p6sol_review.md`（`1ac2e30`）。
> 本文件 = P1-v2 的日常决策载体（argue 文档回归历史审计）。

## 被证伪的假设（本阶段）

1. recovery_v1 混采（任意剂量/对齐/类别/冻结）可提升闭环 —— **证伪**（四轴穷举全败；详见 `docs/p1_2_diagnosis.md`）。
2. "绝对目标监督 + 状态可辨识"默认成立 —— **证伪**（受控探针：恢复行为在低速带被条件化重写；正常/失败态在 obs 上不可分）。
3. 剂量/清洗类搜索有边际信息 —— **证伪**（0.25/0.05/0.02 × 对齐 × 类别组合均崩）。
4. V1（ds 拉伸）可作教师上限证据并开 4B 门 —— **修正**（同时改变曲率；改列 calibration oracle）。

## 保留的证据

- P1-0：起点 pri512@e020；V1 峰值 0.778@1.30（timeout 0.236→0.086；vs s100 z=4.10）；V3<0.75；等价操作口径；引擎 run-to-run 随机性地板 0.2–1%。
- P1-1：重放确定性 strict 全覆盖；数据门 13/13（数量/覆盖/质量）；反事实错位 9,377 行留档为勘误样本。
- P1-2：A2（仅配方）0.30/0.31 稳定；两层根因链完整证据（批次级取证 + 受控离线探针）。
- P1-3：三损失实现（默认关、逐位不变）；traj×geometry cos=−0.73。
- 工具与纪律：fail-fast、探针体系、进程同步/GL wrapper/内存门纪律。

## 停止项（本轮不做）

recovery_v1 再训练；新 mix ratio / freeze scope / 子集组合；无 gate 的直接残差；未过 FP 门的 geometry 联合训练；
P1-4A/4B；K-anchor；GRPO/Best-of-G；WM/MoE 重构；容量扫描。

## 下一假设（P1-v2 主线）

"当前 observation 能否识别'此刻需要恢复'"可被证明/否证；若可（ROC-AUC ≥0.80 且 precision@10% ≥0.60），
则 gated residual（`final = base + gate(obs,base) × residual(obs,base)`）以严格停止条件推进小 pilot；
否则转 observation 改造或全局策略优化。

## Gate（三步，逐级 Go/No-Go，不自动晋级）

1. **可辨识性 Gate**：ROC-AUC ≥0.80 且 precision@10% ≥0.60；报告必须含 PR-AUC、2–4 m/s 子集、
   tollgate/merge/roundabout 分报、**matched-negative 对照**（同 episode 时间邻窗负样本）。
   总体过但关键切片失败 → 仅"有条件通过"，不得直接进入全量 gated residual。
2. **数据 Gate**（recovery_v2 pilot 2–3k 行）：同状态匹配 100%；随机重放 ≥1,000 行；episode split 无泄漏；
   expert 恢复率 ≥80%；任一失败 → 先修 collector，不训练。
3. **Pilot Gate**（gated residual，clean150）：success ≥ baseline−2pp；off 不增；coll ≤+2pp；
   正常态 gate 激活 <5%；intervention recall ≥60%；首步 MAE/6 点 ADE 恶化 ≤10%；第二 epoch 不塌。
   **不通过即停，不追加复杂度**（不加更大 gate / 更复杂 MoE / 更多过采样）。

## Step 1 结果（2026-10-11 晚）：conditional_pass

- **正式 Gate：通过**。F1（当前 obs）matched_natural ROC-AUC **0.934**（LR）/0.933（MLP）、P@10 **0.943**/0.909；cross 0.981/0.984；关键切片 2–4 m/s **0.899/1.000**、tollgate 0.987、merge 0.965、roundabout 0.899 → 无失败切片。
- **防泄漏自检 → 有条件**：time-only **0.940 ≥ F1**（全集）；时间邻近负样本 F1 0.906 vs time-only 0.909；2–4 m/s 同（0.885/0.895）；同格（60 步）F1 0.895（唯一"独走"，P@10 0.578 @8.2:1）。F1 头部系数 = route_completion / 横向偏移。匹配负样本构造性全早期（step 5–75 vs 正 55–490；60/60 集早期现计划失效）→"晚段健康态"负样本缺失。
- **诚实上界**：F2∩/F3∩ 0.969/0.964（F2/F3≈1.0 属与标签定义重叠的构造循环，不作证据）。
- **判定**：正式过门 + **净增量未证** ⇒ 不进全量 gated residual；先做 **Step 1b 负样本增广复核**（晚段健康态 + progress 匹配对照；判据：progress 匹配集 F1 ≥0.80 且相对 time/confound 净增量 ≥0.05 → 确认进入 Step 2；<0.05 维持存疑；<0.80 按现状不成立）。产物：`runs/p1_v2/probe_robustness.json` + 更新 `final_verdict`。
- 数据语义坑（随档）：CPU 重放与 CUDA 评测结局 50/60 一致（正/负标签自洽于 CPU；"成功段"负样本实为早期段）；matched_speedmatched 退化（713 < 1660）；P@10 比率敏感（报告必须带 n_pos/n_neg）；SIGKILL 根因 = npz 逐索引重复解压（已修，教训归档）。

### Step 1b/1c 终判（18:20）：F1-only `no_increment`；特征重测过线 → **Step 2 进入**

- **决定性 progress 匹配集**（rc±ε + 速度带 + 路型，跨 episode，时间不参与）：F1 AUC **0.682(LR)/0.731(MLP)**、净增量 +0.177（time-only 0.490≈chance ⇒ 匹配生效）。按预注册规则 → **`no_increment`**：F1-only 可辨识性不成立（Step 1a 的 0.934 系进度/时间混淆）。弱档：**prevention 0.633/0.712**（计划仍有效、margin 退化——pilot gate 单独验）。
- **特征重测（按 ChatGPT 带结构 0.65–0.80="加 margin/plan 特征重测"）**：同集 **F2m（自车可算：lane_lat + 自身 plan footprint/inside_ratio + 历史，无锚点 oracle）= 0.804(LR)/0.843(MLP)** ≥0.80 ⇒ 路径重开。注记：F2/F3 的 0.996 属定义性 oracle 循环（d_*/is_near_anchor），**不作证据**；F2m 仍含负样本选择用到的 footprint/inside_ratio（部分选择重叠）——**pilot 的正常态激活<5% / recall≥60% 为最终仲裁**。
- 晚段健康负样本 2,781 条 / 51 集（免重放挖出）；残余混淆：其历史通道或含近期失败痕迹（已注记）。
- **决定：Step 2（recovery_v2 collector）GO**——同状态契约 + control 行（晚段/高 rc）；数据门为硬门（任一失败先修 collector、不训练）；gate 设计须含 plan/margin 派生特征（`gate(obs, base_plan)` 架构内）；数据门通过后以 v2 数据复测干净特征集（F2i/F2m），供 Step 3 gate 设计输入。

### Step 2 结果（19:10，已独立抽验）：六门全 PASS → **Step 3 进入**

- pilot = **2,693 行 / 110 episodes**（takeover 1,292 + control 1,401；labeled intervention 836＝correction 660/prevention 400/late 232；232 windows；expert recovery **89.2%**）。
- **六项硬门全 PASS**：①同状态 100%（全行离线重算 max_plan_diff 6.7e-06、0 失败 + 1,216 行抽样重放 0 失配）②重放 ≥1,000（1,216 行，含 471 k≥1；obs/pose/baseline_plan max diff 0.0）③split 88/22 无泄漏 ④recovery 89.2% ≥80% ⑤action-chain 一致 0 失败 ⑥control 覆盖（1.08 比、晚段 75.2%、rc≥0.5 36%）。
- **v1 根因正面实证**：k≥1 行存 Pass-2 同状态 obs（vs Pass-B 逐位 0.0；vs Pass-1 同 step 中位差 33.2）。
- **probe_v2_check**：干净特征集 **F2m 在 v2 上 AUC 0.87–0.92（natural）/ 0.867–0.895（progress 1:1）**、净增量 +0.33~+0.39；F1 fair（progress 1:1）0.83/0.84。
- 注记：mode 邻步切换率 0.611 ⇒ 仅分层（不得作训练目标）；难度偏 easy（56%，分层时注意）。
- **决定：Step 3（gated residual pilot）GO**——架构 `final = base + gate(obs, base_plan) × residual(obs, base_plan)`；**base = pri512@e020 冻结锚定**；gate 输入须含 F2m 类自车特征（**禁锚点 oracle 列**）；验收按 round-10（正常态激活 <5%、intervention recall ≥60%、首步 MAE/6 点 ADE 退化 ≤10%、clean150 ≥base−2pp、off 不增、coll ≤+2pp、二 epoch 不塌）；**不通过即停、不追加复杂度**。

### Step 3 结果（19:31–19:35，已独立复核）：**clean150 主门 FAIL（−3.33pp）→ 按预注册停止**

- Stage A/B1/B2 全过：5/5 单测（gate=0⇒final==base **逐位**）；训练 gate AUC 0.870（二 epoch 不塌）；离线：正常行激活 **3.99%<5%**、recall **61.1%≥60%**、MAE/ADE 无退化（intervention 行还改善）；数据纪律记录（raw 31.3%/exposure 3%/eir 0.096，无 >10× 过采样）。
- **clean150 闭环**（`specs_val_only150` + lqr/plan-reference，6 workers）：gate-off 对照**逐位复现官方基线**（0.3400/0.4933/0.0333）；gated（阈值 0.05）succ **0.3067（−3.33pp < −2pp 门）**、off 0.4800（−1.33pp）、coll 0.0267（−0.67pp）；配对 **13 成功→失败 vs 8 失败→成功**（净 −5）；timeout 23→30；均速 3.39→3.27。
- **机制（已验证）**：闭环 gate 激活率 **16.1% 步**（p95 47%、142/150 集）vs 离线 3.99% ⇒ **部署分布漂移下 gate 过触发**，把"减速+回正"残差施加到正常态。根因数据面：F2m 的 footprint 特征需活 env ⇒ gate=0 监督只能取 v2 control（覆盖窄）。
- **协议合规**：未调参/未加复杂度/未做过采样调整（`stop_rule` 留档）；残差本身有增益面（8 例翻正、off/coll 降）——瓶颈在 **gate 精度**。
- **下一步：停在本步，转 round-12 决策**。候选方向（未实施）：① gate=0 监督覆盖修复（部署分布 + 活 env 特征）；② "激活<5%"明确为**闭环度量**并做闭环操作点标定；③ footprint/margin 特征离线可计算化（扩大数据源）；④ 补做"gate 精度 ↔ 残差增益"反事实分析（8 翻正 vs 13 翻负的激活模式）。

### Round-12A 结果（2026-10-11 21:36）：**NO-GO（无有效区域）→ 关闭 gated residual 主线**

- 15 配置（阈值/scale/debounce/2×2×2 交叉）**无一通过九条件**（`runs/p1_v2/round12a/{attribution_report.md, go_no_go.json}`）；绑定失败面：①"保留 ≥6/8 历史翻正"全网格无人达标（最好 5/8）；② CPU/CUDA 无实质分叉全配置 False（40 eps 子集 6 集翻转）。
- **三问答案**：① gate 误报且**无选择性**（闭环激活 16.1% 步 vs 离线 3.99%；hist_8 17.07% ≈ hist_13 15.87% ≈ 其他 16.02%；保守配置下**反选择**：t050_s050_d2 1.2% vs 4.7%）；② **residual 1.0 强度偏强**（@t050 s050>s100），方向"救余量、伤进度"（timeout 23→30、均速 −0.12；目标 ds 中位 −1.06）；③ **连续触发普遍**（最长 55 连续决策步、2,336 个分叉后激活）＋**后端离散化放大**（单步跨界 ≤0.12%，e2e 6/40 集翻转，材料分离中位 40 env steps）。
- 记录（不改判）：**t050_s050_d2 aggregate 反超基线**（succ .3533 / off .4667 / coll .02 / timeout .1733——四率门+destr+非孤立+消除 12/13 全过；唯"保留翻正 1/8"+CPU/CUDA 未过）。
- **行动**：按 round-14 预授权**关闭 gated residual 主线**——不补 hard negatives、不扩 gate、不调采样比例；全部产物保留；**转 round-15 定 fallback**（协议所列：V1 calibration 可部署分解 / action·plan 纵向尺度校准 / RL 合法信用链〔4B 门保持〕）。
- 口径修正（供后续引用）：destructive-FP 预注册口径**结构性近空**（首激活即分叉因，prefix 内每集仅 1 窗口）；common-prefix 仅适合同设备对照；"翻正集"由激进配置（t005）定义（条件张力已记录）；**跨后端 ~15% 集级差** ⇒ 后续数据采集/校准/评测统一 CUDA 口径或显式量化。

## 并行诊断旁线（不得直接改训练 target）

1. **dθ-only 反事实**（补 ④ 分解缺口；①②已完成：V1 0.776–0.778@1.25–1.30 vs V2 0.658–0.676@1.15）；
2. **phase3_bias_calibration 离线诊断**（既有机制：signed-bias 惩罚，权重默认 0、指标常记）；
3. **Δκ / progress 监控**。

升级条件：bias 修正在不同速度带 / 道路类型 / 困难场景**均稳定改善**，才可升为训练候选。

## 冻结（全阶段）

recovery_v1 训练；P1-4；K-anchor；GRPO/Best-of-G；WM/MoE 重构；容量扫描；未验证 geometry 联合训练。

## 产物

- `runs/p1_v2/intervention_probe.json` + `runs/p1_v2/intervention_probe_by_slice.json`（Step 1）；
- `docs/recovery_v2_contract.md`（Step 2 契约）；
- 后续 pilot 产物按统一 11 项。
