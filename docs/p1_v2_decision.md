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
