# deepseek-v4.1 argue / 主文档（**R41 当期快照**）

> **用法（用户口径）**：本文件按**当期快照**维护——每开新一轮**整篇重写**；旧轮走 git：`git log --oneline -- docs/deepseekv4p1_argue.md` ／ `git show <commit>:docs/deepseekv4p1_argue.md`。
> **配套**：`chatgpt5p6sol_review.md`（GPT 规格/裁决原文）、`repo_handbook.md`（架构/训练/门/附录 D 归档指针）、`README.md`（索引+卫生规则）、`architecture_contract.md`。
> **GPT 规格链**：R39 `647f909`+`fe5de55`（✅ 已执行：工程闭合/统一 WorldState）｜R40 `c416095`（✅ 已执行：验证批 V1–V8）｜**R41 `e8e9c7a`（当前执行规格）**：证据链正式化 → 横向瓶颈定位 → 统一 WorldState → Memory/detach；§六 learner-state 后置；§七 probes；§八 验收。

## 0. 目标与基线

- **总目标**：Eval500（3 seeds+bootstrap+McNemar）**success > IDM+Pure Pursuit** 且 coll/off 不劣；三级集禁混比：Calib50（筛选，**不得作收益声明**）／Clean500（晋级）／Eval500（正式）。
- **研究基线**：`ws_learned` = Calib50 **0.56 / tmo 0.06 / coll 0.18 / off 0.20 / rc 0.811**（生产默认仍 `world_state_api=off`）。
- **R41 门（取代 R40 的离线 c1/c2 硬门）**：**横向门**（vs ws_learned：帧间曲率符号翻转率 **↓≥50%**、steer 饱和 **↓≥40%**、高速横向误差不恶化、连续 footprint 越界下降、coll ≪0.18、off ≪0.20）｜**Calib50 筛选**：succ ≥0.56 ∧ tmo ≤0.06 ∧ coll ≤0.08 ∧ off ≤0.18 ∧ rc ≥0.815｜**Clean500 晋级**：vs C1_HEAD succ +5pp、coll/off 不劣、tmo 下降、rc 不降｜**工程门**：干净 clone 可跑 residual 训练、核心测试不得因 `.slim` 缺失而 skip、oracle/速度干预有正式 CLI 与模型接线、几何模式入 protocol hash、h1 同 batching max diff=0。
- **参数门**：`total ≤ 2,000,000`（五栏 total/reachable/trainable/frozen/dormant）；**`ACTIVE_LIMIT=1.05` 仍参与断言（代码事实）⇒ 按 R41 要么单独获批、要么降为警告（待裁）**。
- **未过 Clean500 安全门前**：不得设 `analytic_learned` 为默认；不得启动 MoE/DAgger/PPO/GRPO。

## 1. R40 验证批结论（已执行；R41 据此取舍）

1. **五猜想判定**：#4 特征同源 ✅（250,622 全量逐位 `max_abs_diff=0`）｜#2 双链漂移 ❌ 否证（数值精度内一致；下沉单源化=整洁性）｜#3 fixed 几何混杂 ✅ 成立但**方向为负**（离线 c1 −0.471；闭环 off+fixed coll 0.08→0.16；V6：闭环 footprint 对几何轴不可分辨，唯一活跃差=margin 符号轴，剩余差归噪声 p=0.45）｜#1/#5 ❌ 闭环否证（oracle 物理状态替换不改善 Hard；**ego oracle 撤销纵向增益** ⇒ 增益依赖状态伪影；横向 oracle 以碰撞换越界 Δoff +0.48 p=1e-6；降速不去碰撞但**实际操纵量弱**〔计划弦速 6.8 ≫ 实驶 3.5〕）。
2. **根因定位（V8 直接机制）**：命令-实驶落差 `ref−v=3.65` = 预览/尾段膨胀 3.26 + 一等步落差 0.39；**主因=横向执行耗损**——全油门窗口 **99.1% 处于 max|steer|≥0.7**，净加速 **+0.03 m/s²**（干净转向档 1.54–2.82）；**计划一等步横向命令 5× 过量、帧间 75% 变号（直道亦然）**；去耗损 sim **v→7.17** ⇒ **κ 抖动+过量 → 转向饱和 → 吃掉纵向预算 → 低速锁死+横向误差/碰撞/越界**。
3. **R41 对"V4B 决定性否证"的裁定**：**先补正式实现与可复现证据**（不接受结论本身）。**我方接受**——三项工程缺口已代码证实：①`pipeline/eval_runner.py` 的 `teacher_force/inject_truth_sha16/speed_scale/speed_oracle` **仅 protocol hash 占位、无执行路径**（`net/model.py` 零注入引用；实注入在 `.slim` harness 运行时补丁）②`tools/train_residual.py` **硬 import** `.slim/.../a1_matrix.py`+`j36/j1_train.py` ③几何模式为**进程级常量+环境变量**（`RLP_READOUT_LANE_GEOMETRY`），无模型实例配置。**同时保留我方对结论范围的限定**：V4B 为自教师两遍法、n=50、弱速度操纵、C0/C1 取自修前版本（默认路径逐位已证）——正式复现须按 §2.1 增强后才算定案。

## 2. R41 受理 + 我方推敲（分歧/细节，供裁）

**受理**：四阶段全盘（§二 证据链正式化 ≠ 训练主模型；§三 横向瓶颈；§四 统一 WorldState〔**验收=状态一致性，不预设安全收益**〕；§五 Memory/detach；§六 learner-state 后置）；§七 六类 probes；§八 门与纪律。

1. **teacher-forcing 的"动作条件一致性"（§七.1 auto-reject 规则的落实口径）**：V4B 注入帧来自**上一遍（同臂）动作序列**，与"学生 action + 专家 future"不同，但仍非当遍动作条件。**建议正式实现**：第二遍**用记录动作序列驱动 env 重放**并注入记录状态（动作条件严格一致）；或 self-teacher 迭代至不动点。**该口径决定 §三 H/C 臂证据的合法性**（请裁）。
2. **`L_replan` 过平滑风险**：跨帧重投影一致性会惩罚一切变化，可能对抗合理重规划（避障场景**应当**变）⇒ 建议**按状态变化门控/加权**（仅惩罚超出状态隐含变化的残差），并把 tmo/coll 纳入回归监控（防"僵直"）。
3. **标签可得性已核（利好，§三.2 可行）**：`path_exec_v1.action=(N,6,2)`（**专家 6 步计划**）+ `s_anchor/anchor_xy/anchor_off`（30 点路径锚）+ `lat_err/head_err`（30 维跟踪误差）+ `valid30` ⇒ κ_near/mid/far、横向偏移、航向误差监督**可直接派生，无需扩展采集**。
4. **§三.4 LQR 可执行性监督的实现代价**：训练环内可微 LQR 建议**先做非梯度的 batch 重加权指标**（预计 steer/饱和/跟踪误差/速度损失），收益确认后再考虑辅助 loss。
5. **§四.3 底层状态 residual 的预算/最小实现**：structured 1,510,570 ⇒ **余量 489,430**；建议先给参数量估算（ego Δspeed/Δaccel/Δyaw-rate + OD Δpos/Δvel/Δheading/presence + LD Δgeometry/validity）与最小可训版本，"四列标量 residual 保留为诊断对照"。
6. **阶段排序建议**：§二 内**先做 ①正式接线 + ③几何实例化**（二者同为 §三.1 几何 A/B 的前置），再做 ②residual 链入库、④预算常量、⑤h1–h6 parity；§三（训练）与 §四（重构）并行推进。

## 3. 执行状态

- **本轮（R41 第一批 = §二 5 项）**：🕐 待开工（本 turn 按用户口径**仅排模+重写台账，不改代码**）；无在飞 lane。
- **已入库**（R40 验证批）：V1 `42318b1`｜V3 `6cdb93c`｜V5 `db3ded8`｜V4B `7235f65`/`4c71569`｜V6 `0cb0978`｜台账 `7eec9cf`；pytest **1145**；各波默认关逐位 all_pass。
- **门复核材料**：`.slim/deepwork/p1_v2/round40/gate_review_draft.md`（E1–E18+候选口径+22 点重算）**转为参考**（R41 §八 已给新门）；`policy_train_prereg.md`（P1/P2/…）**并入 §三 训练设计**（几何 A/B = G 臂）。

## 4. 挂账（逐项）

1. **R41 §二**：①oracle/teacher-force/speed 反事实正式接线（`pipeline/eval_runner.py`+`net/model.py`+`tools/test.py`+config/tests；每开关：CLI/执行路径/protocol hash/cache key/metrics+manifest/默认关逐位）②residual 链拆 `pipeline/residual_dataset.py`+`pipeline/residual_features.py`+runner+config，移除 `.slim` 硬依赖 + 小型入库 fixture CI ③几何改模型实例配置（config/ckpt metadata/protocol hash/cache key/metrics，废弃进程级/env 正式路径）④预算常量清理（`ACTIVE_LIMIT` 处置待裁）⑤h1–h6 分布 parity（Mahalanobis/P50/P90/max）。
2. **R41 §三（训练主线）**：几何 A/B（Policy-L/Policy-C 等预算 + 交叉测试）＋横向时序监督（κ_near/mid/far + `L_Δκ` + `L_replan`〔门控版〕）＋速度条件化（a_y=v²κ/翻转率/饱和预测/clearance）＋LQR 可执行性（先指标后 loss）；**纵横向联合验收**。
3. **R41 §四（架构并行）**：`net/kinematics.py`+`net/geometry.py` 单源（保留旧行为 fixture）＋WorldState 补全（ego/OD/LD physical+latent）＋底层状态 residual（四列降为诊断）＋非原地 `transition(state,action)`（step/source/version + before/after hooks）。
4. **R41 §五**：M0/M1/M2（M1≫M0 才 M2；对齐规则照抄）＋D0–D3（2/3-step BPTT，禁六步全反传）。
5. **R41 §六**：learner-state DAgger 接口（后置：本轮状态与 Policy 基础门通过后）。
6. 其他：Eval500 3-seeds；`runs/` 归域执行；doc_stats 节拍。

## 5. 纪律与索引

- **纪律**：默认关+逐位；生产改动随 pytest（当前 1145）与契约门；**门修订/选点必须预注册**；无放行不训练；不设默认/不启 MoE-DAgger-PPO-GRPO 直至 Clean500 安全门过；docs≤5 快照制（本文件每轮整篇重写）／历史走 git／runs 域命名／数据-spec 登记制／重资产保留制／doc_stats 节拍。
- **索引**：手册 `repo_handbook.md`（附录 D 归档指针）｜评估链 `.slim/deepwork/p1_v2/a3/eval_a3.py`、`round38/c6_protocol.py`、`round39/b1_matrix.py`、`round40/{v4a,v4b,v6,v8}*`｜登记表 `datasets/README.md`、`env/specs/README.md`｜R40→R41 关键 commit 见 §3。
