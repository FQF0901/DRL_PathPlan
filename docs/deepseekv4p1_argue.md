# deepseek-v4.1 argue / 主文档（**R40 当期快照**）

> **用法（用户口径）**：本文件按**当期快照**维护——每开新一轮**整篇重写**；旧轮一律走 git：`git log --oneline -- docs/deepseekv4p1_argue.md` ／ `git show <commit>:docs/deepseekv4p1_argue.md`。
> **配套**：`chatgpt5p6sol_review.md`（GPT 方案/裁决原文）、`repo_handbook.md`（架构/训练/门/**附录 D 归档指针**）、`README.md`（索引+卫生规则）、`architecture_contract.md`（架构契约）。
> **GPT 规格**：R39 主方案 `647f909` + 裁决 `fe5de55`（已执行：R39-A 全闭、R39-B 结案）｜**R40 方案 `c416095`（当前执行规格）**：五猜想验证 → 一致状态模型重构；验收见其 §五。

## 0. 目标与基线

- **总目标**：闭环 success **超越 IDM+Pure Pursuit**；最终门 = Eval500 3-seeds > IDM 且 coll/off 不劣。
- **三级集（禁混比）**：**Calib50**（开发；历史锚 legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58 / 统一基线 0.56 / **IDM 0.756**）｜ **Clean500**（晋级级；canonical `env/specs/scenarios_clean500.json`，sha `087db3f5…`；**@HEAD 重登记**：IDM+PP **0.742** ／ C0_HEAD **0.588** ／ C1_HEAD **0.390**；plan 协议 `c082b57d…`）｜ Eval500（正式：3 seeds+bootstrap+McNemar）。
- **研究基线（R39-B 结案产出）**：`ws_learned`（WorldState API + 学习残差头）= calib50 **succ 0.56 / tmo 0.06 / coll 0.18 / off 0.20 / rc 0.811**（vs `ws_off`=C1_HEAD 0.40/0.34/0.08/0.18）——**纵向修复兑现、横向@speed 为绑定约束**；**几何口径修正（R40-V3）**：`fixed` 对冻结策略为负（off+fixed 臂 coll 翻倍 0.08→0.16）；learned 臂统一 legacy 口径后 **coll 0.12**（off 仍 0.20、rc 0.815），但 legacy 保留缺陷仅作对照臂；**生产默认仍 `world_state_api=off`**（安全门未过不得设默认）。
- **参数门（R40 更新）**：总参数 **≤ 2,000,000**（GPT 放宽；原 1.5M/+2% 冻结作废），但每次报告 **total/reachable/trainable/frozen/dormant** 五栏。
- **协议/hash**：`evaluation_protocol_hash` + model/schema/ckpt 四 hash；正式重测必带（口径 `c6_protocol.py` v1）。

## 1. 根因状态（当前最优解释）

1. **三层根因（R36–R38 裁决链）**：①**结构层**（纵向保速先验 + LQR 弦长=速度指令）→ ②**接口层**（`path_*` 静默 0/缺键；已由 Q4-A 修）→ ③**横向层**（@speed 曲率质量/抖动）。
2. **R39-B 闭环定位**：WorldState API（analytic+学习残差）**修好纵向**（succ +16pp / tmo −28pp / 爬行 17.3→0.7 s）但 **coll +10pp**；剂量扫描**无安全区间**且碰撞对剂量不平坦（机制开即饱和）⇒ **绑定约束 = lateral@speed**。碰撞定靶 = **tollgate×3 + ramp_in/uturn/split/roundabout**（几何约束场景；含"降速仍撞"例）⇒ **横向精度**问题。
3. **R40 验证批判定（V1–V3；2026-10-16 已完成）**：
   - **猜想 4 ✅ PASS**（train/eval 特征同源）：250,622 全量逐位、6 项 `max_abs_diff=0`（唯一跨 batch 0.048 系 TF32 GEMM tiling 效应，已固化\"同 batching\"口径）——`42318b1`。
   - **猜想 2 ❌ 否证**（双 ego-motion 链）：数值精度内一致（ego pose 位级相等；OD 位置 ≤4.8e-5 m=op 序 float32 舍入；动作局部增益≈0）⇒ 下沉单源化**仅整洁性**价值、非缺陷。
   - **猜想 3 ✅ 成立，但方向 = `fixed` 对冻结策略为负**（离线+闭环双证）：离线 fixed 使 B1 c1 **−0.471（拖累）**（统一 legacy 后 B1 c1=**12.7365** > fixed 口径 12.2654；B2p=14.0553/2.7169）；闭环 **`off+fixed` 臂 coll 0.08→0.16（翻倍）**、learned 臂 **legacy 0.12 vs fixed 0.18**；**学残差 succ 增益两口径一致（+0.18, p=0.004）⇒ 几何只改安全轴**。**遗留**：①A4\"fixed 闭环≈中性\"结论需复核（新证 G1 coll +8pp）；②legacy 保留已知缺陷（foot 伪影 `|lat|≈0.02`、margin 符号互换）⇒ **只作对照臂不作候选**；③**敞口问题：几何应否\"训练侧 A/B\"解决（冻结头 OOD 假说）**。
   - **c2 阈值语义绑定几何**：统一 fixed 口径下 hold 基线 c2=**2.3829 > 2.31** ⇒ 门修订证据齐备（**须先预注册后跑**）。
4. **猜想 1/5（碰撞归因）**：学习残差只修 4 个读出标量、未同步 ego/OD/LD anchors 与 latent ⇒ \"纵向可加速、横向仍旧世界\"假说：**离线表示层检验（V4A）🕐 在飞**（O0–O7 oracle 替换臂）；**闭环 H0–H6 + 速度反事实（V4B）** 待 V4A 结果与 oracle 口径确认后派。

## 2. R39 状态（已完成/结案）

- **R39-A 工程门 ✅ 全闭**：A1 `ed83e57`（入口结构化装载/迁移+LoadReport/forward_mode 全迁删 shim/四 hash）｜A2 `f1f52f5`（双账本 + detach 31 项登记 + CI 守卫）｜A3（Clean500@HEAD 重登记）｜A1b `ba9e456`（装配全开、与 a3 逐位、四 hash 轴敏感）。
- **R39-B 结案**：B1 `c78aee0`+`7443ca7`（analytic-only 阴性对照：12.265/2.782 ❌ 门、补 B0 缺口 75%）｜B2 `23bee92`（学习残差 6,148 参数，c1 首过 13.909 ✓/c2 2.648 ✗，B2>B1 ✓）｜B2b `74b3c06`（口径对齐：学得更干净但**链上推力∝目标噪声** ⇒ B2 的 c1 过线系副产物；22 点无交点；门复核=原则可满足需强条件性）｜B2c `1db2aea`（闭环分诊）｜B2d `2259c24`（剂量扫描：区间空、绑定=lateral@speed）。
- **R39-C（Memory 表示）/ R39-D（learner-state）未开始**（R40 排入）。

## 3. R40 方案（`c416095`）——受理 + 我方排序提案与细化

- **受理（五猜想 + 重构 + 验收）**：①五猜想（横纵状态不一致 / 双推进漂移 / fixed 几何混杂 / train-eval 特征同源 / 碰撞=速度 vs 横向 vs 协同）；②重构（WorldState 补齐 ego/OD/LD physical+latent；残差从 4 标量升级为**底层状态残差**；距离/TTC/margin 由**最终 anchors 统一 readout**；**删除双 ego-motion 推进**（下沉唯一 `net/geometry.py`+`net/kinematics.py`）；transition 返回**新** WorldState（禁原地改）；接入 Imagined Memory consumer；M0/M1/M2 与 D0–D3 对照；nav 不可推断 ⇒ **missing 不得 held**；训练 runner/config/split/feature-parity **正式入库**）；③验收（离线单变量门、状态一致性门、Memory 门、梯度门、参数门 ≤2.0M 五栏、闭环门：候选须 ≥0.56/≤0.06/coll≪0.18/off≪0.20，Clean500 四版配对，晋级 +5pp 且安全不劣）。
- **我方排序提案（关键，供裁）**：**先控口径、再谈归因**——**V1 几何控（G0–G3 四臂：off/analytic × legacy/fixed）** + **V2 双链漂移对拍** + **V3 特征同源硬门**（三项零训练/当天可得）→ **重订离线门读数**（B0/B1/B2 全部在同口径下重跑；若 c2 门需修订，**先预注册再跑**）→ 再跑 **H0–H6 与速度反事实闭环臂**（H1/H5）→ 依结果做 §三 重构。理由：猜想 3 已代码证实 ⇒ **在混杂口径上继续归因会把 fixed 的贡献算到 WS API/残差头上**（反之亦然），代价是整轮返工。
- **我方细化/反驳（4 条）**：①**G2 可行性**：analytic 路径若无法用 legacy 几何（readout 依赖 fixed 的 foot 排除），须实现"legacy-aligned readout"变体或明确"不可构造"，并据此把 B1/B2 增量拆成 {几何贡献, 解析贡献, 残差贡献}；②**参数工具同步**：`net/param_probe.py` 的基线/上限（1,510,570/+2%）须改为 **2.0M 门 + 五栏报告**，否则 A2 断言与 R40 门冲突；③**c2 口径预注册草案**（若裁修订）：`|Σds−真|` 均值对"回归均值"天然有利（B0 天然 2.3086）⇒ 建议改为**加速需求加权**或**按桶归一**（`|Σds−真|/max(Σds_real,ε)`）并与 c1 组成联合门；**先预注册后跑**；④**训练工装入库**：`b2_train.py` 等应迁 `tools/`（含 config/split/命令），保证"train=eval 特征"是可复现事实而非声明。
- **生产/研发边界（遵守）**：生产 `off`；研发主线 `analytic_learned`；**安全门未过不得设默认、不得启 MoE/DAgger/RL**。

## 4. 挂账与待裁

1. **待 GPT/CG（更新）**：①**闭环 H 臂的 oracle 口径确认**（teacher-forced 定义：k≥1 用\"该步实际到达状态\"替换—由 V4A 预注册给出建议稿）；②**几何训练 A/B 是否列为本轮正式臂**（fixed vs legacy 重训残差/策略，检验冻结头 OOD 假说）；③c2 口径预注册草案（新证据：门语义绑定几何）；④**训练工装正式入库**（`b2_train.py`→`tools/`，含 config/split/命令；V1 遗留③）。
2. **R40 执行状态**：V1 ✅ `42318b1`（特征同源 + 参数门 2.0M 五栏）｜V2 ✅（猜想 2 否证）｜V3 ✅ `6cdb93c`（几何控臂 + 默认关开关；装配由编排代跑其自带装配器，lane 停摆）｜**V4A 🕐**（离线 oracle 归因）｜V4B ⏳（闭环 H 臂，待口径确认）。
3. **挂账**：R39-C（Memory M0/M1/M2）、R39-D（learner-state/DAgger）、B3/B4（BPTT 前置检查）、Eval500 3-seeds、`runs/` 归域、`net/world_residual.py` 内联上限统一 2.0M（V1 遗留①）、doc_stats 节拍。
4. **工程事实**：pytest 当前 **1137**；各波默认关逐位 all_pass；四 hash 常态记录。

## 5. 纪律与索引

- **纪律**：默认关+逐位；生产改动随 pytest 与契约门；无放行不训练；清理六规则（docs≤5 快照制／历史走 git／runs 域与命名 official=BTC<ts>_<name>、dev=`runs/dev/roundNN/<arm>`／数据-spec 登记制／重资产保留制／doc_stats 节拍）；**门修订必须预注册**。
- **索引**：手册 `repo_handbook.md`（附录 D 归档指针）；评估链 `.slim/deepwork/p1_v2/a3/eval_a3.py`、`round32/canary_eval.py`、`round38/c6_protocol.py`、`round39/b1_matrix.py`；登记表 `datasets/README.md`、`env/specs/README.md`；R39-A/B 波次 commit 见 §2。
