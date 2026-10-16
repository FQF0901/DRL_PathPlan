# deepseek-v4.1 argue / 主文档（**R41 当期快照 v2**）

> **用法（用户口径）**：本文件按**当期快照**维护——每开新一轮**整篇重写**；旧轮走 git：`git log --oneline -- docs/deepseekv4p1_argue.md`。
> **配套**：`chatgpt5p6sol_review.md`（GPT 规格/裁决原文）、`repo_handbook.md`（架构/训练/门/附录 D 归档指针）、`README.md`、`architecture_contract.md`。
> **GPT 规格链**：R39 `647f909`+`fe5de55`（✅）｜R40 `c416095`（✅ 验证批）｜**R41 `e8e9c7a` + 复审 `deb8ee9`（当前执行规格；六点修改全部受理）**。

## 0. 目标与基线

- **总目标**：Eval500（3 seeds+bootstrap+McNemar）**success > IDM+PP** 且 coll/off 不劣。三级集禁混比：Calib50（筛选，**不得作收益声明**）／Clean500（晋级）／Eval500（正式）。
- **研究基线**：`ws_learned`（Calib50 0.56 / tmo 0.06 / coll 0.18 / off 0.20 / rc 0.811）；**生产默认 `world_state_api=off`**；未过 Clean500 安全门：不得设默认、不得启 MoE/DAgger/PPO/GRPO。
- **R41 门**：**横向门**（vs ws_learned：κ 帧间符号翻转率 ↓≥50%、steer 饱和 ↓≥40%、高速横向误差不恶化、连续 footprint 越界下降、coll≪0.18、off≪0.20）｜**Calib50 筛选**：succ≥0.56 ∧ tmo≤0.06 ∧ coll≤0.08 ∧ off≤0.18 ∧ rc≥0.815｜**Clean500 晋级**：vs C1_HEAD succ +5pp、coll/off 不劣、tmo 下降、rc 不降｜**工程门**：干净 clone 可跑 residual 训练、核心 CI 不因 `.slim` 缺失 skip、oracle/速度干预有正式 CLI+模型接线、几何入 protocol hash、h1 同 batching max diff=0。
- **参数门**：`total ≤ 2,000,000`（五栏）；`ACTIVE_LIMIT=1.05` **仍参与断言**（代码事实）⇒ 按 R41 单独获批或降为警告（本轮清理项）。

## 1. 结论分层（按 R41 复审要求）

### 1.1 ✅ 正式可复现结论（已入库、可重现）

- **训练/评测特征同源**：250,622 全量逐位 `max_abs_diff=0`（V1 `42318b1`；h1）。
- **双运动学链数值近似一致**（V2；float32 op 序差 ≤4.8e-5 m）。
- **zero-shot fixed 几何对冻结 Policy 不利**（V3 `6cdb93c`：离线 c1 −0.471；闭环 off+fixed coll 0.08→0.16；V6 `0cb0978`：闭环 footprint 对几何轴不可分辨、唯一活跃差=margin 符号轴、剩余差归噪声 p=0.45）。
- **`ws_learned` 改善 timeout 但恶化安全**（V3/V4B 两口径一致：succ +0.18〔p=0.004〕、tmo 0.34→0.06、coll 0.08→0.18、off 0.18→0.20）。
- 工程事实：正式入口 `--teacher-force/--speed-*/--record-truth` **仅 hash 占位、无执行路径**；`tools/train_residual.py` 硬依赖 `.slim`；几何为进程级+env；pytest 1145。

### 1.2 ⚠️ UNVERIFIED_IN_REPO（未入库研究脚手架的暂定观察；**不得用于终止 WorldState 方向**，正式接线复现后再裁决）

- oracle **OD/LD** 替换无收益、**ego oracle 撤销纵向增益**（⇒ 增益或依赖状态伪影）；
- 横向命令 **5× 过量**、帧间 **75% 方向翻转**、全油门窗 **99.1% 高转向**、净加速 +0.03 vs 干净档 1.54–2.82、去耗损 sim v→7.17（V8 诊断）；
- R40-V4B 的"H3/H4/H5 否证"与"横向 oracle 有害"结论（自教师两遍法、n=50、弱速度操纵、C0/C1 修前版本）。
- 上述第二、三项即**当前最高优先级性能线索**（横向幅值/变号/饱和），但须先完成证据链正式化与消费者图。
- **正式接线后裁决（A1 `78db917`）**：多步（k≥2）合法 oracle 注入在当前闭环**结构性不可得**——MetaDrive engine 无 snapshot/clone（`present=[]`），两遍法+五类漂移门在真实冒烟中仅"帧 0 步 1"合法（12 步），链步/后续帧 `auto_rejected`（如 ds 差 0.037>5e-3）、episode 整体剔除；legacy V4B truth 全 `unproven`（46/46）。⇒ **V4B 式多步 oracle 臂在设计上不可正式复现**；后续 oracle 诊断只保留**一步合法**形式（或需引擎级快照支持后再议）。

### 1.3 排模事实（本轮新增，代码为准）

- **① 环境快照/克隆**：`env/*.py` **无** `dump_state/load_state/snapshot/clone` 封装 ⇒ R41"最优方案（snapshot/clone 一步合法 transition）"需先核实 MetaDrive engine API 并实现；否则按"两遍重放 + 漂移报告门"（replay pose/OD state divergence、traffic RNG 一致性、termination 一致性、逐步状态误差；超阈 episode 不得纳入 oracle 裁决）。
- **② lane_anchor 捷径（证实+量化，C 审计）**：`lane_anchor_xy` 是 **path head 的必需输入**（`path_head.py:72/317/826`），评测由 `--lane-anchor inject`（`select_target_lane` 教师口径）注入；**梯度/有限差分弹性为全部输入源最高（eval G=160.3 / F=241.5，约次高 ego_phys 6.25 的 2 个数量级）**——因 κ_ref 由锚**二阶差分解析提取含 1/Δs² 增益**；遮挡读数 Z=0.83 非最高；**κ=κ_ref+残差 且残差在抵消锚曲率（corr(κ_ref,res)=−0.65，残差 RMS=1.31×κ）** ⇒ 锚是最高增益通道+解析先验、非盲目跟踪内容源，但**横向训练前必须修正该输入结构**（平滑/低阶拟合/≥2m 重采样后再差分）。
- **③ LD 消费者（修正我方此前论断）**：**"LD latent 无消费者"仅对 `net/world_state.py` 的 readout LD 列/旧 token 语境成立**；`obs['ld']` 物理与潜**经 plan_context/semantic 有真实消费者**（`net/model.py:1500/1507`、非 semantic `:1345`），且 LD 物理是**全部源中遮挡读数最高（Z=4.76）**。⇒ 这也重解释了 V4A/V4B 的 "LD oracle ≡ C1"：被替换的是 **WS readout 的 LD 列**，非 `obs['ld']` 路径。**target lane token 对 κ 结构性零消费者**（只进旧 PolicyHead）；**Memory 仅 `others_ctx` 有消费者**（ego/od/ld_ctx 全零）。
- **④ 其他 C 事实（P1/P2 设计约束）**：评测帧 **99.1% 锚视界 <29.75 m ⇒ κ_far 恒 ~0**（κ_mid/far 监督必须带**锚视界有效性掩码**）；rollout 6 步 aux：锚在 k1 峰值后衰减、ego/LD 状态全步持续；建议锚注入加 2–5 cm 噪声增强；probe≡`model.encode` 逐位（fidelity 0.0）。

## 2. R41 六点修改（全部受理）+ 执行批次

1. **V4B 移出正式结论** ⇒ 已按 §1.1/§1.2 分层（UNVERIFIED_IN_REPO 标记）；
2. **teacher-force 优先 snapshot/clone 一步合法 transition**；两遍重放须报告环境漂移、不得自动视为动作条件一致（见 §1.3①）；
3. **residual runner 移除 `.slim` 硬依赖**（拆 `pipeline/residual_dataset.py`+`residual_features.py`；核心 CI 不得因 harness 缺失 skip）；
4. **横向训练前补 State Consumer Graph**（κ_near/mid/far 对 ego/OD/LD latent、lane_anchor、target-lane、Memory 的敏感度）；
5. **底层状态 residual 第一版收缩**：ego Δspeed/Δyaw_rate + OD Δpos/Δvel/presence；**LD 先保持解析静态几何、只学 validity**（不自由改地图几何）；
6. **LQR 可执行性重加权用 student−expert 执行差距**（`w ∝ max(0, sat_student − sat_expert)`），不得惩罚所有高 steer 样本。

**批次**：**第一批**（①正式接线 ②residual 链入库 ③几何模型实例属性 ④几何入 hash/缓存键/ckpt ⑤h1–h6 parity）→ **第一批半**（消费者图）→ **第二批**（横向训练：**P0 现基线 / P1 +当前帧 κ 监督 / P2 +速度分桶横向监督 / P3 P2+跨帧重规划一致性〔`w(ΔS)·‖κ_t−Warp(κ_{t−1})‖` 门控版〕/ P4 P3+student−expert 重加权**；几何必须 legacy训练/legacy评测 + corrected训练/corrected评测 + 交叉测试）→ **第三批**（WorldState 一致性：统一 kinematics/geometry、ego latent 入 WorldState、底层 ego/OD residual、readout 全由 final state；**验收=状态一致/消费者真实存在/不污染安全**，不预设更大闭环收益）。
**后置**：Memory（M0–M2）/detach（D0–D3）/learner-state DAgger / MoE/RL。

**R41 进度**：**第一批半 ✅**（C：消费者图，见 §1.3②③④——已修正 LD 论断；输入结构修正项已列入第二批前置）｜**第一批**：① ✅（A1 `78db917` 五要件接线+注入 hook+漂移门；多步 oracle 结构性不可得的裁决见 §1.2 末）、② ✅（B `38298d9` 零 `.slim` 依赖）、③④⑤ 🕐（Wave-2：D 几何实例化+预算常量 / E h1–h6 parity）。

## 3. 挂账

1. 第一批 5 项（在飞）＋第一批半消费者图（在飞）；2. 第二批 P0–P4 + 几何 A/B；3. 第三批 WorldState；4. Memory/detach/DAgger 后置；5. Eval500 3-seeds；6. `runs/` 归域、doc_stats 节拍；7. `ACTIVE_LIMIT` 处置。

## 4. 纪律与索引

- **纪律**：默认关+逐位；生产改动随 pytest（≥1145）与契约门；**预注册**（门修订/选点）；无放行不训练；安全门前不设默认/不启 MoE-DAgger-PPO-GRPO；docs≤5 快照制／历史走 git／runs 域命名／登记制／重资产保留／doc_stats 节拍。
- **索引**：手册 `repo_handbook.md`（附录 D）｜评估链 `.slim/deepwork/p1_v2/a3/eval_a3.py`、`round39/b1_matrix.py`、`round40/{v4a,v4b,v6,v8}*`｜登记表 `datasets/README.md`、`env/specs/README.md`｜关键 commit：V1 `42318b1`｜V3 `6cdb93c`｜V5 `db3ded8`｜V4B `7235f65`｜V6 `0cb0978`。
