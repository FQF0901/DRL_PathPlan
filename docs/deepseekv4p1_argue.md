# deepseek-v4.1 argue / 主文档（R37 诊断+零训练收口：M3=保速语义+`path_dynamic_ttc`→0 错配（77%）；M4=C2 机制生效 coll −0.06/F3 证伪；M5=A 探针 ✗（off 0.44）；**M6=接口修复 ✅ 零训练 +8pp（0.34→0.42，翻正 4/0）但非完整解** ⇒ **P0=训练层 B（IDM+PP 蒸馏，`dagger_collect` 就绪）+ eval 装配一致化**；C3 其后（§5.64））

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-15 下午·十）**：**R37 阶段 1 四路诊断收口**：M2（C0 0.580 vs C1 0.340，C1 不晋级）、**M3**（头在训练分布内没坏〔bias +0.056/MAE 0.149〕但**命令语义=保速**；坍缩=闭环特有自锁；**装配错配主凶=`path_dynamic_ttc` 缺键→0**〔训练 30.6 s「无冲突」vs 评测 0「立即碰撞」，占消融效应 77%〕）、**M4**（C2 warp 机制生效、**collision 0.14→0.08**、Hard 不劣化；**F3 证伪外部配速**：v 8.9 但 coll 0.48）、**M5**（A1/A2/A3 先验探针意图层生效〔v_ref 8.49/6.31/5.37、残差不变式 50/50〕但达成层不升〔v 3.56/3.01/2.85〕，**Hard 全恶化〔off-road 0.12→0.44/0.28/0.24〕**；A1 steer 饱和 0.81+蛇形）。**横向侧谱（主控复核）**：加速后 |err_ψ| 0.017→0.241、plan |κ|p90 0.023→0.081、κ 抖动 ×4.7 ⇒ **横向能力（曲率质量）是共阻塞**。**零训练通道总账：F3 ✗ / A ✗ / C2 ~中性** ⇒ **P0 定为主训练 lane（B 第一候选：教师=IDM+PP 规则专家，`tools/dagger_collect.py` 就绪；训练须在 eval 装配/指纹统一下做）**；C3（WM/想象步动力学）其后；**接口修复探针 M6 ✅ 落地**（`--long-exog-fill neutral`：**c1_fill succ 0.34→0.42，翻正 4/翻负 0**、tmo −0.04、rc +0.058、v +0.31；a1_fill off 0.38 不晋级）⇒ 零训练 +8 pp、非完整解，剩余=保速教师+@speed 横向质量（训练层）。M1 已入库 `0d63bbc`（966 tests/契约门 6/6/off 逐位）。
> **本版说明**：第五十轮补遗2（402 → 432 行）——新增 §5.66 阶段 1 复盘（进展/发现/结果/分析/问题 + 起步/限速/监督/验证集四问核查结论）；§5.64 追加 M3–M7 结果；§0/§7 刷新；上版 418 行 = `git show d66cc0e:docs/deepseekv4p1_argue.md`；历版全文=本文件 git log。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：最高可执行 d2=0.706）；P1 起点 pri512@e020 0.340。
- **主线锚点（calib50）**：legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58 / **MoE-off 统一基线 0.56** / IDM 0.756。
- **主线结论链（压缩）**：1b no_increment → 12A NO-GO（gate 选择性瓶颈）→ A3 判负（动作可持性）→ R21 语义重建（3C 首现强正向）→ 3D-3 未过门 → R25 首超 legacy（0.400/+7.3pp）→ R26 Q1（0.50）→ R27 R1（sat ✓ 但 tmo +26pp ✗）→ R28 28A 归因（F3/F1/F4）+ K1 首步解耦不足 → R29A T2 唯一正向（0.58）未过门。
- **R30–R33 裁决链**：R30 Reality Audit 定新主问题=**部署链一致性**（WM 固有 TF 0.02–0.04 vs 部署链 0.20–0.32，8–10×）；R31 审计包（Q2=中等风险/无冲突；K1 base≡M0；mirror 动作响应弱 24×–5612×；契约 S3≠S2）；R32 Level-1（**D2→D3=97.0% 主瓶颈；动作条件化失效**；real latent 充分；A_k ×2.1）⇒ GO_ACTION_CONDITIONING；**canary 四臂**（Gate 0 ✓；Gate 1/2 fidelity 级有效；**Gate 3 全 FAIL；Gate 4 AC1 succ 0.600✓ 但 coll 0.180✗**）⇒ 不自动进 C-B；**R33**：33A 三问 + 33B warp NO-GO ⇒ **GO_EGO_ACTION_CONDITIONING_V2**；**R33C**：B1=**稳定器（接口修复 −10~−12%）但硬门 6/11 未过、不晋级**；**R34A**：**失败=路径级**（通道选择 6/越界 6/撞障碍 6；跟踪/表示缺失/未响应 0；覆盖率 94.7%）；**R34B0**：覆盖门 ✅（10/10、7/7、17/17）⇒ 问题是选择/生成；摘要非杠杆；**R35**：**外部 target-lane 规则=toll 失败集 0/10 缺陷源**；merge=横向几何；无 token 自主选择不可靠；**R36（§5.53/5.54）**：终止回退 `acd22c1`；直接执行 `7c0186f` 执行优先——H1 骨架（Gate 0 8/8）+ H2 given-target 数据规格（99.98% 标签、门 15/15；T4 晚换道 caveat）+ H3 hooks（555 键）；**R36 修正（§5.55）**：**认账 H1 `PathExecutionHead` 结构性偏离**（第二生成器+绕过自回归）⇒ 重构 `StructuredPolicyHead`（每步调用；辅助仅 κ_mid/far+未来 progress）+ `architecture_contract.md`；K1 开工。**R36 车道保持阶段（§5.56–5.58）**：H1/K1/I1/J1/J2 全收口；**T1 越界 5.75%（baseline 8.75%）≈专家地板 5.2%**（1-epoch 早停；`plan_head.fusion.2` 必须开〔head-only 欠拟合 0.479 vs 0.058〕）；**T2 终端收敛 0.458→0.706（超专家 0.586）**；GPT 5 条批评核验 + 3 条反驳；**L-A 硬化入库 `bbbe541`**（旧头不前向/oneshot 守卫/运行时断言）+ **L-C G6 前置全绿**；**门校准（相对门）交 CG**；L-D warp 层收口（`9b00761`）。**R37（§5.62–5.64）**：M1 warp 主路径入库 `0d63bbc`；**M2 C0/C1 同协议分诊：0.580 vs 0.340 ⇒ C1 不晋级**（timeout 20/50）；**诊断收口（M3/M4/M5）**：头在分布内没坏但**保速语义**+**`path_dynamic_ttc` 缺键→0 装配错配（占消融 77%）**；C2 warp 机制生效（**coll −0.06**、k≥1 全分化）；F3/A1–A3 零训练配速/先验探针**全 ✗**（coll 0.48 / off 0.44）⇒ **P0=训练层：B 教师蒸馏（IDM+PP，`dagger_collect` 就绪）+ eval 装配一致化**；M6 接口修复探针在飞。**§5.66 四问核查**：spawn=6.0 m/s（非 0）/限速仅编码器可见（head 显式输入无）/起步加速监督近乎缺失（v<1 占 2.67%）/calib50=主线开发集口径（与 0.40–0.58 锚点及 IDM 0.756 同集可比）。
- **旁线+收敛一句话**：dθ-only 0.728（+2.2pp）；lane semantic 前三层已正证；未闭合=normal-state preservation + plan-level trackability（28A：F3/F1/F4）。
- **约束（持续有效）**：plan 部署/评测协议；单 GPU；fail-fast；task-session 解耦 + 20 min 节拍；conda/python 只读；正式 run 不清理。**冻结维持**：60/40 Policy 数据适配、DAgger、MoE、PPO/GRPO、BPTT、Policy 解冻、LQR 修改、WM 主体扩容、recovery 扩采、几何反传、gated residual 复活（细目见 §1 与历轮）。

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
- **评测纪律**：fail-fast 加载校验（missing/unexpected/shape_mismatch=0）为硬条件；跨版本只作方向参考；footprint 规格 = `arc_step` 位姿 + 0.5–1 m 插值 + 同款 drivable 判定；引擎 run-to-run 随机性地板 0.2–1%（判读噪声 ±0.2–0.4pp）；正式实验记录 = manifest/CSV/decision.md，不进定期清理。
- **Gates**：**G1-diagnosis ✅ / G1-ceiling ❌**（"超越路径"须由配速修复 / 更强教师 / RL 超教师之一建立）；**G2 追平**（3-seed mean ≥0.74；off ≤0.098/≤0.097；collision ≤0.144/≤0.174）；**G3 超越**（mean ≥0.80、worst ≥0.77、paired LB>0、McNemar p<0.05、非单一路型）。
- **R32/R33 新增纪律**：生产改动一律"**默认关 + 逐位兼容 + 966 tests**"；录制确定性已验证（重复噪声=0）；**配对 bootstrap** 为默认统计口径；canary/实验 ckpt 一律"启用开关写 metadata + 载入 fail-fast"（v2 起）。

---

## 2. 压缩存档：P0 → P1-v2（关键结论）

- **P0 三问**：上限 **d2=0.706** > oracle 0.608 < **IDM 0.756**；P1=计划侧（>50% 带 plan-infeasibility）；保持 plan 协议（`repeat_action` 归零）；fail-fast 两拦。
- **排摸收官（分配证据）**：pri512（eval500 0.332）、rou64（0.370@e015）胜出；组合负交互被拒。
- **P1 要点**：**V1（ds 拉伸）0.762/0.776/0.778**@1.20/1.25/1.30；反事实错位 9,377 行（已修，takeover_only 后 2,317）；P1-2 根因=绝对目标恢复监督 × 失败/低速态不可分 × ~40× 过采样；P1-3 cos(grad_traj, grad_geometry)=**−0.73** ⇒ 几何 FP 门 FAIL（`ld_margin_weight` 维持 0）。
- **P1-v2 / 12A（闭合）**：1b `no_increment`（0.682/0.731）；Step3 闭环激活 **16.1% ≫ 离线 3.99%** ⇒ clean150 0.3067 FAIL；12A 15 配置全未过九条件 ⇒ **gated residual 主线关闭**。教训：闭环度量不可迁移；gate 选择性才是瓶颈；跨后端统一 CUDA。

---

## 3. P1-v2 执行纪要（Step 1–3；压缩存档）

- 锁板/冻结：可辨识性 probe → recovery_v2 采集 → gated residual pilot → 几何 FP 门 → 消融 → candidate recipe；每步 Go/No-Go；无条件轨=诊断旁线。
- Step1/2（GO）：1a conditional_pass（0.940）→ 1b no_increment → 1c F2m 0.804/0.843 重开；2,693 行/110 eps、六硬门全 PASS；v1 根因=同状态 obs 中位差 33.2。
- Step3（FAIL）：gate=0 逐位==base；离线过（AUC 0.870）但闭环激活 16.1% ⇒ 判负（旁线：E=常态 μ 无系统 bias；D=dθ-only 0.728 ⇒ V1=组合效应）。

---

## 4. Round-12A 零训练诊断与主线关闭（压缩存档）

- 授权链：零训练复盘三机制 + 同架构仅一训；九条件（允许"无有效区域"=判负关闭）。
- 执行：逐 step 遥测 → 逐 episode 复现（0.3400/0.3067，13 翻负/8 翻正一致）→ CPU/CUDA 全链（40 eps 6 集翻转）→ 15 配置扫描；全未过：**gate 误报且无选择性**（16.1% vs 3.99%）、残差 1.0 偏强、连续触发放大。
- **NO-GO → 关闭 gated residual**（产物 `runs/p1_v2/round12a/`；零生产改动）。

---

## 5. round-15→33 执行批次（结果导向；往来残档已并入）

### 5.1–5.28 round-15→29 压缩存档（每轮关键读数）

- **15/16 + 15A**：关闭 gated residual、反对 RL 优先；A1 共变量偏移（learner 6.4×/16.5×、sign 错 21.7%）；V1 s130 **0.778**；q1 天花板 0.72–0.78；C1 union 0.790<0.82 FAIL；几何 FP 门 FAIL（5 项）。
- **17 A3（判负）**：三臂确定性尾全 0.000/off 1.000；A3×chain 0.3533 ≈ 对照 ≈ 基线 ⇒ 远期六步链"无害亦无明显收益"；**瓶颈=动作可持性（holdability）**（同 ckpt 隔离证明）。
- **18/19/18A**：七项代码论断全成立；伪边率 **76.5%**；translate/rotate ±0.002（**~50× 过弱**）；zero_history 5–7× ⇒ input_topology 缺陷 + primary_response 弱（最小语义特征路线）。
- **20 M0/M1/M2**：M1 0.3533（无害不过门）、M2 0.0000（崩）⇒ 预注册转数据侧；后更正（§5.11）：M2=接线缺口（`plan_step` 恒 masked_mean）。
- **21/22 M2a/M2b**：Gate A 9/9 过；闭环未过（α 1.2%）⇒ **Outcome D → R21 语义重建**。
- **R21-0..2 + 3A..3C**：target-lane 纯函数 + 12 维通道 + 归一化/分组池化；**3C 首现强正向**（sign 99.4%、lane 4.31×、curv 20.1%）⇒ 催生 3D。
- **R21-3D**：3D-1/2 门过（独立集 5/5、curv 25.0%、**static_role_only 默认**）；**3D-3 未过**（−8~9pp）⇒ 定位=替换式初始化代价 + 语义未被 action 采用。
- **R25**：残差化（β=0 逐位）+ mask 修复 + 动作级监督；**P1 首超 legacy（0.400/+7.3pp、off −20.7pp）** 但 tmo/sat ✗；B 行标签 bug 修复（corr −0.156→+0.606；3B/3C/3D-3/首跑 P0/P1 受污染追注）。
- **R26 Q1**：B 标签 v2 fail-fast + delta 监督 + 幅度软锚 + 5×5 镜像；Q1 硬门 6/7（teacher +39.6% ✗）⇒ 停止；**calib50 0.50**；MoE 口径差 ±2–4pp（结论稳健）。
- **R27**：27A/27B 门过（sat 驱动=trackability 36.3%/重规划抖动 34.9%/首步 27.8%）；R1 修饱和（sat −9.4pp ✓）但 **tmo +26pp ✗** ⇒ 停止；定位=ds 进度与转向权限未分离。
- **R28**：28A 归因 ✅（**F3 速度参考** tmo 12%→8%；**F1 κ-ds 强耦合**；**F4 LQR 尾段**；chord 效应仅 0.005 m/s）；K1 仅首步解耦不足（ds≡legacy 未消除 tmo）、S2 解冻崩塌 ⇒ **横向 S2 支路正式关闭**。
- **R29A**：T1/T3 机制性失败（含 M0 对照）；**T2 唯一正向（0.52→0.58）未过门**（off 0.14/coll 0.10/tmo 0.18）；Δκ 瓶颈=**首步钉死**。

### 5.29–5.31 R30 Reality Audit 压缩存档（零训练）

- **四格矩阵（8 run）**：exact 行结构退化（仅消费 `plan[0]`）⇒ 实为 3 系统；S1≈S2；**S3 严格劣（0.020/0.000）**；v2 复跑逐 episode 8/8 一致。
- **R4**：E_if/E_deploy=1.54→1.22（≫50%）⇒ 接口主瓶颈；E_z_ego 0.203→0.316；**E_real≈0.40 恒定**（策略误差地板）。
- **WM 条件化（M0）**：动作诱导 Δz 弱于状态自然差异 1–3 数量级（ds 通道 ~1e-4）；a_next 反向修正极性（~0.90）。
- **梯度账本**：detach 截断 / policy→WM 不可达 / Stage-B WM 全冻三布尔全实证（0.0 vs 2.226）。
- **TF 消歧**：纯 WM TF 误差 0.021→0.042 vs 部署链 0.203→0.316（**8–10×**）⇒ **WM 固有能力 OK；主问题=部署链一致性**（策略态分布漂移 + 0.5s 重规划替换 + LQR 残差）。

### 5.32–5.35 R31 审计包压缩存档（五 lane 全只读）

- **Q2 判决**：Stage B 监督=中等间接性代价 + **无梯度冲突** + 非灾难 ⇒ 不构成一级非法监督（E_state 0.022→0.065；rollout ADE 0.544 vs oracle 0.202；oracle 代价亚米级）。
- **M2**：部署 E_z 非 pose/执行错配驱动；**重规划替换真实**（|Δds|→2.0m、松一致率 0.24→0.06）。
- **K1 base≡M0 逐位**（K1-only=20 键）⇒"K1 破坏 latent"排除；expert 协议 vs 部署 E_state **8–10× 放大**。
- **mirror**：scene/action 响应比 **24×/5612×/1392×**（动作/位姿条件化弱）；restricted 一致率 0.842。
- **契约**：S3≠S2（结构级）；两 eval 栈默认相反；exact 仅 `plan[0]`；MoE 口径 ±2–4pp ⇒ **MoE-off 0.56 锁定为统一基线**。

### 5.36–5.37 R32 提案/裁决压缩存档

- GPT（`910b49b`）全面接受 R31 + WM 五结构弱点核验全成立（W1 动作仅经 pose 且 detach；W2 无 ego 物理头；W3 OD t0 恒速外推不回灌；W4 node_mask 固定 t0；W5 latent target detach 共适应风险）。
- 我方 5 条修正全收（Q2 首轮已答；部署状态归因；**Action Jacobian 须有限差分**；MoE-off 0.56；mirror 不锁 95%）；批准**三级决策树**（Level 1 四 lane；Level 2/3 条件触发）。

### 5.38 R32 Level-1 结果（四 lane 全只读）

- **P1（D0–D3，50 ep MoE-off）**：**D0→D1 2.9% / D1→D2 0.0% / D2→D3 97.0%** ⇒ 主瓶颈=**latent-state evolution block**；overshoot 0.469→1.660；D0 逐位复现 R4。
- **P2**：动作条件化失效——ds 通道 FD 敏感度 26–50× 弱；Swap top-1 **10.2%**/margin≈0（Gate B 未过）。
- **P3**：real latent 充分（lane ld_slots 0.90/0.93）⇒ 排除 state-rep 支路；WM lane 保留 0.61/0.77；od/ld ego 泄漏。
- **P4**：A_k ×2.1（温和）；Lipschitz 中位 0.73–0.78、尾 6× ⇒ 接口次要。
- ⇒ 草案 `GO_ACTION_CONDITIONING`；GPT 精确化："action conditioning=首个获得直接失败证据的子机制"。

### 5.39 round-32 答复受理（GPT `cde2e51`；压缩）

- 接受 Level-1 与主方向；批准**最小可逆 canary**（AC0/AC1/P-only/AC1-P；C-A→C-B；Gate 0–4；10 条锁定）；采纳 ΔR² 条件化增益 + 语义串扰防线（action 不无差别广播）；我方 3 条执行注记（数据域/接线点/§五 vs §七 按 §七 因子设计）。

### 5.40 R32 canary 执行（压缩；全文 `docs/round32_canary_report.md`）

- **基建**：生产 +103/−3（默认关/惰性）；**Gate 0 七项全 PASS**（默认=HEAD 键集 168、旧 ckpt strict 0/0/0、关闭/α=0 逐位、RNG 不变、梯度只流新参数、OD/LD 不广播、z-path 等价）+ 966 tests。**α 冷启动=1（训练口径）**——审计闭合：ckpt vs M0 **共享键逐位 0 差异**；step1 `new_grad=2/5`（末层）→step2 `5/5`；终值 α=1.1267/0.7743。
- **P3b**：ΔR²_WM state 仅 +0.025/+0.042 ⇒ "保留"≈惯性；**ΔR²_action +0.002~0.028**（latent 演化几乎不依赖动作）；泄漏 od_pool +0.32 / ld_pool +0.90。
- **四臂（C-A expert-only，各 22,532 步/4ep）**：**Gate 1/2 fidelity 级有效**（top-3 0.635→**0.917**✓；FD 38.7→**2.74×**✓；ds 单调 0.743→**0.946**✓；ΔR²_WM 全正；保留改善；泄漏单通道 ✗）；**Gate 3 全 FAIL**（AC1 E_if **+20.9%**/E_action +17.2%/overshoot **+16.5%** 恶化；AC1-P 近中性）；**Gate 4：AC1 0.600✓/0.120✓/0.180✗/tmo5**（succ +4pp 但 coll 翻倍）；P-only≡base（机制：phys 损失经 z_ego_next 塑形 action 分支，AC1-P 中体现）。
- ⇒ **不自动进 C-B**；AC1/AC1-P 不作新 baseline。

### 5.41 round-33 答复受理（GPT `f0ebfae`；压缩）

- **P0 结案**：反驳接受（冷启动按设计；归因有效）；初始化不再审计。**批准开工**：33A（限时只读三问）→ 33B（W0/W1/W2 warp）→ 33C（B0/B1/B2）。**v2 统一规格**：隐藏层正常初始化 + 输出层 zero-init + **残差增益固定 1（删可学习 α）**。我方 3 条方法论补充（单调率口径 / warp 探针可解释性 / 碰撞统计功效）与 A5 增项（plan-shift vs WM-response）获采纳为 33A 问题。

### 5.42 round-33 执行（33A 三问 + 33B NO-GO）+ 分析/问题

- **33A-Q1 碰撞**（双证据链：录制启发式 + 同协议复跑读 MetaDrive 接触真值，13/13 repro=exact）：**最大簇=收费站静止障碍 6/13**（撞岗亭 3——非车辆、不在 OD/LD、仅经 others 走廊段；撞 OD 可见静止/排队车 3）；侧碰/并线 4/13；低速追尾 3/13；**交叉口冲突 0**。AC1 新增 5 例不集中单类（侧 2/静 2/追 1）但 **2/5 在收费站岗亭**；末决策步仍在给油（plan ds 2.51–3.43 m/0.5s）、min_ttc 1.42–1.73s、v_ego 2.9–5.1 m/s。
- **33A-Q2 overshoot**（逐样本恒等式 O=A+B+C 残差=0；与 c4 口径 0 差异）：h5 ΔO=**+0.2736m**（+16.5%）——**(ii) 规划/WM 接口项 B=plan−mu 主导（Δ=+0.3226，share +117.9% [108,141]）**；(i) 初始策略项 A 反向（−0.0552，−20.2%；mu +0.050 但 expert +0.117）；(iii) 正反馈 C 可忽略（+0.0062，+2.3%；无晚段累积，Spearman NS）。h1–5 聚合 B +115.1%；速度混淆不解释（B +17.8% ≫ 速度 +6.3%）。
- **33A-Q3 噪声**：录制**确定性成立**（gate4_ac1 vs record_ac1 逐位一致）⇒ 重复噪声=0；episode 配对 bootstrap（B=10000）半宽 **6.06/6.12/8.90pp ≪ 恶化 20.9/17.2/16.5%** ⇒ **恶化真实**。
- **33B warp（W0/W1/W2，base 模型，各 50 ep）**：**NO-GO**（严格 0/4；宽松 W1 2/4、W2 0/4）。W1：E_if −2.0%、E_action −1.7%、overshoot −2.4%、lane +3.9%（split 不稳）、OD 泄漏增加；**W2（纯坐标重表达）≈0**。自证：off 逐位（27 输出键 max|Δ|=0）、w0 重录 vs base **50ep×40 键逐位一致**；warp 代码默认关保留。
- **分析（机制综合）**：
  1. **图景一致**：canary 使 WM 对动作敏感（Gate 1/2 ✓），但部署链上**规划层放大 ds**（B +117.9%）→ overshoot/E_if 恶化；碰撞集中于**收费站静止障碍**（岗亭不在 OD/LD、仅 others 5 维走廊段）——与"ds 放大 → 不减速"同源假设一致（末步仍给油）。
  2. **泄漏结构**：action 仅进 ego transition，但 od_pool progress 泄漏 +41~66% ⇒ 图消息传递把 ego 运动带入 OD latent（与 P3 同型）；lane 保留反而改善（+0.065/+0.186）。
  3. **33B 边界**：冻结 WM 消费新坐标系=off-distribution ⇒ 负结果只判"零-shot 不可用"、不判"方向错误"；W2≈0 说明 ST-GNN 对绝对坐标系不敏感（相对几何不变）⇒ warp 价值若存在需重训验证（不自动扩）。
  4. **噪声结论强化**：录制确定性 ⇒ 所有配对结论（+16.5%/+20.9% 等）均为真实模型效应，非 run-to-run 噪声。
- **问题（open）**：①规划层放大未拆到模块级（plan head 的 κ-compose/投影 vs LQR 参考消费）——B 已定位、未细拆（不扩审计，留待 33C 后）；②od_pool 泄漏未解（v2 Gate 2 保留"泄漏不增"门）；③收费站静止障碍感知通道盲区（岗亭仅 others 5 维走廊段；**others 数值未记录** ⇒ "策略是否收到该信号"未验证；v2 评测建议分层读数）；④ds-progress 单调率口径（off-manifold）待 per-arm 探针修正；⑤warp 仅证"零-shot 无效"，"重训后有效"未测；⑥33C 训练**待授权**。
- **决策与下一步**：**decision=GO_EGO_ACTION_CONDITIONING_V2**（不自动进入训练）。**33C 建议（待 CG）**：**B0/B1 两臂**（B2=B1+warp 因 NO-GO 删）；B1 = 特征 `[ds,dθ,κ]` + 隐藏层正常初始化 + 输出层零初始化 + **固定增益 1** + ego 物理监督 `{delta_progress, delta_speed, delta_heading}`；硬门沿用 GPT 列表（top-1≥0.35 / top-3≥0.85 / 单调≥0.90 / 接口恶化≤3% / lane 不退化 / 泄漏不增 / succ≥0.58 / off≤0.14 / coll≤0.08 / tmo≤11 / rc≥0.862；恶化>5% 或碰撞再恶化即停）；~35 min/臂。
- **产物**：`docs/round33_report.md`、`runs/round33/*`（a1_ckpt_audit / a2_collision / a5_overshoot_decomp / a6_noise / b_warp / b_warp_compat）、脚本 `.slim/deepwork/p1_v2/round33/*`。

### 5.43 round-33C 批准受理（GPT `49e04d7`，487 行）：目标改写 + 四任务 + 长期路线图；我方 4 条补充；33C 开工

- **受理**：①**表述修正采纳**——"多步正反馈"→"imagined latent 变化被 PolicyBackbone/PolicyHead **一次性纵向过度解释**"（与 A5 的 B 项 +117.9% 一致）；②**B0/B1**（B1=特征 [ds,dθ,κ] + 隐藏层正常初始化 + 输出层零初始化 + **固定增益 1** + ego 物理监督 {delta_progress, delta_speed, delta_heading}；暂不加 warp/OD·LD 动作注入/Policy 分布数据）；**目标改写**="让 WM 对 action 产生物理正确、可被 Policy 稳定消费的状态变化，同时不提高静态障碍 collision"；③**四任务**：AC V2 / **分层闭环评测**（收费站/静态障碍/merge·ramp/curve/普通道路）/ **WM→Policy 四增益链**（G_WM=‖Δz_ego‖/‖Δa‖、G_backbone=‖Δh‖/‖Δz_ego‖、G_head=|Δds|/‖Δh‖、G_total=|Δds_plan|/|Δds_in|）/ **安全门**（硬门沿用 + **收费站碰撞 ≤ 基线**）；④**三分支决策树**（全门过→60/40 Policy 分布适配；动作过但接口恶化→Policy 接口稳健训练 π(z_WM)≈stopgrad π(z_real)；仍收费站碰撞→静态障碍显式表示/走廊占用/path-conditioned 纵向剖面；不启 RL）；⑤**长期路线图登记**（R33C→R34 接口一致性→R35 动作表示迁移（横向 κ 路径 + 纵向速度/进度）→R36 结构化 WM（warp 重训 + 显式 anchor + 学习残差）→R37 DAgger→R38 MoE→R39 PPO；依赖顺序不颠倒；**现行冻结不因路线图自动解除**）。
- **我方补充（4 条）**：
  1. **静态障碍簇二分**：6/13 中 **3 例对象在 OD 可见**（同车道静止/排队车 4.5–5.1m）⇒ 缺口不止"表示缺失"，**纵向条件化（停车距离）至少同等重要**；安全门诊断须区分"表示缺口（岗亭类）vs 条件化失败（OD 可见类）"。
  2. **四增益协议化**：有限差分口径（δds=±0.25、与 P2 同归一化；逐 horizon k=1..6；配对 recorded 状态；G_total 输入扰动=决策步 0 动作）；**B0 与 B1 同协议对照**——G_WM 合理而 G_backbone/G_head 大 ⇒ 接口为主（分支 2 判据）。
  3. **分层标签已验证可用**：`episodes.csv.primary`（tollgate/ramp_in/merge/curve/… 11 类）+ `dec__road_class`；B0 分层基线先用既有录制建立（零新跑），B1 沿用同映射。
  4. **warp 登记同意 + 边界注记**：R36 重训验证须报告"锚残差 vs latent consistency"分离项；"零-shot 不可判方向"结论保留。
- **开工**：lane A（v2 生产 + 训练器：3 维特征/删 α 固定增益 1/delta_speed 目标 + Gate 0 逐位 + smoke）；lane B（只读：四增益链探针 + 分层评测工具；先出 B0/AC1 基线数）。B1 训练在 lane A 落地后由主控启动。

### 5.44 round-33C 执行完成（AC v2 = B0/B1；四任务）：**接口修复（Gate 3 −10~−12%）但硬门 6/11 未过**；不晋级；分支 3 适用（报告 `docs/round33c_report.md`）

- **执行**：生产 v2（3 维特征 + 删 α 固定增益 1；Gate 0 6/6 PASS、966 tests）；训练器 v2（ego 物理目标 delta_speed）；评测链三工具适配（自证 BIT-EXACT）；B1 训练 22,532 步/4ep（latent 0.0406 / ego_phys 0.0080）。
- **硬门**：✓ = top-3 0.914 / **接口全改善**（E_if −10.1%、E_action −11.9%、overshoot −12.3%、E_real −4.3%）/ lane 保留改善 / coll 0.06（toll 2→1）/ tmo 9；✗ = **top-1 0.192**（<0.35）/ **单调 0.60**（<0.90）/ **泄漏 od_pool +46.9%、agg speed +19.1%**（>10%）/ **succ 0.56**（3正/3负）/ **off 0.20**（+3 新：merge×2 + tollgate×1）/ rc 0.855（<0.862）。
- **对照 AC1**：AC1 的接口恶化（+17~21%）被 v2 修复为改善（−10~−12%）；FD 比值 38.7→**2.98**（AC1 7.34）；ΔR²_WM 全臂最佳（+0.027/+0.030）；**G_total(h6) 0.348→0.157（−55%，仍 ×7.3 base；放大仍在 WM 侧）**。
- **分层/收费站**：tollgate **0 succ（两跑皆然**；coll 2→1、off 3→4；min_ttc 1.43→2.28）；**merge off 2→4**；t_intersection succ 3→5（+0.4）。7784 末步纵向余量 **+8.84m 转正仍碰撞** ⇒ **非纯纵向（路径级嫌疑）**。
- **分析**：v2=**稳定器而非推进器**（接口修复、coll/tmo 改善；succ 未动、off 回退、top-1/单调/泄漏未过门）；单调 0.60 与 FD 2.98 组合="敏感度已好、方向性未好"；泄漏（od_pool +46.9%）说明"action 只进 ego transition"不足以防消息传递串扰。
- **决策建议（按三分支）**：分支 1/2 不适用；**分支 3 适用**（仍收费站碰撞 ⇒ 静态障碍显式表示/走廊占用/path-conditioned 纵向剖面；不启 RL）。我方补充：①先做收费站/merge **路径级只读归因**（plan κ/dθ 剖面 vs 几何）；②泄漏防线重估；③**B1 不晋级**（保留为诊断参考）。
- **产物**：`docs/round33c_report.md`、`runs/round33/canary_v2/*`（ckpt/eval_b1/gate3_b1/gate4_b1）、`runs/round33/{c_gain_chain_b1,c_stratified_b1}.*`。

### 5.45 round-34A 批准受理（GPT `82e815b`，533 行）：B1 结案不晋级；启动 34A 只读审计 → 34B 新动作表示；我方 5 条补充；34A 开工

- **受理**：①**B1 结案**（不晋级/不进 60/40/不启 DAgger·MoE·RL；ckpt 保留为诊断分支）；②**34A**（只读）：**Task1 收费站/merge 连续 footprint 路径归因**（plan 六步几何/30 点插值/扫掠/footprint/首次无效点/LQR preview 命中；**6 类分类**：路径撞障碍/路径越界/路径可行但 LQR 出界/目标通道错误/障碍未进模型/进模型但 Policy 未响应）+ **Task2 静态障碍输入契约审计**（岗亭是否进 obs·字段·距离响应 + zero/mirror/distance/corridor 干预 + Encoder/Policy 逐层响应）+ **Task3 hooks**（输入/几何/网络/rollout/数据契约；默认关 + bit-exact）；验收：≥90% 归因/坐标系单位 mask 可验/干预可复现/开关 bit-exact；③**34B**（34A 后）：最小原型 = 横向 3 曲率控制点（near/mid/far）→ κ(s) 积分路径 + 纵向路径条件化 v_ref[6]/s(t)；**外部接口保持**（ds_t、dθ_t=κ_t·ds_t）⇒ LQR/环境/评测不变；④**35**：结构化 WM（ego-motion 补偿 + 显式 anchor + 学习残差）**重训项保留**（不得用 zero-shot NO-GO 永久关闭）；⑤Stage A/B 交替适配（WM 步/Policy 步/DAgger 分标签）入实施计划。
- **我方补充（5 条）**：
  1. **可证伪性**：6 类分类须把 **LQR 跟踪误差**与**目标通道选择**列为可能主导类——**不预设"静态障碍表示缺失"为根因**（R33 已证 6/13 中 3 例对象 OD 可见）；若跟踪/通道主导，34B 设计与优先级须相应调整。
  2. **others 记录**：Task2 需把 `others`（静态走廊/岗亭）原始数值纳入录制（recorder 扩展，默认关 + bit-exact）——否则"岗亭是否进 obs"无法验证（R33 已知盲区）。
  3. **hooks 落地范围**：以**探针侧**为主（只读脚本），生产只加必要几何/rollout 挂点；"新模块强制 hooks"从 34B 起执行。
  4. **泄漏因果性**：od_pool +46.9% 是否**伤害**闭环尚无因果证据——34A 顺带"逐 episode 泄漏 vs 结果"相关（复用 lane B 工具，零新跑）；弱相关 ⇒ R35 优先级下调。
  5. **34B 预注册**：外部兼容 + 可回滚 + **与现 policy 同预算 A/B 对照**（同数据/同 epoch，避免表示迁移与训练预算混淆）。
- **开工**：lane D1（路径级归因）、lane D2（静态障碍契约）、lane D3（泄漏因果性 + hooks 注记）。

### 5.46 round-34A 执行完成（三 lane）：**失败=路径级（通道选择 6 + 越界 6 + 撞障碍 6；跟踪/表示缺失/未响应 0）；覆盖率 94.7%**；34B 建议范围待确认（报告 `docs/round34a_report.md`）

- **D1（19 ep：b1 11 + base 8）**：覆盖率 **18/19 = 0.947**；分类 = **目标通道选择错误 6**（收费站：路径最近接近位姿落在岗亭车道、未完成必需变道、同 road 有空闲车道）+ **路径越界 6**（merge 为主：路径首无效比执行早 26–837 步、路径余量 ≤0.01m vs 执行 1.0–1.5m）+ **路径撞障碍 6**；**path_ok_track_out / static_missing_from_input / obstacle_in_input_no_response 全 0**；未归因 1（base/7579 超时）。**Q1 收费站路径级 10/10（跟踪级 0/10）；Q2 merge 路径级 6/7（跟踪级 0/7，跟踪残差仅 0.09–0.90m）**。自证：A 链 lane_lat 重算 **0.0m 误差**；B 链 3 ep **exact**（preview 1326/1326、obs 静态段 1326/1326 逐位一致）。**发现既有工具 bug**（`forensics_closed_loop._make_on_lane_fn` heading filter）已绕开并记录。
- **D2（8 runs）**：岗亭**确实进 obs**（`others[16:21]` static 5 维）**且被消费**（置零 → enc_ctx relΔ 0.71–0.81、latent 0.13–0.19），**但 PolicyHead 衰减到 |Δds| 0.031–0.060m 且方向反转（presence→加速；corridor_in 反向加速）** ⇒ 判定 `present_static_5d / consumed_head_attenuated (7/8) / presence_increases_ds (8/8)` ⇒ **"表示缺失"归因不成立**（static 仅 5 维摘要、无形状/精确横向 ⇒ 形状级避让仍需新表示）；自证全过（默认关 bit-exact、复跑 812 步×6 数组 0 差异、干预可复现）。
- **D3**：**泄漏伤害闭环 = 无证据**（预注册主口径；附秩口径 12/12 弱不利相关、中位 ρ=−0.335、主要来自 WM 侧——非因果，建议干预对照结案）；**hooks 38 项**：纯缺口=effective rank / slot permutation / ckpt hash / schema version；部分=normalized obs、LD slot identity、frame/units/valid_range、ego pose delta、anchor warp 前后、**部署侧 PolicyBackbone latent / plan 累计 pose / LQR reference 遥测**；M1–M7 最小实现清单已列。
- **综合结论**：①失败是**路径级**（跟踪 0、表示缺失 0、未响应 0）；②最大类=**通道/车道选择（6/19）**；③静态障碍信号需**符号修正 + 幅度提升**（34B path-conditioned 纵向可解）；④泄漏非因果（R35 优先级不上调）；⑤34B 输入条件：**route/lane-target token + 走廊占用 + 路径上障碍距离**；**LQR 保持不动**。
- **34B 建议范围（待确认）**：横向 κ 3 控制点（near/mid/far）→ κ(s) 积分路径；纵向 v_ref[6] 路径条件化；外部 `dθ_t=κ_t·ds_t` 兼容（LQR/环境/评测不变）；同数据/同 epoch **A/B vs 现 policy**；默认关/可回滚。
- **产物**：`docs/round34a_report.md`、`runs/round34/{d1_path_attribution,d2_static_contract,d3_leakage_causality}.*`、`runs/round34/d3_hooks_note.md`。

### 5.47 round-34B0 批准受理（GPT `5fb3718`，503 行）：34B 修订采纳（弃"走廊摘要"核心化）；nav 排模答复；34B0 开工

- **受理 + 修订采纳**：①**"软走廊"定性**：`others[16:21]`=人工静态障碍摘要（非完整走廊）；它**没解决规划问题**且被学成**危险 shortcut**（presence→加速/入走廊加速/置零减速）⇒ **不强化外部"软走廊策略"**；长期=**显式结构化静态对象 slots**（type/rel x·y/heading/L·W/mask/置信度）+ 网络自判候选路径冲突；②**旧摘要三阶段**：阶段1 保留但降级诊断（zero/permute/present-only/gap-only 严格消融）→ 阶段2 新模型并存 + 高 dropout（50→80→100%）→ 阶段3（闭环不退化 + 收费站成功提升 + 干预方向正确）才删；③**34B 修订（接受）**：横向=**多候选**（保持/左/右，route+target-lane 条件化）；纵向=对每条候选读**几何层客观量**（静态距离/TTC/越界/可通行长度/曲率）→ v/s 剖面 → score head 选择；**关键区别**：net 生成候选、几何层算客观关系（确定性算子≠外部策略）、net 学选择与配速；④**34B0/34B1/34B2** 顺序 + **硬门**（输入职责：slots 重放误差 0 + 旧摘要置零不崩；候选覆盖：收费站/merge 失败集 ≥95% 存在无碰撞/不越界候选；选择：oracle top-1 ≥70%/top-3 ≥95%、通道错误 ↓≥50%；纵向：障碍近→v 单调降 ≥95%、曲率增→速度不升、TTC 降→进度不增；闭环：succ ≥0.62、off ≤0.10、coll ≤0.08、tmo ≤0.12、rc 不降）。
- **nav 排模答复（GPT 提问）**：当前给 net 的导航=**路线级（route-level）**，非车道级——`NAV_DIM=11` = 3×路线顶点（自车系 x,y；顶点=路段终点）+ 4 维命令（forward/left/right one-hot，弦向叉积）+ route_completion；来源 `net/mem.py::nav_features_from_world`（与 env NavChannel 同口径、t0 冻结、detach）。**车道级信息在 LD 通道/semantic**（`ld_lane_role`/`ld_route_associated`/`ld_offset_s_norm` + R21 target-lane token），**不在 nav** ⇒ 34B1 的 "target-lane 条件" 需**新增**。
- **我方补充（3 条）**：①**slots 提取同属外部先验**——原则="**暴露几何、不暴露决策**"（slots 提供形状/位置/mask；不做"哪个障碍相关/是否安全"的压缩）；②**候选覆盖证明的构造**：候选集来源=专家轨迹（`dec__expert_action` 合成）+ 车道中心线左右偏移，先证 ≥95% 覆盖再训 selector；③**34B0 复用 D1 工具**（footprint/几何评估器同口径，避免分叉）。
- **开工**：lane E1（静态摘要因果消融，闭环 5 变体）、lane E2（显式静态对象 slots 提取器 + 自证）、lane E3（候选路径几何评估器 + 覆盖证明）。

### 5.48 round-34B0 执行完成（三 lane）：**覆盖门 ✅（收费站 10/10、merge 7/7、严格 17/17）**；摘要结果层依赖≈0；34B1 建议=车道锚定多候选（报告 `docs/round34b0_report.md`）

- **E1（摘要消融 5 变体 × 50 ep）**：表示/Encoder/动作层确实消费（ctx relΔ 0.58–0.78、latent 0.27–0.32、|Δds| 0.11–0.17m）**但结果层 ≈0**（zero 不减少收费站 coll 2→2/off 3→3；**0/50 结果翻转**；static 段仅 5/50 ep 非零）⇒ **摘要方向错误非充分/主导原因**；D2"presence→加速"仅在 **raw 头**复现（−8.9mm 5/5），**部署 ds（kappa-compose）符号混合**、闭环去摘要反而更快（near20 +0.57 m/s）⇒ **旧摘要可废弃但非杠杆**。
- **E2（slots）**：`e2-static-slots-v1`（8×8：type/rel_x·y/heading/L·W/mask/dist；60m 全部静态对象、cap 8；**暴露几何不暴露决策**）；收费站几何=岗亭面↔路缘 3.5m、跨向岗亭对 7.0m、边界多边形可表达；自证：重放 **16/16 零误差**、自车系 4691/4691 逐位、领先摘要 10–20 步；**merge 8/8 无静态对象**（与 D1 一致）。
- **E3（覆盖证明）**：评估器（复用 D1，**1857 步 × 9 字段 0 不一致**）；**收费站 10/10 有 ≥1 无碰撞候选、merge 7/7 有不越界候选、严格合取 17/17、逐步率 1.000** ⇒ **问题是选择/生成而非存在性**；最佳候选=**车道锚定**（`lane_full_p` 17.5–37.1m 静态余量 0.824m / `lane_center` 14.3–20.6m）；plan 失败证据=压白实线 8.56–8.93m / 岗亭重叠 static_clr=0.0。
- **34B1 设计输入（我方建议，待确认）**：候选参数化**以车道中心线为锚**（δ∈{0,±0.5w,±1.0w}；κ 控制点用于平滑/连续性）；多候选（当前/左/右）+ route/target-lane 条件（nav 现为路线级、需新增车道级）+ score head；**纵向监督用部署 ds 口径**（semantic_off/kappa-compose，非 raw 头）；merge 靠横向路径（无静态对象）；外部 `dθ=κ·ds`、LQR 不动；同预算 A/B。
- **产物**：`docs/round34b0_report.md`、`runs/round34/{e1_ablation,e2_static_slots,e3_candidate_geometry}.*`、`runs/round34/e2_static_slots/*.npz`。

### 5.49 round-35 批准受理（GPT `07f8c4b`，479 行）：核心转向=优先重构 Policy 规划能力；四工作包 + 全套门；我方 6 条补充；开工

- **受理（方向成立，与 34A/34B0 证据链一致）**：①**转向**：不再单线扩 WM，**优先重构 Policy 规划能力**——外部只提供**道路级 route + 客观几何**；**车道/闸口/变道时机/候选选择必须网络内部完成**；几何算子（footprint/碰撞/距离）属确定性层、**不得输出"最佳车道/闸口/编号"**；旧 5 维摘要不强化；固定 LQR。②**四工作包**：**WP1 导航与 target-lane 责任审计**（输入清单 + 三干预〔改 target lane / 改 route / 移除 target-lane token〕+ 失败三分拆 ≥90%）；**WP2 动态 hooks**（输入/网络消费/rollout/干预桩；默认关 + bit-exact + RNG 不变 + 不写 ckpt）；**WP3 网络内候选车道/路径生成**（候选集=保持/左/右/route 续接；每候选 3 κ 控制点→κ(s) 积分；外部 `dθ_t=κ_t·ds_t` 兼容；外部只给车道几何/连接/静态几何，**不得给 selected_lane/gate/best_candidate**）；**WP4 路径条件化纵向**（每候选读曲率/静态距/TTC/可通行长度/边界余量/导航一致性/当前速度→6 步 s(t)/v(t)）。③**训练流程**：候选覆盖（≥95%/99%）→ IL（多安全正例、分项 loss）→ WM 适配（新表示稳定后）→ DAgger（分标签）。④**门**：导航角色（route 变→评分变；移除 target-lane 后仍能自主选道）；候选选择（top-1 ≥70%、top-3 ≥95%、收费站错误闸口 ↓≥50%、merge/路口 ↓≥50%）；横向（连续/有界曲率/镜像 ≥95%/in-road ≥99%）；纵向（障碍近→速度降 ≥95%、TTC 降→进度不增、曲率增→速度不增）；闭环（**succ ≥0.64**、off ≤0.10、coll ≤0.08、tmo ≤0.10）→ clean150（≥0.60、+5pp、收费站显著非零、merge off ↓≥50%）→ 3 seeds ≥0.68 → DAgger/RL 冲 0.74/0.80。
- **我方补充（6 条）**：
  1. **数据规模前置**：覆盖门要求 ≥100/100/100/200 eps，而 calib50 仅 5/4/~8/…——先做**场景普查 + 字段可用性 recon**（expert5k_v8 计数 + lane/static/dynamic slots 可离线提取性）；数据不足时先在 calib50 失败集做机制原型（E3 已 100% 覆盖），**避免"覆盖门"因数据不足阻塞计划**。
  2. **审计须先答"外部规则错还是 Policy 错"**：补充统计**外部 target-lane 规则在失败集的正确率**；若规则常错 ⇒ "外部决定"本身是缺陷源（与原则一致）；**修复现有规则 bug 属"移除错误先验"、非"新增外部决策"**——是否允许作短期止损待裁。
  3. **hooks 按需最小实现**（复用 D3 M1–M7；随工作包增量，避免一次性大工程）。
  4. **候选参数化 = 车道锚定**（E3 证据：胜出候选全为 `lane_full_p`/`lane_center`；κ 控制点仅用于平滑/连续性）——与 WP3 候选集一致，显式写死。
  5. **merge 属"横向路径生成"而非"选道"**（E2/E3：merge 无静态对象、`lane_center` 可行）⇒ 横向头须先保"本车道中心线续接"再谈变道。
  6. **现 policy 保留为 fallback**（A/B + 默认关）——大重构期间闭环不倒退。
- **开工**：lane F1（导航职责审计）、lane F2（hooks 最小实现）、lane F3（场景普查 + 字段可用性 recon）。

### 5.50 round-35 审计执行完成（F1 导航 + F2 hooks + F3 数据）：**外部 target-lane 规则 = toll 失败集 0/10 正确的缺陷源**；merge = 横向路径几何；WP3 建议待令（报告 `docs/round35_report.md`）

- **F1（5 arms × 18 ep 闭环；bit-exact 18/18）**：①**失败三分拆 17/17 = 1.00**——`external_target_wrong` **10（全部 toll）**、`policy_path_wrong_target_ok` 7（全部 merge）、第三类 0（3 例被优先级吸收）；②**外部规则正确率：episode 级 0.412；tollgate 0/10 = 0.0**、merge 7/7；③三干预：`swap_target` → μ/plan max|Δ| 1.35/1.94m、**merge 3 例失败→成功**（b1:7559/b1:7608/base:7559）、toll 变道意图 0→1.0 但未完成；`route_rot` → 强消费（1 成功损失）；**`no_token` → 仍自行变道但 0 成功**（自主雏形、不可靠）；④字段清单：`target_lane`=**规则生成且不查障碍**（lane_semantic.py:355-405）；nav/route=道路级；ref_lanes 仅被规则消费。
- **F2（hooks）**：默认关 P1–P6 全 PASS；覆盖 input 40 / network 351（含 **effective rank** 全仓首例）/ rollout 17（含 **LQR reference**）/ lqr 11；D3 M1–M5 ✅、M6 ◐；**候选分数接口就绪、无候选头**（`num_anchors=0`）。
- **F3（数据）**：expert5k **455/441/906/907 ≥ 目标**；缺口=独立 v7 场景（54/55/109/110）+ policy 失败集（21/16/28/28）；字段矩阵 + 数据计划（**DP-1 离线提取 ~2002 ep + DP-4 static slots 先行**；覆盖门不因总量阻塞）。
- **综合**：①toll 失败 = **外部错误 target + Policy 跟随**（原则获证据支持）；②merge = **横向路径几何**（target 正确）；③无 token 自主选择不可靠 ⇒ 网络内候选+选择（WP3）为正解；④数据/hooks 就绪。
- **我方建议（待裁）**：**诊断臂 = 障碍感知 target-lane 规则**（默认关、仅诊断，量化"正确 target 下 Policy 天花板"；**不作生产路径**）；**WP3 设计输入** = 候选含**收费站空闲闸口车道**（从可见车道生成）+ 车道锚定横向 + merge 保本车道中心线续接；数据 DP-1+DP-4 先行；**WP3 最小原型待 CG 确认**。
- **产物**：`docs/round35_report.md`、`docs/navigation_scope_audit.md`、`runs/round35/{f1_nav_audit,f2_hooks,f3_data_census}.*`。

### 5.51 round-WP3 批准受理（GPT `acd22c1`，468 行）：启动 Network-Internal Lane and Path Selection；我方 3 条补充；开工

> **〔作废/回退（2026-10-15，CG 指令）：`acd22c1` 方案不佳——本节受理与 G1/G2/G3 队列**全部终止**；G2/G3 已中止（产物 `runs/round35/g2_diag_target/`=partial、脚本留档不再执行）。替代方案见 §5.52（直接执行 `7c0186f` 执行优先）。〕**

- **受理**：①**批准 WP3**（命名 Network-Internal Lane and Path Selection）；②**输入边界**（外部只给 route/nav、当前·下一路段可见车道集合、车道几何·拓扑连接、静态·动态 slots、ego 态；**禁 selected_lane/selected_gate/best_candidate/recommended_lane_change_direction**）；③**候选集**（保持当前/本车道中心线续接/左/右/route 续接/收费站可见闸口车道；**不得由规则过滤成"最佳候选"**，所有合法可见且 route 拓扑兼容的候选都交给网络评分）；④**横向表示 = 车道中心线锚 + 网络预测 near/mid/far 曲率残差**（非从零生成；"最新候选覆盖实验已证最佳候选全部车道锚定"）；⑤**候选评分输入**=route 一致性/拓扑/静态最小余量/TTC/边界余量/曲率·变化率/变道距离/当前速度/候选纵向；⑥**逐候选独立 s(t)/v(t)**（不共享同一 ds）；⑦**token 三阶段迁移**（教师辅助 dropout → 30/60/90% → 部署删除）；⑧**多正例标签**（碰撞/越界/不连 route=负；安全且连 route=全正；专家选择仅其一，不强制唯一）；⑨**门**：数据（≥95/95/95/99%）→ 网络响应（route/静态位置/去 token/镜像）→ 选择（top-1 ≥70%/top-3 ≥95%/错误 ↓≥50%）→ 闭环（**0.64**/off ≤0.10/coll ≤0.08/tmo ≤0.10）；未过不进 DAgger/MoE/RL；⑩**诊断臂锁定**（默认关/仅诊断/不进生产/不作 baseline）；⑪WM 暂不扩容（新表示稳定后适配）；⑫**数据提取与 hooks 先行、训练后置**。
- **我方补充（3 条）**：
  1. **分阶段原型 WP3a/WP3b**（降风险）：**WP3a = 候选生成 + 评分器 + 车道锚定路径（残差=0，纯中心线跟随）+ 复用现 ds**——先测"选择+车道跟随"核心假设（toll 通道选择 / merge 压线是否即被修复）；**WP3b = 曲率残差头 + 逐候选纵向头**（学习绕障与配速）。先 A 后 B，各自可回滚。
  2. **逐候选纵向目标口径预注册**：专家数据只给"实际执行"剖面；非选中候选的 s(t)/v(t) 监督目标需明确定义（建议二选一：仅选中候选监督 + 评分器用几何特征惩罚不可行；或对可行候选用几何约束的最速安全剖面作软目标）——避免引入隐式规则。
  3. **复用既有 K-anchor 机制**（recon 结论）：`PlanHead` 已有 `num_anchors>0` 模式（`plan = Σ p_k·(anchor_k + residual_k)`；`net/model.py:50-59/342-356/1042-1044/1346`）——**动态车道锚候选可直接接入**（anchor=车道锚定 6 步计划、p=网络评分、residual=曲率残差〔WP3b〕），显著降低实现风险；`plan[:,0]==action_mu` 与 LQR 消费契约的保持方式需在实现中明确文档化。
- **开工**：lane G1（DP-1 离线提取 + 数据门）、lane G2（诊断臂 + 闭环天花板量化）、lane G3（WP3a 骨架：候选生成 + 评分器 + 车道锚定计划；默认关 + smoke）。

### 5.52 CG 指令（2026-10-15）：**终止并回退 `acd22c1` 方案；直接执行 `7c0186f`（执行优先）**

- **终止/回退**：`acd22c1`（Network-Internal Lane and Path Selection）方案作废——§5.51 受理与 G1/G2/G3 队列全部终止（G2/G3 已中止；产物 `runs/round35/g2_diag_target/`=partial〔10/50 ep〕、脚本 `round35/g{2,3}_*.py` 留档不再执行；如需删除另行指示）。
- **直接上 `7c0186f`（336 行；执行优先）**：先证明"**给定合法目标车道后**网络能稳定生成车道内可执行路径并完成保持/变道/merge/收费闸口执行"，再谈自主选道（理由：否则无法区分"选错车道"与"选对但不会执行"）。要点：①**横向**=车道中心线**锚** + 网络预测 near/mid/far **曲率残差**（不从零生成）；②**纵向**=路径条件化（曲率/边界余量/静态距/TTC/速度/nav 进度）→ 6 步 s 或 v；外部 `ds_k=s_k−s_{k−1}`、`dθ_k=κ_k·ds_k`（LQR/环境/评测不变）；③**target-lane 边界**：允许外部合法 target（测执行）+ **token dropout 20%→50%**；外部障碍感知规则=仅诊断上界；④**训练**：横向头/纵向头/必要 PolicyBackbone 末层/几何特征编码；**冻结**：WM/Encoder 主体/Memory/旧 PolicyHead/MoE/ValueHead/LQR；⑤**任务序**：T1 车道保持 → T2 给定目标变道 → T3 merge 续接 → T4 收费站给定闸口执行（**给定正确闸口仍失败 ⇒ 禁入自主闸口选择**）；⑥**hooks**：输入/横向路径/纵向/footprint 连续插值/LQR 对照/网络性状/干预（默认关 bit-exact）；⑦**门**：G0 契约 → G1 保持（succ ≥0.90/in-road ≥98%/碰撞≈0）→ G2 变道（完成 ≥90%/方向 ≥95%）→ G3 merge（越界 ↓≥50%/续接 ≥90%）→ G4 收费站执行（碰撞 ↓≥80%/换道 ≥85%/succ 目标 ≥60%）→ G5 纵向单调 → **G6 闭环（0.62/off ≤0.10/coll ≤0.08/tmo ≤0.10）** → clean150；⑧**禁**：自主选道训练/删 token/WM 修改/warp 重训/DAgger/MoE/PPO·GRPO/LQR 改/扩容。
- **我方补充（4 条）**：
  1. **G1 门子集口径**：需定义普通道路子集与样本量（calib50 普通段仅 9 ep）——建议专用 normal 集（clean150 normal 子集或 50-ep normal spec），否则门欠功效。
  2. **given-target 监督规格**（并入 H2 数据提取）：逐决策步 target 标签（保持=自车道/变道=专家未来轨迹收敛车道/merge=本车道或续接/toll=专家闸口车道/路口=转向目标车道）+ 专家路径在**目标车道锚系**下表达。
  3. **执行先行的两分支都要记录**：F1 已示 toll 变道意图 0→1.0 但未完成 ⇒ 若 T4 给定正确闸口仍失败，失败在**执行头**（符合本阶段设定）；若成功率高，则收益主要来自选道——G4 逐项记录。
  4. **G2 残留**（终止前 10/50 ep）：部分 ep 行为改变（b1:7784 collision→out_of_road、b1:7789 off→off）——**不构成结论**（未完成且方案作废）；诊断上界规则依 `7c0186f` §二 3 仍允许作 G4 预测，若需可复用脚本重跑全量。
- **开工**：lane H1（Gate-0 骨架：车道锚定横向曲率头 + 路径条件化纵向头 + 组合层；默认关/逐位）、lane H2（given-target 监督数据规格 + 数据门）、lane H3（新头 hooks 扩展）。

### 5.53 round-36 执行（H1/H2/H3 全收口）：执行优先骨架 + hooks + given-target 数据规格就绪；DP-6 提取开工（报告 `docs/round36_report.md`）

- **H1 骨架**（`net/path_head.py` 791 行 + `net/model.py` +134；**已入库 `6944436`**）：车道锚 κ_ref（3 控制站位）+ near/mid/far **曲率残差**（tanh×0.1、末层零初始化 ⇒ 初始 κ=锚曲率）+ 路径条件化纵向头（→6 步 s）+ 组合层（`plan[:,0]==action_mu` 逐位、`dθ=κ̄·ds` 精确积分）；**Gate 0 8/8**（默认关三口径逐位 max|Δ|=0、旧 ckpt strict 0/0/0、LQR 契约 interp ≤3.2e-6、**966 passed**）；smoke 3/3（梯度仅 allowlist 18 键=`path_head.`+`plan_head.fusion.2.`；dropout 0.2→0.5）。Caveats：曲率空间锚定（起点不搬到锚线）、dθ 截断、启用后 rollout 尾段由组合层一次给出。
- **H2 数据规格 + 数据门**（`h2_data_spec.*` + 7 样例）：标签规格（T1 keep/T2 change/T3 merge/T4 gate/T5 junction；专家路径在**目标车道锚系**下的 lat_err/head_err/s_anchor/valid）；计数：T1 907 ep/53,606 步、T2 2,579 ep/11,818 步、T3 441 ep/24,746 步、T4 闸口 **317/455 ep（69.7%）**/15,543 步（gate gap 3.5m、匹配 p50 0.084m）、T5 906 ep；全库 **265,801 usable、99.98% 标签、0 error**；自证 15/15 与 F3 一致；新增 **DP-6**（离线提取；DP-2 不再是前置）。
- **H3 hooks**（`h3_hooks_ext.py` 子类化复用 F2；555 键）：§五 7 类全覆盖（横向/纵向/**footprint 连续 ≤1m**〔非 6 点〕+执行侧/LQR 对照四象限〔样例 60 双好/0 路径错/**2 路径好但跟踪错**〕/dropout 状态/8 臂干预）；默认关三方逐位、开关只增不改、前向 max|Δ|=0、RNG/ckpt sha 不变。
- **关键 caveat（训练前必须处理）**：**T4 专家多为晚换道**（换道步距闸口 p50 25.8m）⇒ 直接模仿会编码"晚换道"；G4"提前换道 ≥85%"需训练侧单独处理（课程/加权/首步约束）。
- **待裁**：①G1 普通道路子集口径（专用 normal 集）；②T4 晚换道策略。
- **下一步**：**DP-6 全量提取（I1 开工）** → 训练接线（T1→T2→T3→T4）与 G1–G6 评测。

### 5.54 R36 实施方案登记版（执行优先；随进展更新）

- **阶段与门（顺序锁定）**：① **I1 = DP-6 全量提取**（T1–T4 数据集：target 标签 + 目标车道锚系残差 + 纵向外源几何；8 workers；**进行中**）→ ② **J1 = 训练接线**（数据加载 + `path_head` 训练循环 + ckpt/metrics；smoke 用 `h2_samples.npz`）→ ③ **T1 保持 → G1**（succ ≥0.90 / in-road ≥98% / 碰撞≈0 / tmo ≤5% / 曲率 ≤ expert P95×1.25 / 饱和低于基线）→ ④ **T2 变道 → G2**（完成 ≥90% / 越界 ≤5% / 方向 ≥95% / 回正 ≥90% / 反复换道 ≤5%）→ ⑤ **T3 merge → G3**（计划越界 ↓≥50% / 实际 off ↓≥50% / 续接 ≥90% / 跟踪失败率不上升）→ ⑥ **T4 闸口 → G4**（覆盖 ≥95% / 岗亭碰撞 ↓≥80% / 提前换道 ≥85% / succ 目标 ≥60% / 碰撞 ≤ 基线）→ ⑦ **G5 纵向单调**（95%/95%/90%/90%）→ ⑧ **G6 闭环**（0.62 / off ≤0.10 / coll ≤0.08 / tmo ≤0.10 / rc 不降）→ clean150 → 3 seeds。
- **实现约束**：可训 = `path_head.` + `plan_head.fusion.2.`（18 键）；冻结 = WM/Encoder 主体/Memory/旧 PolicyHead/MoE/ValueHead/LQR；token dropout 0.2→0.5；生产改动一律默认关 + 逐位（Gate 0 8/8 已过）；外部 `dθ=κ·ds` 兼容（LQR/环境/评测不变）。
- **禁**（`7c0186f` §七）：自主选道训练 / 删 token / WM 修改 / warp 重训 / DAgger / MoE / PPO·GRPO / LQR 改 / 扩容。
- **进展（2026-10-15 上午）**：H1 ✅（已入库 `6944436`）、H2 ✅（数据门 15/15）、H3 ✅（hooks 555 键）；**I1 进行中**；J1 已派发。
- **待裁**：①G1 普通道路子集口径（建议专用 normal 集——calib50 普通段仅 9 ep）；②T4 晚换道策略（课程/加权/首步约束）。

### 5.55 round-36 修正（GPT `ac78f76`，529 行）：**认账——H1 `PathExecutionHead` 结构性偏离 Route A**；受理 8 条修正 + 5 条细化；`StructuredPolicyHead` 重构开工

- **认账（偏差事实）**：H1 `PathExecutionHead` 启用时**一次输出六步 plan 并接管 rollout 尾段**（`plan_override` 模式）⇒ ①事实上的**第二轨迹生成器**；②**绕过 Policy–WM 自回归**（后五步不再读逐步更新的状态）；③`plan[:,0]==action_mu` 只是表面兼容（a0 旧头、a1–a5 新头）。H1 caveat 里记录但未作为结构问题上报 ⇒ **疏漏认账**，方向以本修正为准。
- **受理 GPT 8 条立即要求**：①暂停把 PathExecutionHead 接入正式 rollout 训练；②重构/改名 **`StructuredPolicyHead`**（`lateral` branch + `longitudinal` branch + deterministic composition）；③PolicyHead 唯一动作生成者；④每 WM step 只消费当前动作 `a_k=(ds_k,κ_k)`、`dθ_k=κ_k·ds_k`；⑤`a_{k+1}` 必须由同一 PolicyHead 读 `z_{k+1}` 重新产生；⑥`κ_mid/far` 与未来 progress 仅辅助意图/监督；⑦保留曲率积分/footprint/LQR/数据标签/hooks；⑧产出 **`docs/architecture_contract.md`**（唯一生成器 / WM 输入输出 / 六步 rollout 调用图 / 辅助与主边界 / 训练部署同构）。
- **受理 §七 5 条细化**：①纵向头不再默认一次六步（**当前步 ds 为主**，未来仅辅助）；②`ds_init = clamp(v_current·dt, bounds)` + 专家 ds P50/P90 校验（替代固定 3.5m）；③κ 残差限幅**数据驱动**（专家残差 P95/P99 + 速度条件化 `κ_max(v)`，替代固定 0.1）；④T4 晚换道不用人工首步约束：早期可决策状态加权 + 目标方向/路径进度监督 + **"剩余距离可行性"损失**（DAgger 后置）；⑤正确闸口仅课程/上界、**不进部署输入**。
- **门**：**架构契约门**（生成器数=1 / a0..a5 同参数 / 每步读 z_k / WM 调用 5 次 / 辅助不进 LQR）+ **梯度门**（当前动作 loss→PolicyHead；WM loss→WM；辅助 loss→辅助分支）+ **消融门**（旧 PolicyHead+WM rollout vs Structured+WM rollout vs 一次性诊断臂〔仅诊断上界〕）+ 闭环门（分层同前）。
- **我方执行注记（3 条）**：①**imagined 步的外源几何来源**（实现关键）：k≥1 的静态几何由"t0 静态地图 + 想象位姿"确定性重算；动态 TTC 由 WM `od_pred` 或保持当前步——写入 architecture contract；②现状注记：生产 rollout 的 **K-anchor 模式同样消费 t0 plan**（`anchor_plan[:,k,:]`）——本次只对**新结构化头**执行"每步重算"；是否同步收紧 K-anchor 模式待裁；③J1 训练器已通知按新契约调整；I1 数据继续有效（每步局部标签由 trainer 从锚系专家路径派生）。
- **开工**：lane K1（重构 `StructuredPolicyHead` + 契约门 + `docs/architecture_contract.md` + ds/κ 数据驱动 + 梯度门 + 消融脚手架）。

### 5.56 round-36 车道保持阶段受理（GPT `0912cb9`，536 行）：5 条代码批评核验（4 成立 + 1 已被 J2 覆盖）；**我方 3 条反驳/修正**；执行补齐待令

- **受理（核验成立，代码证据）**：①旧 `PolicyHead` 在 structured 模式**仍前向**（`net/model.py:1624` 调 `self.policy`；其输出仅存 `policy_action_mu`，**"单 action_mu 进 env/WM/LQR"的硬保证已成立**）⇒ 补"旧头跳过 + 单 action_mu 断言 + 命名统一"；②**`oneshot` 缺 eval/deploy fail-fast**（仅文档串声明"仅诊断"）⇒ 补 `--diagnostic-oneshot` 守卫；③**想象步纵向几何为近似**（`step_geometry`：`static/d2end=clamp(d_t0−progress)`、`dynamic/ttc/nav/speed=hold`）⇒ **不得用于证 G5**（与 K1 caveat 一致，需结构化 WM 才真·路径条件化）；④**首轮 allowlist 同开 `plan_head.fusion.2`**（2 张量）⇒ 补 J1a/J1b 分组；⑤横向缺位置收敛监督——**J2 `_plan_anchor_loss` 已是"沿程 lat/head 残差 vs 专家曲线（评测同口径）"**（部分覆盖）；余：**终点加权 + 变道完成度/回正 + 绝对收敛项**。
- **我方反驳/修正（3 条）**：
  1. **Gate 4「连续 footprint 越界率 ≤2%」需校准**：同口径**专家自身**越界率 = **5.2%**（T1 行 `max|lat_err|>0.85m`；0.85 = 1.75 走廊半宽 − 0.9 footprint 半宽），T1 现为 **5.75%**（baseline 8.75%）⇒ ≤2% 比专家还严、**不可达**；建议改**相对门**（≤ 专家地板 + 余量，或"优于 baseline 且不劣于专家分布"），并在同一子集重算两臂。
  2. **结构化 WM 不打包 gate 本阶段「车道保持闭环」（范围异议）**：证据 = T1（**冻结旧 WM**）已达专家地板；T2 阻塞 = **落点收敛**（监督问题，J2 已消 80% 漂移），**非 WM**。建议**并行**：先闭 G1–G4（终端收敛 + 契约硬化，小时级）；结构化 WM 先做 **Step 1（解析坐标层 + 契约测试，零训练）**，**Step 2 训练由 Gate 2 的 4 个 WM 项在现有 WM 上的实测裁决触发**（horizon 增长 / 尾段 vs frozen / 接口 h3–6 / 物理预测 vs 恒速·静态基线）。
  3. **J1a/J1b 以对照实验落地**：跑 **head-only（16 键）vs 18 键** 的同 G1 口径对照；head-only 若不欠拟合则正式采用（归因干净）；欠拟合再按 **0.1× LR 分组**开 fusion 末层。
- **证据补充（供裁）**：**T1（res ep1 @2000）**：越界 **5.75%** vs baseline 8.75%（−34%）、kappa_ok 0.966；**1-epoch 早停**（@500：ep1 0.038 → ep2 0.168 → ep3/4 0.132）；**T2**：方向 0.994✓ / 反复换道 0.03✓、**收敛 0.458✗（末点 |lat|<0.5）、越界 27%✗**（起步正确 lat_first≈−3.5、落点 ±0.5–2m；多 epoch 无改善）；**Gate 5「普通道路子集」需定义**（calib50 普通段仅 9 ep；≥0.90/off ≤0.02 须先测 baseline 同口径）。
- **补充（AGENT.md 压缩为 56 行）**：保留 13 条守则 + 阶段门表 + 基线表；v3 的**作废/待裁标记、9 条禁令、G0–G6 数字、锚+残差不变量、闭环门 0.62** 不在其中——建议在 argue/契约文档保留（或 AGENT.md 加引用行），以免未来"合法偏离"。
- **执行补齐清单（待令）**：①契约硬化（旧头跳过 / 单 action_mu 断言 / oneshot fail-fast / J1a-b 分组）；②**T2 终端加权 + 完成度/回正监督** → G2 复评；③门校准提案（Gate 4 相对门 / Gate 5 子集定义）交 CG；④**结构化 WM Step 1**（零训练契约层）并行；⑤G6 前置（评测链 `--path-head` + env `lane_anchor_xy`）。

### 5.57 round-36 答复受理（GPT `7dcf6bd`，276 行）：事实更正（代码已入库 `631ad45`）+ 受理"完整系统验收"；3 条注记；L-D 开工

- **事实更正（关键）**：GPT 复核的 diff **仅含 argue 文档** ⇒ 其 `IMPLEMENTATION_NOT_YET_VERIFIED` 系**不完整 diff 所致**；代码重构**已入库 `631ad45`**（`net/path_head.py` 791→1242、`net/model.py` +287、`docs/architecture_contract.md` 148 行；981 insertions），并附：**契约门**（生成器数=1、头调用 6、WM 推进 5、`plan_equals_stacked_head_actions=True`、扰 `policy.*` 不改 plan、辅助零化/扰动 ⇒ plan/mu/traj 逐位不变）+ **梯度门**（主/辅 loss→allowlist 18/18、`main_touches_frozen=[]`、WM loss→`st_gnn` 33 参、`wm_loss_touches_path_head/policy=[]`、`wm_action_stopgrad=True`）+ Gate 0 三口径逐位 + **966 tests**。已产出 **`docs/r36_implementation_evidence.md`**（GPT 10 条要求逐条 → 代码/运行证据映射；真缺口 = #7 warp、#8 运行时断言）。
- **受理**：①**相对门校准**获 GPT 接受（"≤2% 若低于专家地板 ⇒ 相对门：显著优于旧基线且不劣于专家同口径分布 + 统计余量"）✓；②**结构化 WM 不整体后置**——接受"车道保持 = 完整系统（StructuredPolicyHead + structured WM + ego-motion 补偿 + 六步递归 + 固定 LQR）验收"；③阶段 1（坐标 + 架构契约，零训练）先行；④10 条中 #7/#8 补齐。
- **我方注记（3 条）**：①**并行执行**：阶段 1 的 warp 层（L-D，独立模块先行、集成待 L-A）与策略侧修复（L-A 硬化 / L-B T2 终端收敛）并行——后者属阶段 3 口径（冻结 WM），不冲突且小时级；②**坐标门阈值基线校准**（LD warp P90 ≤0.10m、同速前车 P90 ≤0.20m、mirror 等）按"校准一次后锁定"流程先测（L-D 交付）；③**WM 门 4 项先测**（现有 WM：h1–6 lateral/heading 增长 / tail vs frozen / h3–6 接口误差 / 物理预测 vs 恒速·静态基线）——先建立基线，再定结构化 WM 训练的验收口径（避免无基线重训）。
- **开工**：**L-D**（`net/ego_warp.py`：LD SE(2) 逆变换 / OD 先验+warp / anchor schema + 5 项 hook；与 R33-B `_ego_warp_pose_k` 逐位等价单测 + 镜像/物理/恒等单测 + 默认关 bit-exact）；L-A 追加**运行时断言**（policy 实例一致 / policy 调用 6 / WM 推进 5 / aux 不进 LQR / 非 oneshot 不得 plan_override；eval·deploy fail-fast）；L-B（T2 终端收敛）继续。

### 5.58 round-36 执行批次（L-A/L-B/L-C 收口；L-D 在飞）：架构硬化入库 + J1a/J1b 对照（fusion 必须开）+ T2 终端收敛（已超专家）+ G6 前置全绿

- **L-A 架构硬化（已入库 `bbbe541`）**：structured/oneshot **默认不再前向旧 PolicyHead**（诊断经 `set_policy_diag`/`set_path_head_trace` 恢复；`action_logstd` 不再随结构化输出提供，RL 接线前须统一均值/方差来源）；**oneshot 默认拒绝**（`allow_oneshot=True`/`--diagnostic-oneshot`；三工具装载前置守卫）；**5 项运行时契约断言**默认开（同头实例/头调用 6/WM 推进 5/aux 不进 LQR/非 oneshot 不用 plan_override；eval·deploy 违规 `RuntimeError`，训练路径只计数落盘）。自证（对改动前 net 包子进程）：默认关三口径逐位、**structured 66 公共键 max|Δ|=0**（键差恰为两诊断键移除 + `contract_*` 新增）、旧头调用 **0/0**（改前 1/1）、**966 tests**；文档漂移同步修复（`net/path_head.py` 注释 + `architecture_contract.md` 三处）。
- **L-B（训练器 + 对照实验）**：①**J1a/J1b（T1，1ep）**：head-only（16 张量）G1 越界 **0.479** vs full-18（+`plan_head.fusion.2`）**0.058** ⇒ **head-only 明显欠拟合 ⇒ fusion 必须开**（GPT 规则"欠拟合才开"以数据成立；可 0.1× LR 分组）；②**T2 终端收敛**（`--plan-res-terminal-weight 3.0` + `--terminal-coef`：末点 |lat|→0/航向/完成度/回正，change 行 ×2）：**收敛 0.458→0.706**、越界 0.29→**0.221**、方向 0.987、κ_ok 0.893；③**专家同口径地板**：评测 `expert_converge_rate=0.586`（独立复算：变道行末点 |lat|<0.5 仅 ~36%）⇒ **模型 0.706 已优于专家** ⇒ "收敛 ≥90%"属超专家门。
- **L-C（G6 前置）**：三工具 `--path-head {off,structured}`（默认 off 逐位自证：npz 全键逐位；csv/scalars 仅新增字段/墙钟差）+ `--lane-anchor {off,inject}`；注入器（env `select_target_lane` + E3 链采样 0.5m + world→ego t0，视界 max(20, v·3+10)m）**对拍训练锚 6/6 行、388 点 max|Δ|=7.6e-6m**；structured 装载实证（missing 20←K1e2 回填、unexpected 0、18 path_head 键逐位来自 ckpt、`plan[:,0]==action_mu`、6 头调用/5 WM 推进）；**2-ep 闭环烟测通**（anchors 120/120、plan0/ref mismatch 0）；G6 正式命令草案（两套门阈值 + 相对门）。
- **门校准汇总（交 CG）**：①**越界门**（Gate4/六步）：keep 行专家地板 **5.2%**；变道行 0.85m 规则**不适用**（专家 97.7–100% 会"越界"——lat_err 相对目标锚、起点差一车道）⇒ 变道行走廊须用"**目标锚 ∪ 自车道中心**"并集（评测已是该口径）；②**收敛门**：专家 0.586（评测口径）/ ~0.36（末点 |lat|<0.5）⇒ 改相对门（模型 0.706 已优）；③建议统一为"**相对门 = 显著优于旧基线 ∧ 不劣于专家同口径分布 + 统计余量**"（GPT 已接受此形式）。
- **在飞**：**L-D**（warp 层 + 坐标门阈值基线校准）、**L-B run2**（分析报告）。

### 5.59 round-36 warp 层收口（L-D，入库 `9b00761`）：`net/ego_warp.py` + **关键发现=OD 先验速度语义错误**；坐标门阈值校准交 CG

- **交付**：`net/ego_warp.py`（纯函数/无状态/**未接入主路径**）：LD 逆变换 `p'=R(−Δθ)(p−Δp)`（与 R33-B `_ego_warp_pose_k` **逐位等价**）、`assemble_pose_k(off|w1|w2)`（**off 逐位=rollout 现状**）、`AnchorBatch` schema v1 + **契约映射**（↔ `frame.pose`/`od_feat`/`od_anchor`/`ld_anchor`）、`warp_ld/od_anchors`、**5 项 hook（默认关 ⇒ 零张量操作/零 RNG）**；单测 **8/8 PASS**（含**模型内部 `st_gnn.pose_k` pre-hook 18/18 逐位**、镜像等变 0.0、物理一致性）+ **966 tests**；注记入库 `docs/r36_warp_layer_note.md`。
- **关键发现（集成错误）**：warp 口径（t0 系=静态场景）下 OD 先验速度必须为 **t0 场景系速度 `v_OD = v_rel + v_ego`**；现行 R33-B wiring 传相对速度 ⇒ **系统性重复扣 `|Δp|`**——同速前车 P90 **1.95m → 0.43m**（spec 口径）、恒速假设成立子集 **1.21 → 0.072m**、静止先验 2.25 → 0.427。⇒ **结构化 WM 接入前必须改用 `od_velocity_t0`**；`off` 保持 `v_rel`（随体语义），两口径不可混用。
- **坐标门基线校准**（只读；10 shard + 262 录制，167 s）：静态 LD warp（250,549 对/13.8M 锚点）内部投影 **P90 4.5e-6 m**（GPT 的 0.10m 门余量 ~4 数量级）⇒ 建议门 **P90 ≤0.01 m + churn 护栏**（pair P90 ≤0.05m ∧ 覆盖 ≥0.80）；OD 同速前车 **0.427m（全子集）/ 0.072m（恒速子集）** ⇒ **0.20m 门仅恒速子集成立**（收窄到该子集并锁 0.10–0.20m；全子集读数登记为 **WM 残差基线**）；端到端 k≥1 **P90 4.5m @k=1**（含追踪/计划质量）⇒ **坐标门必须与位姿链输入误差分离登记**。
- **四 lane 全收口**：L-A ✅ `bbbe541` / L-B ✅（链结果）/ L-C ✅（G6 前置）/ **L-D ✅ `9b00761`**；L-B run2（分析报告）在飞。
- **下一步（阶段 1→2）**：①warp 集成 lane（`assemble_pose_k` 逐位替换 `model.py:1499-1505` + `od_velocity_t0` 修正 + hooks 接入；默认关）；②**WM 门 4 项先测**（现有 WM：h1–6 lateral/heading 增长 / tail vs frozen / h3–6 接口 / 物理预测 vs 恒速·静态基线）；③坐标门阈值锁定（CG 裁）；④G6 闭环跑批（L-C 草案）。

### 5.60 L-B 分析补充（收购 L-B run2）：T2 epoch 取舍 + 残余失败归因 + 专家地板全表

- **epoch 取舍（T2，init=T1 ep1）**：1 ep conv **0.706**/ool 0.221/**repeat 0.0385✓**；**2 ep conv 0.7655/ool 0.159/repeat 0.09✗**（超 5% 门）；3 ep 探索 0.7895/0.1475。⇒ conv↔重复换道存在 trade-off（1–2 ep 之间取值）。
- **残余失败归因（关键）**：2 ep 的 9%（180/2000）失败为"**投影边缘行**"（`|shift|≤0.5`、目标车道起点在 ego 前 ~7m、专家末点 lat p50 −4.45、**专家自身 0/180 收敛**）⇒ 属**口径伪影行**；**剔除后 conv 0.824（达标）/ool 0.085**。
- **专家地板（同口径 action 回放，只读）**：T2 conv **0.587** / ool **0.1195** / rep **0.061**；T1 keep ool **0.0915** ⇒ **现行三目标（conv ≥0.8 / ool ≤5% / rep ≤5%）均严于专家自身** ⇒ 建议统一按专家地板改**相对门**（显著优于旧基线 ∧ 不劣于专家同口径 + 余量）。
- **梯度/契约自证**（全 run）：head_calls=6/wm=5；plan-res 与 terminal 的 loss graph 仅 allowlist（18）且触达 lateral/longitudinal；结构探针 ok、`terminal_probed=True`、forbidden=0；`frozen_grad=0`、optimizer ids 匹配。

### 5.61 本轮收口清单（R36 `7dcf6bd` 轮）：进展/发现/结果/分析/问题（全部完成并推送）

- **进展**：四 lane 全收口（L-A 架构硬化 ✅`bbbe541`；L-B 训练器+T2 终端收敛 ✅；L-C G6 前置 ✅；L-D warp 层 ✅`9b00761`）+ 事实更正与证据索引（`docs/r36_implementation_evidence.md`、`docs/r36_warp_layer_note.md`）。
- **发现**：①**OD 先验速度语义错误**（warp 下须 `v_rel+v_ego`，现行 wiring 重复扣 `|Δp|`：同速前车 P90 1.95→0.43m〔spec〕、恒速子集 1.21→0.072m）；②**三目标（越界/收敛/重复换道）均严于专家自身**（keep 越界专家 5.2–9.15%；T2 conv 0.587/ool 0.1195/rep 0.061）；③**变道行 0.85m 规则不适用**（专家 97.7–100% 会"越界"）；④**投影边缘行是口径伪影**（`|shift|≤0.5` 行专家 0/180 收敛；剔除后 conv 0.824/ool 0.085）。
- **结果**：①T1 越界 **5.75% vs baseline 8.75%**（≈专家地板；fusion 必须开：head-only 0.479 vs full-18 0.058）；②T2 终端收敛 **0.458→0.706（1ep）/0.7655（2ep）**，剔除伪影行后 **0.824 达标**；③L-A 硬化：structured 66 公共键 max|Δ|=0、旧头调用 0/0、966 tests；④L-C：注入器对拍 7.6e-6m、structured 装载/2-ep 烟测通；⑤L-D：8/8 单测（R33-B 逐位 + 模型内 pre-hook 18/18）。
- **分析**：T2 的 conv↔rep trade-off（1ep rep✓/conv 低；2ep conv 高/rep✗）；残余失败定位到"投影边缘行"而非能力缺陷；坐标门与位姿链误差需分离登记。
- **问题（open，待 CG/集成）**：①**门阈值锁定**（相对门 = 显著优于旧基线 ∧ 不劣于专家 + 余量；坐标门 LD ≤0.01m+churn、OD 收窄恒速子集）；②**warp 集成**（`assemble_pose_k` 逐位替换 + `od_velocity_t0` 修正 + hooks；默认关）；③**G6 闭环未跑**（结构化头闭环成功率未知，命令草案就绪）；④**WM 门 4 项未测**（结构化 WM 训练缺验收基线）；⑤AGENT.md 56 行版执行细节缺口（argue/契约保留）。
- **下一步（待令）**：warp 集成 lane → WM 门 4 项先测 → 阈值锁定 → G6 闭环跑批 → 阶段 2（WM 车道保持训练）。

### 5.62 round-37 受理（GPT `66497a2`，551 行）：结构化 Policy + 坐标解耦 WM 的**闭环验证**；我方排序提案（37A→37B→37C）；4 条事实注记；M1/M2 开工

- **受理（总方向）**：**统一闭环基准**——**C0**（旧 Policy+旧 WM 坐标+固定 LQR，基线 ~0.56）/ **C1**（结构化 Policy+旧 WM，验证新动作表示单独贡献）/ **C2**（+analytic ego warp，验证坐标补偿直接贡献）/ **C3**（+重训 residual WM，正式候选）；四臂同数据/同调用/horizon/LQR/episodes；Calib50→Clean150→Eval500（3 seeds/paired bootstrap/McNemar）；**字典序 + 安全硬门**（Primary=success；Hard=collision·off-road；Secondary=timeout·rc；Diagnostic=离线），**禁模糊加权总分**。代码修正 6 项：Policy 收编（`policy.structured_action.*`）/ 均值方差同源（logstd 请求 fail-fast）/ **warp 主路径接入（`ego_warp_mode: off|analytic`，默认 off 逐位）** / **WM 学解析后 residual**（`A_{k+1}=Warp(A_k,a_k)+ΔA_θ`，分项报 解析先验·学习残差·最终误差）/ **imagined 步动态更新**（WM 物理预测；缺则 `dynamic_valid=0`，禁止静默 t0；hold/WM/oracle 只读对照）/ **调用口径三分**（Policy 6 / WM 决策推进 5 / ST-GNN 总 6）。训练序：坐标层→WM residual→Policy（分支 1.0×、backbone 末层 0.1× LR）→预测态一致性→交替。门 A–E（**Gate D 采纳我方"投影边缘行过滤规则须预先定义"**）；**Calib50 门槛 0.62/off ≤0.10/coll ≤0.08/tmo ≤0.10 过前不进 Clean150/DAgger/MoE/RL**。
- **我方排序提案（关键分歧点，供裁）**：**C1/C2 两臂不需要任何训练即可跑**（C1=已训练结构化头〔`t1_keep_res` ep1〕+旧 WM；C2=+analytic warp，仅需集成）⇒ 建议 **37A 集成（warp 主路径 + `od_velocity_t0` 修正 + 五项 hook + 分项读数）→ 37B C0/C1/C2 闭环分诊（Calib50，小时级出数）→ 37C 依 37B 的 C2−C1 信号决定 WM residual 重训（C3）**。理由：①闭环是终局标准，**先用最小成本拿到"新动作表示单独贡献 / 坐标补偿直接贡献"两个数字**（正是本轮要回答的）；②WM 重训是数日级投入，应由 C2−C1 差值触发；③避免"先重训后才发现 C2 无益"的浪费。
- **事实注记（4 条）**：①L-A 已使 structured 模式**不 forward 旧头**（单一动作源风险由运行时断言兜底）⇒ "折叠"=命名/归属迁移（`policy.structured_action.*` + ckpt 键重映射，默认兼容），建议与 37A 同批但**不阻塞 37B**；②"WM 只调用 5 次"的精确口径已在 `architecture_contract.md` §2 记录（6 次 ST-GNN、前 5 推进）；③**Gate B 分项**（恒速/同速前车/加减速/cut-in·merge）——L-D 校准已有前三类，补 cut-in/merge 子集；④**Gate D 边缘行预定义提案**（先注册后执行）：`|shift|≤0.5 ∧ 目标车道起点在 ego 前方 >5m ∧ 专家末点不收敛` ⇒ 伪影行。
- **开工**：**M1**（`net/`：warp 主路径接入〔`assemble_pose_k` 逐位替换 + `od_velocity_t0`〕+ 五项 hook + 分项读数；off 逐位 + 966 tests）；**M2**（**C0/C1 Calib50 同协议闭环分诊**先行，C2 待 M1；产出 paired 表）。

### 5.63 round-37 答复受理（GPT `effea53`，168 行）：接受执行序；**唯一关键修正（采纳）＝C2 ≠ C3 开关**；口径澄清；"收编"=技术债

- **受理（执行序，6 条）**：GPT 接受 **37A 工程接入 → 37B C0/C1/C2 闭环分诊 → 37C WM 适配**；接受 C0/C1 先跑、warp `off|analytic`、OD 速度修正、同 episodes paired、主指标/安全硬门/次指标、离线指标不替代闭环结论。
- **关键修正（采纳，更正 §5.62 表述）**：**C2 = "旧 WM 上的零训练 warp 诊断臂"，不是决定是否训练 C3 的开关**。旧 WM 从未在补偿后 OD/LD anchors 上训练 ⇒ C2 无提升甚至退化**不能**否决 C3（证据：零训练 warp 此前仅 ~2% 改善；OD 速度语义修正进一步证明新旧坐标口径不同）。**§5.62 的"C2−C1 信号决定 C3"作废**，改为 **C3 触发条件（任一成立即继续）**：①C1 的 Policy 离线指标好但六步闭环明显差；②C2 的解析 anchor 明显更准但 Policy 接口未改善；③C2 相对 C1 退化且退化源于新旧坐标分布不匹配；④OD/LD 的 ego-motion 泄漏仍显著；⑤WM 多步车道状态仍随 horizon 漂移。⇒ **C3 由坐标与接口证据联合决定**。
- **口径澄清（采纳）**：①**timeout 门＝episode 比例**：Calib50 `timeout rate ≤10%`（≤5 例）；统一门 = `success ≥0.62 / off-road rate ≤0.10 / collision rate ≤0.08 / timeout rate ≤0.10 / rc 不降`；②**"Policy 收编"可后置但属短期技术债**，硬约束保留：正式模式**唯一运行中的动作生成器**、旧 Policy **不得静默前向**、**均值与方差不得来自两个头**、RL 接线前完成同源 action distribution（现状：L-A 已满足前两条 + `logstd` fail-fast）。
- **处置**：M1/M2 按原 scope 继续（无变更）；37B 结果按上述 **5 条联合判读**；行数勘误并入本 commit（此后版本注用"≈"）。

### 5.64 R37 阶段 1 首批闭环结果（M2）+ **纵向语义根因**（5 证据链）；修订优先级提案（**P0=Policy 速度目标化**）；M1 入库、M3/M4 在飞

- **M2（C0/C1 同协议 calib50，n=50；`runs/round37/m2/m2_c0c1.{md,json}`）**：

| 指标 | C0（K1e2 旧头） | C1（结构化头 t1 ep1 + 注入） | C1−C0 |
|---|---|---|---|
| success | **0.580** | **0.340** | **−0.240** |
| off-road | 0.180 | 0.120 | −0.060 |
| collision | 0.120 | 0.140 | **+0.020** |
| timeout | 0.140 | 0.400 | **+0.260** |
| route completion | 0.872 | 0.741 | −0.131 |
| min-TTC 均值 / steer 饱和 | 6.99 / 0.624 | 7.47 / 0.026 | +0.49 / −0.599 |

配对：succ 翻正 2 / 翻负 14（McNemar p=0.0042）；timeout 翻正 14 / 翻负 1。**字典序判决：C1 不晋级**（Hard 先看：collision 0.12→0.14 恶化；Primary 亦 C0 胜）。
- **协议口径**：历史 K1e2 "0.56" 系 kappa-compose 口径（off 0.14/coll 0.08）；**本协议 C0=0.580/0.180/0.120**（off/coll 变差、succ 略升）⇒ **Round-37 统一基准固定为"本协议"，历史数字不可直比**。并发改动中性性：c0/c1 各自 r1 vs r2 **逐 episode 字段级逐字一致（50/50）** ⇒ M1 默认关口径闭环级复证。
- **根因（新，5 证据链）**：C1 失败形态=**欠动/爬行**（throttle≈0、v 自锁 1.3–2.2 m/s、crawl 中位 52 s、timeout 20/50），机制链：
  1. **LQR 参考速度 = plan 弦长/dt**（`env/tracking.py:498–504`；`:537` `accel = speed_gain·(v_ref−v)`）⇒ **plan 几何即速度指令**；
  2. C0 plan **递增**（首决策 ds 2.69→3.73；第 31 决策 3.38→5.09）⇒ 预览点 v_ref 6.9–7.4 ⇒ throttle +0.2~0.3 ⇒ 保速/加速（v 6.0→6.29）；
  3. C1 plan **平坦且随 v 下滑**（首决策 1.95→2.10；第 6 决策 0.60→0.68；v[0..9]=6.0→4.09→2.74→1.86→1.49→1.21…）⇒ v_ref≈v ⇒ throttle≈0/−0.05 ⇒ **任意减速不可逆**（设计自锁点 = clamp 下界 0.72 m/步 ≈1.44 m/s）；
  4. 结构化头**纵向参数化**：`ds = clamp(v_current·dt, bounds) + 残差`、longitudinal 末层**零初始化**（`net/path_head.py:994`；bounds `[0.72, 6.25]`）⇒ **零残差=保速先验**；从 v=6 逃逸到 8.3 m/s 需 +2~3 m 正残差；
  5. 监督标签（DP-6 = expert5k_v8 录制弦长）：**残差 vs clamp(v·dt) 全速段 ≈0**（65% 行贴 8.33 m/s 上限、P50 4.143；|resid|>0.2 仅 16% 且以负为主）⇒ 数据几乎不含"加速指令"信号。另：**k=1（v=6.0）头输出 1.95 < 保速先验 3.0 < 同桶专家目标 3.42** ⇒ 另有**幅值/输入分布**问题（1 ep；`--init=M0` 无语义键 vs eval `--ld-mode semantic` + 20 键回填；obs 指纹 v8 疑点）——**M3 在裁**。
- **与 GPT R37 的关系（分歧点，供裁）**：R37 的杠杆设定=WM 坐标语义/泄漏（C2/C3）。**我方证据：C1（新动作表示单独贡献）已 −24 pp，且根因在 Policy 纵向语义（保速先验 + 标签无加速信号）；C2/C3 都不会修 C1 的爬行**。⇒ **P0 = 结构化 Policy 纵向"速度目标化"**，三候选：**A** 先验改目标速度（`ds_init` 由 v_current 改为 v_target≈limit/cruise，或纵向头直接输出速度目标）；**B** 教师蒸馏（**用 C0 的 plan 作纵向教师**——闭环行为直接迁移；需一次 rollout 采集）；**C** 标签/损失重构（速度剖面/终端加速显式加权 + eval 装配下重训）。**C3（WM residual）仍按 R37 推进，但角色重定位=想象步动力学**（不是 C1 爬行的解）。
- **M3（离线诊断收口；`.slim/deepwork/p1_v2/round37/m3_report.md`）**：①**头在训练分布内没坏**（bias +0.056 / MAE 0.149 / r 0.983；[2,4) 剖面 +0.43 爬升）；②**但命令语义=保速**（残差仅 +0.03~+0.14 m/步，`v_ref/v≈1.00`）⇒ **坍缩是闭环特有的"保速+无加速裕量"自锁**；③**训练/评测装配错配成立，主凶=`path_dynamic_ttc` 缺键→0**（训练 100% present、均值 30.6 s「无冲突」vs 评测 0「立即碰撞」；单去 TTC 即把 [2,4) 斜率 +0.43→+0.00、[4,6) +0.24→−0.26，占消融效应 **77%**）；④语义 20 键 z 扰动存在但方向相反（+0.14）非驱动；⑤同输入双配置 `eval_on−train` mean Δds **+0.138**（融合输入 β·δ/legacy=2.72 正交）⇒ **训练须在 eval 装配下做**。建议：接口补键/中性填充（先行）→ 部件重训 → P0 头侧目标速度化。
- **M4（C2/F3；`m4_report.md`）**：**C2（analytic warp）机制生效**（plan k=0 与动作逐位同、**k≥1 全分化**；off 中性到 trace 逐位 78/78）：succ 0.32（−1 ep）、**collision 0.14→0.08（3 解除/0 新增）**、tmo 0.48 ⇒ Hard 不劣化、**不主张晋级**；**F3（`--cf-ref-speed-scale 2.0`）证伪外部配速**：v 8.92 m/s 但 coll 0.48、a_lat p95 4.76、lon 饱和 0.637。
- **M5（P0-A 先验探针；`m5_report.md`）**：**意图层生效**（`dec__ds_init` 逐位=声明值 max|Δ|≤1.5e-7；v_ref 2.63→**8.49/6.31/5.37**；残差不变式 `logit(ds/10)−logit(ds_init/10)` 与 C1 逐 id 相等 50/50）**但达成层不升**（v 仅 3.56/3.01/2.85；A1 94% 步 throttle>0.9 而池化 accel≈0，转向饱和步 accel −0.21 m/s²）⇒ **Hard 恶化（off-road 0.12→0.44/0.28/0.24；A1 succ 0.46/coll 0.06/tmo 0.08）**，失败模式从爬行迁移为**车道保持失守**（A1：steer 饱和 0.81、heading err p50 0.237 rad、蛇形极限环）⇒ **P0-A 单一先验不构成解**。
- **横向侧谱（主控复核 trace，50 ep）**：|err_y|/|err_ψ| = C0 0.465/0.111、C1 0.055/0.017、A1 0.988/0.241；plan |κ|p90 = 0.049/0.023/0.081；κ 抖动 = 0.015/0.008/0.037 ⇒ **加速后横向能力（曲率质量+抖动）是共阻塞**：训练层须同时修"纵向目标速度"与"横向 @speed 质量"。
- **零训练通道总账（结论）**：F3 ✗（碰撞）/ A ✗（越界）/ C2 ~中性（coll↓ 但 tmo↑）/ **接口修复 M6 ✅ 方向正确但非完整解** ⇒ **P0 定为主训练 lane**：**第一候选 B = 教师蒸馏（IDM+PP 规则专家；`tools/dagger_collect.py` 就绪：学生状态 roll-in + 规则专家 (ds,dθ) 标签〔自带加速/曲率-速度协同〕+ train-spec 硬隔离）**，训练须在 **eval 装配（指纹统一）** 下进行（M3 ⑤）；**C3（WM/想象步动力学）排其后**（不修爬行，只修想象步演化）。
- **M6（接口修复探针，已落地；`m6_report.md`）**：中性值**最小后悔标定** ⇒ `path_dynamic_ttc`=**17.7 s**（候选 30.1/60 的离线后悔更大：mean|Δds₀| 0.0664 vs 0.078/0.096）；`dynamic/static_distance` 填 0 与缺键**逐位同**（0 即训练主态，97.4%/88% 主态）⇒ **被毒化的实际只有 TTC 一键**。**c1_fill vs C1**：succ **0.34→0.42（翻正 4/翻负 0）**、coll 0.14→0.10、off 0.12 平、tmo 20→18、rc +0.058、v 2.47→2.78、crawl 34.9→24.8 s、**plan 剖面斜率全桶改善且高 v 桶翻正**（主控复核均值 −0.134→−0.086）；**同 t0 obs（状态逐位同）plan 首步 50/50 分化、Δds₀ 均值 +1.13 m/步**（闭环工作点灵敏度 ≫ 离线 +0.05~0.25）。**a1_fill**：succ 0.48 但 **off 0.38 ⇒ Hard 退化、不晋级**。未达 Calib50 绝对门（0.42<0.62、tmo 0.36>0.10）⇒ 接口修复=**方向正确的非完整解**；残余（低速桶斜率仍负、timeout 0.36）须训练层（M3 建议 2/3）或 **env 侧真值回填**（**M7 在飞**：按训练侧同公式从 OD 槽算真值 dyn/TTC，替代中性填充）。

- **M7（真值回填探针，已落地；`m7_report.md`）**：按**训练侧同公式**在 eval 现场重算三键（`m7_fill.py` 单一出处；A–D 公式核验全过：4000 时刻×16 槽 vs DP-6 复写**逐行逐位**、`ODChannel` urgency 933/933、静态 1055/1055 present、矩形距离 max|Δ| 5.7e-14）。**c1_true vs C1**：succ **0.400（+4/−1）**、coll **0.080**（+1/−4，全臂最优）、**off 0.180（+4/−1）**、tmo 0.340、v **2.89**、crawl **17.3 s**、斜率 **−0.059**（三项全臂最优）；**vs C1_fill：off 硬退化（+5/−2）⇒ 字典序不晋级**。机制：真值档**更强释放速度**（crawl −7.5 s、throttle 0.073→0.110、斜率 −0.091→−0.059），但**横向执行跟不上**（cut_in 例 |err_y| 峰值 0.92 m vs C1 0.13、180 步出界；新增 off 含 2 tollgate 碰撞→出界）⇒ **neutral 仍为当前最佳可用接口填充；true 定位为 P0-B 配套接口、训练后重测**。真值分布 vs 训练（cap60 占比 0.671 vs 0.365、p50 60 vs 17.7 s）差异来自**状态分布**（eval 低 v + 稀疏交通），非口径错误。

### 5.65 R37 阶段 1 收口清单（进展/发现/结果/分析/问题/待令）

- **进展**：M1 warp 主路径入库 `0d63bbc`；M2（C0/C1 分诊）、M3（离线诊断）、M4（C2/F3）、M5（A 探针）、M6（接口修复）、**M7（真值回填）**全收口；**M8（P0-B 数据链试点：IDM 教师标签语义验证 + H2/I1 抽取就绪度 + CG 审批材料）在飞**。
- **发现（三层，互为放大）**：①**结构**：保速先验 `ds=DS_HIGH·σ(ds_raw+logit(clamp(v·dt)/DS_HIGH))` ⇒ 零残差=保持速度、无加速权限（自锁 ≈1.44 m/s）；②**接口**：`path_dynamic_ttc` 缺键→0="立即碰撞"语义（占装配错配效应 77%），修复零训练 +8pp；③**横向**：加速后 plan 曲率质量/抖动恶化（|κ|p90 0.023→0.081、κ 抖动 ×4.7、|err_ψ| 0.017→0.241）⇒ off-road。
- **结果（calib50 同协议；succ/off/coll/tmo）**：C0 0.580/0.18/0.12/0.14 ｜ C1 0.340/0.12/0.14/0.40 ｜ **C1_fill 0.420**/0.12/0.10/0.36 ｜ **C1_true 0.400**/0.18/**0.08**/0.34 ｜ C2 0.320/0.12/**0.08**/0.48 ｜ A1 0.46/0.44/0.06/0.08 ｜ F3 0.30/0.24/0.48/0.02。
- **分析**：历史闭环无此问题=旧头是**速度目标生成器**（plan 隐含 v_ref>v，主动拉回速度）；离线门（越界/收敛）只测横向 ⇒ 纵向盲区；接口三形态（缺键/中性/真值）均指向同一瓶颈=**高速下横向执行**；⇒ 训练层修复须**同时**覆盖"纵向目标速度"与"横向 @speed 质量"，且在 **eval 装配（指纹统一）**下训练。
- **待令（CG）**：①**接口修复固化形态**（**neutral=当前最佳可用**〔+8pp、Hard 不劣化〕/ true〔待训练层配套后重测〕/ env 侧直供）；②**P0-B 训练放行**（教师=IDM+PP 规则专家，`tools/dagger_collect.py` 就绪；M8 试点给出标签语义与数据链缺口清单后一并呈报；待定：roll-in 策略、train spec 池、预算、门〔建议 Calib50 快筛 → Clean150 晋级〕）；③C3（WM/想象步动力学）时机；④基准口径登记：**本协议 C0=0.58/0.18/0.12** 与历史 kappa-compose 0.56/0.14/0.08 分离，不再混用。

### 5.66 R37 阶段 1 复盘（进展/发现/结果/分析/问题）+ "起步/限速/监督/验证集"四问核查结论

- **进展**：M1–M7 七条 lane 全收口（warp 主路径入库 `0d63bbc`；C0/C1 分诊；离线诊断；C2/F3；A 先验探针；接口修复；真值回填）；**M8（P0-B 数据链试点：IDM 标签语义 + H2/I1 就绪度）在飞**；本轮全部零训练、判定均在 calib50 dev 口径。
- **发现（四条新事实，已代码/数据核验）**：
  1. **起步速度**：calib50 episode 的 spawn 速度 = **6.0 m/s**（非 0；`env__speed[0]=6.0`）⇒ C1 的缺陷是"**掉速不可恢复**"（保速律 + `ds_init` 地板 0.72 m/步 ⇒ 自锁 ≈1.44 m/s）；**从静止起步会在同一点卡死**（部署风险）。历史轮次未暴露=旧头是速度目标生成器（plan 隐含 v_ref>v）+ 历史门只测横向。
  2. **限速可见性**：限速存在于 **LD 7 维第 5 维**（`env/obs/ld.py`）与 **others 上下文**（`speed_limit_norm_mps`，`env/obs/builder.py`）⇒ **net（冻结编码器）能看到**；但 **StructuredPolicyHead 显式输入（long_exog 7 键）无限速列、无目标速度输出**，标签=保速弦长 ⇒ "该开多快"只能间接感觉 + 模仿"你现在多快"。
  3. **训练集覆盖**：v<0.3 = **1.18%**、v<1 = 2.67%、v<2 = 7.93%；低速行**瞬时**残差≈0偏负（[0,2) −0.17，以减速/排队为主）；v<1 行 6 步未来链 ds 0.23→0.54（≈0.25 m/s²，缓升且弱）⇒ **有刹停/低速跟随、几乎无"静止起步加速"监督**；弱信号仅在 aux 未来链且被 SmoothL1 抹平。
  4. **纵向设计三缺陷**：①**先验语义**：`ds_init=clamp(v·dt)`=保速先验（应为目标速度 `min(限速, √(a_y/κ), 前车约束)`）；②**通道**：加速意图在 aux `ds_profile[:,1..5]`（系数 0.5、契约不进动作/LQR），动作只吃专家**瞬时**弦长；③**损失**：均值回归抹加速尾部；`σ`×`DS_HIGH=10` 尺度下逃离低速需 raw +0.2~+2.2 vs 实测 −0.57；缺速度剖面/jerk/终端加速项；κ 限幅表按专家残差标定、不含提速横向裕量。
- **结果（calib50 dev；succ/off/coll/tmo）**：C0 0.580/0.18/0.12/0.14 ｜ C1 0.340/0.12/0.14/0.40 ｜ **C1_fill 0.420**/0.12/0.10/0.36 ｜ C1_true 0.400/0.18/**0.08**/0.34 ｜ C2 0.320/0.12/**0.08**/0.48 ｜ A1 0.46/0.44/0.06/0.08 ｜ F3 0.30/0.24/0.48/0.02。
- **分析**：三层根因（结构保速 / 接口 TTC 毒化占 77% / 高速横向质量）互为放大；接口三形态（缺键/中性/真值）全部指向**同一绑定点=高速下横向执行**；零训练通道穷尽（唯一有效=接口修复 +8pp，非完整解）⇒ **训练层 P0 必要性成立**。
- **训练放开建议（供裁）**：①必须：`path_head.*` + `plan_head.fusion.2.*`（已证 head-only 0.479→0.058）；②建议：新增 head 前**小适配器**或微调编码器**末层**（吸收装配位移 Δds +0.138 / β·δ 2.72× 正交）；③**不放**：WM 主干/编码器主体/Mem/旧 PolicyHead/MoE/ValueHead（C3 只放 residual/physical heads）；④κ 限幅表在闭环执行数据上重标定。
- **验证集口径（答疑）**：**calib50 非新集**=主线自 R25 起的开发集；台账锚点（legacy 0.40/Q1 0.50/K1 0.52/T2 0.58/统一基线 0.56/**IDM 0.756**）皆 calib50 口径 ⇒ 同集可比；GPT 三级协议（Calib50 开发→Clean150 晋级→Eval500 正式：3 seeds/paired bootstrap/McNemar）保持不变；n=50 ⇒ ±1 集 ≈ ±2pp 只作筛选（配对翻转/McNemar 作证据强度）；历史数字若来自其它集需按待令④逐条登记。
- **问题（新增）**：⑤**"0 起步/低速恢复"应作为训练与门的显式考核项**（提案：新增"起步-恢复"评测切片：v≈0/低速注入 → 要求 v 在 N 步内恢复至场景限速量级）；⑥接口修复的固化路径（builder 直供 vs 模型侧缺键中性默认）需与训练装配一并定稿。

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`、`docs/recovery_v2_contract.md`、`docs/round30_audit.md`、`docs/stage_a_config_consumption.md`
- **报告**：`docs/p1_0_report.md`、`docs/p1_2_diagnosis.md`、`docs/r21_lane_semantic_contract.md`、`docs/r21_target_lane_selection.md`、`docs/semantic_v1_dataset_card.md`、`docs/r21_3a_semantic_wiring.md`、`docs/recovery_v1_dataset_card.md`、**`docs/round32_canary_report.md`（canary Gate 0–4 全测）**、**`docs/round33_report.md`（33A 三问 + 33B warp NO-GO）**
- **数据**：`datasets/recovery_v1|v2/`、`datasets/semantic_v1_{pilot,independent,3d3}/`（9,515 / 8,953 / 53,765）
- **产物**：`runs/p1_v2/round12a|15a|…|30/`、`runs/p1_v2/`（probe 家族）、**`runs/round32/canary/`（gate0_compat/eval_*/gate3_*/gate4_*/record_*/monitor）**、**`runs/round33/`（a1_ckpt_audit / a2_collision / a5_overshoot_decomp / a6_noise / b_warp / b_warp_compat）**、`runs/p1_0|p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_v2/`（round12a→33 脚本 + logs）；R30 队列=`round30/*.sh`；canary 队列=`round32/{canary_train,canary_eval,canary_compat_check,c4_gate3_analyze,p3b_conditional_gain}.py` + `run_r32_*.sh`；R33 队列=`round33/{a2_collision_attribution,a5_overshoot_decomp,a6_noise,b_warp_compat,b_warp_record,b_warp_analyze}.py`
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮；`61bf7ab`=Reality Audit 提案、`acf1b8f`=终审、`c3bf69f`=round-33 答复、`f0ebfae`=round-33 修订）
- **关键提交**：逐轮全文=本文件 git log；近期锚点：`557527b`（§5.39）、`78ec71e`+`7015cc8`（canary 基建+报告/§5.40）、`0188218`（§5.41）、`5ced90e`+`ce190f4`（R33 执行/§5.42）

---

## 7. 存档说明（历次精简 / 去哪找）

- **第五十轮补遗2（本版；402 → 432 行）**：新增 §5.66（阶段 1 复盘 + 四问核查结论 + 训练放开建议 + 新问题⑤⑥）；上版 418 行 = `git show d66cc0e:docs/deepseekv4p1_argue.md`。
- **第五十轮补遗（402 → 418 行）**：§5.64 追加 M3/M4/M5/M6/M7 结果 + 横向侧谱 + 零训练总账（M6 接口修复 ✅ +8pp；M7 真值：coll 0.08 最优/off 0.18 不晋级）+ P0 定案提案（B 第一候选）；§0/§7 刷新；上版 402 行 = `git show 0ecbad6:docs/deepseekv4p1_argue.md`。
- **第五十轮（377 → 402 行）**：新增 §5.64（M2 闭环分诊 + 纵向语义根因 5 证据链 + P0 优先级提案 + M3/M4 在飞）；§0/§7 刷新；上版 377 行 = `git show a0bb6e6:docs/deepseekv4p1_argue.md`。
- **第四十九轮（369 → 377 行）**：新增 §5.63（R37 答复受理：执行序接受 + C2≠C3 修正 + 口径澄清）；§0/§7 刷新；上版 369 行 = `git show 832e02a:docs/deepseekv4p1_argue.md`。
- **第四十八轮（361 → 369 行）**：新增 §5.62（R37 受理 + 排序提案 + 4 条注记 + M1/M2 开工）；§0/§7 刷新；上版 361 行 = `git show bf70293:docs/deepseekv4p1_argue.md`。
- **第四十七轮（351 → 361 行）**：新增 §5.61（本轮收口清单：进展/发现/结果/分析/问题）；§0/§7 刷新；上版 351 行 = `git show 0981ea1:docs/deepseekv4p1_argue.md`。
- **第四十六轮（335 → 351 行）**：新增 §5.59（warp 层收口 + OD 速度语义发现 + 坐标门校准）+ §5.60（L-B 分析补充）；§0/§7 刷新；上版 335 行 = `git show 90eca34:docs/deepseekv4p1_argue.md`。
- **第四十五轮（326 → 335 行）**：新增 §5.58（L-A/L-B/L-C 收口 + 门校准汇总）；§0/§7 刷新；上版 326 行 = `git show bbbe541:docs/deepseekv4p1_argue.md`。
- **第四十四轮（318 → 333 行）**：新增 §5.57（答复受理 + 事实更正 + 3 条注记 + L-D 开工）；§0/§7 刷新；上版 318 行 = `git show 494d9ae:docs/deepseekv4p1_argue.md`。
- **第四十三轮（306 → 318 行）**：新增 §5.56（车道保持阶段受理：5 条批评核验 + 3 条反驳 + 证据补充 + 执行清单）；§0/§7 刷新；上版 306 行 = `git show 631ad45:docs/deepseekv4p1_argue.md`。
- **第四十二轮（296 → 306 行）**：新增 §5.55（认账 + 8 条修正 + 5 条细化 + 3 条注记 + K1 开工）；§0/§7 刷新；上版 296 行 = `git show 8f44851:docs/deepseekv4p1_argue.md`。
- **第四十一轮（287 → 296 行）**：新增 §5.54（R36 实施方案登记版：阶段与门/实现约束/禁令/进展/待裁）；§0/§7 刷新；上版 287 行 = `git show 4a35dcb:docs/deepseekv4p1_argue.md`。
- **第四十轮（263 → 287 行）**：新增 §5.53（R36 准备完成 + 下一步）；§0/§7 刷新；上版 263 行 = `git show 7c0186f:docs/deepseekv4p1_argue.md`。
- **第三十九轮（263 → 277 行）**：新增 §5.52（CG 终止回退 + 直接执行 `7c0186f` + 4 条补充）；§5.51 加作废注记；§0/§7 刷新；上版 263 行 = `git show 7c0186f:docs/deepseekv4p1_argue.md`。
- **第三十八轮（253 → 263 行）**：新增 §5.51（WP3 受理 + 3 条补充 + 开工）；§0/§7 刷新；上版 253 行 = `git show 272dd08:docs/deepseekv4p1_argue.md`。
- **第三十七轮（243 → 253 行）**：新增 §5.50（R35 审计执行完成 + 综合 + 建议）；§0/§7 刷新；上版 243 行 = `git show 74e0bf0:docs/deepseekv4p1_argue.md`。
- **第三十六轮（230 → 243 行）**：新增 §5.49（R35 受理 + 6 条补充 + 开工）；§0/§7 刷新；上版 230 行 = `git show 404738d:docs/deepseekv4p1_argue.md`。
- **第三十五轮（221 → 230 行）**：新增 §5.48（34B0 执行完成：三 lane + 综合 + 34B1 建议）；§0/§7 刷新；上版 221 行 = `git show ba11c77:docs/deepseekv4p1_argue.md`。
- **第三十四轮（213 → 221 行）**：新增 §5.47（34B 修订受理 + nav 答复 + 3 条补充 + 34B0 开工）；§0/§7 刷新；上版 213 行 = `git show 06adf69:docs/deepseekv4p1_argue.md`。
- **第三十三轮（203 → 213 行）**：新增 §5.46（34A 执行完成：三 lane 结果/综合/34B 范围）；§0/§7 刷新；上版 203 行 = `git show df3760e:docs/deepseekv4p1_argue.md`。
- **第三十二轮（191 → 203 行）**：新增 §5.45（34A 批准受理 + 5 条补充 + 开工）；§0/§7 刷新；上版 191 行 = `git show 6f2e6bf:docs/deepseekv4p1_argue.md`。
- **第三十一轮（180 → 191 行）**：新增 §5.44（33C 执行完成：硬门表/增益链/分层/分析/分支判定）；§0/§7 刷新；上版 180 行 = `git show a616595:docs/deepseekv4p1_argue.md`。
- **第三十轮（169 → 180 行）**：新增 §5.43（33C 批准受理 + 4 条补充 + 开工）；§0/§7 刷新；上版 169 行 = `git show 5fbc0ef:docs/deepseekv4p1_argue.md`。
- **第二十九轮（405 → 169 行）**：**压缩清理**——§3/§4/§5.1–5.37 归档化（保留锚点/关键读数/教训）；§5.42 补全"分析/问题"；§0/§6/§7 刷新。上版 405 行 = `git show ce190f4:docs/deepseekv4p1_argue.md`。
- **历次精简留档**：第二十八轮（392→405，`5ced90e`/`ce190f4`）；第二十七轮（377→392，`0188218`）；第二十六轮（363→377，`7015cc8`）；第二十五轮（354→363，`557527b`）；第二十四轮（343→354，`57d2391`）；第二十三轮（334→343，`3d14f9d`）；第二十二轮（319→334，`cd6f1f6`）；第二十一轮（308→319，`87253cf`）；第二十轮（298→308，`440790b`）；第十九轮（281→298，`7beb7e9`）；第十八轮（264→281，`b8bb326`）；第十七轮（299→264，`5ed40e4`）；第十二轮（280→270，`e5ae48e`）；第十轮（267→252，`52ce959`）；第八轮（300→252，`ee2651b`）；第六轮（342→261，`e638b76`）；第一轮（1114→166，`7b83730`）。
- **压缩时间线**：
  - 2026-10-09/10：P0 三件套 → 排摸收官 → round-3–8 与 P1 锁板；
  - 2026-10-10→11：P1 执行 → round-9/10 定板 → P1-v2 三步 + 旁线 D/E → **12A NO-GO**；
  - 2026-10-11 深夜–10-12：round-15→20（15A 全批次 → **A3 判负** → 18A probe+审计 → M0-M2 消融 → 停 LD 架构、转数据侧）；
  - 2026-10-12 深夜–10-13 凌晨：R21-0..2 门全过；3A..3C（首现强正向）；
  - 2026-10-13 上午：3D-1/2 门过、**3D-3 未过**；中午：**R25 首超 legacy**、B 行 bug 修复；
  - 2026-10-13 下午：**R26 Q1 全面生效** → R27 R1（tmo +26pp）→ R28 **28A 归因 + K1 停止**；
  - 2026-10-13 晚：R29A T 族投影（T2 唯一正向未过门）；
  - 2026-10-13 夜–10-14 凌晨：R30 Reality Audit（新主问题=部署链一致性）→ R31 审计包（Q2/M2/契约/MoE-off 0.56）；
  - 2026-10-14 凌晨–上午：R32 提案/裁决 → **Level-1（D2→D3=97%、动作条件化失效）→ canary 批准 → 四臂执行（Gate 3 全 FAIL）**；
  - 2026-10-14 下午：R33——**33A 三问闭合 + 33B warp NO-GO ⇒ GO_EGO_ACTION_CONDITIONING_V2；33C 待令**。
