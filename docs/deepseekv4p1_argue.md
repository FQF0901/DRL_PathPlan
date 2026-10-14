# deepseek-v4.1 argue / 主文档（R36 准备完成：H1 执行头骨架（Gate 0 8/8；已入库）+ H2 given-target 数据规格（99.98% 标签、数据门 15/15）+ H3 hooks（555 键）；关键 caveat=T4 专家晚换道；DP-6 提取开工（§5.53））

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-15 上午·四）**：**R36 准备完成**（H1/H2/H3 全收口；报告 `docs/round36_report.md`）——①**H1 执行头骨架**（`net/path_head.py` 791 行；车道锚 κ_ref + near/mid/far **曲率残差** + 路径条件化纵向头〔→6 步 s〕+ 组合层；**Gate 0 8/8**：默认关三口径逐位、旧 ckpt strict 0/0/0、LQR 契约 ≤3.2e-6、**966 passed**；已入库 `6944436`）；②**H2 given-target 数据规格**：T1 保持 907 ep/53,606 步、T2 变道 2,579 ep/11,818 步、T3 merge 441 ep、T4 闸口 **317/455 ep（69.7%）**/15,543 步；全库 **265,801 usable、99.98% 标签、0 error**；数据门 15/15；**关键 caveat=T4 专家多为晚换道（p50 25.8m）⇒ 直接模仿编码"晚换道"，G4 提前换道需训练侧处理**；③**H3 hooks**（555 键；§五 7 类全覆盖；footprint 连续 + LQR 四象限；默认关 bit-exact）。**下一步**：DP-6 提取开工（I1）→ 训练接线 T1–T4 → G1–G6；待裁：G1 子集口径 + T4 晚换道策略。
> **本版说明**：第四十一轮（287 → 296 行）——新增 §5.54（R36 实施方案登记版：阶段与门/实现约束/禁令/进展/待裁）；§0/§7 刷新；上版 287 行 = `git show 4a35dcb:docs/deepseekv4p1_argue.md`；历版全文=本文件 git log。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：最高可执行 d2=0.706）；P1 起点 pri512@e020 0.340。
- **主线锚点（calib50）**：legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58 / **MoE-off 统一基线 0.56** / IDM 0.756。
- **主线结论链（压缩）**：1b no_increment → 12A NO-GO（gate 选择性瓶颈）→ A3 判负（动作可持性）→ R21 语义重建（3C 首现强正向）→ 3D-3 未过门 → R25 首超 legacy（0.400/+7.3pp）→ R26 Q1（0.50）→ R27 R1（sat ✓ 但 tmo +26pp ✗）→ R28 28A 归因（F3/F1/F4）+ K1 首步解耦不足 → R29A T2 唯一正向（0.58）未过门。
- **R30–R33 裁决链**：R30 Reality Audit 定新主问题=**部署链一致性**（WM 固有 TF 0.02–0.04 vs 部署链 0.20–0.32，8–10×）；R31 审计包（Q2=中等风险/无冲突；K1 base≡M0；mirror 动作响应弱 24×–5612×；契约 S3≠S2）；R32 Level-1（**D2→D3=97.0% 主瓶颈；动作条件化失效**；real latent 充分；A_k ×2.1）⇒ GO_ACTION_CONDITIONING；**canary 四臂**（Gate 0 ✓；Gate 1/2 fidelity 级有效；**Gate 3 全 FAIL；Gate 4 AC1 succ 0.600✓ 但 coll 0.180✗**）⇒ 不自动进 C-B；**R33**：33A 三问 + 33B warp NO-GO ⇒ **GO_EGO_ACTION_CONDITIONING_V2**；**R33C**：B1=**稳定器（接口修复 −10~−12%）但硬门 6/11 未过、不晋级**；**R34A**：**失败=路径级**（通道选择 6/越界 6/撞障碍 6；跟踪/表示缺失/未响应 0；覆盖率 94.7%）；**R34B0**：覆盖门 ✅（10/10、7/7、17/17）⇒ 问题是选择/生成；摘要非杠杆；**R35**：**外部 target-lane 规则=toll 失败集 0/10 缺陷源**；merge=横向几何；无 token 自主选择不可靠；**R36（本版）**：**终止回退 `acd22c1`（§5.51 作废）；直接执行 `7c0186f`=执行优先**——H1 骨架（Gate 0 8/8）+ H2 given-target 数据规格（99.98% 标签、门 15/15；T4 晚换道 caveat）+ H3 hooks（555 键）；DP-6 提取开工。
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

- **第四十一轮（本版；287 → 296 行）**：新增 §5.54（R36 实施方案登记版：阶段与门/实现约束/禁令/进展/待裁）；§0/§7 刷新；上版 287 行 = `git show 4a35dcb:docs/deepseekv4p1_argue.md`。
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
