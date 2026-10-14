# deepseek-v4.1 argue / 主文档（R33 执行完成：33A 三问闭合（碰撞=收费站静止障碍簇 6/13；overshoot=规划层放大 +117.9%；噪声<恶化）+ 33B warp **NO-GO**（0/4）；decision=GO_EGO_ACTION_CONDITIONING_V2；33C 待令（§5.42））

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-14 下午·三）**：GPT 修订答复（`f0ebfae`，97 行）——**反驳接受、初始化结案**；批准 33A（限时只读三问）→ 33B（W0/W1/W2）→ 33C（B0/B1/B2；v2 统一：隐藏层正常初始化 + 输出层 zero-init + **残差增益固定 1（删可学习 α）**）。**执行完成（本版 §5.42）**：①碰撞=**收费站静止障碍簇 6/13**（岗亭 3 + 排队车 3；交叉口 0；双证据链 13/13 repro=exact）；②overshoot=**规划/WM 接口项主导 +117.9%**（A −20.2% 反向、C +2.3% 可忽略）；③噪声：录制确定性 ⇒ 重复噪声=0、bootstrap 半宽 6.1/6.1/8.9pp ≪ 恶化 20.9/17.2/16.5% ⇒ 恶化真实；④33B **warp NO-GO**（严格 0/4；W2 纯重表达≈0；W1 小效应来自移除 OD 恒速先验）⇒ 按树 **decision=GO_EGO_ACTION_CONDITIONING_V2**。33C 建议 B0/B1 两臂（B2 建议删）+ 硬门沿用；**待 CG 确认后训练**。
> **本版说明**：第二十八轮（392 → 404 行）——新增 §5.42（R33 执行完成：33A 三问 + 33B NO-GO + 决策）；§0/§7 刷新；上版 392 行 = `git show 0188218:docs/deepseekv4p1_argue.md`；历版全文=本文件 git log。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **主线结果（压缩；详见 §2–§5）**：条件轨 1b `no_increment` → 12A NO-GO（关闭）；A3 确定性尾全 0（**动作可持性**）；LD 结构缺陷+响应 ~50× 弱（M2=接线缺口）→ R21 语义重建（3C 首现强正向）→ 3D-3 未过门 → **R25 首现闭环超 legacy（0.400/+7.3pp、off −20.7pp）** → **R26 Q1**（Δp99 1.017、calib50 0.50 vs 0.40）→ **R27 R1**（sat ✓ 但 tmo +26pp ✗）→ **R28 28A 归因 ✅**（F3 速度参考主轴；F1 κ-ds 耦合/F4 LQR 尾段）+ **K1**（teacher +1.1%、κ MAE −71.6%、sat −12.5pp、calib 0.52/0.12；ds≡legacy 未消除 tmo ⇒ 停止）→ **R29A**（代码修复 ✅；T1/T3 机制性失败、T2 正向未过门；Δκ 瓶颈=首步钉死 ⇒ 停止）。关键锚：legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58（calib50）；IDM 0.756。
- **round-30 结果（§5.31）**：Reality Audit 第一轮（R30-0..6b、零训练、零生产改动）——矩阵 3 系统（S3 训练栈严格劣：0.02/0.00）、R4 接口主瓶颈、**TF 消歧：WM 固有 0.02–0.04 vs 部署链 0.20–0.32（8–10×）⇒ 主问题=部署链一致性**（overshoot drift）；账本三布尔全实证；契约 v1 + 配置审计落档；**round-31 裁决单在 §5.31 尾**。
- **round-31（§5.32–§5.35）**：宣言（`1c1d91b`）全范式重排（A[WM]→B0–B4[IL/DAgger/MoE]→C0–C4[RL]）→ 我方受理（M1–M5）→ 答复 2（`5872f57`）70/20/10 → 答复 3（`f4faec8`）收敛（GPT 基本站我方）→ **R31 审计包五 lane 收口（§5.35）**：Q2 判决草案（中等风险/无冲突）、**8–10× 状态分布放大**（K1 base≡M0）、WM 条件化弱（24×–5612×）、契约 I1–I7 + MoE ±2–4pp；零训练。
- **round-32 提案（§5.36）**：GPT（`910b49b`）**全面接受 R31** + 自排 WM 五结构弱点 W1–W5（核验全部成立）+ Round-32 十探针 + 联合训练 Phase 0–4 + 验收门 + **MoE-off 基线 0.56**；我方 5 条修正/补充。
- **round-32 裁决（§5.37）**：GPT（`d86e91a`）**5 修正全收** + 批准**三级决策树**（Level 1=D0–D3 归因/Jacobian FD/Swap/Ego 充分性/Lipschitz；decision.md 7 枚举）；统一口径 K1e2+MoE-off、基线 0.56；**Level 1 已开工（4 lane 只读）**；执行注记 3 条。
- **round-32 Level-1（§5.38）**：四 lane 收口——**Gate E 主瓶颈 D2→D3=97.0%（latent transition）**、Gate B 未过（动作条件化失效）、real 充分 / WM lane 保留不足（0.61–0.77）、接口温和（A_k ×2.1）；decision 草案=`GO_ACTION_CONDITIONING`。
- **round-32 答复（§5.39）**：GPT（`cde2e51`）接受主方向 + **批准最小可逆 canary**（AC0/AC1/P-only/AC1-P；C-A/C-B；Gate 0–4；ΔR² 增益 + 泄漏防线）；措辞精确化（latent-state evolution block）；**P3b 只读开工、canary 待令**。
- **round-32 canary（§5.40）**：四臂执行完毕——**Gate 0 ✓**（默认关/逐位/966 tests）；**Gate 1/2 fidelity 级有效**（top-3 0.635→0.917✓、FD 38.7→2.74×✓、ds 单调 0.946✓、ΔR²_WM 全正、保留改善；泄漏单通道 ✗）；**Gate 3 全 FAIL**（AC1 E_if +20.9%/overshoot +16.5% 反向恶化；AC1-P 近中性）；**Gate 4：AC1 succ 0.600✓（+4pp 首破 0.58）但 coll 0.180✗、AC1-P 未过、P-only≡base** ⇒ **不自动进 C-B**；报告 `docs/round32_canary_report.md`。
- **round-33 答复（§5.41）**：GPT（`c3bf69f`）——canary 不晋级（"动作辨识变强 ≠ rollout 更正确"）；33A/B/C 计划 + Gate A–E；**P0"双零死锁"质疑 ⇒ 我方只读审计闭合**（冷启动 α=1 按设计；**ckpt vs M0 共享键逐位 0 差异**；step1 `2/5`→step2 `5/5`）；3 条方法论补充 + A5 增项；待 CG 确认开工。
- **round-33 执行（本版；§5.42）**：33A 三问闭合——碰撞=**收费站静止障碍簇 6/13**（AC1 新增 5 例 2/5 岗亭；交叉口 0；双证据链 13/13 repro=exact）；overshoot=**规划/WM 接口项 +117.9%** 主导（A −20.2% / C +2.3%；恒等式残差 0）；噪声=**重复 0 + bootstrap 半宽 6.1/6.1/8.9pp ≪ 恶化 20.9/17.2/16.5%**。33B **warp NO-GO**（严格 0/4；W2 纯重表达≈0）⇒ **decision=GO_EGO_ACTION_CONDITIONING_V2**；33C 建议 B0/B1（B2 删）+ 硬门沿用；待令。
- **旁线+收敛一句话**：dθ-only 0.728（+2.2pp、组合效应）；常态 μ 无系统性 bias；lane semantic 前三层已正证；未闭合=normal-state preservation + plan-level trackability（28A 解构：F3/F1/F4）；**R28/R29 收敛：横修→六步路径→LQR 尾段×速度耦合；R30 再定位=部署链一致性（想象 vs 现实 8–10× 差、WM 固有误差小）**。
- **约束**（持续有效）：plan 部署/评测协议；单 GPU；fail-fast；task-session 解耦 + 20 min 节拍；conda/python 只读；正式 run 不清理。**冻结维持**：P1-4、K-anchor、GRPO、WM/MoE、容量、recovery 扩采、几何反传；A4=冻结（条件见 §5.12）；禁制详见 §5.1/§5.4/§5.10/§5.13。

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
- **评测纪律**：fail-fast 加载校验（missing/unexpected/shape_mismatch=0）为硬条件；跨版本（s11）只作方向参考；footprint 规格 = `arc_step` 位姿 + 0.5–1 m 插值 + 同款 drivable 判定（引擎无可靠 signed distance 时退化 `inside_ratio/first_invalid_pose`，**不伪造连续距离**）；引擎 run-to-run 随机性地板 0.2–1%（判读噪声 ±0.2–0.4pp）；正式实验记录 = manifest/CSV/decision.md，不进定期清理。
- **Gates**：
  - **G1 拆分**：G1-diagnosis ✅（主要损失层=计划/表示侧）；**G1-ceiling ❌**——尚无任何已验证链 ≥0.756，"超越路径"须由 **配速修复 / 更强教师 / RL 超教师** 三者之一建立。
  - **G2 追平**：3-seed mean ≥0.74；off-road：eval500 ≤0.098、clean500 ≤0.097；collision ≤ IDM（eval500 ≤0.144、clean500 ≤0.174）；双集成立。
  - **G3 超越**：mean ≥0.80、worst ≥0.77、paired bootstrap LB>0、McNemar p<0.05、collision ≤0.144/≤0.174、提升非单一路型；声明 0.82–0.85 需 5 seeds。

---

## 2. 前序结果压缩存档（P0 → P1）

### 2.1 P0 审计结论（三问三答，2026-10-10 定稿）

- **上限**：d2 **0.706** > oracle 0.608 < **IDM 0.756** ⇒ 差距在"学到的 plan/动作质量"；关键行（succ/off/coll）=IDM 0.756/0.068/0.144、laneplan 0.442、D2 0.706/0.038/0.020、v8 模型 0.22–0.33（全表 `runs/forensics/`）。
- **Q2/Q3+防线**：保持 plan 协议（`repeat_action` 归零、A-hold 仅 RL 记账口径）；失败 76–78% off-road/压线、**>50% 带 plan-infeasibility** ⇒ **P1=计划侧**；d2 timeout rc 中位 0.849（55% ≥0.8）⇒ d2 可能低估上限；fail-fast 两拦（`08582f2`/`5e6e1b3`）。

### 2.2 排摸收官（容量分配证据）

- 赢家：**pri512**（eval500 0.332；clean500 0.334@e020）、rou64（0.370@e015）；attn2 温和+；组合 C-N1 负交互被拒；峰值依配置（e005 / e015–e020）；仅分配证据（C1）。

### 2.3 P1 前序结果要点

- **P1-0**：起点 pri512@e020；V3 平坦 <0.75；**V1（ds 拉伸）0.762/0.776/0.778**@1.20/1.25/1.30（tmo 0.236→0.086；vs s100 z=4.10；vs IDM s120–135 打平）；等价口径 PASS（499/500 逐位）。
- **P1-1**：17,256 行/1,493 eps、数据门 13/13；**勘误：k≥1 行反事实错位 9,377 行**（obs=学生态×标签=专家帧）；`takeover_only` 后 2,317 行。
- **P1-2（四轴全败，闭合）**：两层根因=①反事实错位（已修）；②**绝对目标恢复监督 × 失败/低速态不可分 × 高倍过采样**（偏移集中 v∈[2,4)、Δds +0.44~0.57、~40×/epoch）；详见 `docs/p1_2_diagnosis.md`。
- **P1-3**：三损失就绪（默认关、逐位不变）；**cos(grad_traj, grad_geometry)=−0.73**；**待几何 FP 门**（后判 FAIL，§5.3）。

### 2.4 P1 执行手册 v1 锁定要点（存档）

- **范围**：P1-0A/B/C → P1-1 → P1-2 → P1-3；条件 **P1-4A**（启动=监督模型 eval500 ≥0.60 + probes 全 PASS + sampled action 贯通）；不承诺 P1-4B / K-anchor 重启 / GRPO / Best-of-G / WM-MoE 对照 / 容量扫描。
- **手册 3 问**：① V3 配速 ≥0.80？（答：<0.75）；② rou64@e015 vs pri512@e020？（答：不显著取 pri512）；③ recovery+几何把 ~0.37 推到 ≥0.60？（**未达成**）。细则原文 git 历史 §7.10；契约 `docs/p1_implementation_contract.md`、`docs/p1_probe_spec.md`。

---

## 3. P1-v2 执行纪要（Step 1–3；压缩存档）

- **锁板/冻结**：可辨识性 probe → recovery_v2 采集 → gated residual pilot → 几何 FP 门 → 独立消融 → candidate recipe；intervention 标签=同状态"baseline 将失败∧expert 可恢复"；每步 Go/No-Go；无条件轨=诊断旁线（不改训练 target）；**冻结**：recovery_v1 训练、P1-4、K-anchor、GRPO、WM/MoE、容量。
- **Step 1/2（GO/PASS）**：1a conditional_pass（time-only 0.940 ≥ F1 0.934）→ **1b `no_increment`**（0.682/0.731）→ **1c F2m 0.804/0.843 重开**；**2,693 行/110 eps**（intervention 836、recovery 89.2%）、六硬门全 PASS（重放 1,216 行 0 失配）；v1 根因实证=同状态 obs 中位差 33.2；注记：mode 邻步切换 0.611（不得作训练目标）。
- **Step 3（FAIL）+旁线**：gate=0 逐位==base；离线过（AUC 0.870）但闭环激活 **16.1%≫离线 3.99%** ⇒ **clean150 0.3067（−3.33pp）FAIL**（部署迁移过触发，footprint 需活 env）；旁线 E=常态 μ 无系统性 bias；D=dθ-only **0.728（+2.2pp）**、饱和 0.132→0.043 ⇒ V1=组合效应。零生产改动；verdict=FAIL。

---

## 4. Round-12A 零训练诊断与主线关闭（2026-10-11 晚；压缩存档）

**授权链（round-12→14）**：零训练复盘三机制（gate 误报/残差幅度/时序抖动）+ 同架构仅一训；三约束（CPU/CUDA 全链一致性、common-prefix 首分歧、破坏性误触发 <5%）；**12A LOCKED**、统一 CUDA 口径、九条件（**允许"无有效区域"=判负关闭**）；并入闭环重跑+逐步遥测、12B 同状态双 rollout。

**执行+终判（零训练）**：逐 step 遥测 → 逐 episode 复现（gate-off 0.3400/gated 0.3067，13 翻负/8 翻正一致）→ CPU/CUDA 全链探针（40 eps 6 集翻转）→ **15 配置扫描**；全未过九条件：**gate 误报且无选择性**（闭环 16.1% vs 离线 3.99%、保守配置反选择）、残差 1.0 偏强（"救余量伤进度"）、连续触发+后端离散化放大；t050_s050_d2 反超 +1.33pp 但机制不可信。

**教训**：闭环度量不可迁移；destructive-FP 口径近空；common-prefix 仅同设备；**跨后端 ~15% ⇒ 统一 CUDA**；坑（npz SIGKILL、footprint 需活 env、P@10 带 n_pos/n_neg）；**gate 选择性才是瓶颈**。**NO-GO → 关闭 gated residual 主线**（不补 hard negatives/不扩 gate；产物 `runs/p1_v2/round12a/`；零生产改动）。

---

## 5. round-15→30 执行批次（结果导向；残档=已并入执行的往来记录）

### 5.1 round-15/16 授权链残档（`b2f96de`/`24bfaba`；已执行）

- **15**：关闭 gated residual；**反对 RL 优先**；根因重估="教师行为与训练目标不等价"（A 监督重构〔门 clean150 ≥0.50→0.60–0.65 才谈 RL〕/B V1 机制分解/C IDM+PP 安全残差〔≥0.82 门〕）。**16**：批准 A1/A2 零训练；禁制=gated residual 复活/recovery 扩采/tail decoder/PPO·GRPO/K-anchor/WM·MoE/容量/几何反传；预注册 **DAgger=teacher 低∧learner 高**、**A2 Go=succ ≥0.72∧off ≤0.08∧coll ≤IDM**。
- **Track/RL 门（存档）**：A=动作 BC+确定性尾（=A3）/B=分层 signed bias/C=IDM+PP（C1 FAIL 已关）；RL 启动门 6 条（基线 ≥0.60–0.65、动作决定执行、同 PlanHead+LQR 口径、clipping<5%、饱和可控、value 识风险）。

### 5.3 Round-15A 结果（2026-10-12 00:51；压缩存档）

- **A1**：teacher-forced（26,767 行）MAE_ds **0.064 m**/dθ 0.0060、sign 错 3.1%；learner（836）**0.406/0.0986（6.4×/16.5×）**、sign 错 21.7% ⇒ **共变量偏移**（链/当前 2.79）；唯一稳定 bias=learner 态 ds（−0.33…−0.51 m/step、4/4 路型）。
- **A2/Track C**：repeat 族最优 **V1 s130 0.778**（V1 ds×s 0.728 ≫ V2 0.676 ≈ V3 0.69 ⇒ 增益="拉伸 ds 使参考曲率相对速度下降"）；decay 全不敌（geo-decay 0.000–0.042）；**q1 天花板：正确当前动作+简单尾 0.72–0.78**、q2 饱和最低 dth_s085 0.043、q4 执行参考无需六步链。C1 union **0.790<0.82 FAIL（不实现）**；**P1-3 FP 门 FAIL（5 项）⇒ `ld_margin_weight` 维持 0、几何反传禁止**；结论：①共变量偏移已证；②V1 式 ds 拉伸=有效自由度；③"远期链=历史主损"可信未证——A3 同预算对照=裁决工具。**天花板提示：特权动作+最佳尾 ≈0.778≈IDM+2pp ⇒ G3 须超越教师或更优执行**。

### 5.4 round-17 往来残档（`732ba96`；已并入执行）

- 无异议开工：4 条落法（训练/尾解耦、隔离、对照臂、A4 契约）+ 验收全并入——结果见 §5.5。

### 5.5 Round-17 A3 结果（2026-10-12 08:49）：**FAIL（决定性负结果）**

- **机要数字（clean150；基线 0.3400）**：三臂确定性尾均 **0.000/off 1.000**；A3×chain **0.3533** ≈ 对照×chain 0.3267 ≈ 基线 0.3400 ⇒ **远期六步链监督"无害亦无明显收益"**；oracle×三尾 0.687/0.747/0.780；门（A3 R）succ✗/off✗/learner-误差✗（ds −6.0%）/steer-饱和✗（5.1–7.5× oracle）⇒ **判负；A4 不启动**。
- **同 ckpt 隔离（关键）**：chain 0.3533 vs 确定性尾 0.0000 ⇒ 瓶颈=**动作可持性（holdability）**，非尾形/非配方；A1 复跑 teacher ds ↓13% 但 learner 态 +6% ⇒ 共变量偏移未触及。
- **机制**：直道 dθ 抖 ±0.09–0.23 rad → repeat 3s 使 LQR 预瞄成紧弯 → 饱和 0.69 → 10–20 步甩出；**学习链充当执行噪声缓冲**；①动作可持性=独立属性；②learner 态缺口=第一瓶颈（→round-18 probe-first）。生产 `f8169fd`。

### 5.6 round-18 往来残档（`95f23b0`；已并入执行）

- 接受 probe-first（两层根因：primary 动力学 shortcut + learner 态放大；lanekeep 归 primary）；2 处事实核查（LD 7 维已隐式含 lane-relative 几何；`ld_hist` 是输入≠不进 memory）；4 条落法与 3 验收补丁已并入 §5.8。

### 5.7 round-19 往来残档（`1ea85dc`；已并入执行）

- **七项代码论断逐条核验=全部成立**（`model.py:505-507` PlanHead 只看 ld_pool 均值；`spatial.py:103-106` 伪边；`mem.py:699-705` LD 仅当前帧；`model.py:627-632` K-anchor lane_ctx=0；`ld.py` 环优先排序/ego.lane 优先；注释漂移）——量化见 §5.8；0A/0B/0C 前置+全部修正（全语义镜像/输出非充分/Probe B 分层/primary_only 旁路）+4 条实施路线已执行。

### 5.8 Round-18A 结果（2026-10-12 19:2x）：**拓扑/输入结构性缺陷 + primary 几何响应 ~50× 过弱（动力学主导）**〔round-20 复核见 §5.10〕

- **0A/0B/0C**：伪边率 **76.5%**；ego lane 95.07%；**断点：失败 0.5146 vs 成功 0.0363（14×）**（非因果、共现因子）。**四 probe（n=500）**：zero_ld 0.0153；translate/rotate ±0.002/±0.0026（~50× 过弱）；zero_history 0.0134（几何 5–7× ⇒ 动力学主导）；镜像不闭合；primary_only≈full ⇒ `input_topology` 缺陷→最小语义特征路线；`primary_response` 弱→显式池化（已全部执行）。
- **长期坑**：LD 消费走 `ld_hist[:,-1]` ⇒ 干预须六帧一致。

### 5.9 round-20 往来残档（`37ee932`；已并入执行）

- 接受三项收紧（伪边=架构证据非闭环因果、几何响应相对口径、zero_history OOD 温和对照）+ 批准 M0/M1/M2 消融+20A/20B 探针（验收门：**M2<0.45→转数据侧**）+3 条操作补充——已执行（§5.10）。

### 5.10 Round-20 结果（2026-10-12 20:3x）：**M2 触发预注册停止——停 LD 架构精修、转数据侧**

- 同批 M0=0.3267；20A 模型对槽序置换**本就近似不变**（|Δmu|≈2–5e-06）⇒ 伪边输出层"惰性"；20B 失败 switch 0.511 vs 0.037、判别量=动作基噪声；M1 **0.3533（+2.67pp）** 不过门但无害、M2 **0.0000（崩）** ⇒ 预注册触发转数据侧（eval500 跳过）。坑：先 set 再 load；**离线 loss 不预示闭环保崩**；**更正（§5.11）**：M2=接线缺口、"穷举"撤回；M3/重采/A4 未启动。

### 5.11 round-21 代码复核残档（`ed9637d` 前轮；已并入执行）

- **指控成立**：`plan_step`（`model.py:534-536`）恒 `masked_mean` ⇒ PlanHead 全路径恒用均值，M2 attended 读出只进 policy/value；`set_attn` 一次改 3 变量（关链+LD 全排除+随机初始化）、无 gate、全灭 NaN 风险 ⇒ **§5.10"穷举/停 LD"撤回**。立场（已批准）：M2a/M2b 修正阶梯（统一 `aggregate_ld`、α=0 逐位、4 契约测试）；R21 语义重建=fail 分支预备案。**教训：此后"替换 X"类断言必须配消费测试。**

### 5.12 round-22 裁决残档（`ed9637d`；已并入执行）

- 批准四臂 M0/M1/M2a/M2b + Gate A/B + clean150/eval500（限 1 session；失败自动转 R21）；注记：same-lane-as-ego 代理、M0/M1 重训保同批、先 set_ld_mode 再 load——结果见 §5.13。

### 5.13 Round-22 结果（2026-10-12 23:0x）：**Outcome D——M2a/M2b 未过开发门 → 按预注册转 R21 语义重建**

- Gate A 全过（9/9；α=0 逐位、消费契约；M0/M1 与 20 批 168/168 max|Δ|=0.0）；**Gate B/闭环未过（量级不足）**：α 0.0132/0.0129（≈1.2%）；clean150 M2a 0.28/M2b 0.3267（tmo +8.67pp）、e002 回落 ⇒ **量级问题非接线**（条件化读出不被支付）。

### 5.14 R21 语义重建结果（R21-0→R21-2，2026-10-12 24:0x）：**三阶段门全过；主因定位经 §5.15 更正为"待 R21-3 裁决"**

- **R21-0/1（门过）**：唯一纯函数 target-lane 选择（route 优先/ego 兜底；4,632 探针 0 切换）+12 维通道+LD 逐槽补充；schema v7、909 passed、可行性 99.4%；pilot **9,515 行**（A 40.4%/B 42.7%/C 16.9%；同状态 1,013/1,013、expert 恢复 92.6%）。
- **R21-2（零训练）**：raw 全过（R²≈1）而模型侧全灭（ld_pool lat 0.191、fused −0.046）；sign 73%、curv 8.4%、**history_shuffle≫lane（15×）** ⇒ **唯一行动=lane_semantic 显式接线+复测**。产物 `runs/p1_v2/round21/`、`datasets/semantic_v1_pilot/`。

### 5.15 round-23 裁决残档（`085f4f3`；已并入执行）

- **三项指控核验成立**（net 零消费/12 维未归一/topology-margin 语义）⇒ 批准 R21-3A..3C（归一化+改名+target token+分组池化+统一消费；canary；复测门）+S0/S1（弃 S2）+正式采集后置；§5.14 措辞更正；注记：并行采集按保守、改名保 pilot、六帧铁律——结果见 §5.16。

### 5.16 R21-3A..3C 结果（2026-10-13 01:0x）：**接线+契约+Canary 全过；3C 表示 3/3 过、action 4/5（曲率 20.1% 近门）——首现强正向；门后停止待裁**

- 三项修正（归一化单一出处、改名逐位不变、margin 降级）+接线（target_lane_token 零+逐槽增广+分组池化+`build_geometry_context`）；16 新测、**925 passed**；3C：token 1.000/0.998/1.000、**sign 99.4/99.2%**（旧 73%）、lane **4.31×**、**曲率 20.1%（近门）**→ 催生 3D；产物 `runs/p1_v2/round23/`。

### 5.17 round-24 裁决残档：**批准 R21-3D（代码语义一致性 + 独立集复测 + 小规模闭环 canary）；正式采集仍后置——已开工**

- **三项指控核验成立**（semantic 仍带链/rollout 重复 t0 动态语义/target 正确性缺审计）⇒ 批准 R21-3D（3D-1 一致性/3D-2 独立复测/3D-3 canary；正式采集后置）；曲率措辞更正（"方向强证、幅值未闭合"）；未达门只定位、不转 DAgger/MoE/RL——结果见 §5.18。

### 5.18 R21-3D 结果（2026-10-13 10:1x）：**3D-1/2 门过（独立集 5/5，含曲率 25.0%）；3D-3 小 canary 未过门（S0 −8.7pp/S1 −8.0pp）→ 按预注册停止并定位**〔追注：3D-3/B 行标签错配污染，见 §5.20〕

- **3D-1/2（门过）**：chain 更优（lane 4.65×；冻结栈混淆在案）⇒ 保留链；**static_role_only 最优**（lane 6.75×、curv 29.9%）⇒ 默认；target correctness 983 状态 **1.00/1.00/严重 0.00**；独立集 8,953 行：sign 99.6/99.8%、mirror 98.4/99.6%、lane 6.17×、curv 25.0%。
- **3D-3（未过门→停止）**：3d3 53,765 行；S0 0.240/S1 0.247（−8~9pp）；独立集 sign 0.56/0.54；定位=**替换式初始化代价+语义未被 action 采用**（chain/rollout/target 嫌疑全清）→ 催生 R25；产物 `runs/p1_v2/round24/`，零 20 万级采集。

### 5.19 round-25 裁决残档（`8b35e84`；已并入执行）

- **两项指控核验成立**（替换式接线分布冲击/`_head_tokens:796-798` target token mask 恒 1）⇒ 批准 R25：残差化（β=0 逐位）+mask 修复+分阶段解冻+动作级监督（加权/margin/paired）+联合 selection；三臂 P0/P1/P2；门 A–E；失败即停（不堆 loss/数据、不转 DAgger/MoE/RL）——结果见 §5.20。

### 5.20 Round-25 结果（2026-10-13 12:3x）：**残差平滑接入成功安装"采纳"；P1 首现闭环超 legacy（succ +7.3pp、off −20.7pp）但门未全过 → 按预注册停止+定位；另发现并修复 R21-1 遗留 B 行标签状态错配**

- **25A/Gate A 全过**：`geometry=legacy+β·Δ`、`target_token_eff=β_token·token`（β 初值 0 进账本；target token 折叠进 summary 槽）；β=0 全链 max|Δ|=**0.0**；**937 passed**。
- **重大数据 bug（R21-1 遗留）+修复**：B 行 obs=置位态×标签=恢复窗后（corr **−0.156**）→ 修复=置位后立即查询（10,803 行、99.96%、corr **+0.606**）；**追注：3B/3C、3D-3 S0/S1 与首跑 P0/P1 均受污染**。
- **三臂+门+定位**：P1 采纳（βc +0.10~0.12、sign ~1.0、lane 106–184×）vs **P0 不采纳**（lane×0.13–0.6）⇒ **动作级监督=采纳必要条件**；B 4/5（mirror-lat 0.891✗）；C 3/4（teacher +57%✗）；D **succ 0.400（+7.3pp✓）/off 0.347（−20.7pp✓）** 但 tmo +14pp✗、sat 0.894✗；定位=量级过强不对称（2–4× expert）+lat×heading 交互未覆盖；P2 劣于 P1 ⇒ Primary 冻结；`runs/p1_v2/round25/`。

### 5.21 round-26 裁决残档（`1427ed3`；已并入执行）

- 核验成立 ⇒ 批准 R26 四项同改：①B 标签 v1 全链禁用+v2 fail-fast；②监督=相对 legacy 修正量；③分组幅度软锚（P95/P99）；④5×5 全镜像 paired。Q0=P1 重跑（基准）/Q1=+②③④；门 A–E（D：succ ≥0.42 且 +7pp）；注记：L_delta 绑定语义 delta+基底冻结+正常态零修正——结果见 §5.22。

### 5.22 Round-26 结果（2026-10-13 14:0x）：**Q1 全面优于 Q0（26B/C/D 均生效）；硬门仍未全过（teacher +40~44%、sat +14.5pp）→ 按预注册停止+定位**

- **26A–D 落地**：标签 v2 fail-fast（**v1 暴露 0**、**ep1091 4 行剔除**、corr +0.606）；delta 监督（`mu_on−mu_off.detach()` vs `expert−legacy`、基底冻结、4 单测）；`L_amp`；5×5 镜像 paired（12 字段）；幅度限额 P95/P99=**0.199/0.329 rad**。
- **关键结果**：**Q1 e1 硬门 6/7（仅 teacher +39.6% ✗）** ⇒ 无连续两 epoch ⇒ **停止**（D/E 未进入）；Gate C 4/6（pert **−53.6%**、learner **−51.5%**、sign err −66.3%、Δp99× **1.017✓**；✗=teacher +43.7%、sat +14.5pp）；**calib50 Q1=0.50**（legacy 0.40）；幅度对齐 0.076–0.127 vs expert 0.065–0.106。
- **未闭合+坑**：teacher 保持性（+40~44%）；steer 饱和（0.949 vs 0.804）；坑：eval/gate 默认 MoE=on 与训练口径不一致（`--moe-off` 诊断、结论不变）；独立集分母索引/semantic_off z_ld 已修；`runs/p1_v2/round26/`；待 round-27。

### 5.23 round-27 裁决残档（`268151e`；已并入执行）

- 核验成立 ⇒ 批准 R27：27A 饱和归因（六机制、≥90% 可归类、**无归因不得设计新饱和 loss**；禁全局 teacher 零修正强压）、27B N0/N1/C 分解、R0=Q1 复现、R1=+条件 preservation（仅 N0）+归因驱动 trackability；门 A–E（E：clean150 ≥0.43 且 +8pp）——结果见 §5.24。

### 5.24 Round-27 结果（2026-10-13 15:5x）：**27A/27B 门过、R0 逐位复现 Q1；R1 修好饱和（sat −9.4pp ✓）但换取 tmo +26pp 与 teacher +59.6%、plan Δκ 2.1–2.3× ⇒ 按预注册停止+定位**

- **27A（✅）**：Q1 sat **0.952** vs legacy 0.816；**73.7% 场景必要**；驱动=**trackability 36.3%/重规划抖动 34.9%/首步 27.8%**（尾段 1.0%、Δκ 0%）。**27B（✅）+R0**：N0/N1/C=**14.3/30.0/55.7%**（教师行 21.7/45.3/33.0）；`load_samples_v2` 硬拒 B 行 v1；**R0=Q1 逐位复现**（4 epoch+calib50 全同）⇒ 同批基准成立。
- **R1**：**sat 0.949→0.710 ✓、learner −72.2%、N0 P95 ↓40%**；**但 tmo 0.12→0.34 ✗✗、teacher +43.7%→+59.6% ✗、plan Δκ 2.12× ✗（expert 0.0169 vs R0 0.0393/R1 0.0359）** ⇒ 停止（D/E、eval500 未跑）；定位=**ds 进度与转向权限未分离**；`runs/p1_v2/round27/`。

### 5.25 round-28 裁决残档（`2acca30`；已并入执行）

- **核验成立、答案认账（根因升级=横纵耦合）** ⇒ 批准 R28：28A 零训练耦合审计（物理量换算+遥测补齐；F1 固定曲率/F2 固定 ds/F3 独立 v_ref 覆写/F4 尾段；≥90% 归因、逐 episode paired）+ 28B **K0=Q1 复现 / K1=仅横向 κ residual（ds≡legacy、dθ=κ·ds、外部接口不变）**；门 A–E（C：ds 逐位=legacy；D：succ ≥0.43 且 +8pp）；K1 败 ⇒ 不加深纵向头。——结果见 §5.26。

### 5.26 Round-28 结果（2026-10-13 17:5x）：**28A 归因 ✅（主轴 F3 速度参考；F1 κ-ds 耦合、F4 LQR 尾段命中）；K1 仅首步解耦不足（ds≡legacy 未消除 tmo；"否定"表述经 §5.27 更正——S1 价值保留）+ S2 崩塌 ⇒ 预注册停止**

- **28A（Gate A ✅，覆盖率 1.0）**：物理量（Q1 vs legacy）——throttle 饱和 **0.50 vs 0.19**、|e_y| 0.99 vs 0.51、raw steer 2.88 vs 1.40；**chord 效应仅 0.005 m/s**（参考速度差异来自 plan ds 本身）。四族反事实（paired 50）：**F3（几何不动、v_ref×1.15）tmo 12%→8%、off 不变、succ 52% ✓**；**F1（κ 不变同缩）tmo→0 但 off 30%→52% ✓（κ-ds 强耦合）**；**F4（只改尾段）raw steer +0.18、tmo 12%→2% ✓（LQR preview/chain 主耦合源）**；F2 未命中。13 问题 episode 全被 ≥1 族改善。
- **28B**：**K0=Q1 逐位复现**；**K1 S1 e1/e2 强**（teacher **+1.1%**、learner κ MAE **−71.6%**、lane sign 1.000、**sat −12.5pp vs legacy**、plan Δκ 3.90×→2.78×、calib50 **0.52/off 0.12/rc 0.863**），**S2 解冻末层即崩塌**（e3/e4 teacher 0.28、calib 0.0、off 1.0）。
- **门/结论**：B 4/5（plan Δκ 2.78× ✗、κ mirror 0.70 ✗〔common-ds 复核=真横向不对称〕）；C 3/4（**ds 逐位 ✓ max|Δ|=0；tmo +20pp ✗**）⇒ 无过门 epoch ⇒ 停止（无 clean150/K2）；**ds≡legacy 未消除 tmo**（0.28 vs 0.24 vs legacy 0.08）⇒ **主因再定位=横向修正改变六步路径 × LQR 尾段预瞄 × 速度耦合（F3/F4）**；S2 解冻崩塌第三次（R1/K1）⇒ 该支路**正式关闭**；`runs/p1_v2/round28/`。

### 5.27 round-29 收编：**接受"否定"表述更正（K1 S1 价值保留）；批准 29A 零训练尾段投影（T0–T3+独立 v_ref；S2 正式关闭）——已执行（§5.28）**

- **措辞更正（认账）+主因再定位**：被否定的只是"**仅**固定 ds + κ residual 即可自动消除 tmo"；**K1 S1 价值保留**（teacher +1.1%、κ MAE −71.6%、sat −12.5pp、calib 0.52/off 0.12）；横修改变六步路径 → LQR 尾段预瞄 → 横误/速度/油门耦合 ⇒ 即便 ds 不变仍掉速+tmo；"语义横修如何形成可执行六步路径 + path-conditioned 速度跟踪"=当前主问题（对齐 PLUTO/AlignDrive）。
- **29A**：固定 K1 S1 ckpt；T0 原案 / T1 Δκ 约束投影（expert 分组 P95/P99、`κ′₀=κ₀`）/ T2 尾段 ρ 衰减至车道曲率 / T3=T1+v_ref×1.15；门 A（Δκ/raw steer/sat/|e_y|/方向）→B（tmo/rc/throttle/均速）→C（succ ≥0.54/off ≤0.12/coll/tmo ≤0.12）→clean150/eval500；**S2 解冻=正式关闭**；条件批准 LQR 曲率前馈（默认关、零训练对照）。代码修复+注记见 §5.28。

### 5.28 Round-29A 结果（2026-10-13 20:2x）：**代码修复落地；T1/T3 机制性失败（含 M0 对照）、T2 唯一正向（succ 0.52→0.58）未过门；Δκ 瓶颈="首步钉死" ⇒ 预注册停止**

- **代码修复（✅，8 新测、966 passed）**：`plan_valid_delta_kappa`（pairwise ds≥0.5 真掩码）；低 ds **三态 κ 门**（禁误判 N0）；负 ds **fail-fast** + Δκ 标签字段；tracking 4 项只读耦合遥测 + 曲率前馈（默认关）；`--tail-projection`（默认 none、既有行为逐位不变）。
- **投影（calib50，paired 50；T0 复现 K1e2 逐值）**：**T1** Δκ 0.052→0.0058 但 raw steer 3.81→**6.78**、|e_y| 1.22→2.04、**off 0.12→0.64、succ 0.52→0.30**；**T3** 更糟（off 0.90/succ 0.08）；**M0+T1 off=1.00** ⇒ **"平滑本身"机制性失败**（expert 分组 Δκ 0.003–0.009 ≪ 模型 0.052 + `κ′₀=κ₀` 钉死 ⇒ 尾段拖成近恒曲率弧），非 K1 语义尾问题。**T2 唯一正向**：**succ 0.52→0.58 ✓、sat 0.679→0.580、方向 0.737→0.967 ✓、rc +0.013、tmo 0.28→0.18**；但 **off 0.14（>0.12）、coll 0.10、tmo 0.18 ⇒ Gate C 未过**。
- **新结构定位**：**plan Δκ P95 由首步对（κ0→κ1）主导**——T1/T2 均钉 `κ′₀=κ₀` ⇒ 仅改尾段无法降该指标；下轮压 Δκ 须同时约束**首步或首-尾连续性**。T2=唯一正向确定性层但未过门（可作蒸馏候选，前置=先解决 off/coll/tmo）。
- **收口（→round-30）**：T2 蒸馏/首尾 Δκ 并入 R30 禁训清单（§5.30）；失败备选路线（tracker 重设计/速度相关前视/曲率前馈 ↔ Frenet 路径条件化）由 R30"部署链一致性"新主问题覆盖（§5.31）。产物：`runs/p1_v2/round29/`；零训练。

### 5.29 round-30 开场：CG5.6 全栈总审核验 + 攻防（压缩归档；输入 `61bf7ab`）

- **总判**：CG5.6 全栈总审（Stage A/B/C Reality Audit 提案）§二 7 条代码论断**主体成立**；4 处校正：C1 `loss=direct_multi_step` 过时（真实主监督=latent consistency 1.0，`stages.py:2229,2328`；od/ld 为诊断 0.1/0.02）；C2 被切=**状态链 credit assignment**（共享转移头/空间 MP 受全 6 步监督）；C4 实际 optimizer=**排除式**（除 `policy./value.` 全参、YAML `trainable` 未读；MoE off 死参）；C7 执行器=**LqrTracker**（非 Exact）。**真空白=R4 一致性 / WM 动作条件化 / 逐 horizon 梯度 / oracle·frozen 臂**；C6 尾 5 步无直接动作标签（仅 traj6 位置 0.1；phase3 才有 `action_chain:0.2`）；R1/R5 已有等价物。
- **攻防**：接受其纪律（不删 detach/不启六步 BPTT/不新增容量·loss/不启 DAgger·MoE·RL/LQR 冻结）＝与现行冻结一致；反驳 4 点（无需暂停训练——本线皆零训练/小 canary；绝对门与 latent 口径不符→改接口门 `π(ẑ)≈π(z)` 实测后锁；BPTT 课程前置=梯度可达性否证（只读）+受控 canary 裁决；部署迁移 Gate 先量化）。C1 升格为**配置契约 bug**（键级清单见 §5.31 R30-5）。
- **处置（→已全部并入 §5.30/§5.31 执行）**：R4 一致性 > 部署迁移量化（零代码 `{exact,lqr}×{plan,repeat_action}`）> WM 动作条件化 > 逐 horizon 账本；"训练/部署执行口径对照表"=docs 交付；漂移修复=独立小项、冻结期不动代码。

### 5.30 round-30 收编：CG5.6 终审受理——缩减版 Reality Audit（R30-0..6b）开工（压缩归档；终审 `acf1b8f`）

- **批准范围**=R30-0..5（冻结/矩阵/R4/WM 条件化/梯度账本/执行契约）+ 条件 R30-6（R4 显著即三臂）；**本轮禁训**：T2 蒸馏、首尾 Δκ、删 detach、BPTT、action token、DAgger、MoE、PPO/GRPO、LQR 修改、扩数据（T2 保留诊断 oracle）。
- **2 条执行注记（留档，round-31 复核用）**：①Stage B 现象非"状态标签错配"（R25 B 行才是真错配、已修）＝**预测态内部链上的间接轨迹级监督**，风险=WM 冻结质量 × 尾段直接动作监督缺失；②LQR+repeat 系 P0-1 显式口径修复（执行依赖记账变量 a_t；`plan` 尾=记账外变量，`trainer.py:7678-7703`）；"切 Exact 主训练"须先过（teleport 语义 reward/碰撞/终止审计 + 部署口径 A/B）。
- **验收口径（首轮，实测后锁）**：R4 分层带（interface/deploy <20% / 20–50% / >50%）+ WM 相对门（方向/mirror/单调/latent/next-action）+ 矩阵 paired 转换。→ 执行结果见 §5.31。

### 5.31 R30 缩减版 Reality Audit 第一轮结果（2026-10-13 夜·五；R30-0..6b 全执行、零训练）

- **结果**：
  - **R30-1 四格矩阵（8 run）**：exact 行结构退化（0.5s 决策协议仅消费 `plan[0]`，5 子步后 re-arm）⇒ 实为 3 系统；**S1≈S2**（K1e2 0.540 vs 0.520 配对平局；pri512 0.260 vs 0.360，S1 更差）；**S3 严格劣**（0.020/0.000，跨策略 0 翻正）⇒ 训练栈≠部署栈（结构级）；**v2 复跑（reward 复采 + 复现性）：逐 episode 8/8 全一致**（审计 §1.5）。
  - **R30-2 R4（calib50、7,970 决策步）**：E_if/E_deploy = 1.54→1.22（≫50%）⇒ 接口主瓶颈（预注册带判定成立）；E_z_ego 0.203→0.316；E_real ≈0.40 恒定（策略误差地板）。
  - **R30-3 WM 动作条件化（M0，N=200）**：动作诱导 Δz 弱于状态自然差异 1–3 数量级（ds 通道最弱 ~1e-4）；a_next 呈"反向修正"极性（~0.90）；**mirror 门按现构造不可判**（需全场景镜像重造）。
  - **R30-4 梯度账本（B=32）**：detach 截断 / `policy→WM 不可达` / `Stage-B WM 全冻` 三布尔全实证（决定性对照 0.0 vs 2.226）；共享 transition 全 horizon 有梯度。
  - **R30-5**：`stage_a_config_consumption.md`（70 键：CONSUMED 51 / IGNORED 14 / LEGACY 2 / DERIVED 3）+ 执行契约 v1（exact 参考消费语义、键级漂移清单）。
- **发现/分析**：
  - **R30-6 三臂（离线）**：k≥2 起 WM 尾段动作不如 frozen-hold（WM 更优 50%→10%→0%），oracle 平坦 0.40；机制=**尾段 ds 上冲**（2.56→3.64 vs mu 1.94/expert 2.30）+ δ-方向 2–3 步失配 + 幅度膨胀（overshoot drift）。
  - **R30-6b TF 消歧（全量 35,695 anchors、双模型）**：纯 WM TF 误差 ego 1−cos = **0.021→0.042**（pri512；M0 0.032→0.074），比 R4 部署链小 **8–10×** ⇒ **WM 固有能力 OK；根因=部署链一致性**（策略动作/状态分布漂移 + 0.5s 重规划替换尾参考（open-loop 语义）+ LQR 跟踪残差）。
- **问题（round-31 裁决单，待裁）**：①**链路一致性修复三支**：训练分布（learner-state/DAgger）、协议侧（cadence/参考消费语义审计）、监督侧（overshoot drift 指标化后预注册）；②**Stage C 口径**：S3≪S2 结构级 ⇒ 键级 fail-fast + 口径重选（Exact 主训练前置=§5.30 注记 2）；③**工程尾巴**：mirror 门重造（全场景镜像）、配置 fail-fast/删键批次、road_class 旁记。
- **产物**：`runs/p1_v2/round30/`（matrix×8 + v2 复跑[reward]、r4_K1e2、wm_action_probe_M0、grad_ledger、wm_tf_check×2、matrix_report）、`docs/round30_audit.md`（全文）、`docs/stage_a_config_consumption.md`、`.slim/deepwork/p1_v2/round30/`（journal/monitor/run_*.sh）；零训练、零生产代码改动。

### 5.32 round-31 宣言（GPT `1c1d91b`）受理审阅：范式流水线受理 + 5 条修正/前置 + 排队方案（待裁；本轮零实施）

- **受理总判（整体接受）**：全范式流水线（A[WM]→B0 单步 IL→B1 oracle-state rollout IL→B2 预测态一致性→B3 DAgger→B4 MoE→C0 Value warmup→C1 ExactTracker PPO→C2 部署同构 canary→C3 MoE-RL→C4 GRPO）+ 执行器政策（Exact=采集/上界/C0；LQR=部署+晋级门）+ 今晚 Stage-A-only 队列（Task1 契约修复 / Task2 复核 / Task3 pilot 50–100k / Task4 五探针 / Task5 动作忠实度+whole-scene mirror / Task6 A4 canary）+ 五条自动停止规则 + decision.md 五枚举。
- **与 R30 结果的适配核验（逐条）**：S3≪S2 → Exact/LQR 用途政策=吸收 §5.30 注记 2 前置精神 ✓；WM 固有 TF 0.02–0.04 → A2.2（policy 态一步退化）成 A4 前提的关键上界 ✓（细则 M1）；链误差 8–10×/overshoot → A2.3/A2.4+A4 目标 ✓（归因前置 M2）；账本三布尔 → A5 的 2-step BPTT canary 三条件一致 ✓；mirror 不可判 → 升级 whole-scene mirror ✓（**工具已有**：`r26_mirror.py::mirror_obs` 对合+实测校验）；配置漂移（F5）→ Task1 白名单/fail-fast ✓（细化 M5）。
- **修正/前置 5 条（分歧点，供再讨论）**：
  - **M1（A4 前提预注册）**：R30-6b 已证 WM 固有误差 ≪ 部署链误差 ⇒ A4（WM 分布适配）只对"policy 分布漂移"分量有效。建议：① A2.2 补"同模型 expert-state 一步误差"**内部对照**（不跨模型引用 0.02–0.04），以**差值**判定；② 预注册：差值小（≤~2× expert-state 量级）⇒ A4 难达 E_interface −40%，优先走 M2/M3 分解支路；差值大 ⇒ A4 前提成立；③ "尾段 ds overshoot −50%" 同步先过 Task4b 归因校准（上冲与预测态漂移相关 ⇒ A4 有望部分修复，幅度需先测）。
  - **M2（新增 Task 4b：三分分解，复用 R30 录制，<1h 实现+跑）**：把 R4 的 0.203→0.316（1−cos）拆为 (i) policy 分布（A2.2）/ (ii) 跟踪·执行残差 / (iii) 重规划尾替换。规格：a) **action-faithful replay**——用 `env__x/y/theta` 实测位姿链替预测 pose 链喂 st_gnn（z/掩码/步号从 R4 npz 重建），得 ẑ′_k vs z_real(i+k)；若 ẑ′≪ẑ ⇒ WM 转移在真实输入下无碍、主因=(ii)+(iii)；b) 替换量化——从 pose 重建实际执行动作 a_{i+k}，对齐 dec/env 双时钟，测与 plan(i)[k] 一致率。**该探针直接决定 A4 / 2-step BPTT / 协议三支的优先级**。
  - **M3（协议支路入决策树）**：decision.md 五枚举缺协议分支；建议增 `GO_PROTOCOL_AUDIT`（触发=M2 显示 (ii)+(iii) 主导）。候选：replan cadence（0.5s 重规划 vs 3s 参考的错配）、LQR 预览取点与参考消费窗口、kappa-compose 语义（exact 仅消费 plan[0] 的发现同族）。
  - **M4（数据 schema 口径）**：① pilot 每条 transition **同时记 raw 与 deploy-composed 两套动作**（K1 的 ds≡legacy 语义；实测：`kappa-compose` 目前只存在于诊断侧 `eval_a3`，采集/训练链路无此开关 ⇒ 需移植）；② A4 训练所用 action 必须与部署 rollout 的 pose 链语义一致；③ next-obs 对齐用 R4 双时钟（dec/env）；④ 失败窗 10% 分布项=两轮法（小批 policy 收集→按 outcome 定向补采）；⑤ 采集基座=dagger_collect（student+`{lqr,exact}`+同 schema 已有），需扩展显式 `policy_action` 字段（现 `prev_action`=专家上一动作，`dagger_collect.py:1446`，非学生动作）+ 上述语义位。
  - **M5（门的可操作化）**：① whole-scene mirror ≥95%：先定义度量子集（符号/方向类限 |base|≥阈；latent 距离类按镜像比值），首轮以**诊断**呈现——非等变网络未必能过、勿误杀；② config 三方等式（declared==optimizer==非零梯度）过强 ⇒ 改"declared==optimizer 硬门 + 非零梯度按模块报告（warn）"，注册表须含脚本链消费者（`epochs/batch/micro` 走 run_config→train.sh）；③ "未知键=0" 按 **stage 域**注册（A/C 分表），否则误伤他段键。
- **我方排队方案（时间盒；单 GPU 串行）**：Task1（契约，~2–4h 开发+测试；含 Stage A 白名单与 fail-fast）→ Task2（矩阵已 v2 100% 复现；仅 R4 抽 20% 复算 ~0.5h）→ **M2/Task4b（~1h）** → Task3（pilot：先 50k、按资源冲 100k；工具扩展后 ~2–3h）→ Task4/5（探针沿用 R30 工具 + `r26_mirror`；~2h）→ Task6（A4 canary：先建 60/40 混合数据〔合并文件或扩展采样器，`stages.py:1930-1935` 现为单源〕；3–5 epoch/1 seed；R4 选 ckpt）。跑不完按停止规则交 decision.md，不越门。
- **命名勘误（契约文档统一）**：宣言 §三 A1=数据 pilot 与 §四 Task6 "A1 WM Adaptation Canary" 撞名；统一为 A1=pilot / A4=adaptation（以 §三 为准）。
- **supersede 标注**：上一版禁训清单中"扩数据"指 recovery/DAgger 类扩采；本宣言新授权 pilot 采集（policy-conditioned transitions）——建议契约显式记录 supersede；其余禁项（T2 蒸馏/DAgger 训练/BPTT/action token/MoE/PPO·GRPO/删 detach）继续有效。
- **开门条件**：M1–M5 经用户/CG 裁决（接受或修订）后即可按队列开工（Task1→…）；本轮按指令**零实施**（未动任何代码、未采集、未训练）。

### 5.33 round-31 答复 2（GPT `5872f57`）受理审阅：70/20/10 受理 + Q1–Q3 对账（Q1/Q3 已有实测；Q2=新增审计项）+ Q2 规格（待裁；本轮零实施）

- **受理**：接受其补充（Stage B rollout-supervision 风险、Q1–Q3 框架、"路线 A=WM+Policy+rollout 联合系统（Dreamer 式），不追求 WM 单独 100 分"、KPI Tier1/2/3 分层、"先答三问再动训练"）；与我方 §5.32 方向一致（70% 接受）；本轮零实施。
- **对账（关键——三问现状）**：
  - **Q1（Policy 真态 vs WM 态，E_interface h1~h6）＝已有实测**：R4 部署链 E_if 0.64→1.74（≫E_real 0.40 地板）⇒"Q1 大"成立；**但归因未闭合**：R30-6b 证 WM 固有 TF 误差 0.02–0.04（≪链误差，8–10×）⇒ Q1 大 ≠ WM 转移差；分支"优先修 WM action conditioning"须先过 §5.32-M2 三分分解（policy 分布/执行残差/重规划尾替换）。R30-3 另证 latent 动作条件化弱 1–3 数量级（ds 通道 ~1e-4）——"action conditioning"可修对象是它，与 Q1 的"链效应"是两件事。
  - **Q3（Stage C 训练 vs 部署）＝已有实测**：四格矩阵——S3（repeat+LQR）严格劣 S2（0.020/0.000 vs 0.520/0.360，跨策略 0 翻正）；exact 行结构退化（仅消费 plan[0]）⇒"Q3 大"成立且已量化。
  - **Q2（Stage B rollout supervision 合法性）＝新增，尚无实测**：注意 R4 审的是**部署链**（K1e2+LQR 闭环），不是 Stage B 训练分布；Q2 需独立审计（规格见下）。R30 审计已完成（§5.31），Q2 属 round-31 新增项。
- **对"与 recovery_v1 一类"表述的反驳（保留 §5.30 注记 1 立场）**：B-label 是**数据错位**（obs=置位态×标签=他时窗）；Stage B 的 traj6（`traj_aux_weight=0.1`，`trainer.py:3601/2984`）与 phase3 action_chain（0.2，`trainer.py:977/2478`）是**同一 t0 帧、目标定义明确**的间接监督（冻结模型下的 model-based IL；rollout 状态链 detach=`net/model.py:1128`）——本质风险=想象态漂移 × 尾段直接监督缺失，**不是同类错配**；是否构成"一级风险"由 Q2 实测裁决（同意必须优先排除，不同意预设定性）。
- **Q2 审计规格（我方提案；零训练、只读；预计 <2h）**：
  - **Q2-a 训练分布接口差**：Stage B 协议下（val 行、policy rollout、冻结 WM）测 π(ẑ_k) vs π(z_k^real)（k=1..5），与 R4 部署链版同口径对照。
  - **Q2-b 监督合法性分解（梯度冲突）**：同批数据计算 L_action（首步）/ L_traj（traj6 0.1）/ L_chain（phase3 action_chain 0.2，若模型启用）在共享参数（policy/plan 头/编码器）上的梯度余弦——显著负冲突 ⇒"教错误状态上的修正"的直接证据。
  - **Q2-c 完美 WM 反事实（oracle rollout）**：把 rollout 的 ẑ_k 替换为真实未来 latent（编码 real_obs_{i+k}）重放，量化尾动作/traj 损失对 WM 不完美的依赖度（依赖大 ⇒ 监督建立在漂移态上；依赖小 ⇒ 近似合法）。
  - **Q2-d（条件）**：traj6 权重 0 vs 0.1 训练消融 canary——需训练授权，后置（本轮不做）。
- **supersede 澄清（需裁定）**：`1c1d91b` 的今晚队列（Task3 pilot 50–100k、Task6 A4 canary）与 `5872f57`"冻结…大规模数据采集（至三问实证）"**冲突**；我方建议以**新版为准**（证据先行）：先做 Q2 审计 + M2 分解（全零训练），暂不采 pilot、不训 A4。M1–M5 状态：M1=被新版冻结吸收；M2=本版核心工具（Q1 归因）；M3（协议支路）=仍不在 GPT 决策树、保留；M4/M5=随 pilot/契约批次后置。
- **开门条件**：若裁定（①Q1/Q3 以 R30 实测收账；②Q2 规格接受；③冻结以新版为准）⇒ 我方即可开工 Q2 审计 + M2 分解（零训练、零生产代码改动、<2h），随后按结果再议 A4/协议支路。

### 5.34 round-31 答复 3（GPT `f4faec8`）受理审阅：收敛——GPT 基本站我方；R31 只读审计包已开工

- **受理（收敛）**：GPT 本轮"基本站 Coding Agent 这边"：接受 ①Q1 已答（"真正炸的是 Policy(预测状态)"：oracle 0.40 vs WM-state 1.43）；②Q3 已答（S3≈0 vs S2≈0.52，"Stage C 当前训练系统根本不是部署系统"）；③Reality Audit > 继续调 loss；并**采纳 §5.30 注记 1 的定性**（"Stage B 风险≠recovery_v1 错配，而是预测状态链上的间接监督"=Scheduled Sampling 类分布漂移；"值得成为一级审计对象"）。**根因排序（GPT）**：P0 训练链≠部署链（Stage C）> P1 WM rollout vs real 动作分布分叉（E_if≫E_real）> P2 Stage B 预测态监督合法性（未知）> P3 lane semantic（降级为已解决）。
- **裁决**：批准 **Reality Audit 继续 + Q2 审计 + WM action fidelity + Stage C contract 审计**；**暂停** A4 adaptation / action token / 2-step BPTT / DAgger / MoE / PPO / GRPO；**Q2 完成前不启动任何新主训练**。⇒ §5.33 的 supersede 冲突**已解决**（冻结为准；昨晚队列 pilot/A4 不启动）。
- **注记 2 条（非阻塞，供下轮参考）**：
  1. **"A 一定存在"措辞**：R30-6b 已排除 WM 的"固有"误差形态（TF 0.02–0.04 ≪ 链 0.20–0.32）；A（WM 不够好）在部署链上的**贡献份额**与 B（Policy 从未在 rollout 态训练）/C（执行·重规划）尚未分离——由 M2 三分分解裁决，先分解再排序。
  2. **工作清单细化**：R31-1/2/3 照单接受；**追加 M2 三分分解**（服务 P1 归因；复用 R30 录制）与 **Q2 规格 Q2a–c**（训练分布接口差 / 梯度冲突 / 完美 WM 反事实）。
- **开工（按指令；全只读、零训练、零生产代码改动）**：R31 审计包五 lane——①Q2-a/c（Stage B 协议 rollout 一致性 + oracle rollout 反事实）；②Q2-b（L_action vs L_traj 梯度冲突）；③R31-2 mirror（whole-scene，复用 `r26_mirror`）+ 弱响应诊断；④R31-3 Stage C contract（collect/update/eval/deploy 四映射一致性）；⑤M2（action-faithful replay + 重规划替换量化）。产物落 `runs/round31/`、`docs/round31_*`；完成后按结果更新本档并待裁。

### 5.35 R31 审计包第一轮结果（五 lane 全收口；全只读、零训练；2026-10-14 凌晨·五）

- **Q2 判决（我方）**：Stage B 监督 = "中等间接性代价 + 无冲突证据 + 非灾难" ⇒ 不构成"一级非法监督"（GPT 的"一级风险"降级为**已量化的中等风险**）：①Q2-a 协议内退化中等（E_state 0.022→0.065、E_if L2 0.145→0.357；rollout ADE 0.544 vs oracle 0.202=2.7×）；②Q2-b **无梯度冲突**（cos 全正、0 conflict）；③Q2-c oracle 代价亚米级（ΔADE=0.341/ΔFDE=0.796）。
- **M2**：部署 E_z **不由 pose/执行错配驱动**（实测位姿重放 Δ≤8.7% rel；位姿敏感度弱）；**重规划替换真实**（执行≈当前 μ，|Δds|→2.0m、松一致率 0.24→0.06）。
- **本轮最大新发现（量级级）**：**K1 的 base 网络与 M0 逐位相同**（K1-only=20 键 semantic_wiring+β）⇒"K1 训练破坏 latent"排除；同一 base 下 expert 协议 E_state=0.026→0.08 vs 部署 E_z=0.203→0.316（**8–10×**）⇒ **WM 转移误差强依赖 policy 状态/未来分布**（候选=起始态分布/目标分布/交通耦合/重规划语义）——为"policy-conditioned WM 适配（A2.2/A4）"提供直接量级支持。
- **WM 条件化**：whole-scene mirror 坐实 scene/action 响应比 **24×/5612×/1392×**（动作/位姿条件化弱）；mirror 一致率 restricted 0.842（<0.95，首轮诊断）。
- **契约**：S3≠S2（结构级）；两 eval 栈默认相反；exact 仅 plan[0]；**MoE 口径差 ±2–4pp（训练口径略优，结论稳健）**。
- **下一步候选（供裁，零训练优先）**：①部署状态归因探针（分离起始态分布 vs 目标分布；需补录 obs，~1–2h）；②action/pose conditioning（A5；证据已足、归因先行）；③契约修复批次；④A4 canary（前提已具）；⑤2-step BPTT canary（后置）。
- **产物**：`docs/round31_audit_summary.md`（全文）、`docs/round31_stagec_contract_audit.md`、`runs/round31/probes/*`（5 探针）、`runs/round31/moe_off/*`；零训练、零生产代码改动。

### 5.36 round-32 提案（GPT `910b49b`，1641 行）受理审阅：R31 全收 + WM 五结构弱点核验（全成立）+ 5 条修正/补充（待裁；本轮零实施）

- **受理（收敛加深）**：GPT 全面接受 R31（S3≠Deploy 坐实；WM 固有非主问题；Q1/Q3 已答；**Stage B 非 recovery_v1 同类**——"危险但不属于犯规"）；路线 A 加固（S2 0.52 vs S3 0.02 ⇒ 尾 5 步有价值、路线 B 非正解）；反对现在 DAgger（链路未闭合、治标不治本）；根因排序 P0 Deploy chain > P1 rollout→Policy 接口漂移 > P2 Stage B 监督 > P3 WM 条件化 > P4 lane semantic；冻结 A4/DAgger/MoE/PPO/GRPO/大规模采集。
- **WM 五结构弱点核验（逐条 file:line 复核，全部成立）**：W1 动作仅经 pose 间接进入 + pose 被 detach（`net/model.py:1104`；实测 24×–5612×）✓；W2 **无 ego 物理头**（`net/st_gnn.py:69-78` 仅 od/ld head；`ego_next_pred` 由 plan head 产出、非 z_ego_next，`stages.py:1977-1978`）✓；W3 OD 图位姿 t0 恒速外推、预测不回灌（`model.py:1099-1103`）✓；W4 node_mask 固定 t0（`model.py:1066/1112`）、presence/entry 仅诊断 ✓；W5 latent target=embed(future).detach() 而 encoder 同训（`stages.py:8/1983`）⇒ 共适应风险 ✓。
- **5 条修正/补充（分歧点，供裁）**：
  1. **Q2 首轮已答，非待做**：R31 已执行 Q2a/b/c（E_state 0.022→0.065、梯度 cos 全正、ΔFDE=0.796）⇒"Q2 必须做"应改述为"Q2 首轮已答（中等风险/无冲突）；Round-32 做 **tail-specific 深化**（尾 5 步动作链质量 vs LQR preview 依赖）"。
  2. **补 1 探针：部署状态归因**（§5.35 候选①）——GPT 清单缺"部署 8–10× 的状态分布 vs 目标分布归因"；建议并入 Round-32（需小录制补 raw obs）。
  3. **Action Jacobian 须有限差分（关键修正）**：`pose.detach()`（`model.py:1104`）+ 状态链 detach（`:1128`）⇒ **∂z_k/∂a_0 解析恒 0**；探针=(a) 数值差分 + (b) 解析=0 作为 detach 语义证据（两者并报）。
  4. **MoE-off 基线口径采纳**：**0.56 取代 0.52** 为同口径基线（我方敏感性已证）；写入契约（训练/评测一致；MoE-on 待专家训练后）。
  5. **mirror ≥95% 门**：首轮按"诊断+子集口径"呈现（实测 0.842；非等变模型 + t0 冻结简化）——门值待 EMA/等变训练后锁。
- **联合训练策略（Phase 0–4）+ 验收门（W0/W1/W2/P1/J1/Deploy）**：原则接受（禁"全解冻+总 loss 端到端"；交替+梯度隔离；Phase 4 才有限联合梯度+冲突投影）；阈值按"首轮相对门、实测后锁"。
- **开工条件**：5 条经裁（接受/修订）即可启动 Round-32 只读探针包（10+1 探针；默认 MoE-off、零训练、零生产改动、hook 默认关/不写 ckpt/不改 RNG）。

### 5.37 round-32 裁决（GPT `d86e91a`，677 行）：5 修正全收 + 批准三级决策树——Level 1 开工

- **受理**：GPT 接受全部 5 条修正（Q2 首轮已答→tail-specific 深化；部署状态归因升为最高优先；有限差分 Jacobian〔并确认"解析=0 只证梯度契约、不证物理不响应"〕；**MoE-off 0.56 锁定为 Round-32 开发基线**〔注：+4pp 仅 2 episode 翻转，不作稳定收益〕；mirror 首轮分解诊断、不锁 95%）。**批准 Round-32 只读审计**，重排为**三级决策树**：**Level 1**=D0–D3 部署链四段归因 / 有限差分 Jacobian+Action Swap+zero/opposite / Ego 状态充分性 / Policy Lipschitz；**Level 2**（仅当 L1 指向 WM 内部）=latent rank / prior-vs-residual / graph oracle / mask turnover / target drift；**Level 3**（仅当需联合适配）=梯度冲突 / joint drift hook。**decision.md 7 枚举**（+GO_POLICY_INTERFACE_ROBUSTNESS / GO_ROLLOUT_PROTOCOL_FIX / GO_DYNAMIC_GRAPH_FIX）。**本轮禁训**：训练/action embedding/ego physical head/predicted pose 回灌/EMA/BPTT/联合解冻/DAgger·MoE·PPO·GRPO/LQR 修改。
- **统一口径**：K1e2 + **MoE-off** + K-anchor-off + 既有最佳 rollout_semantic + LQR 不变；baseline 0.56（仅开发参考）；hook 规则（默认关 / max_abs_diff=0 / RNG 不变 / 不写 ckpt）。
- **我方执行注记（3 条，非阻塞）**：①**D0→D1 已有首测**（R31-lane E：Δ≤8.7% rel，MoE-on 录制）——P1 增量=D2/D3 + **统一口径（MoE-off）重录**；②**P1 录制需扩字段**（raw obs / OD·LD 原始量 / 位姿，供 D2 oracle 几何与 D3 重编码），并建议**两层法**：先 val 协议 D2/D3（数据集现成、零录制），再部署协议 D0–D3；③**A_k（Gate D）与 Lipschitz 复用现有 R4/lane A 数据**（零新录制）。
- **开工（本轮；全只读、零训练、零生产改动）**：Level 1 四 lane 已派发——A：P1 录制器扩展 + D0–D3 replay；B：P2 有限差分 Jacobian + Action Swap/zero/opposite；C：P3 Ego 状态充分性（real/WM/frozen latent 解码）；D：P4 Policy Lipschitz + A_k 放大因子。产物 `runs/round32/*`、脚本 `.slim/deepwork/p1_v2/round32/*`。

### 5.38 Round-32 Level-1 结果（四 lane 全收口；全只读）：主瓶颈=D2→D3 latent transition；decision 草案=GO_ACTION_CONDITIONING

- **P1（D0–D3，50 ep MoE-off）**：Gate E **D0→D1 2.9% / D1→D2 0.0% / D2→D3 97.0%**（逐 k 0.925–1.004）⇒ pose/执行与图几何非主因；D0 ego 0.191→0.298（static 0.358→0.390，k5 仅 1.3× 优于"不变"）；overshoot 0.469→1.660；D0 逐位复现 R4。
- **P2（Action Fidelity）**：ds 通道 FD 敏感度 26–50× 弱于 dθ/κ；Swap matched **top-1=10.2% / margin≈0** ⇒ **动作条件化失效**（Gate B 未过）。
- **P3（Ego 充分性）**：real latent 充分（lane ld_slots 0.90/0.93）⇒ 排除 state-rep 支路；WM 保留 ego ✓（0.91–1.10）/ lane ✗（0.61/0.77）；附带：WM 的 od/ld latent 泄漏 ego 运动量。
- **P4（Gate D）**：A_k ×2.1（温和；分子 ×3.1、分母 +44%）；Lipschitz 中位 0.73–0.78、尾部 6× ⇒ 接口次要。
- **决策草案（`docs/round32_decision.md` / `runs/round32/decision.md`）**：**`GO_ACTION_CONDITIONING`（主）**——下一步=零初始化动作残差 canary（待批）；未选支路各附排除理由（protocol 2.9%/graph 0.0%/state-rep real 充分/interface 温和/Level-3 未到/BLOCKED 不成立）。
- **产物**：`docs/round32_audit_summary.md`、`docs/round32_decision.md`、`runs/round32/{p1_dchain,p2_action_fd,p3_ego_sufficiency,p4_lipschitz}.*`、`runs/p1_v2/round32/r32_K1e2_moeoff`（50-ep 录制）；零训练、零生产改动。

### 5.39 round-32 答复（GPT `cde2e51`，515 行）：方向受理 + 措辞精确化 + 分臂 canary 批准；P3b 只读开工、canary 待令

- **受理**：GPT 接受 Level-1 与 GO_ACTION_CONDITIONING 主方向，并精确化：**"主瓶颈=latent-state evolution block；其中 action conditioning 是首个获得直接失败证据的子机制"**（D2→D3 含全部未解释差异，不能严格单一归因）；D1→D2=0 因位姿通道不敏感而证据有限 ⇒ **动态图支路不永久关闭**（action 修好后再看几何通道是否可消费）。
- **采纳的补充（3 条）**：①**条件化增益指标**：ΔR²_WM=R²_WM−R²_frozen（frozen 对照已 ~0.87–1.08 ⇒ "保留"可能只是状态惯性/自相关）；ΔR²_action=correct−shuffled；②**语义串扰防线**：action 不得无差别广播进 OD/LD transition（P3 已见 ego 泄漏）——Gate 2 增"lane 保留不降 >5pp、OD/LD ego 泄漏不增 >10%"；③**分臂设计**：AC0 / AC1（ego-only action conditioning，零初始化输出+α=0）/ P-only（仅 ego 物理锚）/ AC1-P；数据 C-A（expert-only）先行、C-B（60/40）后置。
- **我方执行注记（3 条）**：①**canary 数据域**：K1e2 为 semantic 模式 ⇒ C-A 训练数据须含语义键（如 `semantic_v1_3d3`）；C-B 的 policy transition 需 Stage-A 兼容 schema（录制扩字段）；②**接线点**：`st_gnn.forward` 新增 action 入参（`net/model.py` rollout 与 `stages.py::_wm_predictions` 两处调用点）+ 零初始化残差 + α 标量；**α=0 前向逐位不变**（Gate 0）；③**§五 vs §七 不一致**：§五把 ego physical decoder 列入 AC1，§七要求其独立成臂——按 §七 因子设计执行。
- **处置**：**P3b（只读）已开工**（ΔR²_WM 精确值 + ΔR²_action shuffled 扩展 + 泄漏指标形式化）；**canary 已获 CG 批准（最小可逆版 + 10 条锁定条件）**，但涉及**首批生产代码改动 + 4 臂小训练**——按"不要实施任何改动"暂持，等一句"开工 canary"即启动。

### 5.40 round-32 canary 执行（用户令）：基建 + 四臂训练 + Gate 0–4 全测；**fidelity 级有效、部署门未过**（总报告 `docs/round32_canary_report.md`）

- **基建（生产代码 +103/−3；默认关/惰性/零行为变化）**：`st_gnn` ego 动作残差（末层零初始化 + α 标量，惰性构造）+ `ego_phys` 惰性读出 + 两处接线（`_rollout` 用 a_{k-1}、`_wm_predictions` 用 raw_action）；**Gate 0 七项全 PASS**（默认=HEAD 键集 168、旧 ckpt strict 0/0/0、关闭/α=0 逐位、RNG 不变、梯度只流新参数、OD/LD 不广播、baseline↔semantic z-path 等价）+ 966 tests passed。
- **训练（C-A expert-only v8；K1e2 血统 + 参数注入）**：AC1 / P-only / AC1-P 各 22,532 步/4ep（~35 min/臂）；**P3b 收口**（ΔR²_WM state 仅 +0.025/+0.042 ⇒ "保留"≈惯性；**ΔR²_action +0.002~0.028 ⇒ latent 演化几乎不依赖动作**；泄漏 od_pool +0.32 / ld_pool +0.90）。
- **Gate 1/2（独立池 N=200/全量）**：top-1 0.101→**AC1 0.284**（0.35 未达）/ top-3 0.635→**0.917 ✓**；FD 比值 38.7→**7.34（AC1）/ 2.74（AC1-P）✓**；ds-progress 单调 0.743→0.202（AC1）/ **0.946（AC1-P）✓**；ΔR²_WM 全正（AC1-P 最佳 +0.023/+0.029）；保留均改善；泄漏聚合 ✓ 但 od_pool prog 单通道 +41~66% ✗；P-only 动力学 ≡ base（机制：phys 损失经 z_ego_next 塑形 action 分支，在 AC1-P 中体现）。
- **Gate 3（50-ep 部署录制）全 FAIL**：E_if 1.734→2.097（AC1 **+20.9%**）/ 1.795（AC1-P +3.5%）；overshoot@h5 1.660→**1.933（+16.5%）**/ 1.695（+2.1%）⇒ **AC1 反向恶化**（动作敏感性放大 rollout 自反馈漂移）；AC1-P 近中性、E_real −8.9% ✓。
- **Gate 4（calib50）**：**AC1 0.600✓ / 0.120✓ / 0.180✗ / tmo5**（succ 首破 0.58，+4pp；**coll 0.08→0.18 翻倍**）；P-only ≡ base；AC1-P 0.540✗ / 0.160✗ / 0.020✓ / tmo14✗。
- **结论**：结构方向在 fidelity 级被验证（条件化可学习；FD/单调/保留改善），但**部署门未过**（Gate 3 全 FAIL、Gate 4 无臂全过）⇒ **不自动进入 C-B/60-40**（预注册纪律）。
- **裁决建议（5 项，待用户/CG）**：①AC1 碰撞/overshoot 归因（逐 episode 路型/速度 + plan_ds 剖面，只读）；②分支正则/限幅对照（ds 残差幅值约束 / α 上限 / 仅 dθ 或 ds 冻结）；③C-B 前提重议（未过门不自动）；④AC2（相对场景条件化）后置；⑤回滚选项零风险（默认关即逐位不变）。
- **产物**：`docs/round32_canary_report.md`、`runs/round32/canary/*`（gate0_compat / eval_* / gate3_* / gate4_* / record_* / monitor）；脚本 `.slim/deepwork/p1_v2/round32/{canary_train,canary_eval,canary_compat_check,c4_gate3_analyze,p3b_conditional_gain}.py` + 跑批 sh；生产代码已随本轮提交（默认关）。

### 5.41 round-33 答复受理（GPT `c3bf69f`，905 行）：P0"双零死锁"= 已按设计冷启动规避（审计闭合）；33A/B/C 全采纳 + 3 条方法论补充

- **P0 核验（只读审计已完成 = 33A-A1 主体）**：GPT 推理本身正确（α=0 ∧ 末层零初始化 ⇒ 分支梯度恒 0），但**训练从未使用该默认态**——训练器在载入 init 后对缺 key 的 α **冷启动置 1**（`canary_train.py:440-443`；`--alpha-init` 默认 1.0；docstring 已写明）。证据链：①日志首行 `alpha=1.0000 trainable=5 frozen=166`；②步级梯度 step1 `new_grad=2/5`（与"末层 W2/b2 立即获梯度、前置层与 α 被零初始化阻断"机制精确一致）→ step2 `5/5` → α 自 step2 起更新（1.0007→1.1267）；③ckpt 终值 α=1.12667、末层 ‖W2‖=13.59/‖b2‖=0.77（非零）；④**ckpt vs M0 init：168 共享键 max|Δ|=0（逐位）** ⇒ 不存在"其他参数被意外解冻"；⑤optimizer=Adam(allowlist 校验集，`RuntimeError` 守卫)；⑥provenance：AC1/p_only 跑在惰性化前版本、ac1p 跑在惰性版（78ec71e），两版数值等价（Gate 0 已证）。**⇒ 本轮归因有效**；GPT 列的可能性 #1 即事实，且已在训练器文档化——但**未写进 R32 报告（文档缺口，已补）**；同意 v2 起：ckpt metadata（enable/feature_version/alpha_mode）+ 载入 fail-fast + optimizer id 级断言。
- **采纳（无异议）**：不晋级 C-B、AC1/AC1-P 不作新 baseline；顺序 **33A（梯度/碰撞/特征）→ 33B（analytic ego-warp oracle W0–W4）→ 33C（修正版 canary）**；Gate A–E（C 门"先做到不放大，再谈改善"）；v2 设计：**方案一（末层零初始化 + α≡1 固定）**、3 维特征（删重复 speed-intent）、`valid_kappa` mask + ds 分层报告、ego_phys 增 `delta_speed`、三臂 B0/B1/B2、不放大 physical 权重；泄漏四问（horizon/slot/同车道前车/碰撞相关）。
- **补充/修正（3 条）**：
  1. **ds-progress 单调率口径**：现探针在"模型自身 rollout latents"上拟合、施加于"常量动作分支 latents"——AC1 的 latent 强依赖动作 ⇒ off-manifold，0.202 不宜直接读作"因果方向错乱"；v2 建议 per-arm 在分支分布内拟合探针（现口径保留作对照）。
  2. **warp 探针可解释性**：W1–W3 由**冻结 WM** 消费新坐标系 = off-distribution，负结果只能判"不重训无效"、不能判"方向错误"；**W4（oracle）是唯一信息充分项**；建议增设 **W4a=oracle 位置但保持 t0 帧**（分离"锚精度 vs 坐标系"两因素）。
  3. **碰撞统计功效**：n=50、4→9 例，配对 McNemar 双侧 p≈0.06 ⇒ 方向红旗成立、Gate D 不放松，但需配对表（A2）+ 次级证据（min-TTC/急刹）与一次 50-ep 复跑佐证。
- **我方 33A 增项（A5）**：**plan-shift vs WM-response 分解**——同 recorded 状态下 (i) 策略自身 plan（action_mu）base vs canary 是否移动；(ii) 同一 plan 下 WM 预测 progress 是否膨胀 ⇒ 定位 overshoot 是"WM 放大"还是"策略被带偏"。
- **度量噪声地板**：Gate C"恶化 >5% 停止"需先标定接口指标 run-to-run 噪声（建议 base 同协议重复录制一次）。
- **处置**：33A-A1 已闭合（证据见上，`runs/round33/a1_ckpt_audit.json`）；33A 其余项与 33B 为只读 probe——**待 CG 确认即开工**。

### 5.42 round-33 执行完成（GPT `f0ebfae` 批准后）：33A 三问闭合 + 33B warp **NO-GO**；decision=GO_EGO_ACTION_CONDITIONING_V2（33C 待令）

- **33A-Q1 碰撞**（双证据链：录制启发式 + 同协议复跑读 MetaDrive 接触真值，13/13 repro=exact）：**最大簇=收费站静止障碍 6/13**（撞岗亭 3——非车辆、不在 OD/LD、仅经 others 走廊段；撞 OD 可见静止/排队车 3）；侧碰/并线 4/13；低速追尾 3/13；**交叉口冲突 0**。AC1 新增 5 例不集中单类（侧 2/静 2/追 1）但 2/5 在收费站岗亭；末步仍在给油（plan ds 2.51–3.43 m/0.5s）。
- **33A-Q2 overshoot**（逐样本恒等式 O=A+B+C 残差=0）：h5 ΔO=+0.2736m（+16.5%）——**(ii) 规划/WM 接口项 B=plan−mu 主导（Δ=+0.3226，share +117.9% [108,141]）**；(i) A 反向（−0.0552，−20.2%；mu +0.050 但 expert +0.117）；(iii) C 可忽略（+0.0062，+2.3%；无晚段累积，Spearman NS）。h1–5 聚合 B +115.1%；速度混淆不解释。⇒ **WM 预测改变后规划层放大 ds**（与 E_if +20.9% / E_real +2.4% 互证）。
- **33A-Q3 噪声**：录制确定性（`gate4_ac1` vs `record_ac1` 逐位一致）⇒ 重复噪声=0；episode 配对 bootstrap（B=10000）半宽 6.06/6.12/8.90pp ≪ 恶化 20.9/17.2/16.5% ⇒ **恶化真实**。
- **33B warp（W0/W1/W2，base 模型，各 50 ep）**：**NO-GO**（严格 0/4；宽松 W1 2/4、W2 0/4）。W1：E_if −2.0%、E_action −1.7%、overshoot −2.4%、lane +3.9%（split 不稳）、OD 泄漏增加；W2（纯重表达）≈0。机制：对冻结 WM，ego 帧重表达本身几乎无效，W1 小效应来自移除 OD 恒速先验；零-shot 效应比 canary 恶化小一个数量级。自证：off 逐位（27 键 max|Δ|=0；w0 重录 vs base 50ep×40 键逐位一致）；warp 代码默认关保留。
- **决策（按树）**：**GO_EGO_ACTION_CONDITIONING_V2**（不自动进入训练）。
- **33C 建议（待 CG）**：**B0/B1 两臂**（B2=B1+warp 因 NO-GO 删）；B1 = 特征 [ds,dθ,κ] + 隐藏层正常初始化 + 输出层零初始化 + **固定增益 1（删可学习 α）** + ego 物理监督 {delta_progress, delta_speed, delta_heading}；硬门沿用 GPT 列表；训练 ~35 min/臂；评测建议加**收费站/静止障碍分层读数**（不新增 probe）。
- **产物**：`docs/round33_report.md`、`runs/round33/*`（a2/a5/a6/b_warp*）、脚本 `.slim/deepwork/p1_v2/round33/*`。

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`、`docs/recovery_v2_contract.md`、**`docs/round30_audit.md`（R30 全审）**、**`docs/stage_a_config_consumption.md`（键级契约审计）**
- **报告**：`docs/p1_0_report.md`、`docs/p1_2_diagnosis.md`、`docs/r21_lane_semantic_contract.md`、`docs/r21_target_lane_selection.md`、`docs/semantic_v1_dataset_card.md`、`docs/r21_3a_semantic_wiring.md`、`docs/recovery_v1_dataset_card.md`、**`docs/round32_canary_report.md`（canary Gate 0–4 全测）**、**`docs/round33_report.md`（33A 三问 + 33B warp NO-GO）**
- **数据**：`datasets/recovery_v1|v2/`、`datasets/semantic_v1_{pilot,independent,3d3}/`（9,515 / 8,953 / 53,765）
- **产物**：`runs/p1_v2/round12a|15a|17|18a|20|21|22|23|24|25|26|27|28|29/`、**`runs/p1_v2/round30/`（matrix×8 / r4_K1e2 / wm_action_probe_M0 / grad_ledger / wm_tf_check×2）**、`runs/p1_v2/`（probe 家族）、**`runs/round32/canary/`（gate0_compat / eval_* / gate3_* / gate4_* / record_* / monitor）**、**`runs/round33/a1_ckpt_audit.json`（P0 审计）**、**`runs/round33/`（a2_collision / a5_overshoot_decomp / a6_noise / b_warp / b_warp_compat）**、`runs/p1_0|p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`、`.slim/deepwork/p1_v2/`（round12a→32 脚本 + logs；R30 队列=`round30/{journal,monitor,run_r30_stage1,run_r30_stage2,run_r30_r4}.sh`；canary 队列=`round32/{canary_train,canary_eval,canary_compat_check,c4_gate3_analyze,p3b_conditional_gain}.py` + `run_r32_{canary,eval,gate34}.sh`；R33 队列=`round33/{a2_collision_attribution,a5_overshoot_decomp,a6_noise,b_warp_compat,b_warp_record,b_warp_analyze}.py`）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮；`61bf7ab` 起=Reality Audit 全栈提案、`acf1b8f`=终审、`c3bf69f`=round-33 答复）
- **关键提交**：逐轮全文=本文件 git log；近期锚点：`423ae7c`（§5.25）、`8933049`+`c5a44df`（R28/§5.26）、`62f4202`+`97e35c6`（round-29/§5.27）、`d160582`+`711bc47`（R29A/§5.28）、`15ec4fa`（第十四轮）、`61bf7ab`（CG5.6 总审输入）、`acf1b8f`（终审输入）。

---

## 7. 存档说明（历次精简 / 去哪找）

- **第二十八轮（本版；392 → 404 行）**：新增 §5.42（R33 执行完成：33A 三问 + 33B NO-GO + 决策）；§0/§7 刷新；上版 392 行 = `git show 0188218:docs/deepseekv4p1_argue.md`。
- **第二十七轮（377 → 392 行）**：新增 §5.41（round-33 答复受理 + P0 审计闭合 + 3 条方法论补充）；§0/§7 刷新；上版 377 行 = `git show 7015cc8:docs/deepseekv4p1_argue.md`。
- **第二十六轮（363 → 377 行）**：新增 §5.40（canary 执行 + Gate 0–4 全测 + 结论/建议）；§0/§7 刷新；上版 363 行 = `git show 557527b:docs/deepseekv4p1_argue.md`。
- **第二十五轮（354 → 363 行）**：新增 §5.39（R32 答复受理 + P3b 开工/canary 待令）；§0/§7 刷新；上版 354 行 = `git show 57d2391:docs/deepseekv4p1_argue.md`。
- **历次精简留档**：第二十四轮（343→354，`57d2391`）；第二十三轮（334→343，`3d14f9d`）；第二十二轮（319→334，`cd6f1f6`）；第二十一轮（308→319，`87253cf`）；第二十轮（298→308，`440790b`）；第十九轮（281→298，`7beb7e9`）；第十八轮（264→281，`b8bb326`/`39f0569`/`278a187`）；第十七轮（299→264，`5ed40e4`）；第十六轮（276→299，未单独提交、并入第十七轮）；第十五轮（255→276，`fdc6d2d`）；第十三轮（随 R28/R29 新增；`c5a44df`/`711bc47`）；第十二轮（280→270，`e5ae48e`）；第十一轮（269→261，`6e2e2cd`）；第十轮（267→252，`52ce959`）；第九轮（269→248，`48bb8be`）；第八轮（300→252，`ee2651b`）；第七轮（305→283，`a1986f3`）；第六轮（342→261，`e638b76`）；第五轮（355→310，`96df7cb`）；第四轮（299→282，`58ac4c3`）；第三轮（`74f522f`）；第二轮（262→~210，`249c51c`）；第一轮（1114→166，`7b83730`）。
- **压缩时间线**：
  - 2026-10-09/10：round-1/2 收拢；P0 三件套（兼容设计/矩阵/归因，§2.1）→ 排摸收官（§2.2）→ round-3–8 与 P1 锁板（§2.4）；
  - 2026-10-10→11：P1 执行（§2.3）→ round-9/10 定板 → P1-v2 三步 + 旁线 D/E（§3）→ **round-12A NO-GO（§4）**；
  - 2026-10-11 深夜–10-12：round-15→20（§5.1–5.10）：15A 全批次 → **A3 判负** → 18A probe+审计 → M0-M2 消融 → **停 LD 架构、转数据侧**；
  - 2026-10-12 深夜–10-13 凌晨：round-21/22 → **R21-0..2 门全过**（§5.11–5.14）；凌晨：round-23 → **3A..3C（首现强正向）**（§5.15–5.16）；
  - 2026-10-13 上午：round-24 → 3D-1/2 门过、**3D-3 未过**（§5.17–5.18）；中午：round-25 → **首超 legacy、B 行 bug 修复**（§5.19–5.20）；
  - 2026-10-13 下午：round-26 **Q1 全面生效**（§5.21–5.22）→ round-27 R1 修饱和但 **tmo +26pp**（§5.23–5.24）；傍晚：round-28 **28A 归因 + K1 停止**（§5.25–5.26）；
  - 2026-10-13 晚：round-29 → 29A T 族投影（§5.27–5.28）；
  - 2026-10-13 夜：round-30——开场核验（§5.29）→ 终审受理开工（§5.30）→ **第一轮 Reality Audit 完成、新主问题=部署链一致性（§5.31）；待裁**。
  - 2026-10-14 凌晨：GPT round-31 宣言（`1c1d91b`）到达——全范式流水线 + 今晚 Stage-A 队列；**受理审阅（§5.32：M1–M5 + 排队方案）；零实施、待裁**。
  - 2026-10-14 凌晨·二：GPT 答复 2（`5872f57`）——70/20/10（接受 §5.32 反驳 + 补 Stage B 风险 + Q1–Q3 三问 + 冻结大规模采集）；**受理审阅（§5.33：Q1/Q3 收账、Q2 规格、supersede 澄清）；零实施、待裁**。
  - 2026-10-14 凌晨·三：GPT 答复 3（`f4faec8`）——**收敛：基本站我方**；批准 Q2 审计 + WM fidelity + Stage C contract（暂停一切训练）；**§5.34 受理 + R31 只读审计包开工**。
  - 2026-10-14 凌晨·五：R31 审计包**五 lane 全收口**（Q2-a/b/c、mirror、contract、M2、MoE 敏感性）——**§5.35：Q2 判决草案（中等风险/无冲突）+ 8–10× 状态分布放大 + 条件化弱 + 契约清单；下一步候选 5 项待裁**。
  - 2026-10-14 凌晨·六：GPT 长答复（`910b49b`，1641 行）——**全面接受 R31** + WM 五结构弱点（核验成立）+ Round-32 十探针 + 联合训练 Phase 0–4 + MoE-off 基线 0.56；**§5.36 受理 + 5 条修正待裁；零实施**。
  - 2026-10-14 凌晨·七：GPT 裁决（`d86e91a`，677 行）——**5 修正全收 + 批准三级决策树**；**§5.37 受理 + Level 1 开工（4 lane 只读：D0–D3/Jacobian FD/Ego 充分性/Lipschitz）**。
  - 2026-10-14 上午：**Round-32 Level-1 四 lane 全收口**——**§5.38：Gate E 主瓶颈 D2→D3=97.0%；Gate B 未过（动作条件化失效）；decision 草案=GO_ACTION_CONDITIONING（待裁）**。
  - 2026-10-14 上午·二：GPT 答复（`cde2e51`，515 行）——**接受主方向 + 批准最小可逆 canary（AC0/AC1/P-only/AC1-P；Gate 0–4）**；措辞精确化（latent-state evolution block）；**§5.39：P3b 只读开工、canary 待令**。
  - 2026-10-14 下午：**canary 执行完毕**（用户令）——基建 C1（Gate 0 七项 PASS、966 tests）+ 四臂训练（C-A expert-only）+ Gate 0–4 全测；**§5.40：fidelity 级有效（top-3 0.917✓/FD 2.74×✓/单调 0.946✓/ΔR² 全正）、部署门未过（Gate 3 全 FAIL；AC1 succ 0.600✓ 但 coll 翻倍）⇒ 不自动进 C-B；五条建议待裁**。
  - 2026-10-14 下午·二：GPT round-33 答复（`c3bf69f`，905 行）——canary 不晋级 + 33A/B/C 计划 + Gate A–E + P0"双零死锁"质疑；**§5.41：我方只读审计闭合 P0（冷启动 α=1 按设计；ckpt vs M0 共享键逐位 0 差异；step1 2/5→step2 5/5）+ 全采纳 + 3 条方法论补充 + A5 增项；待 CG 确认开工**。
  - 2026-10-14 下午·三：GPT 修订答复（`f0ebfae`，97 行）——反驳接受、初始化结案、批准 33A/B/C 开工；**§5.42：执行完成——33A 三问闭合（碰撞=收费站静止障碍簇 6/13；overshoot=规划层 +117.9%；噪声<恶化）+ 33B warp NO-GO（0/4）⇒ decision=GO_EGO_ACTION_CONDITIONING_V2；33C 建议 B0/B1 + 硬门；待 CG 确认后训练**。
