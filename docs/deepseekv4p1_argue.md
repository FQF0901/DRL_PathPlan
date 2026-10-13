# deepseek-v4.1 argue / 主文档（R31 宣言（`1c1d91b`）受理审阅——5 条修正/前置待裁（§5.32）；本轮零实施）

> **目标（双方一致）**：大幅提升闭环 success，争取**超越 IDM（0.756）**。
> **状态（2026-10-14 凌晨）**：GPT round-31 宣言（`1c1d91b`，948 行）到达：全范式流水线 A→B0..B4→C0..C4 + 今晚 Stage-A-only 队列（Task1–6：契约/复核/pilot 50–100k/五探针/A4 canary + 五条自动停止规则 + decision.md 五枚举）。**§5.32=受理审阅：整体接受 + 5 条修正/前置（M1–M5）+ 排队方案；按指令本轮零实施（不动代码/不采数据），待裁。** R30 结果（§5.31）：矩阵 3 系统（S3 严格劣）、R4 接口主瓶颈（E_if/E_deploy 1.54→1.22）、TF 消歧（WM 固有 0.02–0.04 vs 部署链 0.20–0.32，8–10×）⇒ 主问题=部署链一致性。
> **本版说明**：第十八轮（264 → 280 行）——新增 §5.32（R31 宣言受理审阅：M1–M5 + 排队方案）；§0/§7 刷新；上版 264 行 = `git show 7394a91:docs/deepseekv4p1_argue.md`；历版全文=本文件 git log。

---

## 0. 一页纸（当前局面）

- **目标锚**：IDM 0.756（上限审计：**最高可执行 d2=0.706**）；P1 起点 pri512@e020（clean150 **0.3400/0.4933/0.0333**；eval500 0.332）；G2 追平 ≥0.74、G3 超越 ≥0.80（§1）。
- **主线结果（压缩；详见 §2–§5）**：条件轨 1b `no_increment` → 12A NO-GO（关闭）；A3 确定性尾全 0（**动作可持性**）；LD 结构缺陷+响应 ~50× 弱（M2=接线缺口）→ R21 语义重建（3C 首现强正向）→ 3D-3 未过门 → **R25 首现闭环超 legacy（0.400/+7.3pp、off −20.7pp）** → **R26 Q1**（Δp99 1.017、calib50 0.50 vs 0.40）→ **R27 R1**（sat ✓ 但 tmo +26pp ✗）→ **R28 28A 归因 ✅**（F3 速度参考主轴；F1 κ-ds 耦合/F4 LQR 尾段）+ **K1**（teacher +1.1%、κ MAE −71.6%、sat −12.5pp、calib 0.52/0.12；ds≡legacy 未消除 tmo ⇒ 停止）→ **R29A**（代码修复 ✅；T1/T3 机制性失败、T2 正向未过门；Δκ 瓶颈=首步钉死 ⇒ 停止）。关键锚：legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58（calib50）；IDM 0.756。
- **round-30 结果（§5.31）**：Reality Audit 第一轮（R30-0..6b、零训练、零生产改动）——矩阵 3 系统（S3 训练栈严格劣：0.02/0.00）、R4 接口主瓶颈、**TF 消歧：WM 固有 0.02–0.04 vs 部署链 0.20–0.32（8–10×）⇒ 主问题=部署链一致性**（overshoot drift）；账本三布尔全实证；契约 v1 + 配置审计落档；**round-31 裁决单在 §5.31 尾**。
- **round-31 宣言（本版；§5.32）**：GPT 全范式重排（A[WM]→B0–B4[IL/DAgger/MoE]→C0–C4[RL]）+ 今晚 Stage-A-only 队列（Task1–6）；**受理=整体接受 + 5 条修正/前置（M1 A4 前提预注册；M2 新增 Task4b 三分分解；M3 协议支路入决策树；M4 raw/composed 数据口径；M5 门可操作化）**；零实施、待裁。
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

---

## 6. 文档与产物索引

- **决策/契约**：`docs/p1_v2_decision.md`、`docs/p1_implementation_contract.md`（v1）、`docs/p1_probe_spec.md`（v1.1）、`docs/p0_eval_compat_design.md`、`docs/recovery_v2_contract.md`、**`docs/round30_audit.md`（R30 全审）**、**`docs/stage_a_config_consumption.md`（键级契约审计）**
- **报告**：`docs/p1_0_report.md`、`docs/p1_2_diagnosis.md`、`docs/r21_lane_semantic_contract.md`、`docs/r21_target_lane_selection.md`、`docs/semantic_v1_dataset_card.md`、`docs/r21_3a_semantic_wiring.md`、`docs/recovery_v1_dataset_card.md`
- **数据**：`datasets/recovery_v1|v2/`、`datasets/semantic_v1_{pilot,independent,3d3}/`（9,515 / 8,953 / 53,765）
- **产物**：`runs/p1_v2/round12a|15a|17|18a|20|21|22|23|24|25|26|27|28|29/`、**`runs/p1_v2/round30/`（matrix×8 / r4_K1e2 / wm_action_probe_M0 / grad_ledger / wm_tf_check×2）**、`runs/p1_v2/`（probe 家族）、`runs/p1_0|p1_2/`、`runs/eval/`、`runs/forensics/`
- **流水/脚本**：`.slim/deepwork/p1_0/journal.md`、`.slim/deepwork/p1_v2/`（round12a→30 脚本 + logs；R30 队列=`round30/{journal,monitor,run_r30_stage1,run_r30_stage2,run_r30_r4}.sh`）
- **回合文本**：`docs/chatgpt5p6sol_review.md`（ChatGPT 各轮；`61bf7ab` 起=Reality Audit 全栈提案、`acf1b8f`=终审）
- **关键提交**：逐轮全文=本文件 git log；近期锚点：`423ae7c`（§5.25）、`8933049`+`c5a44df`（R28/§5.26）、`62f4202`+`97e35c6`（round-29/§5.27）、`d160582`+`711bc47`（R29A/§5.28）、`15ec4fa`（第十四轮）、`61bf7ab`（CG5.6 总审输入）、`acf1b8f`（终审输入）。

---

## 7. 存档说明（历次精简 / 去哪找）

- **第十八轮（本版；264 → 280 行）**：新增 §5.32（R31 宣言受理审阅：M1–M5 + 排队方案）；§0/§7 刷新；上版 264 行 = `git show 7394a91:docs/deepseekv4p1_argue.md`。
- **历次精简留档**：第十七轮（299→264，`5ed40e4`）；第十六轮（276→299，未单独提交、并入第十七轮）；第十五轮（255→276，`fdc6d2d`）；第十三轮（随 R28/R29 新增；`c5a44df`/`711bc47`）；第十二轮（280→270，`e5ae48e`）；第十一轮（269→261，`6e2e2cd`）；第十轮（267→252，`52ce959`）；第九轮（269→248，`48bb8be`）；第八轮（300→252，`ee2651b`）；第七轮（305→283，`a1986f3`）；第六轮（342→261，`e638b76`）；第五轮（355→310，`96df7cb`）；第四轮（299→282，`58ac4c3`）；第三轮（`74f522f`）；第二轮（262→~210，`249c51c`）；第一轮（1114→166，`7b83730`）。
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
