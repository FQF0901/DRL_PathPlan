# Round-37 报告（阶段 1）：结构化 Policy 的闭环验证与三层根因定位

- 日期：2026-10-15 ｜ 分支：`Planner_RL` ｜ 范围：R37 阶段 1（坐标层集成 + C0/C1/C2 闭环分诊 + 零训练修复探针）
- 证据口径：全部闭环读数 = **calib50 开发集**（`runs/p1_v2/round25/specs_calibration50.json`，n=50，同协议同 seed 集）
- 约束：**零训练**（未获 CG 放行前不训练）；`.slim/`、`runs/` 不入库；生产改动默认关 + 逐位验证；`net/` 唯一改动 = M1（已入库 `0d63bbc`，966 tests + 契约门 6/6 + off 逐位）

---

## 0. 结论摘要（TL;DR）

1. **新动作表示（结构化头）单独贡献为负**：C1 0.340 vs 旧头 C0 0.580（−24pp，timeout 20/50）⇒ 不晋级。但失败形态是"**欠动/爬行**"而非跟踪失效——根因不在坐标层，也不在 WM。
2. **三层根因（互为放大）**：
   - **结构层**：纵向参数化 `ds = DS_HIGH·σ(ds_raw + logit(clamp(v·dt)/DS_HIGH))`，零残差=**保速先验**；LQR 参考速度 = plan 弦长/dt ⇒ plan 几何即速度指令 ⇒ 掉速不可恢复（自锁 ≈1.44 m/s）。
   - **接口层**：`path_dynamic_ttc` 评测侧缺键→0（训练语义"无冲突"=30.6 s）⇒ "立即碰撞"误读，占装配错配效应 **77%**；修（中性填充）后**零训练 +8pp**（0.340→0.420）。
   - **横向层**：加速后 plan 曲率质量/抖动不足（|κ|p90 0.023→0.081、抖动 ×4.7、|err_ψ| 0.017→0.241）⇒ 高速下蛇形/越界（A1 off-road 0.44）。
3. **零训练通道穷尽**：F3（提 v_ref）✗（coll 0.48）｜ A1–A3（改先验）✗（off 0.24–0.44）｜ C2（warp）~中性（coll −0.06、tmo +0.08）｜ **接口修复 ✅（唯一有效，+8pp，非完整解）** ｜ 真值回填（语义已核验逐位一致）在爬行解除上更强但横向跟不上（off 0.18）。
4. **P0 定案（提案）**：训练层——**B = 蒸馏 IDM+PP 规则专家**（`tools/dagger_collect.py` 就绪：学生状态 roll-in + 规则专家 (ds,dθ) 标签 + train-spec 硬隔离），**且训练必须在 eval 装配（obs 指纹统一）下进行**；C3（WM/想象步动力学）排其后。
5. **待 CG 裁决**：见 §9（接口修复固化形态 / P0-B 放行 / C3 时机 / 口径登记 / 新增考核项）。
6. **M8（P0-B 数据链试点）**：验证"IDM 标签是否含加速信号 + H2/I1 抽取就绪度"，在飞。

---

## 1. 本轮范围、协议与约束

### 1.1 目标（GPT `66497a2` Round-37 方案）
统一闭环基准四臂：**C0**（旧 Policy+旧 WM 坐标）/ **C1**（结构化 Policy+旧 WM）/ **C2**（+analytic ego warp）/ **C3**（+重训 residual WM）；同数据/同调用/horizon/LQR/episodes；Calib50→Clean150→Eval500 三级；**字典序 + 安全硬门**（Primary=success；Hard=collision·off-road；Secondary=timeout·rc），禁模糊加权总分。

### 1.2 执行序（我方提案，GPT `effea53` 接受）
**37A 工程接入 → 37B C0/C1/C2 闭环分诊 → 37C WM 适配**；并采纳 GPT 的关键修正：**C2 是"旧 WM 上的零训练 warp 诊断臂"，不是 C3 的开关**（C3 由 5 条坐标/接口证据联合触发）。

### 1.3 同协议命令行（所有臂逐字相同，仅加臂专属 flag）
```
eval_a3 --ld-mode semantic --rollout-semantic static_role_only --moe-off \
  --tracker lqr --tail chain --spec runs/p1_v2/round25/specs_calibration50.json \
  --workers 6 --device cuda --max-steps 1000
```
- C0：`--ckpt runs/p1_v2/round28/K1/ckpt_epoch002.pt --path-head off --lane-anchor off`
- C1：`--canary-ckpt runs/round36/j2/t1_keep_res/ckpt_epoch001.pt --path-head structured --lane-anchor inject`
- C2：C1 + `--ego-warp-mode analytic`；F3：C1 + `--cf-family F3 --cf-ref-speed-scale 2.0`；A1–A3：C1 + `--ds-init-override 4.16|3.0` / `--ds-bounds 2.5,6.25`；fill/true：C1 + `--long-exog-fill neutral|true`

### 1.4 口径登记（重要）
- **不用 kappa-compose**（L-C caveat）：本协议 **C0 = 0.580/0.180/0.120/0.140（succ/off/coll/tmo）、rc 0.872**；历史 K1e2 "0.56/0.14/0.08" 系 kappa-compose 口径，**不可直比**。
- 台账锚点（legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58 / 统一基线 0.56 / **IDM 0.756**）均为 calib50 口径 ⇒ 本报告读数与主线锚点同集可比。
- n=50 ⇒ ±1 集 ≈ ±2pp：只作筛选/方向判定；配对翻转/McNemar 作证据强度；正式声明留 Eval500（3 seeds/paired bootstrap/McNemar）。

---

## 2. 交付与 lane 索引（M1–M8）

| Lane | 目标 | 关键结果 | 产物（`.slim/deepwork/p1_v2/round37/` + `runs/round37/`） | 状态 |
|---|---|---|---|---|
| **M1** | warp 主路径接入（`ego_warp_mode off\|analytic`）+ OD 场景系速度 + 五项 hook + 分项读数 | off 逐位 max\|Δ\|=0（对 HEAD net 包子进程）；966 tests；契约门 6/6；analytic 自检逐位；真实数据复现 OD 语义（spec P90 0.566 vs wiring 2.672） | `m1_*`（gate_a/k1_contract/offline_od_readout/report）；**入库 `0d63bbc`** | ✅ |
| **M2** | C0/C1 同协议闭环分诊（配对） | C0 0.580/0.18/0.12/0.14 vs C1 0.340/0.12/0.14/0.40；succ 翻正 2/翻负 14（p=0.0042）；r1≡r2 逐 episode 逐字一致（默认关中性闭环级复证） | `m2_{run_c0c1,paired_c0c1,net_sha}.py`、`m2_c0c1.{md,json}` | ✅ |
| **M3** | 离线 vs 闭环 ds(v) + 装配错配审计 | 头在训练分布内没坏（bias +0.056/MAE 0.149/r 0.983）；命令语义=保速（v_ref/v≈1.00）；**主凶=`path_dynamic_ttc` 缺键→0（77%）**；语义 20 键非驱动；`eval_on−train` Δds +0.138 | `m3_diag.py`、`m3_report.md`、`runs/round37/m3/m3_diag.json` | ✅ |
| **M4** | C2 臂（analytic warp）+ F3 配速诊断 | C2 0.320/0.12/**0.08**/0.48（coll −0.06、tmo +0.08）；warp 生效（plan k=0/动作逐位同、k≥1 全分化）；**F3 证伪**（v 8.92 但 coll 0.48、a_lat p95 4.76） | `m4_{run,paired}.py`、`m4_report.md`、`runs/round37/m4/{c2_warp_analytic,c1_f3_v2p0,smoke_*}` | ✅ |
| **M5** | A 先验探针（A1/A2/A3） | 意图层生效（`dec__ds_init` 逐位；v_ref 2.63→8.49/6.31/5.37；残差不变式 50/50）但达成层不升（v 3.56/3.01/2.85）；**Hard 全恶化**（off 0.44/0.28/0.24）；A1 steer 饱和 0.861 + 蛇形 | `m5_{run,paired}.py`、`m5_report.md`、`runs/round37/m5/{a1_ds416,a2_ds30,a3_floor25}` | ✅ |
| **M6** | 接口修复探针（long_exog 中性填充） | 中性值最小后悔标定 **TTC=17.7 s**；dyn/static 填 0 与缺键**逐位同**（仅 TTC 被毒化）；**c1_fill 0.420（+8pp、翻正 4/翻负 0）**、tmo 20→18、rc +0.058、v +0.31、crawl 34.9→24.8 s；a1_fill off 0.38 不晋级 | `m6_{calib,run,paired}.py`、`m6_report.md`、`runs/round37/m6/{c1_fill,a1_fill,smoke_*}` | ✅ |
| **M7** | 真值回填探针（三键按训练侧同公式） | 公式核验 A–D 全过（逐位/1e-6/5.7e-14）；c1_true 0.400/0.18/**0.08**/0.34（coll 最优、off 硬退化）；v 2.89、crawl 17.3、斜率 −0.059（三最优）；vs fill：off +5/−2 ⇒ 不晋级；真值分布差异=状态分布（非口径） | `m7_{fill,unit_check,run,paired}.py`、`m7_report.md`、`runs/round37/m7/{c1_true,smoke_*}` | ✅ |
| **M8** | P0-B 数据链试点（IDM 标签语义 + H2/I1 就绪度） | 进行中 | — | 🕐 |

---

## 3. 结果矩阵（calib50 dev，n=50，同协议）

| 臂 | succ | off-road | collision | timeout | rc | 平均速度 | 备注 |
|---|---|---|---|---|---|---|---|
| **C0**（旧 Policy + 旧 WM 坐标） | **0.580** | 0.180 | 0.120 | 0.140 | 0.872 | 3.65 | 本协议基线 |
| **C1**（结构化 + 旧 WM） | 0.340 | 0.120 | 0.140 | 0.400 | 0.741 | 2.47 | 爬行自锁 |
| **C1_fill**（+TTC 中性 17.7 s） | 0.420 | 0.120 | 0.100 | 0.360 | 0.798 | 2.78 | **零训练最优** |
| **C1_true**（+三键真值） | 0.400 | 0.180 ✗ | **0.080** | 0.340 | 0.776 | 2.89 | 横向跟不上 |
| **C2**（+analytic warp） | 0.320 | 0.120 | **0.080** | 0.480 | 0.751 | 2.45 | 机制诊断臂 |
| **A1**（先验 4.16） | 0.460 | 0.440 ✗ | 0.060 | 0.080 | 0.791 | 3.56 | Hard 恶化 |
| **A2**（先验 3.0） | 0.380 | 0.280 ✗ | 0.100 | 0.240 | 0.766 | 3.00 | Hard 恶化 |
| **A3**（地板 2.5） | 0.340 | 0.240 ✗ | 0.060 | 0.360 | 0.765 | 2.85 | Hard 恶化 |
| **F3**（v_ref×2） | 0.300 | 0.240 ✗ | 0.480 ✗ | 0.020 | 0.493 | 8.92 | 证伪外部配速 |

**配对翻转要点**：
- C0→C1：succ **+2/−14**（p=0.0042）、timeout +1/−14（即 14 例由到达变超时）。
- C1→C1_fill：succ **+4/−0**、coll +0/−2、off +2/−2、tmo 20→18。
- C1→C1_true：succ +4/−1、coll +1/−4、**off +4/−1**（净 +3，因此不晋级）。
- C1→C2：succ 0/−1、coll 0/−3（解除 3，无新增）、tmo +4/−0。
- C1→A1：succ +7/−1 但 **off +16/−0**；C1→F3：**coll +17/−0**。

**门对照（Calib50 绝对门：succ ≥0.62 / off ≤0.10 / coll ≤0.08 / tmo ≤0.10）**：**全部臂未过**（最好单项：coll —— M7/C2 达 0.08；succ 最高 C0 0.580）。

---

## 4. 根因分析（三层，互为放大）

### 4.1 结构层：保速先验 + 闭环自锁（纵向）
机制链（6 步）：
1. 结构化头纵向输出 `ds = DS_HIGH·σ(ds_raw + logit(ds_init/DS_HIGH))`，`ds_init = clamp(v_current·dt, [0.72, 6.25])`（`net/path_head.py:994, 1172–1175`；longitudinal 末层零初始化 ⇒ 初始/未充分训练≈保速）。
2. **LQR 参考速度 = plan 弦长/dt**（`env/tracking.py:498–504`；`:537` `accel = speed_gain·(v_ref − v)`）⇒ **plan 几何即速度指令**。
3. ⇒ `v_ref ≈ ds/dt ≈ v` ⇒ 油门 ≈ 0 ⇒ **任意减速不可逆**；地板 0.72 m/步 ⇒ **自锁 ≈1.44 m/s**。
4. 实测（C1 id=7488）：`v[0..9] = 6.0→4.09→2.74→1.86→1.49→1.21…`；ds 1.95→0.55；throttle ≈ 0/−0.05；crawl 中位 52 s。
5. 对照 C0（同 id）：`v[0..9] = 6.0→6.29→…`；plan 递增（2.69→3.73）；v_ref 6.9–7.4；throttle +0.2~0.3 ⇒ **旧头是"速度目标生成器"**。
6. 离线健康性：头在训练分布内 bias +0.056/MAE 0.149/r 0.983（M3）⇒ **坍缩是闭环（反馈）特有**，非拟合误差；历史轮次未暴露=旧头语义不同 + 历史门只测横向。

### 4.2 接口层：long_exog 装配错配（TTC 单键，77%）
- 头消费 `LONG_EXOG_OBS_KEYS` 7 键（`net/path_head.py:229`）；评测 env 不产 `path_*`：margins/nav/d2end 有回退（`lane_semantic` 列 4/5/11、`nav[:,10]`），**static/dynamic_distance + dynamic_ttc 无回退 ⇒ 恒 0**。
- 训练语义：TTC present 100%、均值 30.6 s（"无冲突"）；**评测 0 = "立即碰撞"** ⇒ 头持续下小 ds。
- 受控消融（M3）：单去 TTC 复现 **77%** 的装配效应（mean|Δds| 0.116/0.151），剖面斜率 [2,4) +0.43→0.00、[4,6) +0.24→−0.26（与闭环负斜率同型）。
- 修复三形态：中性（17.7 s，最小后悔）**+8pp**；真值（M7，公式与训练侧**逐位一致**）更快（crawl −7.5 s）但横向跟不上（off 0.18）。
- 溢出观察：同 t0 obs（状态逐位相同）下 plan 首步 50/50 分化、Δds₀ **+1.13 m/步**（闭环灵敏度 ≫ 离线 +0.05~0.25）。

### 4.3 横向层：@speed 横向质量（加速后的绑定约束）
- 侧谱（主控复核 trace，50 ep，|err_y| / |err_ψ| / plan|κ|p90 / κ 抖动）：C0 **0.465 / 0.111 / 0.049 / 0.015**；C1 **0.055 / 0.017 / 0.023 / 0.008**（慢速下"轻松"）；A1 **0.988 / 0.241 / 0.081 / 0.037** ⇒ 速度一起来，plan 曲率量级与抖动都劣于旧头 ⇒ 蛇形/越界。
- M5 机制：A1 94% 步 throttle>0.9 而池化加速度≈0；转向饱和步（81%）accel −0.21 m/s²（**归因为推断**：极限环/轮胎侧向挤压）；steer 饱和 0.861、heading err p50 0.237 rad（C1 0.009）。
- M7 证据：cut-in 例 |err_y| 峰值 0.92 m vs C1 0.13 m、180 步出界；新增 off 含 2 个 tollgate（碰撞→出界）。

### 4.4 为什么以前没暴露 / 为什么离线门看不见
- 以前：C0 语义是速度目标（plan 隐含 v_ref>v）⇒ 无保速自锁；且历史门（越界/收敛）只测横向。
- 离线：训练/评测 obs 装配不一致（TTC），且离线状态来自数据集（有真值输入、非自产状态）⇒ 反馈病理不可见。

---

## 5. 排摸与代码定位（单一真源索引）

| 机制 | 位置 | 说明 |
|---|---|---|
| LQR 参考速度=弦长/dt | `env/tracking.py:478–507, 537–555` | `_ref_speed = chords/dt_sub`；`accel=speed_gain·(v_ref−v)`；遥测 `lqr_ref_speed_mps` 等 |
| 纵向参数化/先验/限幅 | `net/path_head.py:994, 1020–1043, 1172–1175, 316–334, 281–310` | `ds_init=clamp(v·dt,bounds)`；`κ_p95(v)/κ_p99(v)` 分桶；纵向零初始化 |
| long_exog 键 | `net/path_head.py:51–53, 229–236` | 7 键（margins/static/dynamic/ttc/nav/d2end） |
| 评测 obs 装配 | `env/obs/builder.py`（others: nav+speed_limit+signal+static+road_class）、`env/obs/ld.py`（7 维含 speed_limit） | **无 `path_*` 产出** ⇒ 三键缺省 0 |
| OD 槽/TTC 公式 | `env/obs/od.py:75, 216–230, 231–244, 298–312` | 行 `[dx,dy,vx,vy,cosθ,sinθ,L,W,type]`；ttc=dx/(−vx)（dx>0∧vx<−1e-3） |
| 静态距离 | `env/obs/static.py:97–170` | `scan_static_blocker`（gap=s_obj−s_ego−0.5·L_obj） |
| 训练标签口径 | `.slim/.../round36/i1_extract_dataset.py` manifest §LONG_EXOG_COLUMNS；`h2_data_spec.py::label_row` | 弦长/lat_err/head_err/s_anchor/valid30；long_exog 逐列单位/坐标系/mask |
| trainer 损失 | `.slim/.../round36/j1_train.py:1009–1058+` | 主：`ds_steps[:,k]` vs 专家第 k 步弦长；辅：`ds_profile[:,k,m≥1]`（0.5）；`--terminal-coef`；`--plan-res-coef` |
| eval 工具链 | `.slim/deepwork/p1_v2/a3/eval_a3.py` | `--tracker lqr --tail chain`；`--path-head structured`；`--lane-anchor inject`；R37 新增：`--ego-warp-mode`、`--ds-init-override`、`--ds-bounds`、`--long-exog-fill {off,neutral,true}`（全部默认关，off 逐位） |
| DAgger 采集 | `tools/dagger_collect.py:1240–1272` | 学生 roll-in + 规则专家标签（`--labeler idm\|pure_pursuit`）；train-spec 硬隔离；BC 口径 (ds,dθ) 逆映射 |
| spec 池 | `env/specs/scenarios_train*.json`（含 `_slice200`、`_dagger_r1-r3`）、`scenarios_eval500.json`、`scenarios_val.json` | dagger 工具强制与 eval/val 池无交集 |

**数据链（P0-B 依赖）**：`expert_bc.npz`(expert5k_v8) → H2 标签（label_row/E3 sample_lane_chain）→ I1 抽取 → DP-6 分片（360,497 行）→ `j1_train.py`（allowlist=`path_head.`+`plan_head.fusion.2.`）。**P0-B 需把该链换成 dagger 采集源**（M8 检查中）。

---

## 6. 玩家（用户）讨论与答疑纪要（本轮）

### 6.1 流程与卫生（用户提出的要求，已执行）
- **轮次循环**：pull GPT 答复 → 排摸代码 → 有分歧更新 argue 新一轮讨论 / 无分歧开工。
- **git 卫生**：行数勘误不再单独占 commit，**并入本轮收口提交**（本轮已执行；`e364427` 含勘误）。
- **审计纪律**：诊断只在 `.slim/`/`runs/`（gitignored）；仅 `docs/` 入库；生产改动默认关 + 逐位验证；**无 CG 放行不训练**。

### 6.2 技术五问与结论（用户提问 → 核查答案）
1. **"初始速度为 0？为什么以前没出现？"**
   - 核查：calib50 episode 的 spawn 速度 = **6.0 m/s**（`env__speed[0]=6.0`，非 0）。真正的问题是"**掉速不可恢复**"（保速律 + 0.72 m/步地板 ⇒ 自锁 1.44 m/s）；**从静止起步会在同一点卡死**（部署风险），所以该缺陷必须修。
   - 以前未暴露：旧头=速度目标生成器（plan 隐含 v_ref>v，主动拉回速度）；历史门只测横向。
2. **"net 的输入有道路限速么？怎么知道该开多快？"**
   - 限速**存在于** LD 7 维第 5 维（`env/obs/ld.py`）与 others 上下文（`speed_limit_norm_mps`，`env/obs/builder.py`）⇒ **冻结编码器可见**；
   - 但 **StructuredPolicyHead 的显式输入（long_exog 7 键）无限速列、无目标速度输出**，且标签=保速弦长 ⇒ "该开多快"只能间接感觉 + 模仿"你现在多快"。
3. **"训练集里有起步/刹停场景么？"**
   - 覆盖：v<0.3 **1.18%**、v<1 **2.67%**、v<2 **7.93%**；
   - 低速行瞬时残差≈0且偏负（[0,2) −0.17）⇒ **有刹停/低速跟随**；v<1 行 6 步未来链 0.23→0.54（≈0.25 m/s²，弱）⇒ **几乎无"静止起步加速"监督**，且弱信号仅在 aux 未来链、被 SmoothL1 抹平。
4. **"纵向设计与监督是否合理？哪些地方应放开训练？"**
   - **不合理**：①先验语义（保速 vs 应为目标速度 `min(限速, √(a_y/κ), 前车约束)`）；②通道（加速意图在 aux `ds_profile[:,1..5]`，系数 0.5、契约不进动作/LQR）；③损失（均值回归抹加速尾部；σ×`DS_HIGH=10` 尺度下逃离低速需 raw +0.2~+2.2 vs 实测 −0.57；缺速度剖面/jerk/终端加速项；κ 限幅表只按专家残差标定）。
   - **放开训练（建议）**：①必须 `path_head.*` + `plan_head.fusion.2.*`（已证 head-only 0.479→0.058）；②建议加 head 前**小适配器**或微调编码器**末层**（吸收装配位移 Δds +0.138 / β·δ 2.72× 正交）；③**不放** WM 主干/编码器主体/Mem/旧 PolicyHead/MoE/ValueHead（C3 只放 residual/physical heads）；④κ 限幅表在闭环执行数据上重标定。
5. **"为什么用 50 集而不是 eval500/clean500？验证集变了怎么比？"**
   - **calib50 不是新集**：它是主线自 R25 起的开发集；台账锚点（legacy 0.40/Q1 0.50/K1 0.52/T2 0.58/统一基线 0.56/**IDM 0.756**）皆 calib50 口径 ⇒ 本报告各臂与历史锚点**同集可比**；
   - GPT 三级协议（Calib50 开发 → Clean150 晋级 → Eval500 正式：3 seeds/paired bootstrap/McNemar）保持不变；R37 各臂连 dev 硬门未过，无需（也不应）先烧 Eval500；
   - 分辨率：±1 集 ≈ ±2pp（只作筛选）；历史数字若来自其它集需按口径登记逐条标注。

---

## 7. GPT 侧往来（本轮）

| 条目 | 内容 | 处置 |
|---|---|---|
| `66497a2`（551 行） | Round-37 方案：四臂矩阵 + 统一基准 + 代码修正 6 项 + 训练序 5 步 + 门 A–E | 受理；§5.62 提出执行序提案（C1/C2 零训练可跑⇒先分诊）+ 4 条事实注记 |
| `effea53`（168 行） | 接受我方执行序；**关键修正：C2 不是 C3 的开关**（C3 由 5 条联合条件触发）；timeout 门=比例；"Policy 收编"=技术债 4 硬约束 | 采纳（§5.63）；C3 触发条件并入判读 |
| 我方 §5.64–§5.66 | M1–M7 结果 / 收口清单 / 复盘与四问核查 | 已推送（`0ecbad6`→`e364427`） |

---

## 8. 问题与风险（未决/新增）

1. **接口修复固化路径未定**：目前在 eval 侧 flag（`--long-exog-fill`）；正式形态三选一（builder 直供真值 / 模型侧"缺键=中性"默认 / 保留 eval 侧）——需与训练装配一并定稿（否则训练/评测再次分叉）。
2. **真值回填的横向承受度**：真值档更强释放速度但 off 0.18；其收益需在训练层配套（横向质量修复）后重测，不能单独判定为"真值不好"。
3. **κ 限幅表标定口径**：现表来自专家残差 P95/P99，未覆盖"提速时的横向裕量"；A1/A3 的越界说明该表在提速状态下偏紧/失配（建议闭环执行数据重标）。
4. **"0 起步/低速恢复"无考核项**：当前 calib50 首发 6 m/s，掩盖起步缺陷；建议新增"起步-恢复"评测切片（v≈0/低速注入 → N 步内恢复至场景限速量级）。
5. **Eval500/口径**：历史数字的集口径需逐条登记（含 IDM 0.756 的集来源核验），避免跨集混比。
6. **M3 的"装配位移"残余**：即使补齐三键，`eval_on−train` 仍有 Δds +0.138（融合 β·δ 2.72× 正交）——训练必须在 eval 装配下进行，否则差集仍存在。
7. **A1 的蛇形极限环机制**为推断（转向饱和步 accel −0.21 m/s²、heading err 0.237 rad），未做轮胎/控制器级拆解。

---

## 9. 待令（CG）与下一步

**待令**：
1. **接口修复固化形态**（neutral=当前最佳可用；true 待训练配套；env 侧直供）；
2. **P0-B 训练放行** + 设计选择（roll-in 策略、train spec 池、规模/预算、门；建议 Calib50 快筛→Clean150 晋级）——M8 试点数字一并呈报；
3. **C3（WM/想象步动力学）时机**；
4. **基准口径登记**（本协议 C0=0.58/0.18/0.12 与历史 kappa-compose 分离）；
5. **新增考核项**：起步-恢复切片（问题 4）；
6. **训练放开清单确认**（§6.2-Q4 建议）。

**下一步（放行后）**：
- P0-B：`dagger_collect`（train spec）→ H2/I1 抽取（缺口修复）→ `j1_train` 在 **eval 装配** 下训练（头 + fusion.2 + 可选小适配器）→ Calib50 快筛（含接口填充档）→ Clean150 晋级 → Eval500 正式；
- 并行：κ 表闭环重标定、接口固化落地、起步-恢复切片工具；
- C3：WM residual/physical heads（冻结其余）→ 想象步动力学 → 与 warp 联测。

---

## 10. 证据索引

- **决策/复盘**：`docs/deepseekv4p1_argue.md` §5.62–§5.66；本报告；`docs/architecture_contract.md`
- **M1**：`.slim/deepwork/p1_v2/round37/m1_{gate_a,k1_contract_gate,offline_od_readout}.{json,md}`；入库 commit `0d63bbc`
- **M2**：`.slim/.../round37/m2_c0c1.{md,json}`；`runs/round37/m2/{c0_k1e2_e2_r2,c1_t1keepres_e1_r2}/`
- **M3**：`.slim/.../round37/m3_report.md`；`runs/round37/m3/m3_diag.json`
- **M4**：`.slim/.../round37/m4_report.md`；`runs/round37/m4/{c2_warp_analytic,c1_f3_v2p0}`
- **M5**：`.slim/.../round37/m5_report.md`；`runs/round37/m5/{a1_ds416,a2_ds30,a3_floor25}`
- **M6**：`.slim/.../round37/m6_report.md`、`runs/round37/m6/m6_calib.json`；`runs/round37/m6/{c1_fill,a1_fill}`
- **M7**：`.slim/.../round37/m7_report.md`、`runs/round37/m7/m7_unit_check.json`；`runs/round37/m7/{c1_true,smoke_*}`
- **代码真源**：`env/tracking.py`、`env/obs/{od,ld,static,builder}.py`、`net/path_head.py`、`net/model.py`、`.slim/.../round36/{h2_data_spec,i1_extract_dataset,j1_train,j1_eval}.py`、`.slim/.../a3/eval_a3.py`、`tools/dagger_collect.py`
