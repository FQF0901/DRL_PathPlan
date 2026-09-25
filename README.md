# DRL_PathPlan

MetaDrive 城市/高速驾驶规划 RL：观测 → 策略输出 `(ds, dθ)`（下一 0.5 s 的弧长 + 航向变化）→ 网络内部
world model rollout 产生 3 s / 6 点自车轨迹 → MPC/LQR 跟踪该预瞄。分阶段训练：**A 世界模型（teacher forcing）
→ B 规划器 BC（含 rollout 轨迹辅助）→ C PPO RL（KL 锚定 B 快照）**。

> **一句话现状（2026-09-25）**：全链路已端到端跑通（场景生成 → 专家数据 → WM → 策略 BC → PPO → 评测/KPI/监控），
> **瓶颈在策略质量而非管线**：开环模仿指标已接近专家（动作 ds 3.499 m vs 专家 3.500 m；WM ADE 1.647 vs 匀速 3.269），
> 但闭环 KPI 明显落后规则基线（success 0.18–0.40 vs 0.82；off-road 0.40–0.72 vs 0.06）。
> 主要缺口是**横向车道保持**（BC 复合误差）；**Stage C 的 PPO 目前未带来 KPI 增益**——它修好了低速问题并把
> 速度比从 0.36 提到 0.63，但代价是碰撞率与成功率恶化（奖励 hack，见 §3.1-P10）。

---

## 1. 系统架构

### 1.1 接口与数据流

```
MetaDrive 场景（spec JSON）
   │  env/metadrive_env.py + env/obs/*
   ▼
观测（字典，含 mask 与 6 帧历史）
   │  net/model.py: DrivingModel.forward()
   ▼
编码器 → 时序（GRU，hist_valid 门控）→ 空间消息传递 → MoE（primary + 8 specific）
   │                                    └── world model（预测未来 OD/LD）→ 内部 rollout（B1）
   ▼
输出：action_mu/logstd(2)、value(1)、traj_xy(6,2)、od_pred(6,16,5)、ld_pred(6,16,4)、
      router_logits(8)、plan(6,2)、latent(96)
   │
   ├─ 训练时：env 执行 action（0.5 s）；轨迹/未来帧作为辅助监督
   └─ 评测时：env/tracking.py 的 ExactTracker / LqrTracker 跟踪 plan 预瞄（插值成 30 点 @10 Hz）
```

**观测通道**（`env/obs/`，全部可插拔注册）：

| 通道 | 形状 | 说明 |
| --- | --- | --- |
| `ego` | (1, 8) | 自车运动学；**维度 6:8 承载上一策略步 `(ds,dθ)`**（采集/评测注入，见 §4 已修 bug）|
| `od` | (16, 9) | 周围车辆，盒式 scope（前 100 / 后 50 / 左 25 / 右 25 m），TTC 排序 top-16，零填充 + mask |
| `ld` | (16, 7) | 车道线点（同 scope） |
| `nav` | (1, 11) | 路线导航（相对坐标 + route_completion 等） |
| `signal` | (1, 4) | 本场景无信号灯，预留 |
| `od_hist` / `ld_hist` | (6, 16, F) | 6 帧历史 @0.5 s，**对齐到当前自车系（SE(2) 变换）**；含 `hist_valid(6)` 门控（episode 起点不足 6 帧时置 0） |

### 1.2 网络（`net/`）

- **规模**：H=128、8 个 specific expert（128→256→128）→ **1,174,875 参数**（110 tensors，占 1.5M 预算 78.3%）。
  > 注：`net/param_probe.py` 默认配置打印 666,263（不同配置），以训练 ckpt 的 1,174,875 为准。
- **结构**：编码器 → 时序聚合（GRU + `hist_valid` 门控）→ 空间消息传递（OD/LD 间）→ **MoE**（primary 常开 +
  8 个 specific 专家，sigmoid 门、残差零初始化）→ world model（预测未来 OD/LD）→ 策略头 / 价值头。
- **MoE 语义**：primary 承担主干能力，specific 只做残差增量；路由标签来自场景 taxonomy（8 类：
  `cutin_active / cutout_active / crowded / car_following / on_curve / merging / roundabout_near / near_intersection`）。
- **rollout（B1）**：`rollout=True` 时网络用自身预测的未来 OD/LD 迭代展开，产出 `traj_xy`/`plan`（可 detach WM，
  Stage B 强制 detach，Stage A/C 依配置）。

### 1.3 分阶段训练 v1.1（2026-09-25 用户确认并实现）

| 阶段 | 训练什么 | 冻结什么 | 监督目标 | 验收证据 |
| --- | --- | --- | --- | --- |
| **A：WM teacher forcing** | 编码器 / 时序 / 空间 / MoE / WM | 策略头 / 价值头 | 未来 OD/LD 真值（(episode, step+5k) 查表、t0 对齐、mask+valid）；ego 条件 = **专家 GT 动作序列** + 噪声（p=0.5, σ_ds=0.3, σ_dθ=0.05）| **ADE 1.647 / FDE 2.739 vs 匀速 3.269 / 4.483 ✓**（`runs/train/stage_a_matched`）|
| **B：Planner BC** | 主干 / MoE / 策略头 / specific | WM（且 rollout detach） | 专家动作（主损失 1.0）+ **rollout 轨迹小权重辅助 0.3** + router BCE 0.1；先 primary 后 specific | ds **3.499 m**（专家 3.500）、轨迹 MAE **0.633 m**；aux=0 消融 off-road 1.0 → 0.5 |
| **C：PPO RL** | specific / 策略头 / 价值头（WM 先冻后放） | 编码器 / 时序 / 空间 / MoE(primary) | 规则奖励 + GAE；**KL 锚定 Stage-B 快照**（系数 0.05→0 线性衰减）；primary lr×0.1；可选 critic 预热（value-only） | 50 updates 跑通、44–49 steps/s、无坍塌；300-update 版见 §3 |

**为什么 A 在 B 之前（因果一致性）**：`traj_xy` 是内部 rollout 的产物，rollout 依赖 world model。若 WM 未训练就
用 `traj_xy` 做 BC 监督，等于监督"假观测下产出的轨迹"（且梯度会污染 WM）。因此先以真值训练 WM（A），再在
**WM 冻结 + detach** 的前提下用 rollout 轨迹做小权重辅助（B）。

### 1.4 监督与冻结配合（关键规则）

1. **A**：唯一使用 teacher forcing 的阶段（GT ego 动作 + 噪声）；目标帧必须做 t0 帧对齐 + slot 身份匹配
   （原始 OD 槽位按 TTC 排序，跨帧同一物理车会换槽 → 直接回归病态，匹配后 ADE 3.38 → 1.65）。
2. **B**：WM 冻结 + rollout detach → 轨迹辅助只训练策略侧；`traj_aux_weight=0.3` 为**闭环验证过的配方**
   （0.1 会掉横向，见 §3）；`bc_phase_split` 保证 specific 阶段不动 primary 主干。
3. **C**：无 teacher forcing；KL 锚到 **Stage-B 策略快照**（不是专家 BC）；可选 `--bc-anchor`（默认关）；
   WM 在前 `updates//4` 冻结、之后解冻（**当前 PPO loss 不消费 WM 输出，解冻仅为后续 WM 辅助损失预留**，见 §4-P8）。

### 1.5 数据集配合

```
env/specs/*.json（11,250 条已验证，0 失败；11 种可采样几何 + 显式 spawn 车道 + 脚本事件）
        │  tools/collect_expert.py（专家 = 过滤后的 IDMPolicy；roundtrip + terminal-window 过滤；yield ≈ 0.72）
        ▼
BC 数据集（逐帧记录，训练时在线重建 6 帧历史；带 obs_fingerprint 守卫）
   ├─ runs/bc_expert_full：200 场景 → 10,777 样本（早期）
   └─ runs/bc_expert_2k：2,000 场景 → **103,938 样本**（yield 0.718；8 个标签全远超下限；难度×主标签配平）
        │
        ├─→ Stage A：未来 OD/LD 目标（查表 + 对齐 + mask）
        ├─→ Stage B：动作 + traj6 + 逐步标签
        └─→ Stage C：规则奖励环境（rollout 采样）
```

**纪律**：BC 数据必须与观测版本一致（`env/obs/__init__.py::obs_fingerprint` 写入 meta，`BCDataset.load` 不匹配时告警）；
采集/评测/训练共用同一套场景 spec 与专家定义。

---

## 2. 性能现状（证据，2026-09-25）

### 2.1 基线对照（50 条 val slice，冻结参考 `runs/baseline_eval/val_reference*.json`）

| 对象 | tracker | success | collision | off-road | route_completion | speed_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| **规则基线（pure_pursuit_idm）** | exact | **0.82** | 0.10 | **0.06** | 0.925 | 0.734 |
| 1k-BC（10 条协议切片） | exact | 0.40 | 0.10 | 0.50 | 0.641 | 0.750 |
| 2k-BC aux0.1 | exact | 0.30 | 0.00 | 0.70 | 0.594 | 0.765 |
| **2k-BC aux0.3** | exact | **0.40** | 0.20 | 0.60 | **0.675** | **0.796** |
| 1k-BC | lqr | 0.24 | 0.06 | 0.68 | 0.526 | 0.538 |
| 2k-BC aux0.1 | lqr | 0.20 | 0.00 | 0.78 | 0.538 | 0.399 |
| **2k-BC aux0.3** | lqr | **0.26** | 0.00 | 0.72 | 0.539 | 0.364 |
| Stage C v1（50 updates，从 1k-BC） | lqr | 0.20 | 0.10 | 0.70 | 0.474 | 0.496 |
| **Stage C v2（300 updates + critic 预热 10，从 2k-BC aux0.3）** | lqr | **0.18** | 0.14 | 0.68 | 0.453 | **0.631** |
| **Stage C v2**（同上，exact 口径） | exact | **0.10** | 0.50 | 0.40 | 0.398 | **1.093（超速）** |

### 2.2 开环 vs 闭环

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Stage A WM ADE / FDE | 1.647 / 2.739（匀速 3.269 / 4.483） | 胜匀速 ✓ |
| Stage B 动作克隆 `mu_ds` | 3.499 m（专家 3.500 m） | 开环几乎完美 |
| Stage B 轨迹 MAE | 0.633 m（1k 版 1.110 m） | 10× 数据显著改善 |
| **闭环 off-road** | **0.60–0.72**（基线 0.06） | **开环好 ≠ 闭环好：BC 复合误差** |

**失败模式取证**（2k-BC LQR，50 条）：终止原因 `out_of_road 39 / arrive_dest 10 / max_step 1`，失败发生在
途中（rc 0.16–0.94），**不是终点行为**；成功的 10 条 rc≈0.98。

---

## 3. 问题与线索（对抗性评审的靶子）

### 3.1 已定位的核心问题

- **P1 横向车道保持（最高优先级）**：BC 策略在专家状态分布外漂移出车道（复合误差）。10× 数据把开环指标
  推向完美，但闭环反而略降 → 需要 **on-policy 纠正**（Stage C）或 **DAgger 式专家纠偏采集**（候选，未实现）。
- **P2 闭环执行链**：LQR 跟踪比 exact 执行差约 0.12 off-road（0.60 → 0.72）；跟踪器参考已修为 6 点预瞄（见 3.3），
  但增益未做干净标定（此前的扫描指标被"冲过终点后继续开"污染）。
- **P3 critic 几乎无解释力**：`value/explained_var ≈ 0`（value_loss 6–33）→ 优势噪声大。已实现 critic 预热
  （value-only，前 N 步冻结主干），预热期 explained_var −0.036 → +0.012；但**预热冻结主干 → 拟合能力受限**，
  更强 critic 需要解冻（会破坏"预热期策略逐位不变"的验收口径）。
- **P4 低速吸引子（已监控）**：固定探针显示 `speed<2 m/s` 两档 ds 均值 0.32 / 1.05 m → **不是硬不动点**
  （ds > v·0.5，在缓慢加速），但恢复增益小；`probe/low_speed_alert` 每次 update 报警。
- **P5 仿真非确定性**：少数脚本事件场景（cut-in 等）同种子 run-to-run 有微小差异（排除 PYTHONHASHSEED →
  MetaDrive 内部线程/时钟时序）→ **事件类 KPI 有噪声**，Gate 4 需记录；候选修复：事件脚本改纯步数驱动。
- **P6 观测/历史设计未消融**：盒式 scope（前100/后50/左25/右25）、top-16、6 帧历史、`hist_valid` 门控、
  `prev_action` 注入——均为设计决策，未做消融实验。
- **P7 Stage C 的 WM 解冻是"记账"**：PPO loss 目前不消费 WM 输出，解冻只让参数回到优化器（为后续 WM 辅助
  损失预留）→ 当前 Stage C 对世界模型没有直接梯度。
- **P8 评测样本量**：10 条协议切片噪声大（n=10，Wilson CI 宽），50 条 slice 更可信。
- **P9 MoE 路由标签是"场景级"的**：路由 BCE 用逐步标签（taxonomy 打标），未验证路由器是否真的学到可解释分工。
- **P10 Stage C 的 PPO 未带来 KPI 增益（奖励 hack）**：300 updates（critic 预热 10）后，速度比 0.364 → 0.631
  （低速问题被 RL 修好：探针 0–1 m/s 档 ds 0.19 → 5.7 m、`low_speed_alert` 全程解除），但 success 0.26 → 0.18、
  碰撞 0.00 → 0.14（exact 口径更极端：succ 0.4 → 0.1、碰撞 0.2 → 0.5、速度比 0.796 → **1.093 超速**）。
  训练监控显示 u90–210 出现"off-road/步 → −1.28、总奖励转负"的 hack 期、u240 后自行回落，但
  **critic 全程 `explained_var ≈ 0.004`**（无法为终端风险定价）且 KL 锚系数衰减到 0 → 策略向廉价的速度项漂移。
  **修复方向**：速度项按在道状态门控（或乘安全指示）、提高 off-road/crash 权重、KL 系数下限 + primary lr 再降、
  critic 强化（解冻主干 / 更多预热），并且**先修横向弱点**（否则 RL 只是在一个易出界的策略上做速度优化）。

### 3.2 已排除的假设（有数据支撑，避免重复踩坑）

- 奖励排序错误（爬行 vs 碰撞）：已核验 `爬行 +58 < 碰撞 +105 < 正常行驶 +252`，排序正确；
- 策略概率坍塌 / `prev_action` OOD / warmup 不足 / 速度 OOD 单独致病：均被 live probe 证伪；
- Stage B 的 `traj_xy` 目标不匹配（监督的是 3 s 6 点而非 6 帧）：已修；
- 动作损失未接线 / 策略头 sigmoid 饱和：已修。

### 3.3 已修复的关键 bug（本轮）

| bug | 影响 | 修复 |
| --- | --- | --- |
| **跟踪器参考 = 单个动作**（而非 6 点预瞄） | 闭环"偏出车道 + 速度衰减"的元凶（LQR off-road 0.90 → 0.40）| `collect_rollout`/评测改为传 `out["plan"]`（6 点→插值 30 点）|
| 评测 `prev_action` 恒为 0 | 评测观测 OOD（与采集语义不一致）| `_CkptController` 按采集语义注入（由位姿反推）|
| `monitoring.py` 的 `csv` 参数遮蔽模块 | 监控静默失效（loss 曲线丢数据）| 改名 + 递归 flatten（现 42 条序列）|
| spawn worker 无 GL 运行时 | MetaDrive 崩溃（Known Pipes / IndexError）| `pipeline/gl_runtime.py` 守卫（父进程/worker 双保险）|
| 专家数据与观测版本漂移 | 静默错训 | `obs_fingerprint` 守卫 + 重新采集 |
| 未来帧槽位按 TTC 排序跨帧换位 | WM 回归病态（ADE 3.38）| slot 身份匹配（ADE → 1.65）|

---

## 4. 仓库结构与运行方式

| 目录 | 用途 |
| --- | --- |
| `config/` | `default/env/model/train/eval` 五份配置（冻结值见 `config/README.md`）|
| `env/` | MetaDrive 封装：scenario（spec/taxonomy/generator/validator/behaviors/labels）、obs、expert、tracking |
| `reward_model/` | 规则奖励项（9 项）、聚合（dense+terminal+CaRL+potential shaping）、KPI（primary 分组 + Wilson CI）|
| `net/` | 编码器 / 时序 / 空间 / MoE / world model / 策略头 / rollout |
| `pipeline/` | 阶段 A/B/C、trainer、buffer、vector_env、eval_runner、monitoring、gl_runtime |
| `tools/` | `gene_env.sh`、`train.py`、`test.py`、`collect_expert.py`、`baseline_eval.py`、`visualize.py`、`measure/*` |
| `tests/` | 88 项测试（`tools/venv-python -m pytest tests/ -q`）|
| `runs/`、`data/` | 运行产物（gitignored）|

```bash
# 环境（一次性；本机无系统 libGL，需 glvnd 本地解包）
python3 -m venv --without-pip --system-site-packages .venv
python3 -m pip --python .venv/bin/python install metadrive-simulator
python3 -m pip --python .venv/bin/python install "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
bash tools/setup_gl_libs.sh
# ⚠ 所有 MetaDrive 运行统一用 tools/venv-python（自动设置 LD_LIBRARY_PATH）

# 场景 + 专家数据
bash tools/gene_env.sh
tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json --limit 2000 \
    --out runs/bc_expert_2k --workers 6          # 并行采集（输出与单进程逐字节一致）

# 分阶段训练
tools/venv-python -m pipeline.stages --stage A --bc-dir runs/bc_expert_2k --out runs/train/stage_a
tools/venv-python -m pipeline.stages --stage B --bc-dir runs/bc_expert_2k --ckpt runs/train/stage_a/final.pt \
    --bc-epochs 20 --traj-aux-weight 0.3 --out runs/train/stage_b
tools/venv-python -m pipeline.stages --stage C --ckpt runs/train/stage_b/final.pt \
    --pool local --envs 1 --updates 300 --critic-warmup-updates 10 --out runs/train/stage_c

# 评测（exact = Stage B 语义；lqr = Stage C 闭环；--policy baseline 为规则基线）
tools/venv-python tools/test.py --policy ckpt --ckpt runs/train/stage_b/final.pt \
    --spec env/specs/scenarios_val_slice50.json --limit 50 --workers 2 --tracker lqr --out runs/eval
```

---

## 5. 下一步计划

1. **修横向弱点（最高杠杆）**：DAgger 式专家纠偏采集（在策略漂移状态上让专家给动作）→ 直击 P1 复合误差；
   或提高 BC 的横向信号（aux 权重/损失形态）并以 50 条 LQR slice 做闭环选型。
2. **修 RL 的奖励/优化配置**（P10）：速度项按在道状态门控、提高 off-road/crash 权重、KL 系数下限 + primary lr 再降、
   critic 强化（解冻主干 / 更多预热 / 更大 vf_coef）→ 重跑 Stage C（当前配置**无 KPI 增益，不可作为 Gate 4 证据**）。
3. **跟踪器增益标定**（干净指标：rc≥1 前的 off-road / 纯横向 CTE）。
4. **Gate 4 薄切片验收**：按冻结的 KPI 协议（primary 分组 + 弱类 floor + overall_success）出具正式结论。
5. 事件脚本确定性修复（P5）与观测消融（P6）。

---

## 6. 证据索引

| 内容 | 路径 |
| --- | --- |
| 阶段 A 产物/曲线 | `runs/train/stage_a_matched/` |
| 阶段 B（1k / 2k / 2k-aux0.3） | `runs/train/stage_b_matched/`、`stage_b_2k/`、`stage_b_2k_aux3/` |
| 阶段 C（v1 / v2） | `runs/train/stage_c_run1/`、`stage_c_v2/`（含 `monitor/metrics.csv` 42 条序列）|
| 评测记录 | `runs/eval/*/metrics.json` + `episodes.csv` |
| 冻结基线 | `runs/baseline_eval/val_reference.json`、`val_reference_by_primary.json` |
| BC 数据 | `runs/bc_expert_full/`、`runs/bc_expert_2k/`（`report.json` 含过滤/配平统计）|
| P0 测量 / 数据集统计 / 可行性分析 | `docs/p0-measurements.md`、`docs/dataset_stats.md`、`docs/feasibility-analysis.md` |

> 依赖：MetaDrive 0.4.3、numpy<2、Python 3.10、torch 2.3（`.venv` 复用系统已装包）。
