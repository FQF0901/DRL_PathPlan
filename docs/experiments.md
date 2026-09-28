# 实验记录（2026-09-25）

> 本文件把关键实验的数字与命令固化进仓库（`runs/` 为 gitignored，评审者无法离线查看）。
> 所有数字来自实际运行产物（`runs/*/metrics.json`、`runs/*/monitor/metrics.csv`、日志）。
> **2026-09-26 清理**：文中 `runs/*` 产物目录已随 `runs/` 清理删除（可按各节命令重生成）；数据集落在
> `datasets/`（命名 `BTC<时间戳>_expert<N>k`，见 `docs/design-v1.2.md` §3.5），不随 `runs/` 清理。

---

## 0. 运行清单

> 下列目录为历史运行记录（产物已清除），可由对应章节的命令重新生成。

| 目录 | 类型 | 关键配置 | 状态 |
| --- | --- | --- | --- |
| `runs/train/stage_a_matched/` | Stage A（WM teacher forcing） | 10,777 样本 / 30 留出 episode / 10 epochs / batch 256 / GPU / slot 匹配 | ✓ 完成 |
| `runs/train/stage_a/` | Stage A（原始槽位，对照） | 同上去掉 slot 匹配 | ✓ 完成（对照） |
| `runs/train/stage_b_matched/` | Stage B（1k 数据） | 20 epochs（10 primary + 10 specific）/ aux 0.3 / WM 冻结+detach | ✓ 完成 |
| `runs/train/stage_b_2k/` | Stage B（10× 数据） | 同上但 aux 0.1（config 默认） | ✓ 完成 |
| `runs/train/stage_b_2k_aux3/` | Stage B（10× 数据 + 验证配方） | 20 epochs / aux 0.3 | ✓ 完成 |
| `runs/train/stage_c_run1/` | Stage C v1 | 50 updates / rollout 256 / LQR 本地池 | ✓ 完成 |
| `runs/train/stage_c_v2/` | Stage C v2 | 300 updates / critic 预热 10 / 埋点全开 | 运行中 |
| `runs/bc_expert_full/` | BC 数据（200 场景） | 旧观测 scope 重采版 | ✓ |
| `runs/bc_expert_2k/` | BC 数据（2,000 场景） | 6 workers 并行采集 | ✓ |

---

## 1. 数据集

| 数据集 | 场景数 | 样本数 | 产出率 | 过滤 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 场景 spec（全量） | 11,250 | — | — | — | 校验 11,250/11,250，**0 失败**；11 种可采样几何 |
| `bc_expert_full` | 200 | 10,777 | 0.727 | roundtrip / terminal window | 早期版本 |
| `bc_expert_2k` | 2,000 | **103,938** | 0.718 | roundtrip_fail 28,982；terminal_window 11,865 | 8 标签全远超下限（cutin 1777 / cutout 3212 / crowded 15892 / car_following 9961 / on_curve 8453 / merging 19388 / roundabout_near 2794 / near_intersection 11237）；难度×主标签配平（权重 0.62–3.69）|

采集命令（并行版与单进程输出逐字节一致）：
```bash
tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json \
    --limit 2000 --out runs/bc_expert_2k --workers 6
```

---

## 2. Stage A：世界模型（teacher forcing）

配置：ego 条件 = 专家 GT 动作序列（+噪声 p=0.5, σ_ds=0.3, σ_dθ=0.05）；目标 = 未来 OD/LD 帧（(episode, step+5k)
查表、t0 对齐、mask+valid）；直接多步 Huber + 角度 1−cos；训练编码器/时序/空间/MoE/WM，策略/价值冻结。

| 运行 | loss（首→末） | val | **ADE（WM / 匀速）** | **FDE（WM / 匀速）** |
| --- | --- | --- | --- | --- |
| `stage_a_matched`（slot 匹配） | 5.950 → 3.529 | 6.837 → 4.005 | **1.647 / 3.269** | **2.739 / 4.483** |
| `stage_a`（原始槽位，对照） | 6.274 → 3.749 | — | 3.377 / 9.138 | 5.467 / 15.273 |

> **结论**：WM 胜过匀速基线 ✓。槽位身份匹配（原始 OD 槽位按 TTC 排序、跨帧换位）是必要修正（ADE 3.38 → 1.65）。

命令：
```bash
tools/venv-python -m pipeline.stages --stage A --bc-dir runs/bc_expert_2k \
    --val-frac 0.15 --match-future-slots --out runs/train/stage_a_matched
```

---

## 3. Stage B：规划器 BC

配方：先 primary（10 epochs，冻结策略侧只训主干+MoE+router）后 specific（10 epochs，冻结 primary）；
损失 = 动作 BC 1.0 + rollout 轨迹辅助（WM 冻结 + detach）+ router BCE 0.1。

| 运行 | 样本 | 动作 `mu_ds`（策略 / 专家） | 轨迹 MAE | 备注 |
| --- | --- | --- | --- | --- |
| `stage_b_matched`（1k） | 10,777 | 3.319 / 3.260 m | 1.110 m | aux 0.3 |
| `stage_b_2k`（10× 数据） | 103,938 | **3.499 / 3.500 m** | **0.633 m** | aux 0.1（config 默认）|
| `stage_b_2k_aux3`（10× 数据 + aux 0.3） | 103,938 | 3.525 / 3.500 m | 0.633 m | **闭环更好**（见 §4）|

命令：
```bash
tools/venv-python -m pipeline.stages --stage B --bc-dir runs/bc_expert_2k \
    --ckpt runs/train/stage_a_matched/final.pt --bc-epochs 20 \
    --traj-aux-weight 0.3 --out runs/train/stage_b_2k_aux3
```

---

## 4. 评测（冻结协议：primary 分组 + Wilson CI + 弱类 floor）

**统一评测协议（2026-09-27 起）**：`env/specs/scenarios_eval500.json`（**500 条完整 episode**；按 `labels.geometry`
分层、seed=0；配套数据集 `datasets/BTC20260927-1734_expert500val`，与训练池 seed 交集 = 0）。
数据口径：**只允许两个子集** —— train = 5k 完整 episode（全部用于训练）、eval = 这 500 条；
另允许一个 16 条的代码冒烟集（`scenarios_smoke16.json`，非协议）。
命令：`bash tools/test.sh`（默认已是 500 集；可用 `SPEC=`/`LIMIT=`/`WORKERS=` 覆盖）。

| 对象 | 样本 | success [95% CI] | collision | off-road | rc | speed_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 规则基线（PurePursuitIDM） | 500 | **0.756** [0.716, 0.792] | 0.144 | 0.068 | 0.882 | 0.740 |
| v1.2 IL 5k 第 3 轮（`1100_stageB/final.pt`） | 500 | **0.326** [0.286, 0.368] | 0.066 | 0.604 | 0.666 | 0.457 |

分几何（n≈45/类；格式 = 基线 / IL）：split 0.82/0.80、straight 0.76/0.58、ramp_out 0.57/0.61、
ramp_in 0.61/0.50、merge 0.87/0.28、intersection 0.70/0.30、t_intersection 0.87/0.16、
curve 0.74/0.11、uturn 0.87/0.16、roundabout 0.76/0.07、tollgate 0.78/0.02。
难度（基线 / IL）：easy 0.98/0.49、medium 0.67/0.20、hard 0.58/0.26。

> **口径变更说明**：2026-09-27 之前的所有数字（本节下表与 §8/§9/§10）用的是 **50 条 slice**，
> 两套数字**不可直接比较**；n=500 显示 50-slice 对 ckpt / 基线分别高估约 **+0.09 / +0.06**。
> 早期"uturn/tollgate 专家也常失败"的判断（基于 n=4）在 n=45 上被推翻：基线在这两类是 **0.87 / 0.78**。

基线参考（历史产物，已清理；可由 `tools/baseline_eval.py` 重生成）：`runs/baseline_eval/val_reference.json` / `val_reference_by_primary.json`。

| 对象 | tracker | 样本 | success | collision | off-road | rc | speed_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **规则基线** | exact | 50 | **0.82** | 0.10 | **0.06** | 0.925 | 0.734 |
| 1k-BC | exact | 10 | 0.40 | 0.10 | 0.50 | 0.641 | 0.750 |
| 2k-BC aux0.1 | exact | 10 | 0.30 | 0.00 | 0.70 | 0.594 | 0.765 |
| 2k-BC aux0.3 | exact | 10 | 0.40 | 0.20 | 0.60 | 0.675 | 0.796 |
| 1k-BC | lqr | 50 | 0.24 | 0.06 | 0.68 | 0.526 | 0.538 |
| 2k-BC aux0.1 | lqr | 50 | 0.20 | 0.00 | 0.78 | 0.538 | 0.399 |
| 2k-BC aux0.3 | lqr | 50 | 0.26 | 0.00 | 0.72 | 0.539 | 0.364 |
| Stage C v1（50 updates） | lqr | 50 | 0.20 | 0.10 | 0.70 | 0.474 | 0.496 |
| **Stage C v2（300 updates + critic 预热 10）** | lqr | 50 | **0.18** | 0.14 | 0.68 | 0.453 | **0.631** |
| **Stage C v2**（同 ckpt，exact 口径） | exact | 10 | **0.10** | 0.50 | 0.40 | 0.398 | **1.093（超速）** |

命令：
```bash
tools/venv-python tools/test.py --policy ckpt --ckpt <ckpt> \
    --spec env/specs/scenarios_val_slice50.json --limit 50 --workers 2 --tracker lqr --out runs/eval
tools/venv-python tools/test.py --policy baseline --limit 50 --out runs/eval   # 规则基线
```

**失败模式取证**（2k-BC LQR，50 条）：终止原因 `out_of_road 39 / arrive_dest 10 / max_step 1`；失败发生在
途中（rc 0.16–0.94）；成功者 rc≈0.98 → **瓶颈是横向车道保持（复合误差），不是终点行为**。

---

## 5. 消融与调试证据

| 实验 | 结果 | 结论 |
| --- | --- | --- |
| rollout 轨迹辅助（aux=0 消融，LQR） | off-road 1.000 / speed 0.357 | **aux 必需**（0.5 → 1.0）|
| aux 权重 0.1 vs 0.3（2k，LQR） | succ 0.20 → 0.26；off-road 0.78 → 0.72 | **0.3 是验证配方**（开环指标相同：MAE 0.633）|
| 跟踪器参考 = 单动作 vs 6 点预瞄 | LQR off-road **0.90 → 0.40** | 参考修正是闭环元凶修复 |
| WM 未来槽位原始 vs 身份匹配 | ADE **3.38 → 1.65** | 跨帧槽位换位导致回归病态 |
| 奖励排序核验（爬行 / 碰撞 / 正常行驶） | **+58.26 < +105.45 < +251.80** | 排序正确（排除"奖励鼓励爬行"假设）|
| 低速固定探针（BC 快照） | 0–1 m/s → ds 0.32 m；1–2 m/s → ds 1.05 m | **非硬不动点**：ds > v·0.5，在缓慢加速；已加 `low_speed_alert` 监控 |
| critic 预热（value-only，10 步探针） | `explained_var` −0.036 → **+0.0117** | 预热有效；但预热期主干冻结，拟合能力受限 |

---

## 6. Stage C：PPO

| 运行 | updates | 训练观测 | 评测结果 |
| --- | --- | --- | --- |
| v1（`stage_c_run1`） | 50 | 44–49 steps/s；KL 锚 0.05 → 0.0000；无坍塌 | 见 §4（无增益）|
| v2（`stage_c_v2`） | 300 + 预热 10 | ~50 steps/s 稳定；KL 系数 0.05 → 0.0000；无吞吐坍塌 | **无 KPI 增益（奖励 hack）**：速度比 0.364 → 0.631、success 0.26 → 0.18、碰撞 0.00 → 0.14；exact 口径 succ 0.4 → 0.1、碰撞 0.2 → 0.5、速度比 → **1.093** |

**v2 训练期动态**（`runs/train/stage_c_v2/monitor/metrics.csv`，64 条序列）：

| update | speed_ratio | reward/out_of_road | reward/terminal | reward/total | value explained_var | KL 锚 | 探针 ds | low_speed_alert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1–10（预热） | 0.30 | −0.09 | −0.02 | +0.15 | −0.01 | — | 3.28 | 1（报警）|
| 60 | 0.40 | 0.00 | +0.08 | +0.48 | −0.02 | 0.052 | 3.70 | 0（解除）|
| 120 | 0.65 | **−1.28** | **−0.80** | **−1.60** | 0.042 | 0.190 | 5.90 | 0 |
| 210 | 0.63 | −0.81 | −0.51 | −0.76 | 0.290 | 0.077 | 5.48 | 0 |
| 300 | 0.39 | −0.06 | −0.04 | +0.28 | 0.010 | 0.000 | 5.22 | 0 |
| 末 30 步均值 | 0.41 | −0.195 | — | +0.089 | **0.004** | 0.021 | **5.72** | 0 |

**解读**：RL **修好了低速吸引子**（探针 0–1 m/s 档 ds 0.19 → 5.7 m；告警解除；评测速度比 0.364 → 0.631），
但 u90–210 进入"加速 + 驶出道路"的 **奖励 hack 期**（off-road/步 → −1.28、总奖励转负），u240 后自行回落，
最终仍比 BC 起点更激进且更不安全。根因交互：**横向弱点（P1）× 速度项主导的稠密奖励 × critic 解释力不足
（`explained_var` 全程 ≈ 0.004）**，叠加 KL 锚系数衰减到 0 → 策略向廉价的速度项漂移。
→ 修复方向见 `README.md` §5（速度项按在道状态门控、提高惩罚权重、KL 下限、critic 强化、先修横向弱点）。

命令：
```bash
tools/venv-python -m pipeline.stages --stage C --ckpt runs/train/stage_b_2k_aux3/final.pt \
    --spec env/specs/scenarios_train_slice200.json --pool local --envs 1 \
    --updates 300 --rollout-steps 256 --critic-warmup-updates 10 --out runs/train/stage_c_v2
```

---

## 8. v1.2 IL 正式运行（2026-09-26，新架构 + schema v2）

数据（历史数据集，已清理；同规模可由 `tools/collect_expert.py` 重采到 `datasets/BTC<时间戳>_expert2k`）：`runs/bc_expert_2k_v2`（144,763 行入库 / 103,932 可训练；`obs_fingerprint` v2；scope 前 150）。
命令：
```bash
BC_DIR=runs/bc_expert_2k_v2 OUT=runs/train/il_v2_10x10 WM_EPOCHS=10 BC_EPOCHS=10 \
  EXTRA_B="--traj-aux-weight 0.3" bash tools/train.sh
```
（快路径：物化数据 + 宏 batch 1024 / 微 256(A)·512(B) 精确梯度累积；Stage A **GPU 95%**、Stage B **97%**。）

### 8.1 Stage A（WM，10 epochs）

| 指标 | 值 |
| --- | --- |
| loss / val | 2.351 → 0.883 / 0.884 → 0.614 |
| **ADE（WM / 匀速）** | **1.447 / 6.987**（6 个 horizon 全部胜）|
| FDE（WM / 匀速） | 2.729 / 12.254 |
| presence AUC / 正例率 | **0.970** / 0.391（id 轴）|
| entry AUC / 正例率 | 0.702 / 0.0119 |
| 吞吐 | 82.3 s/epoch（data 0.5s + fwd 37.3s + bwd 39.0s）|

### 8.2 Stage B（5 primary + 5 specific）

逐 horizon **加权 MAE (m)** / MSE (m²)：

| horizon | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | 3.0 s |
| --- | --- | --- | --- | --- | --- | --- |
| primary MAE | 0.125 | 0.294 | 0.500 | 0.729 | 0.969 | 1.240 |
| specific MAE | 0.079 | 0.211 | 0.374 | 0.546 | 0.727 | 0.942 |
| primary MSE | 0.088 | 0.420 | 1.097 | 2.170 | 3.627 | 5.496 |

动作（首步 ds/dθ，显式 L1）：weighted **0.1201** / median 0.0652 / p95 0.4010；切片 brake 0.2299、turn 0.2750、
curve 0.1558。router：CE 0.0038 / KL 0.0022（**软目标非占位 ✓**）；top-1 簇准确率 0.265、NMI 0.081、
gate 熵 0.0038；专家混合集中在 expert2 0.297 / expert6 0.232。`mu_ds` 3.522（weighted 3.564；专家 3.218）。

> 度量口径（2026-09-26 修正）：`bc_traj_mae_h{k}_m` = 加权 L1（米）；`bc_traj_mse_h{k}` = 加权 MSE（m²）。
> 旧 `mae=` 实为**未加权 MSE**（含被过滤帧），`sqrt(MSE)` 只是 Jensen 上界（高估 1.89–2.37×），**不可当 MAE**。

### 8.3 闭环（50 条 val slice，LQR 闭环，配对同场景）

| 策略 | success | collision | off-road | rc | speed_ratio |
| --- | --- | --- | --- | --- | --- |
| 规则基线（冻结） | **0.82** | 0.10 | **0.06** | 0.925 | 0.734 |
| v1 2k-BC（aux0.3） | 0.26 | 0.00 | 0.72 | 0.539 | 0.364 |
| v1 Stage-C v2（300 updates） | 0.18 | 0.14 | 0.68 | 0.453 | 0.631 |
| **v1.2 IL（本次）** | **0.38** | 0.04 | **0.52** | **0.648** | 0.407 |

95% CI [0.259, 0.518]；n=50、errors=0；**真均速 4.135 m/s**（旧 `mean_speed_mps`=3.844 实为**末步速度**，已修）、min-TTC 7.37 s、
a_lat95 2.15；**crawl：9.85% 步 < 2 m/s、4.48 s/episode**（22/50 episode 有 crawl）。
分主标签：ramp_out 0.80、merge 0.60、split 0.60、straight 0.60、intersection 0.50、t_intersection 0.50、
ramp_in 0.40、**curve 0.00 / roundabout 0.00 / uturn 0.00 / tollgate 0.00**；难度 easy 0.72 / medium 0.28 / hard 0.07。
（修正版 ckpt 复评与修前**逐位一致** → 度量修正未改动模型。）

### 8.4 本轮修掉 / 记录的问题

| 问题 | 状态 |
| --- | --- |
| 轨迹度量单位混淆（`mae=` 实为未加权 MSE；`bc_traj_err_h*` 实为 m²）| **已修**（新键带单位；旧键保留 alias 并标注）|
| 数据管线瓶颈（GPU 18%、0.44 s/batch）| **已修**（物化 + 宏 batch + 线程放开 → GPU 95%/97%）|
| 物理 batch 1024 OOM（6 步 ST-GNN ≈32 MB/样本）| 用**精确梯度累积**（已验证 ≤1e-4 等价）|
| router 负载集中（8 专家中 2 个占 53%，其余 <0.12）| **记录**：软目标过平滑（top-2 gap p50 = 0.017）→ 待定夺（降 τ / 减边界平滑）|
| 弯道 / 环岛 / 掉头 / 收费站成功率 0% | **已取证定性**（`docs/forensics-2026-09-26.md`）：瓶颈在 **plan 缺车道锚点与偏差回收**（策略 plan 用 ExactTracker 完美执行仍 17/17 失败；换专家路径则 6 arrive + 7 条 rc≥0.72）；机理 = 爬行段横向漂移 + 大转角转向通道塌缩 |
| speed_ratio 0.407（基线 0.734，偏慢）| **记录**；真均速 4.135 m/s、crawl 9.85% 步 < 2 m/s（纵向 P-only v/ref 0.81 为次因）|
| 评测 `mean_speed_mps` 实为末步速度 / ckpt `steer,throttle` 恒 0 | **已修**（真均值 + `final_speed_mps`；ckpt 记录实际 (ds,dθ)，exact 显式 N/A；新增 crawl 指标）|

### 8.5 与 v1 的可比性说明

- 数据：v1 用 200 场景/1.08 万行（旧观测 scope、过滤后只留干净帧）；v1.2 用 2,000 场景/14.5 万行
  （scope 前 150、全帧 + 权重）；**开环数字不可直接跨版本比较**，闭环走同一冻结协议（50 条 val slice）。
- 模型：v1 为 GRU 时序 + 共享 MoE；v1.2 为 mem-bank + 分模态注意力 + plan-head MoE + ST-GNN；
  参数量 1,149,663（H=128）。
- 评测：两版都用 LQR 闭环 + 同一 50 条切片 + 同一位姿/随机种子协议 ✓。

## 9. v1.2 IL 第二轮：5k 数据 × 20/20 epochs（2026-09-26）

数据（历史数据集，已清理；同规模可由 `tools/collect_expert.py` 重采到 `datasets/BTC<时间戳>_expert5k`）：`runs/bc_expert_5k_v2`（**360,509 行入库 / 259,606 可训练**；train spec 前 5,000 条；指纹 v2-e2adf9319719）。
命令：
```bash
BC_DIR=runs/bc_expert_5k_v2 OUT=runs/train/il_5k_20x20 WM_EPOCHS=20 bash tools/train.sh            # Stage A
STAGE=B BC_EPOCHS=20 BC_DIR=runs/bc_expert_5k_v2 STAGE_A_OUT=runs/train/il_5k_20x20/stage_a \
  OUT=runs/train/il_5k_20x20_b2 bash tools/train.sh                                                 # Stage B
```
（聚类在 5k/60k 子采样上重拟合：两段式胜出，max_enrich 2.87×。）

### 9.1 开环（对比 §8 的 2k 版）

| 指标 | 2k 版 | **5k 版（本轮）** |
| --- | --- | --- |
| Stage A ADE（WM / 匀速） | 1.447 / 6.987 | **1.433 / 12.229**（6/6 horizon 胜）|
| Stage A val loss | 0.614 | **0.580** |
| 动作加权误差（首步） | 0.1201 | **0.0710**（median 0.036 / p95 0.251）|
| 逐 horizon 加权 MAE（h1→h6） | 0.125 → 1.240 m | **0.057 → 0.822 m** |
| 轨迹（全局） | — | mae 0.406 m / mse 1.028 m²（加权）；未加权 1.144 m |
| 分切片（brake / turn / curve） | 0.230 / 0.275 / 0.156 | **0.062 / 0.168 / 0.075** |
| router：CE / top-1 簇准确率 / NMI | 0.0038 / 0.265 / 0.081 | **0.0018 / 0.856 / 0.155** |

Stage B 逐 epoch：primary `traj MAE 0.772→0.406`、val `2.600→2.108`；specific `0.386→0.373`、val `1.944→1.922`。

### 9.2 闭环（50 条 val，LQR，配对同场景）

| 策略 | success | collision | off-road | rc | speed_ratio |
| --- | --- | --- | --- | --- | --- |
| 规则基线（冻结） | **0.82** | 0.10 | **0.06** | 0.925 | 0.734 |
| v1.2 IL 2k（§8） | **0.38** | 0.04 | **0.52** | **0.648** | 0.407 |
| **v1.2 IL 5k（本轮）** | **0.30** | 0.04 | **0.64** | 0.550 | 0.439 |

分主标签：curve / roundabout / uturn / tollgate 仍 **0%**；难度 easy 0.667 / medium 0.167 / hard **0.000**（比 2k 更差）。
真均速 4.33 m/s、crawl 4.8% 步 / 2.22 s、min-TTC 7.74、errors=0。

### 9.3 结论（重要，供决策）

**数据 2.5×、epochs 2×、开环全线变好（router 从 26.5% 升到 85.6%），但闭环反而变差（succ 0.38→0.30、off-road 0.52→0.64）。**
与 §8 与 `docs/forensics-2026-09-26.md` 的取证一致：**瓶颈不是数据量 / 训练时长 / 开环保真，而是 plan 缺少"车道绝对锚点 + 横向偏差回收"**
（失败仍集中在同一批几何，机制为"直道爬行段起漂 + 大转角转向通道塌缩"）。
→ 下一步应是 **E2b 扰动增广**（让监督目标包含"从偏移位姿回车道"）与/或 **E3 DAgger-lite**（治 crawl/OOD），而不是继续堆数据或加 epoch。

## 10. v1.2 IL 第三轮：5k × schema v2（mem-bank + router 软目标 + per-epoch val 切分）（2026-09-27）

数据：`datasets/BTC20260926-2343_expert5k`（**360,501 行入库 / 259,583 trainable**；40 分片并行采集、输出与单进程逐字节一致；
obs fingerprint `v2-6a4d5de3f669`）。相对 §9 的变化：**schema v2 观测**（mem-bank 编码）、**router 软目标**（聚类 CE/KL）、
**per-epoch 独立 val 切分**、以及周期 ckpt/resume 基础设施（见 10.3 第 4 条）。

命令（当前零参入口；本轮 Stage A 实际分两块执行：前 10 个 epoch 被 harness 会话轮换静默杀掉 → 用 `RESUME` 原地续跑）：

```bash
bash tools/train.sh                                                                # Stage A（20 epochs；data/epochs/batch 全在 config/train.yaml）
RESUME=runs/BTC20260927-0920_stageA/stage_a/ckpt_epoch010.pt bash tools/train.sh   # 被杀后从 epoch 10 续跑 11–20
STAGE=B bash tools/train.sh                                                        # Stage B（primary 10 + specific 10）
bash tools/test.sh    # 评测（50 条 val slice + lqr）；POLICY=baseline bash tools/test.sh 复现冻结基线
```

### 10.1 开环

| 指标 | §9 5k 第 2 轮 | **§10 5k 第 3 轮（本轮）** |
| --- | --- | --- |
| Stage A 末轮 | loss 0.580 / ADE 1.433 | loss **0.5933** / ADE **1.449**（匀速 12.229；6/6 horizon 胜）；FDE **3.023** vs 21.443 |
| Stage B primary 末轮 | MAE 0.406 m / val 2.108 | **0.404 m / 1.9729**（traj 1.8771 / action 0.0140 / router 0.0818）|
| Stage B specific 末轮 | MAE 0.373 m / val 1.922 | **0.365 m / 1.8461**（traj 1.7545 / action 0.0119 / router 0.0797）|
| 动作加权误差（首步） | 0.0710 | **0.0699**（mean 0.2330；median/p95 已按监控瘦身移除）|
| 逐 horizon 加权 MAE（h1→h6） | 0.057 → 0.822 m | **0.047 → 0.736 m** |
| router：CE / top-1 簇准确率 / NMI | 0.0018 / 0.856 / 0.155 | **0.0018 / 0.849 / 0.146** |

> **跨版本不可直接比较**：schema v2 改变了观测 scope，val 切分也改为 per-epoch 独立段。Stage B 末值 `action_mu_ds` = **3.461 m**
> （加权 3.524；专家 3.197 m）。router 负载仍集中（expert 0 占 **0.823**、expert 5 占 0.090，其余 ≤0.03）。

### 10.2 闭环（50 条 val，LQR，配对同场景）

| 策略 | success | collision | off-road | rc | speed_ratio |
| --- | --- | --- | --- | --- | --- |
| 规则基线（冻结；本轮复现） | **0.82** | 0.10 | **0.06** | 0.925 | 0.734 |
| v1.2 IL 2k（§8） | 0.38 | 0.04 | **0.52** | 0.648 | 0.407 |
| v1.2 IL 5k 第 2 轮（§9） | 0.30 | 0.04 | 0.64 | 0.550 | 0.439 |
| **v1.2 IL 5k 第 3 轮（本轮）** | **0.42** | 0.04 | **0.52** | **0.688** | 0.446 |

95% CI [0.294, 0.558]；n=50、errors=0；min-TTC 7.30 s；真均速 **4.49 m/s**；crawl 5.5% 步（3.11 s/episode）。
分主标签：split **1.00**、ramp_out 0.80、merge 0.60、straight 0.60、t_intersection 0.50、ramp_in 0.40、
intersection 0.25、roundabout 0.25；curve / uturn / tollgate 仍 **0.00**（tollgate 伴 50% 碰撞）。
难度：easy 0.722 / medium 0.278 / **hard 0.214**（§9：0.667 / 0.167 / **0.000**）；compound（22 条）0.273。

### 10.3 结论与下一步

1. **IL 目前最好一轮**：success **0.30 → 0.42**（亦高于 2k 的 0.38），rc **0.688** 新高，off-road 回到最优 **0.52**，
   collision 保持 0.04；**hard 难度 0.000 → 0.214**，split **0.60 → 1.00**。
2. 开环与 §9 基本持平（ADE 1.449 vs 1.433、val 0.5933 vs 0.580，且跨版本不可比），**闭环却明显变好** —— 再次说明
   瓶颈不在拟合质量，而在闭环执行/规划行为。
3. 剩余失败仍集中在 **curve / roundabout / uturn / tollgate（0–25%）**，与 `docs/forensics-2026-09-26.md` 机制取证一致
   （直道爬行段起漂 + 大转角转向通道塌缩；缺"车道绝对锚点 + 横向偏差回收"）→ 下一杠杆仍是 **E2b 扰动增广** 与/或
   **E3 DAgger-lite**（判据见 README §3.1），不是继续堆数据/epoch。
4. **基础设施首次全链路生效**：`ckpt_every=5` + `RESUME=` 在真实被中断（harness 会话轮换静默杀）后从 `ckpt_epoch010.pt`
   原地续跑 11–20（模型/优化器/RNG/val 状态恢复；含 Adam 设备迁移修复 `move_optimizer_state_to_device`）；
   后台长任务改为 `setsid` 分离 + 每阶段单日志（`[detach]`/`[exit]` 证据行）。
   **存活验证（2026-09-27）**：`setsid` 分离的金丝雀连续运行 **75.0 min**（>历史 ~64 min 杀点）零中断
   （101 ticks、最大间隔 45 s）→ 根因修复确认（harness 只清理自身作业进程组，独立会话不受影响）。
5. 产物：`runs/BTC20260927-0920_stageA`（A 1–10）+ `runs/BTC20260927-1019_stageA_p4`（A 11–20）、
   `runs/BTC20260927-1100_stageB`（B）、`runs/BTC20260927-1147_eval_lqr50`、`runs/BTC20260927-1149_eval_baseline50`、
   报告 `runs/BTC20260927-1150_ilreport_5k`（`il_report.md/json` + 11 张 PNG）。

## 11. 去聚类 + MoE 负载均衡 + DAgger 恢复轮（2026-09-27/28；含 2026-09-28 事故更正）

> **⚠️ 事故更正（2026-09-28）**：本节原 DAgger v1 的采集池**取自 eval500 的失败 spec**（`datasets/BTC20260927-2218_dagger1/report.json` 自证：`specs=env/specs/scenarios_eval500.json`、`provenance.from_eval=/tmp/opencode/dagger_pool80.csv`、`failed_ids=162`），且学生驱动模型错配（用 phase 1 `primary.pt` 而非 phase 2）→ **三个 DAgger 臂（0.040 / 0.330 / 0.420）的 500 集分数全部作废（train-on-test，不入任何结论）**；污染数据集/权重/采集池已于 2026-09-28 删除，作废评测目录保留并改名加 `_void` 后缀。**干净证据仅剩：阶段 B phase 1 = 0.274 → phase 2 = 0.396（CI 不重叠）**。
> 教训与措施：**训练数据生成只允许 train spec**；`tools/dagger_collect.py` 硬断言采集池 ∩ eval/val = ∅（命中即 `SystemExit`）；采集池生成器 `tools/make_dagger_pools.py` 分层固定种子并与 eval/val 校验；v2 协议见下。

**方案变更（去聚类）**：取消聚类监督（簇 CE/acc、二值门、hard 切全部删除），输出统一为
`primary + Σ_{i∈top2} g_i·expert_i`（残差 MoE，**全场景生效**）；路由**不再有监督标签**，只有
**Switch 式负载均衡 aux**（`α·E·Σ f_i·P_i`，α=0.01）；**Stage A 与 Stage B-phase1 都关闭 MoE**
（experts+gate 只在 B-phase2 训练）；phase2 数据 = 全量曝光 + 行权重（**worst-50% × 1.0 / 其余 × 0.1**，
有效质量 0.55），可叠加 DAgger 恢复行（权重 1.0；其 traj-aux 掩码为 0）。

命令（统一评测/训练口径：train=`datasets/BTC20260926-2343_expert5k` 全部 360,501 行；
val=`datasets/BTC20260927-1734_expert500val` 36,122 行；评测=`env/specs/scenarios_eval500.json`）：
```bash
STAGE=B bash tools/train.sh                                        # phase1 primary（MoE 关）→ primary.pt
RESUME=<primary.pt> bash tools/train.sh                            # phase2 specific（MoE 开 + 权重）→ final.pt
# phase 2b r{k}（v2 迭代轮；每轮独立新 run 目录）：
RESUME=<run>/stage_b/primary.pt EXTRA="--dagger-dir datasets/BTC<ts>_dagger_r{k}" bash tools/train.sh
```

**负载均衡实测**（phase 2）：train `load_cv ≈ 0.14`（8 专家各 11–14% ✓，无饿死）；`gate_entropy ≈ 0.70`；val `load_cv ≈ 0.31`。
对比旧"聚类 CE"路由（acc 0.65 < 多数类 0.95 的坍缩）✓。

**500 集闭环对照**（LQR，CI ±0.04；✓=干净证据，✗=已作废）：

| 臂 | 配置 | success [95% CI] | coll | off-road | rc | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| **阶段 B phase 1** | primary（评测关 MoE） | 0.274 [0.237, 0.315] | 0.014 | 0.702 | 0.579 | ✓ 干净 |
| **阶段 B phase 2** | +MoE（难例加权，无 DAgger） | **0.396** [0.354, 0.440] | 0.030 | 0.552 | 0.639 | ✓ 干净 |
| phase 2 + DAgger v1 | 合成轨迹进 traj-aux | 0.040 [0.026, 0.061] | 0.006 | 0.948 | 0.405 | ✗ 泄漏 |
| phase 2 + DAgger v1（no-traj） | 全局关 traj-aux | 0.330 [0.290, 0.372] | 0.070 | 0.596 | 0.611 | ✗ 泄漏 |
| phase 2 + DAgger v1（masked） | 逐行掩码 | 0.420 [0.378, 0.464] | 0.024 | 0.508 | 0.652 | ✗ 泄漏 |

分几何（仅干净两臂；n≈45/类，每格 95% CI ≈ ±0.14，方向性参考；按 Δ 排序）：

| 几何 | n | phase 1 | phase 2 | Δ |
| --- | --- | --- | --- | --- |
| ramp_out | 46 | 0.22 | 0.54 | +0.32 |
| uturn | 45 | 0.33 | 0.60 | +0.27 |
| split | 45 | 0.44 | 0.69 | +0.25 |
| ramp_in | 46 | 0.28 | 0.50 | +0.22 |
| merge | 46 | 0.22 | 0.39 | +0.17 |
| straight | 45 | 0.40 | 0.51 | +0.11 |
| t_intersection | 45 | 0.33 | 0.42 | +0.09 |
| intersection | 46 | 0.39 | 0.46 | +0.07 |
| curve | 46 | 0.02 | 0.02 | 0.00 |
| tollgate | 45 | 0.09 | 0.04 | −0.05 |
| roundabout | 45 | 0.29 | 0.18 | −0.11 |

**结论（更正后）**：
1. **MoE 专家分支有真增益（干净）**：phase 2 − phase 1 = **+0.122**（CI 不重叠）；off-road 0.702→0.552；8 专家均衡负载（无需任何人工标签）。
2. **DAgger v1 三个臂全部作废**（采集池泄漏 + 驱动模型错配）；教训与守卫见顶部更正块。
3. **工程发现保留**：DAgger 行只有首步动作标签时，其"常量动作外推"合成 `traj6` 进 traj-aux 会把 experts 教成过度转向（渐进劣化 0.396→0.304→0.040；val（专家分布）指标不变 → 坏在 plan/rollout 侧）；修复 = **逐行掩码**（DAgger 行不吃 traj-aux），已实现并被 v2 沿用（v2 行同样只有首步动作标签）。
4. 附带修复：val "动作误差"口径 bug（误用轨迹误差、与 `traj_mae` 雷同）已修复并加回归断言（`e011b6b`）。

**v2 迭代协议（2026-09-28 起，进行中）**：3 轮（r1/r2/r3）；每轮 =
① 用当前模型在 **train 切片 500**（`env/specs/scenarios_train_dagger_r{1,2,3}.json`：从 5k 覆盖的 5000 条 train spec 里按 `labels.geometry` 分层随机抽、固定种子、三轮互不重叠）闭环采集，
仅保留**失败 episode 的"终止前 10 s"窗口行**（策略帧 0.5 s ⇒ 20 帧；`collision`/`out_of_road`/`terminal` 计失败，`max_step`/timeout 不计）；
② 从冻结 `primary.pt` 重训 specific（多轮数据累积：`--dagger-dir` 可重复）→ 每轮独立 run 目录；
③ 评 `eval500`（**每轮都评**；轮数固定 3、不做早停，终模型 = r3）。
采集吞吐实测：12 specs / 12 workers / **10.2 s**（≈0.85 s/spec 墙钟；瓶颈=仿真步进；须按 worker 限制 OMP 线程数——v1 的 92 s/spec 初步归因于线程超订：6 worker × `torch_threads=14` > 20 核）。

产物（干净）：`runs/BTC20260927-2202_eval500_phase1`、`runs/BTC20260927-2209_eval500_phase2`；
训练目录 `runs/BTC20260927-1019_stageA_p4/stage_b`（`primary.pt` + `final.phase2.pt` + `metrics.phase2.json`）。
作废留档：`runs/BTC20260928-*_eval500_phase2b_v1*_void`（仅供事故追溯）。

## 7. 口径与注意事项

1. **tracker 语义**：`exact` = Stage B 语义（运动学精确执行预瞄，不引入动力学）；`lqr` = Stage C 闭环
   （含转向/纵向动力学与跟踪误差）。两者数字不可直接比较；正式 KPI 以 `lqr` 为准，`exact` 用于阶段验收。
2. **样本量**：10 条协议切片噪声大（Wilson CI 宽，success ±0.2 量级）；50 条 slice 更可信。
3. **`verdict: all_passed=False`**：评测输出的 verdict 是**完整 KPI 协议 vs 冻结基线**（含碰撞/off-road 的 ε 界），
   与阶段 B 的验收阈值（speed_ratio ≥ 0.6、off-road ≤ 0.5）是两套口径。
4. **非确定性**：少数脚本事件场景（cut-in 等）同种子 run-to-run 有微小差异（排除 `PYTHONHASHSEED` →
   MetaDrive 内部线程/时钟时序）；事件类 KPI 有噪声。
5. **BC 数据版本**：观测 scope 变更后必须重采（`obs_fingerprint` 守卫会在不匹配时告警）。
6. 所有 MetaDrive 运行必须经 `tools/venv-python`（设置 `LD_LIBRARY_PATH` 指向 venv 本地 glvnd）。
