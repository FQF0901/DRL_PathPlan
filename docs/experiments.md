# 实验记录（2026-09-25）

> 本文件把关键实验的数字与命令固化进仓库（`runs/` 为 gitignored，评审者无法离线查看）。
> 所有数字来自实际运行产物（`runs/*/metrics.json`、`runs/*/monitor/metrics.csv`、日志）。

---

## 0. 运行清单

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

基线参考：`runs/baseline_eval/val_reference.json` / `val_reference_by_primary.json`。

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

数据：`runs/bc_expert_2k_v2`（144,763 行入库 / 103,932 可训练；`obs_fingerprint` v2；scope 前 150）。
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

95% CI [0.259, 0.518]；n=50、errors=0、mean_speed 3.84 m/s、min-TTC 7.37 s、a_lat95 2.15。
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
| 弯道 / 环岛 / 掉头 / 收费站成功率 0% | **记录**：下一步主攻方向 |
| speed_ratio 0.407（基线 0.734，偏慢）| **记录** |

### 8.5 与 v1 的可比性说明

- 数据：v1 用 200 场景/1.08 万行（旧观测 scope、过滤后只留干净帧）；v1.2 用 2,000 场景/14.5 万行
  （scope 前 150、全帧 + 权重）；**开环数字不可直接跨版本比较**，闭环走同一冻结协议（50 条 val slice）。
- 模型：v1 为 GRU 时序 + 共享 MoE；v1.2 为 mem-bank + 分模态注意力 + plan-head MoE + ST-GNN；
  参数量 1,149,663（H=128）。
- 评测：两版都用 LQR 闭环 + 同一 50 条切片 + 同一位姿/随机种子协议 ✓。

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
