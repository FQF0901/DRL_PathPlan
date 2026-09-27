# 训练监控指标清单（lane B 口径，2026-09-27）

本文是 `pipeline/monitoring.py` 的落盘契约。**保留集 = TB 显示集 = 可评审集（一一对应）**：
下表 tag 同时是（a）`monitor/metrics.csv` 长表行、（b）tensorboard 曲线、（c）`tools/il_report.py`
与 `tools/plot_curves.py` 的读取口径——不存在"只有 CSV/metrics.json 有、TB 没有"的曲线。

## 命名纪律

1. `loss/…` 只放**训练目标**（宏 batch 均值）；KPI 只放度量（ADE/FDE/err/acc/ce…）。
2. **任何 val 口径的曲线不得命名为 loss**：留出集统一 `val/` 命名空间
   （`val/od/…`、`val/ego/…`、`val/router/…`、`val/loss/planner/…` 是**留出损失**，允许）；
   `val_` 前缀机制已彻底废除。
3. 家族 = **同一 run 内 tag 后缀**：多线族逐 sub 写 `add_scalar("<main>/<sub>")`
   （TB 自动并成一张多线图）；**不再用 torch `add_scalars`**（会建 `<main>_<sub>/` sub-run 目录）。
4. router 只保留"选得准不准"：**硬标签**（聚类 top-1，sidecar `cluster_v<k>_assignments.npz`）
   的 CE/acc；软目标 KL/温度/专家混合权重路径已删除（lane B B3）。
5. **lane T 双分支**（2026-09-27）：router 家族拆成两套 KPI ——
   `router/cluster/{ce,acc}` = 8 路 hard 簇标签（**只算难例**，唯一生产监督 =
   `cluster_hard_<ts>.npz`，全量 `cluster_v2*` 仅历史工件）；
   `router/gate/{ce,acc,hard_rate}` = 二值难例门控（**全样本** CE；`hard_rate` = batch 难例
   加权占比，替代无意义的 `acc_majority`）。损失侧新增
   `loss/planner/specific/router_cluster` 与 `loss/planner/<phase>/gate`（`loss/…` 只放训练目标）。

回退：`TrainingMonitor(legacy_tags=True)` 或 CLI `--monitor-legacy-tags`（默认关）=
旧口径（全部 tag 原样落盘 + 旧多线分组），仅供对比/排查旧 run。
**旧 run 不重建**：消费方保留旧名回退（瘦身 v1 `wm/od/*`、`stageB/*`、`val_*` 与更早
`horizon/*`、`train|val/<phase>_bc_*`）。

## Tier-1 保留清单

层级：**E**=阶段/epoch 级聚合，**H**=逐 horizon（h1..h6 = 未来 0.5..3.0 s，步距 0.5 s）。

### Stage A（WM；KPI 在 val 子集上确定性前向）

| tag（canonical CSV） | TB 形态 | 物理意义 | 口径 |
| --- | --- | --- | --- |
| `loss/wm` | 标量 | WM 总损失（OD + ego_next + presence/entry 按权组合） | **train**，宏 batch 均值 |
| `loss/od` | 标量 | OD 未来位置预测损失项（加权 Huber + 角度） | train |
| `loss/ego_next` | 标量 | plan head "下一时刻 ego" 损失项 | train |
| `loss/presence` / `loss/entry` | 标量 | presence/entry BCE 损失项（id 轴） | train |
| `val/od/ade_m/h{k}` | `val/od/ade_m` → h1..h6 + `cv_h{k}`（匀速基线） | OD 预测 ADE（m） | val 子集，按 `od_mask×train_weight` 加权 |
| `val/od/fde_m/h{k}` | `val/od/fde_m` → h1..h6 + `cv_h{k}` | OD 预测 FDE（m，末点） | val 子集，加权 |
| `val/od/presence_auc` / `val/od/entry_auc` | 标量 | OD presence/entry 头 AUC | val 子集（加权池） |
| `val/ego/action/err_weighted` | 标量 | 首步动作加权误差（L1：ds/dθ 平均） | val 子集，`Σw·e/Σw` |
| `val/ego/traj/mae_m/h{k}` | `val/ego/traj/mae_m` → h1..h6 | WM rollout ego 6 点逐 horizon MAE（m） | val 子集，加权 |
| `val/ego/traj/fde_m` | 标量 | WM rollout ego 末点 FDE（m） | val 子集，加权 |

Stage A 的 loss/ADE/FDE/AUC/ego KPI 都在**验证子集**（`--val-frac` 留出 episode）上评估，
故 KPI 进 `val/` 命名空间；训练损失（`loss/*`）是训练 batch 均值。

### Stage B（planner BC；train / 留出成对）

| tag（canonical CSV） | TB 形态 | 物理意义 |
| --- | --- | --- |
| `loss/planner/<phase>/total` | `loss/planner/<phase>` → total/traj/action/router/router_cluster/gate | 相位总损失（train） |
| `loss/planner/<phase>/{traj,action,router}` | 同上 | 轨迹辅助 / 动作主 / router CE 分解（`router` 为 legacy alias） |
| `loss/planner/specific/router_cluster` | 同上 | specific 段 8 路 hard 簇 CE（**只算难例**；lane T） |
| `loss/planner/<phase>/gate` | 同上 | 二值难例门控 CE（specific 段；**全样本**；lane T） |
| `val/loss/planner/<phase>/{total,traj,action,router}` | `val/loss/planner/<phase>` → 同族 | 留出集同族损失 |
| `ego/traj/mae_m/h{k}` | `ego/traj/mae_m` → h1..h6 | 6 点 rollout 逐 horizon 加权 MAE（m，train） |
| `val/ego/traj/mae_m/h{k}` | `val/ego/traj/mae_m` → h1..h6 | 同上（留出） |
| `ego/traj/fde_m` / `val/ego/traj/fde_m` | 标量 | 末点 FDE（m，加权，train / 留出） |
| `ego/action/err_weighted` / `val/ego/action/err_weighted` | 标量 | 首步动作加权误差（L1，train / 留出） |
| `router/cluster/ce` / `val/router/cluster/ce` | `router/cluster` → ce/acc | 8 路 hard 簇硬标签交叉熵（**只算难例**；train / 留出） |
| `router/cluster/acc` / `val/router/cluster/acc` | 同上 | 8 路 hard 簇 top-1 准确率（难例口径） |
| `router/gate/ce` / `val/router/gate/ce` | `router/gate` → ce/acc/hard_rate | 二值难例门控交叉熵（**全样本**；train / 留出） |
| `router/gate/acc` / `val/router/gate/acc` | 同上 | 门控二值准确率（基线 = `hard_rate`） |
| `router/gate/hard_rate` / `val/router/gate/hard_rate` | 同上 | batch 难例**加权**占比（替代 `acc_majority`；解释门控 acc 用） |

`<phase>` = primary | specific；留出集 = 独立 val-dir（`--val-dir` / train-dir 同级
`*_expert500val`，**train-dir 全部行 + val-dir 全部行**）或 legacy 按 episode 比例切分
（缺 val-dir 时告警回退）。`acc_majority` 已删除（保留集 = TB 显示集）。

### 评测（`tools/test.sh` / `pipeline/eval_runner.py`）

| tag | 形态 | 口径 |
| --- | --- | --- |
| `eval/<metric>` | 标量（step=1） | `metrics.json::overall` 全量 KPI（success/collision/off_road/route_completion/速度/动作…） |
| `eval/by_primary/<metric>/<group>` | `eval/by_primary/<metric>` → 组名 | primary 几何分组视图 |
| `eval/by_difficulty/<metric>/<group>` | `eval/by_difficulty/<metric>` → 组名 | difficulty 分组视图 |

逐 spec 明细**不进 TB**（在 `episodes.csv`）。

## TB run 命名（`bash tools/tb.sh`）

run 名由代码生成（`pipeline/run_paths.tb_spec`；每个名字只取最新一个候选，避免重名；
无事件文件则跳过）：

| run 名 | 目录 |
| --- | --- |
| `train_stageA` / `train_stageB` | 最新 `runs/*/stage_a|b/monitor` |
| `eval_stageB` / `eval_stageA` | 最新 `runs/*eval*/monitor`（按 manifest 的 ckpt 归属） |
| `eval_baseline` | 同上（`policy: baseline` 评测） |

```bash
bash tools/tb.sh                 # 零参：自动 spec，端口 6006 起（占用自动 +1）
PORT=6007 bash tools/tb.sh       # 指定端口；日志 /tmp/opencode/tb_<port>.log
```

## 已移除 tag 及原因

| 已移除 | 原因 / 说明 |
| --- | --- |
| val 口径 loss 曲线（`val/od/loss`、`val/ego_next/loss`，原 `wm/od/loss`、`wm/ego_next/loss`） | 命名纪律：val 曲线不得叫 loss；训练损失只走 `loss/*` 标量（lane B B1） |
| router 软目标路径（`router/soft_ce|soft_kl|entropy|nmi|top1_cluster_acc`、`router/<phase>/expert_mix_weight`、`router_temperature`） | 监督改为硬标签 CE + acc/acc_majority；软分布物化/温度/专家混合权重全删（lane B B3） |
| `horizon/*/fde`、`cv_fde` 独立 tag（旧口径） | FDE 改为 `val/od/fde_m` / `ego/traj/fde_m` 的家族 sub |
| `horizon/*/traj_mse_m2` 曲线 | 轨迹误差主口径是 MAE/FDE（m）；MSE 只在 metrics.json 作为损失口径字段 |
| `horizon/*/valid_samples`、`valid_weight_sum`、`slot_count`、`*/n_updates` | 物理计数/贡献次数只是监控记账；训练日志/metrics.json 仍有样本量 |
| `slice/*`、`label/*`（动作误差切片） | 样本少、噪声大，不参与判定 |
| 动作误差 `median`/`p95` | 与加权 mean 冗余 |
| `expert_util_*`、`expert_mix_util_*`、`grad_norm_*`、计时/显存 | 诊断量只留在 stdout/metrics.json |
| `kpi/*`、`moe/*`、`scene_label/*` | 非 A/B 训练 KPI，未列入 Tier-1 |

## 工具兼容

- `tools/il_report.py`：新 tag 优先 + 旧名自动回退（lane B 前 v1 名与更早 `horizon/*`、
  `train/<phase>_bc_*`、metrics.json 快照）；已移除项显示「已移除」。
- `tools/plot_curves.py`：按上表精简面板（删 slice/label/expert-util/median/p95 面板），
  同样多线同图；旧 run 回退旧 tag，缺失自动跳过。
- `tools/tb.sh`：TensorBoard 入口（固定 run 名，见上节）。
- `tools/annotate_clusters.py`：生成 router 硬标签 sidecar（`cluster_v<k>_assignments.npz`）；
  `tools/collect_expert.py` 收尾自动调用一次。**Stage B 只读 sidecar，禁止在线重算**。
