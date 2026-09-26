# 训练监控指标清单（Tier-1 瘦身版，2026-09-27）

本文是 `pipeline/monitoring.py` 的落盘契约：**新 run 只记录下表 Tier-1 tag**
（用户口径："只关心 OD/EGO 的 loss 和 KPI + router 的 loss 和 KPI"）。
tensorboard 只写**成图所需**的写入：多线族 `add_scalars(main, {sub: value})`（同图多线）、
标量 `add_scalar`；canonical 单 tag 不再额外写一份。
CSV 长表 `<log_dir>/metrics.csv`（`step,tag,value`）保留全部 Tier-1 tag（供
`tools/il_report.py` / `tools/plot_curves.py` 读取）。

回退：`TrainingMonitor(legacy_tags=True)` 或 CLI `--monitor-legacy-tags`（默认关）=
旧口径（全部 tag 原样落盘 + 旧多线分组），供对比/排查。

## Tier-1 保留清单

层级说明：**E**=阶段/epoch 级聚合，**H**=逐 horizon（h1..h6 = 未来 0.5..3.0 s，步距 0.5 s）。

| tag（canonical CSV） | main → subs（TB 同图） | 归属 | 物理意义 | 单位 | 口径 | 层级 |
| --- | --- | --- | --- | --- | --- | --- |
| `wm/od/loss/h{k}` | `wm/od/loss` → h1..h6 | A | WM 对 OD（其他车）未来位置预测损失（加权 Huber + 角度项，验证子集确定性前向） | 无量纲（加权损失） | val 子集，`train_weight×wm_valid` 加权平均 | H |
| `wm/od/ade_m/h{k}` | `wm/od/ade_m` → h1..h6（模型）+ cv_h1..cv_h6（匀速基线） | A | OD 预测 ADE（位移平均误差）；`cv_h*` = 同一数据上的匀速外推基线 | m | val 子集，按 `od_mask×train_weight` 加权 | H |
| `wm/ego_next/loss/h{k}` | `wm/ego_next/loss` → h1..h6 | A | Stage A：plan head 预测"下一时刻 ego"的加权 Huber 损失 | 无量纲（加权损失） | val 子集，`train_weight×wm_valid` 加权 | H |
| `wm/loss` | 标量（单线） | A | Stage A WM 总损失（OD + ego_next + presence/entry 的按权组合，训练 batch 均值） | 无量纲 | train，宏 batch 均值 | E |
| `wm/presence_auc` | 标量（单线） | A | OD presence 头 AUC（未来帧同 id 是否仍在盒内） | 1（概率） | val 子集（加权池） | E |
| `wm/entry_auc` | 标量（单线） | A | OD entry 头 AUC（新 id 是否出现） | 1（概率） | val 子集（加权池） | E |
| `ego/traj/mae_m/h{k}` | `ego/traj/mae_m` → h1..h6 | B | Stage B：ego 6 点 rollout 轨迹逐 horizon 加权 MAE（`traj_xy` vs 专家 `traj6`） | m | train，`train_weight×balance_weight` 加权 | H |
| `val_ego/traj/mae_m/h{k}` | `val_ego/traj/mae_m` → h1..h6 | B | 同上，**留出集**（按 episode 留出，与 Stage A 同 seed/val_frac） | m | val，加权 | H |
| `ego/action/err_weighted` | 标量（单线） | B | 首步动作加权误差（显式 L1：ds/dθ 平均，不受 `loss_type` 影响） | 无量纲（ds 的 m 与 dθ 的 rad 平均后） | train，权重 = `train_weight×balance_weight`（`Σw·e/Σw`） | E |
| `val_ego/action/err_weighted` | 标量（单线） | B | 同上，留出集 | 无量纲 | val，加权 | E |
| `stageB/primary/loss_terms/{loss,traj,action,router}` | 一图 4 线 | B | primary 相位损失分解：总/轨迹辅助/动作主/ router 软目标 | 无量纲 | train，batch 均值 | E |
| `val_stageB/primary/loss_terms/{loss,traj,action,router}` | 一图 4 线 | B | 同上（留出） | 无量纲 | val | E |
| `stageB/specific/loss_terms/{loss,traj,action,router}` | 一图 4 线 | B | specific 相位损失分解 | 无量纲 | train | E |
| `val_stageB/specific/loss_terms/{loss,traj,action,router}` | 一图 4 线 | B | 同上（留出） | 无量纲 | val | E |
| `router/soft_ce` | 标量（单线） | B | router 对聚类软目标的 CE（温度软化的 softmax 分布） | 无量纲（nats/样本） | train，样本均值 | E |
| `router/soft_kl` | 标量（单线） | B | 同上，KL（软目标 ∥ 预测分布） | 无量纲（nats/样本） | train | E |
| `router/top1_cluster_acc` | 标量（单线） | B | router 预测 top-1 与聚类硬标签（软目标 argmax）一致率 | 1（比例） | train | E |
| `router/nmi` | 标量（单线） | B | router top-1 与聚类标签的归一化互信息 | 1（[0,1]） | train | E |
| `router/entropy` | 标量（单线） | B | 门控分布平均熵（越接近 0 越确定） | nats | train | E |
| `val_router/soft_ce`（soft_kl/top1_cluster_acc/nmi/entropy 同族） | 标量（单线） | B | 上述 router KPI 的留出集口径 | 同上 | val | E |
| `router/primary/expert_mix_weight/e{i}` | `router/primary/expert_mix_weight` → e0..e7 | B | primary 相位 8 专家 top-2 混合权重均值（`expert_weights`） | 1（8 线和 = top-2 选择率之和） | train | E |
| `val_router/primary/expert_mix_weight/e{i}` | `val_router/primary/...` → e0..e7 | B | 同上（留出） | 1 | val | E |
| `router/specific/expert_mix_weight/e{i}` | `router/specific/expert_mix_weight` → e0..e7 | B | specific 相位专家混合权重 | 1 | train | E |
| `val_router/specific/expert_mix_weight/e{i}` | `val_router/specific/...` → e0..e7 | B | 同上（留出） | 1 | val | E |

### Stage A 的 train/val 说明

Stage A 的 loss/ADE/ego_next/AUC 都在**验证子集**（`--val-frac` 留出 episode）上做确定性
前向，因此没有 `train_`/`val_` 双份；`wm/loss` 是训练 batch 的总损失。
LD（未来车道/可行驶区域）监督已在前一轮移除（`ld_loss=removed`），本轮不新增 LD 曲线。

## 已移除 tag 及原因

| 已移除 | 原因 / 说明 |
| --- | --- |
| `horizon/*/fde`、`cv_fde` 独立 tag | 用户只看 ADE；FDE 与 ADE 冗余。匀速基线改为 `wm/od/ade_m/cv_h*` 的 sub（同一图 12 线） |
| `horizon/*/traj_mse_m2` 曲线 | 轨迹误差主口径是 MAE（m）；MSE 数值仅在 metrics.json 保留为损失口径字段（`bc_traj_mse*`） |
| `horizon/*/valid_samples`、`valid_weight_sum`、`slot_count`、`*/n_updates` | 物理计数/贡献次数只是监控记账，不参与判定；训练日志/metrics.json 仍有样本量 |
| `slice/*`（`slice/brake|turn|curve/action_err`） | **按机动类型的动作误差切片**（急刹 ds<阈值 / 急转 |dθ|≥阈值 / 弯道 on_curve）：样本少、噪声大，用户不关心 |
| `label/*`（`label/on_curve` 等逐标签切片） | **规则标签下的动作误差切片**（数据集 `label_names` 的 0/1 标签分组）：同上，不参与判定 |
| 动作误差 `median`/`p95` | 与加权 mean 口径冗余（L1 回归看加权均值），切片/分位数全部清理 |
| `expert_util_*`、`expert_mix_util_*` | 专家利用率（top-1 占比 / top-2 选择率）是负载均衡诊断，不是 loss/KPI |
| `grad_norm_*` | 梯度范数探针（Stage A 的 plan_head/router/st_gnn 与 PPO 的 grad_norm）只在 metrics.json/stdout 留训练诊断 |
| `train/update_timing_s/*`（data/fwd/bwd/it_s）、`vram_peak_mb` | 计时/显存只留在 stdout 日志与 metrics.json（monitor 不再记录） |
| `kpi/*`、`moe/*`、`scene_label/*` | Stage C episode KPI / MoE 路由窗口 / 场景标签统计：非 A/B 训练 KPI，未列入 Tier-1 |
| `train/cluster_version_num`、`train/cluster_k`、`train/cluster_soft_targets` | 聚类版本/软目标开关只是阶段元数据，只进 metrics.json 元数据（`cluster_version`/`cluster_k`/`cluster_soft_targets`） |
| `train/per_horizon/*` | 已由 `wm/*` / `ego/*` 族取代（避免逐 horizon 重复两份） |
| `cv_ade`/`cv_fde` 常量标量 | 匀速基线是逐 horizon 量（`cv_h*`），聚合常量无信息 |

`metrics.json` 同步瘦身（`pipeline.stages._slim_phase_result`）：`bc_action_err_slice_*`、
`bc_action_err_label_*`、`bc_action_err_median/p95`、`bc_router_expert_{util,mix_util}_*`、
`slices`/`labels` 快照不再写入；**保留**损失口径字段（`bc_traj_mse*`、`bc_traj_mae*`、
`bc_action_err_weighted_mean`、`bc_router_soft_*` 等）与聚类元数据（`cluster_*`）。

## tensorboard 分组（`add_scalars` 语义）

torch `add_scalars(main, {sub: value})` 为每个 sub 建独立 sub-run 目录
`<log_dir>/<main 把 / 换成 _>_<sub>/`，文件内 tag = `main`，标量面板里同 tag 多 run
即一张图多条线。本仓库的分组（与 `tools/plot_curves.py` 的 PNG 面板一致）：

| main | subs |
| --- | --- |
| `wm/od/loss` | h1..h6 |
| `wm/od/ade_m` | h1..h6, cv_h1..cv_h6（12 线） |
| `wm/ego_next/loss` | h1..h6 |
| `ego/traj/mae_m` / `val_ego/traj/mae_m` | h1..h6 |
| `stageB/{primary,specific}/loss_terms` / `val_stageB/...` | loss, traj, action, router |
| `router/{primary,specific}/expert_mix_weight` / `val_router/...` | e0..e7（每相位一图 8 线） |

同族 <2 个 sub 时不写（避免单点噪声，数值仍在 CSV）；标量 tag（`wm/loss` 等）写单线
`add_scalar`。

## 工具兼容

- `tools/il_report.py`：新 tag 优先 + 旧 run tag 自动回退（旧 `horizon/*`、
  `train/<phase>_bc_*`、`val/*`、metrics.json 快照）；已移除项显示「已移除」。
- `tools/plot_curves.py`：按上表精简面板（删 slice/label/expert-util/median/p95 面板），
  同样多线同图；旧 run 回退旧 tag，缺失自动跳过。
- 旧 run 的 CSV 不受影响；需要旧口径新 run 时用 `--monitor-legacy-tags`。
