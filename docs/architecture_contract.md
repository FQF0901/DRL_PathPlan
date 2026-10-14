# Round-36 架构契约（K1：`StructuredPolicyHead` 唯一动作生成器）

- 受理：`docs/chatgpt5p6sol_review.md` §八/§九（GPT `ac78f76`，2026-10-14）；本 lane = K1
  （架构修正：`StructuredPolicyHead` 重构 + 架构契约）。
- 代码出处：`net/path_head.py`（`StructuredPolicyHead` / `step_geometry` / `compose_step_action`）、
  `net/model.py`（`set_path_head` / `set_path_head_ablation` / `_rollout` 逐步调用）。
- 修订关系：H1 的 `PathExecutionHead`（一次调用给 6 步 plan + `plan_override` 钉住 rollout 尾段）
  被判定为**事实上的第二个六步轨迹生成器**、绕过「决策 → WM 推进一步 → 同一头重新决策」闭环
  ⇒ 本 lane 重构为**每步调用**口径；一次性 6 步语义只保留为**诊断臂**（消融 ③，不接入生产）。

---

## 1. 唯一动作生成模块

| 项 | 口径 |
|---|---|
| 生成器 | `net.path_head.StructuredPolicyHead`（模型属性 `model.path_head`；参数前缀 `path_head.`） |
| 内部结构 | `lateral` 分支（锚 κ_ref + 残差 → `κ_near/mid/far`）+ `longitudinal` 分支（`[latent, 几何(κ_near, 外源几何(8))]` → 当前 `ds`，另出 5 步辅助 ds）+ `compose` 确定性几何层（`a=(ds, κ_near·ds)`；**无参数**） |
| 上游（特征） | `plan_head.fusion.2.`（PolicyBackbone 末层）→ `latent`；Encoder/WM 全冻 |
| 旧 `PolicyHead` | **仅服务**默认关路径（`enable_path_head=False`）与消融①；启用结构化头时其输出只作为 `policy_action_mu` 诊断（`action_logstd` 仍来自旧头，RL 未接线） |
| 静态断言 | `model.action_generator_names() == ("path_head",)`（`path_head_mode()=="structured"`）；`set_path_head_ablation` 只允许 `policy|structured|oneshot` |
| 动作口径 | 每步 `a_k = (ds_k, dθ_k = κ_near_k·ds_k)`；`ds∈[0,10]`、`dθ∈[−0.6,0.6]`（掩码 `ds_clamped/dtheta_clamped`） |
| 6 步 `plan` | 由**逐步动作组装**（`_rollout` 内 `torch.stack(actions)`；`plan[:,0] == action_mu`）；供 LQR/评测（`env/tracking.interpolate` → 30 点） |

## 2. World Model 输入输出（`st_gnn`；本 lane 不改 WM）

- **输入**：`z_ego/z_od/z_ld`（当前步 latent 状态）+ `node_mask` + `pose`（图位姿：ego = 想象累积位姿
  （detach）、OD = t0 位姿 + k·dt·v_t0、LD 静止）+ `step_index` + `od_anchor/ld_anchor`（t0 锚）
  + `action = a_{k-1}`（结构化模式下 **stop-gradient**，见 §5 梯度门）。
- **输出**：`z_ego_next/z_od_next/z_ld_next`（下一步状态；喂下一步头之前 **detach**）+
  `od_pred/ld_pred/od_presence_pred/od_entry_pred`（t0 帧物理解码，诊断/监督）。
- **调用次数**：6 步 rollout 内 `st_gnn` 调用 6 次；其中**前 5 次推进的状态用于重算下一步动作**
  （`a1..a5` 分别读 `z1..z5`），第 6 次仅产出末端预测（`od_pred[5]` 等，保持 6 步预测输出契约）。
  ⇒ 契约门读数：`structured_wm_advances == 5`、`structured_head_calls == 6`。

## 3. 六步 rollout 调用图（`net/model.py::_rollout`）

```
encode(obs):
  latent_0 = plan_step(z0)                      # z0 = t0 编码状态
  g0 = t0 静态几何（anchor/外源几何/presence）
  out0 = path_head(latent_0, g0)                # ← 头调用 #1 → a0 = out0.action = action_mu

_rollout(encoded, a0):                          # k = 1..6
  for k in 1..6:
    (a)  pose ← compose_pose(pose, arc_step(a_{k-1}))          # 想象位姿链（可微）
         progress ← progress + ds(a_{k-1}).detach()            # 想象路径进度（几何输入）
    (b)  nav/signal 重建（A4，世界系位姿）
    (c)  z_k, od_pred[k] ← st_gnn(z_{k-1}, pose_k, action=a_{k-1} [stop-grad])   # WM 调用 #k
    (d)  z_k ← detach(z_k)                                      # 状态链切断
    (e)  if k < 6:                                              # 最后一步后无动作
           latent_k ← plan_step(z_k, …)                         # 当前步特征（读 z_k）
           g_k ← step_geometry(t0 静态地图, pose.detach(), progress)   # 想象步几何重算
           a_k ← path_head(latent_k, g_k).action                # ← 头调用 #(k+1)（同一实例/同一参数）
  plan = stack([a0..a5], dim=1)                                 # 逐步组装（LQR 输入）
```

计数：头调用 6（a0..a5 全部来自同一 `path_head` 参数）、WM 调用 6（其中 5 次推进到 `z1..z5` 供决策）、
无第 7 次头调用；`plan_override` 仅在消融③（`oneshot`）存在。

## 4. 辅助头与主动作的边界

| 输出 | 归属 | 去向 | 门 |
|---|---|---|---|
| `ds_exec`（`ds[:,0]`）、`κ_near` | **主** | `a_k` → WM 推进 → `plan` → LQR/环境/评测 | — |
| `ds[:,1:6]`（未来 ds / 辅助预览 `plan_aux`/`progress_aux`/`aux_progress`） | 辅助 | 仅训练监督/诊断（J1 辅助损失）；**不进 LQR** | 契约门：`plan_from_aux==0`（rollout 链） |
| `κ_mid`/`κ_far` | 辅助 | 仅监督（receding 站位意图）/诊断 | 同上（`longitudinal` 分支**只吃 κ_near**） |
| `ds_init`/`kappa_limit`/`kappa_hard` | 诊断 | 标定读数（不参与动作计算路径） | K1 标定一致性检查 |

**辅助不变量（可证伪）**：把 `lateral[-1]` 的 κ_mid/κ_far 行与 `longitudinal[-1]` 的 1..5 列
（零化或随机扰动）后，`plan/action_mu/traj_xy` **逐位不变**（K1 门脚本 `aux_isolation`）⇒
辅助输出不构成动作/LQR 的第二条路径。

## 5. imagined 步几何（k≥1，`net.path_head.step_geometry`）

| 列 | k≥1 口径 | 备注 |
|---|---|---|
| 车道锚 | t0 锚折线刚体逆变换进 ego-k 帧 + 自车在 lane 线上的**垂足**重锚为首点 | 横向偏移/航向误差/`κ_ref`/控制站位前瞻随之重算 |
| `left/right_margin` | `W = left_0+right_0`（t0 静态地图）+ `lat_k` ⇒ `left_k = clamp(W/2 − lat_k, 0)`、`right_k = clamp(W/2 + lat_k, 0)`（MetaDrive `lat` 正 = 右） | t0 恒等；缺 margin ⇒ 0/present 0 |
| `static_distance` / `distance_to_target_lane_end` | `clamp(d_0 − progress_k, 0)`（障碍/lane 末端在 t0 静态地图中，按累计路径进度收缩） | **近似**：不做 lane 坐标精确还原（待 H2/H3） |
| `dynamic_distance` / `dynamic_ttc` | **保持当前步**（`dynamic_mode="hold"`；WM `od_pred` 变体未接入，见 caveats） | 允许分支 |
| `nav_progress` / `speed` | 保持 t0/当前步 | `speed` 无 WM 观测不算 |

几何输入使用 `pose.detach()`/`progress.detach()`：几何是**外源确定性输入**，动作 loss 的梯度
只经动作输出（不经几何重算二次计入）。

## 6. 数据驱动初始化 / 限幅（替代 H1 固定 3.5 m 与 `tanh×0.1`）

- 标定出处：`.slim/deepwork/p1_v2/round36/k1_calibrate_head_init.py` → `runs/round36/k1/calibration.json`
  （源 = DP-6 同一专家 npz `expert_bc.npz`，N=360,497，可用 265,801；DP-6 分片未提取 ⇒ 只读源 npz）。
- `ds_init = clamp(v_current·dt, bounds)`，`bounds = [max(0.5, 0.5·P10), min(10, 1.5·P90)]`
  = `[0.721513, 6.250048]`（专家 ds step0：P10 1.443 / P50 4.143 / P90 4.167 m）；
  分布校验：`v·dt` P50 = 4.156 vs 专家 ds P50 = 4.143（rel 0.3%）。
  纵向末层零初始化 ⇒ 初始 `ds = ds_init(v)`（数据驱动恒速先验，逐样本）。
- 残差限幅：`residual = clamp(tanh(raw)·κ_p95(v), ±κ_p99(v))`，`κ_max(v) = min(表值(v), a_y_budget/v²)`；
  表 = 速度分桶（节点 = 桶速度中位数）专家 `|κ|` P95/P99：
  `(1.943 → 0.1569/0.2137)、(5.092 → 0.1003/0.1420)、(8.330 → 0.0415/0.0830)、(11.1 → 0.0085/0.0100)`；
  `a_y_budget` = 各节点 `max(p·v²)` = 2.8797 (P95) / 5.7598 (P99) 供高速外推。
- **fallback + 标注**：专家 **残差**（`κ_expert − κ_ref(anchor)`）需 DP-6 锚标签（未就绪）⇒ 用
  `|κ_expert|` 代理（偏保守，含锚线曲率），产物字段 `residual_source="abs_kappa_proxy"`；
  `>12 m/s` 桶无数据 ⇒ 端点常数延伸（最近节点值）；标定整体不可用时回退常量
  `DS_INIT_FALLBACK=3.5`、`KAPPA_LIMIT_FALLBACK=0.1`（旧口径，显式标注，勿作生产默认）。

## 7. 消融脚手架（默认关；`DrivingModel.set_path_head_ablation`）

| 臂 | 模式 | 口径 | 用途 |
|---|---|---|---|
| ① | `policy` | 旧 `PolicyHead` + WM 逐步 rollout（= 默认关路径的行为） | 基线对照 |
| ② | `structured`（**默认**） | 结构化头 + WM 逐步 rollout（本文档 §3） | 生产候选 |
| ③ | `oneshot` | H1 一次性 6 步 plan（`plan_override` 钉住尾段） | **仅诊断上界，不接入生产** |

默认（`policy`）与 HEAD 逐位一致；`head enabled + structured` 才走逐步调用。

## 8. 训练与部署同构声明

- **同构**：训练/评测/部署都走 `DrivingModel.forward(obs, rollout=True)` → `_rollout` → 逐步
  `path_head` 调用 → 组装 `plan` → 固定 LQR（`env/tracking`，本 lane 未改）；`eval_runner`
  的 `--eval-reference plan` 与训练同一入口、同一张图、同一参数（无 teacher forcing、无
  一次性 6 步捷径）。`rollout=False` 的 cheap path（PPO/BC collect）**只**给 `action_mu` 与
  **辅助预览**（`plan_from_aux=1`，无 WM），**不得**作为 LQR 参考。
- **不同构之处（显式声明）**：① 消融①②③ 只用于对照/诊断，不参与部署；② `action_logstd`
  仍来自旧 `PolicyHead`（RL 分布口径未接线）；③ 训练期 target-lane dropout（默认 0）与
  训练侧 DAgger/采样权重不属于前向图；④ 部署端 `lane_anchor_xy`/`path_*` 由环境/数据侧
  提供（H2 规格），与训练同一键集。

## 9. 门定义（本 lane 自证；脚本 `.slim/deepwork/p1_v2/round36/k1_*.py`）

| 门 | 判据 | 证据键 |
|---|---|---|
| **G0 默认关逐位** | 默认关（M0 baseline / M0 cheap / K1 semantic）vs `git archive HEAD net` 逐键 `max|Δ| = 0`、键集一致、forward 不消耗 RNG；旧 ckpt strict load 0/0/0 | `gate/*::forward_parity_*`、`strict_load_reports` |
| **契约门（架构）** | 启用后：生成器数 = 1（`action_generator_names`/参数集合）；`a0..a5` 同头（`structured_head_calls == 6`，plan == 逐步动作堆叠，扰 `policy.*` 不改 plan）；每步读 `z_k`（trace + WM 第 3 步扰动 ⇒ a0..a2 不变 / a3..a5 变）；WM 推进 5 次（`structured_wm_advances == 5`）；辅助不进 LQR（`plan_from_aux == 0` + 辅助行扰动隔离） | `contract::*` |
| **梯度门** | 动作 loss → 仅 allowlist（`path_head.` + `plan_head.fusion.2.`）；WM 预测 loss → 仅 WM；辅助 loss → 仅辅助分支/allowlist（无旁路） | `gradient::*` |
| **消融门** | ①②③ 可切换（默认②）；③ 仅诊断（`one_shot_plan_used_as_action == False`） | `ablation::*` |
| **回归** | `tests/` 966 passed | `test_suite` |
| **闭环门（G1–G6）** | 不在本 lane（训练/评测 lane 执行；口径见 H1 manifest G1–G6） | — |

## 10. Caveats（K1 已知边界）

1. 残差限幅用 `|κ_expert|` **代理**（无 DP-6 锚残差）⇒ 限幅偏保守；DP-6 就绪后重跑标定覆盖
   代码常量（脚本自带「代码 vs 数据」逐位一致性断言，防漂移）。
2. 想象步静态距/末端距按**路径进度**收缩（近似）；`dynamic_distance/ttc`、`nav_progress`、
   `speed` 保持当前步。
3. 5 次 WM 推进 vs 6 次 `st_gnn` 调用：第 6 次是**末端预测**（保持 6 步 `od_pred` 输出契约），
   不参与动作生成（§2）。
4. 锚折线需覆盖 6 步前瞻（≈25 m + 控制站位 29.75 m）；过短时 `_interp_nodes` 端点常数延伸。
5. `dθ` 截断 `±0.6`、`ds` 截断 `[0,10]` 会截断极端 `ds×κ` 组合（掩码随输出返回）。
6. 旧 `PolicyHead` 的 `action_logstd` 仍参与分布式动作（RL 未接线；RL 口径由后续 lane 决定）。
7. 训练/部署端 `long_exog` 的动态列覆盖率由 H3 hook 审计（缺键 ⇒ 0 + presence 掩码）。
