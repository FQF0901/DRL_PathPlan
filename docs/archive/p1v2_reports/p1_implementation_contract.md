# P1 实现契约（P1 Implementation Contract）v1

> 依据：`docs/deepseekv4p1_argue.md` §7.10（P1 执行手册 v1，LOCKED @ round-8）。
> 本契约冻结 **P1-1–P1-3 期间的 trajectory-action 与 PPO log-prob 相关口径**；
> **P1-4 属独立 experimental scope**（§7.10-A5/§7.9-A5），其扩展须另立契约、不改本契约语义。
> 变更控制：任何修改 = 新版本 + 变更日志 + 影响评估（fail-closed；禁止静默漂移）。

## 1. 动作与计划语义（冻结）

- **动作** `a_t = (ds, dθ)`：每 0.5 s 策略步一个；`ds ∈ [0, 10]` m，`dθ ∈ [-0.6, 0.6]` rad。
  实现：`net/policy.py::ACTION_LOW/ACTION_HIGH`（squash 压缩）；语义"车体量"，对路径平移/旋转不变。
- **计划** `plan (6,2)`：6 步 `(ds, dθ)` 链（0.5 s/步）。`_rollout`（`net/model.py:693-832`）生成：
  `plan[0] = action_mu`（契约）；step≥1 = 策略头 μ 沿 latent 自回归。
  状态链 detach；**动作/位姿链可微**（traj 损失经 `arc_step/compose_pose` 回传）。
- **弧线积分单一真源**：`env/tracking.interpolate`（`(ds,dθ) → 10 Hz 位姿`）。
  训练/采集/评测凡言"链→轨迹"一律走该实现（`arc_step` 为其单步同构）。

## 2. 执行与跟踪协议（冻结）

- **闭环节拍**：每 0.5 s 决策 → tracker 10 Hz 子步执行。
- **tracker = LqrTracker**（`env/tracking.py`）：
  - 横向：沿参考弧长前视点 `Ld = clip(0.8·v + 2.5, 3.0, 15.0)` m 的 `[e_y, e_ψ]` 反馈；
  - 纵向：`a = speed_gain·(v_ref − v)`；`v_ref = 参考点弧长弦速（dt_sub=0.1s）× ref_speed_scale`；
  - `ref_speed_scale`（默认 1.0，P1-0B 引入）：**仅缩放 v_ref，几何/游标不动**；默认值逐位不变。
- **评测参考口径**：`plan`（默认；`plan[0]=action_mu` 的 6 步预览）或 `repeat_action`（A-hold 同构）。
- **训练 rollout 参考口径**：`repeat_action`（P0-1 后默认；`pipeline/trainer.py:663`）——执行只依赖 PPO 记账的随机变量 `a_t`。

## 3. PPO 记账契约（冻结；P1-4 前不变）

- **采样**：`raw ~ N(raw_mu, exp(log_std))`（`log_std ∈ [-5, 0]`，clamp）→ `a = low + span·σ(raw)`（squash）。
- **log-prob**：raw 空间高斯密度 + **squash 雅可比修正**（`net/policy.py::log_prob`）。
  更新期对**已执行动作**按同式重算 logprob（`pipeline/trainer.py:476`）⇒ on-policy ratio 可比。
- **rollout 存储**：buffer 动作域 = 2 维 `a_t`（`pipeline/buffer.py`）；`repeat_action` 参考由 `repeat(a_t, 6)` 构造（`trainer.py:6523`）。
- **可训梯度路径（P1-1–P1-3）**：仅损失（BC / 六步 traj / recovery 监督 / 几何·曲率·progress 项）→ 指定参数组（freeze recipe）；
  **不改变采样与 logprob 路径**。P1-4 的 action 空间扩展（如 mode+latent）必须独立 scope + 本文档新版本。

## 4. 评测协议（冻结）

- spec = `env/specs/scenarios_eval500.json`（sha256 见 `runs/p1_0/configs/specs_sha256.txt`）；`max_steps=1000`；`tracker=lqr`；`eval-reference=plan`（除非显式声明）。
- 加载纪律：**fail-fast**——`missing/unexpected/shape_mismatch` 三计数必须 0、`loaded_ratio=1.0`（记录于 `metrics.json::meta.ckpt_load` → 提取 `checkpoint_load_report.json`）。
- 臂配置必须显式绑定（例：`runs/p1_0/configs/p1a_pri512.yaml`），不得依赖临场换装共享 `config/model.yaml`。

## 5. P1-4 边界（条件门，未承诺）

- **4A**（条件执行）：base = P1-3 胜者（退化则 P1-2 胜者）；启动条件 = eval500 ≥0.60 + correctness probes 全 PASS + sampled action 确实进入最终计划与 tracker reference。
  实现 = "最小条件化改动"（尾段生成接 a_t）＋四类探针（后五步敏感性/有效控制维数/反事实单调性/执行一致性）＋**尾段状态反馈探针**（v7 Δψ 教训）。不过即停、不进 PPO。
- **4B**：不纳入本轮承诺；启动前须另出 design/distribution/buffer-migration 三文档并满足 §7.10-A7 六条件。

## 6. 事实锚点（供复核）

| 项 | 位置 |
|---|---|
| 动作界/采样/logprob | `net/policy.py`（ACTION_LOW/HIGH、sample、log_prob） |
| 6 步 rollout / plan[0]=mu | `net/model.py:693-832` |
| 训练参考口径/A-hold | `pipeline/trainer.py:659-671, 6516-6523` |
| 已执行动作 logprob 重算 | `pipeline/trainer.py:476` |
| tracker v_ref/前视 | `env/tracking.py`（set_reference、Ld 公式、ref_speed_scale） |
| 弧线积分单一真源 | `env/tracking.py::interpolate` |

**版本**：v1（2026-10-10，P1-0 期间冻结）。
