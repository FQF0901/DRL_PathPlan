# Round-36 实现证据索引（回应 GPT `7dcf6bd`：10 条要求逐条映射）

> 背景：GPT 复核所见的 diff **仅含 `docs/deepseekv4p1_argue.md`**（文档提交），据此判 `IMPLEMENTATION_NOT_YET_VERIFIED`；但**代码重构已先行入库**（提交 `631ad45`，981 insertions），本文件按 GPT 的 10 条要求逐条给出**代码/运行证据**与**真实缺口**。

## A. 相关提交

| 提交 | 内容 |
|---|---|
| **`631ad45`** | `net/path_head.py` 791→1242 行（`StructuredPolicyHead` 重构：lateral+longitudinal+确定性 compose；每步调用；`step_geometry`；`ds_init`/`κ_max(v)` 数据驱动）、`net/model.py` +287（逐步调用接线/消融开关/`stop-grad`）、`net/__init__.py`、**`docs/architecture_contract.md`（148 行/10 节）** |
| `a85c4af` | AGENT.md v3（后被用户压缩为 56 行版 `0912cb9`） |
| 运行证据 | `runs/round36/k1/gate.json`（契约/梯度/标定/消融门）、`runs/round36/{j2_plan_res,j1_eval/*}.json`、`966 passed` |

## B. GPT 10 条 → 证据映射

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| 1 | 删除正式路径 `plan_override` | ✅（结构化路径） | `net/model.py:1395-1405` 仅 `oneshot` 构造；`1578-1581` 仅 oneshot 消费；结构化路径为逐步动作组装（契约门 `plan_equals_stacked_head_actions=True`）；**L-A 正在补 eval/deploy fail-fast** |
| 2 | 每次只生成当前一步 `(ds,κ)` | ✅ | `path_head.compose_step_action`；契约门 `dtheta_equals_kappa_near_times_ds=True` |
| 3 | `a0..a5` 同一 Policy 实例 | ✅ | 契约门 `structured_head_calls=6.0`、`action_generator_names=['path_head']`、扰 `policy.*` ⇒ plan 逐位不变 |
| 4 | WM 实际推进 5 次 | ✅ | 契约门 `structured_wm_advances=5.0`、`wm_calls_before_head=[0..5]`、扰 WM 第 3 次输出仅 Δa3..a5 |
| 5 | 辅助输出不进 LQR | ✅ | 辅助零化/随机扰动 ⇒ `plan/action_mu/traj` 逐位不变；`plan_from_aux`(rollout)=0 |
| 6 | `architecture_contract.md` | ✅ | `docs/architecture_contract.md`（唯一生成器/WM IO/六步调用图/辅助边界/同构声明/门） |
| 7 | **OD/LD 显式 anchor + ego-motion warp 最小实现** | ❌ **未实施** | 现状仅 `step_geometry` 静态近似（static 递减 / dyn·ttc·nav·speed hold）；**lane L-D 开工** |
| 8 | **运行时断言**（调用/账本/LQR 输入） | ◐ 部分 | 契约门**脚本级**已断言（K1 gate 全项）；**运行时**断言由 L-A 补（steer 已发） |
| 9 | hooks 关闭 bit-exact | ✅ | F2/H3 自证：默认关三方逐位、RNG 不变、ckpt/config sha 不变、前向 max\|Δ\|=0 |
| 10 | 调用图/参数账本/梯度账本 | ✅ | 契约 §3 调用图；allowlist 18 张量 + 冻结 164（K1/J1 gate）；梯度门：主/辅 loss→allowlist 18/18、`main_touches_frozen=[]`、WM loss→`st_gnn` 33 参、`wm_loss_touches_path_head/policy=[]`、`wm_action_stopgrad=True` |

## C. 门 → 现状

| 门（GPT §六） | 现状 |
|---|---|
| 架构门 | ✅ 已过（上述 #1–#5 契约门读数） |
| 坐标门 | ❌ 未到（warp 未实现；阈值需基线校准一次后锁定） |
| WM 门（4 项） | ◐ **未测**——先测现有 WM（horizon 增长 / tail vs frozen / h3–6 接口 / 物理预测 vs 恒速·静态基线）建立基线 |
| Policy 门 | ◐ 部分（T1 离线：越界 5.75% vs baseline 8.75%、κ_ok 0.966；G5 纵向/预测态一致性未测） |
| 六步/闭环门 | ◐ 相对门口径获接受（GPT §六）；闭环 G6 前置由 L-C 补（`--path-head` 装载 + `lane_anchor_xy` 注入） |

## D. 真实缺口（next）

1. **OD/LD warp 最小实现 + 5 项 hook**（L-D：`net/ego_warp.py` 独立模块先行，集成待 L-A 落地）；
2. 运行时断言（L-A，已 steer）；
3. 坐标门阈值基线校准（LD warp P90 / 同速前车 P90 / mirror）；
4. WM 门 4 项在现有 WM 上的**先测**（定"结构化 WM 训练"的验收基线，避免无基线重训）。
