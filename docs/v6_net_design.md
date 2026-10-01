# v6 Net 设计冻结（交叉注意力头 / 去池化 / st_gnn 主路径 / nav 修正）

> **状态：冻结（2026-10-01）**。P1 实现以本文件为唯一架构规格；与现行代码/文档冲突处，以本文件为准。
> 关联：[`docs/v6_program_prereg.md`](v6_program_prereg.md)（阶段/Oracle 门/E3 硬闸）、[`docs/rl_reward_v5.md`](rl_reward_v5.md)（奖励 v5）。
> 上游依据：`.slim/deepwork/v6-net-retrain.md`（已锁决策）；现行基线：`docs/design-v1.2.md`、`.slim/deepwork/p2-contract.md`。未提及处沿用现行契约。

## 0. 冻结常量（不变项）

| 项 | 冻结值 | 现行落点 | 备注 |
|---|---|---|---|
| 主干隐藏维 d=H | **128** | **真实落点 = `config/model.yaml::hidden_dim` + `pipeline/stages.py::build_model`（按 config 构造）**；`net/encoders.py:35` 的 `H=96` 仅为代码默认（不采用） | 断言测试按 config/ckpt 口径（H=128），不得以 encoders 默认值建模型 |
| latent | **H=128** | `net/plan_head.py` 输出 | |
| `expert_hidden` | **256**（维持现状） | `config/model.yaml` 与 E-β′ ckpt 实为 256（`net/moe.py` 代码默认 192 不采用） | 用户裁定 2026-10-01：维持 256；P1 断言 manifest/参数量 = 256 |
| MoE | 8 experts / top-2 / primary 常开 | `net/moe.py`、`config/model.yaml` | 结构不变 |
| OD / LD 槽位 | 16 / 16 | `net/encoders.py` | |
| 历史帧 | 6 | `HISTORY_FRAMES` | |
| rollout | 6 步 × 0.5 s | `config/model.yaml::world_model` | |
| 动作 | `(ds, dθ)` 有界，dt = 0.5 s | `net/policy.py` | 不变 |
| encoder 冻结语义 | **取消** | — | 旧代（E-β′ 谱系）"encoders 冻结 / `design` allowlist"约束随**从 Stage A 重训**作废；新基座全链路重训，P4 RL 的冻结清单按新架构另行冻结（见本文件 §5 #7） |

## 1. A1+A2：Policy/Value 头 → 交叉注意力头（去池化）

### 1.1 现状（被替换）
- `net/model.py::_masked_mean`（L212–214）把 OD/LD 逐槽令牌**掩码均值池化**为单 token（`DrivingModel._plan`，L331–332），再交 `net/plan_head.py` 的 `fusion`（6H→H）。
- 问题：逐槽空间信息（左右邻车/车道线位置）被池化抹平；plan head 池化审计已列为待办（`docs/rl_stage_c_v4_report.md` §5.3）。

### 1.2 规格
新增 **CrossAttnHead**（策略/价值各一份，**不共享权重**；落点 P1 定：`net/policy.py` 扩展或新 `net/attn_head.py`；**模块命名须保 `model.policy.*` / `model.value.*`**，见 §5 #7）：

- **查询**：K = **1 个学习查询**（1×128）。Gate0 裁定：K=8 时 q1..q7 无监督信号 → 零梯度死参数。
- **键/值令牌集合**（≤ 37 token）：
  - 对象令牌：OD 16 槽（`encoded.od_ctx`）、LD 16 槽（`encoded.ld_ctx`）、others 聚合 1（`encoded.others_ctx`）；
  - 上下文令牌：`ego_ctx` 1、nav 1、signal 1，外加 **`plan_head` 融合 latent（1，作为 ego-context token）**——**保证 experts → policy/value 的条件通路**（MoE 残差经融合 latent 进注意力头）。Gate0 已裁定保留融合 latent，原"改用 raw `ego_ctx`"备选作废。
- **掩码**：`od_live` / `ld_live` / others（恒有效）/ `nav_mask` / `signal_mask` 拼接为 key mask；`ego_ctx` 与融合 latent 恒有效；全无效行输出 0，不得 NaN。`_context_tokens` 需同时返回 nav/signal mask（现仅返回 token）。
- **注意力**：1 查询、4 头（head_dim = 32）、pre-LN + 残差、**默认 1 层**（1–2 层可配）、d = 128；cross-attn → 输出投影 → 小 MLP（128→128→输出）。
- **输出**：
  - policy：`raw_mu (B,2)` + `raw_logstd (B,2)`，随后**仍走既有分布接口**（`net/policy.py` 的 `raw/squash/log_prob/sample`、sigmoid 有界压缩、clamp [-5,0] 全部不变；调用点清单见 §5 #5）；
  - value：`V (B,1)`。
- **参数预算**（精确算式，单头）：cross-attn 投影 4×(128²+128) = **66,048**；查询嵌入 1×128 = **128**；pre-LN 2×2×128 = **512**；小 MLP 128²+128 = **16,512**；输出层 policy 2×(128×2+2) = **516** / value 128×1+1 = **129** ⇒ policy ≈ **83.7k**、value ≈ **83.3k**（规格概称「**≈83k/头**」）。验收区间 **[60k, 100k]**；默认 1 层。
- **去池化**：`_masked_mean` **仅保留在 plan_head 融合路径**（`_plan` 的 od_pool/ld_pool → `fusion` → latent/MoE 输出），**不再作为 policy/value 输入**；policy/value 直接消费上述令牌集合。

### 1.3 验收（Gate1 证据）
1. **形状/参数量单测**（扩 `tests/test_net_shapes.py` 或新增 `tests/test_v6_attn_heads.py`）：B=1/8；全无效掩码行输出 0；层数 1/2；**K=1（单查询）**；每头参数落在 **[60k, 100k]**；**无死参数断言**（单次 backward 后头内全部参数有非零梯度）；模型按 config/ckpt 口径实例化并断言 **H=128**（不得用 `net/encoders.py` 默认 96 口径）；`action_mu`/`action_logstd`/`value` 形状与 dtype 与现行一致；`log_prob` 数值稳定（无 NaN/Inf）。
2. **信息充分性对照（左/右 cut-in 可区分）**：构造镜像配对样本（同 ego/地图，cut-in 目标分别来自左/右，其余令牌一致），断言 policy 输出差 `|Δmu|` 显著非零且方向与场景一致；同时记录池化版基线（预期不可区分）作为回归对照。样本数与阈值 P1 预注册。
3. **吞吐**：见 §4。

## 2. A3：st_gnn 上主路径（语义收敛）

- 现状：`net/model.py::_rollout`（L369）逐步执行 `st_gnn`（6 步，t0 锚定 + 直接多步预测，见 `net/st_gnn.py` docstring）；但 Stage C 的 collect/update 走 `rollout=False` cheap path——**真实位置 `pipeline/trainer.py:5610`（collect）/ `5918`（update）**（`pipeline/stages.py:3961` 仅是 wm-freeze 弃用注释，不是调用点）；cheap path 与主路径的 heads 输出逐位等价已被 `tests/test_net_shapes.py:543-563` 覆盖。
- **规格（Gate0 裁定）**：
  - **t0 帧单次 `st_gnn` 作为 encoder 一部分**：其对象级输出（OD/LD 节点特征）供注意力头消费；**collect 与 update 两条路径一致执行**；
  - **A1/A3 衔接口径（Gate1 复核接受，2026-10-01）**：t0 头的 OD/LD 令牌用上述单次消息传递的**对象级输出**；rollout 各步的头令牌用**当步 `enc` 的逐槽特征**（`_head_tokens` 缺省路径），**不额外跑消息传递**——t0 pass 已在 `encode` 内，rollout 的 6 次 `st_gnn` 语义/成本原样不动。若改为每步头都额外跑 MP，实测成本 **+68–71%**（远超 P3 abort 线 +25%），不接受；
  - **`no_grad` 副作用（Gate1 记录）**：t0 单次 pass 在 `no_grad` 下执行 ⇒ Stage B 的 **OD/LD encoder←action loss 通路变弱**（该 pass 的梯度不回传 encoder；WM 损失口径不变）；mini 重训须**监控 encoder 梯度范数与 BC 指标**，异常时按 Gate1 记录复核；
  - **6 步 rollout / WM 预测路径保持原样**（`_rollout` 与 Stage A 教师强制 `_wm_predictions` 的逐步 `st_gnn` 语义不变）；
  - **Stage C 中 `st_gnn` 冻结**（现无 WM loss，解冻无梯度，维持 W1 冻结）；**Stage A/B 按原 WM 损失训练**（`st_gnn` 在 A/B 可训，口径不变）；
  - **禁止**默认 cheap path 作为训练口径；cheap path 仅可作显式消融，且须证明与主路径逐位等价（或明确标注不可比）。
- **成本**：update 路径 ≈ **+1 次 `st_gnn`/minibatch**（bench ≈ **12.8 ms** 含 encode；`/tmp/opencode/v6_stageA_cost.md` §6）→ 重训预算按 +7–23% 计（§4）。
- 验收：collect/update 一致性测试（同一 batch 两路径输出逐位一致；**在 `tests/test_net_shapes.py::test_cheap_path_matches_full_forward`（L543-563）基础上扩展**，覆盖"t0 单次 `st_gnn` 已进入两路径"）；吞吐测量（§4）。

## 3. A4：nav 修正（P0-3）

- 现状缺陷：`_rollout` 全程复用 t0 的 `nav_token`/`signal_token`（`net/model.py`；`env/obs/nav.py::NavChannel.build`）；**且 `others` mem 的前 11 维（nav 子向量，`env/obs/others.py::NAV_DIM`）同样 t0 冻结**——证据与影响见 `docs/db44fefe-system-review.md` §P0-3（匝道/并线/路口/环岛的后续 5 步动作缺少正确任务条件）。
- **规格**：
  - **nav token 逐步重建、世界系重建**：将世界系 route/polyline、route progress、下一分叉语义纳入可 rollout 的地图状态；每步按新 ego pose 重建 nav features/token；
  - **others mem 的 nav 维同步重建**：`others_hist` 的 0..10 维（nav 子向量）随步重建（不得只改独立 nav token 而留 others 里的旧 nav）；
  - **teacher forcing 同步**：Stage A `_wm_predictions`（`pipeline/stages.py:1762`）与 trainer 版（`pipeline/trainer.py:2886`）按同一步进语义重建；
  - signal 暂为恒定 unknown 占位，但接口须支持时间推进状态（接入红绿灯后直接使用）。
- **依赖面（P1 必做）**：
  - **obs schema 变更（env lane）**：新增世界系 route/进度等键；`env/obs/__init__.py::OBS_SCHEMA_VERSION` 与 `obs_fingerprint` 同步升版；
  - buffer / `_assemble_obs_batch`（`pipeline/trainer.py:5801`）携带新键（collect → update 全链）；
  - 调用点接线：`net/model.py::_rollout`（L369）/`plan_step`（L354）；Stage A `pipeline/stages.py:1807`、trainer `pipeline/trainer.py:2925`（见 §5 行动清单）。
- 验收（同 P0-3）：设计 3 s 内经过 checkpoint/转向点的样例，比较"逐步重算 nav"与"t0 冻结 nav"的 plan，要求两者不同且前者被实际使用；route-command 切换单测纳入模型 rollout；**others-nav 同步重建断言**（逐步与 nav token 一致）。

## 4. 吞吐测量（Gate1 证据）

- 协议：固定 batch/micro/rollout-steps/设备（与 P3 重训口径一致），分别测 **collect 与 update 两路径**的 `steps/s` + wall-clock；对照 `docs/rl_stage_c_report.md` §3 的 v4 基线表（注明口径差异：cheap path vs 主路径）。
- 参考基线（v6 前口径，`/tmp/opencode/v6_stageA_cost.md` §6）：cheap path 1.83 ms vs 主路径 84.93 ms（@B=256）；`st_gnn` ×1 = 11.62 ms（占全前向 ~14%）；A3 的 +1 次 = **+12.8 ms 含 encode**。
- 判据：**报告制**（不预设通过阈值）；P3 重训预算与 batch 标定以实测为准；读数落 run manifest。

## 5. 现状差异与 P1 行动清单

| # | 差异 | P1 行动 |
|---|---|---|
| 1 | `expert_hidden`：**维持 256**（现状；代码默认 192 不采用） | P1 显式对齐 256 + manifest/参数量断言 |
| 2 | H=128 落点：`net/encoders.py:35` 的 96 只是代码默认 | 按 `config/model.yaml::hidden_dim` + `build_model` 构造；断言测试按 config/ckpt 口径（H=128） |
| 3 | `_masked_mean` 池化进入 policy/value 输入 | policy/value 改注意力头直吃令牌集合；池化仅保留在 plan_head 融合路径（§1.2） |
| 4 | `_context_tokens` 只返回 token、丢弃 mask | 改为返回 `(token, mask)`，供注意力掩码 |
| 5 | 分布 API：头输出 `raw_mu`/`raw_logstd` 后仍走既有分布接口 | 新头命名保 `model.policy.*`/`model.value.*`；需改的调用点：`net/model.py:443,481,485`、`pipeline/stages.py:1807`、`pipeline/trainer.py:2925`、`tests/test_net_shapes.py:485-498`、`tests/test_bc_pretrain.py:125,160`；**PPO 主路径 `pipeline/trainer.py:5928` 不受影响**（消费 `out["action_mu"]/["action_logstd"]/["value"]`） |
| 6 | cheap path 与主路径并存 | §2 接线（t0 单次 `st_gnn` 进 encoder、双路径一致）；一致性测试扩展 `tests/test_net_shapes.py:543-563`；**cheap path 真实位置 = `pipeline/trainer.py:5610/5918`**（非 `stages.py:3961` 注释） |
| 7 | Stage C 冻结清单悬空：`STAGE_C_DESIGN_PREFIXES`（`pipeline/trainer.py:869`）只认 `policy.`/`value.` 前缀 | 新头以 `model.policy.*`/`model.value.*` 命名（替换现有头内部实现）；P1 加**断言测试**：allowlist 冻结后新头全部参数可训、其余模块全冻结 |
| 8 | nav t0 冻结（含 others mem nav 维） | §3 接线：逐步重建 + others nav 同步 + teacher forcing 同步（`stages.py:1762` / `trainer.py:2886`）+ obs schema/buffer；调用点 `net/model.py::_rollout`（L369）/`plan_step`（L354）、`stages.py:1807`、`trainer.py:2925` |
| 9 | 旧 ckpt（含 E-β′）与新架构不兼容 | **不做权重迁移**；新基座从 Stage A 重训（E3 见 [`docs/v6_program_prereg.md`](v6_program_prereg.md) §6）；**E-β′ 回退 = 旧架构代码路径**（保留旧实现与 git tag `stageA-lock-v1`/`stageB-lock-v1`，不得覆写删除） |

## 6. 边界（不做）
- 不改动作空间/跟踪器/评测协议；不改 MoE 结构（8 experts / top-2 / 256）与 `ego_next` 监督；不新增 WM 损失；不做视觉/渲染。**唯一数据结构变更 = A4 的 obs schema（env lane，§3）**。
- 本文件数字为冻结值；任何变更需新预注册。
