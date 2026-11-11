# Memory Bank + Temporal-Spatial Attention + Policy + Ego-Conditioned Latent WM

> 状态：**owner-approved working contract — RL implementation**
> 日期：2026-11-10
> 范围：Real/Imagine Memory Bank、BankWindow、TemporalSpatialBlock、PolicyHead、OD/LD direct geometry decoder、Ego-conditioned latent WM、BC/DAgger 训练、real-env group-relative PPO、WM 独立训练、joint 6-step trajectory consistency；RL 合同已冻结，代码实现中。
> 当前基线：exact DAgger r3，val500 success **66.4%** / collision **17.8%**（见 `docs/baseline_exact.md`）。旧 `docs/baseline_r3.md` 是 with safety fallback 的历史记录。
> RL gate：**test500 success ≥72.8%（one-shot final gate）**；val500 只做 diagnostic；train_2000 固定 500-scene holdout 做 selection/rollback。只有一个 hard gate，不设 gate 2/3。
> 明确不做：不做 failure-window/mid-episode reset；不做独立 cost/Lagrangian（所有惩罚进一个规则 reward model）；WM 不参与 RL；§12 的 RL 合同已冻结，实现和 debug 不改方向。
> 原则：已经走通并接受的现状写成架构与设计细节；已定的 RL/WM/joint 方案集中在 §12；未实现的代码路径明确标注 planned，不冒充现状。

---

## 0. 结论摘要

主线：

```text
单帧 observation
  → online encoder
  → RealMemoryBank.append(FrameRecord)
  → BankWindow：固定 6 帧、固定 16 slot
  → TemporalSpatialBlock
       temporal: 同 stable_id 历史 + frame-level delta_pose
       spatial: 当前 OD/LD + ego alignment token；不含 nav
  → h_t（TSB per-entity context，同时服务 policy 和 WM）
       ├─ policy head
       │    ego/nav query OD/LD
       │    → policy_action
       └─ EgoConditionedLatentWM
            r_z,t = WM(h_t, action_t)
            z_hat_{t+1} = z_hat_t + r_z,t
  → direct OD/LD geometry decoder
  → ImagineMemoryBank.append(z_hat, geometry, pose)
  → 6 个 policy / 6 个 transition
```

已接受的关键决策：

1. `env.obs.stack_history` 退出主路径；历史只由 Bank 管理。
2. Real/Imagine Bank 同 schema、同构；`source` 和 `rollout_depth` 区分真实/想象。
3. OD/LD 都使用 **固定 16 slot**；slot 只是存储位置，`stable_id` 才是身份。
4. Temporal attention 只允许同一 `stable_id` 的历史参与；slot 复用时旧历史必须 mask。
5. **`z` 是 frame-dependent full-state latent**：包含相对位姿、内容、类型等；显式 geometry 仍单独存储，作为监督/诊断/decoder target。
6. **不再做 entity geometry projection**：TSB 和 decoder 都不把上一帧 geometry 投影到当前帧；历史对齐由 `delta_pose(t -> query)` 完成。
7. TSB spatial 只接收当前 OD/LD 和 **ego alignment token**；**nav 不进入 TSB**，nav 只进入 policy。
8. Policy head 使用 ego/nav 对 TSB 输出的 OD/LD 做 cross-attention；policy hidden 不进入 WM。
9. WM 预测 OD/LD 的 latent residual `r_z`；`z_hat` 累积；OD/LD 都有单帧 target encoder 的 content target；当前不启用 EMA。
10. Geometry 使用 **direct absolute decoder**：OD 输出 5D，LD 输出 10D；没有 base projection、没有 geometry residual。
11. 没有独立 risk head，也没有 value head；Plan/aux；collision/offroad/timeout 进 aux。
12. 当前 scope 是当前可见目标、固定 top-16、短时 rollout；不考虑短时遮挡，只保留 `valid`。
13. Rollout 是 **6 个 policy / 6 个 action / 6 个 imagined transition**；t6 是 terminal state。
14. 所有 canonical 路径的 student/lift action（BC/DAgger lift 行、eval、PPO）exact tracking：`requested_action == executed_action`；不做 `v_next` projection、safe-brake、clip 或额外平滑。DAgger expert replay 行按设计执行 expert action。非有限/不可执行动作直接终止或报错。LQR 是后续阶段，才允许 requested/executed 分离。
15. RL 只用 real env + group-relative PPO；WM 不参与 RL。所有奖励/惩罚进一个规则 reward model，不单独维护 cost；per-step active-set group advantage；train 1500 / holdout 500 / test500 one-shot gate；唯一 hard gate 是 test500 success ≥ **72.8%**。WM 独立拟合 env；joint 阶段用 6-step trajectory consistency + KL 让 policy/WM 自洽。

### 0.1 当前基线（BC/DAgger）

当前 accepted baseline 是 **exact DAgger r3**（`runs/bc_dagger_exact_r3_261107031440/best_closed_loop.pt`，epoch 15）。详细的 run recipe、数据量、val500 指标和失败分布放在 `docs/baseline_exact.md`；本节只保留架构相关事实。

- canonical policy：raw Bank → TSB → ego/nav planner；不使用 projected history，不含 value/risk head。
- RL 阶段完全不使用 WM；WM 独立训练并拟合 env；joint 阶段才通过显式 `imagine` 接口把 WM 的 6-step 轨迹与 policy 对齐。
- BC/DAgger closed-loop selection（历史 baseline）：每 5 epoch 跑 100-scene subset，最终跑完整 val500；score 为 success-first lexicographic。
- RL data split：`train_2000` seeds 1000..2499（1500 train）+ seeds 2500..2999（500 holdout）；val500 只做 diagnostic；test500 只做 one-shot final gate。
- `test500` 不参与调参；raw `history_mode=1` 不回退。

---

## 1. 术语和坐标

### 1.1 frame

- `ego frame`：当前时刻自车坐标系；
- `t0 frame`：rollout 起点的自车坐标系；
- `world frame`：world pose 不直接作为网络输入；只用于计算 frame-level `delta_pose` 和诊断/可视化。

每个 `FrameRecord` 保存该帧 `ego_pose`。
**当前版本不对历史 geometry 做投影。** TSB 使用 `ego_pose` 计算：

```text
delta_pose(t -> query) = inverse(ego_pose_t) ∘ ego_pose_query
```

作为历史 token 的 frame alignment 信息。

### 1.2 state latent / geometry / metadata

```text
z（Bank content）: frame-dependent full-state latent
  OD：类别、外观、行为语义、相对位姿等
  LD：车道/道路结构、类型、拓扑、相对位姿等
  注意：z 不要求跨 frame 旋转不变

geometry: 显式 anchor，单独存储
  OD：[x, y, yaw, vx, vy]
  LD：10D polyline
  作用：监督、诊断、decoder target；不参与 TSB projection

metadata:
  stable_id / slot / source / rollout_depth / timestamp / encoder_version / valid
```

**TSB 不从 geometry 做 SE(2) 投影。** Geometry 只作为 target 和 decoder 输出。

### 1.3 depth

```text
rollout_depth
  REAL     = 0
  IMAGINED = 1..6
```

用途：区分真实/想象、控制 FIFO/advance horizon（<=6）、禁止 imagined 回写 RealBank。当前训练没有 depth weighting；`probe_wm` 报告 per-depth content/geometry 误差（见 §12.5/§13）。

---

## 2. Bank record schema

### 2.1 FrameRecord

```text
FrameRecord
  episode_id: int/str
  frame_id: int
  timestamp: float
  ego_pose: [3]            # x,y,yaw in world/t0-consistent frame
  source: REAL | IMAGINED
  rollout_depth: int       # real=0, imagined=1..6
  parent_frame_id: int     # -1 for real
  action_from_parent: [4] | None
  encoder_version: int
  od: tuple[EntityRecord, ...]   # OD fixed-slot records
  ld: tuple[EntityRecord, ...]   # LD fixed-slot records
```

### 2.2 EntityRecord

```text
EntityRecord
  slot: int                # 0..15 storage bucket
  stable_id: int           # semantic identity
  node_type: OD | LD
  content: [H]             # state latent z；frame-dependent
  geometry: [G]            # explicit local geometry
  type_id: int             # OD/LD type，继承 t0，不预测
  valid: bool              # padding/NaN/无效 slot 为 False
```

### 2.3 字段语义

| 字段 | 含义 | 谁维护 |
|---|---|---|
| `slot` | 固定 16 槽中的存储位置，不是身份 | Bank/数据管线 |
| `stable_id` | 实体身份，用于跨帧关联 | env.obs stable ID |
| `valid` | 该 tensor 是否可用于 attention/loss | 数据管线/模型 |
| `source` | REAL 或 IMAGINED | 写入方 |
| `rollout_depth` | 0 或 1..6 | 写入方 |
| `type_id` | OD/LD 类型 | t0 继承；不做未来预测 |

`od_anchor` 与 `od_taxonomy` 是 encoder 输入，不是 audit；`od_age` 当前恒为 0，是 dead feature；`ld_taxonomy` 当前没有真实数据，`_type` fallback 到 road type。

**只保留 `valid`。** 没有 `present` / `alive`。
`valid=1`：当前帧选中或 imagined 记录有效；`valid=0`：padding、丢弃、无效 slot。

### 2.4 固定 16 slot

```text
OD: 16 slots
LD: 16 slots
Bank 只存 OD/LD；Nav/Ego 不进入 Bank。
```

ROI：

```text
前方 80 m
后方 20 m
左右各 25 m
```

OD/LD 先 ROI filter，再按到 ego 距离排序，取最近 top-16；不足 padding `valid=0`；超过丢弃最远。距离相同用 `stable_id` 稳定排序。

时间 attention 的 mask：

```text
temporal_valid[t, n] = valid[t, n]
                       & valid[t-1.., n]
                       & frame_valid[t]
                       & (stable_id[t, n] == stable_id[历史, n])
```

slot 复用时清空旧 content/geometry/type，写入新 stable_id，旧历史 temporal mask=0。

valid OD/LD stable ID 必须非零且同帧唯一；canonical 生成端、loader、BankWindow 构建和模型 fast path 都 fail-fast。skipping/padding slot 才允许 ID=0。生产路径不存在 slot-ID fallback。

---

## 3. RealMemoryBank / ImagineMemoryBank

### 3.1 同构

两者使用同一套 FrameRecord/EntityRecord schema；区别只在 `source`、`valid`、`depth` 和写入权限。

### 3.2 RealMemoryBank

```text
容量：6 帧
写入：真实 observation
source=REAL
rollout_depth=0
```

- 上游是当前帧 online encoder；
- 不接收 WM 输出；
- 不接受 imagined 记录；
- episode/scene 切换 hard reset；
- append 默认 detach；
- 不保存 target encoder 的 `z*`。

### 3.3 ImagineMemoryBank

```text
snapshot = RealBank 的 6 帧副本
real prefix: source=REAL
新帧: source=IMAGINED
rollout_depth: 1..6
```

- 从 RealBank snapshot；
- 每步 append 一个 imagined FrameRecord；
- 不修改 RealBank；
- rollout 结束销毁/clear；
- 不把 imagined 回写 RealBank；
- FIFO 容量 6；imagined depth 最大 6。

### 3.4 encoder / target encoder

RealBank content 来自：

```text
MetaDrive current observation
  → online single-frame encoder
  → z（full-state latent）+ explicit geometry + stable ID
  → RealBank.append
```

Target encoder：

```text
future real observation
  → independent frozen target encoder（当前不启用 EMA）
  → z*（OD + LD；stop-grad）
```

- target encoder 与 online encoder 同结构、不同实例；
- 当前 WM/joint 训练使用 frozen copy；未来 online encoder 参与训练时再评估 EMA；
- `z*` 只进入 WM loss，不进入 RealBank/policy；
- OD/LD 都有 content target；
- target encoder 不做 geometry projection。

---

## 4. BankWindow API

```python
class BankWindow:
    od_content: Tensor       # [B,T,16,H]  z_od
    od_geometry: Tensor      # [B,T,16,5]  显式 OD geometry
    od_type: Tensor          # [B,T,16]
    od_valid: Tensor         # [B,T,16] bool
    od_stable_id: Tensor     # [B,T,16]
    ld_content: Tensor       # [B,T,16,H]  z_ld
    ld_geometry: Tensor      # [B,T,16,10] 显式 LD geometry
    ld_type: Tensor          # [B,T,16]
    ld_valid: Tensor         # [B,T,16] bool
    ld_stable_id: Tensor     # [B,T,16]
    ego_pose: Tensor         # [B,T,3]
    source: Tensor           # [B,T] int64
    rollout_depth: Tensor    # [B,T] int64
    frame_valid: Tensor      # [B,T] bool
```

### 4.1 构造

- 不足 6 帧：左侧 padding，`valid=0`；
- `content` 保存各帧自己的 `z`，**不做旋转/投影**；
- `geometry` 保存各帧自己的显式 geometry，**不做投影**；
- `ego_pose` 保存各帧 pose，用于 TSB 的 `delta_pose(t -> query)`；
- `stable_id` 按帧保存；
- `valid` 按帧保存；
- `source/depth` 保留；
- 同一 slot 的历史只有 `stable_id` 相同且 `valid=1` 时进入 temporal attention。

### 4.2 批处理

```text
B episodes
  → [B,T,N,H] z
  → [B,T,N,G] geometry
  → [B,T,N] masks / stable_id
  → [B,T,3] ego_pose
```

BankWindow 本身不要求 batch；训练 collate 支持 batch。

---

## 5. TemporalSpatialBlock

### 5.1 结构

```text
input: BankWindow（OD/LD）
       + current ego（仅用于 spatial alignment token）
       + 每帧 delta_pose(t -> query)（仅用于 temporal frame alignment）

per-entity temporal attention
  query = current entity token
  key/value = same stable_id 历史 entity token
  token_t = z_t + FrameEmbed(delta_pose_t)
  mask = valid & same stable_id

current-frame spatial attention
  tokens = OD + LD + ego_alignment
  **不含 nav**
  OD ↔ LD ↔ ego
```

TSB 输出：

```text
z_od: [B,16,H]
z_ld: [B,16,H]
```

### 5.2 token 内容

```text
entity token_t = z_t + FrameEmbed(delta_pose_t)
```

- `z_t` 是 frame-dependent full-state latent；
- `FrameEmbed(delta_pose_t)` 是 frame-level ego 相对运动，不是 entity pose；
- **不加入** entity `(x,y,yaw)` token；
- **不加入** nav token；
- **不做** geometry projection；
- 不加入 distance/dyaw/velocity 等手工拆分特征；
- spatial 的 ego token 只用于当前帧对齐/上下文，不代表历史对齐。

### 5.3 输出用途

```text
h_t = TSB(BankWindow_t, ego_t)
  ├─ policy head（额外读 current nav/ego）
  └─ WM（只读 h_t 和 action；不读 policy head hidden）
```

- `h_t` 是 TSB 的 per-entity context，不写入 Bank；
- Bank 保存 raw `z_hat`、geometry、type、pose、valid、stable_id、source、rollout_depth、frame_valid；
- WM 的 residual base 是 Bank 当前 `z_hat_t`，不是 `h_t`；
- Ego/Nav 不写入 Bank。

### 5.4 不做什么

- 不把 `TemporalPool` 原样保留在主路径；
- 不做 variable stable-ID node graph；
- 不把 geometry 投影到 query frame；
- 不在 spatial attention 里加入手工 distance/dyaw/velocity 特征；
- 不用 `stack_history` 作为历史接口；
- 不把 nav 放进 TSB。

---

## 6. PolicyHead

### 6.1 输入

```text
z_od, z_ld                 # TSB 输出
od_valid, ld_valid         # current masks
ego_hidden                 # 当前 ego 12D 特征编码
nav_hidden                 # 当前 nav 编码
```

### 6.2 结构

```text
query = Linear(2*hidden, hidden)([ego_hidden, nav_hidden])
K/V = [z_od, z_ld]（带 valid mask）
attn = CrossAttention(query, K/V)
attn = norm(attn + query)
attn = attn + FFN(attn)
od_pool = masked_mean(z_od)
ld_pool = masked_mean(z_ld)
policy_context = MLP([attn, ego_hidden, nav_hidden, od_pool, ld_pool])
action_mu = sigmoid(Linear(policy_context))
action_logstd = clamp(Linear(policy_context), -5, 0)
aux = aux_head(policy_context)
```

- policy head 用 ego/nav **query OD/LD**；
- 不重复 temporal attention；历史已由 TSB 编码；
- policy hidden 不传给 WM；
- 不设 value head / risk head；
- aux：collision/offroad/timeout/urgency/progress。

---

## 7. EgoConditionedLatentWM

### 7.1 输入

```text
h_od_t, h_ld_t              # TSB per-entity context（6 帧历史）
h_valid = current_valid     # WM 只建模当前帧实体；target-only 新实体 mask
current_valid              # 当前帧真实存在的实体
action_t                    # normalized action
```

WM **不接收**：

- `v0` / ego hidden；
- policy head hidden / Planner latent；
- nav。

历史由 TSB 负责，并以 `h_t` 的形式进入 WM。
WM 的 output 是 residual `r_z,t`；residual base 是 Bank 当前 raw `z_hat_t`：

```text
z_hat_{t+1} = z_hat_t + r_z,t
```

`action_t` 的 `(dx,dy,dpsi)` 同时提供 current -> next 的相对位姿变化条件，因此 WM 不需要额外的 `delta_pose` 或显式变换模块。

### 7.2 action 表示

```text
policy_action:     [0,1]^4 normalized
requested_action:  physical (dx,dy,dpsi,v_next)
executed_action:   physical; 当前 = requested_action
```

- WM training：有 `executed_action` 时用它，否则用实测 `action`（IL/val expert 数据通常没有 executed_action）→ normalize(bounds) → WM；
- WM rollout：`policy_action` → WM；
- ego pose/ego feature：`requested_action` physical；
- 引入 LQR 后，`executed_action` 才允许与 `requested_action` 不同。

### 7.3 latent transition

```text
z_hat_0 = z_0
h_t     = TSB(BankWindow_t, ego_t)
r_z,t   = WM(h_t, a_t)
z_hat_{t+1} = z_hat_t + r_z,t
```

- `r_z` 是 latent residual，不是 geometry residual；
- `z_hat` 是 Bank 中累积后的 full-state latent；
- `h_t` 是 TSB context，不写入 Bank；
- residual base 是 `z_hat_t`，不是 `h_t`；
- `r_z` 最后一层 zero-init，初始化时保持 persistence；
- OD/LD 都用同一规则累积；
- `r_z` 学习跨 frame 差异；不做显式 latent 旋转；
- 历史 frame 的 TSB 对齐由 `delta_pose` 负责；WM 的 current -> next frame 相对位姿变化由 action conditioning 提供，不要求显式变换模块。

### 7.4 direct geometry decoder

```text
g_hat_od,t = Dec_od(z_hat_od,t) -> [x,y,yaw,vx,vy]  # z_hat_t 所在 frame（t）
g_hat_ld,t = Dec_ld(z_hat_ld,t) -> [x0,y0,...,x4,y4] # z_hat_t 所在 frame（t）
```

- 没有 base projection；
- 没有 geometry residual；
- OD 输出 5D，LD 输出 10D；
- **不预测 type**；type 从 t0 继承；
- 新 canonical contract 只建模 current valid 实体；target-only 新实体 mask；
- 当前 legacy one-step code 会把新实体填入 free slot 并监督 geometry；这是待删除的 train/inference 不一致，不作为新 contract；
- geometry 使用 geometry target mask；current valid 与 target mask 的交集才监督。

### 7.5 target 和 loss

```text
z*_od_{t+1} = target_encoder(o_{t+1}).od   # stop-grad
z*_ld_{t+1} = target_encoder(o_{t+1}).ld   # stop-grad

L_od_content = D(z_hat_od, sg(z*_od))
L_ld_content = D(z_hat_ld, sg(z*_ld))
L_od_geo     = L1(Dec_od(z_hat_od), g_od_target)
L_ld_geo     = L1(Dec_ld(z_hat_ld), g_ld_target)
```

- OD/LD 都有 content target；
- OD/LD 都有 direct geometry target；
- geometry target 在 t+1 query ego frame（`target_encoder(o_{t+1})` 所在 frame）；decoder 对 `z_hat_{t+1}` 的输出也按该 frame 监督；不做 projection；
- 网络学的是 OD/LD 相对自车的位姿变化；不要求额外 SE(2) 变换模块；
- content 使用 content target mask；geometry 使用 geometry target mask；
- `z*` 只进入 loss；
- future map 只在 target/oracle，不进入 encoder/decoder 输入；
- 不预测 type；
- 新 canonical contract 不监督 target-only 新实体；新实体 mask；当前 code 的 free-slot 新实体监督是 legacy。

### 7.6 target encoder 阶段

```text
当前 WM / joint 训练: frozen copy of online encoder
当前不启用 EMA: 保持 frozen target
RL: 不使用 target encoder / WM；WM 训练继续 frozen target
```

- `z*` stop-grad；
- target encoder 同结构、不同实例；
- OD/LD 都走 target encoder；
- 当前 WM 训练时 online encoder/TSB 都冻结，因此 target 与 online encoder 不漂移；
- TODO：target drift / latent norm / decoder 误差监控当前尚未实现。
- 只有未来 online encoder 参与训练时才重新评估 EMA。

---

## 8. Rollout

### 8.1 循环

```text
RealBank.snapshot()
  → ImagineBank

for t = 0..5:
    h_t = TSB(BankWindow_t, ego_t)
    policy_t = PolicyHead(h_t, ego_hidden_t, nav_0)
    a_t = policy_t.policy_action

    pose_{t+1} = compose(pose_t, requested_action_t)
    ego_vec_{t+1} = build_ego(ego_vec_t, requested_action_t)
    ego_hidden_{t+1} = shared_ego_encoder(ego_vec_{t+1})

    r_z,t = WM(h_t, a_t)
    z_hat_{t+1} = z_hat_t + r_z,t
    g_hat_{t+1} = Dec(z_hat_{t+1})

    append ImagineBank(
      content=z_hat_od,t+1 / z_hat_ld,t+1,
      geometry=g_hat_od,t+1 / g_hat_ld,t+1,
      type=t0 type,
      source=IMAGINED,
      rollout_depth=t+1,
      valid=...
    )
```

### 8.2 长度

```text
policy_0..policy_5     6 个 policy
a0..a5                 6 个 action
t1..t6                 6 个 imagined transition
t6                     terminal state；没有 policy_6
rollout_depth          1..6
ego_pose               概念 t0..t6；Bank 只保留 t1..t6 六帧；API 不暴露 7-pose 轨迹
actions                [B,6,4]，全部使用
nav/ego policy inputs  6 个（t0..t5）
```

- `policy_n` 是 t_n 时刻给出的 action；
- `a0` 来自 t0 policy；
- `a5` 产生 t6 terminal state；
- ImagineBank FIFO 保留最后 6 帧；
- `imagine_rollout` 返回 6 个 policy dict，不返回 7 个 pose；
- 不把 imagined 回写 RealBank。

### 8.3 对外输出

- **无 rollout**：`model.forward` 只输出 t0 policy；
- **有 rollout**：`model.imagine` / `model.rollout` 输出 policy_0..policy_5。

模型级 rollout 在 `net/world_model.py`；`DriveModel.imagine` / `DriveModel.rollout` 是薄 wrapper。
model-level rollout 只在 `net/world_model.py` / `net/model.py`，不另设 rollout 文件。

---

## 9. Ego / action 接口

### 9.1 action 分类

```text
policy_action     [0,1]^4 normalized（mean 或 sample）
requested_action  physical (dx,dy,dpsi,v_next)
executed_action   env 实际执行
```

BC/DAgger policy 训练路径：

```text
requested_action == executed_action
```

BC/DAgger student/lift 行、canonical eval、PPO 都执行 net 的 requested action；
PSTL feasibility/comfort 只作为诊断，不改写动作。非有限或无法构造轨迹的动作直接
终止 episode 或报错，不使用 safe-brake。DAgger 的 expert replay 行按设计执行
expert action，并记录为 replay，不属于 student exact execution。

### 9.2 action bounds / limits

- `config/model.yaml` 负责 action bounds/limits；
- `env/action/action_space.py` 只负责：
  - normalize/denormalize；
  - PSTL action -> trajectory；
  - 静态诊断；
- 不再持有第二套 canonical `CLEAN_BOUNDS`；
- `denormalize` 不做隐式 clip；
- canonical collection/eval/PPO 不启用 state-dependent fallback；
- `feasible`/`comfort` 只作诊断；不改变动作、不 mask finite infeasible action。

### 9.3 ego 更新

```text
pose_{t+1} = compose(pose_t, requested_action_t)
```

```text
speed     = v_next
a_long    = (v_next - v_prev) / dt
yaw_rate  = dpsi / dt
a_lat     = speed * yaw_rate
curvature = yaw_rate / max(speed, 0.5)
prev_dx   = dx
prev_dpsi = dpsi
v_long    = speed
v_lat     = 0
steer     = bicycle approximation
```

- 共享 `encoder.py` 的 `ego_encoder`；
- `nav` 固定 t0；
- `ego_pose` 不在 12D ego feature 内，单独维护；
- action 的 `(dx,dy,dpsi)` 就是 current -> next frame transform。

---

## 10. 数据流

### 10.1 Real

```text
env current observation（OD/LD）
  → online single-frame encoder
  → z + explicit geometry + stable_id + valid
  → RealBank.append
  → BankWindow
  → TSB（OD/LD + ego alignment；无 nav）
  → h_t（per-entity TSB context）
  → PolicyHead（ego/nav query）
  → action
```

Policy forward **不调用 WM**。WM 只在独立训练和显式 `imagine`/`rollout` 中读取 `h_t + action`。

### 10.2 WM Training

```text
real frame t-5..t, t+1
  → online encoder for z_t（OD/LD）
  → TSB on the 6-frame BankWindow → h_t
  → target encoder for z*_{t+1}（OD/LD, stop-grad）
  → WM(h_t, action) predicts r_z
  → z_hat_{t+1} = z_hat_t + r_z
  → direct OD/LD geometry decoders
  → L_od_content + L_ld_content + L_od_geo + L_ld_geo
```

- WM training action = `executed_action` if present else measured `action`；normalize(bounds)；
- target `z*` 不进 Bank/policy；
- 不做 geometry projection；
- future map 只做 target。

### 10.3 Raw history only

Canonical training/eval data must be raw per-frame history produced by the
current collectors (`history_mode=1`, real per-frame pose). Projected
`stack_history` shards are legacy/archive only and are rejected by the loader;
they are not a training or gate input.

### 10.4 Imagine

```text
RealBank.snapshot()
  → 6-step rollout
  → policy_0..policy_5
  → a0..a5
  → t1..t6
  → t6 terminal
```

### 10.5 不能做的事

- 不能把 future map 作为 decoder 输入；
- 不能把 `z*` 写进 RealBank；
- 不能用 online encoder 代替 target encoder；
- 不能让 imagined 记录回写 RealBank；
- 不能用 slot 直接当 stable identity；valid stable ID 必须非零且同帧唯一，违反 fail-fast；
- 不能把 nav 放进 TSB；
- 不能对 TSB/decoder 做 entity geometry projection；
- 本版本只使用 `valid`。

---

## 11. RL 前的边界与待办

### 11.1 已冻结

1. **Policy cross-attention**
   - 结构见 §6.2：linear query + cross-attention + residual/FFN + pooled OD/LD context；
   - 全 invalid 时只保留数值 anchor，不改变语义；
   - policy hidden 不进入 WM。

2. **WM transition**
   - per-entity residual MLP；
   - 输入 `h_t + action`，residual base 是 raw Bank `z_t`；
   - WM 只建模 current valid 实体；target-only 新实体 mask；当前 code 的 slot cue 是 legacy；
   - OD/LD 都用 direct absolute geometry decoder；
   - 不再使用 geometry projection / residual base。

3. **LD content / geometry**
   - LD content 与 OD 使用同一 masked loss 规则；
   - LD geometry 是 10D 直接回归；
   - `geometry_weight` 默认 0.1；
   - 不再有 LD base projection。

4. **Action execution**
   - 所有 canonical 路径 exact tracking；
   - 不保留 `v_next` projection / safe-brake / postprocess；
   - 非有限或不可执行动作终止 episode 或报错。

5. **RL 与 WM 的边界**
   - RL 只使用 real env；WM 不进入 RL 的 forward、rollout 或 loss；
   - WM 独立训练，拟合 env 的下一帧 content/geometry；
   - joint 阶段才通过显式 `imagine` 接口，把 WM 的 6-step 轨迹与 policy 对齐；
   - WM 不是 reward/termination model；RL reward 是一个规则 reward model，由 env info 计算；不单独维护 cost。

### 11.2 RL 前必须完成

1. **exact DAgger r3 baseline**：`runs/bc_dagger_exact_r3_261107031440/best_closed_loop.pt`，val500 success 0.664。
2. **唯一 hard gate**：real-env RL 后 **test500 success ≥72.8%**（one-shot final gate）。val500 只做 diagnostic；不设 gate 2/3，不把 collision/off-road 单独设成 pass/fail。
3. **real-env group-relative PPO trainer**：多进程 rollout、同 scene seed 的 G 条 group、per-step active-set advantage、PPO clip、规则 reward model、train-holdout closed-loop eval；不复用旧 PPO canary 的 1-update/2-episode 默认。
4. **MetaDrive 同 seed 确定性**：已排模验证：同 seed/同 action 跨进程 ego/traffic max diff=0；新 trainer 必须显式 `reset(seed=scene_seed)`，group 内 policy sampling seed 独立。
5. **action distribution 合同检查**：TruncatedNormal 的采样与 log_prob 已做 Monte Carlo 检查，边界 atom=0、CDF KS 在采样噪声内；保留回归测试，不作为 blocker。
6. **data/eval 冻结**：RL train 1500（seeds 1000..2499）、holdout 500（seeds 2500..2999）、val500 diagnostic、test500 one-shot final；test500 不参与调参。
7. **reward 权重**：用 BC/DAgger calibration split 校准一次；不做权重网格扫参；`adapt_coeff` 第一版固定 1.0，只在 collision/off-road 漂移时启用。

### 11.3 不在本阶段

- 不做 failure-window / mid-episode reset；
- 不做独立 cost/Lagrangian；所有奖励/惩罚进一个规则 reward model；
- 不做 reward/termination 预测头；不做 WM imagined return；
- 不做 LQR requested/executed 分离；
- 不做 birth/death/type 预测；type 沿用当前帧。

---

## 12. 训练阶段

### 12.1 原则

- policy 与 WM 分开训练；BC/DAgger 和 RL 阶段都**不包含** WM loss。
- RL 只使用 real env；WM 不参与 RL 的 forward、rollout、reward 或 advantage。
- RL reward 是一个规则 reward model：每步奖励 + terminal 结果奖励，单流 return；所有惩罚都在 reward 内，不单独维护 cost。
- RL advantage 使用 per-step active-set group baseline；不用 episode-level advantage 广播。
- RL 可训练参数：planner + TSB；encoder 冻结；optimizer 分 param group。
- WM 训练冻结 policy/encoder/TSB/target；只训练 WM/decoder；WM 输出不喂给 policy。
- joint 阶段 policy 和 WM 都更新：policy 用 small lr，TSB 用更小 lr，encoder/target 仍冻结。
- WM 始终用真实下一帧 observation 的 target encoder 输出；target 不进入 Bank/policy。
- action condition 训练优先用 `executed_action`，缺失时用实测 `action`；rollout 用 `policy_action`。
- one-step 先于 multi-step；6-step plan 不等于执行 6 步；horizon 只允许 1/3/6。
- RL train 1500 / holdout 500 / val500 diagnostic / test500 one-shot final；raw `history_mode=1` 不回退。
- detach 边界显式，见 §12.7。

### 12.2 阶段总览

```text
IL expert BC                 # policy only
DAgger rounds                # policy only
RL: real-env group-relative PPO   # policy only; WM 不参与
WM: one-step → 3-step → 6-step    # policy/encoder/TSB frozen; 拟合 env
Joint: 6-step trajectory consistency + KL   # policy + WM 自洽; policy 更新
```

- one-step WM 可以在 RL 并行训练/评测，但**不影响 policy**。并行时必须使用 frozen snapshot copy/process：独立 optimizer、独立输出目录、不共享参数/buffer；不能冻结 live RL 的 encoder/TSB。不能提供 snapshot 时，不要并行训 WM。
- 3/6-step WM 在 RL 通过 test500 72.8% gate 后做；用 RL 的 on-policy 数据 + exact 数据。
- joint 只在 WM 6-step 误差经 owner review 可接受后进行；如果 6-step 误差大，继续优化 WM，不把 joint 退到 3-step。
- 当前 accepted baseline 是 DAgger r3；one-step WM 与 group-relative PPO trainer 已实现；3/6-step WM、新 joint trainer 尚未实现。

### 12.3 Policy loss（BC/DAgger，已有）

```text
L_policy = NLL(action_mu, action_logstd; a_expert)
         + w_point   * point
         + w_heading * heading
         + w_velocity* velocity
         + w_curvature * curvature_hinge
         + w_accel     * accel_hinge
```

- 每行使用 `plan_weight` 加权；失败 student 状态、shield 行、infeasible 行按 `pipeline/train_bc.py` 的 plan-weight 策略处理。
- 当前默认：`nll_coef=1.0`、`w_point=1.0`、`w_heading=1.0`、`w_velocity=0.2`、`w_curvature=0.1`、`w_accel=0.1`。
- 当前默认输入是 raw Bank 序列，`history_mode=1`。
- **没有 WM loss，也没有 value/risk loss。**

### 12.4 RL：real-env group-relative PPO

**初始化与数据**

- 从 exact DAgger r3 `best_closed_loop.pt` 初始化。
- RL scene split（scene-level，按 seed）：
  - train: `train_2000` seeds 1000..2499（1500 scenes）
  - holdout: `train_2000` seeds 2500..2999（500 scenes）
  - calibration: train 1500 内固定 stratified 200 scenes（reward 权重校准，不参与 holdout/test）
  - val500: biased diagnostic only，不参与 selection/promotion
  - test500: final gate，只跑一次
- 训练 scene 按 difficulty 分层；hard/medium 过采样。不做 failure-window / mid-episode reset；每个 rollout 从 scene seed 的 t=0 开始。
- 当前没有专门的 r3-fail+hard+random 数据集；如果以后要 mix，必须新建 scene-level split，不能复用 val500/test500。

**Rollout 与时间单位**

- 每个 policy step = 0.5s segment = 5 个 0.1s env step；只执行 a0。
- `T_i` 是该 episode 的 **policy decision 数**，最多 120；`gamma` 按 0.5s policy decision 折扣。
- `max_steps=600` 是 env step cap（60s），不是固定 episode 长度；success/collision/off-road 提前终止。
- exact execution；非有限/不可执行动作终止 episode，不替换。
- 训练/选择用 holdout 500；val500 只做诊断；test500 只做最终 gate。

**Reward model（规则函数，无 cost 模块）**

每个 policy step 聚合 5 个 env substep；`rc_k` 是 segment 起始的 route completion，`rc_{k+1}` 是 segment 结束：

```text
r_progress_k  = w_progress * (rc_{k+1} - rc_k)
r_comfort_k   = -w_comfort * (max(0, m_k.max_abs_a_long - 2.0)
                              + max(0, m_k.max_abs_a_lat - 3.0)
                              + max(0, m_k.max_abs_jerk_long - 5.0))
r_low_speed_k = -w_low_speed * max(0, 1 - v_end/2.0)
r_wrong_route_k = -w_wrong_route * 1[first segment with rc_{k+1} - rc_k < -0.05]
r_stuck_k       = -w_stuck * 1[first 10s window with speed < 0.1 and rc_delta <= 0]
r_k = r_progress_k + r_comfort_k + r_low_speed_k + r_wrong_route_k + r_stuck_k
```

- `m_k` 是该 0.5s segment 的 `ActionTrajectory.metrics`，包含 `max_abs_a_long`、`max_abs_a_lat`、`max_abs_jerk_long`；comfort 用 segment 级 max，不再按 5 个 substep 二次聚合。
- `wrong_route` / `stuck` 是 **per-step one-shot penalty**，不进 `R_term`；否则它们会被 success/collision/off-road/timeout 的终止优先级覆盖，永远不可达。
- `wrong_route` 来源：`info["route_completion"]` 在相邻 policy step 的下降 >0.05；只罚首次。
- `stuck` 来源：连续 10s（20 个 policy step）`speed < 0.1` 且 `route_completion` 不增加；只罚首次。红灯/路口等待没有可靠观测输入，第一版不排除，作为效率诊断记录；不把它写成 completion 的硬条件。
- terminal 只结算一个事件，优先级：
  ```text
  invalid_action > collision > off-road > success > timeout
  ```
  `success` 要求 `arrive_dest and not collision and not off_road`；不允许“撞到终点也算成功”。
- terminal reward：
  ```text
  R_term = -w_invalid   if invalid_action
           -w_collision if collision
           -w_offroad   if off_road
           +w_success   if success
           -w_timeout * (1 - rc_T) if timeout
  ```
- 所有奖励/惩罚都在同一个 reward model；不单独维护 cost vector/Lagrangian。`adapt_coeff` 是 reward 内 scalar：
  ```text
  R_term_collision = -adapt_coeff * w_collision
  R_term_offroad   = -adapt_coeff * w_offroad
  ```
  第一版固定 `adapt_coeff=1.0`；不做在线自适应。只有 owner 明确批准后才加 drift 指标和慢更新。
- 初始权重 provisional：`w_progress=10`、`w_success=20`、`w_collision=20`、`w_offroad=15`、`w_timeout=10`、`w_invalid=20`、`w_low_speed=0.2`、`w_comfort=0.05`、`w_wrong_route=2`、`w_stuck=2`。
- 权重用 calibration split 校准一次：保证 collision 惩罚 > 提前终止节省的未来 comfort/progress 上界；不跑权重网格，不在 val500/test500 上调。

**Return / advantage（per-step active-set）**

```text
tau_i = terminal decision index
        normal terminal: tau_i = T_i - 1
        invalid_action at decision t: tau_i = t, T_i = t + 1
active_g(t) = {i in group g | t <= tau_i}   # t 时刻仍有 transition；终止步也计入
G_{i,t} = sum_{k=t}^{tau_i} gamma^(k-t) * r_{i,k}
          + gamma^(tau_i-t) * R_term_i
A_{i,t} = G_{i,t} - mean_{j in active_g(t), j != i} G_{j,t}
```

- `G_{i,t}` 必须包含 `R_term_i`，否则 terminal reward 不会被 credit。
- `tau_i` 是 terminal decision index；normal terminal 时 `tau_i=T_i-1`，invalid 时 `tau_i=t`。`T_i` 是 decision 数，不是 env step 数。
- invalid_action 发生在 decision `t`：`r_{i,t}=0`、`tau_i=t`，`R_term` 以 `gamma^0` 结算，active(t) 包含它。
- 用 leave-one-out baseline；`|active_g(t)| < 3` 时该 step 跳过（不产生 advantage）。
- 不整体跳过 all-fail group：all-fail group 若终局不同（collision vs timeout vs stuck），terminal advantage 仍有信号；只有所有轨迹 `G_t` 完全相同（std≈0）时才自然为 0。std 低于 floor 时跳过 advantage 更新。
- 先做 per-group LOO mean；然后对整个 update batch 的 advantage 做一次全局 normalization 到 mean 0/std 1（std 有 floor）。不再做 per-group std 除法，避免双重归一化。
- 第一版单流；two-stream（dense/outcome 分开）只作为 ablation。
- 不做 episode-level advantage 广播。

**Group-relative PPO**

```text
G = 8  # provisional
same scene seed reset G times; env seed same, policy sampling seed independent
no critic / no value head / no GAE
PPO clip = 0.2; epochs = 4; minibatch = 256/512
ratio = pi_new(a|s) / pi_old(a|s)
policy loss = -mean(min(ratio * A_t, clip(ratio, 0.8, 1.2) * A_t))
```

- `gamma=0.99` 起步；60s 末端 success 信号弱时试 0.995；timeout/max_step 按 terminal，bootstrap=0。
- 可选 KL 到 r3 BC policy；监控 logstd/entropy；`approx_kl > 0.02` 或 `clip_frac > 0.3` 提前停该次 inner epochs。
- 不需要 reward/termination 预测头；reward 是规则函数。
- `G`、`minibatch`、`lr`、`clip`、`gamma`、reward weights、`adapt_coeff` 冻结进 config/checkpoint。

**Trainable params**

- planner + TSB 可训练；encoder 冻结；aux head 冻结/排除 optimizer（RL 没有 aux loss）。
- optimizer param groups: planner base lr; TSB 0.1×base lr 或更小; encoder/aux 不进 optimizer。
- `LiveBankRunner` 的 encoder no-grad 保持不变。

**唯一 gate 与交付**

```text
test500 success >= 72.8%   # one-shot final gate
```

- val500 只做 biased diagnostic，不参与 selection/promotion。
- train holdout 500 做 early stop/rollback；`select_checkpoint.py` 的碰撞硬阈值 selector 标为 historical，不进入 RL promotion。
- 不设 gate 2/3；collision/off-road 只记录；如果 72.8% 伴随明显恶化，人工 kill switch，不算自动 gate。
- test500 只跑一次；多次评估会污染。
- **promoted artifact 是 RL checkpoint**；joint 阶段是 research/non-promotable，不消耗 test500。若 joint 要 promotion，必须另立一次性 test500 协议。
- 72.8% 是相对 val500 66.4% 的 expert-parity 目标；test500 没有历史 baseline，报告绝对 success 和 CI，不伪称 delta。

### 12.5 WM：拟合 env 的 one-step / 3-step / 6-step

**One-step**

> 当前 `train_wm.py` 仍用 val500 validation，并保留 legacy target-only free-slot 新实体监督。在切到 train 1500 / holdout 500 split 并删除 free-slot 监督之前，**不能并行跑生产 one-step WM**；只能作为 legacy canary，不进入 selection/diagnostic。

```text
z_hat_{t+1} = z_hat_t + r_z,t
L_wm = λ_z * (D(z_hat_od, sg(z*_od)) + D(z_hat_ld, sg(z*_ld)))
     + λ_od_geo * L1(Dec_od(z_hat_od), g_od_target)
     + λ_ld_geo * L1(Dec_ld(z_hat_ld), g_ld_target)
```

- 输入：真实 BankWindow 的 `h_t`、raw Bank `z_t`、真实 `executed_action`。
- policy/encoder/TSB/target frozen；`h_t`、`z_t`、action、target detach；只训练 WM/decoder。
- WM 输出只用于 WM loss，不喂给 policy；policy 不更新。
- 不预测 birth/death/type head；type 沿用当前帧。
- WM 只建模 current valid 实体；target-only 新实体在 loss 中 mask。当前 one-step code 的 free-slot 新实体监督是 legacy，新 canonical 训练删除；否则 train/inference 不一致。
- imagined rollout / joint 只使用当前 valid 实体；只比较两端都存在的 stable ID；新出现/消失的实体 mask。
- 不做显式 geometry transform；网络学习 OD/LD 相对自车的位姿变化；TSB 的 frame-level `delta_pose` 保持不变。

**3/6-step**

- 在 RL 通过 test500 72.8% gate 后做；用 RL on-policy 数据 + exact/DAgger 数据。
- WM 训练只使用 `train_2000` seeds 1000..2499（1500 scenes）；attention/DAgger/RL 数据加载时按 scene seed 过滤，holdout 500（2500..2999）不得进入训练。
- WM 验证用固定 500-scene holdout，不用 val500；train_wm/train_joint 加载时 assert 无 holdout 重叠。
- horizon curriculum：1 → 3 → 6；不允许 2/4/5。WM 训练/eval entrypoint 断言 `horizon in {1,3,6}`；通用 `imagine` API 可以接受 1..6，但 2/4/5 不用于训练/eval。
- 先用 detached recurrence；稳定后再短 BPTT（2–3 步）。
- scheduled sampling：训练时以概率 p 用模型自己的 prediction，否则用真实下一帧。
- 每个 depth 用 stable ID 对齐真实 target；只对 horizon 两端都存在的实体算 loss；新出现/消失的实体 mask。
- 训练时可以用真实 action + 真实 pose（隔离 WM 误差）；评估时用递归 pose/action（测复合误差）。两种误差分开报告。
- 唯一 WM 诊断：在真实 t0，冻结 collection policy，真实 env 执行 6 个 segment 得到 actual s1..s6；WM imagine 6 步得到 predicted ŝ1..ŝ6；报告 per-depth content MSE / geometry L1 / pose L1，重点看 ŝ2..ŝ6 和近 ego 车辆；同时报告 depth 1 与 depth 6 的误差比。没有自动 pass/fail；是否进入 joint 由 owner 看误差决定。误差大就继续优化 WM，不把 joint 退到 1/3-step。
- 不做 uncertainty 头；不做 reward/termination 头；不单独维护 cost 组件。

**WM 的限制**

- WM 只对当前可见实体建模；新进入的车辆不会出现在 imagined trajectory 中。
- 6-step 在 dense traffic 下可能偏乐观；必须看 2..6 的 per-depth 误差，尤其是近 ego 车辆。
- 6-step 轨迹是预测，不是执行；eval 仍只执行 a0。

### 12.6 Joint：6-step trajectory consistency

**时间约定（0-based）**

本文统一用 0-based policy action：

```text
t0: real current
a0..a5: policy actions at t0..t5
actual s1..s6: 数据采集时 real env 执行 a0..a5 的结果
predicted ŝ1..ŝ6: 从 t0 开始的 6-step WM imagine
```

- 用户口语的 1-based 第 1 点 / 第 2~6 点，对应本文 actual `s1` / `s2..s6`。
- 部署/eval 只执行 `a0`；`a1..a5` 不执行，只是预测。
- 为了得到 actual target，joint 数据采集会冻结 collection policy，在真实 env 执行 `a0..a5`，记录 `s1..s6`；这是数据采集，不是部署执行。
- 比较 `ŝ2..ŝ6` vs `s2..s6`（5 个 depth）；`ŝ1` 也记录，但它是 env anchor，不作为主要优化目标。

**Joint 数据**

- 冻结 collection policy（pre-joint 或 iteration-start snapshot），用 deterministic `action_mu`，在 `train_2000` 的固定 500-scene joint collection subset（**排除 500-scene holdout**）上执行 6 个 segment，记录 observations/actions/poses/stable IDs；同时记录 collection policy checkpoint。
- 500-scene holdout 只做 validation/early stop/rollback，不参与 joint 训练数据采集。
- 每 N 个 update 刷新一次 actual target；target 始终 stop-grad；target 与 collection policy 绑定，不能跨 snapshot 混用。
- 不在 val500/test500 上采集或选择。

**Loss**

```text
L_joint = λ_wm * L_state + λ_policy * L_policy + λ_kl * KL(π_new || π_pre_joint)
```

- `L_state`：在 depth 2..6 比较 predicted od/ld content/geometry 与 actual；使用**数据采集时记录的 collection action**（stop-grad）做 teacher forcing，WM/decoder + shared TSB 收梯度；当前 policy action 不进入这一支。pose 不放在这一支，因为它由 action 计算，action stop-grad 时没有 WM/TSB 梯度。
- `L_policy`：在 depth 2..6 用**当前 policy action**在 WM 里 rollout，比较 predicted pose 与 actual（actual 来自同一 collection snapshot）；action 项只作为 small optional weight（WM 误差下 corrective action 可以合理不同）；policy head + shared TSB 收梯度；WM/decoder 参数冻结；actual target stop-grad。这是 policy 对 collection snapshot 的一致性/纠正目标，不是纯 env 拟合。
- `L_state` 和 `L_policy` 分开 backward；shared TSB 收两边梯度，记录 per-branch grad norm。WM loss 不进入 policy head；policy loss 不进入 WM head。
- joint 阶段 policy 更新（joint 的定义），用 small lr；encoder/target frozen；TSB 用更小 lr；WM 正常 lr。
- 不用 expert NLL；joint trainer 必须拒绝 NLL。当前 `train_joint.py` 是 legacy one-step + expert NLL，不作为新 trainer。
- KL 到 frozen pre-joint policy（encoder/TSB/planner 的 copy）；用 Monte Carlo 估计：

```text
KL ≈ mean_{s in real+imagined} E_{a~π_new}[log π_new(a|s) - log π_old(a|s)]
```

  采样 K 个 a；需要 `TruncatedNormal.log_prob` 和 `kl` helper。不要用 analytic KL 假装能算。
- `L_state`/`L_policy` 的 action/pose/geometry 权重、per-depth 权重、normalization 在实现时固定并记录；不能只报 total。`lambda_wm`/`lambda_policy`/`lambda_kl`、`N`、`K` 在 config 中冻结并写入 checkpoint。
- 部署/eval 只执行 `a0`；`a1..a5` 是预测，不在 env 执行。joint 数据采集为了 actual target 会执行 `a0..a5`，这不是部署行为。若未来要 open-loop 执行 `a1..a5`，需单独定义 execution contract。
- joint 允许 one-step 作为 sanity；目标形态是 six-step；不允许 3-step joint。6-step 误差大就继续优化 WM，不用 3-step 替代。
- test500 是 RL 的一次性 final gate；joint 是 research/non-promotable，不消费 test500。若 joint 要 promotion，必须另立一次性 test500 协议。

**API 需求**

- 当前 `DriveModel.imagine` 只返回 policy dict；joint 需要 per-step predicted content/geometry/pose/valid/stable_id。计划新增 `imagine_trajectory`（或扩展 `imagine` 返回值），支持 B>1、per-step detach/BPTT schedule；当前 `to_frame_record` 会 detach pose/action，不能直接用于 joint。
- 在 API 完成前，joint trainer 不实现。

**Rollback**

- 保存 joint 前 checkpoint。
- joint 过程中每隔 N 个 update，在 `train_2000` 固定 500 个 scene 的 holdout 上跑闭环；使用 paired seeds，不用 val500/test500。
- 如果 success 比 pre-joint 下降超过 1 个绝对点，或 collision 上升超过 1 个绝对点，丢掉 joint checkpoint，回到 pre-joint。
- 这是安全回滚，不是额外 gate。

### 12.7 detach

| 边界 | 默认 | 说明 |
|---|---|---|
| RL → WM | 不连接 | RL 不使用 WM |
| `z*` | detach | target encoder 输出 |
| target encoder | no grad | frozen |
| `Bank z_hat_t` → residual base | detach | WM 只学 residual |
| `h_t`（TSB）→ WM | detach | WM 训练时 TSB 冻结 |
| `action` → WM | detach | 条件/标签 |
| policy head hidden → WM | 不传入 | WM 不接收 |
| `z_hat_{t+1}` → decoder | 允许梯度 | decoder loss 训练 WM/decoder |
| multi-step hidden | 先 detach | 稳定后再短 BPTT |
| geometry target | detach | 只作监督 |

> 上表是 one-step WM 训练的 detach。joint 的梯度图不同：`L_state` 允许 `h_t`/TSB 梯度，但 policy action stop-grad；`L_policy` 允许 action 梯度，但 WM/decoder 参数冻结；actual future 始终 stop-grad。见 §12.6。joint 的 `h_t → WM` 不能按 one-step 的 detach 处理，否则 WM loss 无法更新 shared TSB。

### 12.8 gate / 监控 / 回滚

- **唯一 hard gate**：real-env RL 后 test500 success ≥ **72.8%**（one-shot）；exact execution；`max_steps=600`；deterministic `action_mu`。
- 训练监控/early stop/WM validation/joint rollback 统一用 `train_2000` 内固定 500 个 scene 的 scene-disjoint holdout；paired seeds；**不用 val500**。
- val500 只做 diagnostic；test500 只做最终 gate 一次。
- collision/off-road/timeout/route completion/feasible：只记录，用于 debug；不单独设 pass/fail。若 72.8% 伴随明显恶化，属于人工 review 红线，不是第二个自动 gate。
- WM 6-step 正确性：predicted vs actual `ŝ2..ŝ6` vs `s2..s6`；只做诊断，不设多级 gate；是否进入 joint 由 owner review 决定。
- joint rollback：见 §12.6；不是额外 gate。
- 新 trainer 必须统一 collision/off-road 检测：canonical collision = any of `crash`, `crash_vehicle`, `crash_object`, `crash_building`, `crash_human`, `crash_sidewalk`；off-road = `out_of_road`/`off_road`。`eval.py`、RL trainer、joint rollback 必须共用同一个 `_info_flag` helper；当前 eval 只看 `crash`，old PPO 看多个 key，数字不可直接比较。

### 12.9 开放合同检查

1. **TruncatedNormal 边界**：已完成 Monte Carlo 排模：边界 atom=0；N=100k 时样本 CDF vs 解析 CDF 的 KS≈0.0025–0.0048；log_prob 数值积分质量≈1；不构成 blocker。保留回归测试（边界计数 + CDF/质量）。
2. **MetaDrive 同 seed 确定性**：已排模验证：同 seed/同 action 跨进程 ego/traffic max diff=0；新 trainer 必须显式 `reset(seed=scene_seed)`，group 内 policy sampling seed 独立；同 env 长跑有 mm 级漂移时用 fresh env rebuild。
3. **max_steps=600 是 cap**：不是固定 episode 长度；episode 提前终止；不要按 600 强制 padding。
4. **gamma**：0.99 起步；60s 末端 success 信号仍弱时试 0.995；不要用 0.95。
5. **reward 权重/adapt_coeff**：provisional；用 BC/DAgger calibration split 校准一次；不在 val500/test500 上调。
6. **exact execution**：所有 canonical 路径保持 `requested_action == executed_action`；不引入 fallback。
7. **WM 6-step 只预测不执行**：eval 仍只执行 a0；不要把它写成 open-loop 执行 6 步。
8. **split 隔离**：当前 `train_wm.py`/`train_joint.py` 仍用 val500 做 validation/selection；`config/eval_thresholds.yaml` 仍写 `val500: checkpoint_selection`，`pipeline/select_checkpoint.py` 仍可在 val500 上选。新代码必须切到 `train_2000` 固定 500-scene holdout；val500 只做 diagnostic；test500 只做 one-shot final gate。
9. **probe/horizon 对齐**：当前 `probe_wm.py` 默认跑 1..H，`net/world_model.py` 接受 1..6；新训练/eval entrypoint 只允许 1/3/6。通用 API 的其他 horizon 不作为支持范围。
10. **reward 事件语义**：`invalid_action > collision > off-road > success > timeout`；success 不允许覆盖 collision/off-road；wrong_route/stuck 是 per-step one-shot，不进 terminal 优先级。必须有单元测试覆盖同帧事件、wrong_route/stuck 可达性、timeout rc shaping。
11. **aux head**：RL 没有 aux loss；aux head 冻结/排除 optimizer，避免 AdamW weight decay 改动它。
12. **config/checkpoint**：G、minibatch、lr、clip、gamma、reward weights、adapt_coeff 必须写入 config 和 checkpoint，便于复现。

---

## 13. 文件/职责

```text
config/
  model.yaml              # 网络结构、Bank、rollout、action bounds/limits
  env.yaml
  train_bc.yaml
  train_wm.yaml           # current one-step WM; 3/6-step planned; ckpt=r3 policy checkpoint
  train_joint.yaml        # current legacy one-step + expert NLL; new joint planned
  train_joint_traj.yaml   # planned; 6-step trajectory consistency + KL config
  train_ppo.yaml          # legacy PPO canary
  train_ppo_group.yaml    # implemented; group-relative PPO config
  eval_thresholds.yaml

net/
  mem_bank.py             # EntityRecord/FrameRecord/RealBank/ImagineBank/BankWindow
  encoder.py              # online current-frame encoder + frozen target encoder
  world_model.py          # TSB / WM / direct geometry decoders / rollout
  planner.py              # PolicyHead cross-attention + aux
  model.py                # 顶层组合、forward、imagine/rollout
  build_net.py            # config -> model

env/
  obs.py                  # observation schema / raw_history / ego build
  tracking.py             # Exact/LQR tracking and execution adapter
  action/
    __init__.py
    action_space.py       # normalize/denormalize + PSTL + 静态诊断；无 canonical bounds

dataset/
  collect.py              # expert IL collection
  collect_dagger.py       # DAgger collection
  splits.py               # scene-level split helpers
  freeze_splits.py        # specs/split manifest generation

data/splits/
  train_2000_rl_train_1500.json    # implemented: seeds 1000..2499
  train_2000_rl_holdout_500.json   # implemented: seeds 2500..2999
  train_2000_rl_calibration_200.json # implemented: 200 stratified calibration scenes

pipeline/
  losses.py               # 纯 loss 函数
  reward.py               # implemented: rule-based RL reward model + event aggregation
  train_bc.py             # row-wise / scene-sequence BC+DAgger；闭合选择
  scene_batches.py        # scene-level ordered sequence batches
  train_config.py         # YAML + CLI 合并
  checkpointing.py        # checkpoint load/save + strict load
  train_wm.py             # current one-step WM；3/6-step planned；policy/encoder/TSB frozen
  probe_wm.py             # current depth 1..H content/geometry；1/3/6 predicted-vs-actual planned
  train_joint.py          # legacy one-step + expert NLL；new joint planned
  train_ppo_group.py      # implemented real-env group-relative PPO
  train_ppo.py            # legacy PPO canary；不再作为 RL 主线
  eval.py                 # closed-loop eval；test500 final gate
  eval_metrics.py         # closed-loop summary rows
  select_checkpoint.py    # historical safety selector; not for RL promotion
  live_bank.py            # online RealBank / FIFO runner
  execution.py            # action segment / relative action helpers
  distributions.py        # TruncatedNormal；boundary sample/log_prob tested
```

当前实现状态：one-step WM 已实现；`probe_wm.py` 当前跑 1..H；`train_joint.py` 是 legacy one-step + expert NLL；`train_wm.py` 仍包含 target-only free-slot 新实体监督（新 contract 要求删除）。group-relative PPO、reward model、RL split/loader 已实现；3/6-step WM、新 joint trainer 未实现。RL baseline run 已完成 50 updates（holdout best 0.688 / final 0.680，uncalibrated；holdout 有 IL exposure confound），owner 已停止 RL 线；test500 未消费。详细结果/失败分析见 `docs/deepseekv4p1_argue.md`。旧代码里对 `§12.2` / `§13 Stage C` 的引用已过期，以本文件为准。

职责：

| 路径 | 负责 | 不负责 |
|---|---|---|
| `mem_bank.py` | Bank 状态、FIFO、stable-ID/valid、BankWindow | 网络前向、loss、env |
| `encoder.py` | online encoder、frozen target encoder | Bank 状态、rollout |
| `world_model.py` | TSB、WM、direct OD/LD decoders、模型级 rollout | env step、训练循环、loss |
| `planner.py` | policy cross-attention、Plan/aux | Bank、WM、risk/value head |
| `model.py` | 顶层组合、forward、rollout、raw-history adapter、FIFO-safe alignment | YAML 解析、训练、env |
| `build_net.py` | 读 YAML、实例化、维度校验 | per-episode Bank、训练 |
| `env/action/action_space.py` | normalize/denormalize、PSTL、静态诊断 | 网络、checkpoint、Bank |
| `pipeline/losses.py` | 纯 policy/WM loss | target encoder、数据加载 |
| `pipeline/reward.py` | 规则 RL reward model、5-step 聚合、terminal event 优先级 | 网络、env step、checkpoint |
| `pipeline/train_bc.py` | BC/DAgger、raw Bank 序列、closed-loop validation 调度 | 训练数据采集、WM loss |
| `pipeline/train_wm.py` | 当前 one-step WM；3/6-step WM（planned）；frozen policy/encoder/TSB | policy 更新、closed-loop |
| `pipeline/probe_wm.py` | 当前 depth 1..H content/geometry；1/3/6 predicted-vs-actual（planned） | policy 训练、env step |
| `pipeline/train_joint.py` | 当前 legacy one-step + expert NLL；新 joint（planned） | 当前不是新 joint |
| `pipeline/train_ppo_group.py` | real-env group-relative PPO（implemented） | WM 训练、reward model |
| `pipeline/train_ppo.py` | legacy PPO contract canary | 新 RL 主线、WM 训练 |
| `pipeline/select_checkpoint.py` | optional hard collision/off-road safety selector | canonical train_bc selection |
| `pipeline/live_bank.py` | online BankWindow / FIFO | 网络结构、训练 |

依赖方向：

```text
mem_bank.py
  ↑
encoder.py
  ↑
model.py
  ↑
build_net.py

mem_bank.py
  ↑
world_model.py
  ↑
model.py

planner.py
  ↑
model.py
```

- `mem_bank.py` 不 import encoder/world_model/planner/model；
- `world_model.py` 只依赖 `mem_bank.py`，不依赖 `encoder.py` / `planner.py` / `model.py`；
- `planner.py` 不依赖 `world_model.py`；
- `model.py` 组合 encoder / mem_bank / planner / world_model；
- `build_net.py` 依赖 `encoder.py`、`model.py`、`planner.py`、`world_model.py`；
- `env/action/action_space.py` 不被 mem_bank/world_model 反向依赖；
- `pipeline/losses.py` 不 import model.py。

### 13.1 config/model.yaml 边界

负责：

- encoder/temporal_spatial/world_model/planner 结构与超参数；
- bank.capacity/od_slots/ld_slots；
- rollout.horizon；
- action bounds/limits（只配置，不 build nn.Module）。

`build_net.py` 负责读取、实例化、校验；不创建 per-episode Bank，不训练。

### 13.2 rollout 归属

- 模型级 rollout：`net/world_model.py`；
- 顶层调用：`net/model.py` 的 `imagine` / `rollout`；
- 训练/评测调度：`pipeline/train_wm.py`、`pipeline/probe_wm.py`、`pipeline/eval.py`；
- rollout 只在 `net/world_model.py` / `net/model.py`，不再有独立的 `pipeline/rollout.py`。

---

## 14. 一句话版本

> Real/Imagine Bank 只存 OD/LD；固定 16 slot、stable ID、valid mask；valid stable ID 非零且同帧唯一；`z` 是 frame-dependent full-state latent；TSB 用同 stable-ID 历史 + `delta_pose` 做 temporal，用 OD/LD + ego 做 spatial（不含 nav、不做 geometry projection）；TSB 的 per-entity `h_t` 同时服务 policy 和 WM；policy 用 ego/nav query TSB，policy head hidden 不进入 WM；WM 用 `h_t + action` 预测 latent residual `r_z`，`z_hat_{t+1}=z_hat_t+r_z`，并用单帧 frozen target 监督 OD/LD（当前不启用 EMA）；OD/LD geometry 由 direct absolute decoder 输出；6 个 policy / 6 个 action / 6 个 transition；Plan/aux，无 risk head、无 value head；所有 canonical 路径 exact tracking，requested == executed，不做 projection/safe-brake/clip；RL 只用 real env + group-relative PPO，WM 不参与 RL；WM 独立拟合 env，6-step predicted vs actual 检查；joint 用 6-step trajectory consistency + KL 让 policy/WM 自洽；唯一 hard gate 是 test500 success ≥72.8%（one-shot）；实现状态见 §13。

### 14.1 Scene-level split

Train/validation/test leakage is forbidden by **scene seed**, not by episode id. The same scene can be re-collected as multiple BC/DAgger episodes; splitting by episode can put copies of one scene on both sides. `dataset/splits.py` is the single implementation of this rule. RL uses `train_2000` seeds 1000..2499 (1500 train) and seeds 2500..2999 (500 holdout); val500 is diagnostic only; test500 is one-shot final gate. RL/WM/joint loaders must assert no holdout overlap with training data.
