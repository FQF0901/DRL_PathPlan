# Memory Bank + Temporal-Spatial Attention + Policy + Ego-Conditioned Latent WM

> 状态：**owner-reviewed working contract**
> 日期：2026-10-25
> 范围：Real/Imagine Memory Bank、BankWindow、TemporalSpatialBlock、PolicyHead、OD/LD direct geometry decoder、Ego-conditioned latent WM、6-step rollout 和 action/ego 接口。
> 明确不做：本文件不修改代码、不训练、不产生 gate 结果。
> 原则：只记录已经讨论并接受的决策；未决项集中在 §11。

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
14. Action 当前 exact tracking：`requested_action == executed_action`；LQR 引入后才允许区分。

---

## 1. 术语和坐标

### 1.1 frame

- `ego frame`：当前时刻自车坐标系；
- `t0 frame`：rollout 起点的自车坐标系；
- `world frame`：只用于诊断/可视化，不作为网络输入。

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

用途：区分真实/想象、控制 horizon、loss 加权、误差诊断、禁止 imagined 回写 RealBank。

---

## 2. Bank record schema

### 2.1 FrameRecord

```text
FrameRecord
  episode_id: int/str
  frame_id: int
  timestamp: float
  ego_pose: [3]            # x,y,yaw in world/t0-consistent frame
  ego_velocity: [2]        # optional, diagnostic
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
                       & (stable_id[t, n] == stable_id[历史, n])
```

slot 复用时清空旧 content/geometry/type，写入新 stable_id，旧历史 temporal mask=0。

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
- Phase 0/1/2 当前 frozen copy；未来 online encoder 参与训练时再评估 EMA；
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
- Bank 只保存 raw `z_hat`、geometry、pose、valid、stable_id；
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
query = MLP([ego_hidden, nav_hidden])
K/V = [z_od, z_ld]（带 valid mask）
policy_context = CrossAttention(query, K/V)
action_mu, action_logstd, aux = heads(policy_context)
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
h_valid                     # current valid mask
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

`action_t` 的 `(dx,dy,dpsi)` 同时提供 current -> next 的 ego frame transform，因此 WM 不需要额外的 `delta_pose`。

### 7.2 action 表示

```text
policy_action:     [0,1]^4 normalized
requested_action:  physical (dx,dy,dpsi,v_next)
executed_action:   physical; 当前 = requested_action
```

- WM training：`executed_action` → normalize(bounds) → WM；
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
- 历史 frame 的 TSB 对齐由 `delta_pose` 负责；WM 的 current -> next frame transform 由 action 提供。

### 7.4 direct geometry decoder

```text
g_hat_od,t = Dec_od(z_hat_od,t) -> [x,y,yaw,vx,vy]  # 当前 ego frame
g_hat_ld,t = Dec_ld(z_hat_ld,t) -> [x0,y0,...,x4,y4] # 当前 ego frame
```

- 没有 base projection；
- 没有 geometry residual；
- OD 输出 5D，LD 输出 10D；
- **不预测 type**；type 从 t0 继承；
- 新实体也直接监督 geometry，不依赖历史 base；
- valid mask 来自当前帧。

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
- geometry target 在当前 query ego frame；
- mask 用 `valid`；
- `z*` 只进入 loss；
- future map 只在 target/oracle，不进入 encoder/decoder 输入；
- 不预测 type；
- 新 OD/LD 没有历史 base 也可以监督。

### 7.6 target encoder 阶段

```text
Phase 0/1: frozen copy of online encoder
Phase 2: 当前不启用 EMA；保持 frozen target
Phase 3: 继续 frozen target，不跟随 GRPO policy 更新
```

- `z*` stop-grad；
- target encoder 同结构、不同实例；
- OD/LD 都走 target encoder；
- 当前 WM 训练时 online encoder/TSB 都冻结，因此 target 与 online encoder 不漂移；
- 记录 target drift / latent norm / decoder 误差。
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
ego_pose               t0..t6，共 7 个
actions                [B,6,4]，全部使用
nav/ego policy inputs  6 个（t0..t5）
```

- `policy_n` 是 t_n 时刻给出的 action；
- `a0` 来自 t0 policy；
- `a5` 产生 t6 terminal state；
- ImagineBank FIFO 保留最后 6 帧；
- 不把 imagined 回写 RealBank。

### 8.3 对外输出

- **无 rollout**：`model.forward` 只输出 t0 policy；
- **有 rollout**：`model.imagine` / `model.rollout` 输出 policy_0..policy_5。

模型级 rollout 在 `net/world_model.py`；`DriveModel.imagine` / `DriveModel.rollout` 是薄 wrapper。
`pipeline/rollout.py` 至多是 action preview 或 wrapper，不能复制 WM loop。

---

## 9. Ego / action 接口

### 9.1 action 分类

```text
policy_action     [0,1]^4 normalized（mean 或 sample）
requested_action  physical (dx,dy,dpsi,v_next)
executed_action   env 实际执行；当前 exact tracking，等于 requested_action
```

当前阶段：

```text
requested_action == executed_action
```

只有引入 LQR 后，两者才允许不同。

**Eval/PPO safety fallback 例外**：eval 和 PPO rollout 在 requested action
PSTL-infeasible 时允许显式的 `v_next` projection 或 safe-brake；必须同时记录
`requested_action` / `executed_action`、fallback code，并对 PPO 样本 mask 或
重算 executed log-prob。该例外只属于安全执行层，不是 policy 的 exact-tracking
训练目标。

### 9.2 action bounds / limits

- `config/model.yaml` 负责 action bounds/limits；
- `env/action/action_space.py` 只负责：
  - normalize/denormalize；
  - PSTL action -> trajectory；
  - 静态诊断；
- 不再持有第二套 canonical `CLEAN_BOUNDS`；
- `denormalize` 不做隐式 clip；
- policy 训练路径不启用 state-dependent fallback；
- eval/PPO 安全执行层的 fallback 例外见 §9.1；必须记录 requested/executed 并 mask RL 样本或重算 executed log-prob。

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
  → WM(h_t, action) / decoders（训练辅助）
```

### 10.2 Training

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

- WM training action = executed_action normalized；
- target `z*` 不进 Bank/policy；
- 不做 geometry projection；
- future map 只做 target。

### 10.3 Imagine

```text
RealBank.snapshot()
  → 6-step rollout
  → policy_0..policy_5
  → a0..a5
  → t1..t6
  → t6 terminal
```

### 10.4 不能做的事

- 不能把 future map 作为 decoder 输入；
- 不能把 `z*` 写进 RealBank；
- 不能用 online encoder 代替 target encoder；
- 不能让 imagined 记录回写 RealBank；
- 不能用 slot 直接当 stable identity；
- 不能把 nav 放进 TSB；
- 不能对 TSB/decoder 做 entity geometry projection；
- 本版本只使用 `valid`。

---

## 11. 未决项

1. **LQR 的 executed/requested 分离**
   - 当前 exact tracking；
   - LQR 后如何记录实际执行动作、如何 mask policy log-prob。

2. **Policy cross-attention 细节**
   - query 用拼接还是独立 ego/nav token；
   - cross-attention 层数/heads；
   - 全 invalid OD/LD 时的 fallback。

3. **WM transition 结构**
   - per-entity MLP + global context；
   - 或轻量 cross-attention；
   - 参数量/训练数据/闭环误差的对照。

4. **LD content target 的权重和 gate**
   - LD content loss 是否与 OD 同权；
   - LD latent norm/drift 监控。

5. **action fallback**
   - 当前关闭；
   - 未来 fallback 与 RL feedback 的契约。

---

## 12. 训练阶段（提案，不是 gate 结果）

### 12.1 原则

- policy 数据和 WM 数据分开；
- WM 训练固定使用 data/splits/val_500.json 对应的 held-out val 场景；
- WM 训练时 encoder/TSB/target encoder 冻结，只训练 WM/decoder；
- WM 始终用真实下一帧 observation 的 target；
- action condition 训练用 executed，rollout 用 policy；
- one-step 先于 multi-step；
- 6-step plan 不等于执行 6 步；
- detach 边界显式。

### 12.2 阶段

```text
Phase 0: IL + one-step WM auxiliary
Phase 1: DAgger + one-step WM
Phase 2a: WM horizon curriculum 1→2→3→6
Phase 2b: 可选 joint fine-tune（policy + WM；TSB 小学习率，encoder/target 冻结）
Phase 3: GRPO
Phase 4: LQR / receding horizon
```

- Phase 0/1 的 policy/BC 训练会更新 encoder/TSB/planner；WM 的独立训练阶段则冻结 encoder/TSB/target，只训练 WM/decoder。
- Phase 2b 是**可选**阶段，不是默认训练：
  - 只有在 policy 和 WM 分别收敛、val500 稳定后才允许；
  - encoder/target encoder 仍冻结；
  - TSB 解冻，但使用独立的小学习率（例如 base lr 的 0.01–0.05）和单独 param group；
  - policy head 与 WM 的 loss 同时开启，必须记录各自梯度范数和 validation；
  - 若 policy success 或 WM 指标退化，回滚到 Phase 0/1 checkpoint；
  - test500 仍然不可参与该阶段调参。

### 12.3 Phase 0 loss

```text
L_policy_il = L_action(policy(z_t, nav_t, ego_t), a_expert)
L_wm = λ_z * (D(z_hat_od, sg(z*_od)) + D(z_hat_ld, sg(z*_ld)))
     + λ_od_geo * L1(Dec_od(z_hat_od), g_od_target)
     + λ_ld_geo * L1(Dec_ld(z_hat_ld), g_ld_target)
```

- `z_hat_{t+1} = z_hat_t + r_z,t`；
- `h_t = TSB(BankWindow_t)`；WM 只读 `h_t` 和 action；
- target encoder both OD/LD，单帧、不接 history；
- 不做 geometry projection；
- decoder 对 `z_hat_{t+1}` 保留梯度；
- `z_hat_t`、`h_t`、action、target 默认 detach/no-grad（WM 训练时 TSB 冻结）。

### 12.4 Phase 2 horizon

```text
horizon: 1 → 2 → 3 → 6   # transition 数
```

每个 depth 都有真实 target；每个 depth 记录 vs persistence 的误差。
Phase 2 起启用 ImagineBank；depth 最大 6。

### 12.5 detach

| 边界 | 默认 | 说明 |
|---|---|---|
| `z*` | detach | target encoder 输出 |
| target encoder | no grad | frozen（当前不启用 EMA） |
| `Bank z_hat_t` → residual base | detach | WM 只学 residual |
| `h_t`（TSB）→ WM | detach | WM 训练时 TSB 冻结 |
| `action` → WM | detach | 条件/标签 |
| policy head hidden → WM | 不传入 | WM 不接收 |
| `z_hat_{t+1}` → decoder | 允许梯度 | decoder loss 训练 WM/decoder |
| multi-step hidden | 先 detach | 稳定后再短 BPTT |
| geometry target | detach | 只作监督 |

### 12.6 gate 建议

- Phase 0：one-step content/geometry 不低于 persistence；
- Phase 1：DAgger 在 student 状态上不崩；
- Phase 2：depth 1/2/3/6 误差曲线可接受；
- Phase 3：GRPO 不出现 entropy collapse/saturation/reward hacking；
- Phase 4：LQR 只执行第一段时闭环稳定。

`persistence` 在 latent 空间指 `z_hat_t = z_hat_{t-1}`；geometry 仍由 decoder 输出。
不存在 geometry base/residual persistence。

---

## 13. 文件/职责

```text
config/
  model.yaml              # 网络结构、Bank、rollout、action bounds/limits
  env.yaml
  train_bc.yaml
  eval_thresholds.yaml

net/
  mem_bank.py             # EntityRecord/FrameRecord/RealBank/ImagineBank/BankWindow
  encoder.py              # online current-frame encoder + frozen target encoder
  world_model.py          # TSB / WM / direct geometry decoders / rollout
  planner.py              # PolicyHead cross-attention + aux
  model.py                # 顶层组合、forward、imagine/rollout
  build_net.py            # config -> model

env/
  action/
    __init__.py
    action_space.py       # normalize/denormalize + PSTL + 静态诊断；无 canonical bounds

pipeline/
  losses.py               # 纯 loss 函数
  train_bc.py
  train_wm.py
  train_ppo.py
  eval.py
  rollout.py              # 单步 preview 或 model.imagine 薄 wrapper
```

职责：

| 路径 | 负责 | 不负责 |
|---|---|---|
| `mem_bank.py` | Bank 状态、FIFO、stable-ID/valid、BankWindow | 网络前向、loss、env |
| `encoder.py` | online encoder、frozen target encoder | Bank 状态、rollout |
| `world_model.py` | TSB、WM、direct OD/LD decoders、模型级 rollout | env step、训练循环、loss |
| `planner.py` | policy cross-attention、Plan/aux | Bank、WM、risk/value head |
| `model.py` | 顶层组合、forward、调用 rollout | YAML 解析、训练、env |
| `build_net.py` | 读 YAML、实例化、维度校验 | per-episode Bank、训练 |
| `env/action/action_space.py` | normalize/denormalize、PSTL、静态诊断 | 网络、checkpoint、Bank |
| `pipeline/losses.py` | 纯 loss 函数 | target encoder、数据加载 |

依赖方向：

```text
mem_bank.py
  ↑
encoder.py
  ↑
world_model.py
  ↑
planner.py

model.py
build_net.py
```

- `mem_bank.py` 不 import encoder/world_model/planner/model；
- `world_model.py` 不 import planner/model/build_net；
- `model.py` 只做编排；
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
- 训练/评测调度：`pipeline/train_wm.py`、`pipeline/eval.py`；
- `pipeline/rollout.py` 只保留单步 preview 或 `model.imagine` 薄 wrapper，不是第二个 WM loop。

---

## 14. 一句话版本

> Real/Imagine Bank 只存 OD/LD；固定 16 slot、stable ID、valid mask；`z` 是 frame-dependent full-state latent；TSB 用同 stable-ID 历史 + `delta_pose` 做 temporal，用 OD/LD + ego 做 spatial（不含 nav、不做 geometry projection）；TSB 的 per-entity `h_t` 同时服务 policy 和 WM；policy 用 ego/nav query TSB，policy head hidden 不进入 WM；WM 用 `h_t + action` 预测 latent residual `r_z`，`z_hat_{t+1}=z_hat_t+r_z`，并用单帧 frozen target 监督 OD/LD（当前不启用 EMA）；OD/LD geometry 由 direct absolute decoder 输出；6 个 policy / 6 个 action / 6 个 transition；Plan/aux，无 risk head、无 value head；当前 action exact tracking，LQR 后才区分 requested/executed。
