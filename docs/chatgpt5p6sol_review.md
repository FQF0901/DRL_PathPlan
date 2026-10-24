# Memory Bank + Factorized Temporal-Spatial Attention + Ego-Conditioned Latent WM

> 状态：**owner review draft**，不是最终实现合同。
> 日期：2026-10-24
> 范围：确定 Real/Imagine Memory Bank、BankWindow、TemporalSpatialBlock、OD/LD geometry/type decoder 和 ego-conditioned latent WM 的接口语义。
> 明确不做：本文件不修改代码、不训练、不产生 gate 结论。
> 原则：本文只记录已经讨论并接受的决策；未决项集中放在 §11，不能把“讨论过”当成“已实现”。

---

## 0. 结论摘要

主线收敛为：

```text
单帧 observation
  → 当前帧 encoder
  → RealMemoryBank.append(FrameRecord)
  → BankWindow：固定 6 帧、固定 16 slot
  → TemporalSpatialBlock
       1) per-entity temporal attention
       2) current-frame spatial self/cross attention
  → Plan / Value / aux
  → EgoConditionedLatentWM
       content z_hat
       + OD geometry/type decoder
       + LD polyline/type decoder
  → ImagineMemoryBank.append
  → rollout
```

已接受的关键决策：

1. `env.obs.stack_history` 退出主路径；历史只由 Bank 管理。
2. Real/Imagine Bank 同 schema、同构；`source` 和 `rollout_depth` 区分真实/想象。
3. OD/LD 都使用 **固定 16 slot**；slot 只是存储位置，`stable_id` 才是身份。
4. 时间 attention 只允许同一 `stable_id` 的历史参与；slot 复用时旧历史必须 mask。
5. OD/LD 都保留 content latent；显式 geometry 单独存。
6. Spatial attention 只加入 **relative pose**，不加入 distance / dyaw / velocity 等拆分的手工几何特征。
7. WM 主 target 是 OD/LD content latent；OD 有 geometry decoder，LD 有 polyline/type decoder。
8. LD 不查询 future map；future map 只作为训练 target/oracle。
9. 没有独立 risk head；风险进入 value 或 aux。
10. 当前 scope 是“当前可见目标、固定 top-16、短时 rollout”，不追求完整场景 birth/death 生命周期。

---

## 1. 术语和坐标约定

### 1.1 frame

- `ego frame`：当前时刻自车坐标系；
- `t0 frame`：rollout 起点的自车坐标系；
- `world frame`：可用于诊断/可视化的全局坐标系，不作为网络输入。

Bank 中每个 `FrameRecord` 保存该帧的 `ego_pose`。历史 geometry 在构造 `BankWindow` 时统一变换到 query frame（通常是当前帧 ego frame）。

### 1.2 content / geometry / metadata

```text
content latent：表示“是什么”
  OD：类别、外观、行为语义
  LD：车道/道路结构语义、类型、拓扑属性

geometry：表示“在哪里”
  OD：中心、heading、尺寸、速度等显式 anchor
  LD：polyline、lane heading、chunk/anchor
  Nav/Ego：route、pose、显式运动学

metadata：
  stable_id / slot / source / depth / timestamp / encoder_version
  present / alive / valid
```

**content 不旋转；geometry 才做 SE(2)/SE(3) 投影。**

### 1.3 depth 不是相机深度

本文所有 `depth` 都指：

```text
rollout_depth
  REAL     = 0
  IMAGINED = 1..N
```

它不是相机深度图，也不是 3D 检测 depth。用途：

- 区分真实/想象；
- 控制 rollout horizon；
- 给 imagined loss 做 horizon 加权；
- 诊断误差累积；
- 禁止 imagined 记录回写 RealBank。

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
  entities: tuple[EntityRecord, ...]
```

### 2.2 EntityRecord

```text
EntityRecord
  slot: int                # 0..15 storage bucket
  stable_id: int           # semantic identity
  node_type: OD | LD | NAV | EGO
  content: [H]             # online encoder / WM output
  geometry: [G]            # explicit local geometry; frame documented by FrameRecord
  type: int                # OD class / LD class / EGO/NAV type
  present: bool            # observed in this real frame
  alive: bool              # exists in world/rollout semantics
  valid: bool              # numeric/attention/loss validity
```

### 2.3 字段语义

| 字段 | 含义 | 谁维护 |
|---|---|---|
| `slot` | 固定 16 槽中的存储位置，不是身份 | Bank/数据管线 |
| `stable_id` | 实体身份，用于跨帧关联 | 环境 stable hash / tracker |
| `present` | 当前真实观测是否看到该实体 | Real observation/detector |
| `alive` | 世界/rollout 中是否仍认为实体存在 | tracker / WM rollout |
| `valid` | 该 tensor 是否可用于 attention/loss（非 padding、非 NaN） | 数据管线/模型 |
| `source` | REAL 或 IMAGINED | 写入方 |
| `rollout_depth` | 0 或 1..N | 写入方 |
| `type` | OD/LD 类型 | 当前观测/decoder；静态类型可继承 |

**`present` / `alive` / `valid` 不能合并。**

典型组合：

```text
present=1, alive=1, valid=1   正常可见实体
present=0, alive=1, valid=1   被遮挡/暂不可见，但仍在 Bank/rollout
present=0, alive=0, valid=0   死亡或空 slot
present=1, alive=0, valid=1   不一致，需诊断
present=0, alive=0, valid=0   padding
```

### 2.4 固定 16 slot 规则

本版本采用：

```text
OD: 16 slots
LD: 16 slots
NAV: 1 slot
EGO: 1 slot
```

- 每帧最多 16 个 OD/LD；
- 不足 16 个：padding，`present=0, alive=0, valid=0`；
- 超过 16 个：按当前策略取 top-16，其余不进入 Bank；
- slot 可复用；
- 时间 attention 必须满足：

```text
temporal_valid[t, n] = valid[t, n] & valid[t-1.., n] & (stable_id[t, n] == stable_id[历史, n])
```

否则 slot 复用时会把不同实体混在一起。

> 不采用 variable stable-ID graph nodes 作为当前版本实现。当前 scope 是固定 top-16 当前可见目标；新实体进入、长期遮挡、重识别不建模。

### 2.5 slot 复用

当新实体进入且没有空 slot 时：

```text
选择一个可复用 slot
  → 清空旧 content/geometry/type
  → 写入新 stable_id
  → 所有旧历史 temporal mask=0
  → present=1, alive=1, valid=1
```

如果没有可复用 slot，则丢弃该实体；这是当前 top-16 scope 的明确限制。

---

## 3. RealMemoryBank 和 ImagineMemoryBank

### 3.1 同构，不同语义

两者使用**同一套 FrameRecord / EntityRecord schema**，这样 BankWindow 和 TemporalSpatialBlock 可以共用；区别只在 `source`、`present`、`depth` 和写入权限。

### 3.2 RealMemoryBank

```text
容量：6 帧
写入：真实环境 observation
source=REAL
rollout_depth=0
present=观测值
alive=当前 scope 下由 present/短时跟踪维护
valid=真实数值有效
```

- 上游是**当前帧 online encoder**；
- 不接收 WM 输出；
- 不接受 imagined 记录；
- episode/scene 切换 hard reset；
- append 时默认 detach；
- 不保存 target encoder 的 `z*`。

### 3.3 ImagineMemoryBank

```text
snapshot = RealBank 的 6 帧副本
source：real prefix=REAL，新帧=IMAGINED
rollout_depth：1..6
present：新 imagined 帧为 0
alive：由 rollout 维护
valid：预测数值有效
```

- 开头从 RealBank snapshot；
- 每一步 append 一个 imagined FrameRecord；
- 不修改 RealBank；
- rollout 结束销毁/clear；
- 不把 imagined content/geometry 回写 RealBank。

### 3.4 Real/Imagine 的上游网络

RealBank 的 content 来自：

```text
metaDrive current observation
  → single-frame encoder（online）
  → content latent + explicit geometry + stable ID
  → RealBank.append
```

WM target 的 `z*` 来自：

```text
future real observation
  → independent frozen/EMA target encoder
  → z*（stop-grad）
```

两者关系：

- **同一套网络结构**（通常同 state_dict 形状）；
- **不同实例/参数**；
- target encoder 不能被 online encoder 替代；
- D1/D2 可以是 frozen copy，EMA no-op；Stage E 才启用 EMA；
- `z*` 只进入 WM loss，不进入 RealBank、ST-GNN 或 policy 输入。

---

## 4. BankWindow API

Bank 是 per-episode 状态；训练时由 collate 组成 batched window。

建议接口：

```python
class BankWindow:
    content: Tensor       # [T,N,H]
    geometry: Tensor      # [T,N,G], transformed to query frame
    type: Tensor          # [T,N]
    slot: Tensor          # [N]
    stable_id: Tensor     # [T,N]
    present: Tensor       # [T,N] bool
    alive: Tensor         # [T,N] bool
    valid: Tensor         # [T,N] bool
    source: Tensor        # [T,N] or [T]
    rollout_depth: Tensor # [T] or [T,N]
    ego_pose: Tensor      # [T,3]
```

### 4.1 构造规则

```text
window = bank.window(last=6)
```

- 不足 6 帧：左侧 padding，`valid=0`；
- 历史 geometry 用每帧 `ego_pose` 变换到 query frame；
- content 不旋转；
- `stable_id` 按帧保存；
- `present/alive/valid` 按帧保存；
- `source/depth` 保留；
- 同一 slot 的历史只有在 `stable_id` 相同且 `valid=1` 时才允许进入 temporal attention。

### 4.2 批处理

```text
B 个 episode
  → [B,T,N,H] content
  → [B,T,N,G] geometry
  → [B,T,N] masks / stable_id
```

BankWindow 本身不要求 batch，但训练 collate 必须支持 batch；不能像当前 `BankView` 一样只处理单场景。

---

## 5. TemporalSpatialBlock

### 5.1 结构

```text
input: BankWindow
  ↓
per-entity temporal attention
  query = current frame token
  key/value = same-slot history
  mask = valid & same stable_id
  ↓
current-frame spatial self/cross attention
  OD ↔ ego
  LD ↔ ego
  OD ↔ LD (optional)
  Nav/Ego conditioning
  ↓
current entity latents + ego context
```

这是 factorized temporal-spatial attention，不要求严格图卷积/message passing。

### 5.2 token 内容

```text
token = content latent + relative pose embedding
```

- 只加 relative pose：`(dx, dy, dψ)`；
- **不额外加入** distance、dyaw、velocity、geometry delta 等手工拆分特征；
- relative pose 的参考 frame 固定为当前 query ego frame；
- 如果后续需要 pairwise relative pose，再单独扩展，不在本版本默认实现。

### 5.3 输出

```text
entity_latent: [B,16,H]（OD/LD）
ego_context:   [B,H]
```

供：

- Plan；
- Value；
- aux；
- WM content target / prediction。

### 5.4 不做什么

- 不把 `TemporalPool` 原样保留在主路径；
- 不做 variable stable-ID node graph；
- 不把 historical content 旋转；
- 不在 spatial attention 里加入手工 distance/dyaw/velocity 特征；
- 不用 `stack_history` 作为历史接口。

---

## 6. Ego-Conditioned Latent WM

### 6.1 主 target

```text
z*_{t+1} = target_encoder(o_{t+1})
L_latent = D(z_hat_{t+1}, sg(z*_{t+1}))
```

- `target_encoder`：独立 frozen/EMA copy；
- `z*`：per-entity OD/LD content target；
- `z_hat`：WM 预测的下一帧 content；
- stop-grad；
- target encoder 不进入 RealBank，不进入 policy。

### 6.2 OD geometry decoder

```text
dec_od(z_hat_od) -> (dx, dy, dψ) in current ego frame
type_head(z_hat_od) -> OD type
```

- 训练输入以 `z_hat` 为主；
- 可加 `z*` 作为辅助 probe，但不能只训练 `z*`；
- geometry target 来自真实下一帧显式 anchor，变换到当前 ego frame；
- 训练/推理一致：rollout 使用 `dec_od(z_hat)`。

### 6.3 LD decoder

```text
dec_ld(z_hat_ld) -> polyline [P,2] + type + valid
```

- 输出在当前 ego frame；
- 不是简单的 `(dx,dy,dψ)`；
- **不查询 future ego pose 附近的 map**；
- future map/future LD geometry 只作为训练 target/oracle；
- 推理只使用当前 observation、ego action、Bank 内部状态；
- LD 是静态的，但本版本选择 learned map completion/extrapolation，因为部署时不能依赖 future map query。

### 6.4 训练损失

```text
L = λ_z   * D(z_hat, sg(z*))
  + λ_od  * L1(dec_od(z_hat_od), g_od_target)
  + λ_ld  * L1(dec_ld(z_hat_ld), g_ld_target)
  + λ_type* CE(type_pred, type_target)
  + ...
```

- OD/LD masks 必须进入 loss denominator；
- LD polyline loss 要处理点顺序/重采样；
- future map 只出现在 target，不进入 encoder/decoder 输入；
- type 可以作为静态/aux，不要求每步都预测。

### 6.5 rollout

```text
z_hat_{t+1} = WM(z_t, a_t)
g_od_{t+1}  = dec_od(z_hat_od)
g_ld_{t+1}  = dec_ld(z_hat_ld)
ImagineBank.append(
  content=z_hat,
  geometry=g_od/g_ld,
  type=pred/inherited,
  source=IMAGINED,
  rollout_depth=depth+1,
  present=0,
  alive=...,
  valid=...
)
```

- OD geometry 存 decoded relative pose（或等价的显式 geometry）；
- LD geometry 存 decoded polyline；
- 每一步的 geometry 必须带 frame/ego_pose，之后才能转到 query frame；
- ImagineBank 只做 FIFO，不反写 RealBank。

---

## 7. Heads

```text
Plan
Value
aux
```

- **不设独立 risk head**；
- collision / offroad / timeout / urgency / progress 等进入 aux；
- 如需要 cost-to-go，归入 value 或 aux，不新增 risk 分支；
- aux 不直接作为 plan 输入，除非明确设计并测试。

---

## 8. 数据流

### 8.1 Real

```text
env current observation
  → online single-frame encoder
  → content + geometry + stable_id + present/valid
  → RealBank.append
  → BankWindow
  → TemporalSpatialBlock
  → Plan/Value/aux + WM
```

### 8.2 Training

```text
real frame t, t+1
  → online encoder for z_t
  → target encoder for z*_{t+1}
  → WM predicts z_hat
  → OD/LD decoder
  → L_latent + L_geometry + L_type
```

### 8.3 Imagine

```text
RealBank.snapshot()
  → imagine step
  → WM z_hat + decoders
  → ImagineBank.append
  → BankWindow（混合 real prefix + imagined）
  → TemporalSpatialBlock
  → next action/planning
```

### 8.4 不能做的事

- 不能把 future map 作为 decoder 输入；
- 不能把 `z*` 写进 RealBank；
- 不能用 online encoder 代替 target encoder；
- 不能让 imagined 记录回写 RealBank；
- 不能用 slot 直接当 stable identity；
- 不能把 `present` / `alive` / `valid` 合并成一个 mask。

---

## 9. 验收/测试建议

### 9.1 Bank

- FIFO 容量 6；
- scene/episode 切换 reset；
- RealBank 拒绝 imagined；
- ImagineBank copy-on-write，不回写；
- padding 的 `present=0, alive=0, valid=0`；
- slot 复用后旧历史 temporal mask=0；
- stable_id 变化时 temporal attention 不混历史；
- `source/depth` 正确。

### 9.2 Target encoder

- target encoder 与 online encoder 参数独立；
- target 输出 no grad；
- D1/D2 frozen target，Stage E EMA；
- 不用 target latent 进 Bank/ST-GNN；
- encoder version 可追踪。

### 9.3 TemporalSpatialBlock

- temporal attention 只读同 stable_id；
- history geometry 变换到 query frame；
- content 不旋转；
- B>1、N=16、k<6 padding；
- 不同实体 permutation 不改变当前实体输出；
- spatial attention 不接受手工 distance/dyaw/velocity 输入。

### 9.4 WM / decoder

- `dec_od` 在 `z_hat` 上训练，不在只在 `z*` 上训练；
- OD relative pose 的 frame 一致；
- LD polyline + type 输出 shape/valid；
- future map 置空时 forward 仍可运行（证明没有泄漏）；
- rollout 6 步不越界、不产生 NaN；
- LD 出 t0 范围时 decoder 仍能产生有效 fallback；
- 评估 map completion 的 polyline/type/coverage，而不只看 latent MSE。

### 9.5 训练/评测

- 没有 `stack_history` 主路径；
- batch 化 BankWindow；
- future map 只在 target；
- Plan/Value/aux 无独立 risk head；
- 所有 artifact 记录 commit、checkpoint、split、source/depth、encoder version。

---

## 10. 与旧方案的关系

- 旧的 `env.obs.stack_history` + `TemporalPool` 不再是主路径；
- 旧的 `InstanceBank` slot 继承 / `InstanceQueue` ID matching 只作为参考，不直接照搬；
- 旧 `arch_lock` 中关于 D1/D2 target encoder 的 EMA 冲突仍需要 owner 决策；本文件不修改合同；
- 旧的 `r3b` / history / WM D1/D2 结果继续只作诊断，不能作为本设计的 gate 证据。

---

## 11. 未决项（需要 owner review）

1. **`alive` 的维护规则**
   - RealBank 是否 `alive=present`；
   - 短时遮挡是否保留 alive；
   - ImagineBank 没有 survival head 时是否默认 alive=valid；
   - 死亡后 slot 何时复用。

2. **LD content 是否真的需要 WM 预测**
   - LD 是静态的；
   - 如果 decoder 只做 map completion，content target 是否仍有收益；
   - 需要和“只保留 current LD content + decoder”做 ablation。

3. **LD decoder 的几何表示**
   - polyline 点数、顺序、重采样；
   - type 是每步预测还是 t0 继承；
   - 是否允许 decoder 生成当前观测外的新 lane；
   - 需要 map completion 的评估协议。

4. **relative pose 的参考 frame**
   - 本文件按“相对当前 query ego frame”；
   - 若需要 pairwise relative pose，另立版本；
   - 需要确认 spatial attention 是否只使用 relative pose embedding。

5. **target encoder 的阶段策略**
   - D1/D2 frozen copy；
   - Stage E EMA；
   - 是否允许 target encoder 与 online encoder 完全同结构；
   - target drift/版本检查。

6. **LD future map 的 target 生成**
   - 训练时如何从环境/map 取未来 polyline；
   - 如何处理 t0 观测不到的新 lane；
   - 不能把 future map 误放进推理输入。

7. **固定 16 slot 的 top-k 规则**
   - 按置信度还是按距离/route；
   - 超出 16 个时丢弃谁；
   - slot 复用时 stable_id 分配。

8. **风险表达**
   - 已决定不设独立 risk head；
   - 需要确认 collision/offroad/timeout 在 value 还是 aux，避免双计。

---

## 12. 训练方案与分布漂移控制

> 本节是训练提案，不是 gate 结果，也不表示代码已经实现。
> 目的是把 IL、DAgger、WM 训练、GRPO、LQR 的数据流和 detach 边界定清楚，避免把 policy 分布、WM 分布、动作条件和 rollout 混在一起。

### 12.1 核心原则

1. **policy 数据和 WM 数据分开**
   - policy 用 expert/student/RL 的动作标签或 reward；
   - WM 始终用**真实下一帧 observation 的 latent target**，不能用自己的 imagined latent 当 target。

2. **action condition 用实际执行动作**
   - IL 可以用 expert action；
   - DAgger/RL 应优先用 `a_executed`（经过 policy、PSTL、LQR、fallback 后的实际动作）；
   - 同时记录 requested action 和 executed action，便于诊断。

3. **one-step 先于 multi-step**
   - 第一版 WM 只做 one-step；
   - 没有 rollout 时不需要 ImagineBank；
   - multi-step 先用 horizon curriculum，再进入 policy 使用。

4. **6-step plan 不等于执行 6 步**
   - WM/plan 可以预览 6 × 0.5 s；
   - 真实环境每个 step 只执行第一个 0.5 s；
   - 下一步重新观测、重新 plan，采用 receding horizon。

5. **detach 边界必须显式**
   - target encoder / `z*` / `a_exec` / 初始 `z_t` / policy 对 imagined state 的梯度，默认断开；
   - decoder 对 `z_hat` 的梯度可以保留；
   - multi-step 先 detach，稳定后再考虑短 BPTT。

### 12.2 推荐训练阶段

用户原始顺序是：

```text
IL → DAgger → GRPO → LQR
```

建议把 **WM multi-step curriculum** 插到 GRPO 之前：

```text
Phase 0: IL + one-step WM auxiliary
Phase 1: DAgger + one-step WM
Phase 2: WM horizon curriculum 1→2→3→6
Phase 3: GRPO
Phase 4: LQR / receding horizon
```

如果必须保持 IL → DAgger → GRPO 的顺序，那么至少在 GRPO 使用 WM 之前完成 horizon curriculum；不能让 policy 直接依赖一个只见过 one-step 的 WM。

### 12.3 Phase 0：IL

数据：

```text
RealMemoryBank：真实当前/历史
policy：expert action
WM：expert action + real next latent target
```

训练：

```text
L_policy_il = L_action(policy(z_t), a_expert)
L_wm       = λ_z * D(z_hat_{t+1}, sg(z*_{t+1}))
           + λ_od * L1(dec_od(z_hat), g_od_target)
           + λ_ld * L1(dec_ld(z_hat), g_ld_target)
           + λ_type * CE(type_pred, type_target)
L_total    = L_policy_il + L_wm
```

建议第一版：

```text
z* = target_encoder(o_{t+1})       # stop-grad
z_t = online_encoder(o_t).detach() # 先不让 WM loss 改 policy encoder
a   = expert_action.detach()
z_hat -> decoder                    # 保留 decoder 对 z_hat 的梯度
```

- 不创建 ImagineBank；
- one-step `z_hat` 只用于 WM/decoder loss；
- target encoder 独立，可 frozen；Stage E 再 EMA；
- 如果后续确认 WM auxiliary 对 encoder 有帮助，再以很小系数放开 `z_t` 梯度，并监控 IL loss。

### 12.4 Phase 1：DAgger

数据：

```text
policy 输入：student 访问到的真实状态
policy 标签：expert action
WM 输入：current latent + executed action
WM target：真实 next observation
```

训练：

```text
L_policy_dagger + L_wm_one_step
```

注意：

- DAgger 改变的是 **policy 状态分布**，不是 WM 的 latent teacher forcing；
- WM 仍然用真实 next observation，不改成 free-running predicted latent；
- `a_exec` 用实际执行动作，不使用未执行的 requested action；
- `a_exec.detach()`；
- 低置信度/fallback 动作要记录，不要静默当成正常 expert 动作。

### 12.5 Phase 2：WM horizon curriculum

目的：在 policy 依赖 WM 之前，让 WM 适应自己的预测误差。

```text
horizon: 1 → 2 → 3 → 6
```

训练：

```text
L_wm = Σ_k w_k * D(z_hat_k, sg(z*_k))
       + λ_geo Σ_k L_geo(dec(z_hat_k), g_k)
       + λ_type L_type
```

其中 `k` 是 rollout depth；每个 depth 都有真实 target，不是只在最后一步算 loss。

建议：

- 开始 detach 每个 step 的 hidden state；
- 稳定后再用 2–3 步截断 BPTT；
- `z*` 永远 stop-grad；
- 每个 depth 记录 vs persistence 的误差；
- 如果 depth 2 已经明显差于 persistence，不允许 depth 6 gate。

Phase 2 起才真正启用 `ImagineBank`：

```text
RealBank.snapshot()
  → 每个 imagined step append
  → source=IMAGINED, rollout_depth=k
  → rollout 结束 reset
```

Phase 0/1 的 one-step 只需要 `z_hat`，不需要 ImagineBank。

### 12.6 Phase 3：GRPO

GRPO 不需要 value head，但“只有 reward model 监督 policy”风险较高。

建议：

```text
L_policy = L_grpo
         + λ_kl * KL(policy || reference_policy)
         + λ_aux * aux
```

- `reference_policy` 可用 DAgger 结束时的策略；
- 如果完全不用 KL/BC anchor，要严格监控 entropy、action saturation、fallback、reward hacking；
- reward model 必须有验证集；不能让 reward model 自己无限优化；
- 环境 reward/cost 可用时，优先用环境信号，不把 learned reward 当唯一真值。

GRPO 采样：

```text
同一 state / prompt
  → K 个 policy action
  → 真实环境执行
  → reward
  → group-relative advantage
  → policy update
```

- 不要 value head；
- log-prob 必须对采样动作计算；
- 如果动作经过 gate/fallback/LQR 改写，必须记录 `executed_action` 和 mask；
- 不要对 LQR 后的动作反算 policy log-prob。

WM 在 Phase 3 建议：

- 先 freeze WM；
- 或只做真实 transition 的离线 WM loss；
- 不让 policy gradient 穿过 imagined state 反传 WM；
- 如果要做 model-based RL，必须单独设计，不混进第一版 GRPO。

### 12.7 Phase 4：LQR / receding horizon

每个真实 step：

```text
1. 观测 current
2. WM 想象 6 步 / policy 输出 6 步 plan
3. LQR 只跟踪第一个 0.5 s
4. 下一真实 step 重新观测、重新 plan
```

不要：

```text
6 步 trajectory 开环交给 LQR 执行 3 s
```

如果 policy 输出 4D action，GRPO 的 log-prob 仍针对 4D action；
如果 policy 输出整条 trajectory，动作空间、采样、log-prob、group 都要重新定义，第一版不建议改。

### 12.8 detach 规则

| 边界 | 默认 | 说明 |
|---|---|---|
| `z*` | detach | target encoder 输出，不进梯度 |
| target encoder | no grad | frozen/EMA copy，更新在 optimizer 后 |
| `z_t` → WM | detach | 第一版避免 WM loss 拉坏 policy encoder |
| `a_exec` → WM | detach | 动作是条件/标签，不从 WM loss 反传 policy |
| `z_hat` → decoder | 允许梯度 | decoder loss 需要训练 WM/decoder |
| `z_hat` → policy | detach | 第一版不让 policy 通过 imagined state 训练 WM |
| multi-step hidden | 先 detach | 稳定后再短 BPTT |
| geometry/type target | detach | target 只作监督 |

如果之后要放开 `z_t` 或 imagined state 的梯度，必须单独做对照实验，并监控 policy loss、WM loss、梯度范数和闭环指标。

### 12.9 WM 漂移控制

1. **horizon curriculum**：1 → 2 → 3 → 6；
2. **scheduled sampling**：训练时按概率使用 predicted latent，概率逐步升高；
3. **multi-step loss**：每个 depth 都对真实 target；
4. **residual + normalization**：预测 `Δz`，保留 persistence baseline；
5. **真实重锚定**：每个真实 step 从真实 observation 重新 encode，ImagineBank 不跨真实 step 继承；
6. **action coverage**：expert / DAgger student / RL policy / 扰动动作都要覆盖；
7. **uncertainty/fallback**：高不确定时退回 persistence 或更短 horizon；
8. **geometry consistency**：OD relative pose、LD polyline 每步校验；
9. **receding horizon**：只执行第一段，避免开环 3 s。

### 12.10 policy 漂移控制

- IL → DAgger：用 student 状态上的 expert 标签；
- DAgger → GRPO：保留 KL/BC anchor；
- 训练覆盖 fallback、LQR tracking error、低置信度状态；
- 不要只在干净 IL 状态上训练 policy；
- GRPO 的 reward 必须和真实 safety/cost 对齐，避免只优化 reward model。

### 12.11 训练 gate 建议

- Phase 0：one-step content/geometry 不低于 persistence，decoder 有效；
- Phase 1：DAgger 在 student 状态上不崩，fallback 可追踪；
- Phase 2：depth 1/2/3/6 误差曲线单调可接受，且明显优于 persistence；
- Phase 3：GRPO 不出现 entropy collapse / saturation / reward hacking；
- Phase 4：LQR 只执行第一段时闭环稳定，不出现长时间 open-loop 跟踪漂移。

### 12.12 仍未决

- `z_t.detach()` 是否长期保留，还是 WM 稳定后放开小梯度；
- WM loss 与 policy loss 的权重/调度；
- target encoder 从 Phase 0 就 frozen，还是先 EMA；
- action condition 用 requested 还是 executed（本文件建议 executed）；
- GRPO 的 reference policy / KL 权重；
- reward model 的验证与安全约束；
- 6-step plan 与 LQR preview 的正式接口。

---

## 13. 文件/文件夹管理（owner approved）

> 这是目标结构，不代表本轮已经修改代码。`action_space.py` 的实际迁移在后续单独任务中完成；迁移前保持 pending，不能出现两份 live 实现。

### 12.1 目标目录

```text
config/
  model.yaml              # 网络结构、Bank 配置、rollout 配置
  env.yaml
  train_bc.yaml
  eval_thresholds.yaml

net/
  mem_bank.py             # EntityRecord / FrameRecord / RealBank / ImagineBank / BankWindow
  encoder.py              # online current-frame encoder + target/EMA encoder
  world_model.py          # TemporalSpatialBlock / EgoConditionedLatentWM / OD-LD decoders / rollout
  planner.py              # PlanHead / ValueHead / aux heads
  model.py                # 顶层组合、forward、imagine 编排
  build_net.py            # config/model.yaml -> 构造 net/model.py 的组件

env/
  action/
    __init__.py
    action_space.py       # 原 net/action_space.py：4D action / PSTL / feasibility / bounds

pipeline/
  losses.py               # 纯 loss 函数
  train_bc.py
  train_wm.py
  train_ppo.py
  eval.py
  rollout.py              # 仅保留 action preview / 旧 wrapper，不是 WM rollout
```

### 12.2 模块职责

| 路径 | 职责 | 不负责 |
|---|---|---|
| `net/mem_bank.py` | Real/Imagine 状态、FIFO、SE(2) 补偿、stable-ID/mask、BankWindow | 网络前向、loss、env |
| `net/encoder.py` | 当前帧 online encoder；独立的 frozen/EMA target encoder | Bank 状态、WM rollout |
| `net/world_model.py` | TemporalSpatialBlock、WM、OD/LD decoder、模型级 rollout | env step、训练循环、loss |
| `net/planner.py` | Plan / Value / aux | Bank、WM、risk head |
| `net/model.py` | 顶层组合、forward、调用 `world_model` rollout | YAML 解析、训练、env |
| `net/build_net.py` | 读 YAML、实例化并返回 `Model` | per-episode Bank 单例 |
| `env/action/action_space.py` | 4D action、PSTL、bounds、feasibility、gate/project | 网络、checkpoint、Bank |
| `pipeline/losses.py` | 纯 loss 函数 | target encoder、数据加载 |
| `pipeline/train_*.py` | target 构造、loss 组合、optimizer、checkpoint | 网络结构定义 |
| `pipeline/eval.py` | 真实环境闭环评测 | imagined rollout |

### 12.3 依赖方向

```text
mem_bank.py
  ↑
encoder.py          # 可产生 FrameRecord
  ↑
world_model.py      # 只依赖 mem_bank 的数据类型；rollout 接收 action/action_provider
  ↑
planner.py

model.py            # 组合 encoder + world_model + planner
build_net.py        # 读 config/model.yaml，构造 model.py 的组件
```

约束：

- `mem_bank.py` 不 import `encoder/world_model/planner/model`；
- `world_model.py` 不 import `planner/model/build_net`；
- `model.py` 只做编排，不实现 encoder/WM/planner 细节；
- `env/action/action_space.py` 不被 `mem_bank.py` 或 `world_model.py` 反向依赖；
- `pipeline/losses.py` 不 import `model.py`，只接收预测/target；
- `net/model.py` 保留为顶层组合，不改名为 `drive.py`。

### 12.4 `config/model.yaml` 的边界

`config/model.yaml` 负责：

- `encoder` / `temporal_spatial` / `world_model` / `planner` 的结构与超参数；
- `bank.capacity`、`bank.od_slots`、`bank.ld_slots`；
- `rollout.horizon` 和 decoder 开关；
- `action` 的 bounds/limits（仅配置；不 build `nn.Module`）。

`build_net.py` 负责：

- 读取 YAML；
- 实例化子模块；
- 校验维度；
- 返回 `Model`。

不负责：

- 创建 per-episode `RealMemoryBank` 单例；
- 创建 target encoder 的 EMA 状态作为 inference 参数；
- 训练循环 / loss / env step。

### 12.5 rollout 归属

- 模型级 rollout：`net/world_model.py`；
- 顶层调用：`net/model.py` 的 `imagine` / `rollout` 方法；
- 训练/评测调度：`pipeline/train_wm.py`、`pipeline/eval.py`；
- **不放** `pipeline/imagination.py`，也不在 `pipeline/rollout.py` 里复制一套 WM loop。

`pipeline/rollout.py` 只保留：

- 单步 action preview；
- 或对 `model.imagine` 的薄 wrapper。

它不是第二个 rollout 真源。

---

## 14. 一句话版本

> **Real/Imagine Bank 以同构的 per-entity content + explicit geometry + stable ID + masks 存储短时状态；固定 16 slot；TemporalSpatialBlock 做同 ID 时序 attention 和当前帧 spatial attention；WM 以独立 frozen/EMA target 的 OD/LD content 为监督，OD 用 relative-pose decoder，LD 用 polyline/type decoder（不查询 future map），预测结果写入 Imagine Bank 做 6 步 rollout；Plan/Value/aux 不设独立 risk head。**
