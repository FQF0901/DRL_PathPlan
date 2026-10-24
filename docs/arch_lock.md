# Memory Bank + Factorized Temporal-Spatial Attention + Ego-Conditioned Latent WM

> 状态：**owner review draft**，不是最终实现合同。
> 日期：2026-10-24
> 范围：确定 Real/Imagine Memory Bank、BankWindow、TemporalSpatialBlock、OD content/geometry、LD projection+residual 和 ego-conditioned latent WM 的接口语义。
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
  → Plan / aux
  → EgoConditionedLatentWM
       OD content z_hat
       + OD geometry decoder
       + LD projection + residual
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
7. WM 主 target 只有 **OD content**；OD 有 relative-pose decoder；LD 走 **projection + residual**，不预测完整 LD content/polyline。
8. LD 不查询 future map；future map 只作为训练 target/oracle；LD type 从 t0 继承，不预测。
9. 没有独立 risk head，也没有 value head；Plan / aux；collision/offroad/timeout 进 aux。
10. 当前 scope 是“当前可见目标、固定 top-16、短时 rollout”；不考虑短时遮挡，只保留 `valid`。

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
  Nav/Ego：policy 输入，不属于 Bank；包含 route、pose、显式运动学

metadata：
  stable_id / slot / source / depth / timestamp / encoder_version
  valid
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
  node_type: OD | LD
  content: [H]             # online encoder / WM output
  geometry: [G]            # explicit local geometry; frame documented by FrameRecord
  type: int                # OD/LD type, inherited from t0; not predicted
  valid: bool              # numeric/attention/loss validity
```

### 2.3 字段语义

| 字段 | 含义 | 谁维护 |
|---|---|---|
| `slot` | 固定 16 槽中的存储位置，不是身份 | Bank/数据管线 |
| `stable_id` | 实体身份，用于跨帧关联 | env.obs stable ID |
| `valid` | 该 tensor 是否可用于 attention/loss（非 padding、非 NaN） | 数据管线/模型 |
| `source` | REAL 或 IMAGINED | 写入方 |
| `rollout_depth` | 0 或 1..N | 写入方 |
| `type` | OD/LD 类型 | t0 继承；不做未来预测 |

**本版本只保留 `valid`。**
不考虑短时遮挡生命周期；实体不在当前 top-16 或 imagined slot 无效时，直接 `valid=0`。

```text
valid=1   当前帧选中 / imagined slot 可用的实体
valid=0   padding、丢弃、无效 slot
```

### 2.4 固定 16 slot 规则

本版本采用：

```text
OD: 16 slots
LD: 16 slots
Bank 只存 OD/LD；Nav/Ego 由 policy 从当前环境输入单独读取。
```

top-16 的选择归 `env/obs.py`，不是 Bank 或模型：

```text
ROI：
  前方 80 m
  后方 20 m
  左右各 25 m

OD/LD：
  先做 ROI filter
  按到 ego 的距离排序
  取最近 top-16
  不足 16：padding，valid=0
  超过 16：丢弃距离最远的
```

不引入 confidence；选择是确定性的，距离相同用 stable_id 稳定排序。

- 每帧最多 16 个 OD/LD；
- 不足 16 个：padding，`valid=0`；
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
  → valid=1
```

如果没有可复用 slot，则丢弃该实体；这是当前 top-16 scope 的明确限制。

---

## 3. RealMemoryBank 和 ImagineMemoryBank

### 3.1 同构，不同语义

两者使用**同一套 FrameRecord / EntityRecord schema**，这样 BankWindow 和 TemporalSpatialBlock 可以共用；区别只在 `source`、`valid`、`depth` 和写入权限。

### 3.2 RealMemoryBank

```text
容量：6 帧
写入：真实环境 observation
source=REAL
rollout_depth=0
valid=当前真实帧选中且数值有效
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
valid：预测数值有效；无效 imagined slot 为 0
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
- Phase 0/1 用 frozen copy；Phase 2 one-step 稳定后再启用 EMA；
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
- `valid` 按帧保存；
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
input: BankWindow（只含 OD/LD）
       + current Ego/Nav（不属于 Bank）
  ↓
per-entity temporal attention
  query = current frame token
  key/value = same-slot history
  mask = valid & same stable_id
  ↓
current-frame spatial self/cross attention
  OD ↔ LD
  OD/LD 与 current Ego/Nav conditioning
  ↓
current OD/LD entity latents
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
```

供：

- Plan；
- aux；
- OD content WM / OD geometry / LD residual。

Ego/Nav 由 policy head 从当前输入单独读取，不从 Bank 读取。

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
L_od_content = D(z_hat_od, sg(z*_od))
```

- `target_encoder`：独立 frozen/EMA copy，同结构、不同实例；
- `z*`：**只对 OD content**；
- `z_hat_od`：WM 预测的下一帧 OD content；
- stop-grad；
- target encoder 不进入 RealBank，不进入 policy；
- **LD 不做完整 content target**；LD 走 projection + residual。

### 6.2 OD geometry decoder

```text
dec_od(z_hat_od) -> (dx, dy, dψ) in current ego frame
```

- 训练输入以 `z_hat_od` 为主；
- 可加 `z*` 作为辅助 probe，但不能只训练 `z*`；
- geometry target 来自真实下一帧显式 anchor，变换到当前 ego frame；
- 训练/推理一致：rollout 使用 `dec_od(z_hat_od)`；
- **不预测 OD type**；type 从 t0 继承。

### 6.3 LD projection + residual

LD 不用 OD 式全量 decoder。

```text
base_ld_{t+1} = project(polyline_t, ego_pose_t -> ego_pose_{t+1})
r_hat_ld      = ld_residual(z_ld_t, a_t, base_ld_{t+1})
g_hat_ld      = base_ld_{t+1} + r_hat_ld
```

- `polyline_t`：t0/当前帧的显式 lane polygon/polyline；
- `project`：用 ego motion/action 做显式几何传播；
- `ld_residual`：Bank 里的 LD feature 学习的小残差；
- `z_ld_t` 是当前 Bank feature，不是未来 LD content target；
- **不预测 LD type**；type 从 t0 继承；
- 输出在当前 query ego frame。

future map target：

- 只用于训练 loss/oracle；
- 从 MetaDrive 的 `road_network` / lane graph 提取 future vector polyline；
- 不查询 future map 作为 decoder 输入；
- future map 可以包含 t0 没看到的新 lane，但新 lane 没有 t0 base，residual 接不上；这部分不保证覆盖。

### 6.4 训练损失

```text
L_od_content = D(z_hat_od, sg(z*_od))
L_od_geo     = L1(dec_od(z_hat_od), g_od_target)
L_ld_res     = L1(project(base_ld) + ld_residual(z_ld), g_ld_target)
L_total      = λ_z * L_od_content
             + λ_od * L_od_geo
             + λ_ld * L_ld_res
             + ...
```

- OD/LD mask 用 `valid`；
- LD 只监督 residual/geometry，不监督 LD content；
- 不预测 type；
- future map 只在 target，不进入 encoder/decoder 输入；
- 新 lane 无 base 时不监督 residual，或单独标记为 map-completion unknown。

### 6.5 rollout

```text
z_hat_od_{t+1} = WM(z_od_t, a_t)
g_od_{t+1}     = dec_od(z_hat_od)
base_ld_{t+1}  = project(polyline_ld_t, ego_pose_t -> ego_pose_{t+1})
g_ld_{t+1}     = base_ld_{t+1} + ld_residual(z_ld_t, a_t)
ImagineBank.append(
  content=z_hat_od / z_ld_t,
  geometry=g_od / g_ld,
  type=inherited,
  source=IMAGINED,
  rollout_depth=depth+1,
  valid=...
)
```

- Bank **只存 OD/LD**；Nav/Ego 不进入 Bank；
- OD geometry 存 decoded relative pose；
- LD geometry 存 projection + residual；
- type 继承 t0；
- 每一步 geometry 带 frame/ego_pose，之后转到 query frame；
- ImagineBank 只做 FIFO，不反写 RealBank；

### 6.6 对外输出语义

- **无 rollout**：
  ```text
  model.forward -> 仅 t0 policy 结果
  ```
  WM/decoder 只作为训练辅助；不对外输出 t1–t5 policy。

- **有 rollout**：
  ```text
  model.imagine -> policy_0, policy_1, ..., policy_5
  ```
  其中 `policy_n` 表示 **t_n 时刻给出的 action**。
  `a_0` 来自当前 policy/plan；`a_1..a_5` 来自 rollout 中 policy 的输出。

### 6.7 target encoder 阶段策略

```text
Phase 0/1: frozen copy of online encoder
Phase 2: 稳定后启用慢 EMA（如 0.999）
Phase 3: 继续 frozen/EMA，不跟随 GRPO policy 更新
```

- `z*` stop-grad；
- target encoder 同结构、不同实例；
- target 只服务 OD content；
- LD 不依赖 target encoder 的 future content；
- 记录 target drift / latent norm / decoder 误差。

## 7. Heads

```text
Plan
aux
```

- **不设独立 risk head**；
- **不设 value head**；GRPO 使用 group-relative advantage；
- collision / offroad / timeout / urgency / progress 等进入 aux；
- aux 不直接作为 plan 输入，除非明确设计并测试。

---

## 8. 数据流

### 8.1 Real

```text
env current observation（OD/LD）
  → online single-frame encoder
  → content + geometry + stable_id + valid
  → RealBank.append
  → BankWindow
  → TemporalSpatialBlock（OD/LD）
  → Plan/aux；policy 额外读取 current Nav/Ego
  → OD content WM / OD geometry / LD residual
```

### 8.2 Training

```text
real frame t, t+1
  → online encoder for z_t (OD/LD)
  → target encoder for z*_{t+1} (OD content only)
  → WM predicts z_hat_od
  → OD geometry decoder
  → LD projection + residual
  → L_od_content + L_od_geometry + L_ld_residual
```

### 8.3 Imagine

```text
RealBank.snapshot()
  → imagine step
  → WM OD content + OD geometry decoder
  → LD projection + residual
  → ImagineBank.append
  → BankWindow（混合 real prefix + imagined）
  → TemporalSpatialBlock
  → policy_0..policy_5 (t_n action)
  → next action/planning
```

### 8.4 不能做的事

- 不能把 future map 作为 decoder 输入；
- 不能把 `z*` 写进 RealBank；
- 不能用 online encoder 代替 target encoder；
- 不能让 imagined 记录回写 RealBank；
- 不能用 slot 直接当 stable identity；
- 本版本只使用 `valid`。

---

## 9. 验收/测试建议

### 9.1 Bank

- FIFO 容量 6；
- scene/episode 切换 reset；
- RealBank 拒绝 imagined；
- ImagineBank copy-on-write，不回写；
- padding 的 `valid=0`；
- slot 复用后旧历史 temporal mask=0；
- stable_id 变化时 temporal attention 不混历史；
- `source/depth` 正确。

### 9.2 Target encoder

- target encoder 与 online encoder 参数独立；
- target 输出 no grad；
- Phase 0/1 frozen target；Phase 2 one-step 稳定后再启用 EMA；
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

- `dec_od` 在 `z_hat_od` 上训练，不在只在 `z*` 上训练；
- OD relative pose 的 frame 一致；
- LD 是 projection + residual，不输出完整 polyline/type；
- LD type 从 t0 继承；
- future map 置空时 forward 仍可运行（证明没有泄漏）；
- rollout 6 步不越界、不产生 NaN；
- LD residual 超出 t0 base 时明确 `valid=0`，不假装覆盖新 lane；
- 评估 OD content/geometry、LD residual/projection coverage，而不只看 latent MSE。

### 9.5 训练/评测

- 没有 `stack_history` 主路径；
- batch 化 BankWindow；
- future map 只在 target；
- Plan/aux；无独立 risk head、无 value head；
- 所有 artifact 记录 commit、checkpoint、split、source/depth、encoder version。

---

## 10. 与旧方案的关系

- 旧的 `env.obs.stack_history` + `TemporalPool` 不再是主路径；
- 旧的 `InstanceBank` slot 继承 / `InstanceQueue` ID matching 只作为参考，不直接照搬；
- 旧 `arch_lock` 中关于 D1/D2 target encoder 的 EMA 冲突仍需要 owner 决策；本文件不修改合同；
- 旧的 `r3b` / history / WM D1/D2 结果继续只作诊断，不能作为本设计的 gate 证据。

---

## 11. 未决项（需要 owner review）

1. **LD residual 的几何细节**
   - residual 是点级、polyline 参数级，还是中心线级；
   - residual 的尺度和 clamp；
   - 投影 base 后，哪些点参与 loss；
   - 新 lane 没有 t0 base 时如何标记 `valid=0`；
   - 需要 LD residual/projection 的评估协议。

2. **target encoder 的冻结/EMA 时点**
   - Phase 0/1 建议 frozen copy；
   - Phase 2 one-step 稳定后再 EMA；
   - 具体 EMA decay、更新频率、target drift 监控。

3. **LD future map target 的生成**
   - 从 MetaDrive road_network/lane graph 提取 vector polyline，而不是渲染图；
   - future ego pose/ROI 选择 top-16；
   - 只作为训练 target，不能进入 decoder 输入；
   - 新 lane 目标存在但无 t0 base 时，不作为 residual 正样本。

4. **slot eviction**
   - 固定 16 slot 满时丢最远实体；
   - slot 复用时 stable_id 变更与旧历史 mask；
   - 因为不考虑遮挡，`valid` 是唯一状态。

5. **GRPO**
   - reference policy 与 KL 权重；
   - reward model 的验证与安全约束；
   - 动作 postprocess/fallback 后的 log-prob 记录；
   - 是否允许 WM 在 GRPO 期间继续更新。


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
L_policy_il = L_action(policy(z_t, nav_t, ego_t), a_expert)
L_wm       = λ_z * D(z_hat_od_{t+1}, sg(z*_od_{t+1}))
           + λ_od * L1(dec_od(z_hat_od), g_od_target)
           + λ_ld * L1(project(g_ld_t) + ld_residual(z_ld_t), g_ld_target)
L_total    = L_policy_il + L_wm
```

建议第一版：

```text
z*_od = target_encoder(o_{t+1})     # OD only, stop-grad
z_t    = online_encoder(o_t).detach() # 先不让 WM loss 改 policy encoder
a      = expert_action.detach()
z_hat_od -> dec_od                  # 保留 decoder 对 z_hat 的梯度
z_ld -> ld_residual                 # LD 只学 residual，不做 future content target
```

- 不创建 ImagineBank；
- one-step `z_hat` 只用于 WM/decoder loss；
- target encoder 独立，Phase 0/1 frozen；Phase 2 稳定后再 EMA；
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
L_wm = Σ_k w_k * D(z_hat_od_k, sg(z*_od_k))
       + λ_geo Σ_k L_geo(dec_od(z_hat_od_k), g_od_k)
       + λ_ld Σ_k L1(project(g_ld_k) + ld_residual(z_ld_k), g_ld_target_k)
```

其中 `k` 是 rollout depth；每个 depth 都有真实 target，不是只在最后一步算 loss。

建议：

- 开始 detach 每个 step 的 hidden state；
- 稳定后再用 2–3 步截断 BPTT；
- `z*` 永远 stop-grad；
- 每个 depth 记录 vs persistence 的误差；
- LD 只校验 projection + residual，不要求完整 lane 生成；
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

对外输出：

- 无 rollout：只输出 t0 policy 结果；
- 有 rollout：输出 t0..t5 policy 结果，`policy_n` 是 t_n 的 action。

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
| geometry target | detach | target 只作监督 |

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

### 13.1 目标目录

```text
config/
  model.yaml              # 网络结构、Bank 配置、rollout 配置
  env.yaml
  train_bc.yaml
  eval_thresholds.yaml

net/
  mem_bank.py             # EntityRecord / FrameRecord / RealBank / ImagineBank / BankWindow
  encoder.py              # online current-frame encoder + target/EMA encoder
  world_model.py          # TemporalSpatialBlock / EgoConditionedLatentWM / OD geometry / LD residual / rollout
  planner.py              # PlanHead / AuxHead
  model.py                # 顶层组合、forward、imagine 编排
  build_net.py            # config/model.yaml -> 构造 net/model.py 的组件

env/
  action/
    __init__.py
    action_space.py       # 原 net/action_space.py（已迁移）：4D action / PSTL / feasibility / bounds

pipeline/
  losses.py               # 纯 loss 函数
  train_bc.py
  train_wm.py
  train_ppo.py
  eval.py
  rollout.py              # 仅保留 action preview / 旧 wrapper，不是 WM rollout
```

### 13.2 模块职责

| 路径 | 职责 | 不负责 |
|---|---|---|
| `net/mem_bank.py` | Real/Imagine OD/LD 状态、FIFO、SE(2) 补偿、stable-ID/valid、BankWindow | 网络前向、loss、env |
| `net/encoder.py` | 当前帧 online encoder；独立的 frozen/EMA target encoder | Bank 状态、WM rollout |
| `net/world_model.py` | TemporalSpatialBlock、WM（OD content）、OD geometry decoder、LD projection+residual、模型级 rollout | env step、训练循环、loss |
| `net/planner.py` | Plan / aux | Bank、WM、risk head、value head |
| `net/model.py` | 顶层组合、forward、调用 `world_model` rollout | YAML 解析、训练、env |
| `net/build_net.py` | 读 YAML、实例化并返回 `Model` | per-episode Bank 单例 |
| `env/action/action_space.py` | 4D action、PSTL、bounds、feasibility、gate/project | 网络、checkpoint、Bank |
| `pipeline/losses.py` | 纯 loss 函数 | target encoder、数据加载 |
| `pipeline/train_*.py` | target 构造、loss 组合、optimizer、checkpoint | 网络结构定义 |
| `pipeline/eval.py` | 真实环境闭环评测 | imagined rollout |

### 13.3 依赖方向

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

### 13.4 `config/model.yaml` 的边界

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

### 13.5 rollout 归属

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

> **Real/Imagine Bank 只存 OD/LD；固定 16 slot、stable ID、valid mask；TemporalSpatialBlock 做同 ID 时序 attention 和当前帧 spatial attention；WM 主 target 是 OD content，OD 用当前 ego frame 的 relative-pose decoder，LD 用 projection + residual（不查询 future map），type 从 t0 继承；预测结果写入 Imagine Bank 做 6 步 rollout；Plan/aux 不设 risk head 或 value head。**
