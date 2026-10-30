# env

`env/` 是仿真环境、观测 schema、动作执行和场景定义层。
它不包含训练代码，也不包含数据文件；训练代码在 `pipeline/`，数据在 `data/`。

## 目录

| 路径 | 作用 |
|---|---|
| `env/metadrive_env.py` | MetaDrive 环境封装：按 spec 建图、写限速、安装 events、`build_env()` / reset / step。 |
| `env/obs.py` | **env → net 的唯一观测 schema**（schema v2）。当前帧、raw history、legacy projected history。 |
| `env/action/action_space.py` | 4D action、PSTL 轨迹、动作可行性检查。 |
| `env/action/config.py` | 从 `config/model.yaml` 读取 canonical action bounds/limits。 |
| `env/action/contract.py` | action contract 的统一入口/校验。 |
| `env/tracking.py` | 动作执行层：`ExactTracker`（精确位姿）和 `LqrTracker`（LQR 跟踪）。 |
| `env/scenario/` | 场景 spec：生成、taxonomy、events、validator。 |
| `env/expert/pure_pursuit_idm.py` | expert 基线：pure pursuit + IDM。 |

规范数据路径：

```text
data/specs/source/     # 上游完整场景池
data/specs/            # 固定 train/val/test/mini specs
data/splits/           # 固定 seed manifests
data/episodes/         # 采集后的 NPZ
```

`data/specs` 和 `data/splits` 是一一对应的 canonical 场景集合；不再使用旧的 `env/specs/` 路径。

---

## env 与 net 的交互接口

### 1. 总览

```text
MetaDrive env
  │
  │  env.obs.build_observation()
  ▼
raw current frame dict
  │
  ├─ live: LiveBankRunner
  │    └─ CurrentFrameEncoder.encode_current()
  │         └─ FrameRecord → RealMemoryBank(6) → BankWindow
  │
  └─ offline: train_bc / forward_history
       └─ CurrentFrameEncoder → BankWindow
  │
  ▼
DriveModel.forward_window(window, nav, ego)
  │
  ├─ TemporalSpatialBlock: OD/LD 6-frame history + pose delta + ego
  └─ Planner: temporal tokens + nav + ego
  │
  ▼
normalized action [4]
  │
  ├─ ActionBounds.denormalize(...)
  └─ action_to_trajectory(...)
  │
  ▼
ExactTracker / LqrTracker / env.step
```

**关键边界：**

- `net` 不直接 import MetaDrive，也不直接读 `env.agent`。
- live 路径的状态输入只经过 `env.obs.build_observation()`。
- `net` 实际收到的是 `BankWindow` + `nav` latent + `ego` latent，不是原始 env 对象。
- 动作执行不属于 `net`：`net` 输出 normalized action，`env/action` 和 `env/tracking` 负责转成轨迹并执行。

---

## 2. `env` 给 net 的原始观测

`env/obs.py::build_observation(env, ...)` 返回一个当前帧 dict。
以下是 schema v2 的固定形状：

### OD

```text
od        [16, 18]
od_mask   [16]
od_ids    [16]
od_anchor [16, 3]
od_taxonomy [16, 4]
od_age    [16]
```

`od` 的 18 维：

```text
dx, dy, cos_theta, sin_theta, vx_rel, vy_rel,
length, width, type_id, static_flag, presence,
acc_x, acc_y, age_norm,
semantic_type_id_norm, mobility_type, geometry_type, state_type
```

用途：

- `od` + `od_mask`：实体内容和有效性；
- `od_ids`：Bank 的 stable-ID 对齐，不是 policy 特征；
- `od_anchor`：生成 OD geometry；
- `od_taxonomy`：类型 id；
- `od_age`：已包含在 `od` 的 `age_norm` 中。

### LD

```text
ld        [16, 5, 16]
ld_mask   [16, 5]
ld_ids    [16]
ld_road_ids [16, 2]
ld_anchor [16, 3]
ld_chunk_s [16]
```

每个 lane 有 5 个采样点，点维度 16：

```text
dx, dy, heading_rel, curvature, speed_limit,
left_type, right_type, lane_role, lat_offset,
has_left, has_right, has_prev, has_next,
road_type_norm, is_junction, anchor_s_norm
```

编码器实际读取 `ld`、`ld_mask`、`ld_ids`：
- `ld` 的 5 个点用于 lane content/geometry；
- `ld_mask` 做 lane/point mask；
- `ld_ids` 用于稳定身份。

`ld_road_ids`、`ld_anchor`、`ld_chunk_s` 当前主要用于数据/诊断，不是 policy 的直接 MLP 输入。

### Nav

```text
nav [108]
  = 4 个 route road × 24 维
  + 12 维 global nav
```

每 road 24 维：

```text
entry/mid/exit (dx,dy) / 100,
entry/exit heading_rel,
length / 100, width / 10, lane_count / 8,
turn_sin/cos, out/in_degree / 8, type_id / 15,
is_current, is_next, is_negative, is_toll, is_junction,
route_remaining_ratio, road_progress_frac, valid
```

global 12 维：

```text
route_completion, route_remaining_norm, route_total_norm,
current_road_length_norm, current_road_progress,
remaining_road_count_norm, junction_ahead,
dest_dx, dest_dy, dest_heading_sin, dest_heading_cos, nav_valid
```

`CurrentFrameEncoder.nav_encoder` 把 `[108]` 编成 hidden；
它进入 `Planner`，**不进入 `TemporalSpatialBlock`**。

### Ego

```text
ego      [12]
ego_mask [1]
```

12 维：

```text
speed, a_long, a_lat, yaw_rate, steer, curvature,
prev_dx, prev_dpsi, v_long, v_lat, length_norm, width_norm
```

ego 进入 `TemporalSpatialBlock` 和 `Planner`。
`prev_dx/prev_dpsi` 是上一决策动作反馈，不是独立 action input。

### Pose / metadata

```text
pose [3]       # x, y, heading
meta = {
  schema, schema_version, source,
  scene_seed, timestamp, step, dt
}
```

- `pose` 进入 `BankWindow.ego_pose`，用于 TSB 的 frame delta；
- `pose` 不作为 encoder 的内容特征；
- `scene_seed` 不是 policy 特征；
- `episode_id/frame_id/timestamp` 用于 Bank 组织；
- `meta` 不进入 policy。

---

## 3. `BankWindow`：net 实际收到的历史

live 和 offline 都会先编码成 6 帧 BankWindow：

```text
od_content    [B, T, 16, H]
od_geometry   [B, T, 16, 5]
od_type       [B, T, 16]
od_valid      [B, T, 16]
od_stable_id  [B, T, 16]

ld_content    [B, T, 16, H]
ld_geometry   [B, T, 16, 10]   # 5 points × (x,y)
ld_type       [B, T, 16]
ld_valid      [B, T, 16]
ld_stable_id  [B, T, 16]

ego_pose      [B, T, 3]
source        [B, T]
rollout_depth [B, T]
frame_valid   [B, T]
```

其中：

- `od_content/ld_content` 是 frame-dependent latent；
- `od_geometry/ld_geometry` 给 WM target/decoder 使用；
- `od_stable_id/ld_stable_id` 用于 Bank 的跨帧稳定对齐；
- `ego_pose` 只产生 frame delta，不做实体几何投影；
- `source/rollout_depth` 是 Bank 元数据；
- `frame_valid` 做 padding/无效帧 mask。

### TSB 与 Planner 的分工

```text
TemporalSpatialBlock:
  od_content + ld_content
  + ego_pose frame delta
  + current ego
  + valid/masks/IDs
  → temporal tokens

Planner:
  temporal od/ld tokens
  + nav latent
  + ego latent
  + current valid masks
  → action_mu [4], action_logstd [4], aux [5]
```

`TemporalSpatialBlock` 明确不使用 nav；nav 只进入 Planner。

---

## 4. net 输出与动作执行

### net 输出

```text
action_mu       [B, 4]    # normalized [0,1]
action_logstd   [B, 4]
aux:
  collision, offroad, timeout, urgency, progress
```

当前 policy contract 没有 value head、risk head 或 cost head。
旧 compatible adapter 已删除；`net/` 中没有 value/risk/cost head。

### 动作到 env

```text
normalized action [4]
  → ActionBounds.denormalize(action, clip=False)
  → physical action (dx, dy, dpsi, v_next)
  → action_to_trajectory(...)
  → ExactTracker 或 LqrTracker
  → env.step(...)
```

- canonical bounds/limits 来自 `config/model.yaml`；
- `CLEAN_BOUNDS` 只是兼容别名；
- 反归一化不做隐式 clip；
- 当前 eval/PPO 使用 0.5 s 决策、5 × 0.1 s 子步；
- `ExactTracker` 是精确执行路径，`LqrTracker` 是可选跟踪路径。

动作语义是 4D：

```text
dx, dy, dpsi, v_next
```

其中 `dpsi` 是下一 0.5 s 的航向变化，`v_next` 是下一段速度。

---

## 5. live 和 offline 的接口区别

### Live eval / PPO

```text
env.reset()
  → build_observation()
  → LiveBankRunner.push(obs)
  → CurrentFrameEncoder.encode_current()
  → BankWindow
  → DriveModel.forward_window()
  → action
  → env.step()
```

使用 `pipeline/live_bank.py`，不使用 `stack_history`。

### Offline BC / WM

```text
data/episodes/*.npz
  → train_bc / train_wm load_data()
  → DriveModel.forward_history() 或 BankWindow
```

旧 projected 数据通过 `history_mode=0` 兼容；
新 raw collector 使用 `history_mode=1`，TSB 使用真实 `pose` delta。

---

## 6. env 不提供给 net 的信息

以下信息不会进入 policy：

- `success`、`collision`、`off_road`、`route_completion`
- reward、cost、termination outcome
- expert action / expert audit fields
- `scene_seed`
- `episode_id`、`frame_id`、`timestamp` 的数值内容
- `ld_road_ids`、`nav_road_ids`、`nav_route_id` 的原始 id
- `nav_mask` 目前只产生 `nav_valid`，`DriveModel.forward_window()` 没有继续把它传给 Planner

这些信息要么只用于 Bank 对齐/诊断，要么只用于训练标签、评测统计和数据分析。

---

## 7. 观测历史

- live：`RealMemoryBank` 保存最近 6 帧 raw frame；
- offline collector：`raw_history` 保存 6 帧 raw history；
- legacy 数据：`stack_history` 保存 query-frame projected history，`history_mode=0`；
- 新 collector：`history_mode=1`；
- `DriveModel.forward_window()` 是 live 的 canonical 入口；
- `DriveModel.forward_history()` 是已有 NPZ 数据的兼容入口。

`stack_history` 不再用于 live policy 路径。

---

## 8. 场景 spec

`env/scenario/` 负责：

- `generator.py`：按 seed 生成 block / geometry / traffic / events；
- `spec.py`：`ScenarioSpec`、JSON round-trip、字段校验；
- `taxonomy.py`：geometry、event、difficulty 等标签；
- `behaviors.py`：脚本车、cut-in/cut-out、IDM 交接；
- `validator.py`：场景可实例化性检查；
- `cli.py`：生成 `data/specs/source/train_10000.json` 和 `val_1000.json`。

canonical 数据由 `dataset/freeze_splits.py` 生成：

```text
train_2000   2000 scenes   seeds 1000..2999
val_500       500 scenes   seeds 5000000..5000499
test_500      500 scenes   seeds 5000500..5000999
mini_*        smoke subsets
```

`test_500` 只用于最终闭环验收，不参与训练或调参。

---

## 9. 不要做的事

- 不要在 `net` 中直接读 MetaDrive 对象；状态必须经过 `env.obs`。
- 不要在 live path 使用 `stack_history`。
- 不要绕过 `ActionBounds` / `action_to_trajectory` 直接拼物理动作。
- 不要把 `test_500` 的 episode 数据或指标用于调参。
- 不要把 val/test scene seed 放进 train。
