> **2026-10-26 status update (supersedes details below):** live eval/PPO use
> `RealMemoryBank`/`BankWindow`; collectors write `raw_history`; BC/PPO no
> longer train value/risk/cost heads; `pipeline/losses.py` and bank-native
> `train_wm` are wired; `best_drive.pt` loads through the eval adapter.
> See `docs/deepseekv4p1_argue.md` for the current review status. The older
> details below are retained as history.

# Implementation Status

> 日期：2026-10-24
> 代码基线：`d8d6f28` + 当前工作区变更
> 对照：`docs/arch_lock.md`（唯一合同）
> 状态：**net scaffold 已修复一轮；仍未接入生产 pipeline；没有任何 gate 被提升**

---

## 0. 总结

本轮对 `net/` 的修复主要集中在：

- BankWindow 的 stable-ID → current-slot 对齐；
- encoder 的 pose key / ID / mask / target no-grad；
- TemporalAttention 的 current-valid、历史 pose、stable-ID；
- LD residual base 与 batched `project_ld`；
- WM 的 OD geometry / valid / persistence 初始化；
- `world_model.imagine_rollout`（模块级 rollout）与 `DriveModel.imagine` 薄 wrapper；
- build_net 的配置与维度校验；
- action_space 迁移到 `env/action/`。

**新路径仍不是 live path**：`env.obs.stack_history`、旧 `StructuredPlanner` 和旧 WM 仍被训练/评测代码使用。因此：

```text
new net: implemented scaffold
production: legacy path
gates: unchanged / not promoted
```

---

## 1. 已完成的修复

### 1.1 `net/mem_bank.py`

- `BankWindowBuilder` 现在按 **当前帧 stable_id → 当前 slot** 重排历史：
  - 实体从旧 slot 换到新 slot 时，历史不再丢失；
  - 不是按历史 `rec.slot` 直接覆盖。
- `od/ld` 都按同一规则匹配。
- 重复 stable_id、重复 slot、错误 node_type 均 fail-fast；负 stable_id 是 env.obs 的合法 signed hash，不做符号拒绝。
- `content dtype` 决定 window dtype，不再由 `ego_pose` 的 float64 决定。
- `ImagineMemoryBank`：
  - `rollout_depth > capacity` 报错；
  - snapshot 使用 deep copy，避免 prefix tensor 被外部 in-place 修改。
- 新增 `BankWindow.to_frame_record()` / `BankWindow.advance()`，支持 imagine 的逐帧状态更新。

### 1.2 `net/encoder.py`

- `pose` / `ego_pose` 二者都接受；缺 pose 直接报错，不再静默填零。
- `od_ids` / `ld_ids` 缺失直接报错，不再默认 -1。
- `od_mask` / `ld_mask` 缺失直接报错，不再默认全 valid。
- `_take_last` 按 tensor 维度判断 current/history，不再用 `shape[1] <= 6` 的启发式。
- `TargetEncoder.encode_current` 强制 `@torch.no_grad()`。
- LD type fallback 改为 road type，不再使用 lane_role。
- 输出 `nav_valid` / `ego_valid`。

### 1.3 `net/world_model.py`

- `TemporalAttention`：
  - 正确的 `permute(0,2,1,3)`；
  - 当前帧 invalid 时整行输出为 0；
  - stable-ID mask。
- `TemporalSpatialBlock`：
  - historical token 在进入 temporal attention 前加入 relative pose；
  - 不再把 pose 只加在当前帧 temporal 之后；
  - 不再使用 age/type embedding 作为额外 token 特征；
  - 支持 nav/ego valid mask；
  - `imagine` 支持可选 `nav_seq` / `ego_seq`，默认仍使用当前输入；
  - 校验 slot 数量。
- `project_ld` 支持：
  - 单条 `[10]`；
  - `[N,10]` + `[3]`；
  - `[B,N,10]` + `[B,3]`；
  - `[B,T,N,10]` + `[B,T,3]`。
- `LDResidualDecoder` 接受 `base_ld`，输出 residual；`FactorizedEgoConditionedLatentWM` 输出 `g_ld = base + residual`。
- `ODGeometryDecoder` 作为 relative-pose residual，加在当前 geometry 上。
- `od_delta` 最后一层 zero-init，保留 persistence baseline。
- WM 接受 `od_valid` / `ld_valid`，并输出 validity。
- 新 WM rollout 不再 `@torch.no_grad()`；可做训练期 rollout。

### 1.4 `net/planner.py`

- Plan / aux。
- 没有独立 value head / risk head。
- aux 增加 `urgency`：
  - collision / offroad / timeout / urgency / progress。

### 1.5 `net/model.py`

- `DriveModel.forward_current` 复用 `CurrentFrameEncoder.frame_records`，不再手写重复 FrameRecord 逻辑。
- `world_model.imagine_rollout` + `DriveModel.imagine`：
  - 每个 step 重新跑 TemporalSpatialBlock；
  - OD 用 absolute relative-pose decoder；LD 用 projected base + residual；
  - 使用 `BankWindow.advance` 更新 content/geometry/type/valid/ego_pose/source/depth；
  - 支持 `imagine_bank`，逐步 append 到 ImagineMemoryBank；
  - `imagine_from_real_bank` 使用 RealBank snapshot；
  - 未提供 actions 时，使用上一步 policy 的 action；
  - 支持可选 nav_seq/ego_seq 与 `ImagineBank`；
  - `DriveModel.imagine` 只做参数转发，模型级 rollout 在 `world_model.py`；
  - 支持 `ego_pose` 序列和 `detach`。
- 旧 `StructuredPlanner` 仍保留为 legacy，不用于 `build_net`。

### 1.6 `net/build_net.py` + `config/model.yaml`

- 读取嵌套 `model` / `bank` / `rollout` 配置。
- fail-fast 校验：
  - encoder/temporal/planner/wm hidden 一致；
  - hidden 可被 heads 整除；
  - history_frames/capacity 为 6；
  - od/ld slots 为 16；
  - LD points 与 env.obs 一致；
  - planner/WM action_dim 一致；
- action bounds/limits 仍由 `env/action/action_space.py` 的常量单一真源负责；不保留 config 中的 dead action 段。
- 保留 `bank` / `rollout` runtime 配置，但不创建 per-episode 单例。
- `config/model.yaml` 改为模块化结构。

### 1.7 `net/param_probe.py`

- 已从旧 `StructuredPlanner` 参数账本改为 `build_net` 账本。
- 修复顶层 budget 读取。
- 支持 `python -m net.param_probe` 和 `python net/param_probe.py`。

### 1.8 action_space 迁移

```text
net/action_space.py -> env/action/action_space.py
```

- 新增 `env/action/__init__.py`。
- 所有代码 import / 字符串引用已更新。
- 不再保留 `net/action_space.py`。

---

## 2. 当前参数量

`net/param_probe.py` 输出：

```text
total: 952,794
target: 1,000,000
allowed: 800,000 ~ 1,200,000
within_budget: true
```

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| `encoder` | 136,896 | 14.4% |
| `temporal_spatial` | 546,944 | 57.4% |
| `planner` | 165,901 | 17.4% |
| `world_model` | 103,053 | 10.8% |

主要子模块：

```text
temporal_spatial.temporal_od   66,304
temporal_spatial.temporal_ld   66,304
temporal_spatial.pose          17,024
temporal_spatial.spatial      396,544
planner.plan                  165,256
world_model.od_delta           49,408
world_model.od_geometry        16,899
world_model.ld_residual        18,314
```

**判断：**

- 总参数在预算内；
- temporal-spatial 仍占 57.4%，spatial 396,544 是主要参数集中点；
- world_model 约 103k，占比偏低；
- 当前分布合理与否没有训练/闭环证据，不能视为已优化。

---

## 3. 合同符合度

| 条款 | 状态 |
|---|---|
| Real/Imagine Bank 同 schema | 已实现 |
| 6 帧 FIFO / copy-on-write | 已实现 |
| 固定 16 OD/LD slot | 已实现 |
| stable-ID 历史对齐 | 已实现（当前 slot 顺序） |
| `valid` only | 已实现；没有 `present`/`alive` |
| geometry ego compensation | 已实现 |
| content 不旋转 | 已实现 |
| TemporalAttention 同 ID / current-valid | 已实现 |
| historical token 包含 relative pose | 已实现 |
| no age/type extra token | 已实现（type 只作为 metadata） |
| OD content WM + OD geometry decoder | 已实现 |
| LD projection + residual | 已实现，base/residual 已接通 |
| LD type t0 继承 | 部分：env.obs 没有 `ld_taxonomy`，当前用 road type fallback |
| Plan/aux、无 value/risk | 已实现（旧 legacy 仍有 value/risk） |
| build_net 配置/维度校验 | 已实现 |
| `stack_history` 退出主路径 | **未完成** |
| 生产训练/评测/checkpoint 接入 | **未完成** |
| Target encoder 训练接线 | **未完成** |
| loss / checkpoint / full rollout gate | **未完成** |

---

## 4. 仍存在的重大偏差

1. **live path 仍是 legacy**
   - `train_bc.py`、`eval.py`、`train_ppo.py`、`checkpointing.py`、`train_wm.py` 仍 import 旧 `StructuredPlanner` / 旧 WM。
   - `env.obs.stack_history` 仍在用。
   - 新 `DriveModel` 不能被这些脚本直接训练/评测。

2. **旧类仍在新文件里**
   - `net/model.py` 同时有 `StructuredPlanner`（legacy）和 `DriveModel`（new）。
   - `net/world_model.py` 同时有 `EgoConditionedLatentWM`（legacy）和 `FactorizedEgoConditionedLatentWM`（new）。
   - 旧 `TemporalPool` reshape bug 已修复；legacy 路径仍保留 value/risk 和旧 WM。

3. **训练设施未接**
   - `pipeline/losses.py` 仍不存在；
   - `TargetEncoder` 只用于测试，不进入训练；
   - 没有 OD content / OD geometry / LD residual loss；
   - 没有新 checkpoint schema。

4. **环境边界**
   - env.obs 的 ROI 已改为 x[-20,80], y[-25,25]，LD gate 已改为 s0[-20,80], |lat|≤25；
   - env.obs 的 `pose` 仍是 float64，BankWindow 现在会转换 dtype，但 env 仍不是新 Bank 接口。

5. **imagine 仍有边界**
   - ImagineBank 已可 append；但 ego/nav 的逐步传播仍依赖外部 `ego_pose`，没有内置 ego dynamics；
   - `imagine` 默认仍使用当前 nav/ego；可选 nav_seq/ego_seq 已支持并校验 shape，但没有内置 ego dynamics；
   - 没有正式 multi-step training gate。

---

## 5. 测试

### 定向测试

```text
52 passed
```

覆盖：

- BankWindow / stable-ID reorder / geometry
- encoder / target no-grad
- TemporalAttention / TemporalSpatialBlock
- planner / aux
- build_net / config validation
- imagine / ImagineBank
- action space / kinematics
- checkpointing
- legacy model / WM shape

### 完整测试

```text
106 passed, 1 skipped
1 failed, 7 errors
```

失败/错误仍在 MetaDrive / Panda3D headless 初始化：

- `test_lqr_smoke.py`
- `test_obs_schema.py`

这不是本次 net 改动引入的，但当前环境不能声称全绿。

---

## 6. Gate 状态

| Gate | 状态 |
|---|---|
| A1 exact | 不变 |
| B DAgger/BC | 不变；新 Bank 未接入 |
| C PPO | 不变；旧 v8 不是 clean gate |
| D1 WM | 不变；新 WM 无 D1 证据 |
| D2 WM | 不变；无新的 multi-step probe |
| E/F | 未开始 |
| Test500 | 未运行 |

**本轮没有提升任何 gate。**

---

## 7. 下一步

1. 将 env/obs、collector、eval、PPO、BC 全部切到 RealBank/BankWindow；
2. 接入 `pipeline/losses.py` 和新 TargetEncoder；
3. 更新 checkpoint 到新 `DriveModel`；
4. 移除/隔离 legacy `StructuredPlanner` / `EgoConditionedLatentWM`；
6. 跑严格 net tests + D1/D2 + Val500 + LQR；
7. 在 gate 证据完成前，不宣称新架构可用。
