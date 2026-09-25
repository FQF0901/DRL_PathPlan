# env/obs

目的：把环境状态编码为定长观测（含 6 帧历史），供网络直接消费。

## 组装与通道
- `builder.py::ObservationBuilder.build(env, spec)`：按 config 组装通道 + 历史堆叠；`episode_step==0` 时自动
  `reset()`；`register(channel)` 挂载自定义通道。`base.py` 定义 `ObservationChannel`
  （`build -> (features (N,F), mask (N,))`）与 `FrameAlignment`（SE(2) 对齐；自车系 **x 前向 / y 左向**）。

| 通道 | 形状 | 内容 |
| --- | --- | --- |
| ego | (1,8) | v, a_long, a_lat, yaw_rate, steer, curvature + 末 2 维上一策略步 `(ds,dθ)` |
| od | (16,9) | dx,dy,vx,vy,cosθ,sinθ,L,W,type_id（自车系，vx/vy 为相对速度） |
| ld | (16,7) | dx,dy,heading_rel,curvature,speed_limit,left/right_line_type_id |
| nav | (1,11) | 2 checkpoint（自车系）+ 6 命令 one-hot（3 实际+3 保留）+ route_completion |
| signal | (1,4) | 占位恒 `[0,0,0,1]`（无交通灯，mask=1） |

- OD/LD 为**盒式 scope**（前 100 / 后 50 / 左 25 / 右 25 m）；OD 排序键 = `min(TTC, 5 s)` 后距离；不足
  **零填充 + mask=0**，不做全图 top-k（LD 按 offset 环填充，见 `ld.py`）。
- 历史（`memory.py`）：6 帧 @0.5 s（每 5 个 env step 1 帧），`stack` 时 SE(2) 对齐**当前** ego 系；预热复制
  最旧真实帧并用 `hist_valid` 标 0（`*_hist_mask` 仍=1）。prev_action：调用方写 `env.prev_policy_action=(ds,dθ)`
  → `ego.py` 写入末 2 维；未写入保持 0。

## 版本守卫与可插拔
- `obs_fingerprint()`（本包 `__init__.py`）= `env/obs/*.py` 内容哈希（12 hex）；BC meta 记录，`BCDataset.load` 不一致时告警——观测变更后必须重采 BC 数据；新增通道实现 `ObservationChannel` 并 `builder.register()`。
