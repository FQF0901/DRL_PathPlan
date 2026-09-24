# env/hooks

目的：在不侵入环境逻辑的前提下采集指标、标签与可视化数据。

## 接口
- `on_reset(env, info)`：重置后初始化统计。
- `on_step(env, obs, action, reward, info)`：逐步采集。
- `on_episode_end(env, info)`：汇总并交给 monitoring 落盘。

## 采集内容
- 指标：KPI 原始量（碰撞、越界、TTC、加速度、jerk 等）。
- 标签：每步场景标签，用于场景分布与 MoE 路由分析。
- 可视化：轨迹 / 帧快照，供 `tools/visualize.py` 使用。

## 约定与可插拔
- 多钩子经注册表挂载，彼此独立；钩子异常只记日志、不中断训练。
- 新增钩子实现上述三个回调即可。
