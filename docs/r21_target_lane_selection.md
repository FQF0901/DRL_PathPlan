# R21 target lane 选择（唯一可测试纯函数）

实现：`env/obs/lane_semantic.py::select_target_lane`（无 env、无全局状态；输入
`ego_lane` / `ego_position` / `current_ref_lanes` / `next_ref_lanes`，输出
`TargetLaneSelection{lane, source, route_associated, confidence}`）。
单测：`tests/test_target_lane_continuity.py`。

## 1. 优先级链（不得调换）

```
1. route current_ref_lane
   从 nav.current_ref_lanes 中选"续接最优"者：
   key = (min 端点距离 ||ref.end − next.start||, |Δ lane_id to ego_lane|, lane_id)
   平局规则：ego 所在路由车道与最优续接端点差 ≤ LANE_CONNECT_TOL_M(1.5 m)
             → 保持 ego 车道（防横跳）
   → source=route_current，confidence=1.0
2. route continuation（真实越界 / 无 current_ref 但 ego 在 next 链）
   - 目标车道已被越过末端（投影 long > target.length，nav 尚未更新）且存在 next 链
     → target = next 链上同 lane_id 优先、否则端点最近的续接车道；
   - current_ref 为空且 ego_lane ∈ next_ref_lanes → target = ego_lane；
   → source=route_continuation，confidence=0.8
3. next_ref_lane（无 current_ref 信息）
   同 lane_id 优先 → 距 ego 最近 → lane_id 稳定序
   → source=route_next，confidence=0.6
4. ego.lane 仅兜底（无任何 route 信息）
   → source=ego_fallback，route_associated=False，confidence=0.25
```

无任何车道 → `source=none`，`lane=None`，confidence=0。

**硬约束**：不得沿用 "ego.lane 最高优先"。ego 不在 route 上时 target 仍是 route 车道
（`test_route_current_wins_over_ego_lane`）。

## 2. 稳定性保证（门）

- **横向偏移不切换**：存在 route 信息时选择只依赖 (route 几何, ego 车道 id)，
  **不依赖 ego 横向位置**；±1 m 横向 / ±10° 航向偏移下 selection 逐字段不变
  （`test_lateral_and_heading_offsets_do_not_switch_target`；env 级审计另测）。
- **只有真实 transition 才切换**：
  - 越过 target 车道末端（`long > length`）→ route continuation；
  - 汇入/分流处续接端点差 > 1.5 m 且 ego 在非续接路由车道 → 选续接车道（route 语义，
    触发 `lane_change_required_*`）；
  - 端点差 ≤ 1.5 m（同车道数直行）→ 保持 ego 路由车道。
- **多 current_ref 与 ego 横向无关**（`test_multi_ref_selection_independent_of_ego_lateral_position`）。

## 3. 边界与回退

- `ego_lane` 缺失（罕见）：current_ref 内按横向距离最近；仍无则第一个 ref；
- 投影失败（圆弧 `local_coordinates` 抛 ValueError）：不切换 continuation（保守），
  通道侧记 `target_lane_valid=0`；
- `next_ref_lanes` 缺失（终点路段）：不产生 continuation，`distance_to_target_lane_end`
  = 上限 200 m；
- lane 对象身份比较（`is`）用于去重/角色，lane_id 仅用于**相对**顺序（左右 / 稳定序），
  不进入任何数值特征。

## 4. 与 expert 侧选择的关系

`env/expert/pure_pursuit_idm.py::_plan_target_lane` 是**专家控制策略**（含提前汇入、
静态障碍绕行、变道安全间隙），与本节纯函数**相互独立**：R21 语义通道不消费专家状态，
专家也不消费本通道。两者在汇入/绕行场景的行为差异是 R21-2 "expert current action"
对照的预期来源之一（记录，不强行对齐）。
