# reward_model

目的：规则奖励（训练）与评估 KPI（评测）同口径实现；纯 Python/NumPy，不 import env，可脱离仿真单测。

## 奖励项（`terms.py`，默认权重）
| 项 | 权重 | 口径 |
| --- | --- | --- |
| `route_completion` | 1.0 | 势能塑形 `γΦ(s')−Φ(s)`，γ=1.0 |
| `speed_ratio` | 1.0 | `v / 车道限速` 截断收益（cap=1.0），防爬行 |
| `comfort_lon` / `comfort_lat` | −0.05 / −0.05 | 死区 2.5 / 2.0 后的二次软惩罚 |
| `comfort_jerk` | −0.005 | 死区 5.0 |
| `solid_line` | −2.0 | 跨越连续实线（线型 id 2/3/6/7/8） |
| `speed_limit` | −5.0 | 超速量（默认 5% 容差） |
| `crash` | −10.0 | 任意 `crash*` 标志，终止型 |
| `out_of_road` | −8.0 | 出界，终止型 |

## 聚合（`aggregation.py`）
- 稠密项按权重求和；`shaping=True` 的项可乘 `shaping_decay` 退火（契约 §3 训练后期退火）。
- 终止型项原始值 >0 即终止并记录原因；CaRL 式规则命中时把稠密和乘 `factor`（默认 0，碰撞清零）并可追加惩罚。
- 终局 outcome 仅终局步加一次：`arrive_dest +10` / `collision −5` / `out_of_road −5` / `max_step −2` / `error −5`。
- `credit_assignment="dense"`（PPO/GAE 口径）| `"grouped_discounted"`（GRPO 消融）。

## KPI（`kpi.py`）
- 与 `config/eval.yaml` 同名同序 12 项；按 primary 标签（`spec.labels.geometry`）分组，附带标签单列
  `compound`（仅报告）；每组 n≥30 才判定 + Wilson 95% CI；弱类（baseline<0.7）floor=`max(baseline+0.15, 0.75)`。
- 已验证奖励序（真实 `RewardAggregator`，合成序列）：蠕动 +58.26 < 碰撞 +105.45 < 正常行驶 +251.80，
  蠕动是最差选项 → 已排除"奖励导致 PPO 蠕动"的假设。

接口：奖励项经 `registry.py` 注册（可插拔、单项 `enabled=False` 即关闭）；`compute` 只读 step_ctx
（键与 MetaDrive `info` 对齐），新增项不改动已有项。
