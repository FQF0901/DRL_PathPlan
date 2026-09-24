# reward_model

目的：计算规则奖励与评估 KPI，供训练与评估共用。

## 奖励构成
- 逐步稠密项：安全 / 舒适 / 效率 / 合规四类。
- 终局项：route_completion、arrival、collision、timeout。
- CaRL 风格：违规触发乘性 / 终止型惩罚。
- 进度塑形：基于势能的 shaping，避免刷分。

## 接口
- 奖励项：`compute(state, action, next_state) -> float` + 权重。
- 聚合：按权重求和，可按类别分组统计；KPI 由同一次回放独立计算。

## 可插拔
奖励项与聚合器均经注册表注册；新增项不改动已有项。
