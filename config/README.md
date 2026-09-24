# config

目的：集中声明环境、模型、训练与评估参数（YAML，键均带中文注释）。

- `default.yaml`：主配置，`includes` 指向以下四个子配置。
- `env.yaml`：MetaDrive / 步长 / 观测 / 场景。
- `model.yaml`：网络结构（GRU、空间消息传递、MoE、世界模型、策略头）。
- `train.yaml`：阶段 A/B/C、数据与监控（`steps_per_epoch: 100000`）。
- `eval.yaml`：验证 spec、评估周期（5 epoch）、KPI 与阈值。

## 验收标准（P0 冻结）

1. 数据集抽检：帧间 ego 位移/航向与记录运动一致、物体速度有界、无时间反转，
   即不出现违背物理时间的现象。
2. 各训练阶段的 loss 与 KPI 落在预先声明的期望区间内
   （阈值见 `eval.yaml` 的 `thresholds`，P0 后冻结）。
