# pipeline

目的：训练 / 评估流水线与运行组件。

## 阶段
- A：BC 预热（专家 = IDMPolicy）→ PPO（精确跟踪）。
- B：离线世界模型训练，复用阶段 A 回放，不与环境交互。
- C：联合微调：环境内 MPC/LQR 跟踪器、弱路由监督、KL 锚点、主专家 lr ×0.1。

## 组件
- rollout：策略自回归 ×6。
- trainer：阶段调度与优化；buffer：回放缓冲区。
- tracker：MPC/LQR 跟踪器；eval_runner：独立进程确定性评估，
  每 5 epoch 一次，worker 每 1000 次 reset 重启。
- monitoring：tensorboard + CSV；逐步场景标签与 MoE 路由统计
  （有效专家数 = 路由权重和、各专家权重 / 输出范数、token 数）。

## 可插拔
trainer / tracker / 监控后端按接口替换；阶段流程由 `train.yaml` 驱动。
