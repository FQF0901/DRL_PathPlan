# pipeline

目的：训练 / 评估流水线与运行组件（阶段 A/B/C、跟踪、评测、监控、缓冲）。

## 阶段语义（v1.1，详见 `.slim/deepwork/p2-contract.md` §11）
- **A（WM 教师强制）**：ego 条件 = 专家 GT 动作序列 + ego plan 噪声；目标 = `(episode, step+5k)`
  查表重建的未来 OD/LD（t0 对齐 + mask/valid）；直接多步 Huber + 角度 `1-cos`；训练主干+MoE+WM。
- **B（planner BC）**：primary→specific 两段（先 shared+primary，后 8 个 specific）；动作 BC 主项 +
  rollout 轨迹辅助（WM 冻结 + 输出 detach）+ router BCE；产物是阶段 C 的 KL 锚快照。
- **C（PPO RL）**：规则奖励 + GAE；KL 锚到阶段 B 快照（系数线性衰减）；critic warmup
  （前 N 个 update 只拟合 value 头）；primary lr×0.1；WM 初始冻结、`--wm-freeze-updates` 后解冻。
- `stages.py`：阶段编排 + CLI（`--stage A|B|C` 及全部参数）；`tools/train.py` 是薄入口。

## 组件
- `rollout.py`：B1 策略自回归 ×6 + WM 推演；`reference_from_actions` 复用 `env.tracking.interpolate`。
- `trainer.py`：PPO/GAE、BC 预训练、KL/BC 锚、critic warmup；PPO `update()` 走 cheap path
  （`rollout=False, world_model=False`），BC / Stage-B 才跑完整 rollout。
- `buffer.py`：按帧存储 + 在线重建 6 帧历史（不存堆叠，RAM 关键）；GAE(λ) 按 episode 切断。
- `vector_env.py`：spawn 常驻 env 池（spec 按复用键分组）、按 spec 数回收、MemAvailable 下限检查、
  `env.prev_policy_action` 注入；`LocalEnvPool` 供单进程 exact/LQR 路径。
- `eval_runner.py`：冻结 val 集、独立进程、确定性；primary 标签分组 + compound 单列 + Wilson CI +
  弱类绝对 floor；tracker 参考 = 6 步 rollout `plan` → interpolate → 30 点（preview reference）。
- `monitoring.py`：CSV + tensorboard；递归展平 42 条序列（reward 分解 / advantage / value /
  固定探针动作漂移 + 低速告警），另含场景 label 与 MoE 路由统计（有效专家数 Σw、primary 漂移）。

## 可插拔
tracker（exact/lqr）、env 池（local/vector）、监控后端与阶段流程由 `config/train.yaml` 驱动。
