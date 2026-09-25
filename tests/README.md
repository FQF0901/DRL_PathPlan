# tests

目的：骨架 / 契约 / 回归测试。优先纯 CPU 离线；涉及 metadrive 的依赖延迟到用例内导入。

运行：

    tools/venv-python -m pytest tests/ -q      # 当前 88 passed

裸 `pytest` 可能解析到系统 Python（无 metadrive），一律用 `tools/venv-python`（venv + system-site-packages）。

| 文件 | 覆盖 |
| --- | --- |
| `test_skeleton.py` | 5 个配置文件存在且可被 YAML 解析 |
| `test_net_shapes.py` | net 形状 / `hist_valid` 门控 / mask / MoE / WM 损失 / 运动学 / 参数预算 |
| `test_od_pose_grad.py` | 空 OD 槽位的 `atan2(0,0)` 不得以 NaN 污染 `traj_xy` 梯度 |
| `test_ego_prev_action.py` | ego 末 2 维 prev_action 注入（reserved），未注入/非法值保持 0 |
| `test_reward_terms.py` | 奖励项 / 势能塑形不变性 / CaRL / KPI 分组与判定 |
| `test_kpi_grouping.py` | primary 分组 / compound / Wilson CI / 弱类 floor（纯函数） |
| `test_buffer_gae.py` | 按帧缓冲 + 在线历史重建（SE(2)、`hist_valid`）+ GAE(λ) 手算对照 |
| `test_bc_pretrain.py` | BC 目标对齐（`traj6` 取点）、动作损失非零、sigmoid 参数化与 logprob 契约 |
| `test_collect_fast_path.py` | PPO cheap path（`rollout=False, world_model=False`）与完整前向逐位一致 |
| `test_collect_expert_parallel.py` | `collect_expert --workers` 轮转分配与合并的确定性（不建 env） |
| `test_stage_v11.py` | GL 路径守卫、阶段 A 未来窗口 stride 查表、B 的 WM detach 因果链、freeze 前缀 |
| `test_critic_warmup.py` | warmup 只拟合 value 头、策略/主干逐位不变、第 N+1 次恢复 PPO |

约定：断言以契约为准（形状、门控、因果链、判定逻辑），非快照式回归。
