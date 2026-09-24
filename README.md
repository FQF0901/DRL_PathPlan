# DRL_PathPlan

MetaDrive 驾驶规划 RL。当前状态：**P0 完成**（环境 + 五测通过，见 `docs/p0-measurements.md`）；策略输出 (ds, dtheta)（下一 0.5 s 的弧长 + 航向变化），由 MPC/LQR 跟踪；仅物理仿真，无信号灯、无多智能体（暂缓）。

## 目录结构
| 目录 | 用途 |
| --- | --- |
| `config/` | env / model / train / eval 子配置与 default 主配置 |
| `env/` | MetaDrive 封装：scenario / obs / hooks / expert；`env/specs/` 放生成的场景 spec |
| `reward_model/` | 规则奖励项、聚合与 KPI |
| `net/` | GRU → 空间消息传递 → MoE → 世界模型 → 策略头 / rollout |
| `pipeline/` | 阶段 A/B/C、buffer、tracker、eval_runner、monitoring |
| `tools/` | gene_env.sh / train.py / test.py / visualize.py / inspect_dataset.py |
| `tests/` | 骨架测试；`data/`、`runs/` 为运行产物（已忽略） |

## 设计原则
- 非侵入：只包装/继承/钩子，不打补丁，不动系统 Python。
- 可插拔：观测通道 / 奖励项 / 专家 / 场景构建器均走注册表。
- 配置驱动；hook/probe 采集场景标签、MoE 路由与 KPI（tensorboard + CSV）。

## 环境准备（一次性，P0 已验证）
```bash
python3 -m venv --without-pip --system-site-packages .venv
python3 -m pip --python .venv/bin/python install metadrive-simulator
python3 -m pip --python .venv/bin/python install "numpy==1.26.4" "opencv-python-headless==4.10.0.84"  # 钉 numpy<2，替换 opencv-python
bash tools/setup_gl_libs.sh          # 解包 glvnd 运行库到 .venv/gl（本机无系统 libGL，panda3d 需要）
```
> 所有 MetaDrive 相关运行统一用 `tools/venv-python`（自动设置 LD_LIBRARY_PATH），不要直接调 `.venv/bin/python`。

## 快速开始
```bash
tools/venv-python tools/measure/smoke_env.py --seed 1000 --steps 200   # 冒烟 + FPS/RSS
bash tools/gene_env.sh                                                 # 生成 + 校验场景 spec
tools/venv-python tools/train.py --config config/default.yaml --stage A
tools/venv-python tools/test.py --config config/default.yaml --ckpt <ckpt>
```
依赖：MetaDrive 0.4.3、numpy<2、Python 3.10；`.venv` 复用系统已装包（torch 2.3 等）。P0 冻结参数见 `config/env.yaml` / `config/eval.yaml`。
