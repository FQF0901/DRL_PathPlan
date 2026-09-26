# tools

目的：命令行入口。统一用 `tools/venv-python`（venv + system-site-packages）运行；裸 `python3` 可能解析到无 metadrive 的系统解释器。

| 工具 | 用途与关键参数 |
| --- | --- |
| `gene_env.sh` | 生成 + 实例化校验场景 spec：`--slice [N]`、`--n-train/--n-val`、`--workers`、`--no-validate` |
| `train.py` | 分阶段训练：`--stage A|B|C`、`--ckpt`、`--bc-dir`；B 用 `--traj-aux-weight`（已验证 0.3）；C 用 `--critic-warmup-updates`、`--kl-anchor-coef`、`--pool` |
| `test.py` | 冻结 val 集评测：`--policy baseline|ckpt`、`--ckpt`、`--tracker exact|lqr`、`--limit N`、`--workers 2`、`--name` |
| `collect_expert.py` | BC 采集：`--specs`、`--workers N`（峰值 ≈1.3GB/worker）、`--recycle-every 150`、过滤（终末截断/on_lane/round-trip）与 `--balance weights|cap|none` |
| `baseline_eval.py` | 规则基线批量评测：`--specs`、`--workers`、`--out`、`--render`（调试，强制单进程） |
| `visualize.py` | 场景俯视图抽检：`--specs` + `--random N`/`--ids`/`--all-in-file`、`--policy baseline|idle`、`--frames/--steps` |
| `measure/smoke_env.py` | headless 冒烟：reset 耗时 / FPS / RSS（≥150 FPS 口径） |
| `measure/check_determinism.py` | 同种子跨进程/重复建图的确定性（稀疏指纹对比） |
| `measure/obs_smoke.py` | 观测通道 shape/mask + 单步耗时分解（obs/physics/traffic/reward） |

约定：入口 import 时无副作用（metadrive/torch 延迟到 `main()`）；未识别参数原样透传给下游模块。
