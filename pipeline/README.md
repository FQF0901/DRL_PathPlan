# pipeline layout

`pipeline/` is the orchestration layer.  Training entry points are separate
because their data/contracts differ; shared utilities live in `common/`.

| 目录 | 内容 | 状态 |
|---|---|---|
| `common/` | checkpointing、distributions、execution、live_bank、train_config、batching | active |
| `eval/` | `closed_loop.py`（闭环 eval）、`metrics.py`（汇总/CI）；`python -m pipeline.eval` | active |
| `losses/` | 纯 policy/WM loss | active |
| `rl/` | `reward.py`、`train_group.py`（real-env group-relative PPO） | active（RL 线已停止，代码保留） |
| `train/` | `bc.py`、`wm.py`、`scene_batches.py` | active |
| `wm/` | `probe.py`（WM 自回归诊断，不是 pytest） | active research |

已删除的旧文件（`train_ppo.py`、`train_joint.py`、`select_checkpoint.py`、
`dataset/rl_splits.py`）不再保留兼容 shim；需要历史版本时用 git。

命令：

```bash
python -m pipeline.eval --help
python -m pipeline.rl.train_group --help
python -m pipeline.train.bc --help
python -m pipeline.train.wm --help
python -m pipeline.wm.probe --help
```
