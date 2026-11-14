# pipeline layout

`pipeline/` is the orchestration layer.  Training entry points are separate
because their data/contracts differ; shared utilities live in `common/`.

| 目录 | 内容 | 状态 |
|---|---|---|
| `dataio/` | `collect_expert.py`、`collect_dagger.py`、`freeze_splits.py`、`splits.py` | active |
| `common/` | checkpointing、distributions、execution、live_bank、train_config | active |
| `eval/` | `closed_loop.py`、`metrics.py`、`wm_probe.py`；`python -m pipeline.eval` / `pipeline.eval.wm_probe` | active |
| `losses.py` | 纯 policy/WM loss（单文件，不强行做包） | active |
| `train/` | `bc.py`、`wm.py`、`rl.py`、`rl_reward.py`、`scene_batches.py` | active（RL 线已停止，代码保留） |

命名规则：
- `pipeline.train.*` = 训练入口
- `pipeline.eval.*` = 评测/诊断
- `pipeline.common.*` = 无业务入口的共享工具

命令：

```bash
python -m pipeline.eval --help
python -m pipeline.eval.wm_probe --help
python -m pipeline.train.bc --help
python -m pipeline.train.wm --help
python -m pipeline.train.rl --help
```

已删除的旧文件（`train_ppo.py`、`train_joint.py`、`select_checkpoint.py`、
`dataset/rl_splits.py`、`common/batching.py`）不再保留兼容 shim；需要历史版本时用 git。
