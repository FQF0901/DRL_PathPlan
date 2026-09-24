# tools

目的：命令行入口（P0 为占位，P1/P2 实现）。

| 脚本 | 用途 |
| --- | --- |
| `gene_env.sh` | 调 Python 生成场景 spec 到 `env/specs/` |
| `train.py` | 分阶段训练：`--config --stage {A,B,C}` |
| `test.py` | 评估：`--config --spec --ckpt --out` |
| `visualize.py` | 回放 / 轨迹可视化：`--ckpt --out` |
| `inspect_dataset.py` | 数据集抽检：物理时间一致性（位移 / 航向 / 速度上界） |

- 依赖 `.venv`（`--system-site-packages`）与 `config/default.yaml`。
- 所有脚本 import 时无副作用；未实现功能抛 `NotImplementedError("P1/P2")`。
