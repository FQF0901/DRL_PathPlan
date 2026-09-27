# env/specs

目的：生成并冻结的场景 spec（JSON）与实例化校验报告；本目录 `*.json` 已 gitignore，只提交本说明与 `__init__.py`。

## 文件清单（2026-09-25 生成）
| 文件 | 条数 | 用途 |
| --- | --- | --- |
| `scenarios_train.json` | 10,000 | 训练采样池（Stage C / BC 采集来源） |
| `scenarios_val.json` | 1,000 | 冻结验证集（基线冻结与评测，不参与训练） |
| `scenarios_train_slice200.json` | 200 | 冒烟/薄切片训练 |
| `scenarios_val_slice50.json` | 50 | 冒烟评测切片 |

## schema
- 落盘包装 `{"schema_version": 1, "count": N, "specs": [...]}`；spec 字段 11 个：`id/seed/split/blocks/geometry/traffic/limits/nav/ego/difficulty/labels`（见 `env/scenario/spec.py`）。
- 单位：限速与速度 m/s；`traffic.events[*].trigger_step/duration_steps` 为 10 Hz env step；validator 逐条对账
  blocks/geometry/限速/导航/事件触发。

## 重新生成（`tools/gene_env.sh` → `env.scenario.cli`）
    tools/gene_env.sh --slice 200            # 全量 10k/1k + 切片，生成后默认跑实例化校验
    tools/gene_env.sh --no-validate          # 只生成
    tools/gene_env.sh --n-train 2000 --n-val 200 --workers 6

## 用途边界
- 训练从 train spec 按需采样；评测只跑冻结 `scenarios_val.json`；校验失败或改过的 spec 不得用于训练；观测变更后 BC 需重采（`obs_fingerprint` 守卫）。
> 2026-09-27 追加：`scenarios_eval500.json`（500 条分层评测集：来自 val 1000 条、seed=0、按 labels.geometry 原比例配额，provenance 见文件内；配套 500 episode 数据 `datasets/BTC<戳>_expert500val`）与 `scenarios_smoke16.json`（16 条；**非协议**，仅代码冒烟）。
