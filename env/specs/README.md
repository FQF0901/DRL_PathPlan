# env/specs

目的：存放生成并冻结的场景 spec（JSON）。本目录 `*.json` 已 gitignore（`.gitignore` 中的 `env/specs/*.json`），仓库只提交本说明与 `__init__.py`。

## 文件清单（条数为各文件 `count` 字段实测值）
| 文件 | 条数 | 来源/用途 |
| --- | --- | --- |
| `scenarios_train.json` | 10000 | `tools/gene_env.sh` 生成；训练采样池（`config/train.yaml::data.spec`，`:51`） |
| `scenarios_val.json` | 1000 | 同上；冻结验证集（不参与训练） |
| `scenarios_train_slice200.json` / `scenarios_val_slice50.json` | 200 / 50 | `--slice 200` 切片（冒烟）；切片为独立分层生成，非全量前缀子集 |
| `scenarios_train_5k.json` | 5000 | `tools/make_dagger_pools.py --mode pool5k`：train ∩ expert5k 覆盖集（`config/train.yaml:95` 的 phase 3 池） |
| `scenarios_train_dagger_r1.json` / `_r2` / `_r3` | 各 500 | `tools/make_dagger_pools.py`（默认 rounds 模式）：3 轮互不重叠 DAgger 采集池 |
| `scenarios_eval500.json` | 500 | `tools/make_eval_spec.py --write`：val 分层抽样（`config/eval.yaml::eval.spec`，`:3`） |
| `scenarios_smoke16.json` | 16 | 同上；**非协议**，仅代码冒烟 |

## schema
- 落盘包装 `{"schema_version": 1, "count": N, "specs": [...]}`，部分文件附 `provenance`（来源/配额/隔离断言）；`load_specs` 兼容裸列表（`env/scenario/spec.py:322-349`）。
- spec 字段 11 个：`id/seed/split/blocks/geometry/traffic/limits/nav/ego/difficulty/labels`（`spec.py:66-78`）；单位与取值约束见 `env/scenario/README.md`。

## 生成/再生成命令
    bash tools/gene_env.sh --slice 200            # 全量 10k/1k + 切片（生成后默认跑实例化校验）
    bash tools/gene_env.sh --no-validate          # 只生成
    bash tools/gene_env.sh --n-train 2000 --n-val 200 --workers 6
    tools/venv-python tools/make_eval_spec.py --write          # eval500 + smoke16（默认 DRY-RUN，--write 才落盘）
    tools/venv-python tools/make_dagger_pools.py               # dagger r1–r3（默认 rounds 模式）
    tools/venv-python tools/make_dagger_pools.py --mode pool5k # 5k 覆盖整池

## 用途边界（代码可查）
- train/val seed 区间重叠会被生成器拒绝（`env/scenario/generator.py:477-480`）。
- 5k / dagger 池生成时断言与 `scenarios_eval500.json`、`scenarios_val.json` 无交集（`tools/make_dagger_pools.py:67-73, 119-125`；`:151-155` 为三轮互不重叠断言）。
- `tools/gene_env.sh` 默认生成后做实例化校验，报告写 `validation_<name>.json`（`env/scenario/cli.py:97-119`）；校验失败或改过的 spec 不得用于训练。
- 观测变更后 BC 数据需重采（`obs_fingerprint` 守卫，`env/obs/__init__.py:1-6`）。
