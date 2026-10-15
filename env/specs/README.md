# env/specs

目的：存放生成并冻结的场景 spec（JSON）。本目录 `*.json` 已 gitignore（`.gitignore` 中的 `env/specs/*.json`），仓库只提交本说明与 `__init__.py`。

## 文件清单（条数为各文件 `count` 字段实测值；状态：KEEP / 预留）

| 文件 | 条数 | 用途 | 消费者（代码可查） | 状态 |
| --- | --- | --- | --- | --- |
| `scenarios_train.json` | 10000 | 训练采样池 | `config/train.yaml:51`（`data.spec`）；`pipeline/stages.py:250` 兜底；`tools/make_eval_spec.py` / `make_dagger_pools.py` 来源 | KEEP |
| `scenarios_val.json` | 1000 | 冻结验证集（不参与训练） | `pipeline/eval_runner.py:6,1751`（缺省评测 spec）；`tools/dagger_collect.py:133` 隔离守卫 | KEEP |
| `scenarios_train_slice200.json` | 200 | train 冒烟/短跑切片（独立分层生成，非全量前缀子集） | `pipeline/trainer.py:8337` 默认；`tools/visualize.py:825` 默认；`tools/train.py` / `collect_expert.py` / `dagger_collect.py` 示例；P0-B 采集起步池（R37 P0-B 审批包 §2.1；已归档，取档见 `docs/repo_handbook.md` 附录 D） | KEEP |
| `scenarios_train_5k.json` | 5000 | phase-3 采集整池（train ∩ expert5k 覆盖集） | `config/train.yaml:95`（`spec_pool`）；`pipeline/phase3_loop.py:435` 默认；`tools/reward_audit*.py` | KEEP |
| `scenarios_train_dagger_r1.json` | 500 | 三轮互不重叠 DAgger/RL 采集池之 1 | `config/arms/v7_arm1_offroad.yaml:6`（pin）；v6/v7 预注册 pin（要点见 `docs/repo_handbook.md` 附录 D）；recovery_v1 采集池（已执行，`tools/controls/collect_recovery.py:66,76`） | KEEP |
| `scenarios_eval500.json` | 500 | 统一评测集（val 分层抽样；与 train 系列零交集） | `config/eval.yaml:3`（主）；`config/train.yaml:151`（探针）；`pipeline/phase3_loop.py:444`；`pipeline/run_paths.py:78`；`tools/test.sh:19` | KEEP |
| `scenarios_clean500.json` | 500 | **Clean500 晋级集 canonical 副本**（= `scenarios_val.json` − `scenarios_eval500.json`；sha256 `087db3f5…` 与历史真源逐字节一致） | R39-A3 / `round38/c6_*`（统一基准）；门定义见 `docs/repo_handbook.md` | KEEP |
| `scenarios_smoke16.json` | 16 | 代码冒烟（**非协议**） | `tests/test_dagger_collect.py:390,776,885` | KEEP |

## 已删（CLEAN-2，2026-10-15；本目录 JSON 不入 git，直接删本地文件）

| 文件 | 条数 | 原用途 | 删除依据（零现役消费者） |
| --- | --- | --- | --- |
| `scenarios_val_slice50.json` | 50 | 旧验证切片（早期冒烟/取证） | 仅历史/注释引用：`tools/diagnostics/forensics_closed_loop.py:96` 为陈旧默认值（其默认 ckpt 亦早已缺失）、`tools/diagnostics/README.md:8`（同款陈旧文档）、归档版 `experiments.md`（已进 git 历史）、`runs/` 历史产物；无现役评测消费者 |
| `scenarios_train_dagger_r2.json` | 500 | 三轮互不重叠采集池之 2 | 无现役消费者；recovery_v1 采集已执行完毕；DAgger 后续轮计划池 = slice200 → 5k（R37 P0-B 审批包 §2.1；已归档，取档见 `docs/repo_handbook.md` 附录 D）；`tools/controls/collect_recovery.py:77` 为已执行命令的历史文档行 |
| `scenarios_train_dagger_r3.json` | 500 | 三轮互不重叠采集池之 3 | 同上（历史文档行 `collect_recovery.py:78`） |

> 重生成：`bash tools/gene_env.sh --slice 200` 会同时再生成 `scenarios_train_slice200.json` + `scenarios_val_slice50.json`；`tools/venv-python tools/make_dagger_pools.py`（默认 rounds 模式，固定种子）会同时再生成 r1–r3 + 5k。按需使用、避免误覆盖。

## 新增/登记规则（CLEAN-2 起）

1. **命名**：`scenarios_<split>[_<tag>][_sliceN|_5k].json`——`<split>` ∈ {train, val}；`<tag>` = 用途短标（如 `dagger_r1`）；`_sliceN` / `_5k` = 切片 / 整池量级后缀。
2. **登记**：任何新增 spec 必须同步登记本表（用途/消费者/状态），并注明生成工具与命令；本目录 `*.json` 不入 git，**本 README 是唯一权威清单**。
3. **Clean500 = val − eval500，不"另造"新内容**：本目录保留 **canonical 副本 `scenarios_clean500.json`**（n=500，sha256 `087db3f5…`，与历史真源逐字节一致；CLEAN-1 误随 docs 归档删除后于 2026-10-16 恢复，规则随之修订）。**严禁**以任何其它方式重新生成/改动其内容；重建须 `scenarios_val.json − scenarios_eval500.json` 且 sha 校验（核验见 `runs/round38/c6/clean500_recon.json`）。
4. **删除**：删除 spec 前须以 `net env pipeline tools tests config reward_model` + 顶层 md 为引用面复核；零现役消费者才可删，删除后登记到本表"已删"节。

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
- train/val seed 区间重叠会被生成器拒绝（`env/scenario/generator.py:477-479`）。
- 5k / dagger 池生成时断言与 `scenarios_eval500.json`、`scenarios_val.json` 无交集（`tools/make_dagger_pools.py:74-82, 123-131`；`:151-155` 为三轮互不重叠断言）。
- `tools/gene_env.sh` 默认生成后做实例化校验，报告写 `validation_<name>.json`（`env/scenario/cli.py:101-122`）；校验失败或改过的 spec 不得用于训练。
- 观测变更后 BC 数据需重采（`obs_fingerprint` 守卫，`env/obs/__init__.py:1-6`）。
