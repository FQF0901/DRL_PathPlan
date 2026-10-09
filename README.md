# DRL_PathPlan

MetaDrive 城市/高速驾驶规划 RL：观测（当前帧 + 6 帧历史，含 OD/LD/nav）→ 网络输出动作 `(ds, dθ)`
（下一 0.5 s 的弧长 + 航向变化）→ 网络内 latent rollout 产生 3 s / 6 点 plan → LQR 跟踪执行。
训练链路：**A 世界模型教师强制 → B 规划器 BC（primary→specific，可接 DAgger 迭代）→ C PPO RL
（实验口径，KL 锚定 B 快照）**（`pipeline/stages.py:1-27`；网络数据流与输出契约见 `docs/net_architecture.md`）。

## 快速开始

前置：Python 3.10 + 仓库内 `.venv`（`metadrive-simulator 0.4.3`、`numpy<2`、`opencv-python-headless`）。
**所有 MetaDrive 运行统一用 `tools/venv-python`**（自动注入 venv 内 glvnd 的 `LD_LIBRARY_PATH`，
`tools/venv-python:1-14`）。环境重建：

```bash
python3 -m venv --without-pip --system-site-packages .venv
python3 -m pip --python .venv/bin/python install metadrive-simulator "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
bash tools/setup_gl_libs.sh   # 本机无系统 libGL：在 .venv 内解包 glvnd（不改系统）
```

`env/specs/*.json`、`datasets/`、`runs/` 都是生成物（`.gitignore:8,10,12`），克隆后按最短链生成：

```bash
# 1) 场景 spec：env/specs/scenarios_{train,val}.json（默认 10000 / 1000 条）
bash tools/gene_env.sh
# 评测集：从 scenarios_val.json 分层抽 500 条 → scenarios_eval500.json（默认 DRY-RUN，必须 --write）
tools/venv-python tools/make_eval_spec.py --write

# 2) 专家 BC 数据（--specs / --out 必填；--workers 默认 auto）
tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json --limit 5000 \
    --out "datasets/BTC$(date +%Y%m%d-%H%M)_expert5k"

# 3) 训练（零参可跑：数据自动取最新 datasets/BTC*_expert*；A/B 共用一个 run 根）
bash tools/train.sh            # Stage A（默认）
STAGE=B bash tools/train.sh    # Stage B（自动接最新 stage_a/final.pt）

# 4) 评测（零参可跑：ckpt 自动取最新 stage_b/final.pt、回退 stage_a；默认 LIMIT=50 / TRACKER=lqr）
bash tools/test.sh
tools/venv-python tools/test.py --policy baseline --limit 10 --workers 1   # 规则基线参照

# TensorBoard（run 名 train_stageA/train_stageB/eval_*；端口占用自动 +1）
bash tools/tb.sh
```

直接 CLI 最小形态（绕过脚本）：

```bash
tools/venv-python tools/train.py --stage A --bc-dir datasets/BTC20260926-2343_expert5k --out runs/train/stage_a
tools/venv-python tools/train.py --stage B --ckpt runs/train/stage_a/final.pt \
    --bc-dir datasets/BTC20260926-2343_expert5k --out runs/train/stage_b
tools/venv-python tools/train.py --stage C --config config/arms/v7_arm1_offroad.yaml \
    --ckpt runs/train/stage_b/final.pt --updates 200 --out runs/train/stage_c
tools/venv-python tools/test.py --policy ckpt --ckpt runs/train/stage_b/final.pt --limit 50 --workers 2
```

（`runs/train/…` 为入口默认 `--out` 形态，`tools/train.py:72-73`；数据目录示例取仓库现存 KEEP 数据集，
实际以 `datasets/` 当前内容为准。）

入口与默认值来源：`tools/train.sh:1-6`（零参、`STAGE/RESUME/BC_DIR/WORK_DIR` 等环境变量覆盖）、
`tools/test.sh:1-4`、`tools/run_config.py:56-132`（从 `config/train.yaml` 解析路径与 ckpt）；
Stage C 当前采用臂见 `config/arms/README.md`。

## 仓库布局

| 路径 | 内容 |
| --- | --- |
| `config/` | 运行配置：`default/env/model/train/eval` 五份 YAML + `arms/`（RL 臂）+ `plan_anchors_k6.json`；加载关系与消费方见 `config/README.md` |
| `env/` | MetaDrive 封装（`metadrive_env.py`、`tracking.py`）+ `obs/`（观测通道/历史/schema）+ `scenario/`（spec 生成/校验/标签/脚本事件）+ `expert/`（规则专家） |
| `net/` | 编码器 / 时序注意力 / 空间消息传递 / MoE / latent 世界模型（ST-GNN）/ 策略与价值头 / K-anchor |
| `pipeline/` | 阶段编排 `stages.py`、训练器 `trainer.py`、环境池 `vector_env.py`、评测 `eval_runner.py`、DAgger 循环 `phase3_loop.py`、run 布局 `run_paths.py`、监控 `monitoring.py` |
| `reward_model/` | 奖励项 `terms.py`、聚合 `aggregation.py`、KPI `kpi.py` |
| `tools/` | 入口脚本（`gene_env.sh` / `train.sh` / `test.sh` / `tb.sh`）+ 采集/评测/诊断工具 + `measure/`、`diagnostics/` |
| `tests/` | 契约/回归测试：`tools/venv-python -m pytest tests/ -q`（`tests/README.md:1-9`） |
| `docs/` | 现行文档 + `archive/`（历史归档）+ `cleanup/`（清理记录），见下「文档索引」 |
| `datasets/`、`runs/` | 生成物（gitignored）：专家数据集 / 训练与评测产物 |
| `env/specs/` | 生成物（gitignored）：场景 spec JSON |

## 关键约定

### run 命名（代码生成，不手拼）

- 训练 run：`runs/BTC<北京戳>_<name>`（`pipeline/run_paths.py:106-108`）；`run.work_dir: auto` 由入口解析
  （`config/train.yaml:57-64`、`tools/run_config.py:102-111`）。**同轮 A/B 共用一个 run 根**：Stage B 取
  Stage A 产物所在 run（`pipeline/run_paths.py:5-6`）。
- 评测 run：`runs/BTC<秒级北京戳>_<kind>_<tag>`，由 `eval_run_name` 幂等规范化（`pipeline/run_paths.py:86-103`；
  `tools/test.py:97-101`）；`--name` 传路径时原样使用。
- run 内固定布局：`logs/stage_<x>.log`、`manifest.txt`、`config.snapshot.yaml`、`model.snapshot.yaml`、
  `stage_a/`、`stage_b/`（`pipeline/run_paths.py:148-164`；`tools/train.py:116-125`）。

### ckpt 命名

- A/B：周期 `ckpt_epoch{NNN}.pt`（`pipeline/stages.py:493-494`）、阶段末 `final.pt`
  （`pipeline/stages.py:2709,3461`）、B 另存 `bc.pt`（`pipeline/stages.py:3456`）。
- C：周期 `ckpt_u{NNN}.pt`（`pipeline/stages.py:539-540`）、阶段末 `final.pt`（`pipeline/stages.py:4888`）。
- `run.resume: auto` 从最新「有周期 ckpt、无 `final.pt`、进程未在跑」的 stage 原地续跑
  （`pipeline/run_paths.py:279-287`；`tools/run_config.py:82-99`）。
- phase3 循环稳定导出 `<loop_dir>/best/final.pt` 供下游消费（`pipeline/phase3_loop.py:115,932`）。

### spec 隔离硬校验（train-on-test 防复发）

- 训练数据只允许来自 train spec；`dagger_collect` 的 `--from-eval` 路径已删除
  （`tools/dagger_collect.py:6-9`）。
- 隔离守卫按 `(id, seed)` 与 eval/val 冻结集求交集，命中即 `SystemExit`
  （`tools/dagger_collect.py:905-932`）；Stage B 合并 DAgger 数据前复核来源（`pipeline/stages.py:309-350`）。
- 评测集固定 `env/specs/scenarios_eval500.json`（`config/eval.yaml:3`），与 `scenarios_val.json` 的
  互斥/并集关系与禁训纪律见 `docs/v7_program_prereg.md` §0。

### obs 指纹：改动 `env/obs` 必须重采数据

- `env/obs/__init__.py::obs_fingerprint` = `env/obs/*.py` 内容哈希 + schema 版本（当前 v6，
  `env/obs/__init__.py:38-49`）。
- 采集时写入数据集 meta（`tools/collect_expert.py:1632`）；训练加载时不一致 → 告警并要求重新采集
  （`pipeline/trainer.py:2956-2968`）。
- 规则：scope/通道/历史语义一旦改动（如 OD/LD scope、历史帧结构），旧 BC 数据在新观测下语义不同，
  必须用 `tools/collect_expert.py` 重采（`env/obs/__init__.py:3-5`）。

### 配置加载（细节见 `config/README.md`）

- 训练侧 `pipeline.stages.load_config` 只解析**一层 includes + 顶层平铺合并**（`pipeline/stages.py:138-149`）；
- 评测侧 `pipeline.eval_runner.load_config` 为**递归深合并**（`pipeline/eval_runner.py:1424-1445`）。

## 文档索引

### 现行（当前口径 / 冻结规格）

| 文档 | 内容 |
| --- | --- |
| `docs/LOCKS.md` | 锁版记录（Stage A/B 锁版与 RL 基线 tag） |
| `docs/net_architecture.md` | 网络架构（代码口径：数据流 / 模块 / IO 维度 / 监督） |
| `docs/rl_reward_v5.md` | 奖励 v5 冻结规格（项集 / 聚合结构 / 终局值档） |
| `docs/v7_program_prereg.md`、`docs/v7_program_report.md` | v7 程序预注册（冻结）与收尾报告 |
| `docs/v6_reports/` | v6 报告/预注册/网络设计（目录索引 `docs/v6_reports/README.md`） |
| `docs/v7_reports/` | v7 证据/报告档 + 冻结 specs/figures（目录索引 `docs/v7_reports/README.md`） |
| `docs/reward_audit/` | 奖励审计与 E-β″ 复算（`MANIFEST.md` 为入口） |

### 历史归档

| 路径 | 内容 |
| --- | --- |
| `docs/archive/` | v3/v4 代历史文档（索引 `docs/archive/README.md`） |
| `docs/archive/config_arms_legacy/` | v6 P4 / v7 历史臂配置（索引与映射见其 `README.md`） |

### 清理记录

| 路径 | 内容 |
| --- | --- |
| `docs/cleanup/` | V8 清理报告：`V8_CLEANUP_config_docs.md`（config/arms + docs 归档）、`V8_CLEANUP_data_runs_DONE.md`（datasets/runs 删除执行）、`V8_CLEANUP_data_runs_WILL_DELETE.md`（清单） |

> 运行产物（`runs/`）与数据集（`datasets/`）不进 git；历史数字与复现命令固化在 `docs/` 各报告中。
