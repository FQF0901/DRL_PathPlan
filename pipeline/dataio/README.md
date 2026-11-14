# dataio

`pipeline/dataio/` 是**数据采集与划分代码**，不是数据目录。

相关目录的边界：

```text
pipeline/dataio/  采集 / DAgger / split 代码
data/specs/       固定场景定义
data/splits/      仅 val/test gate seal + RL subset view
data/episodes/    采集后的 NPZ 数据
env/obs.py        env -> net 的观测 schema
```

## 模块

严格说这里是 **4 个功能模块 + 一个空的 `__init__.py`**。

| 文件 | 作用 |
|---|---|
| `collect_expert.py` | 用 `PurePursuitIDMPolicy` 采集 expert IL/BC 数据；生成观测、实测 expert endpoint、PSTL 诊断、audit 字段和 episode outcome。 |
| `collect_dagger.py` | 用 student `DriveModel` 做 DAgger roll-in；逐状态查询 expert，按 replay/lift 生成动作标签；生成 student outcome 和 DAgger 诊断。 |
| `freeze_splits.py` | 从 `data/specs/source/` 生成固定的 train/val/test/mini specs；只给 val/test 写 gate seal manifest（SHA-256），train/mini 的 split 直接写在 spec 内。 |
| `splits.py` | 纯 split 工具：按 `scene_seed` 划分、重叠检查、seed 计数、文件 SHA-256、manifest 校验。 |
| `__init__.py` | 空文件，仅用于声明 Python package。 |

## 数据流

```text
data/specs/source/
  train_10000.json
  val_1000.json
        │
        ▼
pipeline/dataio/freeze_splits.py
        │
        ├─ data/specs/train_2000.json        # 2000 scenes
        ├─ data/specs/val_500.json           # 500 scenes, held-out
        ├─ data/specs/test_500.json          # 500 scenes, final only
        ├─ data/specs/mini_train_100.json
        ├─ data/specs/mini_val_20.json
        └─ data/splits/{val,test}_*.json     # gate seal: seed 顺序 + SHA-256

data/specs/train_2000.json
        │
        ├─ pipeline/dataio/collect_expert.py                → data/episodes/train_2000/il_expert_exact
        └─ pipeline/dataio/collect_dagger.py         → data/episodes/train_2000/il_dagger_exact_r*

data/specs/val_500.json
        │
        └─ pipeline/eval/closed_loop.py                  → 闭环验证（现场 build_env，不落 NPZ）

data/specs/test_500.json
        │
        └─ pipeline/eval/closed_loop.py                  → 最终 held-out 验收
```

## 常用命令

```bash
# 生成/刷新固定的 specs 和 split manifests
python -m pipeline.dataio.freeze_splits

# IL/expert BC 数据
python -m pipeline.dataio.collect_expert \
  --specs data/specs/train_2000.json \
  --out data/episodes/train_2000/il_expert_exact \
  --workers 8 \
  --max-steps 600

# DAgger 数据（默认参数见 config/collect_dagger.yaml）
python -m pipeline.dataio.collect_dagger
```

`collect_dagger.py` 还支持：

- `--expert-ratio`：expert replay 概率
- `--shield-ttc`：TTC 过低时强制 expert replay
- `--stuck-shield-steps`：低速持续超过该步数后强制 expert takeover
- `--difficulties`：只采集指定难度，例如 `--difficulties medium hard`
- `--round` / `--seed`：DAgger 轮次和随机种子
- `--device`：CPU/CUDA

零参数运行时读取 `config/collect_dagger.yaml`；任何 CLI 参数都会覆盖 YAML。

## 输出格式

两个 collector 都输出：

```text
expert_bc.npz
expert_bc.meta.json
```

NPZ 主要字段：

- 历史观测：`od`、`ld`、`nav`、`ego`、mask、IDs、`pose` 等
- action 标签：`action`、`action_lift`、`action_replay`（DAgger）
- label 状态：`label_source`、`label_valid`、`expert_plan_valid`
- 诊断/audit：PSTL、residual、shield、intervention、outcome 等
- `scene_seed`、`episode_id`、`step`

注意：

- `collect_expert.py` 的 `action` 是**实测 expert endpoint 标签**；`action_lift` 仅作为 audit/residual，不替代标签。
- `collect_dagger.py` 的 `action` 在 replay 行是实测 expert 标签，在 student lift 行是 lift 标签；用 `label_source` 区分。
- `success` / `collision` / `off_road` 等是 episode 级 outcome，按行广播；在 DAgger 数据里它们是 **student roll-in 的 outcome**，不是 expert 的 outcome。

## Split 规则

- 只按 `scene_seed` 划分，绝不按 `episode_id`。
- 同一个 scene 的所有 episode 必须落在同一个 split。
- 固定集合：
  - `train_2000`：2000 scenes，用于 IL / DAgger 训练，也是 RL 的环境 seed 集
  - `rl_train_1500`：RL 训练子集，seeds 1000..2499
  - `rl_holdout_500`：RL holdout/selection，seeds 2500..2999
  - `val_500`：500 scenes，用于 BC/WM checkpoint/model selection
  - `test_500`：500 scenes，只用于最终验收，不调参
  - `mini_train_100` / `mini_val_20`：只用于 smoke test
- `train_2000` / `mini_*` 的 split 直接编码在 spec 的 `split` 字段里，不再写 seed manifest。
- 只有 `val_500` / `test_500` 保留 gate seal manifest；RL 只保留两个 subset view。
- `test_500` 没有 episode NPZ；最终评测由 `pipeline/eval/closed_loop.py` 现场生成环境。
- `val_500/expert_exact` 是 exact raw offline BC validation labels，不是闭环 KPI 数据；闭环验证只用 `data/specs/val_500.json` + `data/splits/val_500.json`。

## 布局

路径、split 和 episode pool 说明以 `data/README.md` 为准；本文件只描述 collector 行为、命令和字段。
