# tools/legacy/k_anchor —— K-anchor 计划头快照（R38-P2a）

用途：保存 R38-P2a 减枝波（B 组）从主链删除的 **K-anchor 旧计划头**完整实现快照，
供历史消融/复现/审计使用。本目录 **不被生产代码导入**（`net/`、`pipeline/`、`tools/` 官方
CLI 对 `tools.legacy` 零引用；CI 守卫 #1 的扫描面，见 `docs/code_reachability_audit.md` §5）。

## 清单（快照 commit = `9577078`）

| 快照文件 | 退役符号（出处，行号按快照 commit） | 判定/出处 | 复现约束 |
| --- | --- | --- | --- |
| `anchor.py` | `net/anchor.py` 整文件（371 行）：锚字典加载（`load_anchor_dictionary`）、内置默认 K=6、`lane_follow_dtheta`/`to_lane_dtheta`/`from_lane_dtheta`、`lane_shape_features`、`assign_anchors`、`mixture_probabilities`、`anchor_mixture`、`assigned_plan` | `LEGACY_REPRO`（`docs/deprecation_manifest.yaml` B 组：K-anchor 长期关闭 + 独立计划候选） | 自包含（仅 torch）；纯函数可直接复算 |
| `plan_head.py` | `net/plan_head.py` 整文件（221 行）：`num_anchors/anchor_temperature/anchor_hard` 分支、`set_anchors`/`set_anchor_mode`/`plan_anchors`、锚 buffer（`anchors/anchor_ds/anchor_dth`） | 同上（B 组：锚选择头/速度头/残差头 + buffer） | 需当前 `net/encoders.py`+`net/moe.py`（非退役模块）；唯一改动 = `anchor_mixture` 改为本包内导入 |
| `fit_plan_anchors.py` | `tools/fit_plan_anchors.py`（KMeans 拟合形状锚字典 CLI） | 同上（B 组：拟合工具随锚头归档） | 需 `numpy`+`sklearn` + expert npz；唯一改动 = `--out` 默认指向本目录 |
| `plan_anchors_k6.json` | `config/plan_anchors_k6.json`（K=6 锚字典，与内置默认同源） | 同上（B 组：字典文件随归档删除） | 只读数据（`sha256 79829ef7…73bf`） |

## 模型级集成的出处（未随快照自包含）

`num_anchors>0` 时的模型接线在删除前的 `net/model.py`（快照 commit `9577078`）：
构造参数 `num_anchors/anchor_path/anchor_temperature/anchor_hard`（`:420-453`）、
`load_anchor_dictionary` 调用（`:449-453`）、rollout 尾段锚计划驱动
（`anchor_plan[:, k]`，`:1907-1927`）、输出键 `anchor_logits/anchor_probs/anchor_plan/
anchor_speed/anchor_residual/anchor_ctx`（`:2075-2083`）。**模型内复现需旧 net 版本**：

```bash
git worktree add /tmp/old_net 9577078      # 或 git show 9577078:net/model.py
```

## 运行方式

```bash
# 快照包自检（锚字典加载 + PlanHead K=6 前向）
tools/venv-python -m tools.legacy.k_anchor.selfcheck

# 单测（含 WTA/软混合/车道系快照断言）
tools/venv-python -m pytest tests/test_legacy_k_anchor.py -q

# 锚字典拟合（需 expert npz；输出到本目录）
tools/venv-python tools/legacy/k_anchor/fit_plan_anchors.py --dataset <expert_bc.npz> --k 6
```

## 旧 ckpt 兼容

带 `plan_head.anchor*`（16 键）与 `encoders.lane/ttc`（4 键）的旧 ckpt 由
`tools/legacy/legacy_key_mapper.py` 登记忽略（`IGNORE_PREFIXES`，逐族计数、
默认 fail-fast 不变），见 `docs/checkpoint_migration_contract.md` §4.2/§5。

## 纪律

1. 快照逐行拷贝（唯一改动 = 导入路径/CLI 默认值，均在各模块 docstring 显式标注）；
2. 新增/删除快照时同步本 README 与 `tools/legacy/README.md` 清单；
3. 快照不是生产依赖，不保证与当前 net 接口兼容。
