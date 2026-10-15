# tools/legacy —— 已退役路径的历史复现快照

用途：存放 **从主链删除** 的历史实现快照（逐行拷贝 + 出处标注），供历史消融/复现/审计使用。
本目录 **不被生产代码导入**（`net/`、`pipeline/`、`tools/` 官方 CLI 对 `tools.legacy` 零引用；
CI 守卫 #1 的扫描面，见 `docs/code_reachability_audit.md` §5）。

## 运行方式

```bash
# 模块自检（纯 torch，CPU，无 metadrive）
tools/venv-python -m tools.legacy.oneshot_path

# 单测
tools/venv-python -m pytest tests/test_legacy_oneshot_path.py -q
tools/venv-python -m pytest tests/test_legacy_k_anchor.py -q      # R38-P2a K-anchor 快照
tools/venv-python -m pytest tests/test_legacy_key_mapper.py -q    # R38-P2a 旧 ckpt 键映射

# 历史复现需旧 net（模型内 oneshot 诊断臂）
git worktree add /tmp/old_net 74f0bd4     # 或 git show 74f0bd4:net/path_head.py
```

## 清单

| 快照模块 | 退役符号（出处 file:line，快照 commit） | 判定/出处 | 复现约束 |
| --- | --- | --- | --- |
| `oneshot_path.py` | `compose_path_plan` (`net/path_head.py:641-702`)、`one_shot_plan` (`:1053-1054`)、`forward(one_shot=True)` 叠加块 (`:1288-1305`)、`plan_override` 钉住尾段 (`net/model.py:1911-1916`)、装载守卫 (`net/model.py:986-993`)、`_integrate_kappa` (`:549-572`)、`replay_plan`+圆弧数学 (`:575-601` / `net/model.py:226-247`) | `ARCHITECTURE_FORBIDDEN`（`docs/deprecation_manifest.yaml` A 组：唯一动作源 + 每步单动作；一次性六步 = 第二轨迹生成器） | 快照自包含（仅 torch）可直接复算 6 步 plan；**模型内** 复现需删除前 net（`74f0bd4`） |
| `k_anchor/`（`anchor.py` / `plan_head.py` / `fit_plan_anchors.py` / `plan_anchors_k6.json`） | `net/anchor.py` 整文件（371 行）、`net/plan_head.py` 锚分支（`num_anchors`/`set_anchors`/`plan_anchors`/锚 buffer）、`tools/fit_plan_anchors.py`、`config/plan_anchors_k6.json` | `LEGACY_REPRO`（B 组：K-anchor 长期关闭 + 独立计划候选；旧 ckpt 键经 `legacy_key_mapper` 忽略） | `anchor.py` 自包含；`plan_head.py` 需当前 `net/encoders.py`+`net/moe.py`；**模型内**复现需删除前 net（`9577078`）。详见 `k_anchor/README.md` |
| `legacy_key_mapper.py` | 旧 ckpt 键面三档分类（`IGNORE_PREFIXES` = B 组 20 键；`classify_keys`） | `CHECKPOINT_ONLY`（契约 §4.2/§5；`docs/deprecation_manifest.yaml` B 组第 10 项） | 纯函数（无 torch/IO）；B 组 ignore 集 + 预留 rename/init 接口 |
| `checkpoint_migration.py` | 最小装载入口（`migrate_state_dict`/`load_checkpoint_state`；默认 fail-fast 不变） | 同上（契约 §5 硬约束 1-4） | 未接线（loader 接入属 D 组）；单测 `tests/test_legacy_key_mapper.py` |

## 纪律

1. 每个快照模块头部必须写清：源码出处（`file:line`）+ 快照 commit + "仅历史复现，需旧 net 版本"。
2. 新增/删除快照时同步本 README 清单（来源 = `docs/deprecation_manifest.yaml` 对应条目）。
3. 快照中的常量/数学（动作界、圆弧、ζ 积分）**逐行拷贝**，不做"顺手改进"；与当前 net 的差异
   一律在模块 docstring 里显式标注。
