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

# 历史复现需旧 net（模型内 oneshot 诊断臂）
git worktree add /tmp/old_net 74f0bd4     # 或 git show 74f0bd4:net/path_head.py
```

## 清单

| 快照模块 | 退役符号（出处 file:line，快照 commit `74f0bd4`） | 判定/出处 | 复现约束 |
| --- | --- | --- | --- |
| `oneshot_path.py` | `compose_path_plan` (`net/path_head.py:641-702`)、`one_shot_plan` (`:1053-1054`)、`forward(one_shot=True)` 叠加块 (`:1288-1305`)、`plan_override` 钉住尾段 (`net/model.py:1911-1916`)、装载守卫 (`net/model.py:986-993`)、`_integrate_kappa` (`:549-572`)、`replay_plan`+圆弧数学 (`:575-601` / `net/model.py:226-247`) | `ARCHITECTURE_FORBIDDEN`（`docs/deprecation_manifest.yaml` A 组：唯一动作源 + 每步单动作；一次性六步 = 第二轨迹生成器） | 快照自包含（仅 torch）可直接复算 6 步 plan；**模型内** 复现需删除前 net（`74f0bd4`） |

## 纪律

1. 每个快照模块头部必须写清：源码出处（`file:line`）+ 快照 commit + "仅历史复现，需旧 net 版本"。
2. 新增/删除快照时同步本 README 清单（来源 = `docs/deprecation_manifest.yaml` 对应条目）。
3. 快照中的常量/数学（动作界、圆弧、ζ 积分）**逐行拷贝**，不做"顺手改进"；与当前 net 的差异
   一律在模块 docstring 里显式标注。
