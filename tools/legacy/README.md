# tools/legacy —— 已退役路径的历史复现快照

用途：存放 **从主链删除** 的历史实现快照（逐行拷贝 + 出处标注），供历史消融/复现/审计使用。
本目录 **不被生产代码导入**（`net/`、`pipeline/`、`tools/` 官方 CLI 对 `tools.legacy` 零引用；
CI 守卫 #1 的扫描面，见 `docs/code_reachability_audit.md` §5）。

## 运行方式

```bash
# 模块自检（纯 torch，CPU，无 metadrive）
tools/venv-python -m tools.legacy.oneshot_path
tools/venv-python -m tools.legacy.others_layout        # R38-P3-G：旧 others-28→33 适配器
tools/venv-python -m tools.legacy.monitoring_reader    # R38-P3-G：legacy tag 分组/重放

# 单测
tools/venv-python -m pytest tests/test_legacy_oneshot_path.py -q
tools/venv-python -m pytest tests/test_legacy_k_anchor.py -q      # R38-P2a K-anchor 快照
tools/venv-python -m pytest tests/test_legacy_key_mapper.py -q    # R38-P2a/P2b 旧 ckpt 键映射
tools/venv-python -m pytest tests/test_r38_p2b_warp_legacy_removed.py -q   # R38-P2b warp 快照
tools/venv-python -m pytest tests/test_r38_p2b_policy_fold.py -q           # R38-P2b 双实体收编
tools/venv-python -m pytest tests/test_monitoring_tb_grouped.py -q         # R38-P3-G legacy reader

# 历史复现需旧 net（模型内 oneshot 诊断臂）
git worktree add /tmp/old_net 74f0bd4     # 或 git show 74f0bd4:net/path_head.py
git worktree add /tmp/legacy_net dd0a8c2  # R38-P2b：legacy set_ego_warp(w1/w2) 整臂复现
```

## 清单

| 快照模块 | 退役符号（出处 file:line，快照 commit） | 判定/出处 | 复现约束 |
| --- | --- | --- | --- |
| `oneshot_path.py` | `compose_path_plan` (`net/path_head.py:641-702`)、`one_shot_plan` (`:1053-1054`)、`forward(one_shot=True)` 叠加块 (`:1288-1305`)、`plan_override` 钉住尾段 (`net/model.py:1911-1916`)、装载守卫 (`net/model.py:986-993`)、`_integrate_kappa` (`:549-572`)、`replay_plan`+圆弧数学 (`:575-601` / `net/model.py:226-247`) | `ARCHITECTURE_FORBIDDEN`（`docs/deprecation_manifest.yaml` A 组：唯一动作源 + 每步单动作；一次性六步 = 第二轨迹生成器） | 快照自包含（仅 torch）可直接复算 6 步 plan；**模型内** 复现需删除前 net（`74f0bd4`） |
| `k_anchor/`（`anchor.py` / `plan_head.py` / `fit_plan_anchors.py` / `plan_anchors_k6.json`） | `net/anchor.py` 整文件（371 行）、`net/plan_head.py` 锚分支（`num_anchors`/`set_anchors`/`plan_anchors`/锚 buffer）、`tools/fit_plan_anchors.py`、`config/plan_anchors_k6.json` | `LEGACY_REPRO`（B 组：K-anchor 长期关闭 + 独立计划候选；旧 ckpt 键经 `legacy_key_mapper` 忽略） | `anchor.py` 自包含；`plan_head.py` 需当前 `net/encoders.py`+`net/moe.py`；**模型内**复现需删除前 net（`9577078`）。详见 `k_anchor/README.md` |
| `legacy_key_mapper.py` | 旧 ckpt 键面分类（`IGNORE_PREFIXES` = B 组 20 键；**R38-P2b 双实体**：`PREFIX_RENAME_MAP` = `path_head.* → policy.structured_action.*`，`RENAME_MAP` = `policy.log_std → policy.logstd`，`LEGACY_META_KEYS` = `path_head_ablation/path_head_mode`；`classify_keys`/`classify_meta`） | `CHECKPOINT_ONLY`（契约 §4.2/§4.4/§5；`docs/deprecation_manifest.yaml` B/D 组） | 纯函数（无 torch/IO）；含 rename/drop/reject 三档 + 条件族（改名目标缺席）逐族计数 |
| `checkpoint_migration.py` | 最小装载入口（`migrate_state_dict`/`load_checkpoint_state`；默认 fail-fast 不变；R38-P2b 起应用改名族） | 同上（契约 §5 硬约束 1-4） | pipeline loader 仍**未接线**（.slim 评估链可在调用点显式使用）；单测 `tests/test_legacy_key_mapper.py` |
| `legacy_ego_warp.py` | `set_ego_warp`（w1/w2 模式 setter，`net/model.py:593-628`@dd0a8c2）、`_ego_warp_pose_k`（`:697-717`）、`assemble_pose_k` w1/w2 口径（`net/ego_warp.py:189-227`@dd0a8c2） | `LEGACY_REPRO`（C 组：w1/w2 = 历史 warp 实验口径；R37 M1 正式接口 = `set_ego_warp_mode(off\|analytic)`） | 纯函数自包含（仅 torch）；**逐位参照物 = `tests/fixtures/r38_p2b_warp/`**（m1_gate_a/b_warp_compat/d1_ego_warp_tests 改读 fixture）；模型内整臂复现需删除前 net（`dd0a8c2`） |
| `legacy_policy.py` | 旧 `PolicyHead`/`ValueHead`/`CrossAttnHead` 整模块（`net/policy.py`@dd0a8c2；唯一改动 = `H` 本地化） | D 组"旧 PolicyHead 迁 legacy 模型加载器"落点（正式 net 仍保留 `net/policy.py` 供默认关路径前向） | 自包含（仅 torch）；消费 = 离线诊断 `tools/diagnostics/legacy_policy_diag.py`（`--legacy-ckpt` 独立复算）/ 旧 ckpt `policy.*` 键面审计 |
| `monitoring_reader.py` | `pipeline/monitoring.py` 旧 `legacy_tags` 口径：`_LEGACY_*`（20 常量 + 5 regex）/`_group_of_tag`/旧 `_grouped_scalars` 分支/`_write_legacy_grouped_tensorboard`（`pipeline/monitoring.py:309-391,736-744`@2cfd8e8） | `LEGACY_REPRO`（G 组：正式训练只写新 schema `_slim_tag`；旧 tag 仅用于**读取/重放旧日志**） | 纯 stdlib（csv/math/re）；`legacy_group_of_tag`/`legacy_grouped_scalars`/`read_metrics_csv`/`write_legacy_grouped_tensorboard`；正式 `TrainingMonitor(legacy_tags=...)` 写入即 fail-fast |
| `others_layout.py` | 旧 others 28→33 重排：`net/mem.py::_remap_legacy_others`（`net/mem.py:110-138`@2cfd8e8）+ `net/encoders.py::OTHERS_LEGACY_DIM/OTHERS_HEAD_LEGACY_DIM` | `LEGACY_REPRO`（G 组：静默零填充 static 段违反"缺失静默填充"守则；net 端已改 fail-fast） | 纯函数 `remap_legacy_others_28_to_33`（仅 torch）；旧数据集/旧缓存需**显式**转换后喂 `mem_from_obs` |

## 纪律

1. 每个快照模块头部必须写清：源码出处（`file:line`）+ 快照 commit + "仅历史复现，需旧 net 版本"。
2. 新增/删除快照时同步本 README 清单（来源 = `docs/deprecation_manifest.yaml` 对应条目）。
3. 快照中的常量/数学（动作界、圆弧、ζ 积分）**逐行拷贝**，不做"顺手改进"；与当前 net 的差异
   一律在模块 docstring 里显式标注。
