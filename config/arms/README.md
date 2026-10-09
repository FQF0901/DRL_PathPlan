# config/arms（RL 臂配置）

> **当前状态（2026-10-09）**：本目录仅 **`v7_arm1_offroad.yaml`**（当前采用臂）+ 本 README。
> 历史臂（v6 P4 奖励臂 / v7 奖励臂 / v7 结构臂，共 21 项）已归档到 `docs/archive/config_arms_legacy/`
> （族说明与逐项映射见该目录 `README.md`；操作与引用检查见 `docs/cleanup/V8_CLEANUP_config_docs.md`）。
> 当前臂的加载/奖励守卫测试：`tests/test_arm_configs.py`。

## 当前臂：`v7_arm1_offroad.yaml`

**用途**：v7 P2 首臂（`docs/v7_program_prereg.md` §9）——在 IL 底座（w1 e005）上做 RL 奖励单变量：
相对 v5 bundle（同 arm0）追加 `off_road_edge` 离路距离型稠密项；KL 锚 0.05 → 0.02（慢衰减，方差控制）。
完整口径与执行路径注意见文件头注释（`config/arms/v7_arm1_offroad.yaml:1-18`）。

**关键配置点**：

- `includes`：`config/default.yaml` + 四个叶子子配置（写法原因见下「加载语义」）；
- `stages.C.reward.terms`：rc=1 草案项集 + **恰一个**追加项 `off_road_edge`（weight −0.5、`edge_scale_m` 1.0）；
- `stages.C.reward.aggregation.terminal_values`：rc=1 档终局值（与 `docs/reward_audit/ebeta2/config_draft_rc1.yaml`
  逐位一致，`tests/test_arm_configs.py` 逐参数钉住）；
- `stages.C.kl_anchor_coef` / `kl_anchor_final_coef`：0.05 / 0.02（**记录用**；实际执行由 driver CLI 承载，
  见下「加载语义」）。

**使用示例**：

```bash
tools/venv-python tools/train.py --stage C --config config/arms/v7_arm1_offroad.yaml \
    --ckpt runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt \
    --spec env/specs/scenarios_train_dagger_r1.json --pool local --envs 1 \
    --updates 200 --out runs/train/stage_c_arm1
```

（CLI 参数均存在：`--pool` / `--envs` / `--updates` 见 `pipeline/stages.py:5068,5078,5079`；
`scenarios_train_dagger_r{1,2,3}.json` 由 `tools/make_dagger_pools.py` 生成，见其 docstring；
示例 ckpt 为仓库现存 KEEP run（该 run 为扁平 ckpt 布局，`pipeline/run_paths.py:11`）；
base ckpt 与 spec 以当次 driver 记录为准，driver 为外部编排脚本、不入 repo。）

## 加载语义（重要：一层平铺合并 + `stages` 段整体替换）

训练侧 `pipeline.stages.load_config`（`pipeline/stages.py:138-149`）**只解析一层 `includes`**，
且为**顶层平铺合并**（主文件顶层键整体替换 include 的同名键），不做深度合并。因此臂文件：

- `includes` 同时列出 `config/default.yaml` 与其四个叶子子配置（`config/arms/v7_arm1_offroad.yaml:19-24`）：
  只写 `config/default.yaml` 时，平铺加载器只会带入其 `project`/`includes` 元数据而**丢失
  `train`/`data`/`run`/`monitoring` 等段**（`tests/test_arm_configs.py` 已钉住 includes 清单）。
- 臂文件的 `stages` 段会**整体替换**合并结果里的 `stages`；未显式携带的 `stages.C` 键回落代码默认
  （如 `max_episode_steps` 回落 200：`pipeline/stages.py:512-528`；`tests/test_stage_c_horizon.py`
  动态遍历 `config/arms/*.yaml` 钉住）。
- **KL 锚是 CLI-only**：`run_stage_c` 只读 `args.kl_anchor_coef` / `args.kl_anchor_final_coef`
  （`pipeline/stages.py:4582-4584`；parser 默认值 `5108-5109`）→ 本臂文件的 `kl_anchor_*` 键不会被代码消费，
  实际值由 driver 按文件值传 CLI（文件头注记：driver 侧 fail-closed 断言；见
  `config/arms/v7_arm1_offroad.yaml:9-11,47-49`）。
- 递归深合并加载器 `pipeline.eval_runner.load_config`（`pipeline/eval_runner.py:1424-1445`；
  `tools/test.py --config` 走它）下 includes 递归解析，同一臂文件也可用；若需保留 A/B 段做深度合并，
  可用它先解析再落盘为 run 快照。

## 归档（历史臂）

- **位置**：`docs/archive/config_arms_legacy/`（文件内容原样，未修改）。
- **族**：v6 P4 奖励臂（`arm0_bundle_rc1`、rc 扫档 `arm1–3`、`arm4_lam098`、`arm5_ttc`、车道罚 `arm6–7`、
  碰撞抑制 `arm8_*`）；v7 奖励臂（`v7_arm1_ttc`、`v7_arm1_collision_suppress`、`v7_arm2_bundle_kl`、
  `v7_arm3_bundle_only`、`v7_reward_A/B/C`）；v7 结构臂 K-anchor 三件套
  （`v7_struct_b_v5_{train,model,eval}.yaml`）。
- **退役测试快照**：`*.py.disabled` 同目录（防 pytest 收集）。
- **索引**：`docs/archive/config_arms_legacy/README.md`（逐项原路径 → 现路径映射）；
  归档操作、停移收尾与引用检查：`docs/cleanup/V8_CLEANUP_config_docs.md`。
