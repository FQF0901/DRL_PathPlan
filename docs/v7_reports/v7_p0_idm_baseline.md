# v7-P0 fix-15：IDM 现口径复测 + spec-seed 变体（Gate A 补证 ②）

- 日期：2026-10-02（runner 09:23 排队等 GPU 空闲；4 个 run 10:02:21–10:09:44 串行完成）
- 状态：**完成**（4/4 主 run rc=0、n_error=0；评测全程 GPU/评测串行，无双跑；确定性/深挖补跑 11 次另计，全部串行；结论见文末补记）
- 树：评测时 HEAD `b99e1272869e8846f3f51a833d2bd044ec2ab4bd`（同期自 `f5f0dc3` 前进 1 个 obs/net commit；`env/expert/`、`env/scenario/`、`pipeline/eval_runner.py`、`config/` 无 diff；当前树 `git status --porcelain` = 0）
- 口径（冻结 pin）：`--policy baseline --workers 6 --tracker lqr --eval-reference plan --max-steps 1000 --config config/default.yaml`（baseline 不消费 tracker/eval-reference）
- 产物：`/tmp/opencode/v7_p0_idm_baseline.md`（本档）、`/tmp/opencode/v7_p0_idm_analysis.{md,json}`（全表 + 机读）、`runs/*_v7p0_idm_*`（5 个 run）

## 结论摘要（TL;DR）

1. **eval500 现口径复测 = 历史锚，逐条 500/500 一致**：success **0.756** [0.716, 0.792]、collision **0.144**、off-road **0.068**、rc **0.882**、speed_ratio **0.740**；终止类计数逐项相同（arrive_dest 378 / collision 72 / max_step 16 / out_of_road 34）。逐 episode 除 `duration_s`（墙钟）外所有列完全相同，metrics 仅浮点末位差（≤1e-15）。⇒ 历史 manifest `dirty=1` **无实际影响**；冻结锚 `runs/BTC20260927-1839_eval500_baseline` 可继续作为主判据单 pin baseline（见 §5）。
2. **clean500 IDM 基线（新锚，此前无 baseline run）**：success **0.742** [0.702, 0.778]、collision 0.174、off-road 0.054、rc 0.865、speed_ratio 0.757。
3. **spec-seed 变体 spread**（同 500 模板、per-scenario seed +1e5/+2e5）：success 0.706（A）/ 0.724（B）（原 0.756）⇒ 3 个 seed 集 **mean 0.729、sd 2.5pp、range 5.0pp**（仅 2 变体：0.715 ± 1.3pp）；collision 0.144–0.202（sd 2.9pp / range 5.8pp）；off-road 0.056–0.068（sd 0.7pp / range 1.2pp）。
4. **确定性异常（重要）**：同一 spec 重复评测**并非严格恒等**：全量复评 6 次中出现 **1/500 success↔collision 翻转（id 299：arrive_dest/277 步 ↔ collision/188 步）+ 1–2/500 数值漂移（id 524 steps 337/338/339；id 427 等）**，success 约 ±0.2pp。探针（id 299 单条 spec + `--workers 1` + fresh 进程）仍 3/6 翻转 ⇒ **内禀逐 episode 非确定性**；已排除：PYTHONHASHSEED（hash0 全量仍翻）、numpy 全局 RNG（`np.random.seed(0)` 仍翻）、worker 调度/前序（单条 spec 也翻）、reset 期 seeding（engine/mgr seed 链跨进程一致）。两 flaky 场景均为 `SyS` hard 脚本 cut_in；与 `env/scenario/README.md`「事件类 KPI 运行间噪声」记载一致（其根因归为 MetaDrive 内部线程/挂钟时序，未定位到代码行）。⇒ 预注册"同 `(id,seed)` 评测确定性"**并非对全部 episode 成立**；配对协议可吸收该噪声（±0.2pp），但不能声称"同 spec 恒等重复"。
5. **确定性对照（跨 harness）**：历史锚 54f8c5e vs 现树 b99e127 复评逐 episode 除 `duration_s` 墙钟外**全部列恒等**（500/500）——在默认随机 hash 下这意味着历史锚恰好复现。per-id 跨 seed 稳定性低（success 三 seed 集全过仅 **268/500**，0/3 有 54/500）⇒ 主判据中 baseline 必须与 agent 逐 `(id,seed)` 配对（消实例方差）；spec-seed spread 只作基线稳定性读数，不进入主判据（与预注册 §2.5/§3 一致）。
6. **分层 seed 敏感性**：success range 最大 = hard **14.1pp**、uturn **15.6pp**、merge **13.0pp**、t_intersection **11.1pp**（n≈45）；easy 仅 4.4pp。几何层读数比较时应带 seed 波动量级。

## 方法与执行

**执行纪律**：runner 先等 3637353（v6p4 复跑队列，09:41–09:56）与另一支 09:56–10:02 的评测结束；每次评测前重查 `test/train/spawn_main == 0` 且 MemAvailable ≥ 3GB。4 个 run 严格串行（各 ~107–113s），watcher 证实期间无其它 test 进程并发。

**逐 run 命令**（原样记录，`--name` 由 `tools/test.py` 规范化为 `BTC<ts>_<name>`）：

```bash
# 1) eval500 复测（现树 vs 历史锚）
tools/venv-python tools/test.py --policy baseline --spec env/specs/scenarios_eval500.json \
  --out runs --name v7p0_idm_eval500_rerun --workers 6 --tracker lqr --config config/default.yaml --eval-reference plan
# 2) clean500（val-only，选点集）IDM 基线
tools/venv-python tools/test.py --policy baseline --spec /tmp/opencode/phase3_diag/exp/specs_val_only500.json \
  --out runs --name v7p0_idm_clean500 --workers 6 --tracker lqr --config config/default.yaml --eval-reference plan
# 3/4) spec-seed 变体 +1e5 / +2e5
tools/venv-python tools/test.py --policy baseline --spec /tmp/opencode/v7_p0_specs/scenarios_eval500_seedA.json \
  --out runs --name v7p0_idm_eval500_seedA --workers 6 --tracker lqr --config config/default.yaml --eval-reference plan
tools/venv-python tools/test.py --policy baseline --spec /tmp/opencode/v7_p0_specs/scenarios_eval500_seedB.json \
  --out runs --name v7p0_idm_eval500_seedB --workers 6 --tracker lqr --config config/default.yaml --eval-reference plan
```

**变体生成**（`/tmp/opencode/v7_p0_make_variants.py`，sha256 `a42c1c93…`）：读入 eval500 spec，逐条 `seed += offset` 且 `traffic.seed += offset`；id/geometry/blocks/traffic(除 seed)/limits/nav/ego/difficulty/labels 逐条断言不变；每条过 `ScenarioSpec.from_dict().validate()`；三集 seed 两两互斥。产物与哈希：

| 变体 | 偏移 | seed 范围 | sha256 | 文件 |
|---|---|---|---|---|
| seedA | +100000 | 5100004–5100999 | `e31a87b232777ccc…b6635` | `/tmp/opencode/v7_p0_specs/scenarios_eval500_seedA.json` |
| seedB | +200000 | 5200004–5200999 | `fa644eb49d85999c…e7dc` | `/tmp/opencode/v7_p0_specs/scenarios_eval500_seedB.json` |
| manifest | — | — | `489326e8a8f1ecb5…625f` | `/tmp/opencode/v7_p0_specs/variants_manifest.json` |

**分析**：`/tmp/opencode/v7_p0_idm_analysis.py`（复用 `tools/paired_eval.py` 的 `load_episodes_csv / pair_episodes / paired_stats / mcnemar_exact_p`，bootstrap B=10000 seed=0）。

**异常深挖（主任务外交付，但直接关系评测口径）**：clean 复跑出现 1/500 success 翻转后，追加：(a) 8 条子集探针（`/tmp/opencode/v7_p0_specs/determinism_probe8.json`，含 id 299/524 + 6 条对照）默认 5 跑 + `PYTHONHASHSEED=0` 5 跑 + `--workers 1` 5 跑；(b) id 299 单条 spec `--workers 1` ×6 + `np.random.seed(0)` 包装 ×6；(c) seed 链探针（fresh 进程 build_env+reset）；(d) `PYTHONHASHSEED=0` 下 eval500 全量 ×2 + clean500 ×1。结论：非确定性为内禀逐 episode 熵，锁定在 MetaDrive/Panda3D 内部（未定位到代码行）；hash pin / numpy seed / 调度均非充分修复。证据与读数见文末补记与 `/tmp/opencode/v7_p0_determinism_probe.json`。

## 0. Run 清单与证据哈希

| label | run dir | spec | spec sha256 | episodes.csv sha256 | rows | n_error | wall_s | manifest git |
|---|---|---|---|---|---|---|---|---|
| hist | BTC20260927-1839_eval500_baseline | /workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_eval500.json | 98856105eca17461… | d6269ef238f11f9b… | 500 | 0 | 148 | git: 54f8c5e83a18c521d28584d0080fb38f247ea33f @Planner_RL dirty=1 |
| eval500_rerun | BTC20261002-100221_v7p0_idm_eval500_rerun | /workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_eval500.json | 98856105eca17461… | 8fac7f295f1b9fbb… | 500 | 0 | 110 | git: b99e1272869e8846f3f51a833d2bd044ec2ab4bd @Planner_RL dirty=1 |
| clean500 | BTC20261002-100413_v7p0_idm_clean500 | /tmp/opencode/phase3_diag/exp/specs_val_only500.json | 087db3f5c02e9ced… | 9bf57b196abc49bc… | 500 | 0 | 112 | git: b99e1272869e8846f3f51a833d2bd044ec2ab4bd @Planner_RL dirty=1 |
| eval500_seedA | BTC20261002-100606_v7p0_idm_eval500_seedA | /tmp/opencode/v7_p0_specs/scenarios_eval500_seedA.json | e31a87b232777ccc… | 905e05f36dc63613… | 500 | 0 | 109 | git: b99e1272869e8846f3f51a833d2bd044ec2ab4bd @Planner_RL dirty=1 |
| eval500_seedB | BTC20261002-100757_v7p0_idm_eval500_seedB | /tmp/opencode/v7_p0_specs/scenarios_eval500_seedB.json | fa644eb49d85999c… | 3f4988bc1cc1c8f0… | 500 | 0 | 106 | git: b99e1272869e8846f3f51a833d2bd044ec2ab4bd @Planner_RL dirty=1 |

## 1. 总体读数（现口径复测 vs 历史锚）

| run | n | success | 95% Wilson CI | collision | off-road | rc | speed_ratio | max_step | mean_steps |
|---|---|---|---|---|---|---|---|---|---|
| hist | 500 | 0.756 | [0.716,0.792] | 0.144 | 0.068 | 0.882 | 0.740 | 0.032 | 366.2 |
| eval500_rerun | 500 | 0.756 | [0.716,0.792] | 0.144 | 0.068 | 0.882 | 0.740 | 0.032 | 366.2 |
| clean500 | 500 | 0.742 | [0.702,0.778] | 0.174 | 0.054 | 0.865 | 0.757 | 0.030 | 354.7 |
| eval500_seedA | 500 | 0.706 | [0.665,0.744] | 0.202 | 0.056 | 0.851 | 0.728 | 0.036 | 359.3 |
| eval500_seedB | 500 | 0.724 | [0.683,0.761] | 0.182 | 0.068 | 0.860 | 0.747 | 0.026 | 354.5 |

## 2. eval500 复测 vs 历史锚（逐 (id,seed) 配对；dirty=1 风险核对）

- 配对键: n_common=500, hist_only=0, rerun_only=0（fail-closed 通过）
- **success**: 0.756 → 0.756 (Δ=+0.00pp, 95%CI [+0.00,+0.00]), 2×2 fixed=0 broken=0 both_pass=378 both_fail=122, net=0, z=n/a, McNemar p=1
- **collision**: 0.144 → 0.144 (Δ=+0.00pp, 95%CI [+0.00,+0.00]), 2×2 fixed=0 broken=0 both_pass=72 both_fail=428, net=0, z=n/a, McNemar p=1
- **off_road**: 0.068 → 0.068 (Δ=+0.00pp, 95%CI [+0.00,+0.00]), 2×2 fixed=0 broken=0 both_pass=34 both_fail=466, net=0, z=n/a, McNemar p=1
- **max_step**: 0.032 → 0.032 (Δ=+0.00pp, 95%CI [+0.00,+0.00]), 2×2 fixed=0 broken=0 both_pass=16 both_fail=484, net=0, z=n/a, McNemar p=1
- **逐条完全一致（success/collision/off-road/max_step/rc 全等）**: 500/500
- 不一致事件计数: 无

### 2.1 分层读数 delta（rerun − hist）

| by_primary | n | success Δ(pp) | collision Δ(pp) | off-road Δ(pp) | rc Δ | speed_ratio Δ |
|---|---|---|---|---|---|---|
| split | 45/45 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| straight | 45/45 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| ramp_out | 46/46 | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 |
| ramp_in | 46/46 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| merge | 46/46 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| intersection | 46/46 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| t_intersection | 45/45 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| curve | 46/46 | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 |
| uturn | 45/45 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| roundabout | 45/45 | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 |
| tollgate | 45/45 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 |

| by_difficulty | n | success Δ(pp) | collision Δ(pp) | off-road Δ(pp) | rc Δ | speed_ratio Δ |
|---|---|---|---|---|---|---|
| easy | 183/183 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| medium | 168/168 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| hard | 149/149 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 |

## 3. 分层（几何×难度）: eval500_rerun

### success（行=geometry, 列=difficulty；格式 = 值(n)，n<30 只报不判）
| geometry | easy | medium | hard | all |
|---|---|---|---|---|
| split | 1.000(16) | 0.786(14) | 0.667(15) | 0.822(45) |
| straight | 1.000(14) | 0.941(17) | 0.286(14) | 0.756(45) |
| ramp_out | 1.000(10) | 0.000(13) | 0.696(23) | 0.565(46) |
| ramp_in | 1.000(16) | 0.333(15) | 0.467(15) | 0.609(46) |
| merge | 1.000(18) | 0.846(13) | 0.733(15) | 0.870(46) |
| intersection | 0.947(19) | 0.600(20) | 0.286(7) | 0.696(46) |
| t_intersection | 1.000(16) | 1.000(11) | 0.667(18) | 0.867(45) |
| curve | 1.000(17) | 0.857(14) | 0.333(15) | 0.739(46) |
| uturn | 0.941(17) | 0.846(13) | 0.800(15) | 0.867(45) |
| roundabout | 0.929(28) | 0.471(17) | — | 0.756(45) |
| tollgate | 1.000(12) | 0.762(21) | 0.583(12) | 0.778(45) |

### collision（行=geometry, 列=difficulty；格式 = 值(n)，n<30 只报不判）
| geometry | easy | medium | hard | all |
|---|---|---|---|---|
| split | 0.000(16) | 0.214(14) | 0.267(15) | 0.156(45) |
| straight | 0.000(14) | 0.059(17) | 0.643(14) | 0.222(45) |
| ramp_out | 0.000(10) | 0.000(13) | 0.304(23) | 0.152(46) |
| ramp_in | 0.000(16) | 0.000(15) | 0.333(15) | 0.109(46) |
| merge | 0.000(18) | 0.154(13) | 0.267(15) | 0.130(46) |
| intersection | 0.053(19) | 0.300(20) | 0.714(7) | 0.261(46) |
| t_intersection | 0.000(16) | 0.000(11) | 0.333(18) | 0.133(45) |
| curve | 0.000(17) | 0.143(14) | 0.267(15) | 0.130(46) |
| uturn | 0.059(17) | 0.154(13) | 0.200(15) | 0.133(45) |
| roundabout | 0.071(28) | 0.118(17) | — | 0.089(45) |
| tollgate | 0.000(12) | 0.143(21) | 0.000(12) | 0.067(45) |

### off_road（行=geometry, 列=difficulty；格式 = 值(n)，n<30 只报不判）
| geometry | easy | medium | hard | all |
|---|---|---|---|---|
| split | 0.000(16) | 0.000(14) | 0.000(15) | 0.000(45) |
| straight | 0.000(14) | 0.000(17) | 0.000(14) | 0.000(45) |
| ramp_out | 0.000(10) | 1.000(13) | 0.000(23) | 0.283(46) |
| ramp_in | 0.000(16) | 0.667(15) | 0.000(15) | 0.217(46) |
| merge | 0.000(18) | 0.000(13) | 0.000(15) | 0.000(46) |
| intersection | 0.000(19) | 0.100(20) | 0.000(7) | 0.043(46) |
| t_intersection | 0.000(16) | 0.000(11) | 0.000(18) | 0.000(45) |
| curve | 0.000(17) | 0.000(14) | 0.000(15) | 0.000(46) |
| uturn | 0.000(17) | 0.000(13) | 0.000(15) | 0.000(45) |
| roundabout | 0.000(28) | 0.412(17) | — | 0.156(45) |
| tollgate | 0.000(12) | 0.095(21) | 0.000(12) | 0.044(45) |

### 边际：difficulty
| difficulty | n | success | collision | off-road | rc | speed_ratio |
|---|---|---|---|---|---|---|
| easy | 183 | 0.978 | 0.022 | 0.000 | 0.974 | 0.884 |
| medium | 168 | 0.673 | 0.125 | 0.202 | 0.876 | 0.847 |
| hard | 149 | 0.577 | 0.315 | 0.000 | 0.775 | 0.523 |

### 边际：geometry(primary)
| geometry | n | success | collision | off-road | rc | speed_ratio |
|---|---|---|---|---|---|---|
| split | 45 | 0.822 | 0.156 | 0.000 | 0.908 | 0.665 |
| straight | 45 | 0.756 | 0.222 | 0.000 | 0.854 | 0.636 |
| ramp_out | 46 | 0.565 | 0.152 | 0.283 | 0.796 | 0.745 |
| ramp_in | 46 | 0.609 | 0.109 | 0.217 | 0.886 | 0.722 |
| merge | 46 | 0.870 | 0.130 | 0.000 | 0.921 | 0.674 |
| intersection | 46 | 0.696 | 0.261 | 0.043 | 0.841 | 0.842 |
| t_intersection | 45 | 0.867 | 0.133 | 0.000 | 0.914 | 0.802 |
| curve | 46 | 0.739 | 0.130 | 0.000 | 0.863 | 0.652 |
| uturn | 45 | 0.867 | 0.133 | 0.000 | 0.920 | 0.875 |
| roundabout | 45 | 0.756 | 0.089 | 0.156 | 0.881 | 0.914 |
| tollgate | 45 | 0.778 | 0.067 | 0.044 | 0.918 | 0.622 |

## 3. 分层（几何×难度）: clean500

### success（行=geometry, 列=difficulty；格式 = 值(n)，n<30 只报不判）
| geometry | easy | medium | hard | all |
|---|---|---|---|---|
| split | 1.000(17) | 0.562(16) | 0.769(13) | 0.783(46) |
| straight | 1.000(18) | 1.000(19) | 0.222(9) | 0.848(46) |
| ramp_out | 1.000(13) | 0.000(11) | 0.429(21) | 0.489(45) |
| ramp_in | 1.000(20) | 0.500(16) | 0.333(9) | 0.689(45) |
| merge | 0.941(17) | 0.917(12) | 0.375(16) | 0.733(45) |
| intersection | 1.000(16) | 0.556(18) | 0.455(11) | 0.689(45) |
| t_intersection | 1.000(12) | 0.952(21) | 0.462(13) | 0.826(46) |
| curve | 1.000(13) | 1.000(13) | 0.158(19) | 0.644(45) |
| uturn | 0.933(15) | 0.833(18) | 0.538(13) | 0.783(46) |
| roundabout | 0.917(36) | 0.400(10) | — | 0.804(46) |
| tollgate | 1.000(12) | 1.000(16) | 0.647(17) | 0.867(45) |

### collision（行=geometry, 列=difficulty；格式 = 值(n)，n<30 只报不判）
| geometry | easy | medium | hard | all |
|---|---|---|---|---|
| split | 0.000(17) | 0.438(16) | 0.154(13) | 0.196(46) |
| straight | 0.000(18) | 0.000(19) | 0.556(9) | 0.109(46) |
| ramp_out | 0.000(13) | 0.000(11) | 0.524(21) | 0.244(45) |
| ramp_in | 0.000(20) | 0.062(16) | 0.556(9) | 0.133(45) |
| merge | 0.059(17) | 0.083(12) | 0.500(16) | 0.222(45) |
| intersection | 0.000(16) | 0.222(18) | 0.545(11) | 0.222(45) |
| t_intersection | 0.000(12) | 0.048(21) | 0.538(13) | 0.174(46) |
| curve | 0.000(13) | 0.000(13) | 0.526(19) | 0.222(45) |
| uturn | 0.067(15) | 0.167(18) | 0.462(13) | 0.217(46) |
| roundabout | 0.083(36) | 0.200(10) | — | 0.109(46) |
| tollgate | 0.000(12) | 0.000(16) | 0.176(17) | 0.067(45) |

### off_road（行=geometry, 列=difficulty；格式 = 值(n)，n<30 只报不判）
| geometry | easy | medium | hard | all |
|---|---|---|---|---|
| split | 0.000(17) | 0.000(16) | 0.000(13) | 0.000(46) |
| straight | 0.000(18) | 0.000(19) | 0.000(9) | 0.000(46) |
| ramp_out | 0.000(13) | 1.000(11) | 0.000(21) | 0.244(45) |
| ramp_in | 0.000(20) | 0.438(16) | 0.000(9) | 0.156(45) |
| merge | 0.000(17) | 0.000(12) | 0.000(16) | 0.000(45) |
| intersection | 0.000(16) | 0.222(18) | 0.000(11) | 0.089(45) |
| t_intersection | 0.000(12) | 0.000(21) | 0.000(13) | 0.000(46) |
| curve | 0.000(13) | 0.000(13) | 0.000(19) | 0.000(45) |
| uturn | 0.000(15) | 0.000(18) | 0.000(13) | 0.000(46) |
| roundabout | 0.000(36) | 0.400(10) | — | 0.087(46) |
| tollgate | 0.000(12) | 0.000(16) | 0.059(17) | 0.022(45) |

### 边际：difficulty
| difficulty | n | success | collision | off-road | rc | speed_ratio |
|---|---|---|---|---|---|---|
| easy | 189 | 0.974 | 0.026 | 0.000 | 0.972 | 0.885 |
| medium | 170 | 0.735 | 0.112 | 0.153 | 0.883 | 0.850 |
| hard | 141 | 0.440 | 0.447 | 0.007 | 0.702 | 0.529 |

### 边际：geometry(primary)
| geometry | n | success | collision | off-road | rc | speed_ratio |
|---|---|---|---|---|---|---|
| split | 46 | 0.783 | 0.196 | 0.000 | 0.881 | 0.665 |
| straight | 46 | 0.848 | 0.109 | 0.000 | 0.905 | 0.706 |
| ramp_out | 45 | 0.489 | 0.244 | 0.244 | 0.729 | 0.714 |
| ramp_in | 45 | 0.689 | 0.133 | 0.156 | 0.903 | 0.772 |
| merge | 45 | 0.733 | 0.222 | 0.000 | 0.862 | 0.614 |
| intersection | 45 | 0.689 | 0.222 | 0.089 | 0.833 | 0.853 |
| t_intersection | 46 | 0.826 | 0.174 | 0.000 | 0.897 | 0.825 |
| curve | 45 | 0.644 | 0.222 | 0.000 | 0.818 | 0.631 |
| uturn | 46 | 0.783 | 0.217 | 0.000 | 0.872 | 0.884 |
| roundabout | 46 | 0.804 | 0.109 | 0.087 | 0.896 | 0.921 |
| tollgate | 45 | 0.867 | 0.067 | 0.022 | 0.922 | 0.710 |

## 4. spec-seed 变体 spread（同 500 模板，per-scenario seed 偏移）

| spec | seed 范围 | n | success [95%CI] | collision | off-road | rc | speed_ratio | max_step |
|---|---|---|---|---|---|---|---|---|
| eval500_rerun | 5000004–5000999 (原) | 500 | 0.756 [0.716,0.792] | 0.144 | 0.068 | 0.882 | 0.740 | 0.032 |
| eval500_seedA | 5100004–5100999 (+1e5) | 500 | 0.706 [0.665,0.744] | 0.202 | 0.056 | 0.851 | 0.728 | 0.036 |
| eval500_seedB | 5200004–5200999 (+2e5) | 500 | 0.724 [0.683,0.761] | 0.182 | 0.068 | 0.860 | 0.747 | 0.026 |

### 4.1 跨 seed 分布（success/collision/off-road 为主）

| 指标 | n(seed 集) | mean | sd(ddof=1) | min | max | range(pp) | 各点 |
|---|---|---|---|---|---|---|---|
| success | 含原 spec(n=3) | 0.729 | 0.025 | 0.706 | 0.756 | 5.00 | rerun=0.756, seedA=0.706, seedB=0.724 |
| success | 仅变体(n=2) | 0.715 | 0.013 | 0.706 | 0.724 | 1.80 | seedA=0.706, seedB=0.724 |
| collision | 含原 spec(n=3) | 0.176 | 0.029 | 0.144 | 0.202 | 5.80 | rerun=0.144, seedA=0.202, seedB=0.182 |
| collision | 仅变体(n=2) | 0.192 | 0.014 | 0.182 | 0.202 | 2.00 | seedA=0.202, seedB=0.182 |
| off-road | 含原 spec(n=3) | 0.064 | 0.007 | 0.056 | 0.068 | 1.20 | rerun=0.068, seedA=0.056, seedB=0.068 |
| off-road | 仅变体(n=2) | 0.062 | 0.008 | 0.056 | 0.068 | 1.20 | seedA=0.056, seedB=0.068 |
| rc | 含原 spec(n=3) | 0.864 | 0.016 | 0.851 | 0.882 | 3.07 | rerun=0.882, seedA=0.851, seedB=0.860 |
| rc | 仅变体(n=2) | 0.855 | 0.006 | 0.851 | 0.860 | 0.86 | seedA=0.851, seedB=0.860 |
| speed_ratio | 含原 spec(n=3) | 0.738 | 0.009 | 0.728 | 0.747 | 1.84 | rerun=0.740, seedA=0.728, seedB=0.747 |
| speed_ratio | 仅变体(n=2) | 0.737 | 0.013 | 0.728 | 0.747 | 1.84 | seedA=0.728, seedB=0.747 |

### 4.2 分层 spread：按 geometry 的 success（3 个 seed 集）

| geometry | n | 原 | +1e5 | +2e5 | range(pp) |
|---|---|---|---|---|---|
| split | 45 | 0.822 | 0.756 | 0.844 | 8.89 |
| straight | 45 | 0.756 | 0.800 | 0.733 | 6.67 |
| ramp_out | 46 | 0.565 | 0.500 | 0.522 | 6.52 |
| ramp_in | 46 | 0.609 | 0.652 | 0.696 | 8.70 |
| merge | 46 | 0.870 | 0.739 | 0.804 | 13.04 |
| intersection | 46 | 0.696 | 0.609 | 0.652 | 8.70 |
| t_intersection | 45 | 0.867 | 0.756 | 0.756 | 11.11 |
| curve | 46 | 0.739 | 0.674 | 0.739 | 6.52 |
| uturn | 45 | 0.867 | 0.733 | 0.711 | 15.56 |
| roundabout | 45 | 0.756 | 0.711 | 0.667 | 8.89 |
| tollgate | 45 | 0.778 | 0.844 | 0.844 | 6.67 |

### 4.3 分层 spread：按 difficulty 的 success

| difficulty | n | 原 | +1e5 | +2e5 | range(pp) |
|---|---|---|---|---|---|
| easy | 183 | 0.978 | 0.934 | 0.967 | 4.37 |
| medium | 168 | 0.673 | 0.696 | 0.673 | 2.38 |
| hard | 149 | 0.577 | 0.436 | 0.483 | 14.09 |

### 4.4 同 id 跨 seed 结果翻转（描述性；不进入主判据）

- 三集合共同 id 数: 500/500
- **success**: 各 id 命中数分布 {'0/3': 54, '1/3': 67, '2/3': 111, '3/3': 268}；翻转 原↔A=129/500, 原↔B=102/500, A↔B=125/500
- **collision**: 各 id 命中数分布 {'0/3': 321, '1/3': 111, '2/3': 51, '3/3': 17}；翻转 原↔A=111/500, 原↔B=97/500, A↔B=116/500
- **off_road**: 各 id 命中数分布 {'0/3': 451, '1/3': 20, '2/3': 11, '3/3': 18}；翻转 原↔A=24/500, 原↔B=18/500, A↔B=20/500


[JSON] /tmp/opencode/v7_p0_idm_analysis.json

## 5. dirty=1 风险结论（Gate A 补证点）

**结论：历史 baseline manifest 的 `dirty=1` 是 run_paths 取证代码在**干净工作树**上的假象（至多 1 个无关文件），对 IDM 口径无实际影响；冻结锚 0.756 可继续作为主判据单 pin baseline。**

证据链：

1. **直接复现**：现树 `b99e127` + 现 harness 复跑 eval500，与历史 run `BTC20260927-1839`（54f8c5e）逐 episode 比较：success/collision/off-road/max_step 四类事件 + rc **500/500 全等**（fixed=0、broken=0），终止类计数逐项相同；逐列扫描仅 `duration_s`（墙钟）不同，metrics 汇总仅浮点末位差（route_completion_mean 差 1e-16、mean_duration_s 为墙钟）。
2. **代码面**：`env/expert/pure_pursuit_idm.py`（baseline policy 本体）在 `54f8c5e` 与 HEAD 逐字节一致（sha256 `015db21556ea…`）；`env/scenario/spec.py`、`env/scenario/behaviors.py`、`config/env.yaml` 同样未变。历史 run 之后第一个 commit `e8a5fd6`（很可能是当时的 dirty 工作树）仅改 `config/eval.yaml` 默认 spec + docs + `tools/test.sh`——而该 run 显式传了 `--spec env/specs/scenarios_eval500.json`，不受默认值影响。baseline 路径 `tracker=None`、不消费 `--eval-reference`。
3. **dirty 计数机制（决定性）**：`pipeline/run_paths.py::_git` 实现为 `return proc.stdout.strip() or "unknown"`；干净树上 `git status --porcelain` 输出为空 → 返回字符串 `"unknown"` → `len("unknown".splitlines())` = **1** ⇒ **`dirty=1` 正好是"干净树"的默认输出**（HEAD 与 54f8c5e 的实现逐字相同，历史同样适用）。若树真脏，计数 = 实际改动文件数（如 3、7），仅当恰好 1 个文件时与假象同值。
4. **补强实验**：追加复跑 `v7p0_idm_eval500_rerun_clean` 前/后 `git status --porcelain` 均为 **0**，其 manifest 仍写 `dirty=1` → 证实假象。该复跑 episodes.csv sha 与首跑不同，原因不是树脏，而是评测自身的 hash 非确定性（见 §7；id 299 翻转）——恰好构成非确定性证据。
5. 综合 1–4：`dirty=1` 对 IDM baseline KPI 无观测影响。主判据 pin 继续用 `runs/BTC20260927-1839_eval500_baseline`；`runs/BTC20261002-100221_v7p0_idm_eval500_rerun` 可作现树复现副本归档（单 baseline 原则下不得交叉配对）。

## 6. 对 v7 协议的含义

- **单 baseline pin 安全**：现口径复测与冻结锚逐条等价，主判据无需改 pin（若改 pin 也无实质影响，但应二选一、禁止混用）。
- **评测确定性（新发现，见 §7）**：默认配置下约 1–2/500 episode 逐 run 非确定（success ±0.2pp）；PYTHONHASHSEED / numpy 全局 RNG / worker 调度均非充分原因，**无法靠单 pin 配置消除**（hash0 全量 ×2 仍翻 id 299）。建议：(a) 把 baseline 读数按"分布"理解（run 级噪声 ±0.2pp，远小于 5pt 方差闸与 +3pt 判据，不影响主判据）；(b) 若要严格可复现（如逐 episode 证据链），需专立"MetaDrive 事件/物理非确定性审计"（建议 oracle 立项；候选方向：Panda3D taskMgr/挂钟、脚本 actor 交接时序）；(c) 主判据继续按 `(id,seed)` 配对，噪声部分被配对差抵消但非零；(d) 不要在预注册中沿用"同 spec 恒等重复（确定性）"的表述。
- **spec-seed 变体只作稳定性读数**：success 3 seed 集 mean 0.729 / sd 2.5pp / range 5.0pp（2 变体 0.715 ± 1.3pp；collision 0.176 ± 2.9pp；off-road 0.064 ± 0.7pp）。与预注册「run 级 sd ≤5pt 方差闸」对照：**IDM 自身跨 seed 波动 sd 约 2–3pp，range 到 5–6pp**——方差闸 5pt 与基线波动同量级，配对口径（同 (id,seed)）是消方差的关键；上述读数含少量 hash 噪声，但量级由场景实例变化主导。
- **实例方差主导**：同 id 跨 seed 的 success 只有 268/500 三个 seed 集全成功（0/3 = 54，1–2/3 = 178）；翻转量 原↔A 129/500、A↔B 125/500，而 collision/off-road 翻转 97–116 / 18–24。因此：任何跨 run 汇总必须逐场景配对（`paired_eval.py` fail-closed 同口径）；比较分层读数时按同 seed 集比较。
- **clean500 与 eval500 不可混用读数**：IDM 两集读数不同（success 0.742 vs 0.756；collision 0.174 vs 0.144；off-road 0.054 vs 0.068）。clean500 继续只用于选点/早停（预注册 §4），不得据此推断 eval500 结论。
- **如需更紧的 spread 估计**：可再加 +3e5/+4e5 变体（每个 ~2 min），当前 3 点已可界定量级（≤6pp）。变体 seed 偏移只改场景实例，不改变模板 id/几何。

## 7. 异常与限制

- **评测运行间非确定性（重要）**：同 spec 同树重复全量复评**不是严格恒等**——6 次全量中观察到：id 299 在 `arrive_dest`（success, 277 步）与 `collision`（188 步）两态间翻转（hist/rerun 为 arrive；rerun_clean、hash0_r1 为 collision；hash0_r2 为 arrive），id 427 数值漂移（hash0 对内 14 列）、id 524 steps 337/338/339 三态（默认对内）；其余约 498/500 条逐列一致（`duration_s` 墙钟列除外）。全量 success 波动 ±1 episode ≈ **±0.2pp**。**探针与排除项**（详证 `/tmp/opencode/v7_p0_determinism_probe.json`）：(i) 8 条子集默认 ×5：6 对照全稳，299 = 3×arrive/2×collision，524 三态；(ii) 同子集 `PYTHONHASHSEED=0` ×5 看似全稳，但 **hash0 全量 ×2 仍翻（299 + 427）** ⇒ hash pin 非充分；(iii) id 299 单条 spec + `--workers 1` + fresh 进程 ×6 仍 3/6 翻 ⇒ 与调度/前序无关，为**内禀逐 episode 熵**；(iv) 同设置 + `np.random.seed(0)` ×6 仍 2/6 翻 ⇒ 排除 numpy 全局 RNG；(v) 4 个 fresh 进程 build_env+reset 的 seed 链完全一致（`engine.global_seed=mgr.random_seed=5000299`，engine/mgr np_random 首值 40133/24552）⇒ reset 期 seeding 无熵。两 flaky 场景均为 `SyS` hard 脚本 cut_in（299: cut_in right trigger 42；524: cut_in left trigger 75），与 `env/scenario/README.md`「事件类 KPI 带运行间噪声」一致；**根因未定位到代码行**，最可能为 MetaDrive/Panda3D 内部 taskMgr/挂钟时序（README 原归因），需专项审计。**影响**：配对 `(id,seed)` 评估吸收大部分；单 run baseline 读数有 ±0.2pp 噪声；"同 spec 重复=恒等重复"的预注册表述需修订。
- **manifest dirty=1（两次）**：`_git` 的 clean-tree 假象（§5.3），非真实脏树；对 KPI 无影响（§5.1）。
- **clean500 spec 不在 repo 冻结路径**：`/tmp/opencode/phase3_diag/exp/specs_val_only500.json`（sha256 `087db3f5…`）；若该路径被清理，run 不可原位重放（本 run 的 episodes.csv + spec sha 已归档）。
- **变体 n=2（含原 =3）**：sd 自由度低；报告以 range 为主、sd 为辅，不据此做区间推断。
- **变体语义**：换 seed 同时改变地图/交通实例与随机流，spread 混合「实例难度差异 + 策略响应」，不是纯策略随机性——这是预注册定义的口径（同模板、不同 per-scenario seed），不拆分；其读数也含上述 hash 噪声（单点 ~±0.2pp），量级不改变结论。
- **workers=6 vs 历史 4**：逐条一致（除 hash flaky 条），worker 数不引入差异；`wall_time_s ≈ 110s` 不可与历史 148s 直接比（机器负载不同）。
- **未做**：变体上的几何×难度完整交叉表只在 eval500/clean500 两个主集输出；变体只输出边际（geometry/difficulty）+ 总体（如需变体交叉表可由 `episodes.csv` 即时再生）。

## 8. 证据索引

| 类别 | 路径 | sha256（截断） |
|---|---|---|
| 历史锚 run | `runs/BTC20260927-1839_eval500_baseline/` | ep `d6269ef238f11f9b…` |
| eval500 复测 run | `runs/BTC20261002-100221_v7p0_idm_eval500_rerun/` | ep `8fac7f295f1b9fbb…` |
| eval500 dirty=0 复跑 | `runs/BTC20261002-101320_v7p0_idm_eval500_rerun_clean/` | ep `bbaa8c4a3e583ce5…` |
| eval500 hash0 ×2 | `runs/BTC20261002-101954…_hash0_r1/`、`runs/BTC20261002-102146…_hash0_r2/` | 见补记 C |
| clean500 run | `runs/BTC20261002-100413_v7p0_idm_clean500/` | ep `9bf57b196abc49bc…` |
| clean500 hash0 | `runs/BTC20261002-102339_v7p0_idm_clean500_hash0/` | 见补记 C |
| 变体 A run | `runs/BTC20261002-100606_v7p0_idm_eval500_seedA/` | ep `905e05f36dc63613…` |
| 变体 B run | `runs/BTC20261002-100757_v7p0_idm_eval500_seedB/` | ep `3f4988bc1cc1c8f0…` |
| 探针 runs（27 个） | `runs/*_v7p0_probe8_r*` / `*_probe8_h0_r*` / `*_probe8_w1_r*` / `*_probe299_w1_r*` / `*_probe299_seeded_r*` | — |
| 探针 spec | `/tmp/opencode/v7_p0_specs/{determinism_probe8,probe_299}.json` | — |
| 探针证据 JSON | `/tmp/opencode/v7_p0_determinism_probe.json` | — |
| eval500 spec（冻结） | `env/specs/scenarios_eval500.json` | `98856105eca17461…` |
| clean500 spec | `/tmp/opencode/phase3_diag/exp/specs_val_only500.json` | `087db3f5c02e9ced…` |
| 变体 A spec | `/tmp/opencode/v7_p0_specs/scenarios_eval500_seedA.json` | `e31a87b232777ccc…` |
| 变体 B spec | `/tmp/opencode/v7_p0_specs/scenarios_eval500_seedB.json` | `fa644eb49d85999c…` |
| 变体 manifest | `/tmp/opencode/v7_p0_specs/variants_manifest.json` | `489326e8…` |
| runner / 日志 / 状态 | `/tmp/opencode/v7_p0_idm_runs.{sh,log}`、`/tmp/opencode/v7_p0_idm_runs_status.json` | — |
| 分析（全表/JSON） | `/tmp/opencode/v7_p0_idm_analysis.{md,json}` | — |
| 重建脚本 | `v7_p0_make_variants.py` / `v7_p0_idm_analysis.py` / `v7_p0_seed_probe.py` / `v7_p0_collect_probe.py` / `v7_p0_seeded_test.py` / `v7_p0_hash0_runs.sh` / `v7_p0_extra_rerun.sh` | gen `a42c1c93…` |

完整 command/argv、manifest、逐 run spec/内部哈希见 `/tmp/opencode/v7_p0_idm_analysis.md` §0 与 `analysis.json`。

## 补记：dirty=0 复跑、确定性探针、hash0 读数（10:13–10:35）

**A. dirty=0 复跑**（`runs/BTC20261002-101320_v7p0_idm_eval500_rerun_clean`，rc=0，500 行）：

- 启动前/后 `git status --porcelain` 均为 0，manifest 仍写 `dirty=1` → 证实 §5.3 的 `_git` clean-tree 假象（空输出回退成 `"unknown"`，计为 1 行）。
- 该复跑 episodes.csv sha `bbaa8c4a…` 与首跑 `8fac7f29…` 不同；逐列定位 = **id 299 success 翻转（arrive→collision）+ id 427 数值漂移 + id 524 steps 338→337**（其余同）→ 由此引出确定性探针。

**B. 确定性探针**（证据 JSON `/tmp/opencode/v7_p0_determinism_probe.json`；`id 299` 两态 = arrive_dest/277 步 vs collision/188 步）：

| # | 实验 | 设置 | 结果 |
|---|---|---|---|
| 1 | 8 条子集（299/524 + 6 对照）默认 ×5 | workers=6 | 6 对照全稳；299 = 3×arrive/2×collision；524 = 337/338/339 |
| 2 | 同子集 `PYTHONHASHSEED=0` ×5 | workers=6 | 看似全稳（299 全 collision-188；524 全 337） |
| 3 | **反证**：`PYTHONHASHSEED=0` 全量 eval500 ×2 | workers=6 | **仍翻**（299 collision→arrive；id 427 数值漂移）⇒ hash pin 非充分 |
| 4 | id 299 单条 spec 默认 ×6 | **workers=1** fresh 进程 | **仍 3/6 翻** ⇒ 与调度/前序无关（内禀） |
| 5 | 同 #4 + `np.random.seed(0)` 包装 ×6 | workers=1 | 仍 2/6 翻 ⇒ 排除 numpy 全局 RNG |
| 6 | seed 链探针 ×4 fresh 进程（build_env+reset） | — | `engine.global_seed=mgr.random_seed=5000299`；engine/mgr np_random 首值 = 40133/24552（全一致）⇒ reset 期 seeding 无熵 |

**全量 run 的 id 299 轨迹**：hist=arrive(277)、rerun=arrive(277)、rerun_clean=collision(188)、hash0_r1=collision(188)、hash0_r2=arrive(277) —— 两态约各半，跨 hash 配置不可控。
**结论**：内禀逐 episode 非确定性，锁定在 MetaDrive/Panda3D 内部（最可能 taskMgr/挂钟时序；未定位到代码行），影响 ~1–2/500 episode / 全量 success ±0.2pp；配对协议可吸收，但"同 spec 恒等重复"不成立。

**C. hash0 全量读数（补录；默认口径对照，不入主判据）**：

| run | success [95%CI] | collision | off-road | rc | speed_ratio | terminations |
|---|---|---|---|---|---|---|
| eval500 hash0_r1 | 0.754 [0.714,0.790] | 0.146 | 0.068 | 0.881 | 0.740 | 377/73/16/34 |
| eval500 hash0_r2 | 0.756 [0.716,0.792] | 0.144 | 0.068 | 0.882 | 0.740 | 378/72/16/34 |
| clean500 hash0 | 0.742 [0.702,0.778] | 0.174 | 0.054 | 0.865 | 0.757 | 371/87/15/27 |

- 两 eval500 读数被默认口径 0.756/0.144/0.068 夹住（±1 episode），clean500 与默认完全相同 ⇒ **hash pin 不改变总体结论**，但**不能**消除 id 299 级 flake（见 B#3）。
- 追加 run 目录：`runs/BTC20261002-101954_v7p0_idm_eval500_hash0_r1`、`..._102146_..._hash0_r2`、`..._102339_..._clean500_hash0`；探针 run 目录 `runs/*_v7p0_probe*`（共 27 个）。
