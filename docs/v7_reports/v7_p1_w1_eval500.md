# v7-P1 w1 e005 · eval500 探索性复评报告（Gate B 补证）

- 状态：**done** @ 2026-10-03T06:35 (+0800)
- **标注：探索性复评、不具一次评估独立性**（base 冻结在先：Gate B 已定 w1 e005 为 P2 base 后才跑本次 eval500；round 内 w4 已消耗一次 candidate 评测）
- 冻结 base：`runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt` sha256 `fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9`（只读）
- spec：`env/specs/scenarios_eval500.json` sha256 `98856105eca17461bbdabbf88f203102be7b585fd3de82820b17460358dd4595`（冻结测试集，与预注册 §2.5 一致；未用于选点）
- 命令（实际经 `tools/venv-python` 启动，GPU 串行、启动前 nvidia-smi 无进程）：
  `tools/test.py --policy ckpt --ckpt runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt --spec env/specs/scenarios_eval500.json --out runs --name v7p1dagger_w1_eval500_exploratory --workers 6 --tracker lqr --config config/default.yaml --eval-reference plan`
- run：`runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory`（n=500、n_error=0、wall 600s、workers=6、tracker=lqr、eval-reference=plan）
- 证据：`episodes.csv` sha256 `6597d2eae18b88ee7155b917eef1f608b9da4a3b2916e17f967db79b010e88a7`；`metrics.json` sha256 `3b5dd35a24400253c46d40dd012d30b6b339bb0a951fa108fe84d45056ca6f2e`；原始日志 `/tmp/opencode/v7p1_w1_eval500_run.log`
- 配对产物：`/tmp/opencode/v7_p1_w1_eval500_pairs/{vs_p1b,vs_ebeta2,vs_idm}/paired_eval.{md,json}`（三组各自独立调用，单 baseline pin、fail-closed 通过：500/500 键一致、n_error=0）

---

## 1. eval500 读数（w1 e005）

| 指标 | 读数 | 95% Wilson CI |
|---|---|---|
| success | **0.530** | [0.486, 0.573] |
| collision | 0.074 | [0.054, 0.100] |
| off_road | 0.378 | [0.337, 0.421] |
| solid_line_crossing | 0.378 | [0.337, 0.421] |
| speed_limit_violation | 0.466 | [0.423, 0.510] |
| route_completion (mean) | 0.7336 | — |
| speed_ratio (mean) | 0.4622 | — |
| min_ttc (mean / min) | 7.52 / 0.518 | — |

### 1.1 分层（success；n≈45–46/几何，compound n=218）

| primary | n | success | collision | off_road |
|---|---|---|---|---|
| curve | 46 | 0.283 | 0.087 | 0.609 |
| intersection | 46 | 0.674 | 0.043 | 0.217 |
| merge | 46 | 0.522 | 0.043 | 0.435 |
| ramp_in | 46 | 0.587 | 0.152 | 0.217 |
| ramp_out | 46 | 0.609 | 0.109 | 0.283 |
| roundabout | 45 | 0.378 | 0.089 | 0.489 |
| split | 45 | 0.756 | 0.044 | 0.200 |
| straight | 45 | 0.622 | 0.044 | 0.311 |
| t_intersection | 45 | 0.622 | 0.111 | 0.267 |
| tollgate | 45 | **0.000** | 0.067 | 0.933 |
| uturn | 45 | 0.778 | 0.022 | 0.200 |
| compound(report) | 218 | 0.353 | 0.060 | 0.560 |

| difficulty | n | success | collision | off_road |
|---|---|---|---|---|
| easy | 183 | 0.787 | 0.011 | 0.197 |
| medium | 168 | 0.375 | 0.018 | 0.583 |
| hard | 149 | 0.389 | 0.215 | 0.369 |

**要点**：tollgate 0/45（off_road 0.933）为最大失分块（与 w1 tg45 专项复评 0/45 一致，见 status 第 37 行）；curve 0.283、roundabout 0.378 次之；uturn/split 表现最好（0.778/0.756）。

## 2. 与 clean500 选点读数的一致性

| 集合 | run | success | collision | off_road |
|---|---|---|---|---|
| clean500（val-only，选点用） | `runs/BTC20261003-045912_v7p1dagger_w1_clean500`（ep `042b6330…`） | 0.526 | 0.066 | 0.398 |
| eval500（测试，本次探索性复评） | `runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` | 0.530 | 0.074 | 0.378 |

- 两集合不相交（val-only vs 测试），success 差 **+0.4pp**（0.526 → 0.530）、collision +0.8pp、off_road −2.0pp，均在已知逐 episode 内禀熵量级（≈±0.2pp/全量，预注册 §3 fix-15）附近的窄幅内 ⇒ **无可见的 val→test 选点膨胀**；w1 e005 的 ~0.53 水平在 val/测试两侧自洽。
- 注意：clean500 的 spec 位于 `/tmp/opencode/phase3_diag/exp/specs_val_only500.json`（sha `087db3f5…`，外部依赖，见 §5.6）。

## 3. 三组单 baseline 配对（逐 `(id, seed)`；N=500/500）

| 对照 | baseline success | Δ success | net | z | McNemar p | 配对差 95% CI |
|---|---|---|---|---|---|---|
| vs P1-B eval500（`BTC20261002-164209_v7p1b_eval500_sel`） | 0.312 | **+21.8pp** | +109 | 9.99 | 5.74e-28 | [+18.0, +25.6]pp |
| vs E-β″ eval500（`BTC20261001-200331_v6p3_e3_eval500`） | 0.440 | **+9.0pp** | +45 | 5.13 | 2.42e-07 | [+5.6, +12.4]pp |
| vs IDM eval500（`BTC20260927-1839_eval500_baseline`） | 0.756 | **−22.6pp** | −113 | 8.85 | 3.89e-20 | [−27.2, −18.0]pp |

方向：fixed = baseline 失败→agent 成功；net = fixed − broken；success 越高越好。

### 3.1 2×2 与分项（success / collision / off_road / max_step）

**vs P1-B**（success：fixed 114 / broken 5 / both_pass 151 / both_fail 230）

| 事件 | base→agent | Δ | net | z | p |
|---|---|---|---|---|---|
| collision | 0.072 → 0.074 | +0.2pp | +1 | 0.16 | 1.00 |
| off_road | 0.618 → 0.378 | −24.0pp | −120 | 10.22 | 2.36e-28 |
| max_step | 0.002 → 0.018 | +1.6pp | +8 | 2.53 | 0.0215 |

**vs E-β″**（success：fixed 61 / broken 16 / both_pass 204 / both_fail 219）

| 事件 | base→agent | Δ | net | z | p |
|---|---|---|---|---|---|
| collision | 0.050 → 0.074 | +2.4pp | +12 | 1.66 | 0.126 |
| off_road | 0.498 → 0.378 | −12.0pp | −60 | 6.06 | 7.13e-10 |
| max_step | 0.012 → 0.018 | +0.6pp | +3 | 1.00 | 0.508 |

**vs IDM**（success：fixed 25 / broken 138 / both_pass 240 / both_fail 97）

| 事件 | base→agent | Δ | net | z | p |
|---|---|---|---|---|---|
| collision | 0.144 → 0.074 | −7.0pp | −35 | 3.99 | 8.22e-05 |
| off_road | 0.068 → 0.378 | **+31.0pp** | +155 | 12.45 | 4.38e-47 |
| max_step | 0.032 → 0.018 | −1.4pp | −7 | 1.53 | 0.189 |

`max_step` 事件方向为 lower_is_better；agent 0.018 略高于 P1-B（0.002，p=0.0215），但低于 E-β″/IDM。

### 3.2 分层（success Δpp / McNemar p；agent 列固定为 w1 e005 读数）

| primary | n | agent | vs P1-B | vs E-β″ | vs IDM |
|---|---|---|---|---|---|
| curve | 46 | 0.283 | +28.3 / 2.4e-04 | +4.3 / 0.500 | −45.7 / 5.7e-06 |
| intersection | 46 | 0.674 | +34.8 / 3.1e-05 | +15.2 / 0.065 | −2.2 / 1.00 |
| merge | 46 | 0.522 | +19.6 / 0.0117 | +2.2 / 1.00 | −34.8 / 1.4e-04 |
| ramp_in | 46 | 0.587 | +6.5 / 0.453 | 0.0 / 1.00 | −2.2 / 1.00 |
| ramp_out | 46 | 0.609 | +19.6 / 0.0039 | +6.5 / 0.453 | +4.3 / 0.688 |
| roundabout | 45 | 0.378 | +24.4 / 0.0034 | +2.2 / 1.00 | −37.8 / 7.6e-05 |
| split | 45 | 0.756 | +6.7 / 0.250 | −2.2 / 1.00 | −6.7 / 0.549 |
| straight | 45 | 0.622 | +20.0 / 0.0117 | +11.1 / 0.063 | −13.3 / 0.146 |
| t_intersection | 45 | 0.622 | +42.2 / 3.8e-06 | +26.7 / 0.0042 | −24.4 / 0.0034 |
| tollgate | 45 | 0.000 | 0.0 / 1.00 | 0.0 / 1.00 | **−77.8 / 5.8e-11** |
| uturn | 45 | 0.778 | +37.8 / 1.5e-05 | +33.3 / 6.1e-05 | −8.9 / 0.344 |

| difficulty | n | agent | vs P1-B | vs E-β″ | vs IDM |
|---|---|---|---|---|---|
| easy | 183 | 0.787 | +20.8 / 7.5e-11 | +12.0 / 1.0e-05 | −19.1 / 1.0e-08 |
| medium | 168 | 0.375 | +23.8 / 1.8e-12 | +10.7 / 7.6e-06 | −29.8 / 1.8e-10 |
| hard | 149 | 0.389 | +20.8 / 3.4e-07 | +3.4 / 0.487 | −18.8 / 2.3e-04 |

（分层 n≥30 均判定；`primary_x_difficulty` 细格 n=14–19 <30 只报不判，详见各 `paired_eval.md`。）

### 3.3 主判据状态

- 三组报告 `main_criterion_passed = false`，原因：**n_runs_below_min（n=1 < 5）**；`stability_not_met` 系 n=1 时 sd=null 的占位（非方差发现）。vs IDM 另有 ci_lower_le_zero / iqm_le_zero。
- 单 run 参考判据（预注册 §2.3：Δ ≥ +3pt 且 z ≥ 1.96，仅方向性）：vs P1-B ✓（+21.8pp / z=9.99）、vs E-β″ ✓（+9.0pp / z=5.13）；vs IDM 不适用/不满足。
- **结论边界**：本次为探索性复评，不具一次评估独立性，不单独构成超越/失败结论；主判据须按 §2.2 多 run + 方差闸执行。

## 4. 安全闸读数（对照预注册 §5；仅读数，不裁决）

| 支路 | 读数 | 判读（工具字段/直读） |
|---|---|---|
| collision | 0.074 ≤ 0.10（绝对支路） | 三组配对 `collision_gate=passed`；vs IDM 亦为 −7.0pp 改善 |
| off_road | 0.378 vs IDM 0.068（+31.0pp）；>0.10 且 >IDM+2pt=0.088 | vs P1-B / E-β″ `off_road_gate=passed`（相对各自基线）；**vs IDM 支路 failed** |
| speed_ratio | 0.462 vs IDM 0.740（=0.624×，阈 0.9×） | 直读不达 §5 支路（配对工具不计算该项） |

## 5. 边界与异常（如实）

1. **探索性 / 不独立性**：base 在本次评测前已冻结（Gate B 已定 w1 e005）；round 内 w4 已消耗一次 candidate 评测。本 run 不参与选点，不计为独立的一次性测试集评估证据。
2. **tollgate 全失**：eval500 0/45（off_road 0.933）；w1 tg45 专项复评亦 0/45（`runs/BTC20261003-050855_v7p1dagger_w1_tg45`，status 第 37 行）。IDM tollgate 0.778 ⇒ 与 IDM 的总差（−22.6pp）主要来自 tollgate（−77.8pp）、curve（−45.7pp）、roundabout（−37.8pp）、merge（−34.8pp）。
3. **与 IDM 的安全面**：off-road 支路不达（0.378 vs 0.068），speed_ratio 0.624×；collision 支路通过。若要主张"超越"，按 §5 任一不过即不得宣称。
4. **n=1**：无 run 级 sd、无方差闸可评；工具 JSON 的 `stability_not_met` 仅为占位。单 run 结果对 ±0.2pp 内禀逐 episode 熵敏感（预注册 §3），未做复跑。
5. **确定性**：同 `(id, seed)` 评测并非严格恒等（已登记的 MetaDrive/Panda3D 内禀噪声）；本次未做重复评测。
6. **外部依赖**：clean500 对照依赖 `/tmp/opencode/phase3_diag/exp/specs_val_only500.json`（sha `087db3f5…`），若被清理不可原位重放（沿用既有登记）。
7. **本次未改 repo**：ckpt 只读；仅写 `runs/`（本次 run dir）与 `/tmp/opencode`；manifest `dirty=1` 为既有 clean-tree 假象（未修）。注意：工作树存在**并行进程**的未提交修改（`reward_model/terms.py` mtime 06:27:59、`tests/test_reward_terms.py` 06:28:37，均晚于本评测结束 06:26:34），非本任务产生，不影响已完成评测/配对。
8. 日志/配对输出完整，无进程错误、无 OOM；`n_error=0`、`errors=0`。

## 6. 产物清单

- 报告：`/tmp/opencode/v7_p1_w1_eval500.md`（本文件）
- status：`/tmp/opencode/v7_p1_w1_eval500_status.txt`；主 status 增补 5 行 `stage …`（`/tmp/opencode/v7_p1_dagger_status.txt`）
- 评测：`runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory/`（ep `6597d2ea…`、metrics `3b5dd35a…`）；日志 `/tmp/opencode/v7p1_w1_eval500_run.log`
- 配对：`/tmp/opencode/v7_p1_w1_eval500_pairs/{vs_p1b,vs_ebeta2,vs_idm}/paired_eval.{md,json}`
