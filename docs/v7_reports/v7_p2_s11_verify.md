# v7 P2 arm1 seed11 洁净复验报告

- 生成：2026-10-05T08:05+0800；执行：fixer 子代理（只读主树 + worktree 隔离；GPU 串行）
- 目的：洁净复验 seed11 u150 ckpt——dirty 期 `clean500 0.668` 需复现；`eval500 / tg45 / T3`（void）需重跑
- ckpt：`runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt` sha256=`a7cc091f…ba2e`（与 dirty 期一致）
- 代码基线：worktree `/tmp/opencode/v7_pre_v5` @ `2f4450e`（pre-v5，detached；主树复验期间已推进至 `326c260`）
- 解释器：主树 `.venv/bin/python` 绝对路径 + `LD_LIBRARY_PATH=<主树>/.venv/gl/usr/lib/x86_64-linux-gnu`（worktree 无 .venv/gl，等价 `tools/venv-python`）
- spec sha256 pin 校验通过：clean500 `087db3f5…` / tg45 `27010b0e…` / eval500 `98856105…`

## 隔离证据与 `dirty=1` 澄清

- worktree `git status` 全程干净（复验前、运行中 30 点采样、复验后均为空）。
- manifest `dirty=1` **不等于脏**：`pipeline/run_paths._git` 把空 porcelain 输出映射为 `"unknown"`（`stdout.strip() or "unknown"`），计 1 行。实测 manifest 时刻 porcelain=空。⇒ 本复验 3 个 manifest 的 `dirty=1` 均为**干净树**；历史所有干净 run（10-03 的 w1/s0 等）同为 `dirty=1`；dirty 期 void 三连（u125 clean500/tg45/eval500）为 `dirty=15/16`（真实 WIP 行）。
- 由此：dirty 期 07:05 的 u150 clean500（`dirty=1`，早于 WIP 首次改动 07:12:37）亦为干净树。
- smoke（2 spec）：ckpt `loaded=154/154`，obs 无 `ttc` 形状错（v5 WIP 的标志性错误不复现）→ 确证 worktree 代码为 pre-v5。

## 方法偏差（如实报告）

1. **worktree 无 venv**：直接使用主树 venv 绝对路径解释器并显式导出 glvnd `LD_LIBRARY_PATH`（worktree 内 `ensure_gl_library_path` 找不到本地 gl 目录）。
2. **eval500 spec 走主树绝对路径**：`env/specs/*.json` 被 `.gitignore` 且为生成物，worktree 内不存在 `scenarios_eval500.json`；改用主树绝对路径（sha 与 driver pin 一致 `98856105…`，内容即 dirty 期同款）。纯数据、不影响代码隔离。
3. **T3 用一行补丁副本**：原 `/tmp/opencode/v7_p1b_chain/t3_measure.py` 硬编码 `ROOT=主树` 并 `os.chdir(ROOT)`，直接跑会落到 v5 代码。改用 `/tmp/opencode/v7_p2_s11_verify/t3_measure_pre_v5.py`（仅 `ROOT→/tmp/opencode/v7_pre_v5`，diff 恰 1 行），协议（ids/policies=ckpt,baseline/device/config）与 driver 一致。
4. 命令按 driver `build_eval_cmd` 口径补 `--device cuda`（dirty 期 argv 亦含）。

## 四项读数（洁净）

| 项 | run dir | episodes sha256 | n | n_error | succ | coll | off | 用时 |
|---|---|---|---|---|---|---|---|---|
| clean500 u150 | `runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean` | `6419fc8f…` | 500 | 0 | **0.668** | 0.182 | 0.138 | 918s |
| eval500 u150 | `runs/BTC20261005-074755_v7p2_s11_eval500_clean` | `41790134…` | 500 | 0 | **0.646** | 0.202 | 0.140 | 712s |
| tg45 u150 | `runs/BTC20261005-075959_v7p2_s11_tg45_clean` | `7a0caf38…` | 45 | 0 | **0.000** | 0.556 | 0.444 | 57s |
| T3 (9 ids) | `/tmp/opencode/v7_p2_s11_verify/t3_clean.json`（sha `35f4e0ea…`） | — | 9×2 | 0 | 0/9 S1 | — | — | 49.6s |

### 与 dirty 期对照

| 项 | dirty 期 | 洁净复验 | 判定 |
|---|---|---|---|
| clean500 u150 | 0.668，n=500，err=0（dirty=1） | 0.668，n=500，err=0 | **逐 episode 一致**：500/500 的 success+termination 完全相同，唯一差异列 `duration_s`（墙钟）⇒ 复现成立 |
| eval500 | void：n_error=500/500（`obs['ttc']` 形状错） | 0.646，err=0 | void 替换为有效读数 |
| tg45 | void：n_error=45/45 | 0.000，err=0 | void 替换为有效读数 |
| T3 | void：ckpt 9/9 error（`obs['ttc']` 形状错） | 0/9 S1，ckpt 全部有效 | void 替换为有效读数 |

## 配对（`tools/paired_eval.py`；baseline pin 同 driver）

### clean500 u150（agent `BTC20261005-073128_v7p2_s11_u150_clean500_clean`）

| baseline | n | base succ | Δpt | z | McNemar p | 95% CI |
|---|---|---|---|---|---|---|
| w1 `BTC20261003-045912_v7p1dagger_w1_clean500` | 500 | 0.526 | **+14.20** | 6.30 | 1.72e-10 | [+10.0, +18.4] |
| P1-B `BTC20261002-162639_v7p1b_screen_epoch010` | 500 | 0.314 | +35.40 | 12.74 | 6.86e-45 | [+31.0, +39.8] |
| IDM `BTC20261002-100413_v7p0_idm_clean500` | 500 | 0.742 | −7.40 | 3.31 | 0.00119 | [−11.8, −3.0] |

事件项（vs w1）：collision **+11.60pp**（z=6.41）/ off_road **−26.00pp**（z=10.48）/ max_step +0.2pp。
⇒ 与 dirty 期 `v7_p2_paired/s11_clean500_vs_*` 三项**逐位一致**（Δ/z/p/CI 全同）。

### eval500 u150（agent `BTC20261005-074755_v7p2_s11_eval500_clean`，0.646）

| baseline | n | base succ | Δpt | z | McNemar p | 95% CI |
|---|---|---|---|---|---|---|
| P1-B `BTC20261002-164209_v7p1b_eval500_sel` | 500 | 0.312 | **+33.40** | 12.48 | 1.13e-43 | [+29.2, +37.8] |
| w1 探索性 `BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` | 500 | 0.530 | +11.60 | 5.58 | 1.88e-08 | [+7.6, +15.6] |
| IDM `BTC20260927-1839_eval500_baseline` | 500 | 0.756 | −11.00 | 4.88 | 1.14e-06 | [−15.4, −6.8] |

事件项（vs P1-B）：collision +13.0pp（z=7.22）/ off_road −47.8pp（z=15.33）/ max_step +1.2pp。

### tg45（agent `BTC20261005-075959_v7p2_s11_tg45_clean`，0.000）

| baseline | n | base succ | Δpt | z | McNemar p | 95% CI |
|---|---|---|---|---|---|---|
| IDM `BTC20261002-160154_v7p1b_tg45_idm` | 45 | 0.778 | **−77.78** | 5.92 | 5.82e-11 | [−88.9, −64.4] |
| P1-B `BTC20261002-160114_v7p1b_tg45` | 45 | 0.000 | 0.00 | — | 1.0 | [0.0, 0.0] |

事件项（vs IDM）：collision +48.9pp（z=4.31）/ off_road +40.0pp（z=4.02）。

### T3（0/9 S1；ckpt 全部有效）

| id | ckpt term | plan_cross | ckpt_clearance | base term |
|---|---|---|---|---|
| 34 | collision | True | −1.748 | arrive_dest |
| 243 | out_of_road | True | — | arrive_dest |
| 164 | out_of_road | False | — | arrive_dest |
| 84 | out_of_road | True | — | arrive_dest |
| 166 | out_of_road | True | — | arrive_dest |
| 131 | collision | True | — | arrive_dest |
| 76 | collision | True | — | arrive_dest |
| 147 | collision | True | — | arrive_dest |
| 239 | collision | False | — | arrive_dest |

- `n_s1_resolved=0/9`（与 seed0 一致）；ckpt 4×out_of_road + 5×collision；plan 穿岗亭 7/9；baseline 9/9 arrive_dest。辅助条款无命中。

## 结论

1. **clean500 0.668 复现成立且逐 episode 一致**——dirty 期读数非污染、可直接采用（配对三项亦逐位复现）。
2. **eval500=0.646、tg45=0.0、T3=0/9 为有效读数**（n_error=0），替换 dirty 期 void；seed11 终评数据现已完整。
3. 判读（沿用 driver 口径）：seed11 clean500 vs w1 **+14.2pp / z=6.30**，落在 §9 期望带 +8–15pt 且显著 ⇒ 单 run `strong` 成立；但主判据的 **n≥5 / 方差闸仍不满足**（seed11 n=1；2-seed sd=9.3pt 已标注），不得据此宣称超越。
4. 安全闸：clean500 collision vs w1 **+11.6pp（未过相对闸）**；off_road −26pp（过）；tg45 对 IDM 全面落后（0.0 vs 0.778，coll/off 显著更差）。
5. eval500 0.646 较 seed0（0.512）高 +13.4pp，属跨 seed 方差范围（§2.2 σ≈15.8pt 参考），单点不作臂间结论。

## 异常与备注

- 无运行错误；GPU 全程串行（每项前 nvidia-smi 确认无 compute 进程）。
- `dirty=1` 为命名/取证代码怪癖（空 porcelain→"unknown"），已澄清，不影响任何读数。
- 复验期间主树 HEAD 被其他 lane 推进（`aa4c69c`→`326c260`，v5 接线），与本任务无关；**主树未被本任务改动**（`git status` 干净）。
- **worktree 保留**：`/tmp/opencode/v7_pre_v5` @ `2f4450e`，供 arm2/arm3 等 v4 时代 ckpt 复验复用；不再需要时 `git -C /workspace/01_Proj/DRL_PathPlan worktree remove /tmp/opencode/v7_pre_v5`。
  - 后续复验注意：eval500/tg45 spec 需以主树绝对路径传入（worktree 内无生成物）；T3 需用 `t3_measure_pre_v5.py` 类补丁副本（原脚本硬编码主树 ROOT）。

## 产物

- runs：`BTC20261005-073128_v7p2_s11_u150_clean500_clean` / `BTC20261005-074755_v7p2_s11_eval500_clean` / `BTC20261005-075959_v7p2_s11_tg45_clean`（各含 metrics.json / episodes.csv / manifest.txt / monitor）
- 辅助：`/tmp/opencode/v7_p2_s11_verify/{clean500,eval500,tg45,t3_clean}.log`、`t3_clean.json`、`t3_measure_pre_v5.py`、`paired/{clean500_vs_{w1,p1b,idm},eval500_vs_{p1b,w1,idm},tg45_vs_{idm,p1b}}/paired_eval.{md,json}`

## status

`OK` — 4/4 项完成；clean500 0.668 逐 episode 复现；eval500 0.646；tg45 0.0；T3 0/9；n_error 全 0；主树未动；worktree 保留。
