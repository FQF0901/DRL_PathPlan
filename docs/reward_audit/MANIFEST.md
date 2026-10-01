# P2/Gate2 奖励审计产物入库（tracked）

> **用途**：`runs/` 在 `.gitignore` 下（Gate2 发现 ⑤）→ P2 审计产物（报告 + rc 档配置草案）
> 在此落 tracked 路径，供 P4 臂消费与复核；原始 `runs/` 路径与 sha256 对照见下表。
> **上游**：`docs/rl_reward_v5.md` §7（执行记录/池重叠声明）、`docs/v6_program_prereg.md` §7.1
> （E-β″ 复算规格）。

## 产物清单（逐位一致：入库副本 = 原始文件）

| tracked 路径 | 原始 `runs/` 路径 | sha256 |
|---|---|---|
| `docs/reward_audit/reward_audit.md` | `runs/reward_audit/report/reward_audit.md` | `e35bbfdf404698a26ef8457e34b88d122d1b5b0c333fc1850b88b54220ea8ba5` |
| `docs/reward_audit/reward_audit.json` | `runs/reward_audit/report/reward_audit.json` | `38eb9f95dad216b48151366b61d2cd56db3dfb9738091715265619d2a20cc8e6` |
| `docs/reward_audit/config_draft_rc1.yaml` | `runs/reward_audit/report/config_draft_rc1.yaml` | `18dda809b45a8c8da8fec63e2a58c3f9beffa12d170cdc766c857533c9db8a64` |
| `docs/reward_audit/config_draft_rc3.yaml` | `runs/reward_audit/report/config_draft_rc3.yaml` | `6f2107d2ce6bb1ea85e418710a24c0d7876e0b609f464cb6454b77a4c1ce7e9b` |
| `docs/reward_audit/config_draft_rc10.yaml` | `runs/reward_audit/report/config_draft_rc10.yaml` | `f3f18a0f40bba03f44a5860703c3ae27cc581713b362f6f422ce251507c6437a` |
| `docs/reward_audit/config_draft_rc30.yaml` | `runs/reward_audit/report/config_draft_rc30.yaml` | `0f43d1f092e35f807fe3cd5aa2b60f4359d478075e6174c871e4ecc068a64e0e` |

## 溯源

- 生成：2026-10-01 16:00:32，`tools/reward_audit.py analyze`（生成时工作树；工具落库 commit `a93d285`）。
- 采集：E-β′ `runs/_refs_rlbase/e_beta_prime/final.pt`（sha256 `c9e2d31e4b9d4f784335ffe693b70ddecd4b0ff0e2d4fb062c82aa947f13ea87`）
  在 pre-v6 快照 `031cc1c`（`/tmp/opencode/v6_pre`）下 rollout 1000 episode；池 `env/specs/scenarios_val.json`（1000 条）；
  分类 arrive 441 / out_of_road 490 / collision 34 / max_step 35；审计 501 / 反解 499（互斥）。
- 每档 rc 反解定稿：rc=1 **+30/−19/−15/−23**；rc=3 +28/−20/−16/−24；rc=10 +22/−23/−18/−29；rc=30 +2/−33/−26/−42（`error` −5；复算差 ≤ 0.494）。

## 已知偏差（Gate2 发现，入库保持逐位一致、不就地修改）

- **池重叠**：采集时 `--exclude` 因空值静默失效（summary `exclude:null`），审计池含 eval500 全部 500 条。
  **E-β″ 复算必须排除**（工具已修：默认排除 eval500，空值报错；见 `docs/rl_reward_v5.md` §7）。
- **划分不可交换**：P2 为同类内按序号交替（collision 两半难度 5.34 vs 8.85）；E-β″ 复算改为分层随机 + A/B 互换（`--split-seed`/`--swap-ab`）。
- **报告 schema 早于工具修复**：本报告 meta 无 `head_commit`/`split_seed`/`min_per_class` 字段（复算报告将记录；Gate2 发现 ④）。
- 采集器 obs 帧错位（post-step vs 训练 pre-step；影响 ≪ 容差）见 `.slim/deepwork/v6-net-retrain.md` Gate2 发现 ②。
