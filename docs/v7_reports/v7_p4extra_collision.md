# v7 P4-extra 报告（s11 u150 + 碰撞抑制单变量：terminal collision −22→−32；seeds 0/11）

- 生成：2026-10-05T23:50:15+0800；预注册：`docs/v7_program_prereg.md` §12（content `eec0abe` / anchor `ee1ee49`）
- 臂：`config/arms/v7_arm1_collision_suppress.yaml`（sha `b8697d54eafc…`）= arm1 逐位不变 + 单变量 `terminal_values.collision` −22→−32（v6 §7.5 arm8-A 口径）
- 起点：s11 u150 `runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`（sha `a7cc091fcbda…`）；pre-v5 worktree `2f4450e`；训练 pins：u100 / ckpt_every 25（候选 u25–u100）/ rollout 256 / spec dagger_r1 500 / envs 1 / KL 锚 0.05→0.02
- 判据（§12.3）：clean500 collision ≤ 0.174 且 success ≥ 0.638；eval500 collision ≤ 0.144 且 success ≥ 0.616；**PASS = 四条全过**（单侧 = partial）
- 参照：s11 clean 0.668/coll 0.182、eval 0.646/coll 0.202；IDM clean 0.742/coll 0.174、eval 0.756/coll 0.144；w1 0.526/0.530

## 判据表（每 seed；双读数）

| seed | 采纳 | clean succ | clean coll | eval succ | eval coll | clean 判定 | eval 判定 | 总判定 |
|---|---|---|---|---|---|---|---|---|
| 0 | u75 | 0.678 | 0.154 | 0.652 | 0.164 | 过 | 未过 | **partial** |
| 11 | u50 | 0.654 | 0.16 | 0.646 | 0.154 | 过 | 未过 | **partial** |

> 判据阈值：clean coll ≤ 0.174 / succ ≥ 0.638；eval coll ≤ 0.144 / succ ≥ 0.616；success 下界 = s11 −3pp。

## 结论（判读；§12.3 口径）

- **总判定：partial（2/2 seeds）** —— clean500 四条全过；eval500 **仅 collision 未过**（0.164 / 0.154 vs 阈值 0.144，超 +2.0pp / +1.0pp），success 四条全过（0.678/0.652、0.654/0.646）。
- **碰撞抑制有效（相对 s11；paired_eval 口径）**：clean500 coll **−2.8pp**（z 2.06，p 0.054）/ **−2.2pp**（z 1.46，p 0.185）；eval500 coll **−3.8pp**（z 2.47，p 0.018，显著）/ **−4.8pp**（z 3.05，p 0.0032，显著）；同时 success 不回吐（clean +1.0 / −1.4pp；eval +0.6 / 0.0pp，均 ns）。
- **安全闸（§5）**：clean500 collision 0.154 / 0.160 **≤ IDM 0.174 ✓**；eval500 0.164 / 0.154 **> IDM 0.144 ✗**（缺口自 s11 的 0.202 收窄 −3.8/−4.8pp）。off_road / speed_ratio 仍超 IDM（见下，本臂未承诺修复）。
- **止损/纪律**：u50+u100 双点闸 2/2 seeds 未触发（无 early-collapse）；keep-best 按 §12.3 执行（采纳 u75 / u50）；无 void。
- **归因（单变量）**：`terminal_values.collision` −22→−32 单变量即可在保持 success（−3pp 内）下把 collision 显著压低；clean500 过闸、eval500 差 1–2pp **不宣称过闸**（按预注册记为 partial）。

## seed=0

- train: rc=0 wall=468.7s out=`runs/BTC20261005-2138_v7p4extra_s0_colls` status=**terminal:partial**
- 收尾断言: ok=True [] | off_road_edge={'first': -0.109162, 'last': -0.143197, 'mean_last20': -0.087959} terminal={'first': -0.289062, 'last': -0.289062, 'mean_last20': -0.128945} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/kl_anchor: first=0.0012 last=0.0058 mean20=0.0049 | u25=0.0042 u50=0.0065 u75=0.0054 u100=0.0058
  - train/reward/terminal: first=-0.2891 last=-0.2891 mean20=-0.2279 | u25=-0.1797 u50=0.1133 u75=-0.1797 u100=-0.2891
  - train/returns/mean: first=-6.1736 last=-14.5361 mean20=-13.3435 | u25=-16.6196 u50=-0.588 u75=-8.8171 u100=-14.5361
- u50 闸: trip=False [] | succ=0.647 coll=0.14 | vs s11: net=1 z=0.28 collΔ=-0.06
- u100 闸: trip=False [] | succ=0.64 coll=0.16 | vs s11: net=0 z=0.0 collΔ=-0.04
- 筛查曲线（sub150 u25→u100；succ/coll 与 vs s11 配对）:
  - u25: succ=0.667 coll=0.16 off=0.147 | vs s11 net=4 z=1.15 collΔ=-0.04
  - u50: succ=0.647 coll=0.14 off=0.2 | vs s11 net=1 z=0.28 collΔ=-0.06
  - u75: succ=0.68 coll=0.12 off=0.173 | vs s11 net=6 z=1.41 collΔ=-0.08
  - u100: succ=0.64 coll=0.16 off=0.18 | vs s11 net=0 z=0.0 collΔ=-0.04
- keep-best sub150 top（合格线 0.61→collision 最低）: u75 succ=0.68 coll=0.12 eligible=True; u50 succ=0.647 coll=0.14 eligible=True
- clean500 u50: succ=0.636 coll=0.164 off=0.164 sr=0.45 | vs s11 net=-16 z=2.31 collΔ=-0.018 | vs w1 net=55
- clean500 u75: succ=0.678 coll=0.154 off=0.14 sr=0.454 | vs s11 net=5 z=0.73 collΔ=-0.028 | vs w1 net=76
- 采纳: u75（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§12.3））ckpt=`runs/BTC20261005-2138_v7p4extra_s0_colls/ckpt_u075.pt` sha=4217abe0d67cc231e48b1fe783c3d2fbd87dac5022a091c71d7b1cf4f54dd342
- **clean500 配对 vs s11（首要）**: Δ=+1.0pp CI(-1.6, 3.6) z=0.73 p=0.56
  - collision 项: Δ=-2.8pp CI(-5.4, -0.2) z=2.06 p=0.0541
  - off_road 项: Δ=+0.2pp CI(-2.2, 2.6) z=0.17 p=1
- clean500 配对 vs w1: Δ=+15.2pp CI(11.2, 19.0) z=7.00 p=7.27e-13 ✓显著
- clean500 配对 vs IDM（安全闸参照）: Δ=-6.4pp CI(-10.6, -2.2) z=2.95 p=0.00412
- **eval500**: succ=0.652 coll=0.164 off=0.158 sr=0.455 rc=0.867 sha=50ba36d9ecf6a59d15bfe06c40dba2f841ff070e2b5531b961cd0b32a9b48b21
  - 配对 vs s11（success, pair_stats）: Δ=+0.6pp z=0.46（ns）
  - **collision 项（paired_eval 口径）**: Δ=-3.8pp CI(-6.8, -0.8) z=2.47 **p=0.0183 ✓显著**（coll_fixed=39 / coll_broken=20）；off_road 项 Δ=+1.8pp z=1.57（ns）；max_step Δ=+1.2pp
  - 配对 vs w1 探索性: Δ=+12.2pp z=6.07 | vs IDM: Δ=-10.4pp z=4.79
- tg45: succ=0.0 off=0.889 coll=0.111 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: clean500 collision ≤ 0.174=0.154(✓)；clean500 success ≥ 0.638=0.678(✓)；eval500 collision ≤ 0.144=0.164(✗)；eval500 success ≥ 0.616=0.652(✓)
- 判定: **partial** clean500: succ=0.678 coll=0.154 → 过；eval500: succ=0.652 coll=0.164 → 未过

## seed=11

- train: rc=0 wall=509.7s out=`runs/BTC20261005-2243_v7p4extra_s11_colls` status=**terminal:partial**
- 收尾断言: ok=True [] | off_road_edge={'first': -0.137628, 'last': -0.114723, 'mean_last20': -0.138681} terminal={'first': -0.179688, 'last': -0.109375, 'mean_last20': -0.146289} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/kl_anchor: first=0.0014 last=0.0145 mean20=0.013 | u25=0.0064 u50=0.0087 u75=0.0099 u100=0.0145
  - train/reward/terminal: first=-0.1797 last=-0.1094 mean20=-0.1266 | u25=-0.1797 u50=-0.1797 u75=-0.0117 u100=-0.1094
  - train/returns/mean: first=-5.9654 last=-11.8141 mean20=-13.2353 | u25=-16.1637 u50=-17.7554 u75=-5.1308 u100=-11.8141
- u50 闸: trip=False [] | succ=0.647 coll=0.14 | vs s11: net=1 z=0.28 collΔ=-0.06
- u100 闸: trip=False [] | succ=0.64 coll=0.173 | vs s11: net=0 z=0.0 collΔ=-0.0267
- 筛查曲线（sub150 u25→u100；succ/coll 与 vs s11 配对）:
  - u25: succ=0.647 coll=0.2 off=0.127 | vs s11 net=1 z=0.3 collΔ=0.0
  - u50: succ=0.647 coll=0.14 off=0.2 | vs s11 net=1 z=0.28 collΔ=-0.06
  - u75: succ=0.66 coll=0.173 off=0.153 | vs s11 net=3 z=1.0 collΔ=-0.0267
  - u100: succ=0.64 coll=0.173 off=0.187 | vs s11 net=0 z=0.0 collΔ=-0.0267
- keep-best sub150 top（合格线 0.61→collision 最低）: u50 succ=0.647 coll=0.14 eligible=True; u75 succ=0.66 coll=0.173 eligible=True
- clean500 u50: succ=0.654 coll=0.16 off=0.16 sr=0.425 | vs s11 net=-7 z=1.07 collΔ=-0.022 | vs w1 net=64
- clean500 u75: succ=0.666 coll=0.184 off=0.134 sr=0.469 | vs s11 net=-1 z=0.18 collΔ=0.002 | vs w1 net=70
- 采纳: u50（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§12.3））ckpt=`runs/BTC20261005-2243_v7p4extra_s11_colls/ckpt_u050.pt` sha=711e96a711c1d96b758520dfc76f9dfb78e89eed21106e7a3b0894136bfd387e
- **clean500 配对 vs s11（首要）**: Δ=-1.4pp CI(-4.0, 1.2) z=1.07 p=0.36
  - collision 项: Δ=-2.2pp CI(-5.2, 0.8) z=1.46 p=0.185
  - off_road 项: Δ=+2.2pp CI(-0.2, 4.6) z=1.81 p=0.0989
- clean500 配对 vs w1: Δ=+12.8pp CI(8.8, 16.6) z=6.16 p=3.99e-10 ✓显著
- clean500 配对 vs IDM（安全闸参照）: Δ=-8.8pp CI(-13.2, -4.4) z=3.89 p=0.000125
- **eval500**: succ=0.646 coll=0.154 off=0.164 sr=0.428 rc=0.867 sha=b7b50c4c40ea53149b4037bf6a5b06f54e13a908767b1f96207720fb28e72927
  - 配对 vs s11（success, pair_stats）: Δ=+0.0pp z=0.00（ns）
  - **collision 项（paired_eval 口径）**: Δ=-4.8pp CI(-7.8, -1.8) z=3.05 **p=0.00316 ✓显著**（coll_fixed=43 / coll_broken=19）；off_road 项 Δ=+2.4pp z=1.95（p=0.073 ns）；max_step Δ=+2.2pp（p=0.0034）
  - 配对 vs w1 探索性: Δ=+11.6pp z=5.92 | vs IDM: Δ=-11.0pp z=4.88
- tg45: succ=0.0 off=0.867 coll=0.133 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: clean500 collision ≤ 0.174=0.16(✓)；clean500 success ≥ 0.638=0.654(✓)；eval500 collision ≤ 0.144=0.154(✗)；eval500 success ≥ 0.616=0.646(✓)
- 判定: **partial** clean500: succ=0.654 coll=0.16 → 过；eval500: succ=0.646 coll=0.154 → 未过

## 2 seeds 描述性分布（n=2）

- clean500 Δsucc vs s11: n=2 mean=-0.2pp sd=1.7pt min=-1.4 max=+1.0（判据下界 −3pp）
- clean500 Δcoll vs s11: n=2 mean=-2.5pp sd=0.4pt min=-2.8 max=-2.2（判据阈值 = IDM 0.174）
- eval500 Δsucc vs s11: n=2 mean=+0.3pp sd=0.4pt min=+0.0 max=+0.6（判据下界 −3pp）
- eval500 Δcoll vs s11: n=2 mean=-4.3pp sd=0.7pt min=-4.8 max=-3.8（判据阈值 = IDM 0.144）

## 执行事件 / 现场

- **时间线**：launch 21:38:35；seed0 21:38:35→22:43:33（wall 3898.4s；train 468.7s，u50/u100 闸 2 点未触发，sub150×4，clean500×2，tg45，T3，eval500 一次）；seed11 22:43:33→23:50:15（wall 4001.6s；train 509.7s）；driver rc=0；无 early-collapse、无 void、无 stall/timeout。
- **ckpt 加载**：起点 s11 u150（sha256 `a7cc091f…`，preflight 逐字节断言）；stage C 日志 `载入阶段 B 策略快照 runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`，在 pre-v5 worktree（2f4450e）正常载入（与 s11 同代码期；主树 v5 不可加载 v4 ckpt，故隔离跑）。
- **worktree 细节**：`/tmp/opencode/v7_pre_v5` detached @ `2f4450e`；主树 venv 绝对解释器 + `LD_LIBRARY_PATH=<venv>/gl/usr/lib/x86_64-linux-gnu`；`runs`/`datasets` 符号链接主树、`env/specs/*.json` 符号链接主树（worktree .gitignore 覆盖）；臂配置文件复制入 worktree 并以 GIT_CONFIG `core.excludesFile=/tmp/opencode/v7_pre_v5_exclude` 隔离（tracked status 0 行，driver 每步断言 HEAD=2f4450e）；训练/评测期间未改任何代码。
- **主树变更**：仅 `docs/v7_program_prereg.md`（§12 + 变更记录）与 `config/arms/v7_arm1_collision_suppress.yaml`（内容 `eec0abe` / 锚 `ee1ee49`；主树 HEAD 含锚，driver 断言）。
- **训练期警告（既有、非本臂引入）**：`BC 数据集 obs 指纹不一致（数据 v2-6a4d5de3f669 != 当前 v4-…）` 与 `others 为旧布局（28 维）→ 重排 v4` —— 与 s1/s11 旧 run 日志逐条相同（stage C 不消费 BC 数据，仅探针加载；dagger_r1 场景为旧布局）；不影响读数。
- **GPU 串行**：每 train/eval 前 `nvidia-smi ≤3000 MiB 且无 tools/train|test 进程` 守卫通过（实测 413–452 MiB）；全批串行（seed0 → seed11）。
- **独立复算**：§判据表与配对读数已由盘上 `episodes.csv` 逐 `(id,seed)` 复算一致（success/collision 绝对值与 net/z）；eval500 collision 配对另经 `tools/paired_eval.py` 复算（`/tmp/opencode/v7_p4extra_paired/s{0,11}_eval500_vs_s11/`）；ckpt sha256 与 state 一致（u75 `4217abe0…` / u50 `711e96a7…`）。
- 事件：见 status `/tmp/opencode/v7_p4extra_status.txt` 与逐 seed 日志 `/tmp/opencode/v7_p4extra_s{0,11}.log`。

> 判据（§12.3）：collision ≤ IDM 且 success ≥ s11 −3pp（clean500/eval500 双读数）；PASS = 四条全过。配对为描述性；tg45/T3 无命中条款。
