# v7 §14 报告（s11 u150 + ttc 稠密近失罚单变量：TTCLeadPenalty −0.5/2.0/0.5；seeds 0/11）

- 生成：2026-10-06T08:11:42+0800；预注册：`docs/v7_program_prereg.md` §14（content `7bf18d1` / anchor `5db385e`）
- 臂：`config/arms/v7_arm1_ttc.yaml`（sha `8d68b3b31c36…`）= arm1 逐位不变 + 单变量 terms 追加 `ttc`（weight −0.5 / ttc_threshold 2.0 / ttc_floor 0.5；terminal collision 保持 −22）
- 起点：s11 u150 `runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`（sha `a7cc091fcbda…`）；pre-v5 worktree `2f4450e`；训练 pins：u100 / ckpt_every 25（候选 u25–u100）/ rollout 256 / spec dagger_r1 500 / envs 1 / KL 锚 0.05→0.02
- 判据（§14.3）：collision 再降（至少一侧：clean500 ≤0.162 或 eval500 ≤0.144，同时满足 ≤IDM 锚与 相对 s11 降幅 ≥2pp）且 success ≥ s11−3pp（clean ≥0.638 / eval ≥0.616；或 clean500 配对 Δ ≥+3pt）；**每 seed 独立 PASS/partial/fail，2 seeds 均 PASS 才算臂 PASS**
- 参照：s11 clean 0.668/coll 0.182、eval 0.646/coll 0.202；§12 采纳 s0u75 clean 0.678/coll 0.154、eval 0.652/coll 0.164，s11u50 clean 0.654/coll 0.160、eval 0.646/coll 0.154；IDM clean 0.742/coll 0.174、eval 0.756/coll 0.144；w1 0.526/0.530

## 判据表（每 seed；双读数）

| seed | 采纳 | clean succ | clean coll | eval succ | eval coll | collision 侧 | succ 侧 | 总判定 |
|---|---|---|---|---|---|---|---|---|
| 0 | u75 | 0.648 | 0.194 | 0.62 | 0.194 | ✗ | ✗ | **fail** |
| 11 | u25 | 0.672 | 0.152 | 0.648 | 0.156 | ✓ | ✓ | **PASS** |

> 判据阈值：clean coll ≤ 0.162（宽松读法 ≤0.174 仅作副读数）/ succ ≥ 0.638；eval coll ≤ 0.144 / succ ≥ 0.616；success 下界 = s11 −3pp；或 clean500 配对 Δ ≥+3pt。

## seed=0

- train: rc=0 wall=467.3s out=`runs/BTC20261006-0615_v7s14_s0_ttc` status=**terminal:fail**
- 收尾断言: ok=True [] | ttc={'first': 0.0, 'last': 0.0, 'mean_last20': -0.000143} off_road_edge={'first': -0.109162, 'last': -0.125814, 'mean_last20': -0.140006} terminal={'first': -0.289062, 'last': -0.234375, 'mean_last20': -0.13707} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/kl_anchor: first=0.0012 last=0.0064 mean20=0.0078 | u25=0.0082 u50=0.0062 u75=0.0034 u100=0.0064
  - train/reward/ttc: first=0.0 last=0.0 mean20=-0.0001 | u25=0.0 u50=0.0 u75=0.0 u100=0.0
  - train/reward/terminal: first=-0.2891 last=-0.2344 mean20=-0.2031 | u25=0.1133 u50=0.2266 u75=-0.1797 u100=-0.2344
  - train/returns/mean: first=-6.1736 last=-18.2915 mean20=-17.0849 | u25=-8.8024 u50=-1.3613 u75=-11.9096 u100=-18.2915
- u50 闸: trip=False [] | succ=0.62 coll=0.167 | vs s11: net=-3 z=0.77 collΔ=-0.0333
- u100 闸: trip=False [] | succ=0.627 coll=0.173 | vs s11: net=-2 z=0.5 collΔ=-0.0267
- 筛查曲线（sub150 u25→u100；succ/coll 与 vs s11 配对）:
  - u25: succ=0.66 coll=0.187 off=0.147 | vs s11 net=3 z=0.83 collΔ=-0.0133
  - u50: succ=0.62 coll=0.167 off=0.207 | vs s11 net=-3 z=0.77 collΔ=-0.0333
  - u75: succ=0.64 coll=0.167 off=0.187 | vs s11 net=0 z=0.0 collΔ=-0.0333
  - u100: succ=0.627 coll=0.173 off=0.187 | vs s11 net=-2 z=0.5 collΔ=-0.0267
- keep-best sub150 top（合格线 0.61→collision 最低）: u75 succ=0.64 coll=0.167 eligible=True; u50 succ=0.62 coll=0.167 eligible=True
- clean500 u50: succ=0.626 coll=0.192 off=0.168 sr=0.512 | vs s11 net=-21 z=3.28 collΔ=0.01 | vs p4e net=-26 collΔ=0.038 | vs w1 net=50
- clean500 u75: succ=0.648 coll=0.194 off=0.142 sr=0.484 | vs s11 net=-10 z=1.62 collΔ=0.012 | vs p4e net=-15 collΔ=0.04 | vs w1 net=61
- 采纳: u75（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§14.3））ckpt=`runs/BTC20261006-0615_v7s14_s0_ttc/ckpt_u075.pt` sha=0225c2d35f957f3ed960026c59eca21dc25e81dc7b159ed41e7eebebd8e4b9d6
- **clean500 配对 vs s11（首要）**: Δ=-2.0pp CI(-4.4, 0.4) z=1.62 p=0.143
  - collision 项: Δ=+1.2pp CI(-1.6, 4.0) z=0.83 p=0.488
  - off_road 项: Δ=+0.4pp CI(-1.8, 2.8) z=0.34 p=0.864
- clean500 配对 vs §12 采纳 ckpt: Δ=-3.0pp CI(-5.6, -0.6) z=2.34 p=0.0275
  - collision 项: Δ=+4.0pp CI(2.0, 6.2) z=3.65 p=0.000325
- clean500 配对 vs w1: Δ=+12.2pp CI(8.0, 16.4) z=5.50 p=3.32e-08 ✓显著
- clean500 配对 vs IDM（安全闸参照）: Δ=-9.4pp CI(-13.8, -5.0) z=4.24 p=2.72e-05
- **eval500**: succ=0.62 coll=0.194 off=0.162 sr=0.488 rc=0.853 sha=7c180dae59d7aa72442dd63e878c96607590f51d87aa0a8a5fce3975c9474992
  - 配对 vs s11: Δ=-2.6pp z=2.08 p=n/a
  - collision 项: —
  - 配对 vs §12 采纳 ckpt: Δ=-3.2pp z=2.47 p=n/a | vs w1 探索性: Δ=+9.0pp z=4.48 p=n/a | vs IDM: Δ=-13.6pp z=5.96 p=n/a
- tg45: succ=0.0 off=0.778 coll=0.222 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: clean500 collision ≤ 0.162（≤IDM 0.174 且 Δvs s11 ≥2pp）=0.194(✗)；clean500 success ≥ 0.638=0.648(✓)；eval500 collision ≤ 0.144（=IDM 锚；Δvs s11 ≥2pp）=0.194(✗)；eval500 success ≥ 0.616=0.62(✓)；clean500 配对 success Δ ≥ +3pt=-2.0(✗)
- 判定: **fail** collision 再降: clean=0.194(严格✗/宽松✗) eval=0.194 → ✗；success: clean=0.648 eval=0.62 配对Δ=-2.0 → ✗；collision侧=✗ succ侧=✗

## seed=11

- train: rc=0 wall=452.9s out=`runs/BTC20261006-0711_v7s14_s11_ttc` status=**terminal:PASS**
- 收尾断言: ok=True [] | ttc={'first': 0.0, 'last': -0.003182, 'mean_last20': -0.002279} off_road_edge={'first': -0.137628, 'last': -0.080488, 'mean_last20': -0.091789} terminal={'first': -0.179688, 'last': 0.027344, 'mean_last20': -0.115937} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/kl_anchor: first=0.0014 last=0.0138 mean20=0.0137 | u25=0.0065 u50=0.0059 u75=0.0076 u100=0.0138
  - train/reward/ttc: first=0.0 last=-0.0032 mean20=-0.0023 | u25=0.0 u50=-0.0004 u75=0.0 u100=-0.0032
  - train/reward/terminal: first=-0.1797 last=0.0273 mean20=-0.1549 | u25=-0.0664 u50=0.2266 u75=-0.1797 u100=0.0273
  - train/returns/mean: first=-5.9654 last=-8.8755 mean20=-12.378 | u25=-14.4597 u50=1.4561 u75=-9.9364 u100=-8.8755
- u50 闸: trip=False [] | succ=0.66 coll=0.153 | vs s11: net=3 z=0.83 collΔ=-0.0467
- u100 闸: trip=False [] | succ=0.573 coll=0.247 | vs s11: net=-10 z=2.36 collΔ=0.0467
- 筛查曲线（sub150 u25→u100；succ/coll 与 vs s11 配对）:
  - u25: succ=0.68 coll=0.12 off=0.18 | vs s11 net=6 z=1.6 collΔ=-0.08
  - u50: succ=0.66 coll=0.153 off=0.18 | vs s11 net=3 z=0.83 collΔ=-0.0467
  - u75: succ=0.647 coll=0.173 off=0.18 | vs s11 net=1 z=0.38 collΔ=-0.0267
  - u100: succ=0.573 coll=0.247 off=0.187 | vs s11 net=-10 z=2.36 collΔ=0.0467
- keep-best sub150 top（合格线 0.61→collision 最低）: u25 succ=0.68 coll=0.12 eligible=True; u50 succ=0.66 coll=0.153 eligible=True
- clean500 u25: succ=0.672 coll=0.152 off=0.156 sr=0.449 | vs s11 net=2 z=0.33 collΔ=-0.03 | vs p4e net=9 collΔ=-0.008 | vs w1 net=73
- clean500 u50: succ=0.674 coll=0.164 off=0.14 sr=0.476 | vs s11 net=3 z=0.47 collΔ=-0.018 | vs p4e net=10 collΔ=0.004 | vs w1 net=74
- 采纳: u25（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§14.3））ckpt=`runs/BTC20261006-0711_v7s14_s11_ttc/ckpt_u025.pt` sha=229bbc1e37c6e6fbfbfcdcc96fa6426c6fd0f9f86412d78cecd1c2f088e29b7f
- **clean500 配对 vs s11（首要）**: Δ=+0.4pp CI(-2.0, 2.8) z=0.33 p=0.868
  - collision 项: Δ=-3.0pp CI(-5.6, -0.4) z=2.19 p=0.04
  - off_road 项: Δ=+1.8pp CI(-0.4, 4.0) z=1.62 p=0.15
- clean500 配对 vs §12 采纳 ckpt: Δ=+1.8pp CI(-0.6, 4.2) z=1.41 p=0.211
  - collision 项: Δ=-0.8pp CI(-3.2, 1.4) z=0.69 p=0.608
- clean500 配对 vs w1: Δ=+14.6pp CI(10.4, 18.6) z=6.58 p=2.08e-11 ✓显著
- clean500 配对 vs IDM（安全闸参照）: Δ=-7.0pp CI(-11.2, -2.8) z=3.24 p=0.00156
- **eval500**: succ=0.648 coll=0.156 off=0.158 sr=0.446 rc=0.872 sha=9f82cbb8deeed49b556fbbe100ab86bbf7fbfc15e79ab13847975da284221fb6
  - 配对 vs s11: Δ=+0.2pp z=0.15 p=n/a
  - collision 项: —
  - 配对 vs §12 采纳 ckpt: Δ=+0.2pp z=0.14 p=n/a | vs w1 探索性: Δ=+11.8pp z=5.87 p=n/a | vs IDM: Δ=-10.8pp z=4.70 p=n/a
- tg45: succ=0.0 off=0.822 coll=0.178 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: clean500 collision ≤ 0.162（≤IDM 0.174 且 Δvs s11 ≥2pp）=0.152(✓)；clean500 success ≥ 0.638=0.672(✓)；eval500 collision ≤ 0.144（=IDM 锚；Δvs s11 ≥2pp）=0.156(✗)；eval500 success ≥ 0.616=0.648(✓)；clean500 配对 success Δ ≥ +3pt=0.4(✗)
- 判定: **PASS** collision 再降: clean=0.152(严格✓/宽松✓) eval=0.156 → ✓；success: clean=0.672 eval=0.648 配对Δ=+0.4 → ✓；collision侧=✓ succ侧=✓

## 2 seeds 描述性分布（n=2）

- clean500 Δsucc vs s11: n=2 mean=-0.8pp sd=1.7pt min=-2.0 max=+0.4（判据下界 −3pp）
- clean500 Δcoll vs s11: n=2 mean=-0.9pp sd=3.0pt min=-3.0 max=+1.2（严格阈值 ≤0.162 = IDM 0.174 且 Δ≥2pp；宽松 ≤0.174）
- eval500 Δsucc vs s11: n=2 mean=-1.2pp sd=2.0pt min=-2.6 max=+0.2（判据下界 −3pp）
- eval500 Δcoll vs s11: n=2 mean=-2.7pp sd=2.7pt min=-4.6 max=-0.8（判据阈值 = IDM 0.144）

## 执行事件 / 现场

- 执行环境：pre-v5 worktree `/tmp/opencode/v7_pre_v5` @ `2f4450e`（detached）；主树 venv `/workspace/01_Proj/DRL_PathPlan/.venv/bin/python` + `LD_LIBRARY_PATH`；`runs`/`datasets`/`env/specs` 符号链接主树；worktree tracked 白名单 = §13 已同步 3 个 reward_model 文件（sha256 校验）。
- 主树预注册：内容 `7bf18d1`（config + §14）/ 锚 `5db385e`（变更记录）；主树仅 docs/config 变更。
- 事件：见 status `/tmp/opencode/v7_s14_ttc_status.txt` 与逐 seed 日志 `/tmp/opencode/v7_s14_s{0,11}.log`（异常如实追加于本节）。

> 判据（§14.3）：collision 再降（至少一侧：clean ≤0.162 或 eval ≤0.144，同时满足 ≤IDM 锚与相对 s11 降幅 ≥2pp）且 success ≥ s11 −3pp（或 clean500 配对 Δ ≥+3pt）。配对为描述性；tg45/T3 无命中条款。
