# v7 奖励单变量臂队列报告（§13：A 出界距离罚提前 / B 限速缺口显式罚 / C 窗口化舒适；base = s11 u150）

- 生成：2026-10-06T05:58:24+0800；预注册：`docs/v7_program_prereg.md` §13（内容 `db98501` / 锚 `23ae105`）
- 起点：s11 u150 `runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`（sha `a7cc091fcbda…`）；pre-v5 worktree `2f4450e`；训练 pins：u100 / ckpt_every 25（候选 u25–u100）/ rollout 256 / spec dagger_r1 500 / envs 1 / KL 锚 0.05→0.02
- 判据（§13.3，clean500 主读数、每 seed 独立；eval500 每 seed 采纳 candidate 一次为第二读数）：
  - A：配对 success ≥ +3pt 且 z≥1.96，或 off_road Δ≤−3pt 且 collision Δ≤+3pt；
  - B：speed_ratio ↑≥0.05 且 collision Δ≤+3pt，或 success ≥ s11−3pt；
  - C：jerk_p95 ↓≥20%（≤0.8×s11）且 success ≥ s11−3pt。
- s11 参照：clean succ 0.668/coll 0.182/off 0.138/sr 0.474/jerk_p95 30.07；eval succ 0.646/coll 0.202/sr 0.477/jerk_p95 30.17

## 臂级汇总（2 seeds 描述性）

| 臂 | 单变量 | seed | 采纳 | clean succ | clean coll | clean off | clean sr | clean jerk | eval succ | eval coll | clean 判定 | eval 第二读数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | off_road_edge.edge_scale_m 1.0→2.5（提前 2.5 m 起罚；weight −0.5 不变） | 0 | u25 | 0.646 | 0.158 | 0.192 | 0.4868 | 31.88 | 0.614 | 0.18 | 未过 | 未过 |
| A | off_road_edge.edge_scale_m 1.0→2.5（提前 2.5 m 起罚；weight −0.5 不变） | 11 | u100 | 0.668 | 0.152 | 0.146 | 0.4433 | 31.35 | 0.644 | 0.162 | 未过 | 未过 |
| B | 追加 speed_deficit（weight −0.3；raw=clip(1−v/speed_limit_mps,0,1)） | 0 | u25 | 0.658 | 0.138 | 0.144 | 0.4214 | 28.83 | 0.638 | 0.142 | 过 | 同向过 |
| B | 追加 speed_deficit（weight −0.3；raw=clip(1−v/speed_limit_mps,0,1)） | 11 | u25 | 0.666 | 0.166 | 0.156 | 0.4614 | 31.26 | 0.626 | 0.178 | 过 | 同向过 |
| C | 追加 comfort_jerk_win（weight −0.1/deadband 5.0/window_steps 20） | 0 | u25 | 0.654 | 0.138 | 0.148 | 0.4187 | 29.08 | 0.66 | 0.13 | 未过 | 未过 |

## 结论（§13.3 臂级；每臂 PASS = 2 seeds 均独立满足主判据）

- 臂 A: **fail**（seeds=[0, 11]，verdicts=['fail', 'fail']）
- 臂 B: **PASS**（seeds=[0, 11]，verdicts=['PASS', 'PASS']）
- 臂 C: **fail**（seeds=[0]，verdicts=['fail']）（seed 11 **not-run**：§13.4 预算限制）

## 逐臂逐 seed 明细

### 臂 A — off_road_edge.edge_scale_m 1.0→2.5（提前 2.5 m 起罚；weight −0.5 不变）

#### A seed=0

- train: rc=0 wall=510.0s out=`runs/BTC20261006-0027_v7reward_A_s0` status=**terminal:fail** wall_total=3603.9s
- 收尾断言: ok=True [] | off_road_edge={'first': -0.304918, 'last': -0.315801, 'mean_last20': -0.296226} extra=— terminal={'first': -0.289062, 'last': 0.113281, 'mean_last20': -0.103633} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/reward/terminal: first=-0.2891 last=0.1133 mean20=-0.0432 | u25=-0.1211 u50=-0.1406 u75=0.0273 u100=0.1133
  - train/returns/mean: first=-8.974 last=-16.426 mean20=-21.0791 | u25=-20.493 u50=-14.6818 u75=-20.052 u100=-16.426
- u50 闸: trip=False [] | succ=0.627 coll=0.18 | vs s11: net=-2 z=0.58 collΔ=-0.02
- u100 闸: trip=False [] | succ=0.673 coll=0.18 | vs s11: net=5 z=1.39 collΔ=-0.02
- 筛查曲线（sub150 u25→u100）:
  - u25: succ=0.627 coll=0.147 off=0.227 sr=0.4985 jerk=32.0 | vs s11 net=-2 z=0.71 collΔ=-0.0533 offΔ=0.08
  - u50: succ=0.627 coll=0.18 off=0.193 sr=0.5512 jerk=32.84 | vs s11 net=-2 z=0.58 collΔ=-0.02 offΔ=0.0466
  - u75: succ=0.593 coll=0.207 off=0.207 sr=0.5643 jerk=34.49 | vs s11 net=-7 z=1.94 collΔ=0.0067 offΔ=0.06
  - u100: succ=0.673 coll=0.18 off=0.14 sr=0.5058 jerk=34.01 | vs s11 net=5 z=1.39 collΔ=-0.02 offΔ=-0.0067
- keep-best sub150 top（合格线 0.61→collision 最低）: u25 succ=0.627 coll=0.147 eligible=True; u100 succ=0.673 coll=0.18 eligible=True
- clean500 u25: succ=0.646 coll=0.158 off=0.192 sr=0.4868 jerk=31.88 | vs s11 net=-11 z=2.04 collΔ=-0.024 offΔ=0.054
- clean500 u100: succ=0.674 coll=0.196 off=0.124 sr=0.5012 jerk=34.13 | vs s11 net=3 z=0.46 collΔ=0.014 offΔ=-0.014
- 采纳: u25（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§13.3 同 §12））ckpt=`runs/BTC20261006-0027_v7reward_A_s0/ckpt_u025.pt` sha=0806c224107d571dd3875e7d8018b2ada7421b5c74c2473301b1b303e7faf489
- **clean500 配对 vs s11（主判据）**: Δ=-2.2pp CI(-4.4, -0.2) z=2.04 p=0.0614
  - off_road 项: Δ=+5.4pp CI(2.8, 8.0) z=4.02 p=6.57e-05
  - collision 项: Δ=-2.4pp CI(-5.2, 0.4) z=1.70 p=0.119
- clean500 配对 vs w1: Δ=+12.0pp CI(7.6, 16.2) z=5.22 p=1.75e-07 ✓显著 | vs IDM: Δ=-9.6pp CI(-13.8, -5.4) z=4.46 p=9.69e-06
- **eval500**: succ=0.614 coll=0.18 off=0.186 sr=0.4878 jerk=31.9 rc=0.853 sha=404abbc713a10c74ca8491a2c4c8880e4bca379c22fdf4d921981641f05cf7bf
  - 配对 vs s11: Δ=-3.2pp z=2.41 p=n/a | vs w1 探索性: Δ=+8.4pp z=4.20 p=n/a | vs IDM: Δ=-14.2pp z=6.02 p=n/a
- tg45: succ=0.0 off=0.644 coll=0.356 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: 路径① 配对 success ≥ +3pt 且 z≥1.96=-2.2(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=5.4(✗)
- eval500 第二读数: 路径① 配对 success ≥ +3pt 且 z≥1.96=-3.2(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=4.6(✗)
- 判定: **fail** clean500: 路径① 配对 success ≥ +3pt 且 z≥1.96=-2.2(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=5.4(✗) → 未过；eval500（第二读数）: 路径① 配对 success ≥ +3pt 且 z≥1.96=-3.2(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=4.6(✗) → 未过

#### A seed=11

- train: rc=0 wall=562.4s out=`runs/BTC20261006-0127_v7reward_A_s11` status=**terminal:fail** wall_total=3968.4s
- 收尾断言: ok=True [] | off_road_edge={'first': -0.326184, 'last': -0.292938, 'mean_last20': -0.313411} extra=— terminal={'first': -0.179688, 'last': 0.113281, 'mean_last20': -0.230586} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/reward/terminal: first=-0.1797 last=0.1133 mean20=-0.1232 | u25=-0.2344 u50=-0.1797 u75=0.1719 u100=0.1133
  - train/returns/mean: first=-8.7791 last=-17.0703 mean20=-20.9785 | u25=-26.9916 u50=-25.0479 u75=-19.0569 u100=-17.0703
- u50 闸: trip=False [] | succ=0.68 coll=0.153 | vs s11: net=6 z=1.73 collΔ=-0.0467
- u100 闸: trip=False [] | succ=0.66 coll=0.133 | vs s11: net=3 z=0.77 collΔ=-0.0667
- 筛查曲线（sub150 u25→u100）:
  - u25: succ=0.593 coll=0.167 off=0.24 sr=0.5027 jerk=31.42 | vs s11 net=-7 z=1.81 collΔ=-0.0333 offΔ=0.0933
  - u50: succ=0.68 coll=0.153 off=0.167 sr=0.481 jerk=33.72 | vs s11 net=6 z=1.73 collΔ=-0.0467 offΔ=0.02
  - u75: succ=0.6 coll=0.147 off=0.167 sr=0.4201 jerk=29.96 | vs s11 net=-6 z=1.34 collΔ=-0.0533 offΔ=0.02
  - u100: succ=0.66 coll=0.133 off=0.173 sr=0.4519 jerk=31.39 | vs s11 net=3 z=0.77 collΔ=-0.0667 offΔ=0.0266
- keep-best sub150 top（合格线 0.61→collision 最低）: u100 succ=0.66 coll=0.133 eligible=True; u50 succ=0.68 coll=0.153 eligible=True
- clean500 u50: succ=0.666 coll=0.186 off=0.14 sr=0.4806 jerk=33.79 | vs s11 net=-1 z=0.17 collΔ=0.004 offΔ=0.002
- clean500 u100: succ=0.668 coll=0.152 off=0.146 sr=0.4433 jerk=31.35 | vs s11 net=0 z=0.0 collΔ=-0.03 offΔ=0.008
- 采纳: u100（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§13.3 同 §12））ckpt=`runs/BTC20261006-0127_v7reward_A_s11/ckpt_u100.pt` sha=b5b63e784358a0d9574f8e3a59089c0ba047a70910933764bdb80b2655225ee7
- **clean500 配对 vs s11（主判据）**: Δ=+0.0pp CI(-2.6, 2.6) z=0.00 p=1
  - off_road 项: Δ=+0.8pp CI(-1.6, 3.2) z=0.67 p=0.618
  - collision 项: Δ=-3.0pp CI(-5.8, -0.2) z=2.06 p=0.0534
- clean500 配对 vs w1: Δ=+14.2pp CI(10.2, 18.2) z=6.68 p=8.69e-12 ✓显著 | vs IDM: Δ=-7.4pp CI(-11.6, -3.2) z=3.45 p=0.000717
- **eval500**: succ=0.644 coll=0.162 off=0.152 sr=0.445 jerk=31.31 rc=0.862 sha=0c7a78819f6f51fc7e0a5b2c983177f0623a2c0ccca1eac9596f5826cdbc7a1e
  - 配对 vs s11: Δ=-0.2pp z=0.14 p=n/a | vs w1 探索性: Δ=+11.4pp z=5.73 p=n/a | vs IDM: Δ=-11.2pp z=5.11 p=n/a
- tg45: succ=0.0 off=0.911 coll=0.089 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: 路径① 配对 success ≥ +3pt 且 z≥1.96=0.0(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=0.8(✗)
- eval500 第二读数: 路径① 配对 success ≥ +3pt 且 z≥1.96=-0.2(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=1.2(✗)
- 判定: **fail** clean500: 路径① 配对 success ≥ +3pt 且 z≥1.96=0.0(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=0.8(✗) → 未过；eval500（第二读数）: 路径① 配对 success ≥ +3pt 且 z≥1.96=-0.2(✗)；路径② off_road Δ≤−3pt 且 collision Δ≤+3pt=1.2(✗) → 未过

**臂 A 阶段小结：fail**（seeds=[0, 11]，verdicts=['fail', 'fail']）

### 臂 B — 追加 speed_deficit（weight −0.3；raw=clip(1−v/speed_limit_mps,0,1)）

#### B seed=0

- train: rc=0 wall=621.0s out=`runs/BTC20261006-0233_v7reward_B_s0` status=**terminal:PASS** wall_total=4128.5s
- 收尾断言: ok=True [] | off_road_edge={'first': -0.109162, 'last': -0.162433, 'mean_last20': -0.135104} extra={'first': -0.199186, 'last': -0.173427, 'mean_last20': -0.171141} terminal={'first': -0.289062, 'last': -0.066406, 'mean_last20': -0.229453} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/reward/terminal: first=-0.2891 last=-0.0664 mean20=-0.0023 | u25=-0.1797 u50=-0.4375 u75=-0.1953 u100=-0.0664
  - train/returns/mean: first=-9.1071 last=-9.5071 mean20=-15.306 | u25=-27.1033 u50=-22.8316 u75=-17.9344 u100=-9.5071
- u50 闸: trip=False [] | succ=0.587 coll=0.153 | vs s11: net=-8 z=2.0 collΔ=-0.0467
- u100 闸: trip=False [] | succ=0.627 coll=0.14 | vs s11: net=-2 z=0.47 collΔ=-0.06
- 筛查曲线（sub150 u25→u100）:
  - u25: succ=0.64 coll=0.12 off=0.187 sr=0.4292 jerk=28.87 | vs s11 net=0 z=0.0 collΔ=-0.08 offΔ=0.04
  - u50: succ=0.587 coll=0.153 off=0.247 sr=0.4749 jerk=31.23 | vs s11 net=-8 z=2.0 collΔ=-0.0467 offΔ=0.1
  - u75: succ=0.64 coll=0.153 off=0.187 sr=0.4327 jerk=31.06 | vs s11 net=0 z=0.0 collΔ=-0.0467 offΔ=0.04
  - u100: succ=0.627 coll=0.14 off=0.227 sr=0.5005 jerk=32.75 | vs s11 net=-2 z=0.47 collΔ=-0.06 offΔ=0.08
- keep-best sub150 top（合格线 0.61→collision 最低）: u25 succ=0.64 coll=0.12 eligible=True; u100 succ=0.627 coll=0.14 eligible=True
- clean500 u25: succ=0.658 coll=0.138 off=0.144 sr=0.4214 jerk=28.83 | vs s11 net=-5 z=0.8 collΔ=-0.044 offΔ=0.006
- clean500 u100: succ=0.636 coll=0.16 off=0.194 sr=0.4916 jerk=32.65 | vs s11 net=-16 z=2.31 collΔ=-0.022 offΔ=0.056
- 采纳: u25（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§13.3 同 §12））ckpt=`runs/BTC20261006-0233_v7reward_B_s0/ckpt_u025.pt` sha=f795a3d314f11a3d726ab9bdc909a0fc04f48ac288e5e2b49a234e5b5e0e21df
- **clean500 配对 vs s11（主判据）**: Δ=-1.0pp CI(-3.4, 1.4) z=0.80 p=0.522
  - off_road 项: Δ=+0.6pp CI(-1.8, 3.0) z=0.51 p=0.736
  - collision 项: Δ=-4.4pp CI(-7.0, -1.8) z=3.39 p=0.000941
- clean500 配对 vs w1: Δ=+13.2pp CI(9.2, 17.2) z=6.24 p=2.38e-10 ✓显著 | vs IDM: Δ=-8.4pp CI(-13.0, -4.0) z=3.68 p=0.00029
- **eval500**: succ=0.638 coll=0.142 off=0.154 sr=0.4193 jerk=28.78 rc=0.873 sha=3c59bc08010fe5de25490b873318ab3c3bc3b5245c4b45608d9b86d711ad8c40
  - 配对 vs s11: Δ=-0.8pp z=0.53 p=n/a | vs w1 探索性: Δ=+10.8pp z=5.69 p=n/a | vs IDM: Δ=-11.8pp z=5.19 p=n/a
- tg45: succ=0.0 off=0.933 coll=0.067 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.0526(✗)；路径② success ≥ s11 −3pt=0.658(✓)
- eval500 第二读数: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.0582(✗)；路径② success ≥ s11 −3pt=0.638(✓)
- 判定: **PASS** clean500: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.0526(✗)；路径② success ≥ s11 −3pt=0.658(✓) → 过；eval500（第二读数）: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.0582(✗)；路径② success ≥ s11 −3pt=0.638(✓) → 同向过

#### B seed=11

- train: rc=0 wall=594.5s out=`runs/BTC20261006-0342_v7reward_B_s11` status=**terminal:PASS** wall_total=3774.7s
- 收尾断言: ok=True [] | off_road_edge={'first': -0.137628, 'last': -0.079399, 'mean_last20': -0.099206} extra={'first': -0.200953, 'last': -0.176534, 'mean_last20': -0.189091} terminal={'first': -0.179688, 'last': -0.109375, 'mean_last20': -0.202227} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/reward/terminal: first=-0.1797 last=-0.1094 mean20=-0.1023 | u25=-0.2891 u50=-0.1641 u75=-0.0664 u100=-0.1094
  - train/returns/mean: first=-8.9508 last=-19.5836 mean20=-13.9812 | u25=-28.2458 u50=-21.3683 u75=-10.1786 u100=-19.5836
- u50 闸: trip=True ['succ net=-43 < -10'] | succ=0.353 coll=0.073 | vs s11: net=-43 z=6.14 collΔ=-0.1267
- u100 闸: trip=False [] | succ=0.667 coll=0.18 | vs s11: net=4 z=1.07 collΔ=-0.02
- 筛查曲线（sub150 u25→u100）:
  - u25: succ=0.653 coll=0.173 off=0.173 sr=0.4685 jerk=31.32 | vs s11 net=2 z=0.53 collΔ=-0.0267 offΔ=0.0266
  - u50: succ=0.353 coll=0.073 off=0.553 sr=0.5231 jerk=31.3 | vs s11 net=-43 z=6.14 collΔ=-0.1267 offΔ=0.4066
  - u75: succ=0.627 coll=0.193 off=0.173 sr=0.5105 jerk=32.31 | vs s11 net=-2 z=0.5 collΔ=-0.0067 offΔ=0.0266
  - u100: succ=0.667 coll=0.18 off=0.147 sr=0.5213 jerk=33.51 | vs s11 net=4 z=1.07 collΔ=-0.02 offΔ=0.0
- keep-best sub150 top（合格线 0.61→collision 最低）: u25 succ=0.653 coll=0.173 eligible=True; u100 succ=0.667 coll=0.18 eligible=True
- clean500 u25: succ=0.666 coll=0.166 off=0.156 sr=0.4614 jerk=31.26 | vs s11 net=-1 z=0.18 collΔ=-0.016 offΔ=0.018
- clean500 u100: succ=0.664 coll=0.184 off=0.144 sr=0.5109 jerk=33.48 | vs s11 net=-2 z=0.34 collΔ=0.002 offΔ=0.006
- 采纳: u25（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§13.3 同 §12））ckpt=`runs/BTC20261006-0342_v7reward_B_s11/ckpt_u025.pt` sha=43c896af34cc4553cab000e5f04f44aebf79d40c696a0eebcc31b0234ad8338f
- **clean500 配对 vs s11（主判据）**: Δ=-0.2pp CI(-2.4, 2.0) z=0.18 p=1
  - off_road 项: Δ=+1.8pp CI(-0.4, 4.0) z=1.67 p=0.136
  - collision 项: Δ=-1.6pp CI(-4.0, 0.8) z=1.26 p=0.268
- clean500 配对 vs w1: Δ=+14.0pp CI(9.8, 18.0) z=6.34 p=1.26e-10 ✓显著 | vs IDM: Δ=-7.6pp CI(-11.8, -3.4) z=3.50 p=0.000598
- **eval500**: succ=0.626 coll=0.178 off=0.166 sr=0.4604 jerk=31.16 rc=0.857 sha=959abe15b21af6bf59327f7f7693edcc642e207373fb7d0d227278d03decc506
  - 配对 vs s11: Δ=-2.0pp z=1.54 p=n/a | vs w1 探索性: Δ=+9.6pp z=4.46 p=n/a | vs IDM: Δ=-13.0pp z=5.55 p=n/a
- tg45: succ=0.0 off=0.578 coll=0.422 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.0126(✗)；路径② success ≥ s11 −3pt=0.666(✓)
- eval500 第二读数: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.017(✗)；路径② success ≥ s11 −3pt=0.626(✓)
- 判定: **PASS** clean500: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.0126(✗)；路径② success ≥ s11 −3pt=0.666(✓) → 过；eval500（第二读数）: 路径① speed_ratio ↑≥0.05 且 collision Δ≤+3pt=-0.017(✗)；路径② success ≥ s11 −3pt=0.626(✓) → 同向过

**臂 B 阶段小结：PASS**（seeds=[0, 11]，verdicts=['PASS', 'PASS']）

### 臂 C — 追加 comfort_jerk_win（weight −0.1/deadband 5.0/window_steps 20）

#### C seed=0

- train: rc=0 wall=512.6s out=`runs/BTC20261006-0445_v7reward_C_s0` status=**terminal:fail** wall_total=4084.1s
- 收尾断言: ok=True [] | off_road_edge={'first': -0.109162, 'last': -0.150885, 'mean_last20': -0.111089} extra={'first': -0.001458, 'last': -1.2e-05, 'mean_last20': -0.000457} terminal={'first': -0.289062, 'last': -0.234375, 'mean_last20': -0.139258} kl={'n': 100, 'first': 0.05, 'last': 0.02}
  - train/kl_anchor_coef: first=0.05 last=0.02 mean20=0.0229 | u25=0.0427 u50=0.0352 u75=0.0276 u100=0.02
  - train/reward/terminal: first=-0.2891 last=-0.2344 mean20=-0.1684 | u25=-0.2344 u50=0.0586 u75=-0.1797 u100=-0.2344
  - train/returns/mean: first=-6.1784 last=-15.7394 mean20=-16.0567 | u25=-19.6027 u50=-2.2482 u75=-10.3487 u100=-15.7394
- u50 闸: trip=False [] | succ=0.667 coll=0.147 | vs s11: net=4 z=1.0 collΔ=-0.0533
- u100 闸: trip=False [] | succ=0.633 coll=0.187 | vs s11: net=-1 z=0.3 collΔ=-0.0133
- 筛查曲线（sub150 u25→u100）:
  - u25: succ=0.64 coll=0.133 off=0.167 sr=0.4266 jerk=29.19 | vs s11 net=0 z=0.0 collΔ=-0.0667 offΔ=0.02
  - u50: succ=0.667 coll=0.147 off=0.187 sr=0.496 jerk=33.56 | vs s11 net=4 z=1.0 collΔ=-0.0533 offΔ=0.04
  - u75: succ=0.687 coll=0.12 off=0.18 sr=0.477 jerk=33.11 | vs s11 net=7 z=1.81 collΔ=-0.08 offΔ=0.0333
  - u100: succ=0.633 coll=0.187 off=0.16 sr=0.4687 jerk=33.37 | vs s11 net=-1 z=0.3 collΔ=-0.0133 offΔ=0.0133
- keep-best sub150 top（合格线 0.61→collision 最低）: u75 succ=0.687 coll=0.12 eligible=True; u25 succ=0.64 coll=0.133 eligible=True
- clean500 u25: succ=0.654 coll=0.138 off=0.148 sr=0.4187 jerk=29.08 | vs s11 net=-7 z=0.93 collΔ=-0.044 offΔ=0.01
- clean500 u75: succ=0.688 coll=0.14 off=0.156 sr=0.4696 jerk=33.04 | vs s11 net=10 z=1.71 collΔ=-0.042 offΔ=0.018
- 采纳: u25（规则：全量 clean500 success ≥ 0.638 者中 collision 最低（§13.3 同 §12））ckpt=`runs/BTC20261006-0445_v7reward_C_s0/ckpt_u025.pt` sha=b455763004925b20cf3b5c9e72a140f053e6b975c39e51844c2aeb6fac5f0f54
- **clean500 配对 vs s11（主判据）**: Δ=-1.4pp CI(-4.4, 1.6) z=0.93 p=0.427
  - off_road 项: Δ=+1.0pp CI(-1.4, 3.4) z=0.85 p=0.5
  - collision 项: Δ=-4.4pp CI(-7.2, -1.6) z=3.05 p=0.00319
- clean500 配对 vs w1: Δ=+12.8pp CI(9.0, 16.4) z=6.46 p=3.62e-11 ✓显著 | vs IDM: Δ=-8.8pp CI(-13.2, -4.6) z=3.86 p=0.000142
- **eval500**: succ=0.66 coll=0.13 off=0.148 sr=0.4136 jerk=29.03 rc=0.878 sha=4aea19a58ac569da5e1b232d8cc5e8b065ca0511860f56114a5b0035ab0845ac
  - 配对 vs s11: Δ=+1.4pp z=0.93 p=n/a | vs w1 探索性: Δ=+13.0pp z=6.81 p=n/a | vs IDM: Δ=-9.6pp z=4.31 p=n/a
- tg45: succ=0.0 off=0.911 coll=0.111 | vs IDM: Δ=-77.8pp z=5.92 p=n/a
- T3 S1: 0/9（辅助无命中条款）
- 判据条款: jerk_p95 ↓≥20%=29.084(✗)；success ≥ s11 −3pt=0.654(✓)
- eval500 第二读数: jerk_p95 ↓≥20%=29.0349(✗)；success ≥ s11 −3pt=0.66(✓)
- 判定: **fail** clean500: jerk_p95 ↓≥20%=29.084(✗)；success ≥ s11 −3pt=0.654(✓) → 未过；eval500（第二读数）: jerk_p95 ↓≥20%=29.0349(✗)；success ≥ s11 −3pt=0.66(✓) → 未过

**臂 C 阶段小结：fail**（seeds=[0]，verdicts=['fail']）

## 2 seeds 描述性分布（n=2；不设方差闸结论）

- 臂 A: s0 Δsucc(clean)=-2.2pp Δcoll(clean)=-2.4pp Δsucc(eval)=-3.2pp; s11 Δsucc(clean)=+0.0pp Δcoll(clean)=-3.0pp Δsucc(eval)=-0.2pp
- 臂 B: s0 Δsucc(clean)=-1.0pp Δcoll(clean)=-4.4pp Δsucc(eval)=-0.8pp; s11 Δsucc(clean)=-0.2pp Δcoll(clean)=-1.6pp Δsucc(eval)=-2.0pp
- 臂 C: s0 Δsucc(clean)=-1.4pp Δcoll(clean)=-4.4pp Δsucc(eval)=+1.4pp

## 执行事件 / 现场

- 执行环境：pre-v5 worktree `/tmp/opencode/v7_pre_v5` @ `2f4450e`（detached）；主树 venv `/workspace/01_Proj/DRL_PathPlan/.venv/bin/python`；`runs`/`datasets`/`env/specs` 符号链接主树；同步白名单 ['config/arms/v7_reward_A.yaml', 'config/arms/v7_reward_B.yaml', 'config/arms/v7_reward_C.yaml', 'reward_model/aggregation.py', 'reward_model/registry.py', 'reward_model/terms.py']（sha256 校验；worktree tracked 改动 = 恰 3 个 reward_model 文件）。
- 主树预注册：内容 `db98501` / 锚 `23ae105`；主树仅 reward_model/config/docs/tests 变更。
- GPU 排队（勿抢）：等 `v7_p4extra_done`（23:50）+ `v7_q6q2_diag_done`（23:52）两标志且 lane 无 train/test 进程后于 00:13 同步/预检、00:27 开跑；排队期间未抢跑。
- 00:13 首次 `--check` 失败（driver bug：git status 首行前导空格被整体 strip 吃掉 → 路径解析少一字符）；修复解析后 00:13:52 check OK（不涉实验代码/配置）。
- 00:14 smoke B 首跑失败（**环境问题，非代码**：chain 未导出 `LD_LIBRARY_PATH` → panda3d `getPipeTypes()` 空列表 IndexError，MetaDrive 引擎初始化失败）；补 `LD_LIBRARY_PATH=<venv>/gl/usr/lib/x86_64-linux-gnu` 后 00:26 smoke B / 00:27 smoke C （u2）通过；新项在训练循环内非零（speed_deficit mean20 −0.208 / comfort_jerk_win mean20 −0.00073；断言 ok）。
- batch 实跑 **326.0 min**（A s0/s11 60/66min、B s0/s11 63/66min、C s0 68min）；C seed 11 按 §13.4 预算规则跳过（remaining 34min < 1.5×66min 实测单 seed 墙钟），标 **not-run**（如实记录，不追补）。
- 逐 run `episodes.csv` sha256 / metrics / ckpt sha 见各 seed 明细；事件日志 `/tmp/opencode/v7_reward_arms_chain.log` / `v7_reward_{a,b,c}_s{0,11}.log`。

