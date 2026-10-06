# v7 P3：arm1 n=4 种子分布读数（seeds 0/11/1/2）

- 生成：2026-10-05T21:13:11+0800；来源：`/tmp/opencode/v7_p2_s{seed}.json`；驱动自动合并报告：`/tmp/opencode/v7_p2_arm1.md`
- 臂：`config/arms/v7_arm1_offroad.yaml`（v5 bundle rc=1 + off_road_edge −0.5/scale 1.0；KL 锚 0.05→0.02）；训练 pins：spec dagger_r1 500 / envs 1 / u200 / rollout 256 / u50+u100 双点闸 / keep-best / clean500+tg45+T3+eval500
- 参照：w1 e005 clean500 0.526；P1-B clean500 0.314 / eval500 0.312；IDM clean500 0.742 / eval500 0.756 / tg45 0.778

## clean500 Δ vs w1 分布（首要）

- **n=4 mean=+8.1pp sd=5.6pt min=+1.0 max=+14.2** ⚠方差未达§2.2闸口径
- 逐 seed：s0 +1.0pp (z=0.38, p=0.76)；s11 +14.2pp (z=6.3, p=1.7e-10)；s1 +6.6pp (z=2.84, p=0.0057)；s2 +10.4pp (z=4.6, p=4.9e-06)

## 逐 seed 表

| seed | adopted | clean500 succ | coll | off | Δ vs w1 | tg45 succ | T3 S1 | eval500 succ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | u100 | 0.536 | 0.102 | 0.344 | +1.0pp (z=0.38) | 0.0 | 0/9 | 0.512 | flat |
| 11 | u150 | 0.668 | 0.182 | 0.138 | +14.2pp (z=6.3) | void | void(9/9) | void | strong |
| 1 | u50 | 0.592 | 0.162 | 0.208 | +6.6pp (z=2.84) | 0.0 | 0/9 | 0.55 | positive |
| 2 | u200 | 0.63 | 0.174 | 0.172 | +10.4pp (z=4.6) | 0.044 | 0/9 | 0.622 | strong |

## 闸与 keep-best 记录

- seed 0: u50: trip=False net=-7 offΔ=0.0334 [—]；u100: trip=False net=8 offΔ=-0.06 [—]
  - keep-best sub150 top: u100 net=8 z=1.13；u50 net=-7 z=0.96
  - u200 末段: sub150 succ=0.007 off=0.987 net_vs_w1=-77
- seed 11: u50: trip=True net=-25 offΔ=0.02 [net=-25 < -20]；u100: trip=False net=16 offΔ=-0.18 [—]
  - keep-best sub150 top: u150 net=18 z=2.85；u125 net=17 z=2.53
  - u200 末段: sub150 succ=0.533 off=0.313 net_vs_w1=2
- seed 1: u50: trip=False net=15 offΔ=-0.1733 [—]；u100: trip=False net=-7 offΔ=-0.0333 [—]
  - keep-best sub150 top: u50 net=15 z=2.24；u200 net=12 z=2.0
  - u200 末段: sub150 succ=0.6 off=0.22 net_vs_w1=12
- seed 2: u50: trip=False net=10 offΔ=-0.1533 [—]；u100: trip=True net=-27 offΔ=0.02 [net=-27 < -20]
  - keep-best sub150 top: u200 net=18 z=2.78；u50 net=10 z=1.71
  - u200 末段: sub150 succ=0.64 off=0.193 net_vs_w1=18

## 异常/说明（如实记录）

- seed 0: collision +3.6pp
- seed 11: tg45 void（n_error=n）；eval500 void（n_error=n）；T3 ckpt_error=9/9；collision +11.6pp
- seed 1: collision +9.6pp
- seed 2: collision +10.8pp

