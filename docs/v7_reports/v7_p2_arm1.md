# v7 P2 首臂报告（arm1：w1 e005 + off_road_edge + KL 锚；seeds 0/11）

- 生成：2026-10-05T17:26:02+0800；预注册：`docs/v7_program_prereg.md` §9（content `2ddfa22` / anchor `c37acbb`）
- 臂：`config/arms/v7_arm1_offroad.yaml` = v5 bundle(rc=1) + `off_road_edge`(−0.5 / scale 1.0)；KL 锚 0.05→0.02；训练 pins 同 v6 P4 arm0（spec dagger_r1 500 / envs 1 / u200 / rollout 256）
- base/参照：w1 e005 clean500 0.526（期望锚 0.47–0.51）；P1-B e010 clean500 0.314 / eval500 0.312；IDM clean500 0.742 / eval500 0.756 / tg45 0.778；w1 eval500 exploratory 0.530（仅报告）

## seed=0

- train: rc=0 wall=936.2s out=`runs/BTC20261003-0641_v7p2_s0_arm1` verdict=**terminal:flat**
- 收尾断言: ok=True [] | off_road_edge={'first': -0.139387, 'last': -0.188262, 'mean_last20': -0.118567} kl={'n': 200, 'first': 0.05, 'last': 0.02}
- 训练读数（monitor/metrics.csv）: kl_anchor_coef first=0.05 last=0.02
  - train/kl_anchor: first=0.0064 last=0.0265 mean20=0.0223 | u25=0.0136 u50=0.0112 u75=0.019 u100=0.0304 u125=0.0226 u150=0.0157 u175=0.0238 u200=0.0265
  - train/reward/off_road_edge: first=-0.1394 last=-0.1883 mean20=-0.1186 | u25=-0.1499 u50=-0.2149 u75=-0.1163 u100=-0.1501 u125=-0.0963 u150=-0.0818 u175=-0.0378 u200=-0.1883
  - train/reward/terminal: first=-0.6562 last=-0.1055 mean20=-0.0863 | u25=-0.2344 u50=-0.2344 u75=-0.0547 u100=-0.2344 u125=-0.0547 u150=0.1133 u175=0.2266 u200=-0.1055
  - train/returns/mean: first=-10.8057 last=-12.5531 mean20=-10.6125 | u25=-11.2495 u50=-19.1255 u75=-17.395 u100=-15.8385 u125=-16.5666 u150=-15.4115 u175=-8.4756 u200=-12.5531
- u50 闸: trip=False [] | net=-7 z=0.96 offΔ=0.0334
- u100 闸: trip=False [] | net=8 z=1.13 offΔ=-0.06
- 筛查曲线（sub150 u25→u200；succ/off/coll 与 vs w1 配对）:
  - u25: succ=0.347 off=0.573 coll=0.04 | vs w1 net=-26 z=3.47 offΔ=0.18
  - u50: succ=0.473 off=0.427 coll=0.067 | vs w1 net=-7 z=0.96 offΔ=0.0334
  - u75: succ=0.4 off=0.453 coll=0.073 | vs w1 net=-18 z=2.45 offΔ=0.06
  - u100: succ=0.573 off=0.333 coll=0.06 | vs w1 net=8 z=1.13 offΔ=-0.06
  - u125: succ=0.367 off=0.413 coll=0.053 | vs w1 net=-23 z=3.51 offΔ=0.02
  - u150: succ=0.447 off=0.42 coll=0.1 | vs w1 net=-11 z=1.54 offΔ=0.0267
  - u175: succ=0.447 off=0.467 coll=0.067 | vs w1 net=-11 z=1.43 offΔ=0.0734
  - u200: succ=0.007 off=0.987 coll=0.007 | vs w1 net=-77 z=8.66 offΔ=0.5934
- keep-best sub150 top: u100 net=8 z=1.13; u50 net=-7 z=0.96
- clean500 u50: succ=0.452 off=0.446 coll=0.072 | vs w1 net=-37 z=2.8 | vs p1b net=69
- clean500 u100: succ=0.536 off=0.344 coll=0.102 | vs w1 net=5 z=0.38 | vs p1b net=111
- 采纳: u100 ckpt=`runs/BTC20261003-0641_v7p2_s0_arm1/ckpt_u100.pt` sha=aa4b657a328797c48552af1f1cc1da44bb070762166adcdbff823861d85a5227
- **clean500 配对 vs w1（首要）**: Δ=+1.0pp CI(-4.2, 6.0) z=0.38 p=0.76
- clean500 配对 vs P1-B: Δ=+22.2pp CI(18.0, 26.4) z=9.48 p=6.84e-24 ✓显著
- clean500 配对 vs IDM（安全闸参照）: Δ=-20.6pp CI(-25.6, -15.6) z=7.61 p=8.66e-15
- **eval500 配对 vs P1-B（一次）**: Δ=+20.0pp z=9.05 p=n/a
  - eval500: succ=0.512 off=0.374 coll=0.082 sha=17617b83ea3495f30e0c8d74ba6eec0036aa7d0365b1c8bb5a1fac230721a126
  - eval500 vs w1 探索性: Δ=-1.8pp z=0.70 p=n/a（w1 首测探索 0.53 参照）
  - eval500 vs IDM（参照）: Δ=-24.4pp z=9.09 p=n/a（IDM 0.756 参照）
- tg45: succ=0.0 off=0.911 coll=0.089 | vs IDM: Δ=-77.8pp z=5.92 p=n/a（IDM 0.778 参照）
- T3 S1: 0/9 （辅助无命中条款）
- 判读: **flat** Δ=+1.0pp z=0.38（未达 +3pt/显著）
- 异常/副作用: u200 末段坍缩：sub150 succ=0.007 off=0.987（vs w1 net=-77）；collision +3.6pp (p=0.0153，显著)；off_road -5.4pp (p=0.057，未达显著；越低越好=改善)

## seed=1

- train: rc=1 wall=1.9s out=`runs/BTC20261005-1724_v7p2_s1_arm1` verdict=**train-failed**
- 判读: **None** None

## seed=11

- train: rc=0 wall=1083.2s out=`runs/BTC20261005-0601_v7p2_s11_arm1` verdict=**terminal:strong**
- 收尾断言: ok=True [] | off_road_edge={'first': -0.187949, 'last': -0.113731, 'mean_last20': -0.112835} kl={'n': 200, 'first': 0.05, 'last': 0.02}
- 训练读数（monitor/metrics.csv）: kl_anchor_coef first=0.05 last=0.02
  - train/kl_anchor: first=0.001 last=0.0145 mean20=0.0138 | u25=0.0164 u50=0.0204 u75=0.0312 u100=0.0136 u125=0.0127 u150=0.0256 u175=0.0112 u200=0.0145
  - train/reward/off_road_edge: first=-0.1879 last=-0.1137 mean20=-0.1128 | u25=-0.121 u50=-0.1004 u75=-0.1433 u100=-0.1253 u125=-0.1521 u150=-0.1275 u175=-0.0735 u200=-0.1137
  - train/reward/terminal: first=-0.3828 last=-0.1797 mean20=-0.2211 | u25=-0.1797 u50=-0.2344 u75=-0.1094 u100=-0.3594 u125=-0.4375 u150=-0.2734 u175=-0.1797 u200=-0.1797
  - train/returns/mean: first=-10.2097 last=-20.0473 mean20=-19.2875 | u25=-11.1656 u50=-18.568 u75=-17.2088 u100=-18.9748 u125=-23.6083 u150=-7.3467 u175=-14.3414 u200=-20.0473
- u50 闸: trip=True ['net=-25 < -20'] | net=-25 z=4.23 offΔ=0.02
- u100 闸: trip=False [] | net=16 z=2.67 offΔ=-0.18
- 筛查曲线（sub150 u25→u200；succ/off/coll 与 vs w1 配对）:
  - u25: succ=0.44 off=0.453 coll=0.1 | vs w1 net=-12 z=1.55 offΔ=0.06
  - u50: succ=0.353 off=0.413 coll=0.033 | vs w1 net=-25 z=4.23 offΔ=0.02
  - u75: succ=0.573 off=0.24 coll=0.153 | vs w1 net=8 z=1.21 offΔ=-0.1533
  - u100: succ=0.627 off=0.213 coll=0.087 | vs w1 net=16 z=2.67 offΔ=-0.18
  - u125: succ=0.633 off=0.28 coll=0.093 | vs w1 net=17 z=2.53 offΔ=-0.1133
  - u150: succ=0.64 off=0.147 coll=0.2 | vs w1 net=18 z=2.85 offΔ=-0.2466
  - u175: succ=0.527 off=0.307 coll=0.113 | vs w1 net=1 z=0.15 offΔ=-0.0866
  - u200: succ=0.533 off=0.313 coll=0.107 | vs w1 net=2 z=0.28 offΔ=-0.08
- keep-best sub150 top: u150 net=18 z=2.85; u125 net=17 z=2.53
- clean500 u125: **void（n_error=500/500，全错误 episode；非真实读数）**
- clean500 u150: succ=0.668 off=0.138 coll=0.182 | vs w1 net=71 z=6.3 | vs p1b net=177
- 采纳: u150 ckpt=`runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt` sha=a7cc091fcbda670b25c396a43e18dc39abe089053e50aa292b5fc0f19164ba2e
- **clean500 配对 vs w1（首要）**: Δ=+14.2pp CI(10.0, 18.4) z=6.30 p=1.72e-10 ✓显著
- clean500 配对 vs P1-B: Δ=+35.4pp CI(31.0, 39.8) z=12.74 p=6.86e-45 ✓显著
- clean500 配对 vs IDM（安全闸参照）: Δ=-7.4pp CI(-11.8, -3.0) z=3.31 p=0.00119
- **eval500: void（n_error=500/500，全错误 episode；非真实读数，判读/分布不采用）**
- tg45: **void（n_error=45/45，非真实读数）**
- T3 S1: 0/9 （辅助无命中条款） | ckpt_error=9/9（void）
- 判读: **strong** Δ=+14.2pp z=6.3（≥+8pt 且显著：期望带内）
- 异常/副作用: collision +11.6pp (p=4.11e-11，显著)；off_road -26.0pp (p=2.29e-29，显著；越低越好=改善)

## 2 seeds 分布

- clean500 Δ vs w1（首要）: n=2 mean=+7.6pp sd=9.3pt min=+1.0 max=+14.2 ⚠方差未达§2.2闸口径
- clean500 Δ vs P1-B（次）: n=2 mean=+28.8pp sd=9.3pt min=+22.2 max=+35.4 ⚠方差未达§2.2闸口径
- eval500 Δ vs P1-B: n=1 mean=+20.0pp sd=—pt min=+20.0 max=+20.0

## 执行事件

- 2026-10-03T07:59：seed0 终评完成后 driver 在 `write_report` 崩溃（`_fmt_pair` 把 None 交给 `:.3g` 格式串，`eval500_pair` 为裸 pair struct）；报告未生成、批中止，seed11 未跑。seed0 数据（json/log/runs）完整保留。
- 修复：`_fmt_pair` 及同类报告格式化对 None 安全（→ `n/a`），兼容裸 pair struct；本次刷新后补跑 seed11（`--batch --seeds 11`），报告由 2 seeds 合并。
- 2026-10-05T07:12–07:19（seed11 终评期间）：**另一 lane 在本 repo 并发改动未提交源码**（`env/obs/{ld,schema,__init__,builder,ttc,lane}.py`、`net/{mem,model,encoders}.py` 等；HEAD 亦由 `c37acbb` 前进至 `2f4450e`）。seed11 的 u125 clean500（07:17:45 启动）/ tg45 / T3 / eval500 在改动后启动 → 全部首帧报 `obs['ttc']` 形状错误（n_error=n），已标记 void 并从分布剔除；u150 clean500（07:05:04 启动，早于首次改动 07:12:37）与全部 sub150/训练不受影响。建议：tree 干净后重跑 seed11 的 u125 clean500 + tg45 + T3 + eval500。

> 判据（§9.3）：期望带 +8–15pt 且方差可控；sd > 5pt 标注『方差未达 §2.2 闸口径』；Δ<0/崩解 → 记录等编排决策。
