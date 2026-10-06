# v7 Q6：s11 eval500 碰撞类型仪器化重放

- ckpt: `/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`
- spec: `env/specs/scenarios_eval500.json`（101 条 collision，seed 与 CSV 逐条一致）
- 协议: tracker=lqr / eval-reference=plan / max_steps=1000 / workers=6 / device=cuda
- 重放终止复现率: 101/101 条 collision（CSV 口径）

## 0. 直接回答（二选一 + 另附类别）

- **后车撞自车（rear-end by follower）: 5/101 = 5.0%**（其中对手 policy 分布 {'IDMPolicy': 5}）
- **自车追尾前车（ego rear-ends lead）: 64/101 = 63.4%**
- 侧碰/cut-in: 14/101 = 13.9%
- 撞静态物（建筑/岗亭/object/sidewalk）: 18/101 = 17.8%

## 1. 分类计数

| 类别 | n | 占比 |
|---|---|---|
| 后车撞自车（rear-end by follower） | 5 | 5.0% |
| 自车追尾前车（ego rear-ends lead） | 64 | 63.4% |
| 侧碰（side swipe） | 6 | 5.9% |
| 撞静态建筑（岗亭/建筑） | 17 | 16.8% |
| 撞路缘/人行道（sidewalk） | 1 | 1.0% |
| 侧碰/cut-in（side + cut-in） | 8 | 7.9% |

## 2. 按 primary 分类

| primary | n | 后车撞自车（rear-end by follower） | 自车追尾前车（ego rear-ends lead） | 侧碰（side swipe） | 侧碰/cut-in（side + cut-in） | 撞静态建筑（岗亭/建筑） | 撞静态物（traffic object） | 撞路缘/人行道（sidewalk） |
|---|---|---|---|---|---|---|---|---|
| curve | 14 | 0 | 13 | 1 | 0 | 0 | 0 | 0 |
| intersection | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| merge | 5 | 0 | 3 | 1 | 0 | 0 | 0 | 1 |
| ramp_in | 10 | 0 | 8 | 1 | 1 | 0 | 0 | 0 |
| ramp_out | 13 | 0 | 10 | 2 | 1 | 0 | 0 | 0 |
| roundabout | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| split | 9 | 0 | 7 | 1 | 1 | 0 | 0 | 0 |
| straight | 14 | 1 | 12 | 0 | 1 | 0 | 0 | 0 |
| t_intersection | 7 | 4 | 2 | 0 | 1 | 0 | 0 | 0 |
| tollgate | 25 | 0 | 8 | 0 | 0 | 17 | 0 | 0 |
| uturn | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

## 3. 典型案例（每类 ≤5）

### 后车撞自车（rear-end by follower）

- id=41 primary=straight blocks=SSSS crash_step=291 components=['crash', 'crash_vehicle'] partner=LVehicle(vehicle, policy=IDMPolicy) contact=rear rel=(-0.89,3.72) rel_v=(-1.39,-0.09) partner_a_lon_2s=None lane_changed=None ego_v=3.67 ego_a_lon_2s=-0.30 ego_throttle_min=0.52 brake_frac=0.00 | ego throttle_min_2s=0.5152956649812105, brake_frac=0.0
- id=109 primary=t_intersection blocks=CTS crash_step=603 components=['crash', 'crash_vehicle'] partner=DefaultVehicle(vehicle, policy=IDMPolicy) contact=rear rel=(-3.80,-2.85) rel_v=(-1.22,0.00) partner_a_lon_2s=None lane_changed=None ego_v=3.23 ego_a_lon_2s=-0.88 ego_throttle_min=0.49 brake_frac=0.00 | ego throttle_min_2s=0.48676465662986984, brake_frac=0.0
- id=326 primary=t_intersection blocks=STC crash_step=286 components=['crash', 'crash_vehicle'] partner=DefaultVehicle(vehicle, policy=IDMPolicy) contact=rear rel=(-4.63,-1.02) rel_v=(-0.03,3.83) partner_a_lon_2s=-0.96 lane_changed=True ego_v=5.15 ego_a_lon_2s=1.07 ego_throttle_min=1.00 brake_frac=0.00 | partner lane_changed_2s; follower a_lon_2s=-0.96 m/s^2; ego throttle_min_2s=1.0, brake_frac=0.0
- id=917 primary=t_intersection blocks=STC crash_step=200 components=['crash', 'crash_vehicle'] partner=SVehicle(vehicle, policy=IDMPolicy) contact=rear rel=(-1.45,-3.32) rel_v=(-4.97,7.72) partner_a_lon_2s=None lane_changed=None ego_v=3.63 ego_a_lon_2s=-0.24 ego_throttle_min=0.69 brake_frac=0.00 | ego throttle_min_2s=0.6940228997577398, brake_frac=0.0
- id=926 primary=t_intersection blocks=STC crash_step=478 components=['crash', 'crash_vehicle'] partner=MVehicle(vehicle, policy=IDMPolicy) contact=rear rel=(-4.55,-1.90) rel_v=(0.96,0.07) partner_a_lon_2s=None lane_changed=None ego_v=4.99 ego_a_lon_2s=-0.15 ego_throttle_min=0.59 brake_frac=0.00 | ego throttle_min_2s=0.5939885668962938, brake_frac=0.0

### 自车追尾前车（ego rear-ends lead）

- id=23 primary=curve blocks=SCSCS crash_step=554 components=['crash', 'crash_vehicle'] partner=MVehicle(vehicle, policy=IDMPolicy) contact=front rel=(4.55,-0.23) rel_v=(0.43,-0.12) partner_a_lon_2s=1.20 lane_changed=False ego_v=2.69 ego_a_lon_2s=-0.95 ego_throttle_min=0.45 brake_frac=0.00 | ego a_lon_2s=-0.9518991314514176, throttle_min_2s=0.44774815988623945; lead a_lon_2s=+1.20 m/s^2 (急刹判据)
- id=35 primary=merge blocks=SyS crash_step=215 components=['crash', 'crash_vehicle'] partner=XLVehicle(vehicle, policy=IDMPolicy) contact=front rel=(1.26,2.52) rel_v=(-3.33,-0.14) partner_a_lon_2s=None lane_changed=None ego_v=4.09 ego_a_lon_2s=-0.38 ego_throttle_min=0.46 brake_frac=0.00 | ego a_lon_2s=-0.3841065452981156, throttle_min_2s=0.45622349841936344
- id=53 primary=straight blocks=CSSSS crash_step=351 components=['crash', 'crash_vehicle'] partner=XLVehicle(vehicle, policy=IDMPolicy) contact=front rel=(4.79,-2.48) rel_v=(-0.59,-0.99) partner_a_lon_2s=0.18 lane_changed=False ego_v=3.16 ego_a_lon_2s=-0.53 ego_throttle_min=0.52 brake_frac=0.00 | ego a_lon_2s=-0.5253582329025592, throttle_min_2s=0.518692237982971; lead a_lon_2s=+0.18 m/s^2 (急刹判据)
- id=95 primary=merge blocks=SyS crash_step=336 components=['crash', 'crash_vehicle'] partner=MVehicle(vehicle, policy=IDMPolicy) contact=front rel=(4.68,-1.69) rel_v=(-5.72,-0.00) partner_a_lon_2s=None lane_changed=None ego_v=5.76 ego_a_lon_2s=0.81 ego_throttle_min=0.56 brake_frac=0.00 | ego a_lon_2s=0.8114944315320485, throttle_min_2s=0.5621392460787433
- id=101 primary=tollgate blocks=S$S crash_step=292 components=['crash', 'crash_vehicle'] partner=XLVehicle(vehicle, policy=IDMPolicy) contact=front rel=(0.39,-3.43) rel_v=(-4.15,-0.01) partner_a_lon_2s=None lane_changed=None ego_v=4.43 ego_a_lon_2s=0.55 ego_throttle_min=0.39 brake_frac=0.00 | ego a_lon_2s=0.5481784699743653, throttle_min_2s=0.38887164035225824

### 侧碰（side swipe）

- id=5 primary=ramp_out blocks=SRS crash_step=271 components=['crash', 'crash_vehicle'] partner=MVehicle(vehicle, policy=IDMPolicy) contact=right rel=(0.50,-2.06) rel_v=(-0.91,0.23) partner_a_lon_2s=0.05 lane_changed=False ego_v=3.44 ego_a_lon_2s=-0.52 ego_throttle_min=0.54 brake_frac=0.00 | contact=right rel=(0.50,-2.06)
- id=107 primary=split blocks=CYS crash_step=264 components=['crash', 'crash_vehicle'] partner=SVehicle(vehicle, policy=IDMPolicy) contact=right rel=(3.45,-1.64) rel_v=(0.05,0.44) partner_a_lon_2s=-0.05 lane_changed=False ego_v=4.68 ego_a_lon_2s=0.08 ego_throttle_min=0.41 brake_frac=0.00 | contact=right rel=(3.45,-1.64)
- id=506 primary=merge blocks=SyS crash_step=269 components=['crash', 'crash_vehicle'] partner=SVehicle(vehicle, policy=IDMPolicy) contact=right rel=(3.94,-1.84) rel_v=(-3.09,-0.30) partner_a_lon_2s=0.01 lane_changed=False ego_v=4.56 ego_a_lon_2s=-0.37 ego_throttle_min=0.26 brake_frac=0.00 | contact=right rel=(3.94,-1.84)
- id=854 primary=ramp_in blocks=SrS crash_step=519 components=['crash', 'crash_vehicle'] partner=SVehicle(vehicle, policy=IDMPolicy) contact=left rel=(2.37,1.74) rel_v=(-3.17,0.40) partner_a_lon_2s=0.41 lane_changed=False ego_v=4.50 ego_a_lon_2s=-0.27 ego_throttle_min=0.38 brake_frac=0.00 | contact=left rel=(2.37,1.74)
- id=887 primary=ramp_out blocks=SRS crash_step=436 components=['crash', 'crash_vehicle'] partner=DefaultVehicle(vehicle, policy=IDMPolicy) contact=left rel=(3.62,1.59) rel_v=(-3.41,-0.02) partner_a_lon_2s=0.08 lane_changed=False ego_v=4.34 ego_a_lon_2s=-0.45 ego_throttle_min=0.45 brake_frac=0.00 | contact=left rel=(3.62,1.59)

### 撞静态建筑（岗亭/建筑）

- id=34 primary=tollgate blocks=CS$ crash_step=961 components=['crash', 'crash_building'] partner=TollGateBuilding(tollgate, policy=None) contact=front rel=(7.21,0.10) rel_v=(-0.45,0.00) partner_a_lon_2s=0.00 lane_changed=False ego_v=0.45 ego_a_lon_2s=-1.21 ego_throttle_min=0.86 brake_frac=0.00 | partner=TollGateBuilding
- id=76 primary=tollgate blocks=CS$ crash_step=771 components=['crash', 'crash_building'] partner=TollGateBuilding(tollgate, policy=None) contact=front rel=(7.26,-0.30) rel_v=(-0.56,0.00) partner_a_lon_2s=0.00 lane_changed=False ego_v=0.56 ego_a_lon_2s=-1.26 ego_throttle_min=1.00 brake_frac=0.00 | partner=TollGateBuilding
- id=131 primary=tollgate blocks=S$S crash_step=374 components=['crash', 'crash_building'] partner=TollGateBuilding(tollgate, policy=None) contact=front rel=(7.25,0.35) rel_v=(-0.03,0.00) partner_a_lon_2s=0.00 lane_changed=False ego_v=0.03 ego_a_lon_2s=-1.37 ego_throttle_min=1.00 brake_frac=0.00 | partner=TollGateBuilding
- id=147 primary=tollgate blocks=SS$ crash_step=479 components=['crash', 'crash_building'] partner=TollGateBuilding(tollgate, policy=None) contact=front rel=(7.25,0.32) rel_v=(-0.17,-0.00) partner_a_lon_2s=0.00 lane_changed=False ego_v=0.17 ego_a_lon_2s=-1.38 ego_throttle_min=1.00 brake_frac=0.00 | partner=TollGateBuilding
- id=217 primary=tollgate blocks=CS$ crash_step=392 components=['crash', 'crash_building'] partner=TollGateBuilding(tollgate, policy=None) contact=front rel=(7.23,-0.46) rel_v=(-0.38,0.00) partner_a_lon_2s=0.00 lane_changed=False ego_v=0.38 ego_a_lon_2s=-1.74 ego_throttle_min=0.65 brake_frac=0.00 | partner=TollGateBuilding

### 撞路缘/人行道（sidewalk）

- id=619 primary=merge blocks=CSy crash_step=474 components=['crash', 'crash_sidewalk'] partner=None(None, policy=None) contact=None rel=(None,None) rel_v=(None,None) partner_a_lon_2s=None lane_changed=None ego_v=4.61 ego_a_lon_2s=0.95 ego_throttle_min=0.73 brake_frac=0.00 | no OBB overlap; nearest=DefaultVehicle center_dist=137.58 m; crash_sidewalk

### 侧碰/cut-in（side + cut-in）

- id=104 primary=ramp_in blocks=SrS crash_step=492 components=['crash', 'crash_vehicle'] partner=LVehicle(vehicle, policy=IDMPolicy) contact=right rel=(4.07,-1.23) rel_v=(-3.62,-0.13) partner_a_lon_2s=-0.05 lane_changed=True ego_v=3.83 ego_a_lon_2s=-0.38 ego_throttle_min=0.49 brake_frac=0.00 | partner lane_changed_2s; lateral interaction rel_vy=-0.1258106231689453; contact=right rel=(4.07,-1.23)
- id=278 primary=split blocks=CYS crash_step=398 components=['crash', 'crash_vehicle'] partner=DefaultVehicle(vehicle, policy=IDMPolicy) contact=right rel=(2.49,-1.81) rel_v=(-0.59,1.24) partner_a_lon_2s=1.80 lane_changed=True ego_v=5.47 ego_a_lon_2s=0.13 ego_throttle_min=0.31 brake_frac=0.00 | partner lane_changed_2s; lateral interaction rel_vy=1.2369232177734375; contact=right rel=(2.49,-1.81)
- id=602 primary=t_intersection blocks=STC crash_step=371 components=['crash', 'crash_vehicle'] partner=MVehicle(vehicle, policy=IDMPolicy) contact=left rel=(3.19,1.63) rel_v=(1.78,-1.70) partner_a_lon_2s=1.17 lane_changed=False ego_v=5.23 ego_a_lon_2s=-0.20 ego_throttle_min=0.49 brake_frac=0.00 | lateral interaction rel_vy=-1.6989593505859375; contact=left rel=(3.19,1.63)
- id=695 primary=ramp_out blocks=SRS crash_step=180 components=['crash', 'crash_vehicle'] partner=LVehicle(vehicle, policy=IDMPolicy) contact=right rel=(2.22,-2.01) rel_v=(-0.10,0.58) partner_a_lon_2s=1.29 lane_changed=False ego_v=3.22 ego_a_lon_2s=-1.17 ego_throttle_min=0.37 brake_frac=0.00 | lateral interaction rel_vy=0.5825777053833008; contact=right rel=(2.22,-2.01)
- id=697 primary=straight blocks=CSS crash_step=127 components=['crash', 'crash_vehicle'] partner=XLVehicle(vehicle, policy=IDMPolicy) contact=right rel=(1.52,-3.96) rel_v=(-3.30,3.38) partner_a_lon_2s=-0.14 lane_changed=False ego_v=5.13 ego_a_lon_2s=-0.27 ego_throttle_min=0.53 brake_frac=0.00 | lateral interaction rel_vy=3.3777236938476562; contact=right rel=(1.52,-3.96)

## 4. 重放口径核对

- 全部 101 条重放终止均为 collision。

## 5. 后车撞自车（rear_by_follower）对手证据

- policy 分布：{'IDMPolicy': 5}
- 逐条（follower = 撞自车的后车）：
  - id=41 primary=straight contact=rear policy=IDMPolicy rel=(-0.89,3.72) follower_v=2.2780381490908947 follower_a_lon_2s=None lane_changed=None | ego_a_lon_2s=-0.2988858810809829 ego_brake_frac=0.0 ego_throttle_min=0.5152956649812105
  - id=109 primary=t_intersection contact=rear policy=IDMPolicy rel=(-3.80,-2.85) follower_v=2.0067198465477767 follower_a_lon_2s=None lane_changed=None | ego_a_lon_2s=-0.877394361405649 ego_brake_frac=0.0 ego_throttle_min=0.48676465662986984
  - id=326 primary=t_intersection contact=rear policy=IDMPolicy rel=(-4.63,-1.02) follower_v=6.398646618408948 follower_a_lon_2s=-0.9618773865771111 lane_changed=True | ego_a_lon_2s=1.0671476641907036 ego_brake_frac=0.0 ego_throttle_min=1.0
  - id=917 primary=t_intersection contact=rear policy=IDMPolicy rel=(-1.45,-3.32) follower_v=7.833396881758281 follower_a_lon_2s=None lane_changed=None | ego_a_lon_2s=-0.23979862556037435 ego_brake_frac=0.0 ego_throttle_min=0.6940228997577398
  - id=926 primary=t_intersection contact=rear policy=IDMPolicy rel=(-4.55,-1.90) follower_v=5.946897188286326 follower_a_lon_2s=None lane_changed=None | ego_a_lon_2s=-0.14660195992986003 ego_brake_frac=0.0 ego_throttle_min=0.5939885668962938

## 6. 自车追尾前车（ego_rear_ends_lead）自车行为证据

- n=64；自车 2s 内减速率 >1 m/s² 的条数：15；自车 brake_frac>0 的条数：0；自车 2s 平均 a_lon=-0.5623511175556669
- 前车 2s 急刹（a_lon<-2）条数：0；前车加速（>0.5）条数：39；前车 2s 历史缺失（新生成）条数：15
