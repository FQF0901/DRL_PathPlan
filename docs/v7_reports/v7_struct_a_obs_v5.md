# v7 结构迭代 A / Lane A：obs v5（LD 远场 + 当前车道块 + TTC token）

- 日期：2026-10-05
- 范围：`env/obs/**`、`net/encoders.py`、`net/mem.py`、`net/model.py`（仅令牌集合接线）、
  `tools/collect_expert.py`、`tests/**`。**未动** `net/plan_head.py`、ST-GNN/空间层与 plan 生成路径（Lane B）。
- 内容 commit：`0291f3a`；锚 commit：`aa4c69c`（见文末"提交记录"）
- 结论：schema v5 落地；全量 `pytest` **654 passed**；GPU 冒烟（tollgate spec 采集 + cheap/full/rollout 前向）**无 NaN**。

## 1. 字段表

### 1.1 LD（`env/obs/ld.py`，16×7，字段不变）

| 项 | v4 | v5 |
|---|---|---|
| 采样 offset | {5,10,15,20,30} m | **{20,40,60,80} m**（`LD_OFFSETS_M`） |
| 槽位分配 | 当前车道 5 primary + 环填充 | **当前车道 4 primary** + 其余车道按 20/40/60/80 环填充（环内按车道优先级）；总槽位 16 不变 |
| 特征 7 维 | `dx,dy,heading_rel,curvature,speed_limit,left_line_type_id,right_line_type_id` | 不变 |
| 近场（<20 m） | 由 LD 覆盖 | **不在 LD**；由 `lane` 块显式补偿（其余车道近场本次接受不可见） |

### 1.2 `lane`：当前车道块（新，`env/obs/lane.py`，1×17）

只描述 `ego.lane`（缺失时 `navigation.current_ref_lanes[0]`），单槽、不参与 LD 槽位竞争。

| dim | 名称 | 语义 / 单位 |
|---|---|---|
| 0 | `d_lat` | ego 相对本车道中心线的横向偏差（m）。**正 = ego 在中心线左侧**；= `-lane.local_coordinates(ego.position)[1]`（MetaDrive 车道系 lat 正 = 车道方向右侧）。直道/航向对齐时 `d_lat == -near_dy` |
| 1 | `heading_err` | `wrap(lane.heading_theta_at(s0) - ego.heading_theta)`（rad）。**正 = 车道方向在 ego 航向左侧**（ego 相对车道右偏航）；与 LD `heading_rel` 同号（车道−ego）。收敛目标 `heading_err ≈ +k·d_lat` |
| 2 | `lane_width` | `lane.width`（m） |
| 3 | `curvature` | 投影点 dθ/ds（中心差分；右转负，1/m） |
| 4 | `speed_limit` | 本车道限速原始值（m/s，与 LD 同口径不换算） |
| 5–7 | `near_dx/dy/heading_rel` | `s0+5 m` 中心线点（自车系点 + 车道−ego 航向） |
| 8–10 | `mid_*` | `s0+15 m`（近场-中距锚） |
| 11–13 | `far_*` | `s0+60 m`（远场锚） |
| 14–16 | `near_valid/mid_valid/far_valid` | 1 = 该采样点在车道长度内；否则几何清零 |

- `s0` = ego 纵向投影 clamp 到 [0,length]（投影失败退化 0）；无车道/无 ego → 全 0 + `lane_mask=0`。
- 不套 OD/LD 盒式 scope（只描述自车所在车道，60 m 恒在 150 m 前视内）。
- 对齐元数据：3 个点对 + 4 个角度维（`heading_err` 与 3 个 `heading_rel`，均"车道−ego"，对齐时减 Δθ）；
  当前**不进** 6 帧历史（当前帧上下文 token）。

### 1.3 `ttc`：TTC 上下文 token（新，`env/obs/ttc.py`，1×12）

**只读** OD 槽位渲染（槽序/9 维字段/排序策略不动）；只统计 `presence=1` 的新鲜槽（陈旧槽几何不新鲜，不参与）。

| dim | 名称 | 语义 / 单位 |
|---|---|---|
| 0 | `min_ttc_x` | 新鲜槽 `ttc_x` 最小值，clamp [0,cap]；无 → cap=5 s |
| 1 | `min_ttc_path` | 新鲜槽 `ttc_path` 最小值，同上 |
| 2 | `n_lt3_x` | `ttc_x < 3 s` 的槽数 |
| 3 | `n_lt3_path` | `ttc_path < 3 s` 的槽数 |
| 4 | `resp_index` | 责任槽位下标（0..15）；无 → −1 |
| 5–6 | `resp_dx/dy` | 责任槽位相对位置（m，自车系） |
| 7–8 | `resp_vx/vy` | 责任槽位相对速度（m/s，自车系） |
| 9 | `resp_ttc` | 责任槽位获胜 TTC = min(ttc_x, ttc_path)，clamp cap |
| 10 | `resp_is_path` | 1 = `ttc_path` 获胜（`ttc_path < ttc_x`） |
| 11 | `valid` | 1 = 至少一个新鲜槽有有限 TTC；0 = 无接近对象/无数据 |

- 责任槽位 = 两口径合并 TTC 最小的新鲜槽；并列取最小槽下标（确定性）。
- `ttc_mask=1` ⇔ ego 存在且 OD 通道可用；`valid` 区分"无风险"与"无数据"。

## 2. 几何口径

### 2.1 LD 环填充（槽序）

1. 当前车道（candidate priority 0）占满 4 个 offset（primary，20→80 m）；
2. 其余候选车道按"环优先"填充：先 20 m 环（所有车道），再 40/60/80 m，环内按车道优先级；
3. 采样 `s = clamp(ego 投影, 0, length) + offset`；超车道末端/投影失败 → 该槽 mask=0（通道降级）。

### 2.2 当前车道块符号

- `d_lat`：`-lat_lane`（正 = ego 偏左）；直道已知场景校验 `d_lat == -near_dy`（测试锁定）。
- `heading_err = wrap(θ_lane(s0) - θ_ego)`（正 = 车道方向在 ego 左侧；ego 朝左偏 → 负）。
- 近场点 `near_*` 在 ego 系：直道 ego 在中心线左侧 0.7 m、航向对齐 → `near_dy = -0.7`。

### 2.3 TTC 两口径（恒速、自车系；`r=(dx,dy)`、`v=(vx,vy)` 均为相对量）

1. `ttc_x`（纵向轴，与 OD 紧迫度同口径）：`dx>0 且 vx<-eps` 时 `dx/(-vx)`，否则 inf；
2. `ttc_path`（**沿自车路径投影版**）：解 `|r + v·t| = R` 的最小正根 = 首次进入自车碰撞圆
   （半径 `R=collision_radius_m=2.5 m`）的时间；已在该圆内且仍在接近（`r·v<0`）→ 0；远离/无解 → inf。
   - 设计说明：在 obs 层 ego 路径切线即 ego 航向，把相对运动对"路径上的 ego 圆"做投影，
     等价于沿路径的最近接近/进入时间；与标量 x 轴投影（`ttc_x`）互补，用于捕获横向切入/斜向逼近
     （纯横向 `dx≤0` 时 `ttc_x` 漏检）。
   - 解析校验（测试锁定）：`r=(0,6), v=(0,-5)` → `ttc_path=0.7 s`；`r=(10,0), v=(-2,0)` → `ttc_x=5 s, ttc_path=3.75 s`。

## 3. schema v5 / 采集

- `OBS_SCHEMA_VERSION = 5`；`schema_manifest()["schema_version"] = 5`；新增
  `lane_layout` / `ttc_layout` / `ld_layout`（offset/槽位策略/近场说明）。
- `obs_fingerprint()` 前缀升为 `v5-`（本次冒烟数据集：`v5-190fac36e6be`）；旧 v4 数据集加载必然不匹配（既有纪律）。
- 旧数据回退（A4/v4 模式）：`lane`/`ttc` 缺键 → net 全 0 + mask=0 + **一次性** RuntimeWarning，
  不参与注意力、旧数据前向兼容（`net.mem.context_features_from_obs`）。
- `tools/collect_expert.py`：`CURRENT_CHANNELS` 增 `lane`/`ttc`（逐帧入库）；
  `_channel_dim`/`_channel_slots`/`_npz_schema_manifest` 自动透传（1×17 / 1×12 + mask）。
- net 令牌集合（`net.model._head_tokens`）：
  `[OD16, LD16, **lane**, others, ego, nav, **ttc**, signal, latent]`（T=39；lane 与 LD 主块并列、
  ttc 与 nav 相邻）。lane/ttc 为 **t0 上下文**，rollout 逐步复用（逐步重算属 Lane B）。

## 4. 测试与冒烟

- 全量：`tools/venv-python -m pytest tests/ -q` → **654 passed**（含新增 `tests/test_obs_v5_lane_ttc.py` 16 项：
  LD offset/环填充/短车道、lane 已知场景数值（d_lat/heading_err/曲率/valid）、TTC 数值/责任槽位/cap/
  presence 门控/只读 OD、schema/指纹/采集透传、net 梯度可达 + 旧数据一次性回退）。
- GPU 冒烟（RTX 4070；串行）：
  - 采集：tollgate spec（id 9368，blocks `S$S`）1 条，`--max-steps 600` → 84 行 / 72 可训练（1.1 s）；
  - 前向：`pipeline.frames` 精确查表重建 6 帧历史（`hist_valid` 全 1），组装 obs（含 lane/ttc）
    → `DrivingModel()` cheap/full/rollout 全部**无 NaN**；token 形状 (1,39,96)；cheap vs full `action_mu` 一致；
    `plan[:,0] == action_mu`；
  - 采集值抽查（row 44 / step 220）：`lane=[d_lat −0.0256, heading_err 0.0382, width 3.5, far_valid 0]`；
    `ttc=[min_x 5.0(cap), min_path 5.0, resp_index 1, resp_ttc 5.0, valid 1]`；
    全数据集：`lane_mask` 全 1、`ttc_mask` 全 1；`min_ttc_x` min 2.10 s；`n_lt3_x` max 1。
  - 命令：
    ```bash
    tools/venv-python tools/collect_expert.py --specs /tmp/opencode/struct_a_smoke_spec.json \
        --out /tmp/opencode/struct_a_collect --workers 1 --max-steps 600
    tools/venv-python /tmp/opencode/struct_a_forward_smoke.py
    ```

## 5. 未决 / 后续（Lane B 或接线）

1. **训练管线透传缺口（必须补，1–2 行级）**：`pipeline/trainer.py::BCDataset._obs_keys`（硬编码
   `("ego","od","ld","nav","signal","others","ego_world","route_world")`）与
   `SINGLE_SLOT_CHANNELS`（加 `lane`/`ttc`）、`pipeline/buffer.py::DEFAULT_CHANNELS`（PPO 缓冲形状）
   均未包含新通道 → **本迭代约束下未改 pipeline**，训练/PPO 路径会走"缺键回退"（token mask=0）。
   重采 v5 数据后、开训前必须先接线，否则新 token 不生效（`BCDataset` 会把 npz 里的 lane/ttc 过滤掉）。
2. **TTC 的 rollout 重算**：当前 ttc token 为 t0 上下文；Lane B 可在 rollout 内用预测 OD 重算 per-slot TTC
   （本迭代按约束只做令牌接线）。
3. **ST-GNN 空间节点**：lane 目前只进 policy/value 交叉注意力令牌集合；若要在 ST-GNN 消息传递中作为
   空间节点（与 OD/LD 并列）需改 `net/st_gnn.py`/`net/spatial.py`（Lane B 决策）。
4. **plan_head 融合路径**：lane/ttc 未进 plan head 融合（`net/plan_head.py` 冻结）；Lane B 的 K-anchor
   设计可显式消费 lane token（d_lat/航向误差）。
5. **TTC 参数**：`collision_radius_m=2.5`、`horizon_s=3`、`cap=5` 为初值（可配置）；是否按 ego/对象尺寸
   动态化、是否把 TTC 并入 OD 排序（**本迭代明确不动**）留待评审。
6. **lane 通道历史**：当前不进 6 帧历史；若 Lane B 需要，可加入 `memory.channels`（对齐元数据已就绪：
   点对 + 角度维）。
7. **配置透传**：`config/env.yaml` 未显式写 ld/lane/ttc 段（走代码默认）；如需调参在 `obs.ld/lane/ttc` 段配置即可。

## 6. 提交记录

- 内容 commit：`0291f3ae4cc8c0f33d1a8b73abb6671c699d821e`（feat：obs v5 + net 接线 + collect + tests）
- 锚 commit：`aa4c69c`（docs/v7_program_prereg.md §8 修订锚行，不改动其余内容）
