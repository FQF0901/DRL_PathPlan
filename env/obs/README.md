# env/obs

目的：把环境状态编码为定长观测 + 6 帧历史（schema v6），供网络直接消费。观测实现（`*.py` 内容 + schema 版本）的指纹 `obs_fingerprint()` 写入 BC 数据集 meta 并在加载时比对（`env/obs/__init__.py:37-49`）。

## 组装与接口
- `ObservationBuilder(config)`：内置通道与默认历史通道见 `builder.py:64-76`；`build(env, spec)` 返回当前帧各通道 + mask + 历史键，完整键/形状见 `builder.py:3-19`。
- `episode_step==0` 时自动 `reset()`（清空历史与 OD 槽位表，`builder.py:141-145`）；自定义通道用 `register(channel)` 挂载（`:131`），实现 `base.py::ObservationChannel.build -> (features (N,F), mask (N,))`（`base.py:124-147`）。
- 坐标系约定：自车系 **x 前向 / y 左向**（源码证据见 `base.py:13-24`）；历史帧用 `se2_align` 重表达到当前 ego 系（`base.py:65-101`）。

## 通道（当前帧）
| 通道 | 形状 | 内容/来源 |
| --- | --- | --- |
| ego | (1,8) | v, a_long, a_lat, yaw_rate, steer, curvature, prev_ds, prev_dtheta（`ego.py:1-21, 32`）；末 2 维 = 调用方写 `env.prev_policy_action=(ds,dθ)`，未写保持 0（`ego.py:92-99`） |
| od | (16,9) | dx, dy, vx, vy（相对速度）, cosθ, sinθ, L, W, type_id（`od.py:3-12, 231-244`）；companion `od_id`(int64)/`od_presence`（`od.py:87-97, 316-318`） |
| ld | (16,7) | dx, dy, heading_rel, curvature, speed_limit（m/s 原始值）, left/right_line_type_id（`ld.py:32-43`） |
| nav | (1,11) | 2 checkpoint 自车系 (x,y) + 6 命令 one-hot（3 实际+3 保留）+ route_completion（`nav.py:11, 22-25, 46`） |
| signal | (1,4) | 恒 `[0,0,0,1]`（无灯），mask=1（`signal.py:1-10, 28-32`） |
| others | (1,21+K) | nav(11) + speed_limit(1, 归一化) + signal(4) + static(5) + road_class one-hot(K=12)（`others.py:1-31, 67-72`；K=`taxonomy.GEOMETRY_LABELS`，`:109-119`） |
| ego_world | (1,3) | t0 世界系位姿 (x,y,θ)，rollout/教师强制重算 nav 的锚点（`world.py:77-96`） |
| route_world | (64,2) | 世界系路线折线（首段起点 + 各段终点），不足重复末点、mask=0（`world.py:42-74, 99-120`） |

- OD/LD 共用盒式 scope：默认前 150 / 后 50 / 左右 25 m（`builder.py:76`，`od.py:40-41`）。
- OD 固定槽位（v2）：新对象取空槽，槽满按 `(min(TTC, 5s), 距离)` 紧迫度驱逐"最长未出现且已出盒"的槽；出盒 `presence=0` 保留最近盒内特征，超过 `release_after_s=1.0 s` 释放（`od.py:14-41, 72-73`）。`od_mask`=槽已分配（身份有效），`od_presence`=本帧在盒内观测到（特征新鲜），下游必须都看（`od.py:35-38`）。
- LD 采样 offset = `LD_OFFSETS_M` = **{5,10,15,20,30} m**（`ld.py:76`）；当前车道优先占满全部 offset（primary 5 槽），其余候选车道按 offset 环优先填充，共 16 槽（`ld.py:209-261`）。采样点超车道末端/投影失败 → 该槽 mask=0（`ld.py:227-234`）。
- static 段（others dim 16..20）：自车当前车道走廊（默认覆盖左右各 1 条相邻车道）内前方最近 `BaseBuilding`：present / gap_norm / rel one-hot（`static.py:64-73, 106-200`）。
- 通道不可用时返回全 0 + mask=0（`base.py:104-121`）；`others_mask=1 ⇔ ego 存在`，各分量有确定性回退（`others.py:43-47`）。

## 历史（`memory.py`）
- `FrameMemory(frames=6, interval=5, channels=("ego","others","od","ld"))`（`memory.py:34-35, 50-64`）：每 5 个 env step（0.5 s）入库一帧；帧序旧→新，`stack` 时 SE(2) 对齐到当前帧（`memory.py:1-22, 164-208`）。
- 预热期重复最旧真实帧补满，`hist_valid` 按真实缓冲长度标 0/1（`memory.py:166-178`）；补位帧的伴随数组填缺省值（`memory.py:210-221`）。
- `od_hist` 与 `od_id_hist`/`od_presence_hist` 同槽位、跨帧身份一致，不做 SE(2) 变换（`od.py:94-97`，`memory.py:16-18`）。
- 时间戳非严格递增直接 raise（`memory.py:97-101`）。

## 配置入口
- `ObservationBuilder` 键（主要键，全部可选）：`channels`、`scope` 及 `od`/`ld` 各自的 `front_m/rear_m/left_m/right_m`（顶层扁平同名键兼容旧配置，`:96-102`）、`od.num_slots/ttc_cap_s/release_after_s/physics_dt`、`ld.num_slots/offsets`、`others.speed_limit_norm_mps/static_scan_range_m/static_lat_margin_m/static_lane_span`、`memory.frames/interval/channels`、`physics_dt`（`builder.py:29-39`）。
- `config/env.yaml::obs` 的扁平键 `topk_objects`/`topk_lanes`/`history_frames` 被 `builder` 支持为别名（`builder.py:89-95`）；`scope` 一节被 OD/LD 共用（`:96-102`）。注意：该段**未被运行链路消费**（训练池不传 obs_config；评测侧 `config["env"]` 为空）——实际用 `ObservationBuilder` 内置默认，值与此镜像（见 `config/README.md` 未消费键表）。

## 已知文档与代码不一致（以代码为准，未修改）
- LD 采样 offset 代码为 `(5,10,15,20,30)`（`ld.py:76`），但 `schema.py:239`（`schema_manifest()["frame"]["ld"]["semantics"]`，实机仍输出 `{0,20,40,60,80}`）、`schema.py:18-23`（v5/v6 历史段）、`__init__.py:23-25`、`builder.py:25-27` 的文案仍写 `{0,20,40,60,80}`；`ld.py` 内部注释（`:24-26, :256`）也残留旧口径。
