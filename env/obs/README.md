# env/obs

目的：把环境状态编码为定长观测 + 6 帧历史（schema v7），供网络直接消费。观测实现（`*.py` 内容 + schema 版本）的指纹 `obs_fingerprint()` 写入 BC 数据集 meta 并在加载时比对（`env/obs/__init__.py:45-57`）。

## 组装与接口
- `ObservationBuilder(config)`：内置通道与默认历史通道见 `builder.py:71-84`；`build(env, spec)` 返回当前帧各通道 + mask + 历史键，完整键/形状见 `builder.py:3-24`。
- `episode_step==0` 时自动 `reset()`（清空历史与 OD 槽位表，`builder.py:152-156`）；自定义通道用 `register(channel)` 挂载（`:142-144`），实现 `base.py::ObservationChannel.build -> (features (N,F), mask (N,))`（`base.py:127-160`）。
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
| lane_semantic | (1,12) | v7（R21）：相对 target lane 的语义特征（横向偏移/航向差/近远曲率/左右 margin/target 有效性/route 关联/变道需求/到车道末端距离等），归一化除数见 `lane_semantic.LANE_SEMANTIC_NORMALIZATION`；mask=1 ⇔ 选出 target lane（`lane_semantic.py:128, 159, 569-580`；布局 `schema.py:255-275`） |

LD 伴随数组（不进 feats/mask，随 ld 进 6 帧历史为 `<key>_hist`，缺帧按 -1/0 填充）：`ld_lane_role`/`ld_offset_s_norm`/`ld_group`/`ld_route_associated`（`ld.py:48-53, 189-197`；`memory.py:164-224`）。

- OD/LD 共用盒式 scope：默认前 150 / 后 50 / 左右 25 m（`builder.py:86`，`od.py:46-48`）。
- OD 固定槽位（v2）：新对象取空槽，槽满按 `(min(TTC, 5s), 距离)` 紧迫度驱逐"最长未出现且已出盒"的槽；出盒 `presence=0` 保留最近盒内特征，超过 `release_after_s=1.0 s` 释放（`od.py:14-41, 72-73`）。`od_mask`=槽已分配（身份有效），`od_presence`=本帧在盒内观测到（特征新鲜），下游必须都看（`od.py:35-38`）。
- LD 采样 offset = `LD_OFFSETS_M` = **{5,10,15,20,30} m**（`ld.py:96`）；当前车道优先占满全部 offset（primary 5 槽），其余候选车道按 offset 环优先填充，共 16 槽（`ld.py:235-290`）。采样点超车道末端/投影失败 → 该槽 mask=0（`ld.py:253-268`）。
- static 段（others dim 16..20）：自车当前车道走廊（默认覆盖左右各 1 条相邻车道）内前方最近 `BaseBuilding`：present / gap_norm / rel one-hot（`static.py:64-73, 106-200`）。
- 通道不可用时返回全 0 + mask=0（`base.py:104-121`）；`others_mask=1 ⇔ ego 存在`，各分量有确定性回退（`others.py:43-47`）。

## 历史（`memory.py`）
- `FrameMemory(frames=6, interval=5, channels=("ego","others","od","ld"))`（`memory.py:35, 47-58`）：每 5 个 env step（0.5 s）入库一帧；帧序旧→新，`stack` 时 SE(2) 对齐到当前帧（`memory.py:1-22, 164-208`）。
- 预热期重复最旧真实帧补满，`hist_valid` 按真实缓冲长度标 0/1（`memory.py:166-178`）；补位帧的伴随数组填缺省值（`memory.py:210-221`）。
- `od_hist` 与 `od_id_hist`/`od_presence_hist` 同槽位、跨帧身份一致，不做 SE(2) 变换（`od.py:94-97`，`memory.py:16-18`）。
- 时间戳非严格递增直接 raise（`memory.py:97-101`）。

## t0 段纵向外源几何（`path_*`，R38-Q4A）
- `build()` 在通道全部产出后追加 **8 列** `path_*`：`path_speed`、`path_left_margin`、`path_right_margin`、`path_static_distance`、`path_dynamic_distance`、`path_dynamic_ttc`、`path_nav_progress`、`path_distance_to_target_lane_end`，各 `(1,)` float32，并同步产出与值同源的 `<key>__present` 掩码列（`path_field.py:1-42`）。
- 值口径：speed/left/right/nav/d2end = 同帧 `ego[:,0]` / `lane_semantic[:,4]` / `lane_semantic[:,5]` / `nav[:,-1]` / `lane_semantic[:,11]`（= 头侧既有回退口径，逐位）；static/dynamic_distance/dynamic_ttc = 真值公式（`long_exog.py`；静态 = `static.scan_static_blocker` 同函数，动态 = OD 槽矩形最近距 / `dx/(−vx)` cap 60 s）。
- 源通道不在场 ⇒ 该列**不产出**（不留静默 0）；`static/dynamic` 三列恒产出（无真值 ⇒ 0 + `__present=0`，显式主态）。消费方 `net/path_head.long_exog_from_obs` 对 dynamic 两列**缺键 fail-fast**（R38-Q4A 禁静默 0）；`__present` 在场时用于 `long_present`（否则退回"键在场 ⇒ 1"旧口径）。
- 该 8 列**不进** 6 帧历史、不进 `feature_spec`（非通道）；`tools/collect_expert.py` 的数据集白名单（`CURRENT_CHANNELS`/companion）不含它们 ⇒ 不改变既有数据集 schema。

## 配置入口- `ObservationBuilder` 键（主要键，全部可选）：`channels`、`scope` 及 `od`/`ld` 各自的 `front_m/rear_m/left_m/right_m`（顶层扁平同名键兼容旧配置，`:106-112`）、`od.num_slots/ttc_cap_s/release_after_s/physics_dt`、`ld.num_slots/offsets`、`others.speed_limit_norm_mps/static_scan_range_m/static_lat_margin_m/static_lane_span`、`memory.frames/interval/channels`、`physics_dt`（`builder.py:34-45`）；`lane_semantic` 无参数（`builder.py:127`）。
- `config/env.yaml::obs` 的扁平键 `topk_objects`/`topk_lanes`/`history_frames` 被 `builder` 支持为别名（`builder.py:99-105`）；`scope` 一节被 OD/LD 共用（`:106-112`）。注意：该段在标准链路下未被实际传参——训练池 `build_pool` 无 obs_config 参数（`pipeline/trainer.py:7211-7221`）；评测/采集链路透传的是 `config["env"]["obs"]`（`pipeline/eval_runner.py:1794,1832`；`tools/dagger_collect.py:1301` 等），平铺配置下无顶层 `env:` 键 → 空 dict。因此实际用 `ObservationBuilder` 内置默认，值与此镜像（见 `config/README.md` 未消费键表）。

## 已知文档与代码不一致（以代码为准，未修改）
- LD 采样 offset 代码为 `(5,10,15,20,30)`（`ld.py:96`），但 `schema.py:339`（`schema_manifest()["frame"]["ld"]["semantics"]`，实机仍输出 `{0,20,40,60,80}`）、`schema.py:21-27`（v5/v6/v7 历史段）、`__init__.py:23-34`、`builder.py:30-32` 的文案仍写 `{0,20,40,60,80}`（含 `builder.py:40` 的 config 示例）；`ld.py` 内部注释（`:4-7, :22-30`）也残留旧口径。
