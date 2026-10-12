# R21 lane_semantic 契约（schema v7）

状态：R21-0 实现契约（2026-10-12）；实现出处 `env/obs/lane_semantic.py`（唯一）、
`env/obs/ld.py`（LD 点补充伴随数组）、`env/obs/schema.py`（机器可读清单）。
依据：`docs/chatgpt5p6sol_review.md` round-21 第二部分；`docs/deepseekv4p1_argue.md` §5.11–5.13。

## 1. 通道定义

`lane_semantic`：**1 槽 × 12 维**（float32；`mask=1 ⇔ 选出 target lane`）。
特征顺序（即 `LANE_SEMANTIC_FEATURE_NAMES`）：

| # | 名称 | 单位 / 口径 |
|---|---|---|
| 0 | `ego_to_target_lane_lateral_offset` | m；MetaDrive 车道系横向坐标（**正 = 车道方向右侧**，与 `lane_lateral_offset`/reward 同号） |
| 1 | `ego_heading_error_to_target_tangent` | rad；`wrap(ego.heading − lane.heading@投影点)`（正 = 自车相对切线逆时针偏左） |
| 2 | `target_lane_curvature_near` | 1/m；投影点前视 +10 m（clamp 车道末端），右转（顺时针）为负 |
| 3 | `target_lane_curvature_far` | 1/m；前视 +30 m（同上） |
| 4 | `left_same_direction_road_surface_margin` | m（≥0）；ego 到左侧最近不可行驶边界的距离（跨同 road 同向相邻车道累加；对向/缺失即停）。**仅输入特征：不查实线/护栏/route 权限/footprint 连通 ⇒ 非 legal corridor，禁作监督真值/安全 loss**（round-23 §2.3 降级） |
| 5 | `right_same_direction_road_surface_margin` | m（≥0）；右侧同理 |
| 6 | `target_lane_valid` | 0/1；1 ⇔ mask=1 且 ego 投影成功、全部特征有限 |
| 7 | `route_association_confidence` | [0,1]；按选择来源固定：route_current=1.0 / route_continuation=0.8 / route_next=0.6 / ego_fallback=0.25 / none=0 |
| 8 | `lane_change_required_left` | 0/1；ego 不在 target 且需左移（同 road 按 lane_id 大小；跨 road 按 ego 在 target 车道系的横向位置，\`|lat|\` > width/2 才置位） |
| 9 | `lane_change_required_right` | 0/1；对称 |
| 10 | `distance_to_lane_end` | m（≥0）；`target.length − clamp(投影 long)` |
| 11 | `distance_to_target_lane_end` | m；存在 next 链时 = `distance_to_lane_end`；否则 = 上限 `LARGE_DISTANCE_CAP_M=200`（round-23 改名：旧名 `distance_to_next_topology_change` 语义强于实际，**值不变**、不实现新地图搜索） |

**入口固定归一化（round-23 §2.1）**：网络入口逐元素除以 `LANE_SEMANTIC_NORMALIZATION` =
`(5, π, 0.1, 0.1, 25, 25, 1, 1, 1, 1, 200, 200)`（与上表顺序一一对应；常数单一出处
`env/obs/lane_semantic.py`，入 schema/模型快照 + 逐位单测；`net.semantic.LaneSemanticEncoder`
消费同一常量，不得在训练脚本里散落）。

回退语义：

- `mask=0`：没有任何 route 信息且 ego 无 lane（source=none）→ 12 维全 0；
- `mask=1` 但投影失败（圆弧 `local_coordinates` 抛错 / 非有限）→ 12 维全 0，
  `target_lane_valid=0`（供有效率审计，不崩通道）；
- 任何几何异常不抛异常（契约同 `ObservationChannel`）。

**禁 raw lane_id 数值输入（硬约束）**：通道 12 维与 LD 点补充均不含 lane id 数值；
lane 身份一律用 role / 同 road / 左右 / route 关联 / 局部 group 表达。单测
`tests/test_lane_semantic_contract.py::test_features_invariant_to_lane_id_relabeling`、
`tests/test_ld_group_topology.py::test_lane_id_relabeling_preserves_supplements` 锁定。

## 2. LD 点补充（逐槽伴随数组，不进 net 输入）

`env.obs.ld.LDChannel` 在 feats/mask 之外输出（`companions()`；shape `(16,)`，
随 ld 进 6 帧历史为 `<key>_hist`，缺帧填 -1/0）：

| 键 | dtype | 语义 |
|---|---|---|
| `ld_lane_role` | int64 | `-1` 无效槽；`0` other；`1` target（与 target 同车道对象）；`2` target_continuation（next 链上同 lane_id 优先、否则端点最近的续接车道）；`3` left_adjacent（同 road 且 lane_id 更小，0=最左）；`4` right_adjacent |
| `ld_offset_s_norm` | float32 | `clamp(s / lane.length, 0, 1)`；同车道内随 offset 严格递增 |
| `ld_group` | int64 | **仅观测内局部 group**：同 lane 对象 → 同 id（按槽序首次出现 0,1,2,…）；无效槽 -1；不暴露全局 lane_id |
| `ld_route_associated` | float32 | 1 ⇔ 该槽车道 ∈ `current_ref_lanes ∪ next_ref_lanes`（对象身份判定） |

## 3. target lane 选择

唯一纯函数 `env.obs.lane_semantic.select_target_lane`，契约见
`docs/r21_target_lane_selection.md`。**不得沿用"ego.lane 最高优先"**。

## 4. schema / 指纹

- `OBS_SCHEMA_VERSION = 7`；`schema_manifest()["schema_version"] == 7`；
  新增 `frame.lane_semantic`（1×12）、`lane_semantic_layout`、`ld_semantics_layout`；
- `obs_fingerprint()` 前缀 `v7-`（目录内容哈希）——观测语义变更后旧数据集/旧运行必然不匹配；
- `lane_semantic` 进 `builder.DEFAULT_CHANNELS`（列尾）；LD 点补充随 `ld` 通道输出。

## 5. 门（R21-0）

| 门 | 阈值 | 证据 |
|---|---|---|
| target lane 有效（`target_lane_valid`） | 总体 ≥99%；普通路 ≥99.5% | `runs/p1_v2/round21/r21_0_audit.json` |
| 横向偏移不切换 target（非真实 transition） | 0 次切换 | 单测 + 审计（±0.25/0.5/1.0 m） |
| 同 lane `offset_s` 严格递增 | 全组满足 | `tests/test_ld_group_topology.py` |
| 同组不混入他 lane | 全组满足 | 同上 |
| lane-relative 特征 vs MetaDrive 几何查询 | lat <0.05 m / heading <0.5° / curv 相对 <10% | `r21_0_audit.json` |
| 坐标 round-trip | 误差 <1e-6 m | 单测 + 审计 |
| CUDA 统一后端 | 记录设备；torch 计算一律 CUDA | `r21_0_audit.json::device` |

## 6. 旧数据集 / 旧 ckpt fail-fast 方案（R21-1 前置）

- **数据集**：`pipeline.stages.validate_bc_dataset` 在指纹与当前 `obs_fingerprint()` 不一致时
  **硬失败**（`SystemExit`），逃生门 `--allow-legacy-dataset` / `ALLOW_LEGACY_DATASET=1`
  （放行时 metrics 标注 `legacy_dataset=true`）。`pipeline.trainer.BCDataset.load` 保留
  运行期警告（供 R21-2 探针显式加载旧口径数据）。
- **ckpt**：R21 采集器要求显式 `--allow-legacy-obs-ckpt` 才加载**未记录 obs 指纹**的旧
  ckpt（如 pri512/M1），并在 manifest 记录 `ckpt.obs_fingerprint`（缺失 = `"<legacy>"`）
  与 `obs_fingerprint` 供追溯；新训练（R21-3+，未批准）必须在 ckpt meta 写入 obs 指纹并在
  eval 入口比对。
- **eval 入口**：R21-1 采集/回放只允许 `ckpt` 作为 C 段"learner-state"来源；不用于主模型
  训练。R21-2 的模型侧探针显式加载 M0 ckpt 并记录 `legacy_obs=true` 溯源标记。
