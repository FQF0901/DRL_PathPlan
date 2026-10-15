# Round-36 warp 层交付注记（L-D：`net/ego_warp.py` + 坐标门基线校准）

> 来源：lane L-D（GPT `7dcf6bd` §四.3 / §五 阶段 1）。本文件把 `.slim/`（gitignored）产物中的关键结论固化入库。
> 原始证据：`.slim/deepwork/p1_v2/round36/{d1_ego_warp_evidence,d2_coord_gate_baseline,d3_ld_report}.{json,md}`、`runs/round36/c_smoke/*`。

## 1. 交付物

- **`net/ego_warp.py`**（新增，~374 行，纯函数/无状态/**未接入 model.py 主路径**）：
  - `warp_pose_t0_to_ego_k` / `warp_pose_k`：LD 逆变换 `p'=R(−Δθ)(p−Δp)`、`ψ'=ψ−Δθ`——与 R33-B `DrivingModel._ego_warp_pose_k` **逐位等价**（同 op 序）；
  - `advance_od_pose` / `od_velocity_t0` / `assemble_pose_k(mode=off|w1|w2)`：`off` **逐位等于** rollout 现状；`w1/w2` = 整图 ego_k 系重表达（OD 无/有恒速先验）；
  - `AnchorBatch`（schema v1：xy/theta/vel/size/live/slot_id/kind + `to_pose()` + `validate_anchor` + `FIELD_MAP` ↔ `frame.pose`/`od_feat`/`od_anchor`/`ld_anchor` 契约映射）、`warp_ld_anchors`/`warp_od_anchors`；
  - `WarpHooks`（5 项 hook：warp 前 / 解析 warp 后 / 真实下一帧 / learned residual / 最终；**默认关 ⇒ 零张量操作、零 RNG**）。
- 单测 `d1_ego_warp_tests.py`：**8/8 PASS**——R33-B 逐位（16 例 max\|Δ\|=0）、off 恒等（含**模型内部 `st_gnn.pose_k` pre-hook 18/18**）、镜像等变（0.0）、物理一致性、默认关 bit-exact + RNG 不变；**966 tests passed**。

## 2. 关键发现（集成错误）：OD 先验速度语义

- warp 口径把 **t0 系当静态场景** ⇒ OD 恒速先验必须用 **t0 场景系对象速度 `v_OD = v_rel + v_ego`**（`od_velocity_t0`）。
- 现行 R33-B wiring 传的是**相对速度** ⇒ 系统性**重复扣除** `|Δp|`：同速前车 P90 **1.95 m → 0.43 m**（spec 口径）；恒速假设成立子集 **1.21 → 0.072 m**；静止先验对照 2.25 → 0.427。
- 结论：**接入结构化 WM 前必须改用 `od_velocity_t0`**；`off` 路径保持 `v_rel`（随体坐标语义）——两种口径不可混用。

## 3. 坐标门基线校准与阈值建议（锁前请 CG 裁）

| 项 | 读数（只读，10 shard + 262 录制，167 s） | 建议 |
|---|---|---|
| 静态 LD warp（250,549 对 / 13.8M 锚点） | 内部投影 **P90 4.5e-6 m**（pair 级 8e-6；覆盖热点 p05 0.842） | 门：同 target row + 内部投影 **P90 ≤0.01 m** + churn 护栏（pair P90 ≤0.05 m ∧ 覆盖 ≥0.80）；GPT 的 0.10 m 有 ~4 个数量级余量 |
| OD 同速前车（94,101 配对，k=1） | spec 口径 P90 **0.427 m**；**恒速子集 0.072 m**；wiring 口径 1.95 m；静止先验 2.25 m | 0.20 m 门仅在"**恒速假设成立子集**"成立 ⇒ 门收窄到该子集（锁 0.10–0.20 m）；全子集读数登记为 **WM 残差基线** |
| 端到端 k≥1 | k=1 P90 **4.5 m**（旧 ckpt 录制，含追踪/计划质量；k=5: 14.5 m） | 端到端 0.10 m 当前不可达 ⇒ **坐标门与位姿链输入误差分离登记** |

## 4. 集成注意

- `assemble_pose_k` 可**逐位替换** `net/model.py:1499-1505` 的内联分支（已由模型内部 pre-hook 门验证）；`net/__init__.py` 导出方式由集成 lane 决定；hooks 勿接入生产必经路径。
- ③"真实下一帧 anchor"部署侧暂无落盘（离线用 path_exec_v1 / replay 作源）；⑤当前 = ②（无 learned residual）；LD 覆盖缺口 ~12%（评分/残差对齐按内部投影或弧长口径）。
- 部署链读数来自 **K1 标定前**旧 ckpt 录制 ⇒ K1 落地后需重测（脚本已就绪，一条命令）。
