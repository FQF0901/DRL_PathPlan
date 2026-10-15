# 仓库手册（DRL_PathPlan）

- 用途：把"代码唯一真理"的**网络架构**、**训练流程**与**当前问题**固化成一册可核查工程手册；供 onboarding、code review、以及后续 lane 启动前对齐。
- 依据与口径：
  - 第 1/2 章主要依据 = E1 事实清单 `.slim/deepwork/p1_v2/round37/handbook_facts.md`（2026-10-15 代码核验版；本手册写作时**逐条复核了关键行号并实测 3 项数据**：参数量、测试收集数、DP-6 行数；漂移项状态见 §附录 A）；
  - 第 3 章为**文档依据**（台账 `docs/deepseekv4p1_argue.md` §0/§5.62–§5.66 + `docs/round37_report.md` 各节），不是本手册的代码核验结论；
  - "逐位" = 严格相等（`torch.equal` 级）；"默认关" = 不启用开关时与基线 HEAD **键集/前向/RNG 逐位一致**。
- 本手册核验命令：`tools/venv-python net/param_probe.py`（参数）；`tools/venv-python -m pytest tests/ --collect-only -q`（**1043 tests collected in 1.55 s**，实测 2026-10-15 R38 四波落地后）。
- 本手册为新增文件（写域仅此文件）；不修改任何既有文件。

---

## 第 1 章 网络架构

### 1.1 总览（主链，`net/model.py`）

```
obs(dict)
  │  mem_from_obs                                  net/mem.py:288
  ▼
MemBank（4 模态 × 6 帧；index 0 最老、-1 当前）
  │  MemEncoder.encode                            net/mem.py:688-759
  │    ego/od/others → TemporalAttention（掩码池化+帧龄）；LD **不做时序**（取当前帧）
  ▼
t0 latent：z_ego=embed_ego(ego_now) ─ z_od=embed_od(od_now,od_live) ─ z_ld=embed_ld(ld_now,ld_live)
  │                                               net/model.py:1304-1308
  ▼
PlanHead：[z_ego, mean(z_od), mean(z_ld), others_ctx, nav, signal] → fusion → MoE
  │      → latent (B,H) + ego_next (B,6)          net/model.py:1361-1365
  ▼
动作生成（单实体 `model.policy`；模式读出 = 派生 `path_head_mode()` net/model.py:979-987）
  ├─ 旧 PolicyHead（默认关路径）：交叉注意力 K=1 查询 × 令牌集 T=39   net/policy.py:119
  └─ StructuredPolicyHead（`set_path_head(True)` 后注册为 `policy.structured_action`；唯一动作生成器）
      每步调用 a_k=(ds_k, κ_near_k·ds_k)           net/path_head.py:1021
  ▼
_rollout：6 步 × dt=0.5 s                          net/model.py:1507（步循环 :1630 起）
  每步：(a) pose 链 arc_step/compose_pose（不 detach）→ (b) 图位姿 assemble_pose_k
        → (c) st_gnn 单步（latent 转移 + t0 锚物理解码）→ (d) z_*_next.detach() → 下一步
  ▼
plan (B,6,2) / traj_xy (B,6,2)   ──►  LQR 参考（`plan[:,0]==action_mu` 硬契约）/ 环境执行
```

### 1.2 `net/` 模块表（15 个 `.py` = `__init__` + 14 个模块）

| 模块 | 角色 | 关键类/函数（file:line） |
|---|---|---|
| `__init__.py` | 公共 API 再导出（`DrivingModel`/`StructuredPolicyHead`/`WarpHooks`/`step_geometry` 等 41 符号）；导入无副作用 | `net/__init__.py:43-85`（`__all__`） |
| `model.py`（2163 行） | `DrivingModel`：前向主链/开关/`encode`/`_rollout`；运动学 `arc_step`(:223)/`compose_pose`(:238)/`interpolate_actions`(:247)；A4 世界系 nav 重建 `world_state_from_obs`(:318)/`rebuild_nav_from_world`(:337)/`advance_pose_world`(:365)/`advance_signal`(:378)；R38-D1 探针装载 `set_probes`(:652) | `forward:1913`、`encode:1260`、`_rollout:1507` |
| `encoders.py` | `ObsEncoders`（逐通道线性投影+类型/轨道 id 嵌入+LayerNorm）、`FrameEncoding`；维度常量 `H=96`(:38) `EGO_DIM=8`(:39) `OD_DIM=9`(:40) `LD_DIM=7`(:41) `NAV_DIM=11`(:42) `SIGNAL_DIM=4`(:43) `OD_SLOTS=LD_SLOTS=16`(:44-45) `HISTORY_FRAMES=6`(:46) `DEFAULT_OTHERS_DIM=33`(:56) `OTHERS_LEGACY_DIM=28`(:59) `ID_BUCKETS=64`(:65) `NUM_NODE_TYPES=5`(:74，v8 删 lane/ttc 后 7→5) | `net/encoders.py:77-204` |
| `mem.py` | 4 模态 mem 输入契约（只读）；`MemBank`(:204)+拷贝隔离 `clone/shift_*`(:232-286)；`mem_from_obs`(:288，旧 28→33 others 重排与缺键回退)；世界系 nav 纯函数族(:437-646)；`EncodedMem`(:648-675)；`MemEncoder.encode`(:688-759) | LD 取当前帧 `:705`；`od_now/ld_now=*_hist[:,-1]` `:711-712` |
| `temporal.py` | `TemporalAttention`：6 帧掩码注意力池化 + 帧龄嵌入（**GRU 已被 v2 弃用**，docstring 说明理由） | `net/temporal.py:1-27`(docstring)、`:41`(类) |
| `spatial.py` | 2 层消息传递（adjacency：OD↔ego / LD↔ego / LD 槽位相邻 / OD kNN 对称化）；`LdSetAttention`(:61)、`LdAggregate`(:100) | `build_adjacency:272`、`build_edge_features:313` |
| `st_gnn.py` | `SpatioTemporalGNN`：latent 自回归转移 + t0 锚定物理小解码；`OD_PRED_DIM=5`/`LD_PRED_DIM=4`(:44-45)；`od_state_from_features`(:164)、`ld_state_from_features`(:184) | 类`:83`；单步 `forward:236-322` |
| `moe.py` | `MoEBlock`：primary 常开 + 8 experts(top-2 softmax 路由)；`top_k_softmax`(:52)、`switch_load_balance_aux`(:63)；`moe_enabled` 初值 True(:115)，关闭时输出严格=primary(:171) | `net/moe.py:80` |
| `plan_head.py` | `PlanHead` = fusion MLP(:51) + MoE(:57) + ego_next 头(:66)；**K-anchor 分支已随 R38-P2a 迁 `tools/legacy/k_anchor/`** | `net/plan_head.py:37-89` |
| `policy.py` | `PolicyHead`(:119)/`ValueHead`(:203) 交叉注意力头（K=1 查询 × 令牌集）；动作界 `ACTION_LOW=(0,-0.6)`/`ACTION_HIGH=(10,0.6)`(:49-50)；`ATTN_HEADS=4`/`ATTN_LAYERS=1`(:55-56) | docstring `net/policy.py:1-44` |
| `path_head.py` | `StructuredPolicyHead`(:1021)（唯一动作生成器的结构化形式；R38-P2b 注册于 `model.policy.structured_action`，键面 `policy.structured_action.*`；旧别名 `PathExecutionHead`/一次性 `compose_path_plan` 已删，快照 `tools/legacy/oneshot_path.py`）；几何/组合纯函数 `anchor_reference`(:462)/`replay_plan`(:569)/`compose_step_action`(:598)/`step_geometry`(:636)；obs 适配 `anchor_from_obs`(:778)/`target_lane_valid_from_obs`(:802)/`long_exog_from_obs`(:864，dynamic 两列缺键 fail-fast)；常量 `NUM_STEPS=6`/`DT=0.5`/`LAT_HORIZON_M=35.0`(:166-174)+数据驱动标定(:352-437)；输出分组 `output_groups`(:1280) | `net/path_head.py` |
| `semantic.py` | `LaneSemanticEncoder`(:73) + `SemanticWiring`(role-aware 分组读出,:105)；监督目标名 `SEMANTIC_AUX_TARGETS`(:67) | `net/semantic.py` |
| `ego_warp.py` | R36/R37 坐标层纯函数+hook：`WARP_MODES=("off","analytic")`(:103)（**R38-P2b：legacy w1/w2 已退役**，逐位 oracle `tests/fixtures/r38_p2b_warp/`、快照 `tools/legacy/legacy_ego_warp.py`）、`ANCHOR_SCHEMA_VERSION`(:99)、`od_velocity_t0`(:180)、`masked_pose_error`(:190)、`assemble_pose_k`(:202)、`AnchorBatch`(:255)、`WarpHooks`(:363) | `net/ego_warp.py` |
| `probes.py` | R38-D1 §四 七类强制 probes/hooks 统一容器（**默认全关**）：`StateSourceHooks`(:163)/`PolicyHooks`(:212)/`WorldModelHooks`(:277)/`MemoryHooks`(:320)/`ChainHooks`(:444)；统一出口 `ProbeRecorder`(:645)；⑦ 梯度账本 `grad_ledger`(:943)/`state_dict_hash`(:90)；默认关 ⇒ 前向逐位、零 RNG、不写回输入（`net/probes.py:1-100` 契约） | `net/model.py:652-663`（装载） |
| `param_probe.py` | 参数预算 CLI 探针（实例化模型打印各模块参数量并断言 ≤1.5M） | `PARAM_BUDGET=1_500_000` `net/param_probe.py:30`（断言 `:114`） |

### 1.3 obs 键全表（builder 实际输出；`env/obs/builder.py:3-24`）

当前帧（全部 float32，除标注）：

| 键 | 形状 | 来源/语义 |
|---|---|---|
| `ego` / `ego_mask` | (1,8) / (1,) | `[v, a_long, a_lat, yaw_rate, steer, curvature, reserved0, reserved1]`；**reserved0/1 = 上一策略步动作 (ds,dθ)**（由 trainer/collect 侧写 `env.prev_policy_action`）——`env/obs/ego.py:1-17` |
| `od` / `od_mask` / `od_id` / `od_presence` | (16,9) / (16,) / (16,) int64 / (16,) | OD 槽位固定 = track id；语义见 §1.4——`env/obs/od.py:1-46` |
| `ld` / `ld_mask` | (16,7) / (16,) | LD 7 维；`LD_OFFSETS_M=(5,10,15,20,30)`——`env/obs/ld.py:38-40, 96` |
| `nav` / `nav_mask` | (1,11) / (1,) | 2 checkpoint(x,y) + 6 command one-hot + route_completion——`net/encoders.py:15` |
| `signal` / `signal_mask` | (1,4) / (1,) | 恒"未知/无灯"占位——`net/encoders.py:16` |
| `others` / `others_mask` | (1,21+K) / (1,) | nav(11)+speed_limit(1 归一化/30 m/s)+signal(4 占位)+static(5)+road_class one-hot(K=12→33)——`env/obs/others.py:1-47` |
| `ego_world` / `ego_world_mask` | (1,3) / (1,) | t0 世界系位姿——`env/obs/world.py` |
| `route_world` / `route_world_mask` | (64,2) / (64,) | 世界系路线折线；上限 64 点、不足重复末点+mask=0——`env/obs/world.py:19, 42, 103-118` |
| `lane_semantic` / `lane_semantic_mask` | (1,12) / (1,) | v7（R21）target lane 相对语义（横向/航向/曲率/margin/target 有效性/route 关联/变道需求/d2end 等；固定归一化除数）——`env/obs/lane_semantic.py:110-147` |

历史键（FIFO 6 帧 @0.5 s，SE(2) 对齐到当前帧）：

| 键 | 形状 | 备注 |
|---|---|---|
| `ego_hist`(+mask)、`others_hist`(+mask) | (6,1,8)、(6,1,21+K) | 默认历史通道 `("ego","others","od","ld")`——`env/obs/memory.py:34-35, 164-224` |
| `od_hist`(+mask)、`od_id_hist`、`od_presence_hist` | (6,16,9)、(6,16) int64、(6,16) | 跨帧身份一致、不做 SE(2) 变换 |
| `ld_hist`(+mask) | (6,16,7) | 伴随数组随 ld 进历史：`ld_lane_role`(int64)/`ld_offset_s_norm`/`ld_group`(int64)/`ld_route_associated` → `<key>_hist (6,16)`（缺帧填 -1/0）——`env/obs/builder.py:22-24`、`env/obs/ld.py:47-54` |
| `hist_valid` | (6,) | warmup 补位帧=0 |

- schema：`OBS_SCHEMA_VERSION = 7`；`obs_fingerprint = "v7-<md5 12hex>"`（对 `env/obs/*.py` 全量哈希）——`env/obs/__init__.py:45-57`；变更史 v2–v7 见 `:8-34`。
- ✅ **外源几何键（R38-Q4A 已补生产者，`9577078`，2026-10-15）**：t0 段 8 列（`path_speed` + 7 键，`LONG_EXOG_OBS_KEYS` `net/path_head.py:256-265`）由 `env/obs/path_field.py` 生产、`env/obs/builder.py:189-190` 挂载，并同步产出与值同源的 `<key>__present` 掩码（公式单一出处 `env/obs/long_exog.py`）。头侧消费 `net/path_head.py:864-963`（`long_exog_from_obs`）：**dynamic_distance/dynamic_ttc 缺键 = fail-fast**（`LONG_EXOG_STRICT_COLUMNS=(4,5)`，:268-271）；static 缺键 → 0+present=0（诊断列）；margin/nav/d2end 缺键仍回退 `lane_semantic`/`nav`。Q4-A 前 eval 装配缺三键 ⇒ 回退 0（`dynamic_ttc=0` = 分布外"立即碰撞"语义，M3 占装配错配效应 77%）——该行为已被取代；离线数据集（H2/I1）与 `.slim/.../m7_fill.py`（re-export `env.obs.long_exog`）为同公式消费方；`--long-exog-fill neutral` 缓解保留待 P3 固化形态。
- ⚠️ **`lane_anchor_xy/lane_anchor_mask` 同样不在 env obs**：结构化头启用后缺键直接 `ValueError`（不静默回退）——`net/path_head.py:778-799`；注入器 = R36 lane C（默认关）`.slim/deepwork/p1_v2/round36/c_lane_anchor.py:1-6, 126-254`。
- 结构化头的**可用回退**：margin(左/右)/d2end/valid ← `lane_semantic` 列 4/5/11/6；speed ← `obs['ego']` 第 0 维；nav_progress ← `obs['nav']` 末维——`net/path_head.py:802-821, 842-922`。

### 1.4 OD/LD 槽语义

| 通道 | 维度 | 语义 | 出处 |
|---|---|---|---|
| OD | 9 | `[dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]`；**vx/vy 为相对速度**（对象速度 − ego 速度） | `env/obs/od.py:3-9` |
| OD mask 双轨 | — | `od_mask` = 槽位已分配（身份有效）；`od_presence` = 本帧盒内被观测（特征新鲜）；**下游必须都看** | `env/obs/od.py:37-44` |
| LD | 7 | `[dx, dy, heading_rel, curvature, speed_limit, left_line_type_id, right_line_type_id]` | `env/obs/ld.py:38-40` |

- mem 侧当前帧：`od_now = mem.od[:, -1]`、`ld_now = mem.ld[:, -1]`；`od_live = od_mask∧od_presence`（末帧）、`ld_live = ld_mask[:,-1]`——`net/mem.py:11-15, 708-712`。
- OD 槽驱逐/释放：槽满按 `(min(TTC,5s), 距离)` 紧迫度驱逐"最长未出现且已出盒"槽；超过 `release_after_s=1.0 s` 释放——`env/obs/od.py:14-41, 72-73`。

### 1.5 WM 与 rollout

- `st_gnn` 单步返回：`z_ego_next (B,H) / z_od_next (B,S,H) / z_ld_next (B,L,H) / od_pred (B,S,5) / ld_pred (B,L,4) / presence_logit (B,S) / entry_logit (B,S)`——`net/st_gnn.py:264-266, 322`。
- 物理解码 = **t0 锚先验 + 零初始化残差头**；OD 先验按 `k·dt` 恒速外推、LD 先验静止——`net/st_gnn.py:290-319`（OD prior `:298-308`、LD prior `:309-317`）；预测空间 OD 5 维 `[dx,dy,vx,vy,heading]`(`:165`)、LD 4 维 `[dx,dy,heading,curvature]`(`:185`)。
- 步数/步长：`DrivingModel` 默认 `wm_steps=6`、`wm_dt=0.5`（`net/model.py:409-410`）；config：`world_model.rollout_steps: 6`（`config/model.yaml:49`）、`policy.rollout_steps: 6`（`:54`）、`action.dt: 0.5`（`:60`）；`env.timing.policy_dt 0.5 / mpc_dt 0.1`（`config/env.yaml:16-19`）。
- rollout 循环语义（每步 k=1..6）：位姿链（**不 detach**）→ A4 世界系位姿推进+逐步重建 nav/signal → 图位姿 `pose_k` 经 `assemble_pose_k` → `st_gnn` 单步 → **状态链切断** `z_*_next.detach()` 后作下一步输入——`net/model.py:1626-1819`（detach `:1720-1721`）。
- **detach 契约（固定不可配）**：来自真实 obs 的 t0 状态不 detach（检测任务梯度直达编码器）；rollout 合成后续状态进入下一步前 detach；action/pose 链保持可微——`net/model.py:20-23, 1585`。
- **`plan[:,0] == action_mu` 硬契约**：输出契约（评测/跟踪器/PPO 依赖）`net/model.py:131-137`；运行时断言（structured+rollout，eval 违反 fail-fast；训练只落 `contract_*` 键）`net/model.py:2012-2047`。
- cheap path：`forward(rollout=False, world_model=False)` 不跑 st_gnn/rollout，`action_mu/action_logstd/value` 与完整前向**逐位一致**（PPO 收集用）——`net/model.py:139-141, 1873-1878`；`world_model=True` 必须 `rollout=True`（否则 `ValueError`，`:1981-1982`）；`wm_detach` 形参保留但**恒为 no-op**（v8 latent 状态链 detach 固定，`:152-153`）。
- **warp 两种速度语义（不可混用）**：`off`（默认）把 t0 系当随体坐标用**相对速度** `od_velocity = enc.od_now[...,2:4]*od_live`（`net/model.py:1595`）；`analytic` 把 t0 系当静态场景用 `od_velocity_t0`（`v_OD = v_rel + v_ego`，`net/model.py:1652-1660`、`net/ego_warp.py:180`）。混用后果 = 重复扣自车位移（实测同速前车 P90 1.95 m → 0.43 m，E1 §2）。`assemble_pose_k(mode)`：`off`=[Δp, od_t0+k·dt·v_t0, ld_t0]（t0 帧）/`analytic`=（R37 M1 解析口径，op 序 = 旧 `w2`）；**R38-P2b：legacy `w1`/`w2` 已退役**（`WARP_MODES=("off","analytic")`；逐位 oracle `tests/fixtures/r38_p2b_warp/`）；warp 只改 st_gnn 输入图位姿，`traj_xy/plan` 仍是 t0 帧累积位姿口径——`net/model.py:1665`、`net/ego_warp.py:202-254`。
- `step_geometry` 语义（t0 静态地图 + 想象位姿 → 想象步几何；仅 `dynamic_mode="hold"`，否则 `ValueError`）——`net/path_head.py:664-706`：**重算**车道锚刚体逆变换 + 垂足重锚（`:729-748`）、左右 margin = t0 路面宽 W/2 ∓ 新横向偏移（`:750-759`）、`static_distance`(列 3) 与 `distance_to_target_lane_end`(列 7) 按累计进度收缩（`:760-762`）；**保持** speed(0)/dynamic_distance(4)/dynamic_ttc(5)/nav_progress(6)（`:763`、`:689-690`）。
- R37 五项 hook（`WarpHooks.capture`）读数键：`warp_prior_err / warp_residual_contrib / warp_final_err (B,T)` [m] + `warp_readout_valid`；数据来自 obs `warp_anchor_actual_next (B,T,N,3)` / `warp_anchor_residual_learned`（缺省 = n/a/0）——`net/model.py:97-108, 1451-1460, 1851-1860`。
- 计划曲率监控（只读派生）：`plan_kappa = plan[...,1]/plan[...,0].clamp_min(1e-6)` 等——`net/model.py:2057-2067`。

### 1.6 关键开关（全部默认关/默认口径）与工程约定

| 开关 | 取值（默认在前） | 位置 |
|---|---|---|
| `set_ld_mode` | `baseline` / `no_chain` / `set_attn` / `attn` / `attn_no_chain` / `semantic` / `semantic_no_chain` | `net/model.py:519-582` |
| `set_rollout_semantic` | `full` / `t0_only` / `static_role_only` | `net/model.py:583-600` |
| `set_ego_warp_mode`（R37 主路径） | `off` / `analytic`（单状态机；**R38-P2b 删除 legacy `set_ego_warp` 与互斥守卫**） | `net/model.py:601-626` |
| `set_warp_hooks` / `set_warp_readout` | `None`=关 / `True`=开（默认关） | `net/model.py:628-639 / 641-650` |
| `set_probes(recorder)` | 默认 `None`=关（R38-D1 只读探针；前向逐位/零 RNG/不写回输入） | `net/model.py:652-663` |
| `set_moe(enabled,...)` | 推理默认开；off ⇒ 输出严格=primary | `net/model.py:820-829`（委托 `net/moe.py:120-123`） |
| `set_action_ego` / `set_ego_phys` | 惰性构造；ego_phys 仅训练/评测消费、不进 forward | `net/model.py:831-840 / 841-862` |
| `set_path_head(enabled=True)` | 默认关；惰性构造 + 构造期保存/恢复 CPU RNG；**R38-P2b：头注册为 `policy.structured_action`（键面 `policy.structured_action.*`）** | `net/model.py:874-907` |
| `set_path_head_ablation(mode)` | `policy` / `structured`（**R38-P1/P2b：降为两档 legacy 诊断入口；`oneshot` 写入即 `ValueError`**） | `net/model.py:909-931` |
| `set_policy_diag()` | **R38-P2b：fail-fast 桩**（指路离线工具 `tools/diagnostics/legacy_policy_diag.py`） | `net/model.py:945-957` |
| `set_contract_asserts` | **默认开** | `net/model.py:959-977` |
| `set_target_lane_dropout` | 默认 0 | `net/model.py:997-1009` |

> K-anchor 已退役（R38-P2a）：`plan_anchor.enabled`/`num_anchors` 配置与模型面已删除，实现整体迁 `tools/legacy/k_anchor/`（`net/plan_head.py` 不再含锚分支）。
> `forward_mode` 枚举（H 组：替换 `rollout×world_model` 隐式布尔）：**P3 进行中**——当前全仓 0 命中（`c5_reach_scan` 复核），落地时须附旧布尔组合 → 枚举的逐位对照表。

工程约定（默认关 + 逐位兼容 + RNG 保存恢复）：

- 默认关 ⇒ 键集/前向/RNG 与 HEAD 逐位一致（R36 K1 Gate 0 自证 `.slim/deepwork/p1_v2/round36/k1_contract_gate.py`；R37 M1 `off` 经 `assemble_pose_k(mode="off")` 逐位复现旧内联分支）——`net/model.py:50-55, 88-110`、`net/ego_warp.py:5-13`。
- 惰性构造不扰动既有初始化抽样序列（构造期 save/restore CPU RNG）——`net/model.py:874-907`；`path_head_parameter_names()` 默认空元组（`:1011-1022`）。
- 新参数默认严格 0 键差异：`semantic_wiring/beta/target_proj`（`:511-517, 580-593`）、`ld_set_attn/ld_aggregator`（`:566-579`）、`st_gnn.action_ego`（`net/st_gnn.py:26-33, 128-135`）、`ego_phys`（`net/model.py:456-462`）。
- 各新头初始化为恒等/零残差：MoE experts 零初始化（`net/moe.py:86`）、latent 转移头零初始化恒等（`net/st_gnn.py:51-61`）、path_head lateral/longitudinal 末层零初始化（`net/path_head.py:1140-1148`）。**K-anchor 选择头（原 `net/plan_head.py:90-107`）已随 R38-P2a 退役**（迁 `tools/legacy/k_anchor/`）。
- 结构化 cheap path：无 WM ⇒ 只给 `plan_aux` 预览并显式标 `plan_from_aux=1`；**生产 LQR 参考必须 `rollout=True`**——`net/model.py:2002-2019`（`plan_from_aux` 标记 `:2007`/`:2043`）。
- `path_head_mode()`（派生读出，R38-P2b）语义：未启用/`policy` → 旧 `PolicyHead` 为动作源；`structured` → 旧头默认**不前向**（输出 dict 无 `policy_action_mu/action_logstd`），唯一动作源 = `policy.structured_action`；`action_generator_names()` 返回 `("policy.structured_action",)`——`net/model.py:979-995`。

### 1.7 参数量与 `config/model.yaml` 关键键

实测（`tools/venv-python net/param_probe.py`，本手册复核；H=128、K-anchor 关、path_head 关、MoE 常开）：

| 配置 | 总参数 | 备注 |
|---|---|---|
| **config 口径（生产默认）** | **1,278,879** | encoders 19,072 / mem_encoder 136,128 / plan_head 总 531,247（含 moe 407,401）/ st_gnn 384,011 / policy 108,452 / value 99,969 |
| `set_path_head(True)` | 1,326,248 | path_head 47,369 |
| `set_action_ego(True)` / `set_ego_phys(True)` | 1,312,287 / 1,295,778 | 惰性支路 |
| `set_path_head(True)` + `set_ld_mode("semantic")` | **1,510,570** | ⚠️ **超出 1.5M 预算**（`net/param_probe.py:30` 断言）；semantic_wiring 167,808 |

`config/model.yaml` 关键键：`hidden_dim: 128`(:2)；spatial 2 层(:12-15)；MoE primary 768 / experts 8×76 / router 384 / top_k 2(:17-31)；world_model 6 步(:46-50)；policy attn 4 头 1 层 / trunk 160(:52-62)；value 256(:64-65)；**`plan_anchor` 块已随 R38-P2a 删除**（K-anchor 实现迁 `tools/legacy/k_anchor/`；现 `config/model.yaml` 共 65 行）。`build_model` 恒显式传 128（`pipeline/stages.py:155-175`，`section.get("hidden_dim", 128)`）——`net/encoders.py:38` 的 `H=96` 仅为类级默认。

⚠️ 两处 config 漂移（详见附录 A）：`temporal.type: gru`(:6-9) 与实际 `TemporalAttention` 不符；`param_budget_estimate: 585000`(:3-4) 为 H=96 时期历史估算（注释自认），以实测 1,278,879 为准。

---

## 第 2 章 训练流程

### 2.1 通用训练栈（`pipeline/`）

- 入口：`tools/train.py`（分阶段 CLI `--stage A/B/C`、`--phase3 <dagger_dir>`、`--phase3-loop`、`--phase3-chain`、`--ckpt/--spec/--bc-dir/--out`；未识别参数透传）——`tools/train.py:48-74, 77-139`；训练编排全在 `pipeline.stages.main`（`pipeline/stages.py:5945-5959`）。零参入口 `tools/train.sh`（A→B（+phase3 链））——`tools/train.sh:1-20`。
- 模型构造：`build_model` 读 `hidden_dim/moe/policy/value/spatial/world_model`（**K-anchor/`plan_anchor` 已随 R38-P2a 删除**）——`pipeline/stages.py:155-175`。

| 阶段 | 编排函数 | 数据/目标 | 关键口径 |
|---|---|---|---|
| **A（WM 教师强制）** | `run_stage_a` `pipeline/stages.py:1733` | ego 条件 = 专家 GT 动作序列；目标 = `(episode, step+5k)` 查表未来 OD/LD（对齐 t0 + id 匹配 + `wm_valid` 门控）`stages.py:5-19` | 损失：latent consistency **1.0** 主 + 物理 od 0.1 / ld 0.02 + plan head ego_next 0.1 + presence/entry BCE 0.1（`config/train.yaml:80-86`；实现 `pipeline/trainer.py:2391/2262/2323/2820`）；可训练集 = `[encoders, mem_encoder, plan_head, st_gnn]`，**MoE 关**（`config/train.yaml:86`、`stages.py:1758`）；产物 `<out>/final.pt`（`stages.py:2727-2730`） |
| **B（planner BC）** | `run_stage_b` `pipeline/stages.py:2962` | 两段式：primary 段（`primary_phase_split=0.5`，MoE **关**）→ `primary.pt` → worst-50% 行权重挖掘 → specific 段（MoE **开**） | 冻结清单 `_PRIMARY_PHASE_FREEZE`（st_gnn/value/experts/router/ego_phys）`stages.py:2752-2763`；`_SPECIFIC_PHASE_FREEZE`（+encoders/mem_encoder/fusion/norm/ego_next/primary/policy+K-anchor 锚头前缀）`:2774-2786`；行权重 worst 1.0/mild 0.1、动作逐维 `[1.0, 69.4]`（`config/train.yaml:177-186`）；负载均衡 aux α=0.01（`:176`）；主损失=首步动作+6 点轨迹辅助（WM 冻结+detach，`:174-175`） |
| **phase3（迭代恢复）** | `run_stage_b_phase3` `pipeline/stages.py:4193` | 数据 = `--phase3 <dagger_dir>` 当轮失败窗口行（无 worst/mild 权重、无 mining）`:4196-4198` | 冻结配方 `specific_only`（默认）/`trunk_only`/`recovery`/`recovery_experts`/`all`（`config/train.yaml:138-141`、`stages.py:4262-4295`）；冻结时上游监督自动降 0（`PHASE3_WM_LOSS_KEYS`，`pipeline/trainer.py:4951, 5058-5082`）；损失默认系数 action 1.0 / action_chain 0.2 / bias_calib 0 / ego_next 0.1 / od 0.005 / ld 0.002 / presence 0.1 / entry 0.1 / traj_aux 0 / anchor_ce/wta 0 / load_balance 0.01（`config/train.yaml:116-137`）；LR 分组缩放 base 0.25 / specific 0.5（`:113-115`）；epoch 级 clean150 守护可选（fail-closed 抛 `Phase3GuardCollapse`，`stages.py:4216-4219, 4107-4111`）；全自动循环 `pipeline/phase3_loop.py:1-3` |
| **C（PPO RL，实验）** | `run_stage_c` `pipeline/stages.py:5129` | `PPOTrainer` 主循环 `pipeline/trainer.py:7273`（`collect_rollout :7660` / `update :7898` / `train :8163`） | KL 锚 = 阶段 B 快照（coef 0.05→0 线性衰减）、primary lr ×0.1、`trainable_scope=design`、`spec_rotation=episode`、`max_episode_steps=200`（`config/train.yaml:196-233`）；GAE λ=0.95 / minibatch 1024 / `target_kl=null` / adv_norm=global（`:14-45`）；cheap path（`plan_reference=repeat_action` ⇒ 收集走 `rollout=False`，`stages.py:26, 5192`）；周期 ckpt `ckpt_u<NNN>.pt`（每 25 update，`stages.py:541-543`） |

- BC 主循环 `pretrain_bc`（`pipeline/trainer.py:3839`）：权重感知 `Σw·x/Σw`(:3853-3856)、动作目标=首步 `targets["action"][:,0,:]`(:3851-3852)、微批梯度累积(:3870-3872)、`apply_freeze_prefixes`→`build_optimizer`(:3911-3912)、留出集每 epoch 末 `evaluate_bc`(:4545)。phase3 循环 `pretrain_bc_phase3`(:5630)、损失项组装 `_phase3_loss_terms`(:5380)、`evaluate_bc_phase3`(:6344)。
- 优化器分组 `build_optimizer`(:506-537)：参数名含 `primary` → `lr×primary_lr_scale`；`value.*` 仅当 `value_lr_scale≠1.0` 时单列；组序 `[other, value?, primary]`；Adam。phase3 LR 分组（`:540-630`）、recovery 分组（`:631-699`）。
- 冻结/allowlist：`apply_freeze_prefixes`(:1126)、`apply_trainable_allowlist`(:1157)、`trainable_param_groups`(:1188)。
- ckpt/resume：`save_checkpoint` 固定键 `{model, meta, optimizer, epoch, val_metrics, rng_state, config_hash}`(:8350-8381)；`load_checkpoint` 严格失败自动回退 strict=False 并记 missing/unexpected(:8384-8407)；**resume `epoch` = 已完成全局 epoch 数，从 `epoch+1` 继续**(:8363-8365、`stages.py:734-758`)；RNG `capture/restore_rng_state`(:8212-8273)；`config_snapshot_hash`(:8206)。周期 ckpt `ckpt_epoch{N:03d}.pt`（`stages.py:474-497`；默认 N=5 `config/train.yaml:46-48`）；阶段末恒写 `final.pt`（A `stages.py:2727`/B `:3514`/C `:5594`）；B 另有 `primary.pt`(`:3370`)。
- 日志/监控：`pipeline/monitoring.py`（tensorboard+CSV，hook 式非侵入；开关 `config/train.yaml:235-239`）；每阶段 `metrics.json`；run 目录布局单一来源 `pipeline/run_paths.py:1-5`；`tools/train.py:112-125` 写 manifest。
- 数据侧：`BCDataset`（npz 契约+meta，`pipeline/trainer.py:3158-3428`；`load :3240`、`build_obs_batch :3348`）与预物化 `MaterializedBCDataset`(:3430-3505，等价性测试 `tests/test_fast_data_path.py`)；语义标签守卫 `pipeline/semantic_labels.py:1-3`（B 行标签 NPZ 入口**硬拒 v1**，唯一有效来源 = `round26/baseline_labels.npz`）；历史/未来窗口精确查表单一真源 `pipeline/frames.py:1-3`；回放缓冲 + GAE `pipeline/buffer.py:1-2`。
- ⚠️ 评测 tracker 两套默认：`pipeline.eval_runner` argparse 默认 `--tracker exact`（只衡量规划轨迹本身）`pipeline/eval_runner.py:1716-1718`；包装 `tools/test.sh` 默认显式传 `TRACKER=lqr`（含控制器跟踪误差）`tools/test.sh:5`。默认 `--workers 2 / --max-steps 1000`（100 s @0.1 s）`pipeline/eval_runner.py:119-120`；产物 `<out>/<run_name>/{metrics.json,episodes.csv,monitor/}` `:1914-1919`。

### 2.2 结构化头训练（当前主线，R36 J1/J2）

- **数据链**：`expert_bc.npz`（源 `datasets/BTC20261007-2202_expert5k_v8`）→ H2 标签规格 `h2_data_spec.py::label_row`(:339) + E3 `sample_lane_chain`(:1177) → I1 抽取 `i1_extract_dataset.py`（DP-6）→ `datasets/path_exec_v1/`（manifest.json + `shard_0000..0009.npz`，**rows = 360,497**，本手册实测 manifest）→ `j1_train.py`。
- `j1_train.py`（`.slim/deepwork/p1_v2/round36/j1_train.py`）：
  - 默认：`--data datasets/path_exec_v1`(:121)、初始化 `runs/p1_v2/round22/M0/ckpt_epoch002.pt`(:118)、模型配置 `runs/p1_v2/round22/M0/model.snapshot.yaml`(:119)、联合装载 `--base-dataset datasets/BTC20261007-2202_expert5k_v8`(:120)、冒烟 `--samples runs/round36/h2_samples.npz`(:122)。
  - **allowlist = `("policy.structured_action.", "plan_head.fusion.2.")` = 18 键**（R38-P2b 收编后前缀；:126-127；与 `net/path_head.py:283-286` 的 `TRAIN_ALLOWLIST_PREFIXES` 在 main() fail-fast 核对 :1434-1439）；`--head-only` 收窄为 `("policy.structured_action.",)`(:128-129, :1466)；`freeze_to_allowlist`(:851)；`--fusion-lr-scale` 分组 lr（:1476-1479；CLI 默认 :1992）。
  - 损失项与默认系数（argparse :1966-1992）：主损失 `--ds-coef 1.0`、`--kappa-coef 1.0`；辅助 `--aux-ds-coef 0.5`、`--aux-progress-coef 0.5`、`--aux-kappa-coef 0.5`、`--aux-path-coef 0.0`（关）；J2 计划级锚系残差 `--plan-res-coef 0.0`（关）+ `--plan-res-terminal-weight 3.0`；`--terminal-coef 0.0`（关）；`--target-lane-dropout 0.2`；`--lr 1e-3`、`--bs 64`、`--epochs 4`。
  - 链口径 `--chain imagination`（默认，训练/部署同构）vs `step0`（诊断）——:125, 79-81。
  - **契约门**：生成器数=1 / 6 头调用 / 5 WM 推进 / 辅助不进 LQR + 梯度门 + 结构探针（:27-29, 1735-1934）；产物 `ckpt_epoch{NNN}.pt` + `ckpt_{tag}.pt` + `metrics_{tag}.json` + `train_log_{tag}.txt`(:71-72)。
- **离线门 `j1_eval.py`**（只做离线；闭环走 G6 草案）：G1 车道保持（footprint 越界率 ≤1 m 连续插值 + 9 点 footprint；曲率界 `|κ| ≤ 专家 P95 × 1.25`；饱和率）`j1_eval.py:22-24`；G2 给定变道（方向正确率/回正率末点 `|lat|≤0.5 m`/`repeat_change_rate` :322/镜像门 :487）；G3 merge（:28）；G4 收费站（:30）；G5 纵向单调性 4 项响应（:379）；**专家地板**注释（末有效点 `|lat_err|≤0.5 m` 参照，专家 3 s 未收敛行同计 fail，:326）；自证 S1–S6（docstring :37-44）。

### 2.3 评测/部署链

- `.slim/deepwork/p1_v2/a3/eval_a3.py`：A3 推理适配器（obs/history→动作→确定性尾/LQR→0.5 s 重规划）。基础 flag（`--ckpt/--config/--spec/--ids/--tail {repeat,scale,chain}/--tracker {lqr,exact}/--tail-scale/--oracle-action/--ld-mode/--rollout-semantic/--moe-off/--canary-ckpt/--path-head {off,structured}/--lane-anchor {off,inject}/--diagnostic-oneshot/--workers/--device/--max-steps/--out/--name`）`:920-1000`（`--tracker` :941、`--diagnostic-oneshot` :975、`--path-head` :977），`--tracker` 默认 `lqr`。⚠️ `--diagnostic-oneshot` 与装载守卫 `allow=` 形参为 R38-P1 冻结期残留（该文件当时在 C6 pin 中，P1b 跳过清扫；解冻后删，见 `.slim/deepwork/p1_v2/round38/p1b_report.md` §4.1）。
- **R37 新增 flag（全部默认关）**：`--ego-warp-mode {off,analytic}`(:969, M4)、`--ds-init-override`(:974)+`--ds-bounds`(:978, M5)、`--long-exog-fill {off,neutral,true}`(:982, M6/M7)；行为说明 `:38-76`（`neutral` 值出处 `runs/round37/m6/m6_calib.json`；`true` = `m7_fill.py` 同公式真值：TTC cap 60 s / 最近矩形距离 / `scan_static_blocker` 同函数）。
- `.slim/deepwork/p1_v2/round32/canary_eval.py`：R32 canary Gate 1–2 评测 harness（Action Fidelity / 物理语义；base 路径逐位不变）`:1-30`；`.slim/deepwork/p1_v2/a3/run_evals.py`：clean150 编排（两臂×三尾 + oracle 对拍）`:1-13`。
- 通用入口：`tools/test.py`（冻结验证集评测，委托 `pipeline.eval_runner.main`；`--policy baseline|ckpt` + 透传）`tools/test.py:1-30`、`pipeline/eval_runner.py:1684-1741`；`tools/baseline_eval.py`（规则基线批量评测）。⚠️ **不存在名为 `run_eval*` 的文件**（全库 0 命中；`run_eval` 只是 `.slim/.../run_post.sh`、`round24/run_clean150_s0s1.sh` 等 shell 里的函数名）。

### 2.4 数据采集 / DAgger

- `tools/dagger_collect.py`：策略驱动 roll-in + 规则专家"空问"标签 + 失败窗口。
  - 隔离协议：`--specs` 必填，载入后硬校验池 `(id, seed)` 与 `scenarios_eval500.json`/`scenarios_val.json` 交集为空（命中 `SystemExit`；`--from-eval` 路径已删除）——`tools/dagger_collect.py:1-30`。
  - roll-in = 学生策略驱动（复用 `pipeline.eval_runner._CkptController`；访问状态 = 学生自己踩到的状态）；标签 = 每个策略步对当前状态调用规则专家 `act()`（默认 `PurePursuitIDMPolicy`，`DEFAULT_LABELER="pure_pursuit"` :130），首步换算成 BC 口径 `(ds,dθ)`——docstring §2–§5；flag 表（`--ckpt`/`--out`/`--specs` 必填，`--labeler {pure_pursuit,idm}`、`--tracker {lqr,exact}` 默认 lqr、`--workers 4`、`--max-steps 600`、`--target-fails 1000`、`--window-fail-before`、`--fail-terminations collision,out_of_road,terminal`、`--balance {weights,cap,none}`）`:1240-1272`。
  - 产物：`expert_bc.npz` + `expert_bc.meta.json`（`dagger` 块含 driver ckpt sha256/labeler/过滤统计/`roll_in: true`）+ `report.json` + `dagger.json`。

### 2.5 常用命令块（可复制）

```bash
# 0) 参数/测试（度量口径）
tools/venv-python net/param_probe.py                      # 参数预算探针（≤1.5M 断言）
tools/venv-python -m pytest tests/ -q                     # 全量单测（collect: 1043 tests）

# 1) calib50 闭环评估（R37 同协议；臂专属 flag 见 docs/round37_report.md §1.3）
tools/venv-python .slim/deepwork/p1_v2/a3/eval_a3.py \
  --ld-mode semantic --rollout-semantic static_role_only --moe-off \
  --tracker lqr --tail chain --spec runs/p1_v2/round25/specs_calibration50.json \
  --workers 6 --device cuda --max-steps 1000 \
  --ckpt runs/p1_v2/round28/K1/ckpt_epoch002.pt --path-head off --lane-anchor off \
  --out runs/round37/m2 --name c0_recheck

# 2) 结构化头训练（正式例；smoke 加 --device cpu --epochs 1 --max-steps 10 --bs 2）
tools/venv-python .slim/deepwork/p1_v2/round36/j1_train.py --task t1_keep \
  --data datasets/path_exec_v1 --device cuda --epochs 4 --bs 64 --lr 1e-3 \
  --target-lane-dropout 0.2 --out runs/round36/j1/t1_keep

# 3) 结构化头离线门（smoke，CPU；臂=structured|baseline，链=imagination|step0）
tools/venv-python .slim/deepwork/p1_v2/round36/j1_eval.py --arm structured --chain imagination \
  --device cpu --tag smoke --out runs/round36/j1_eval

# 4) 生成 DP-6 数据集（datasets/path_exec_v1；IO 密集，默认 8 workers 由脚本自管）
tools/venv-python .slim/deepwork/p1_v2/round36/i1_extract_dataset.py \
  --out-dir datasets/path_exec_v1 --out-json runs/round36/i1_extract.json \
  --out-md runs/round36/i1_extract.md

# 5) DAgger 采集（train spec 必填；标签专家 idm|pure_pursuit）
tools/venv-python tools/dagger_collect.py --ckpt <ckpt.pt> --out runs/round37/collect_idm \
  --specs env/specs/scenarios_train_dagger_r1.json --labeler idm --tracker lqr --workers 4
```

---

## 第 3 章 当前问题（**文档依据**：台账 §0/§5.62–§5.66 + `docs/round37_report.md`）

> 本章不重复第 1/2 章的代码核验；结论与数字以台账/轮报告为准（括号内为节号）。代码位置仅作线索，行号以报告口径为准。

### 3.1 三层根因（互为放大）（report §4；argue §5.64/§5.65）

| 层 | 机制 | 关键证据（数字） | 引用 |
|---|---|---|---|
| **结构层**：保速自锁（纵向） | `ds = DS_HIGH·σ(ds_raw + logit(clamp(v·dt)/DS_HIGH))`，纵向末层零初始化 ⇒ 零残差=**保速先验**；LQR 参考速度 = plan 弦长/dt ⇒ plan 几何即速度指令 ⇒ 掉速不可恢复 | C1 实测 `v[0..9]=6.0→4.09→2.74→1.86→1.49→1.21…`；throttle≈0/−0.05；crawl 中位 **52 s**；自锁 ≈**1.44 m/s**（地板 0.72 m/步）；离线头健康 bias +0.056/MAE 0.149/r 0.983 ⇒ 坍缩是**闭环特有** | report §4.1；argue §5.64 |
| **接口层**：`path_dynamic_ttc` 缺键→0 | 训练语义 TTC present 100%、均值 30.6 s（"无冲突"）vs 评测 0 = "立即碰撞" ⇒ 头持续下小 ds | 单去 TTC 复现装配效应 **77%**（mean\|Δds\| 0.116/0.151）；中性填充（TTC=17.7 s 最小后悔）**零训练 +8pp**（0.340→0.420，翻正 4/翻负 0） | report §4.2；argue §5.64/§5.65 |
| **横向层**：@speed 横向质量 | 加速后 plan 曲率质量/抖动不足 ⇒ 蛇形/越界 | 侧谱 C0 **0.465/0.111/0.049/0.015**、C1 0.055/0.017/0.023/0.008、A1 **0.988/0.241/0.081/0.037**（\|err_y\|/\|err_ψ\|/plan\|κ\|p90/κ 抖动）；A1 steer 饱和 0.861、heading err p50 0.237 rad；**蛇形归因为推断**（未做轮胎/控制器级拆解） | report §4.3, §8.7；argue §5.64 |

- 补强：M7 真值回填（按训练侧同公式，单元核验 A–D 全过：逐位/1e-6/5.7e-14）更强释放速度（crawl −7.5 s、斜率 −0.059 三项最优）但**横向跟不上**（cut-in 例 \|err_y\| 峰值 0.92 m vs C1 0.13 m、180 步出界）——report §4.2/§5，argue §5.64。
- 为什么以前没暴露：旧头是"**速度目标生成器**"（plan 隐含 v_ref>v，主动拉回速度）；历史门（越界/收敛）只测横向 ⇒ 纵向盲区；离线门状态来自数据集（非自产状态）⇒ 反馈病理不可见——report §4.4，argue §5.66。

### 3.2 结果矩阵（calib50 dev，n=50，同协议）与零训练总账（report §3；argue §5.65）

| 臂 | succ | off-road | collision | timeout | rc | 平均速度 | 备注 |
|---|---|---|---|---|---|---|---|
| **C0**（旧 Policy+旧 WM） | **0.580** | 0.180 | 0.120 | 0.140 | 0.872 | 3.65 | 本协议基线 |
| **C1**（结构化+旧 WM） | 0.340 | 0.120 | 0.140 | 0.400 | 0.741 | 2.47 | 爬行自锁（timeout 20/50） |
| **C1_fill**（+TTC 中性 17.7 s） | 0.420 | 0.120 | 0.100 | 0.360 | 0.798 | 2.78 | **零训练最优**（+8pp） |
| **C1_true**（+三键真值） | 0.400 | 0.180 ✗ | **0.080** | 0.340 | 0.776 | 2.89 | coll 最优但 off 硬退化⇒不晋级 |
| **C2**（+analytic warp） | 0.320 | 0.120 | **0.080** | 0.480 | 0.751 | 2.45 | 机制诊断臂（Hard 不劣化） |
| **A1**（先验 4.16） | 0.460 | **0.440** ✗ | 0.060 | 0.080 | 0.791 | 3.56 | Hard 恶化 |
| **A2**（先验 3.0） | 0.380 | 0.280 ✗ | 0.100 | 0.240 | 0.766 | 3.00 | Hard 恶化 |
| **A3**（地板 2.5） | 0.340 | 0.240 ✗ | 0.060 | 0.360 | 0.765 | 2.85 | Hard 恶化 |
| **F3**（v_ref×2） | 0.300 | 0.240 ✗ | **0.480** ✗ | 0.020 | 0.493 | 8.92 | 证伪外部配速 |

- Calib50 绝对门（succ ≥0.62 / off ≤0.10 / coll ≤0.08 / tmo ≤0.10，report §3）：**全部臂未过**（最好单项 coll = M7/C2 达 0.08）。
- 配对翻转要点（report §3）：C0→C1 succ +2/−14（McNemar p=0.0042）；C1→C1_fill succ +4/−0；C1→C1_true off +4/−1（净 +3 ⇒ 不晋级）；C1→A1 off +16/−0；C1→F3 coll +17/−0。
- **零训练通道总账**（report §0/§9；argue §5.65）：F3 ✗（coll 0.48）｜ A 族 ✗（off 0.24–0.44）｜ C2 ~中性（coll −0.06、tmo +0.08）｜ **接口修复 ✅**（唯一有效，+8pp，**非完整解**）｜ 真值回填（语义已核验逐位一致；更强但 off 0.18，不晋级）。

### 3.3 目标与差距（口径登记）

- 目标锚：**IDM 0.756**（calib50 口径；上限审计最高可执行 d2=0.706）——argue §0。
- **口径登记（重要）**：本协议 **C0 = 0.580/0.180/0.120**（succ/off/coll，rc 0.872）vs 历史 **kappa-compose 口径 0.56/0.14/0.08**——**不可直比**；历史锚点（legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58 / 统一基线 0.56 / IDM 0.756）均为 calib50 口径，本报告读数与之同集可比——report §1.4、argue §5.64/§5.65 待令④。
- 分辨率：n=50 ⇒ ±1 集 ≈ ±2pp，只作筛选/方向判定；正式声明留 Eval500（3 seeds / paired bootstrap / McNemar）——report §1.4。
- 三级协议：Calib50 开发 → Clean150 晋级 → Eval500 正式（argue §5.62/§5.66）。

### 3.4 待裁（CG 六项）与已知风险（report §9/§8；argue §5.65/§5.66）

待令（CG）：

1. **接口修复固化形态**：三选一（builder 直供真值 / 模型侧"缺键=中性"默认 / 保留 eval 侧 flag）；neutral = 当前最佳可用（+8pp、Hard 不劣化），true 待训练层配套后重测——report §8.1/§9.1。
2. **P0-B 训练放行** + 设计选择（roll-in 策略、train spec 池、规模/预算、门；建议 Calib50 快筛→Clean150 晋级）——M8 试点数字一并呈报；report §0/§9.2（M8 在飞）。
3. **C3（WM/想象步动力学）时机**：由坐标与接口 5 条联合条件触发（C2 ≠ C3 开关）——argue §5.63。
4. **基准口径登记**：本协议 C0 与历史 kappa-compose 分离，不再混用——report §9.4。
5. **新增考核项**：起步-恢复切片（v≈0/低速注入 → N 步内恢复至场景限速量级；spawn=6.0 m/s 掩盖起步缺陷）——report §8.4/§9.5。
6. **训练放开清单确认**：必须 `policy.structured_action.*`+`plan_head.fusion.2.*`（R38-P2b 收编后前缀 = 旧 `path_head.*` 18 键；head-only 已证欠拟合 0.479 vs 0.058）；建议加 head 前小适配器或微调编码器末层（吸收装配位移 Δds +0.138 / β·δ 2.72× 正交）；**不放** WM 主干/编码器主体/Mem/旧 PolicyHead/MoE/ValueHead；κ 限幅表在闭环执行数据上重标定——argue §5.66。

已知风险/未决（report §8）：

- 接口修复若定稿不当会**训练/评测再次分叉**（§8.1）；真值回填的横向承受度需训练层配套后重测（§8.2）；κ 表未覆盖提速横向裕量（§8.3）；"0 起步/低速恢复"无考核项（§8.4）；Eval500/历史数字集口径需逐条登记（含 IDM 0.756 集来源核验）（§8.5）；**M3 装配位移残余**：即使补齐三键 `eval_on−train` 仍有 Δds +0.138 ⇒ 训练必须在 eval 装配（obs 指纹统一）下进行（§8.6）；A1 蛇形极限环机制为**推断**（§8.7）。

---

## 附录

### A. 已知文档-代码漂移（E1 ⚠️ 8 条）与状态

> E1 = `.slim/deepwork/p1_v2/round37/handbook_facts.md`。任务口径中的 "R1–R3" 编号在本仓库未见记载，故按条目给出**可核验**状态；本手册不代改任何文件。

| # | 漂移项 | 证据 | 状态（2026-10-15） |
|---|---|---|---|
| 1 | `path_*` 外源几何键 env builder **不产出**（此前 7 键只在 `net/path_head.py` 声明） | `env/obs/builder.py:189-190`、`env/obs/path_field.py`、`env/obs/long_exog.py` | **已修（R38-Q4A `9577078`）**：t0 段 8 列 + present 由 env 生产；dynamic 两列缺键 fail-fast；E 组 B 类（掩码消费/static 下沉）归 P3 |
| 2 | eval 装配缺 static/dynamic/ttc 三键 ⇒ 回退 0（"立即碰撞"语义） | `.slim/.../a3/eval_a3.py:55-58`、`round37/m7_fill.py:5-8` | **已收敛（R38-Q4A）**：生产者补全后 eval 装配有键（`m7_fill` re-export 同公式）；缺键语义 = dynamic 两列 fail-fast；`--long-exog-fill neutral` 缓解保留待 P3 固化 |
| 3 | `lane_anchor_xy/lane_anchor_mask` 不在 env obs（结构化头缺键 fail-fast） | `net/path_head.py:778-799`；注入器 `round36/c_lane_anchor.py:1-6, 126-254` | **设计现状**（工具注入，默认关） |
| 4 | `config/eval.yaml:3` 注释口径（"train=5k、仅 train/eval500 两个子集"）vs `env/specs/` 实际 10 个 spec 文件 | `config/eval.yaml:3` vs `env/specs/` 清单 | **待修**（配置文件未改） |
| 5 | 不存在 `run_eval*` 文件；评测入口 = `tools/test.py` / `python -m pipeline.eval_runner` / `tools/baseline_eval.py` / `a3/eval_a3.py` | E1 §5（全库 find 0 命中） | **文档误导项**（以本手册为准；`run_eval` 仅为 shell 函数名） |
| 6 | 评测 tracker 两套默认：`eval_runner` 默认 `exact` vs `tools/test.sh` 默认 `lqr` | `pipeline/eval_runner.py:1716-1718`、`tools/test.sh:5` | **行为现状**（易踩坑，本手册 §2.1 已记录） |
| 7 | `config/model.yaml:6-9` `temporal.type: gru` vs 实际 `TemporalAttention`（GRU 已被 v2 弃用） | `config/model.yaml:6-9` vs `net/temporal.py:1-27` | **待修**（config 未改；README 侧已注记并随 R38 波次入库） |
| 8 | `param_budget_estimate: 585000` 为 H=96 时期历史估算 vs H=128 实测 1,278,879 | `config/model.yaml:3-4`；本手册 §1.7 实测 | **待修/已注记**（config 注释自认，以 `net/param_probe.py` 实测为准） |
| 9 | `path_head` 双实体/双别名口径（`PathExecutionHead`、`path_head_ablation`、`enable_path_head`、oneshot/plan_override） | `net/model.py:864-872, 909-931`；`tests/test_r38_p1_oneshot_path_removed.py`、`tests/test_r38_p2b_policy_fold.py` | **已修（R38-P1/P2b：`596747c`+`48cb5eb`+`939d1a2`）**：单一出口 `model.policy.structured_action`；`path_head` 只读 alias；`path_head.*` 键面经 `legacy_key_mapper` 改名 |
| 10 | K-anchor 旧计划头（`net/anchor.py`、`plan_anchor` 配置、`num_anchors`、锚损失/字典） | `git show dd0a8c2 --stat`；`tools/legacy/k_anchor/` | **已退役（R38-P2a：`a226285`+`dd0a8c2`）**：实现/字典迁 legacy；旧 ckpt 键经 `legacy_key_mapper` 忽略 |
| 11 | legacy `set_ego_warp`（w1/w2）与 `WARP_MODES` 旧档 | `net/ego_warp.py:103`；`tests/fixtures/r38_p2b_warp/` | **已退役（R38-P2b：`939d1a2`）**：`WARP_MODES=("off","analytic")`；w1/w2 逐位 oracle 冻结为 fixture |

> 9–11 = R38 波次（P1/P2a/P2b）**已消除**的实现/口径项；1–2 为 Q4-A 收敛更新。未列项（3–8、补充）状态不变。

补充（E1 §总结 10.a，同族漂移）：LD 采样 offset 代码 = `(5,10,15,20,30)`（`env/obs/ld.py:96`）vs 文档/注释 `{0,20,40,60,80}`（`env/obs/schema.py:339`、`env/obs/builder.py:31, 40`、`env/obs/__init__.py:24`、`env/obs/ld.py:4`）——**待修**（README 已按"以代码为准，未修改"记录）。

工作区证据：写作时那批 README 行号/schema 引用刷新**已随 R38 波次入库**（如 `net/README.md`@`939d1a2`、`config/README.md`@`dd0a8c2`、`env/obs/README.md`@`9577078`）；**当前工作区变更 = `AGENT.md`、`opencode_mobile.md`**（2026-10-15 复核，非本手册写域）。漂移项 7/8/补充项的本体（config 注释/LD offset 注释）仍"待修/已注记"。

### B. 工程纪律

- **默认关 + 逐位验证**：新功能一律默认关且与基线逐位兼容（Gate 0 自证）；生产改动随 `1043 tests`（`tools/venv-python -m pytest tests/`，2026-10-15 R38 四波后 collect 口径）与契约门——`net/model.py:50-55`、argue §1（R32/R33 新增纪律）。
- **无 CG 放行不训练**：本轮（R37 阶段 1）全部为零训练诊断；P0-B 训练须待 CG 放行——report 头部约束、§6.1、argue §5.65。
- **目录卫生**：`.slim/deepwork/`（`.gitignore:2`）、`runs/`（`.gitignore:10`）、`datasets/`（`.gitignore:8`）均不入库；诊断只在 `.slim/`/`runs/`，仅 `docs/` 入库。
- **docs 归档规则**：见 `docs/README.md`——被代码按路径引用的文档一律留 `docs/` 根；轮次关闭 ≥2 轮且无代码引用才移入归档桶（`archive/`、`v6_reports/`、`v7_reports/`）；归档件正文不改，只由 README 索引；移动前必跑引用扫描（模板见 `docs/README.md` §5.3）。
- **口径登记**：跨集/跨口径数字必须逐条标注来源池（argue §1 C8；report §1.4）。

### C. 相关文档索引

| 类别 | 文档 | 角色 |
|---|---|---|
| 决策台账 | `docs/deepseekv4p1_argue.md` | 主文档（§0 一页纸 + §5.x 历轮；R37 = §5.62–§5.66） |
| 本轮报告 | `docs/round37_report.md` | R37 阶段 1 全量报告（lane 索引/结果矩阵/根因/待令） |
| 上一轮 | `docs/round36_report.md`、`docs/r36_implementation_evidence.md`、`docs/r36_warp_layer_note.md` | R36 报告与证据/注记 |
| 契约/规格 | `docs/architecture_contract.md`、`docs/p1_v2_decision.md`、`docs/p0_eval_compat_design.md`、`docs/p1_probe_spec.md` | 架构契约（生成器/调用图/训练-部署同构）、P1-v2 决策、评测兼容、探针规格 |
| 语义/恢复线 | `docs/r21_lane_semantic_contract.md`、`docs/r21_target_lane_selection.md`、`docs/r21_3a_semantic_wiring.md`、`docs/semantic_v1_dataset_card.md`、`docs/recovery_v1_dataset_card.md`、`docs/recovery_v2_contract.md` | 语义三件套 + 数据集卡/契约 |
| 问题框架 | `docs/round30_audit.md`、`docs/rl_reward_v5.md`、`docs/v7_program_prereg.md` | Reality Audit、奖励口径、v7 预注册（现由 `tools/legacy/k_anchor/anchor.py` 引用；`net/anchor.py` 已随 R38-P2a 删除） |
| 报告归档 | `docs/archive/`（含 `p1v2_reports/` 15 件、`v8_reports/` 2 件）、`docs/v7_reports/`、`docs/v6_reports/` | 历史代际/已关闭线；索引 `docs/README.md` §3–§4 |
| 代码侧工作区 | `.slim/deepwork/p1_v2/README.md`（现行主工具表：`a3/eval_a3.py`、`round36/j1_train.py`、`j1_eval.py`、`i1_extract_dataset.py` 等）、`.slim/deepwork/p1_v2/round37/handbook_facts.md`（本手册事实清单） | gitignored 本地工作区 |
