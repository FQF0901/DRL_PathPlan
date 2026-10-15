# net

P2 驾驶策略网络：mem-bank 观测 → 时序聚合 → **latent 世界模型**（ST-GNN 自回归转移）→ plan head/MoE → 策略/价值头。
主入口 `net.model.DrivingModel`（`net/model.py:392`）；按配置构造走 `pipeline.stages.build_model`（`pipeline/stages.py:155`，读 `config/model.yaml` 根键）。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `encoders.py` | 各模态线性投影 + 类型/轨道 id 嵌入 + LayerNorm；无效槽位严格清零。OD 9 维 / LD 7 维 / ego 8 维 / others `F_o`（默认 33），节点顺序 `[ego, od_0..15, ld_0..15]`（`net/encoders.py:24,48-56,133-153`）。 |
| `mem.py` | mem-bank 输入契约（`mem_from_obs`，`net/mem.py:288`）、缺失键回退与旧 others 布局重排、`MemBank.clone/shift_*`、`MemEncoder`（`net/mem.py:678`）；世界系 nav 重建 helper（`nav_features_from_world` / `rebuild_nav_from_world` / `advance_pose_world`）。 |
| `temporal.py` | `TemporalAttention`：6 帧掩码注意力池化 + 帧龄嵌入；整列全无效时 pooled=0 → 经 `out` 层（带 bias 的 Linear，`net/temporal.py:85`）后输出为该层 bias（**非严格 0**，实测范数 ≈0.5–0.6（随机初始化）；与下方 mask 约定的**已知实现偏差，待代码修正**）（`net/temporal.py:41-85`）。 |
| `spatial.py` | 2 层消息传递（层数可配）：ego↔OD/LD、LD 槽位相邻、OD 近邻 KNN（对称化）；边特征 = 相对位姿 5 维 `[Δx,Δy,cosΔθ,sinΔθ,‖Δp‖]/scale`（`net/spatial.py:244-353`；M1/M2 消融组件 `LdSetAttention`/`LdAggregate` 见 `:61-243`）。 |
| `st_gnn.py` | `SpatioTemporalGNN`：latent 自回归转移（3 个 `2H→H→H` 头，输出层零初始化 ⇒ 初始恒等）+ presence/entry logits（先验 bias +2/−2）+ 物理小解码头（od 5 维 / ld 4 维，先验+零初始化残差，诊断用）+ 惰性 `action_ego`（默认关）（`net/st_gnn.py:83-322`）。 |
| `plan_head.py` | 6 个 token 融合（z_ego/od_pool/ld_pool/others/nav/signal）→ MoE → 融合 `latent` + 下一 ego 特征前 6 维（reserved 2 维由采集侧写入，`net/encoders.py:8-10`；v8 rollout 不合成 raw mem/ego）；K-anchor 计划头（默认关）（`net/plan_head.py:41-123,202-221`）。 |
| `path_head.py` | R36 K1 `StructuredPolicyHead`（**唯一动作生成器**的结构化形式）：横向分支（锚 κ_ref + 残差限幅 → κ_near/mid/far）+ 纵向分支（κ_near + 外源几何 → 当前 `ds`）+ 无参组合层 `compose_step_action`；`step_geometry` 按 t0 静态地图 + 想象位姿重算 imagined 步几何；H1 旧 `PathExecutionHead`/`compose_path_plan` 仅消融③；默认关（`set_path_head` 惰性构造）（`net/path_head.py:987-1242`）。 |
| `moe.py` | primary 恒开 + 8 个零初始化残差 expert，top-2 软混合；Switch 式负载均衡 aux（`net/moe.py:80-182`）。 |
| `policy.py` | 交叉注意力策略/价值头（K=1 查询 × 令牌集合；4 头、默认 1 层可配）；sigmoid 压缩有界动作 `ds∈[0,10] m`、`dθ∈[-0.6,0.6] rad`，logstd clamp `[-5,0]`（`net/policy.py:43-51,119-221`）。 |
| `semantic.py` | R21-3A 语义接线（`ld_mode=semantic`，默认关，惰性构造）：`lane_semantic → target_lane_token`（固定归一化 + 2 层 MLP，mask=0 ⇒ 严格 0）；LD 逐槽增广（role/route/offset/group-relation embed）+ 六组 masked mean 读出 → `ld_context`（`net/semantic.py:73-191`）。 |
| `ego_warp.py` | R36 L-D / R37 M1 坐标层（纯函数、无状态）：t0 → ego_k 逆变换、OD 恒速先验（`od_velocity_t0` = 相对速度 + ego t0 速度）、`assemble_pose_k(off/w1/w2)`、anchor schema（`AnchorBatch`/`validate_anchor`）+ 5 项 hook（`WarpHooks`，默认关零开销）（`net/ego_warp.py:119-220,254-390`）。 |
| `anchor.py` | K-anchor 工具：锚字典加载/内置默认 K=6、lane↔ego 航向剖面、WTA 分配、软混合（`net/anchor.py:98-130,294-327`）。 |
| `model.py` | 装配与惰性开关（`set_moe/set_ld_mode/set_ego_warp_mode/set_path_head/set_warp_hooks`）、`arc_step` 运动学、`encode/plan_step/_rollout/forward/rollout`、`SUPERVISED_LABELS`（`net/model.py:210-218,225-275,392,880,923,1292,1503,1864`）。 |
| `param_probe.py` | 参数探针 + 前向/rollout 形状冒烟；断言总参数 ≤1.5M（`net/param_probe.py:29-34,114`：预算 + 断言）。 |
| `probes.py` | R38-D1 §四 七类强制 probes/hooks 统一容器（**默认全关**）：①状态来源 ②Policy ③World Model ④Memory ⑤六步调用链（含 ST-GNN 总调用单列）只读 hook + 统一出口 `ProbeRecorder.write`（张量字典 + JSON）；⑥ plan 连续 footprint 首次越界/碰撞纯函数；⑦ 逐参数梯度账本 `grad_ledger`（第 1/3/6 步 loss 单独反向）。默认关 ⇒ 前向逐位一致/零 RNG/不改 optimizer/ckpt（`net/probes.py:1-100` 契约 + `.slim/deepwork/p1_v2/round38/d1_off_bitwise.py` 自证）。 |

## 数据流（v8，latent 世界模型）

1. **输入**：4 个 per-modality mem（6 帧；index 0 最老、-1 当前帧）→ `mem_from_obs` 组装 `MemBank`。net 只读，绝不写回（`net/mem.py:39-43`）。
2. **编码**：`MemEncoder.encode` 对 OD / ego / others 各做 `TemporalAttention`；**LD 不做时序**，直接取当前帧（`net/mem.py:688-706`）。
3. **t0 latent 状态** = 当前帧编码：`z_ego=embed_ego(ego_now)`、`z_od=embed_od(od_now, od_live)`、`z_ld=embed_ld(ld_now, ld_live)`（单帧，非时序；`net/model.py:1311-1314`）。
4. **Plan head**：`[z_ego, mean(z_od), mean(z_ld), others_ctx, nav, signal]` → fusion → MoE → `latent` + `ego_next`（`net/model.py:1362-1367`，`net/plan_head.py:202-221`）。
5. **策略/价值头**：交叉注意力吃令牌集合 **T=39** = `[z_od 16, z_ld 16, od_pool, ld_pool, others, z_ego, nav, signal, latent]`（`net/model.py:1211-1262`；semantic 模式的 target token 折叠进 `ld_pool` 槽，不追加新 key ⇒ 仍 T=39）。
6. **rollout（6 步）**：每步 `st_gnn` 以 `[z_ego+step_embed(k), z_od, z_ld]` 为节点做 MP → `z_*_next = z_* + head([h_*, z_*])`（无效槽位保持 0）+ t0 帧物理解码；`z_*_next.detach()` 后作为下一步输入；动作/位姿链可微（`net/st_gnn.py:232-322`，`net/model.py:1616-1721`）。
7. **A4 nav 重建**：每步按世界系位姿推进并重建 nav token（缺 `route_world/ego_world` 时回退 t0 冻结，逐位兼容旧输入；`net/model.py:1263-1291,1641-1643`）。

## I/O 契约（B 维在前）

**输入**（env schema v2 规范键；形状/dtype 校验见 `net/mem.py:288-433`）：

- `ego_hist (B,6,8)` + `ego_hist_mask`；末 2 维 = 上一策略步 `(ds,dθ)`（reserved）
- `od_hist (B,6,16,9)` + `od_hist_mask` + `od_id_hist (B,6,16)` + `od_presence_hist (B,6,16)`
- `ld_hist (B,6,16,7)` + `ld_hist_mask`（只作输入）
- `others_hist (B,6,33)` + `others_hist_mask`：`nav(11)+speed_limit(1)+signal(4)+static(5)+road_class(12)`；旧 28 维布局自动重排 + 零填充 static 段 + 一次性告警（`net/mem.py:110-139`）
- `hist_valid (B,6)`（warmup 补位帧为 0；**必需**）
- 可选上下文：`nav (B,11)`、`signal (B,4)`、`ego_world (B,3)`、`route_world (B,M,2)`、`route_world_mask (B,M)`；当前帧 `ego/od/ld/others/od_id/od_presence` 作回退

**输出**（`forward`，`net/model.py:1864-2082`）：

- 恒有：`action_mu/action_logstd (B,2)`、`value (B,1)`、`latent (B,H)`、`ego_next (B,6)`、`router_logits (B,8)`、`expert_weights (B,8)`（top-2 混合权重）；MoE 开启且 `α>0` 时 `load_balance_loss`
- `rollout=True`：`traj_xy (B,6,2)`（t=0.5..3.0 s 端点，t0 自车系）、`traj_theta (B,6)`、`plan (B,6,2)`（实际执行的 6 个动作，`plan[:,0] == action_mu`）
- `world_model=True`（必须同时 `rollout=True`）另返回：`od_pred (B,6,16,5)`、`ld_pred (B,6,16,4)`、`od_presence_pred/od_entry_pred (B,6,16)`（logits）、`z_ego_pred/z_od_pred/z_ld_pred`（latent 诊断）
- K-anchor 开启（`num_anchors>0`）另返回：`anchor_logits/anchor_probs/anchor_plan/anchor_speed/anchor_residual/anchor_ctx`
- `forward(rollout=False, world_model=False)` 是 **PPO cheap path**：不跑任何 st_gnn 消息传递，`action_mu/action_logstd/value/router_logits/expert_weights/latent` 与完整前向逐位一致（`net/model.py:1970-1985`）；`rollout=True, world_model=False` 仍跑 rollout 但不返回预测键。
- R36 K1（结构化头启用时，非默认）：`plan_from_aux/structured_*` 逐步组装证据键 + `contract_*` 契约断言键；`policy_action_mu`/`action_logstd` 仅在 `set_policy_diag(True)`/`set_path_head_trace(True)` 诊断时产出（`net/model.py:1997-2052`）。
- R37 M1（`set_warp_readout(True)` 时，非默认）：`warp_readout_valid/warp_prior_err/warp_residual_contrib/warp_final_err`（`net/model.py:1988-1996`）。

## rollout / detach 语义（固定，无消融开关）

- v8 不再拷贝/滑动 raw mem，直接迭代 latent 状态；真 mem 仅作 nav 回退锚点（`net/model.py:1519-1523`）。
- 来自真实 obs 的 t0 状态**不 detach**（检测任务）；rollout 合成的后续 latent 进入下一步前 detach（预测任务）；**action/pose 链保持可微**（`net/model.py:1627,1720-1721`）。
- `wm_detach` 形参保留但**恒为 no-op**（`net/model.py:1883`）。
- 图位姿（t0 帧口径）：ego = 累积位姿（detach）、OD = t0 位姿 + `k·dt·v_t0`、LD 静止（`net/model.py:1646-1663`；R37 M1 `set_ego_warp_mode("analytic")` 时改由 `assemble_pose_k(w2)` + `od_velocity_t0` 组装，默认 `off` 逐位不变）。
- 物理解码先验：OD 匀速外推 / LD 静止，t0 帧口径（`net/st_gnn.py:276-311`）。

## MoE 口径

- `out = primary(x) + residual_scale · Σ_{i∈top2} g_i·expert_i(x)`；top-2 在被选 logit 上 softmax（恰 2 个非零、和为 1）；无硬切/二值门（`net/moe.py:1-27,149-182`）。
- 运行时开关 `DrivingModel.set_moe(enabled=...)`（非参数，不进 state_dict）：阶段 A / 阶段 B phase 1 关闭（输出严格 = primary）；phase 2 与推理默认开启（`net/model.py:880-890`）。
- 负载均衡 aux = `α·E·Σ f_i·P_i`（`f` detach、`P` 可微；值域 `[1,E]`）；`α>0` 时输出 `load_balance_loss`；诊断 `expert_load/load_cv/gate_entropy`（`net/moe.py:63-77,131-146`）。
- 固定监督标签顺序 `SUPERVISED_LABELS`（8 项，必须与 `config/model.yaml::moe.router.supervised_labels` 一致；`net/model.py:210-218`）。

## K-anchor 计划头（默认关）

- `num_anchors>0` 时 PlanHead 增加选择头（零初始化 ⇒ 初始均匀）、连续速度头（初始 ≈3.5 m/步）、逐锚 6×2 残差头（零初始化）；`plan = Σ p_k(anchor_k+residual_k)`，`p=softmax(logits/τ)`，推理可 argmax（`net/plan_head.py:82-123,161-200`）。
- rollout 尾段（k≥1）由 t0 锚混合驱动，step0 仍 = `action_mu`（输出契约不变）；锚字典缺文件时回退内置 K=6（`net/anchor.py:98-130`）。
- lane 通道已从 obs v6 删除 → `plan_anchors` 的 `lane_ctx` 恒为 `zeros(B,3)`（valid=0 → 恒等变换）；`anchor_lane_context` 保留但不再接线（`net/model.py:1369-1371`）。

## Round-36 K1 结构化动作头（`net/path_head.py`；默认关）

- **开关**：`DrivingModel.set_path_head(True)` 惰性构造 `StructuredPolicyHead`（构造期保存/恢复 CPU RNG ⇒ 不扰动既有初始化抽样序列）；默认关 ⇒ 键集/前向/RNG 与 HEAD 逐位一致，旧 ckpt strict 载入 0/0/0（`net/model.py:923-950`）。
- **每步调用（唯一动作生成器）**：`structured` 模式（默认）下 `_rollout` 每步用**同一实例、当前** `z_k` 产出 `a_k=(ds_k, κ_near_k·ds_k)` → WM 单步推进 → 同一头重算；6 步 `plan` 由逐步动作**组装**供 LQR/环境（`plan[:,0]==action_mu`），辅助 `κ_mid/κ_far/未来 progress` 不进动作（`net/path_head.py:1-70`，`net/model.py:1616-1721`）。消融脚手架：`policy`①（旧头 + WM rollout）/`structured`②（生产默认）/`oneshot`③（H1 一次性 6 步诊断臂，默认拒绝，需 `allow_oneshot=True`）（`net/path_head.py:254-255`，`net/model.py:952-975`）。
- **横向头**：锚 = 外部给定目标车道中心线折线 `lane_anchor_xy (B,P,2)`（ego t0 系）→ 三控制站位 `KAPPA_CTRL_S = (5.25, 17.5, 29.75) m`（= `KAPPA_CTRL_FRACTIONS (0.15, 0.50, 0.85) × LAT_HORIZON_M 35.0`）的 `κ_ref (B,3)`；头输出 `κ_near/mid/far` 残差 = `clamp(tanh(raw)·κ_p95(v), ±κ_p99(v))`（速度条件化数据驱动限幅，标定 `runs/round36/k1/calibration.json`；数据不足回退常量 `KAPPA_LIMIT_FALLBACK = tanh×0.1`），末层零初始化 ⇒ 初始 `κ(s) = κ_ref`（锚中心线）；分段线性 κ(s)（`net/path_head.py:139-190,281-360`）。
- **纵向头**：输入 = `κ_near`(1) + 外源几何 8 维（速度/左右余量/静态距/动态距/TTC/nav 进度/目标末端距；缺键默认 0 + `long_present` 掩码）→ 当前步 `ds`（有界 `[0,10]`；末层零初始化 ⇒ 初始 `ds = clamp(v_current·dt, bounds)`，速度缺失回退常量 3.5 m/步）（`net/path_head.py:193-245,316-335`）。
- **组合层**：`compose_step_action` → `a=(ds, κ_near·ds)`（策略动作界截断，无参数）；H1 一次性 `compose_path_plan`（`ds_k = s_k−s_{k−1}`、`dθ_k = κ̄_k·ds_k` 段内平均曲率精确积分）仅消融③使用（`net/path_head.py:562-663`）。
- **imagined 步几何**（:func:`step_geometry`）：k≥1 的车道锚/边界余量/静态距/目标车道末端距由 **t0 静态地图 + 想象位姿**确定性重算；动态距离/TTC/nav/speed 保持当前步（`net/path_head.py:664-777`）。
- **R36 L-A 旧头消歧**：`structured`/`oneshot` 下旧 `PolicyHead` 默认**不再前向**，输出 dict 无 `policy_action_mu`/`action_logstd`（诊断经 `set_policy_diag(True)`/`set_path_head_trace(True)` 恢复）；`set_contract_asserts`（默认开）在 eval/deploy 违反「6 次头调用 / 5 次 WM 推进 / 唯一动作源」契约时 fail-fast（`net/model.py:989-1017,2018-2052`）。
- **训练/冻结集**（manifest：`.slim/deepwork/p1_v2/round36/h1_manifest.json`）：可训 `path_head.` + `plan_head.fusion.2.`（PolicyBackbone 末层；`TRAIN_ALLOWLIST_PREFIXES`，`net/path_head.py:248-251`）；WM/Encoder/Memory/旧 PolicyHead/MoE/ValueHead 全冻。target-lane token dropout：`set_target_lane_dropout(p)`（默认 0；20%→50% 斜坡由训练侧调度）（`net/model.py:1033-1044`）。
- **Gate 0 证据**：`runs/round36/h1/gate0_compat.json`（逐位/strict/LQR 契约/966 tests）+ `runs/round36/h1/smoke.json`（开/关前向 + 梯度 allowlist + dropout）。

## config/model.yaml → 模型构造键（`pipeline/stages.py:155-197`）

| 键 | 映射 |
| --- | --- |
| `hidden_dim` | `DrivingModel(hidden)`（`DrivingModel` 默认 96；`build_model` 缺键默认 128） |
| `spatial.layers` | `spatial_layers`（缺省 2） |
| `moe.primary.hidden_dim` | `primary_hidden`（缺省 768） |
| `moe.experts.count / hidden_dim` | `num_experts / expert_hidden`（缺省 8 / 76） |
| `moe.router.hidden_dim` | `router_hidden`（缺省 384） |
| `policy.trunk_hidden` | policy 主干隐藏维（缺省 160） |
| `policy.attn_heads / attn_layers` | 注意力头数（缺省 4）/ 层数（缺省 1）；**同时用于 value 头** |
| `value.net_hidden` | value 头隐藏维（缺省 256） |
| `world_model.rollout_steps` | `wm_steps`（缺省 6） |
| `plan_anchor.enabled / num_anchors / path / temperature / hard` | K-anchor（`enabled=false` → `num_anchors=0`，模型与旧版逐位一致） |
| `moe.router.supervised_labels` | 不由 build_model 读取；由 `pipeline.trainer.load_supervised_labels` 校验 BC 数据集标签顺序 |

未读取的段：`temporal.*`、`world_model.enabled/conditioned_on/loss`、`moe.primary.enabled/router_gated`、`moe.experts.init`、`moe.router.type/supervision/top_k`（top-2 在 DrivingModel 默认值固定为 2）、`spatial.type/hidden_dim`（build_model 只读 `spatial.layers`）、`policy.rollout_steps`、`policy.action.*`、`value.*`（除 `value.net_hidden`）、`param_budget_estimate`（预算以 `net/param_probe.py` 实测为准，见「规模与测试入口」）。

## 规模与测试入口

- 参数预算 ≤1.5M 由 `net/param_probe.py` 断言。实测（`tools/venv-python net/param_probe.py`）：默认构造（H=96、expert_hidden=192）= **1,000,639**；H=128 / expert=76（primary/router/trunk/net 取代码缺省）= **1,278,879**。实际规模随 `config/model.yaml`（如 `moe.primary.hidden_dim`、`policy.attn_layers`）变化。
- 测试：`tools/venv-python -m pytest tests/test_net_shapes.py tests/test_mem_rollout.py tests/test_v6_attn_heads.py tests/test_v6_world_nav.py tests/test_plan_anchor.py tests/test_od_pose_grad.py -q`。

## 约定与坑

- `hist_valid=0` 的补位帧整帧不参与时序注意力；编码器嵌入与 ST-GNN 无效槽输出严格为 0（`net/encoders.py:153`、`net/st_gnn.py:284-287`）；`TemporalAttention` 整列全无效时输出 = out 层 bias（非严格 0，见上表 temporal 行的已知偏差）。
- OD 槽位 = track id（跨帧稳定），身份由 `od_id_hist` 标注；`od_live = od_mask & od_presence`（当前帧）决定哪些槽进入头令牌/图节点。
- `od_pose` / `od_state_from_features` 对退化槽位（cos=sin=0）做 atan2 防 NaN 处理（`net/encoders.py:179-195`，`net/st_gnn.py:164-181`）。
- 动作 `(ds,dθ)` = 下一 0.5 s 的弧长 + 航向变化；`arc_step` 是 net 内唯一圆弧实现，`interpolate_actions` 提供 6→30 点一致性检查（`net/model.py:225-275`）。
- 零初始化：ST-GNN 转移/物理头输出层与 policy `mu` 末层零初始化 ⇒ 初始 ST-GNN 为恒等转移、初始动作 = 界中点 `(5.0 m, 0 rad)`；`logstd` 头权重非零（std=0.01）保证注意力/主干经该路径有梯度（`net/st_gnn.py:51-62,142-147`，`net/policy.py:143-155`）。
