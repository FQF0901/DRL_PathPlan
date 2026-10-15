"""``DrivingModel``：mem-bank 输入 → 注意力聚合 → plan head(MoE) → latent 自回归 rollout。

v8（B3）数据流（latent 世界模型）
--------------------------------
1. **输入**：4 个 per-modality 真 mem（``ego_hist/od_hist/ld_hist/others_hist`` + mask、
   ``od_id_hist/od_presence_hist``，见 :mod:`net.mem`）。**net 只读**；真 mem 的更新只由 env 完成。
2. **编码**：:class:`net.mem.MemEncoder` 把 6 帧 mem 编码成"编码 mem"——OD / Ego /
   Others 各做掩码注意力池化，**LD 不做时序**（直接用当前帧，理由见该模块 docstring）。
3. **状态 latent**（t0）：``z_ego/z_od/z_ld`` = **当前帧编码**（``embed_ego(ego_now)`` /
   ``embed_od(od_now, od_live)`` / ``embed_ld(ld_now, ld_live)``；单帧，非时序聚合）。
4. **Plan head**：``[z_ego, mean(z_od), mean(z_ld), others_ctx, nav, signal]`` → MLP 融合 →
   **MoE**（primary 常开 + 8 个 specific 专家、top-2 软混合）→ 产出**下一时刻 ego 特征**
   与策略/价值 latent；``router_logits(8)``/``expert_weights(8)`` 原样输出。
5. **ST-GNN（latent 转移）**：节点 = ``[z_ego + step_embed(k), z_od, z_ld]``，MP 2 层 →
   ``z_*_next = z_* + head([h_*, z_*])``（3 个 ``2H→H→H`` 头）+ presence/entry logits +
   **物理小解码头**（od 5 维 / ld 4 维，先验 + 零初始化残差，t0 帧口径；诊断/旧消费者）。
6. **rollout（6 步，不再滑动 raw mem）**：每步 plan head/policy 用**当前步 latents**；
   st_gnn 得 next latents，**作为下一步输入前 detach**（状态链切断）；A4 nav 逐步重建
   （nav 是上下文，允许 raw→encode）。真 mem 绝不被写入。
7. **detach 语义（规格第 5 条，固定不可配）**：来自真实 obs 的 t0 状态**不 detach**
   ——第一步预测/轨迹的梯度直达编码器（检测任务）；rollout 合成的后续状态在进入下一步
   前 detach（预测任务不回传状态链）。被切的是"状态（latent）"，**action/pose 链保持可微**：
   ``traj_xy`` 的梯度沿 pose 链训练 plan head 的动作链。

v6 增补（docs/repo_handbook.md 附录 D；A1/A2 去池化仍有效）
--------------------------------------------------
- **A1/A2 去池化**：policy/value 不再吃池化 token，改为**交叉注意力头**
  （:class:`net.policy.CrossAttnHead`）直吃令牌集合（OD/LD 逐槽 + 池化 + others + ego +
  nav + signal + plan_head 融合 latent；``_head_tokens``）；池化仅保留在 plan head 融合路径。
- **A4 nav 修正**：``_rollout`` 内逐步按**世界系** ego 位姿重建 nav/signal
  （:func:`rebuild_nav_from_world` / :func:`advance_signal`；无 ``route_world``/``ego_world``
  键时回退 t0 冻结，逐位兼容旧输入）。v8：others 上下文为 t0 静态 token，不再逐步同步
  others mem 的 nav 维（rollout 不重编码 raw mem）。

v8 增补（参数再分配 + lane/ttc 移除 + WM latent 重构）
-----------------------------------------------------
- 参数再分配：MoE ``primary/experts/router`` 与 policy trunk / value net 的隐藏维由
  ``config/model.yaml`` 新键控制（见 :class:`net.moe.MoEBlock` / :class:`net.policy.PolicyHead`）；
- **lane/ttc 移除**：obs v6 删除 ``lane``/``ttc`` 通道 → 编码器线性层/类型嵌入与
  ``net.mem.context_features_from_obs`` 一并删除（``NUM_NODE_TYPES`` 7→5）。
- **头令牌集合 T=39**：``[z_od 16, z_ld 16, od_pool 1, ld_pool 1, others 1, z_ego 1, nav 1,
  signal 1, latent 1]``（``od_pool/ld_pool`` = t0 上下文 ``masked_mean(od_ctx/ld_ctx)``）；
  **去掉 t0 单次 MP 进头**（``_t0_object_features`` 保留供诊断，不再接入令牌）。
- **监督口径**：WM 主损失 = **latent consistency**（``z_k`` vs 未来帧编码目标，掩码加权
  smooth_l1，见 ``pipeline.trainer.weighted_latent_consistency_loss``）；物理小解码与
  presence/entry/ego_next 为辅助/诊断项。

Round-36 K1（结构化动作生成器；默认关）＋ R38-P2b 双实体收编
-----------------------------------------------------------
- ``set_path_head(True)`` 惰性构造 :class:`net.path_head.StructuredPolicyHead`
  （**唯一动作生成器的结构化形式**：``lateral`` 分支 + ``longitudinal`` 分支 + ``compose``
  确定性几何层），收编后注册为 ``self.policy.structured_action``（键面
  ``policy.structured_action.*``；旧 ``path_head.*`` 键经 ``legacy_key_mapper`` 重映射）；
  **默认关 ⇒ 键集/前向/RNG 与 HEAD 逐位一致**
  （Gate 0 自证：``.slim/deepwork/p1_v2/round36/k1_contract_gate.py``）。
- **每步调用**：``_rollout`` 每步用当前 ``z_k`` 调**同一**头 → ``a_k=(ds_k, κ_near_k·ds_k)``
  → ``st_gnn`` 推进一步 → 同一头重算；6 步 ``plan`` 由逐步动作**组装**（``plan[:,0]==action_mu``
  契约不变）。``κ_mid/κ_far``、未来 progress/ds 为辅助输出（不进动作/LQR）。影像步静态几何
  由 t0 静态地图 + 想象位姿确定性重算（:func:`net.path_head.step_geometry`）。
- 启用后输入契约：``obs['lane_anchor_xy'] (B,P,2)``（给定目标车道中心线折线，ego t0 系；
  可选 ``lane_anchor_mask``）；外源几何键 ``path_*`` 可选（缺省 0 且 presence 掩码随输出返回）。
- **R38-P2b 收编口径**：``model.policy`` 单实体（GPT §七-D）——``structured_action`` 子结构 =
  结构化头；动作分布 / 辅助意图为头部**输出分组**（``StructuredPolicyHead.output_groups``；
  ``action_distribution`` 的 ``log_std`` 在 RL 同源方差接线前 fail-fast，均值与方差不得来自两个头）。
  ``enable_path_head`` 镜像属性与 ``path_head_ablation`` 模式枚举**已删除**；``path_head`` 兼容
  只读 alias、``set_path_head_ablation`` 降为 legacy 诊断入口（``policy``/``structured`` 两档，
  ③ 一次性六步臂 R38-P1 已删，写入即 fail-fast，历史复现看 ``tools/legacy/oneshot_path.py``）；
  ``action_generator_names()`` 给「生成器数 = 1」静态断言入口（structured ⇒
  ``("policy.structured_action",)``）。cheap path 无 WM ⇒ ``plan`` 仅为辅助预览（``plan_from_aux=1``），
  生产 LQR 参考走 ``policy_wm_rollout``（旧 ``rollout=True``；R38-P3-H 枚举）。
- ``set_target_lane_dropout(p)``：训练期 target-lane token dropout（20%→50% 斜坡由训练侧调度；
  默认 0 = 关；``p=0`` 不消耗 RNG；仅 ``model.training`` 且存在 target token（semantic 模式）时生效）。

Round-36 L-A（架构硬化：旧头消歧 + 一次性六步守卫；R38-P2b 收编后）
------------------------------------------------------------------
- **旧头消歧**（GPT ``0912cb9`` 批评 #1）：``path_head_mode() == "structured"`` 时 ``forward``
  **不前向旧 ``PolicyHead``** —— 同一 forward 只有一个动作源（结构化头），旧头输出
  （``policy_action_mu`` 诊断）与 ``action_logstd`` **从输出 dict 中移除**（显式 KeyError 优于
  静默 NaN）。``action_logstd`` 来源 = 旧 ``PolicyHead``（**RL 未接线**：方差来源与
  ``action_mu`` 不同源，RL 接线前必须统一）。
- **R38-P2b：旧头诊断离线化**（`LEGACY_REPRO`）：``set_policy_diag`` 已 fail-fast（
  ``RuntimeError``，指路 ``tools/diagnostics/legacy_policy_diag.py``）；``set_path_head_trace``
  只记 trace、不再恢复旧头前向；正式 forward **只保留一个动作出口**（structured ⇒ 结构化头 /
  默认关 ⇒ 旧头），不在主 forward 并行计算旧头诊断。
- **一次性六步路径已删除**（R38-P1 减枝；= 原 L-A 守卫的 ③ 诊断臂）：该路径是第二个六步轨迹
  生成器（绕过一个决策步一次头调用）⇒ `ARCHITECTURE_FORBIDDEN`；实现迁
  ``tools/legacy/oneshot_path.py``，模式名写入 ``set_path_head_ablation`` 即 ``ValueError``
  fail-fast（评测/部署工具侧原 ``--diagnostic-*`` 守卫随之退役）。
- **运行时契约断言**（GPT ``7dcf6bd`` §四.1/§四.8；``set_contract_asserts``，**默认开**）：
  structured + wm rollout 模式（``policy_wm_rollout``/``world_model_teacher_forcing``）时每
  forward 常数开销地校验 ① 同一头实例（a0..a5）、
  ② 头调用 6 次、③ WM 推进 5 次、④ 辅助未来输出不进 LQR（``plan`` 逐步组装且
  ``plan[:,0]==action_mu``）、⑤ 一次性钉住路径**不存在**（结构断言：历史模式名回归或
  ``one_shot_plan`` 复活 ⇒ 门失败）；eval/deploy 违反 ⇒ ``RuntimeError`` fail-fast，
  训练路径只落盘 ``contract_*`` 输出键。

Round-37 M1（warp 主路径接入 + OD 速度语义修正 + 五项 hook + 分项读数；默认全关）
------------------------------------------------------------------------------------
- ``set_ego_warp_mode("off"|"analytic")``（默认 ``off``）：``analytic`` 时 rollout ``pose_k``
  构造走 :func:`net.ego_warp.assemble_pose_k` ``analytic``（op 序 = 旧 ``w2`` 口径）——t0 系当
  **静态场景**，OD 先验速度必须是 **t0 场景系对象速度** ``v_OD = v_rel + v_ego``
  （:func:`net.ego_warp.od_velocity_t0`；``v_ego = enc.ego_now[..., 0]``），LD 静止锚做 ego
  SE(2) 逆变换；整图重表达进 ego_k 帧（ego 槽恒 ``(0,0,0)``）。**禁止**用相对速度推进后再扣
  自车运动（重复扣 ``|Δp|`` 的集成错误）。``off`` 经同一函数（``mode="off"``，相对速度=随体
  坐标语义）逐位复现旧内联分支。**R38-P2b**：legacy 实验开关 ``set_ego_warp``（``w1``/``w2``）
  已删除（`LEGACY_REPRO`；生产 net 只有一套 warp 状态机）——逐位参照物 = 
  ``tests/fixtures/r38_p2b_warp/``（生成器 ``round38/p2b_oracle_freeze.py``），legacy 纯函数
  快照 = ``tools/legacy/legacy_ego_warp.py``；``w1``/``w2`` 模式名一律 ``ValueError`` fail-fast。
- ``set_warp_hooks(WarpHooks)``（默认 ``None`` = 关）：每步经
  :meth:`net.ego_warp.WarpHooks.capture` 记录 ① ``anchor_pre``（warp 前 t0 帧）/②
  ``anchor_analytic``（ego_k 帧）/③ ``anchor_actual_next``（真实下一帧，由 ``obs`` 提供；缺省
  n/a）/④ ``residual_learned``（缺省 None = 0，接口留给 WM residual 训练）/⑤ ``anchor_final``
  （= ② + ④）。
- ``set_warp_readout(True)``（默认关）：分项读数（诊断输出，训练/评测可关）——
  ``warp_prior_err``（解析先验误差）/``warp_residual_contrib``（学习残差贡献）/``warp_final_err``
  （最终 anchor 误差），``(B,T)`` 掩码均值 [m]；缺真实下一帧（部署侧）⇒ ``warp_readout_valid=0``
  （n/a，不产出误差键）。
- 数据接口（可选 obs 键，形状 ``(B,T,N,3)``，N = 1+OD+LD 节点数，T = rollout 步数）：
  ``obs['warp_anchor_actual_next']``（ego_k 帧真实下一帧 anchor；离线/训练数据提供）、
  ``obs['warp_anchor_residual_learned']``（逐 step learned residual）。缺省 ⇒ 诊断按 n/a/0。
- 默认全关 ⇒ 前向逐位不变（``warp_mode="off"`` 的 ``assemble_pose_k`` 已由 R36 L-D
  pre-hook 门逐位验证；Gate A：M0/cheap/structured/遗留 w1/w2 全输出 ``max|Δ|=0``）。

动作与运动学
------------
动作 ``(ds, dθ)`` = 下一个 0.5 s 的弧长 + 航向变化；:func:`arc_step` 是 net 内部唯一
圆弧实现（与 ``env/tracking`` 同一约定），:func:`interpolate_actions` 提供 6→30 点插值
一致性检查入口。``traj_xy (B,6,2)`` = t=0.5..3.0 s 的 6 个动作端点（t0 自车系）。

输出契约（评测/跟踪器/PPO 依赖，保持不变）
------------------------------------------
``action_mu/action_logstd (B,2)``、``value (B,1)``、``traj_xy (B,6,2)``、``plan (B,6,2)``
（rollout 实际执行的 6 个动作，``plan[:,0] == action_mu``）、``router_logits (B,8)``、
``expert_weights (B,8)``、``latent (B,H)``；另含 ``od_pred (B,6,16,5)``、
``ld_pred (B,6,16,4)``（物理解码，诊断 + 未来 LD 监督）、``od_presence_pred (B,6,16)``、
``od_entry_pred (B,6,16)``（logits）、``traj_theta (B,6)``、``z_*_pred``（latent 诊断）。

``forward(..., forward_mode=...)`` 是显式模式枚举（R38-P3 引入；**R39-A1 起唯一开关**）：
- ``forward_mode ∈ {policy_single_step, policy_wm_rollout, world_model_teacher_forcing,
  diagnostics}``（映射表 = :data:`FORWARD_MODE_FLAGS`；唯一入口 :func:`resolve_forward_mode`）；
- 缺省（``forward_mode=None``）≡ ``world_model_teacher_forcing``（旧签名默认 (True, True)
  逐位保留）；
- **旧 ``rollout=/world_model=`` 布尔兼容口已删除**（R39-A1；全仓调用点已迁移）：残留布尔
  调用 ⇒ ``ValueError`` fail-fast 并指路 ``forward_mode``（不再静默兼容）。

旧口径（枚举语义）：``policy_single_step`` 是 PPO cheap path：省略 traj/rollout 多步键
（不跑 st_gnn 消息传递），``action_mu/action_logstd/value`` 与完整前向逐位一致。

向后兼容
--------
- 输入契约（:func:`net.mem.mem_from_obs`）= env schema v2 规范键
  （``ego_hist/od_hist/od_id_hist/od_presence_hist/ld_hist/others_hist`` + masks +
  ``hist_valid``；others 维默认 33 = schema v4：nav(11)+speed_limit(1)+signal(4)+
  static(5)+road_class(12)；旧 28 维数据 ⇒ R38-P3-G fail-fast + 显式适配器
  ``tools/legacy/others_layout.py``）；
  缺失 ``ego_hist/others_hist`` 或 ``od_id_hist/od_presence_hist`` 时按模块 docstring
  的回退规则处理（旧数据集可直接复用）；
- ``nav/signal`` 键保留为可选上下文 token（v2 的规范上下文在 ``others`` 里）；
- ``wm_detach`` **形参已撤（R38-P3-H）**：v8 的 latent 状态链 detach 语义是固定的，从来
  没有消融开关；迁移期旧签名 ``wm_detach=...`` 由 ``**legacy_signature`` 吞掉（恒 no-op，
  逐位同结果；回归测试 ``tests/test_stage_v11.py::test_wm_detach_kwarg_is_accepted_noop``），
  不再出现在 ``forward`` 签名中；其他未知关键字 ⇒ ``TypeError`` fail-fast。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from net.encoders import (
    DEFAULT_OTHERS_DIM,
    H,
    HISTORY_FRAMES,
    LD_SLOTS,
    NAV_DIM,
    OD_SLOTS,
    SIGNAL_DIM,
    ObsEncoders,
)
from net.mem import (
    EncodedMem,
    MemBank,
    MemEncoder,
    mem_from_obs,
    squeeze_batch_singletons,
)

try:  # A4：env lane 在 net/mem.py 提供实现（签名见本模块 rebuild_nav_from_world）；未就绪用占位
    from net.mem import advance_pose_world as _mem_advance_pose_world
    from net.mem import nav_features_from_world as _mem_nav_features_from_world
    from net.mem import world_state_from_obs as _mem_world_state_from_obs
except ImportError:  # pragma: no cover - helper 未就绪时的回退
    _mem_advance_pose_world = None
    _mem_nav_features_from_world = None
    _mem_world_state_from_obs = None

from net.ego_warp import WarpHooks, assemble_pose_k, masked_pose_error, od_velocity_t0
from net.path_head import (
    LAT_HORIZON_M,
    PATH_HEAD_DEFAULT_MODE,
    PATH_HEAD_MODES,
    StructuredPolicyHead,
    anchor_from_obs,
    long_exog_from_obs,
    replay_plan,
    step_geometry,
    target_lane_valid_from_obs,
)
from net.plan_head import PlanHead
from net.policy import ACTION_HIGH, ACTION_LOW, PolicyHead, ValueHead
from net.probes import ProbeRecorder
from net.semantic import SemanticWiring
from net.spatial import LdAggregate, LdSetAttention, wrap_angle
from net.st_gnn import SpatioTemporalGNN

#: router 监督标签（固定顺序，必须与 config/model.yaml::moe.router.supervised_labels 一致）
SUPERVISED_LABELS: tuple[str, ...] = (
    "cutin_active",
    "cutout_active",
    "crowded",
    "car_following",
    "on_curve",
    "merging",
    "roundabout_near",
    "near_intersection",
)

_ARC_EPS = 1e-6

# ---------------------------------------------------------------------- 前向模式
#: R38-P3-H（GPT §七-H）：``forward`` 显式模式枚举（替代 ``rollout × world_model`` 隐式布尔
#: 组合）。映射表 = :data:`FORWARD_MODE_FLAGS`；``diagnostics`` 当前 = 全量前向旗标对
#: （与 ``world_model_teacher_forcing`` 相同），独立枚举名用于**隔离诊断/可视化调用方**与
#: 训练/评估口径（未来诊断面扩展——probes/readout 强制——挂靠此值，不动生产模式）。
FORWARD_MODES: tuple[str, ...] = (
    "policy_single_step",
    "policy_wm_rollout",
    "world_model_teacher_forcing",
    "diagnostics",
)

#: ``forward_mode`` → ``(rollout, world_model)``（行为映射；每组合逐位对照见
#: ``tests/test_r38_p3_forward_mode.py``）。
#: - ``policy_single_step``：单步策略（cheap path；无 st_gnn 消息传递/rollout）；
#: - ``policy_wm_rollout``：策略 + 6 步 **latent WM rollout**（轨迹/plan 逐步组装；不返回
#:   WM 物理解码键 —— 修复"关 WM 仍 latent rollout"的旧命名歧义：本模式 WM 明确在跑）；
#: - ``world_model_teacher_forcing``：全量前向（WM 预测键 ``od_pred/ld_pred/z_*_pred``
#:   随 rollout 返回；WM 训练/评估口径。命名沿 GPT 术语：当前实现为 latent 自回归 rollout，
#:   真正的教师强制输入链（真实未来帧/动作）是 Stage A ``_wm_predictions`` 路径，未接入本
#:   入口）；
#: - ``diagnostics``：当前 = 全量前向（同 WMTF），语义保留给诊断/可视化调用方。
FORWARD_MODE_FLAGS: dict[str, tuple[bool, bool]] = {
    "policy_single_step": (False, False),
    "policy_wm_rollout": (True, False),
    "world_model_teacher_forcing": (True, True),
    "diagnostics": (True, True),
}


def resolve_forward_mode(forward_mode: str | None) -> tuple[bool, bool]:
    """``forward_mode`` → ``(rollout, world_model)``（唯一模式入口；R39-A1 起无布尔路径）。

    - ``forward_mode=None`` ⇒ ``world_model_teacher_forcing``（旧签名默认 (True, True) 逐位保留）；
    - 非法值 ⇒ ``ValueError``（列出允许值）。
    """
    mode = "world_model_teacher_forcing" if forward_mode is None else str(forward_mode)
    if mode not in FORWARD_MODE_FLAGS:
        raise ValueError(f"forward_mode={mode!r} 非法；允许值 = {FORWARD_MODES}")
    return FORWARD_MODE_FLAGS[mode]


#: R39-A1：已删除的布尔兼容口（`forward` 收到即 fail-fast；文案指路 forward_mode）。
_REMOVED_FORWARD_BOOL_KWARGS: tuple[str, ...] = ("rollout", "world_model")


def _consume_legacy_forward_kwargs(kwargs: dict[str, Any]) -> None:
    """旧签名兼容（R39-A1）：``wm_detach`` 恒 no-op 吞掉；已删除布尔口 ⇒ fail-fast。

    - ``wm_detach``：v8 的 latent 状态链 detach 语义固定（``_rollout`` 内固有），从未提供
      消融开关 ⇒ 吞掉并忽略（逐位同结果；回归 =
      ``tests/test_stage_v11.py::test_wm_detach_kwarg_is_accepted_noop``）；
    - ``rollout=`` / ``world_model=``：R38-P3 兼容 shim、**R39-A1 删除**（全仓调用点已迁移
      到 ``forward_mode``）⇒ ``ValueError`` fail-fast，不做静默兼容；
    - 其他未知关键字 ⇒ ``TypeError`` fail-fast（不给静默透传留口）。
    """
    removed = [key for key in _REMOVED_FORWARD_BOOL_KWARGS if key in kwargs]
    if removed:
        raise ValueError(
            f"forward() 的布尔开关 {removed} 已删除（R39-A1；兼容 shim 不再保留）："
            f"请改用 forward_mode ∈ {FORWARD_MODES} —— "
            "rollout=False ∧ world_model=False ⇒ 'policy_single_step'；"
            "rollout=True ∧ world_model=False ⇒ 'policy_wm_rollout'；"
            "rollout=True ∧ world_model=True（含缺省全量前向）⇒ 'world_model_teacher_forcing'；"
            "诊断/可视化调用方 ⇒ 'diagnostics'。"
        )
    kwargs.pop("wm_detach", None)
    if kwargs:
        raise TypeError(
            f"forward() 收到未知参数 {sorted(kwargs)}；"
            f"模式开关请用 forward_mode ∈ {FORWARD_MODES}"
        )


# ---------------------------------------------------------------------- 运动学
def arc_step(ds: Tensor, dtheta: Tensor, eps: float = _ARC_EPS) -> tuple[Tensor, Tensor]:
    """``(ds, dθ)`` → 自车系圆弧位移 ``(dx, dy)``（恒曲率圆弧，起点原点、初始航向 0）。

    推导：对恒曲率 ``κ = dθ/ds``，半径 ``R = ds/dθ``，
    ``dx = R·sin(dθ)``、``dy = R·(1-cos(dθ))``（y 左向 ⇒ dθ>0 向左偏移）；
    ``|dθ| < eps`` 退化为直线 ``(ds, 0)``。
    """
    small = dtheta.abs() < eps
    safe = torch.where(small, torch.ones_like(dtheta), dtheta)
    radius = ds / safe
    dx = torch.where(small, ds, radius * torch.sin(dtheta))
    dy = torch.where(small, torch.zeros_like(ds), radius * (1.0 - torch.cos(dtheta)))
    return dx, dy


def compose_pose(pose: Tensor, dx: Tensor, dy: Tensor, dtheta: Tensor) -> Tensor:
    """把"当前帧自车系下的位移"叠加到自车系位姿上，返回新位姿 ``(B,3)``。"""
    theta = pose[..., 2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    new_x = pose[..., 0] + cos_t * dx - sin_t * dy
    new_y = pose[..., 1] + sin_t * dx + cos_t * dy
    return torch.stack([new_x, new_y, wrap_angle(theta + dtheta)], dim=-1)


def interpolate_actions(actions: Tensor, dt: float = 0.5, hz: float = 10.0) -> Tensor:
    """把 ``(B,K,2)`` 个 ``(ds,dθ)`` 圆弧插值为 ``(B,K·substeps,3)`` 位姿序列。

    与 p2-contract §1 ``env/tracking.interpolate`` 同约定：``substeps = round(dt·hz)``
    （默认 0.5 s × 10 Hz = 5），输出第 i 个点是第 ``i+1`` 个子步（0.1 s）结束时的位姿
    （自车系、起点原点，共 ``K·substeps`` 个点，末点即第 K 个动作的终点）。
    """
    substeps = int(round(float(dt) * float(hz)))
    if substeps < 1 or abs(substeps - float(dt) * float(hz)) > 1e-6:
        raise ValueError(f"dt·hz 必须为正整数，收到 dt={dt}, hz={hz}")
    batch, steps, _ = actions.shape
    pose = torch.zeros((batch, 3), dtype=actions.dtype, device=actions.device)
    poses: list[Tensor] = []
    for k in range(steps):
        ds = actions[:, k, 0]
        dtheta = actions[:, k, 1]
        for _ in range(substeps):
            dx, dy = arc_step(ds / substeps, dtheta / substeps)
            pose = compose_pose(pose, dx, dy, dtheta / substeps)
            poses.append(pose)
    return torch.stack(poses, dim=1)


def ego_next_features(
    ego_prev: Tensor, ds: Tensor, dtheta: Tensor, *, dt: float, prev_speed: Tensor
) -> Tensor:
    """由刚执行的动作解析构造下一帧 ego 特征（8 维）——**参考语义**。

    v2 的 plan head 直接用学习到的 head 预测 ego 特征前 6 维，后 2 维 reserved 写执行
    动作（与 p2-contract §8.4 的"上一动作占用保留维"一致）。本函数保留为解析参考
    （同样的运动学关系），供数据侧构造教师强制目标/一致性核对。
    """
    speed = ds / dt
    a_long = (speed - prev_speed) / dt
    yaw_rate = dtheta / dt
    a_lat = speed * yaw_rate
    return torch.stack(
        [speed, a_long, a_lat, yaw_rate, ego_prev[:, 4], ego_prev[:, 5], ds, dtheta], dim=-1
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """``(B,S,H)`` 按 ``(B,S)`` 掩码求均值；全无效时返回 0。

    v6（§1.2）：**只服务 plan head 融合路径**（``_plan`` 的 od_pool/ld_pool → fusion）；
    policy/value 的交叉注意力头直吃逐槽令牌，不再经过池化。
    """
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


def _warp_diag_steps(
    obs: Mapping[str, object], key: str, *, batch: int, steps: int, nodes: int, ref: Tensor
) -> Tensor | None:
    """R37 M1：``obs[key] (B,T,N,3)`` warp 诊断序列（缺省 None）→ 形状校验 + 投到 ``ref`` 设备。

    仅服务 5 项 hook 与分项读数（默认关路径不触碰）；形状不符 ⇒ ``ValueError``（fail-fast，
    不静默降级）。
    """
    raw = obs.get(key)
    if raw is None:
        return None
    tensor = torch.as_tensor(raw)
    if tuple(tensor.shape) != (batch, steps, nodes, 3):
        raise ValueError(
            f"obs[{key!r}] 形状应为 ({batch},{steps},{nodes},3)（B,T,N,3；缺省 = n/a），"
            f"收到 {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=ref.dtype, device=ref.device)


# ------------------------------------------------------------- A4：nav 逐步重建
def world_state_from_obs(
    obs: Mapping[str, object]
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """A4：从 obs 提取 ``(ego_world, route_world, route_world_mask)``；缺键返回全 None。

    优先委托 ``net.mem.world_state_from_obs``（env lane 实现，签名同）；未就绪时本地回退。
    """
    if _mem_world_state_from_obs is not None:
        return _mem_world_state_from_obs(obs)
    if not isinstance(obs, Mapping):
        return None, None, None
    ego_world = obs.get("ego_world")
    route_world = obs.get("route_world")
    if not torch.is_tensor(ego_world) or not torch.is_tensor(route_world):
        return None, None, None
    mask = obs.get("route_world_mask")
    return ego_world, route_world, mask if torch.is_tensor(mask) else None


def rebuild_nav_from_world(
    mem: MemBank,
    ego_pose_world: Tensor,
    route_world: Tensor,
    route_world_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """A4：世界系 route + ego 世界位姿 → 重建 nav ``(nav_features (B,NAV_DIM), nav_mask (B,1))``。

    **纯函数版**（不原地改写 ``mem.others``；v8 起 others 上下文为 t0 静态 token，
    nav 由独立 token 承载，不再回写 others）。
    这里用 ``net.mem.nav_features_from_world``（其纯函数内核）+ 相同回退语义：路线不可用
    行回退到 mem 当前帧的 nav（t0 冻结值），mask=0 行由调用方掩码。helper 未就绪时回退
    占位实现 = mem 当前帧 nav（t0 冻结语义）。
    """
    if _mem_nav_features_from_world is not None:
        feats, mask = _mem_nav_features_from_world(ego_pose_world, route_world, route_world_mask)
        valid = mask.reshape(-1) > 0.5
        if not bool(valid.all()):
            fallback = mem.others[:, -1, :NAV_DIM].detach()
            feats = torch.where(
                valid.unsqueeze(1), feats, fallback.to(dtype=feats.dtype, device=feats.device)
            )
        return feats, mask
    nav = mem.others[:, -1, :NAV_DIM].clone()
    mask = torch.ones((mem.batch, 1), dtype=nav.dtype, device=nav.device)
    return nav, mask


def advance_pose_world(pose_world: Tensor, ds: Tensor, dtheta: Tensor) -> Tensor:
    """A4：世界系位姿按一步动作 ``(ds,dθ)`` 推进（与教师强制同一步进语义）。

    优先委托 ``net.mem.advance_pose_world``；未就绪时回退 ``compose_pose`` 等价实现
    （``arc_step`` + SE(2) 合成，见 ``net.model`` 运动学节）。
    """
    if _mem_advance_pose_world is not None:
        return _mem_advance_pose_world(pose_world, ds, dtheta)
    dx, dy = arc_step(ds, dtheta)
    return compose_pose(pose_world, dx, dy, dtheta)



def advance_signal(
    signal_token: Tensor, signal_mask: Tensor, step_index: int
) -> tuple[Tensor, Tensor]:
    """A4：signal 时间推进占位（v6：恒为 unknown 占位）。

    预留 ``step_index``：接入红绿灯后在此按时间推进 signal 状态并重编码，net 侧调用
    接口不变（``_rollout`` 每步调用一次）。
    """
    return signal_token, signal_mask


# ---------------------------------------------------------------------- 主模型
class DrivingModel(nn.Module):
    """P2 驾驶模型（v2：mem-bank + 注意力聚合 + plan-head MoE + ST-GNN 递归 rollout）。"""

    def __init__(
        self,
        hidden: int = H,
        od_slots: int = OD_SLOTS,
        ld_slots: int = LD_SLOTS,
        history_frames: int = HISTORY_FRAMES,
        others_dim: int = DEFAULT_OTHERS_DIM,
        spatial_layers: int = 2,
        od_knn: int = 4,
        num_experts: int = 8,
        expert_hidden: int = 76,
        router_hidden: int = 384,
        primary_hidden: int = 768,
        moe_top_k: int = 2,
        wm_steps: int = 6,
        wm_dt: float = 0.5,
        action_low: tuple[float, float] = ACTION_LOW,
        action_high: tuple[float, float] = ACTION_HIGH,
        log_std_init: float = -1.0,
        attn_heads: int = 4,
        attn_layers: int = 1,
        trunk_hidden: int = 160,
        net_hidden: int = 256,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.history_frames = int(history_frames)
        self.others_dim = int(others_dim)
        self.rollout_steps = int(wm_steps)
        self.dt = float(wm_dt)

        self.encoders = ObsEncoders(hidden, od_slots, ld_slots, others_dim=others_dim)
        self.mem_encoder = MemEncoder(hidden)
        self.plan_head = PlanHead(
            hidden,
            num_experts,
            expert_hidden,
            router_hidden,
            primary_hidden=primary_hidden,
            top_k=moe_top_k,
        )
        self.st_gnn = SpatioTemporalGNN(
            hidden, od_slots, ld_slots, steps=self.rollout_steps, dt=self.dt,
            spatial_layers=spatial_layers, od_knn=od_knn,
        )
        # ---- R32 canary：ego 物理锚 decoder（**惰性构造**，与 ld_aggregator/semantic_wiring 同口径：
        # 默认不新增参数 → 既有 ckpt 键集/冻结审计/配方分组零变化；仅 P 臂训练/评测消费，不进 forward）----
        # ``set_ego_phys(True)`` 时构造 ``ego_phys(z_ego_next) → [delta_progress, delta_speed, delta_heading]``
        # （R33C v2 监督口径；R32 = ``[delta_progress, speed, delta_heading]``）。
        self.ego_phys: "nn.Sequential | None" = None
        #: R32 canary：ego 物理锚开关（普通属性，**不进 state_dict**；默认关）。
        self.enable_ego_phys = False
        # ---- Round-36 K1：StructuredPolicyHead（唯一动作生成器的结构化形式）----
        # **R38-P2b 双实体收编**：结构化头注册为 ``self.policy.structured_action``（键面
        # ``policy.structured_action.*``；旧 ``path_head.*`` 经 legacy_key_mapper 重映射）——
        # 单策略实体 = ``model.policy``（``structured_action`` 子结构；动作分布 / 辅助意图为其
        # 输出分组，见 ``net.path_head.StructuredPolicyHead.output_groups``）。
        # **惰性构造**（``set_path_head``；默认不注册/不新增参数 ⇒ 默认路径逐位不变；
        # 构造期保存/恢复 CPU RNG，不扰动既有初始化抽样序列）。
        # 每步调用：``_rollout`` 用当前 ``z_k`` 调同一子模块产出 ``a_k=(ds, κ_near·ds)``；
        # 旧 ``PolicyHead`` 仅服务默认关路径（不再是第二动作生成口）。
        #: R36 K1 结构化头启用开关（单状态：``set_path_head``；R38-P2b 删除 ``enable_path_head``
        #: 镜像属性与 ``path_head_ablation`` 模式枚举）。
        self._path_head_enabled: bool = False
        #: 结构化 rollout 的逐步调用 trace（默认 None = 关；门证据用，不参与前向数值）。
        self.path_head_trace: list[dict[str, object]] | None = None
        #: R36 L-A：运行时契约断言开关（**默认开**；仅 structured 模式 + wm rollout 模式；每
        #: forward 常数次整数比较 ⇒ 零可测开销；eval/deploy（``training=False``）违反 ⇒
        #: fail-fast RuntimeError；训练路径只计数落盘（``contract_*`` 输出键）。默认关路径不触碰。
        self.contract_asserts_enabled: bool = True
        #: 本次 forward 的 structured 头调用实例 id 收集（契约①；每次 forward 重置）。
        self._path_head_call_ids: list[int] = []
        #: 结构化 rollout 喂给 WM 的动作是否 stop-gradient（默认 True，梯度门：WM 预测 loss 只到 WM）。
        self.path_head_wm_action_detach: bool = True
        #: 训练期 target-lane token dropout 概率（默认 0 = 关；0 不消耗 RNG）。
        self.target_lane_dropout_p: float = 0.0
        #: Round-20 消融（**默认 baseline = 既有行为逐位不变**）：
        #: ``baseline`` | ``no_chain``（M1：关 LD↔LD 槽位链）| ``set_attn``（M2：关链 + LD set-attention）。
        #: M2 的注意力模块**惰性构造**（baseline 路径不新增参数 → 既有 ckpt 键集不变）。
        #: Round-22 新增 ``attn``（M2a）/``attn_no_chain``（M2b）：``LdAggregate`` 仅替换
        #: **pooled LD summary**（``ld_summary = mean + alpha·attn``、alpha₀=0），同样惰性构造。
        #: R21-3A 新增 ``semantic``：lane_semantic → target_lane_token + role-aware 分组读出
        #: （``net.semantic.SemanticWiring``，惰性构造；默认关 ⇒ baseline 逐位不变）。
        self.ld_mode: str = "baseline"
        #: R21-3D：rollout 阶段 semantic 注入口径（``full`` = R21-3A 现状）
        self.rollout_semantic_mode: str = "full"
        #: R37 M1：warp **主路径**口径（``off`` = 逐位不变；``analytic`` = 解析 warp +
        #: OD t0 场景系速度先验，见 :meth:`set_ego_warp_mode`）。
        #: R38-P2b：legacy ``ego_warp`` 属性（``w1``/``w2`` 第二状态机）已删除（`LEGACY_REPRO`）。
        self.ego_warp_mode: str = "off"
        #: R37 M1：5 项 warp hook 容器（默认 None ⇒ 零开销；见 :meth:`set_warp_hooks`）。
        self.warp_hooks: WarpHooks | None = None
        #: R37 M1：warp 分项读数开关（默认 False；诊断输出，训练/评测可关）。
        self.warp_readout: bool = False
        #: R38-D1：§四 ①..⑤ 强制 probes/hooks 统一容器（默认 None = 关 ⇒ 前向逐位一致、
        #: 零开销、不消耗 RNG；只读诊断，不参与数值；见 :meth:`set_probes`）。
        self.probes: ProbeRecorder | None = None
        self.ld_set_attn: "LdSetAttention | None" = None
        self.ld_aggregator: "LdAggregate | None" = None
        self.semantic_wiring: "SemanticWiring | None" = None
        #: R21-25A：残差式语义接入（β 初值 0 ⇒ 载入旧 ckpt 后逐位 = baseline；
        #: 惰性构造，默认关 ⇒ 旧 ckpt 键集不变）。target_proj：target token → summary 槽的
        #: 投影（bias 零初始化 ⇒ β=0 时贡献严格 0；R21-25B「必要 target-token projection」）。
        self.beta_context: "nn.Parameter | None" = None
        self.beta_token: "nn.Parameter | None" = None
        self.target_proj: "nn.Linear | None" = None
        # v6 A1：policy/value = 交叉注意力头（K=1 查询 × 令牌集合；命名保 policy./value. 前缀，
        # 见 STAGE_C_DESIGN_PREFIXES 的 allowlist 契约）
        self.policy = PolicyHead(
            hidden,
            action_low,
            action_high,
            log_std_init=log_std_init,
            num_heads=int(attn_heads),
            layers=int(attn_layers),
            trunk_hidden=int(trunk_hidden),
        )
        self.value = ValueHead(
            hidden,
            num_heads=int(attn_heads),
            layers=int(attn_layers),
            net_hidden=int(net_hidden),
        )

    def set_ld_mode(self, mode: str = "baseline") -> "DrivingModel":
        """LD 消融开关（默认 ``baseline`` = 逐位不变）。

        - ``no_chain``（M1）：``st_gnn.spatial.include_ld_chain=False``（仅移除 LD↔LD 槽位链边）；
        - ``set_attn``（Round-20 M2，保留但不再用于实验）：再置 ``exclude_ld_nodes=True``
          （LD 绕过 MP、独立编码）并**惰性构造** :class:`net.spatial.LdSetAttention`；
        - ``attn``（Round-22 M2a）：**保留 LD-LD 链与 LD 节点 MP**，惰性构造
          :class:`net.spatial.LdAggregate`（``alpha`` 初值 0）——仅替换 pooled LD summary
          （``plan_step`` 融合输入 + ``encode`` 的 head-token summary）；
        - ``attn_no_chain``（Round-22 M2b）：M2a 单变量增量（**仅**额外关 LD-LD 链）；
        - ``semantic``（R21-3A）：惰性构造 :class:`net.semantic.SemanticWiring`
          （lane_semantic → target token + 分组读出）；LD-LD 链与 LD 节点 MP 保持 baseline
          口径。该模式下 ``plan_step`` **禁止** ``masked_mean`` 回退（缺 context 报错），
          且必须提供 ``lane_semantic``/LD companion 输入（旧数据 fail-fast）。
        - ``semantic_no_chain``（R21-3D 单变量）：semantic 接线 + **仅**关闭 LD-LD 槽位链
          （``include_ld_chain=False``；LD 节点 MP 与其余一切不动）。round-24 倾向默认。
        """
        mode = str(mode).lower()
        if mode not in ("baseline", "no_chain", "set_attn", "attn", "attn_no_chain", "semantic", "semantic_no_chain"):
            raise ValueError(
                f"未知 ld_mode={mode!r}（baseline | no_chain | set_attn | attn | attn_no_chain | "
                "semantic | semantic_no_chain）"
            )
        spatial = self.st_gnn.spatial
        #: Round-20 ``set_attn`` 语义保持原样（关链 + LD 排除 MP）；Round-22 ``attn`` 保留两者，
        #: ``attn_no_chain`` 仅关链（单变量）；R21-3A ``semantic`` 与 baseline 同（链/MP 不动）；
        #: R21-3D ``semantic_no_chain`` = semantic + 仅关 LD-LD 链。
        spatial.include_ld_chain = mode in ("baseline", "attn", "semantic")
        spatial.exclude_ld_nodes = mode == "set_attn"
        self.ld_mode = mode
        if mode == "set_attn" and self.ld_set_attn is None:
            hidden = int(getattr(self.plan_head, "hidden_dim", 0) or 0)
            if hidden <= 0:
                hidden = int(self.st_gnn.hidden)
            self.ld_set_attn = LdSetAttention(hidden, heads=4, ctx_dim=3 * hidden).to(
                next(self.parameters()).device
            )
        if mode in ("attn", "attn_no_chain") and self.ld_aggregator is None:
            hidden = int(getattr(self.plan_head, "hidden_dim", 0) or 0)
            if hidden <= 0:
                hidden = int(self.st_gnn.hidden)
            self.ld_aggregator = LdAggregate(hidden, heads=4, ctx_dim=3 * hidden).to(
                next(self.parameters()).device
            )
        if mode in ("semantic", "semantic_no_chain") and self.semantic_wiring is None:
            hidden = int(getattr(self.plan_head, "hidden_dim", 0) or 0)
            if hidden <= 0:
                hidden = int(self.st_gnn.hidden)
            device = next(self.parameters()).device
            self.semantic_wiring = SemanticWiring(hidden).to(device)
            # R21-25A：残差门控（标量 Parameter，初值 0；进 optimizer 账本）
            self.beta_context = nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
            self.beta_token = nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
            # target token → summary 槽投影（bias 零初始化 ⇒ β=0 时严格 0 贡献）
            proj = nn.Linear(hidden, hidden).to(device)
            nn.init.zeros_(proj.bias)
            nn.init.normal_(proj.weight, std=0.02)
            self.target_proj = proj
        return self

    def _is_semantic(self) -> bool:
        """semantic 系模式（``semantic`` / ``semantic_no_chain``）统一判定。"""
        return self.ld_mode in ("semantic", "semantic_no_chain")

    def set_rollout_semantic(self, mode: str = "full") -> "DrivingModel":
        """R21-3D：rollout 阶段 semantic 注入口径（**仅推理/训练行为开关，非参数**）。

        - ``full``（默认，= R21-3A 现状）：rollout 每步用当步 latent + **t0 语义全量**
          （lat/heading/margins/distance 等动态量逐步重复注入）；
        - ``t0_only``：语义只在 t0 计算一次；rollout 1–5 步直接复用 **t0 的 geometry
          context / target token**（不重复注入动态语义）；
        - ``static_role_only``：t0 用全量语义；rollout 1–5 步仅保留**静态**语义
          （curvature/valid/conf/lane-change/拓扑身份），**动态**（lat/heading/margins/
          distance_to_lane_end/distance_to_target_lane_end）置零后重建 context
          （round-24 §3.2 倾向口径）。
        """
        mode = str(mode).lower()
        if mode not in ("full", "t0_only", "static_role_only"):
            raise ValueError(f"未知 rollout_semantic={mode!r}（full | t0_only | static_role_only）")
        self.rollout_semantic_mode = mode
        return self

    def set_ego_warp_mode(self, mode: str = "off") -> "DrivingModel":
        """R37 M1：warp **主路径**口径（纯推理行为开关，非参数；默认 ``off`` = 逐位不变）。

        - ``off``（默认）：rollout 图位姿 = 现行 t0 帧口径（OD 先验用通道**相对速度** =
          随体坐标语义）——经 :func:`net.ego_warp.assemble_pose_k` ``mode="off"`` 逐位复现旧
          内联分支；
        - ``analytic``：解析 ego-motion warp（``assemble_pose_k`` ``analytic``；op 序 = 旧 ``w2``）
          ——t0 系当**静态场景**，OD 先验速度 = **t0 场景系对象速度** ``v_OD = v_rel + v_ego``
          （:func:`net.ego_warp.od_velocity_t0`；``v_ego = enc.ego_now[..., 0]`` 沿自车 x 轴），
          LD 静止锚做 ego SE(2) 逆变换，整图重表达进 ego_k 帧（ego 槽恒 ``(0,0,0)``）。
          **禁止**用相对速度推进后再扣自车运动（系统性重复扣 ``|Δp|``；R36 L-D 实测同速前车
          P90 1.95 m → 0.43 m）。几何自洽不变量：任意两节点距离/相对航向不变。

        R38-P2b：legacy ``set_ego_warp``（``w1``/``w2``）已删除（`LEGACY_REPRO`）——生产 net
        只有这一套 warp 状态机；传入 ``w1``/``w2`` ⇒ ``ValueError`` fail-fast，逐位参照物 =
        ``tests/fixtures/r38_p2b_warp/``（legacy 纯函数快照 = ``tools/legacy/legacy_ego_warp.py``）。
        """
        mode = str(mode).lower()
        if mode not in ("off", "analytic"):
            raise ValueError(
                f"未知 ego_warp_mode={mode!r}（off | analytic）；legacy w1/w2 已退役（R38-P2b）"
                "——逐位参照 = tests/fixtures/r38_p2b_warp/，纯函数快照 = "
                "tools/legacy/legacy_ego_warp.py"
            )
        self.ego_warp_mode = mode
        return self

    def set_warp_hooks(self, hooks: WarpHooks | None = None) -> "DrivingModel":
        """R37 M1：装载 5 项 warp hook 容器（默认 ``None`` = 关；只读诊断，不参与数值）。

        ``hooks.enabled=True`` 时 rollout 每步经 :meth:`net.ego_warp.WarpHooks.capture` 记录
        ① ``anchor_pre``（warp 前 t0 帧）/② ``anchor_analytic``（ego_k 帧）/③
        ``anchor_actual_next``（真实下一帧；``obs['warp_anchor_actual_next']``，缺省 n/a）/④
        ``residual_learned``（``obs['warp_anchor_residual_learned']``，缺省 None = 0，接口留给
        WM residual 训练）/⑤ ``anchor_final``（= ② + ④）。``enabled=False`` ⇒ 零张量操作、
        不消耗 RNG、前向逐位不变。
        """
        self.warp_hooks = hooks
        return self

    def set_warp_readout(self, enabled: bool = True) -> "DrivingModel":
        """R37 M1：分项读数开关（诊断输出；默认关，训练/评测可关）。

        开启后（且提供 ``obs['warp_anchor_actual_next']``）rollout 返回逐步位置误差
        ``warp_prior_err``（解析先验 vs 真实）/``warp_residual_contrib``（学习残差贡献 = ⑤−②）/
        ``warp_final_err``（最终 anchor vs 真实），``(B,T)`` 掩码均值 [m]；缺真实下一帧
        （部署侧）⇒ ``warp_readout_valid=0``（n/a，不产出误差键）。
        """
        self.warp_readout = bool(enabled)
        return self

    def set_probes(self, probes: ProbeRecorder | None = None) -> "DrivingModel":
        """R38-D1：装载 ①..⑤ 强制 probes/hooks 统一容器（默认 ``None``/``enabled=False`` = 关）。

        默认关 ⇒ 全链零短路（每次 ``capture`` 首行返回）：前向逐位一致、输出键集不变、
        不消耗额外随机数、不改 optimizer/ckpt（Gate A 自证：``d1_off_bitwise.py`` 双跑
        HEAD/改动后 net 包 ``max|Δ|=0``）。开启后每步经 :mod:`net.probes` 记 ①状态来源 /
        ②Policy / ③World Model（warp 五项仍由 :class:`net.warp_hooks` 单列）/ ④Memory /
        ⑤六步调用链（``probes.chain.st_gnn_total_calls`` = **ST-GNN 总调用**单列）；
        输出 dict 契约不变，探针经 :meth:`net.probes.ProbeRecorder.write` 统一出口落盘。
        传容器时同步 ⑤ 期望 ST-GNN 总调用 = ``rollout_steps``。
        """
        self.probes = probes
        if probes is not None:
            probes.chain.expected_st_gnn_calls = int(self.rollout_steps)
        return self

    #: ``static_role_only`` 下 rollout 后段置零的**动态**语义特征索引
    #: （lat/heading/margins/distance_to_lane_end/distance_to_target_lane_end）。
    #: 静态 = curvature(2,3)/valid(6)/conf(7)/change(8,9) + 拓扑身份（role/route/group）。
    SEMANTIC_DYNAMIC_FEATURES: tuple[int, ...] = (0, 1, 4, 5, 10, 11)

    def _rollout_step_geometry(
        self,
        z_ld: Tensor,
        ld_live: Tensor,
        z_ego: Tensor,
        nav_token: Tensor,
        others_ctx: Tensor,
        *,
        semantic: Mapping[str, Tensor] | None,
        t0_delta: Tensor | None,
        t0_target: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """R21-25A：rollout 单步 plan 融合 context = **legacy(z_ld_k) + β_context·delta_k**。

        rollout 口径只门控**语义 delta**（基底始终逐 step legacy —— 保证 β=0 与 baseline 逐位）：

        - ``full``：delta_k = wiring(z_ld_k, t0 语义全量)；
        - ``t0_only``：delta_k = t0 delta（不重复注入动态语义）；
        - ``static_role_only``：delta_k = wiring(z_ld_k, t0 语义的**静态部分**；
          动态 lat/heading/margins/distance 置零)。
        """
        legacy = self.aggregate_ld(z_ld, ld_live, z_ego, nav_token, others_ctx)
        mode = self.rollout_semantic_mode
        if mode == "t0_only":
            delta, raw_target = t0_delta, t0_target
        elif mode == "static_role_only" and semantic is not None:
            static_semantic = dict(semantic)
            feats = semantic["lane_semantic"].clone()
            for index in self.SEMANTIC_DYNAMIC_FEATURES:
                feats[:, index] = 0.0
            static_semantic["lane_semantic"] = feats
            delta, raw_target = self.semantic_wiring(z_ld, ld_live, **static_semantic)
        else:
            delta, raw_target = self.semantic_wiring(z_ld, ld_live, **semantic)
        context = legacy + self.beta_context * delta if delta is not None else legacy
        target = self.beta_token * raw_target if raw_target is not None else None
        return context, target

    def stgnn_flags(self) -> tuple[bool, bool]:
        """``(include_ld_chain, exclude_ld_nodes)``——LD 邻接/消息传递开关的只读视图（诊断用）。"""
        spatial = self.st_gnn.spatial
        return bool(spatial.include_ld_chain), bool(spatial.exclude_ld_nodes)

    def aggregate_ld(
        self,
        z_ld: Tensor,
        ld_live: Tensor,
        ego_token: Tensor,
        nav_token: Tensor,
        others_ctx: Tensor | None = None,
    ) -> Tensor:
        """Round-22：**LD pooled summary 的统一读出**（t0 / rollout / policy 同一入口）。

        - ``baseline``/``no_chain``/``set_attn``（未接聚合器）：``masked_mean(z_ld, ld_live)``
          —— 既有路径逐位不变；
        - ``attn``/``attn_no_chain``：:class:`net.spatial.LdAggregate`
          （``mean + alpha·attn(q=cat[ego,nav,others], kv=z_ld)``；alpha₀=0 ⇒ 逐位 = mean）。
        """
        aggregator = self.ld_aggregator
        if aggregator is not None and self.ld_mode in ("attn", "attn_no_chain"):
            return aggregator(z_ld, ld_live, ego_token, nav_token, others_ctx)
        return _masked_mean(z_ld, ld_live)

    # ------------------------------------------------- R21-3A：语义接线（唯一入口）
    def semantic_parameter_names(self) -> tuple[str, ...]:
        """semantic 接线新增的参数/buffer 名（旧 ckpt fail-fast 白名单）。"""
        if self.semantic_wiring is None:
            return ()
        names = [f"semantic_wiring.{name}" for name in self.semantic_wiring.parameter_names()]
        names += ["beta_context", "beta_token"]
        if self.target_proj is not None:
            names += [f"target_proj.{name}" for name, _ in self.target_proj.named_parameters()]
        return tuple(sorted(names))

    def _semantic_inputs_from_obs(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """obs → semantic 接线输入（current frame；缺键/形状不符直接报错 = 旧数据 fail-fast）。

        契约（env schema v7）：``lane_semantic (B,1,12)``、``lane_semantic_mask (B,1)``、
        ``ld_lane_role (B,16) int64``、``ld_offset_s_norm (B,16)``、``ld_route_associated (B,16)``、
        ``ld_group (B,16) int64``。缺任一键 → ``ValueError``（**不得静默零填充**）。
        """
        required = (
            "lane_semantic",
            "lane_semantic_mask",
            "ld_lane_role",
            "ld_offset_s_norm",
            "ld_route_associated",
            "ld_group",
        )
        missing = [name for name in required if not torch.is_tensor(obs.get(name))]
        if missing:
            raise ValueError(
                f"semantic 模式缺少观测键 {missing}（schema v7 语义通道/LD companion）；"
                "旧 schema/dataset 必须 fail-fast 而非静默回退（round-23 §6A-7）"
            )
        lane_semantic = squeeze_batch_singletons(obs["lane_semantic"].float(), 2, "lane_semantic")
        if tuple(lane_semantic.shape) != (int(lane_semantic.shape[0]), 12):
            raise ValueError(f"obs['lane_semantic'] 形状应为 (B,12)，收到 {tuple(lane_semantic.shape)}")
        batch = int(lane_semantic.shape[0])
        lane_mask = self._normalize_mask(obs["lane_semantic_mask"], batch, lane_semantic.device, "lane_semantic_mask")
        role = obs["ld_lane_role"]
        group = obs["ld_group"]
        if role.ndim != 2 or int(role.shape[0]) != batch:
            raise ValueError(f"obs['ld_lane_role'] 形状应为 (B,L)，收到 {tuple(role.shape)}")
        return {
            "lane_semantic": lane_semantic,
            "lane_semantic_mask": lane_mask,
            "role": role.to(torch.int64),
            "offset_s": squeeze_batch_singletons(obs["ld_offset_s_norm"].float(), 2, "ld_offset_s_norm"),
            "route_assoc": squeeze_batch_singletons(obs["ld_route_associated"].float(), 2, "ld_route_associated"),
            "group": group.to(torch.int64),
        }

    def build_geometry_context(
        self,
        z_ld: Tensor,
        ld_live: Tensor,
        z_ego: Tensor,
        nav_token: Tensor,
        others_ctx: Tensor | None = None,
        *,
        semantic: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """**统一几何读出入口**（PlanHead t0 + rollout 1–6 + policy/value head tokens）。

        R21-25A **残差式接入**（round-25 §2）：

        - ``legacy = aggregate_ld(...)``（baseline = masked_mean；attn 模式 = LdAggregate）——
          作为**基底**（注意：这反转了 round-23 的"semantic 禁 masked_mean"语义）；
        - ``geometry_context = legacy + beta_context · semantic_delta``；
        - ``target_token_eff = beta_token · target_lane_token``（无效语义 ⇒ token 严格 0）；
        - ``beta_*`` 初值 0 ⇒ 载入旧 ckpt 后与 baseline **逐位一致**（Gate A）。
        - ``semantic`` 模式必须提供 ``semantic`` 输入（否则报错，无静默回退）。
        """
        if self._is_semantic():
            if semantic is None or self.semantic_wiring is None:
                raise RuntimeError(
                    "semantic 模式必须提供 lane_semantic 输入（build_geometry_context semantic=None）"
                    "——禁止静默回退（round-25 §2；残差基底 = legacy aggregate_ld）"
                )
            delta, target_token = self.semantic_wiring(z_ld, ld_live, **semantic)
            legacy = self.aggregate_ld(z_ld, ld_live, z_ego, nav_token, others_ctx)
            context = legacy + self.beta_context * delta
            target_eff = self.beta_token * target_token
            return context, target_eff
        return self.aggregate_ld(z_ld, ld_live, z_ego, nav_token, others_ctx), None

    def set_moe(self, *, enabled: bool = True, load_balance_coef: float = 0.0) -> "DrivingModel":
        """MoE 运行时开关 + 负载均衡 α（lane U1；非参数，不进 state_dict）。

        phase 1（primary）关闭：专家不参与、输出严格 = primary；
        phase 2（specific）打开：``out = primary + Σ_{i∈top2} g_i·expert_i``（全场景生效）。
        推理默认开启（与训练 phase 2 口径一致）。
        """
        self.plan_head.set_moe(enabled=enabled, load_balance_coef=load_balance_coef)
        return self

    # ---------------------------------------------------------------- R32 canary 开关
    def set_action_ego(self, enabled: bool = True) -> "DrivingModel":
        """R33C v2：ego 动作条件化开关（委托 ``st_gnn``；**惰性构造**，默认关 = 与 HEAD 逐位一致）。

        ``enabled=True`` 首次调用才在 ``st_gnn`` 内构造 ``action_ego``（(H+3)→H→H，隐藏层
        正常初始化 + 输出层零初始化，**残差增益固定 1**；R33C v2 已删可学习 ``alpha_ego``）
        ⇒ 默认模型不新增参数（旧 ckpt strict 加载 0/0/0；同 ``ego_phys``/``semantic_wiring`` 口径）。
        """
        self.st_gnn.set_action_ego(bool(enabled))
        return self

    def set_ego_phys(self, enabled: bool = True) -> "DrivingModel":
        """ego 物理锚 decoder 开关（R32 canary；**惰性构造**；仅训练/评测消费，不进 forward）。

        输出 3 维物理量：R33C v2 训练目标 = ``[delta_progress, delta_speed, delta_heading]``
        （R32 = ``[delta_progress, speed, delta_heading]``；模块结构/键集不变，仅监督口径更新，
        见 ``.slim/deepwork/p1_v2/round33/canary_v2_train.py``）。默认关 ⇒ 模型不新增参数
        （既有 ckpt 键集 / 冻结审计 / 配方 LR 分组零变化）。构造期保存/恢复 CPU RNG 状态
        （不扰动既有初始化抽样序列）。
        """
        self.enable_ego_phys = bool(enabled)
        if enabled and self.ego_phys is None:
            device = next(self.parameters()).device
            rng_state = torch.random.get_rng_state()
            self.ego_phys = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, 3),
            ).to(device)
            torch.random.set_rng_state(rng_state)
        return self

    # ---------------------------------------------------------------- R36 K1 开关
    @property
    def path_head(self) -> "StructuredPolicyHead | None":
        """R38-P2b 兼容别名（只读）：结构化头 = ``self.policy.structured_action``（收编后单实体）。

        旧 ``path_head.*`` 键面已 fold 到 ``policy.structured_action.*``（legacy_key_mapper
        负责旧 ckpt 键名重映射）。本属性保留为**读访问**迁移 alias（.slim 评估链 / probes /
        测试）；写入（``model.path_head = ...``）⇒ ``AttributeError`` fail-fast——装配必须经
        :meth:`set_path_head`（构造参数），不再有第二实体开关。
        """
        return self.policy._modules.get("structured_action")  # type: ignore[return-value]

    def set_path_head(
        self,
        enabled: bool = True,
        *,
        horizon: float = LAT_HORIZON_M,
        mode: str | None = None,
    ) -> "DrivingModel":
        """Round-36 K1：``StructuredPolicyHead`` 开关（**惰性构造**；默认关 = 与 HEAD 逐位一致）。

        ``enabled=True`` 首次调用才构造 :class:`net.path_head.StructuredPolicyHead`
        （横向分支 + 纵向分支 + 确定性组合层 + 两个几何特征编码层；``compose`` 无参数），
        注册为 ``self.policy.structured_action``（键面 ``policy.structured_action.*``）；
        构造期保存/恢复 CPU RNG ⇒ 不扰动既有初始化抽样序列（同 ``ego_phys`` 口径）。

        R38-P2b：``enable_path_head`` 镜像属性已删除（单状态 = ``self._path_head_enabled`` +
        ``self.policy.structured_action``）；``mode`` 形参保留走 :meth:`set_path_head_ablation`
        legacy 入口（``policy``/``structured``）。

        启用后 ``forward`` 要求 ``obs['lane_anchor_xy']``（给定目标车道中心线折线），并走
        **逐步调用**口径：``_rollout`` 每步用当前 ``z_k`` 调同一头 → ``a_k=(ds_k, κ_near_k·ds_k)``
        → WM 单步推进 → 同一头重算；6 步 ``plan`` 由逐步动作组装（见
        ``docs/architecture_contract.md``）。``enabled=False`` 只关闭消费路径（已构造的头
        参数保留，不进 forward）。
        """
        self._path_head_enabled = bool(enabled)
        if mode is not None:
            self.set_path_head_ablation(mode)
        if enabled and self.policy._modules.get("structured_action") is None:
            device = next(self.parameters()).device
            rng_state = torch.random.get_rng_state()
            head = StructuredPolicyHead(self.hidden, horizon=float(horizon)).to(device)
            self.policy.structured_action = head  # type: ignore[assignment]
            torch.random.set_rng_state(rng_state)
        return self

    def set_path_head_ablation(
        self, mode: str = PATH_HEAD_DEFAULT_MODE
    ) -> "DrivingModel":
        """R38-P2b：**legacy 诊断入口**（原消融脚手架模式枚举已删除；单实体 = ``model.policy``）。

        - ``structured``（生产口径）：开启结构化头消费路径（``set_path_head(True)`` 等价；
          头未构造时仅登记开关，构造仍由 ``set_path_head`` 惰性完成）；
        - ``policy``（消融①，历史口径）：``set_path_head(False)``（关闭消费路径 = 默认关路径）。
          注意：R38-P2b 起 structured 模式旧头**永不前向**（诊断已离线化，见
          :meth:`set_policy_diag`）；``policy`` 只是"不启用结构化头"的历史名称。

        一次性六步诊断臂（H1）已于 R38-P1 删除（第二个六步轨迹生成器）⇒ 写入该模式名一律
        ``ValueError`` fail-fast，历史复现见 ``tools/legacy/oneshot_path.py``。
        """
        if mode == "oneshot":
            raise ValueError(
                "path_head_ablation='oneshot'（H1 一次性六步诊断臂）已在 R38-P1 减枝删除"
                "（ARCHITECTURE_FORBIDDEN：唯一动作源 + 每步单动作）；仅支持 policy | structured。"
                "历史复现见 tools/legacy/oneshot_path.py。"
            )
        if mode not in PATH_HEAD_MODES:
            raise ValueError(f"path head ablation 必须是 {PATH_HEAD_MODES} 之一，收到 {mode!r}")
        self._path_head_enabled = str(mode) == "structured"
        return self

    def set_path_head_trace(self, enabled: bool = True) -> "DrivingModel":
        """结构化 rollout 逐步调用 trace 开关（门证据用；不改变任何前向数值）。

        开启后 ``forward`` 会重置 ``self.path_head_trace`` 并记录每次头调用
        ``{step, latent, action, ds, kappa_near, wm_calls_before, wm_advance_calls}``
        （step 0 = t0 编码期调用；step k = rollout 第 k 步调用）。默认关 ⇒ 无开销。
        R38-P2b：trace 不再隐式打开旧头诊断前向（诊断已离线化；见 :meth:`set_policy_diag`）。
        """
        self.path_head_trace = [] if enabled else None
        return self

    def set_policy_diag(self, enabled: bool = True) -> "DrivingModel":
        """R38-P2b：旧 ``PolicyHead`` 诊断前向**已离线化**——本入口 fail-fast。

        原语义（R36 L-A）：structured 模式下显式恢复旧头前向以产出 ``policy_action_mu`` /
        ``action_logstd``（诊断键）。R38-P2b 双实体收编后：正式 forward **只有一个动作出口**
        （structured ⇒ 结构化头；默认关 ⇒ 旧头），旧头诊断改为离线工具
        ``tools/diagnostics/legacy_policy_diag.py``（不占主 forward、不并行计算）。
        """
        raise RuntimeError(
            "set_policy_diag 已离线化（R38-P2b 双实体收编：正式 forward 只保留一个动作出口）："
            "旧头诊断请用离线工具 tools/diagnostics/legacy_policy_diag.py "
            "（policy_action_mu/action_logstd 离线复算，不改变任何前向数值）。"
        )

    def set_contract_asserts(self, enabled: bool = True) -> "DrivingModel":
        """R36 L-A：structured 运行时契约断言开关（**默认开**；GPT ``7dcf6bd`` §四.1/§四.8）。

        断言项（仅 ``path_head_mode()=="structured"`` 且 wm rollout 模式时评估）：

        ① ``policy_instance_ids_are_identical``：a0..a5 来自同一 ``StructuredPolicyHead`` 实例；
        ② ``policy_call_count_is_six``：6 次头调用；③ ``world_model_call_count_is_five``：
        WM 推进 5 次；④ ``auxiliary_future_outputs_not_consumed_by_lqr``：``plan`` 由逐步头动作
        组装（``plan_from_aux=0``）且 ``plan[:,0]==action_mu``（进环境/LQR 的动作源唯一）；
        ⑤ ``deployment_does_not_use_plan_override``：structured 模式的一次性钉住路径**不存在**
        （结构断言语义：历史模式名或 ``one_shot_plan`` 复活 ⇒ 该门失败；见 :meth:`_rollout`）。

        eval/deploy 路径（``training=False``）违反 ⇒ ``RuntimeError`` fail-fast；训练路径
        **不抛**，仅以 ``contract_*`` 输出键计数落盘。开销 = 每 forward 常数次整数/张量比较
        （无逐 step 钩子）；``set_contract_asserts(False)`` 关闭后不收集、不产出这些键。
        默认关路径（path head 未启用）不受影响。
        """
        self.contract_asserts_enabled = bool(enabled)
        return self

    def path_head_mode(self) -> str:
        """当前生效的动作生成模式：``policy``（默认关）| ``structured``（生产）。

        R38-P2b：模式由**派生**得到（``self.policy.structured_action`` 是否构造 +
        ``_path_head_enabled``）；不再有 ``path_head_ablation`` 模式枚举属性。
        """
        if self.policy._modules.get("structured_action") is None or not self._path_head_enabled:
            return "policy"
        return "structured"

    def action_generator_names(self) -> tuple[str, ...]:
        """动作生成器模块名（架构契约门「生成器数 = 1」的静态断言入口）。

        - ``policy``：``("policy",)``（旧 PolicyHead；默认关路径）；
        - ``structured``：``("policy.structured_action",)``（收编后单实体内的结构化头子结构）。
        """
        return ("policy",) if self.path_head_mode() == "policy" else ("policy.structured_action",)

    def set_target_lane_dropout(self, p: float = 0.0) -> "DrivingModel":
        """训练期 target-lane token dropout 概率（默认 0 = 关；非参数，不进 state_dict）。

        ``p > 0`` 且 ``self.training`` 且存在 target token（semantic 模式）时按**样本**独立
        伯努利置零 ``target_lane_token``（``keep = rand(B) >= p``；rollout 各步共用 t0 掩码）。
        掩码经 ``encode()['target_lane_dropout_keep']`` / 前向 ``target_lane_dropout_keep``
        暴露。**默认 0 ⇒ 不消耗 RNG、无行为变化**；``eval()`` 下不生效（确定性评测）。
        """
        p = float(p)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"target-lane dropout 概率必须在 [0,1)，收到 {p}")
        self.target_lane_dropout_p = p
        return self

    def path_head_parameter_names(self) -> tuple[str, ...]:
        """path head（惰性构造）新增参数/buffer 名；默认关 → 空元组（旧 ckpt strict 加载不受影响）。

        R38-P2b：键面前缀 = ``policy.structured_action.``（收编后命名空间）。
        """
        head = self.policy._modules.get("structured_action")
        if head is None:
            return ()
        names = [f"policy.structured_action.{name}" for name, _ in head.named_parameters()]
        names += [f"policy.structured_action.{name}" for name, _ in head.named_buffers()]
        return tuple(sorted(names))

    # ---------------------------------------------------------------- 输入解析
    def _mem_from_obs(self, obs: Mapping[str, Tensor]) -> MemBank:
        """obs → 真 mem（v2 键名 + 回退，见 :func:`net.mem.mem_from_obs`）。只读，不写回。"""
        mem = mem_from_obs(obs, others_dim=self.others_dim, history_frames=self.history_frames)
        if int(mem.od.shape[2]) != self.od_slots or int(mem.ld.shape[2]) != self.ld_slots:
            raise ValueError(
                f"mem 槽位数与模型不匹配：od {int(mem.od.shape[2])}/{self.od_slots}、"
                f"ld {int(mem.ld.shape[2])}/{self.ld_slots}"
            )
        return mem

    @staticmethod
    def _normalize_mask(mask: Tensor | None, batch: int, device: torch.device, name: str) -> Tensor:
        """上下文掩码 → ``(B,1)``（缺省 = 全 1）。"""
        if mask is None:
            return torch.ones((batch, 1), dtype=torch.float32, device=device)
        mask = mask.float()
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        mask = squeeze_batch_singletons(mask, 2, name)
        if tuple(mask.shape) != (batch, 1):
            raise ValueError(f"obs[{name!r}] 形状应为 (B,1)，收到 {tuple(mask.shape)}")
        return mask

    def embed_context(
        self,
        nav: Tensor | None = None,
        nav_mask: Tensor | None = None,
        signal: Tensor | None = None,
        signal_mask: Tensor | None = None,
        *,
        batch: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """raw nav/signal（``(B,11)``/``(B,4)``，可带单例维）→ 编码 token + mask 四元组。

        返回 ``(nav_token (B,H), nav_mask (B,1), signal_token (B,H), signal_mask (B,1))``。
        公开入口：teacher forcing / A4 逐步重建 nav 的调用点用它把重建出的 nav 特征重新
        编码为 token（``plan_step`` 消费），无需触碰 ``encoders``。缺失通道 → 0 token +
        0 mask（编码器输出 0，掩码路径保证不污染输出）。
        """
        for tensor in (nav, signal):
            if torch.is_tensor(tensor) and batch is None:
                batch = int(tensor.shape[0])
            if torch.is_tensor(tensor) and device is None:
                device = tensor.device
        if batch is None:
            raise ValueError("embed_context 需要 nav/signal 张量或显式 batch")
        if device is None:
            device = torch.device("cpu")

        if torch.is_tensor(nav):
            nav = squeeze_batch_singletons(nav.float(), 2, "nav")
            if tuple(nav.shape) != (batch, NAV_DIM):
                raise ValueError(
                    f"obs['nav'] 形状应为 (B,{NAV_DIM})（或 (B,1,{NAV_DIM})），收到 {tuple(nav.shape)}"
                )
            nav_mask = self._normalize_mask(nav_mask, batch, nav.device, "nav_mask")
            nav_token = self.encoders.embed_nav(nav.unsqueeze(1), nav_mask.reshape(-1, 1))[:, 0]
        else:
            nav_token = torch.zeros((batch, self.hidden), dtype=torch.float32, device=device)
            nav_mask = torch.zeros((batch, 1), dtype=torch.float32, device=device)

        if torch.is_tensor(signal):
            signal = squeeze_batch_singletons(signal.float(), 2, "signal")
            if tuple(signal.shape) != (batch, SIGNAL_DIM):
                raise ValueError(
                    f"obs['signal'] 形状应为 (B,{SIGNAL_DIM})（或 (B,1,{SIGNAL_DIM})），收到 {tuple(signal.shape)}"
                )
            signal_mask = self._normalize_mask(signal_mask, batch, signal.device, "signal_mask")
            signal_token = self.encoders.embed_signal(
                signal.unsqueeze(1), signal_mask.reshape(-1, 1)
            )[:, 0]
        else:
            signal_token = torch.zeros((batch, self.hidden), dtype=torch.float32, device=device)
            signal_mask = torch.zeros((batch, 1), dtype=torch.float32, device=device)
        return nav_token, nav_mask, signal_token, signal_mask

    def _context_tokens(
        self, obs: Mapping[str, Tensor], batch: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """obs 的 nav/signal（可选；v2 的规范上下文在 ``others`` mem 里）→ token+mask 四元组。

        v6 §5 #4：同时返回 nav/signal mask，供交叉注意力头的 key mask；缺失通道用
        0 token + 无效掩码（不报错——避免与 env 的通道裁剪策略耦合）。
        """
        return self.embed_context(
            obs.get("nav"),
            obs.get("nav_mask"),
            obs.get("signal"),
            obs.get("signal_mask"),
            batch=batch,
            device=obs["hist_valid"].device,
        )

    # ---------------------------------------------------------------- 编码/规划
    def plan_step(
        self,
        z_ego: Tensor,
        z_od: Tensor,
        z_ld: Tensor,
        od_live: Tensor,
        ld_live: Tensor,
        others_ctx: Tensor,
        nav_token: Tensor,
        signal_token: Tensor,
        *,
        geometry_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """**公开单步 plan head 入口**（rollout / Stage A/B 教师强制共用）。

        v8（B3）：融合输入 = ``[z_ego_k, mean(z_od_k), ld_summary_k, others, nav, signal]``
        （状态 latent 掩码均值池化 + t0 静态 others 上下文；nav 可为逐步重建 token）。
        返回 ``(latent, ego_next, moe_aux)``；``ego_next`` 监督"下一 ego 特征"，
        让 plan head/MoE 在教师强制路径中保持梯度（design-v1.2 §2.3 方案①）。

        R21-3A：``ld_summary_k`` 统一走 :meth:`build_geometry_context`——非 semantic 模式
        自行调用（baseline 逐位不变）；**semantic 模式必须由调用方传入 ``geometry_context``**
        （缺 → 报错，禁止 ``masked_mean`` 回退）。
        """
        nav_token = squeeze_batch_singletons(nav_token, 2, "nav_token")
        signal_token = squeeze_batch_singletons(signal_token, 2, "signal_token")
        od_pool = _masked_mean(z_od, od_live)
        if self._is_semantic():
            if geometry_context is None:
                raise RuntimeError(
                    "semantic 模式下 plan_step 缺 geometry_context——禁止静默回退"
                    "（round-25 §2：残差基底 = legacy aggregate_ld，但必须由 build_geometry_context "
                    "统一构造；调用方不得自行拼接）"
                )
            ld_pool = geometry_context
        else:
            # Round-22：LD pooled summary 统一读出（baseline = masked_mean，逐位不变）
            ld_pool = self.aggregate_ld(z_ld, ld_live, z_ego, nav_token, others_ctx)
        return self.plan_head(z_ego, od_pool, ld_pool, others_ctx, nav_token, signal_token)

    def _t0_object_features(self, encoded: EncodedMem) -> tuple[Tensor, Tensor]:
        """A3：t0 帧**单次 st_gnn 消息传递** → 对象级特征 ``(od (B,S,H), ld (B,L,H))``。

        经 :meth:`SpatioTemporalGNN.node_features` 公共委托执行（step_index=1），不跑解码器
        ——注意力头只要对象级节点特征；模型不再直连 ``st_gnn.spatial``/``st_gnn.step_embed``。
        在 ``no_grad`` 下执行：st_gnn 的训练信号保持 WM 损失口径（traj/policy 损失不得
        回传 st_gnn；tests/test_stage_v11.py 锁定），本 pass 只提供"当前权重下的特征"。

        v8（B3）：**不再接入令牌集合**（头令牌改为 latent 状态）；保留供诊断/外部调用。
        """
        with torch.no_grad():
            return self.st_gnn.node_features(
                ego_ctx=encoded.ego_ctx,
                od_ctx=encoded.od_ctx,
                ld_ctx=encoded.ld_ctx,
                node_mask=encoded.frame.node_mask,
                pose=encoded.frame.pose,
                step_index=1,
            )

    def _head_tokens(
        self,
        z_ego: Tensor,
        z_od: Tensor,
        z_ld: Tensor,
        od_live: Tensor,
        ld_live: Tensor,
        od_pool: Tensor,
        ld_pool: Tensor,
        others_ctx: Tensor,
        nav_token: Tensor,
        nav_mask: Tensor,
        signal_token: Tensor,
        signal_mask: Tensor,
        latent: Tensor,
        target_token: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """交叉注意力头的令牌集合 + key mask（v8 B3，T=39；semantic 模式仍 T=39）。

        顺序 = ``[z_od 16, z_ld 16, od_pool 1, ld_pool 1, others 1, z_ego 1, nav 1, signal 1,
        融合 latent 1]``。R21-25A：target token **不追加新 key**，而是经 ``target_proj``
        折叠进既有 ``ld_pool`` 槽（``ld_slot = ld_pool + target_proj(target_token)``）——
        等价消除 round-24 指出的"追加 token mask 恒 1"隐患（无新 key ⇒ 无效 token 不影响
        任何输出；bias 零初始化 + β=0 ⇒ 贡献严格 0，逐位 = baseline）。
        ``z_*`` = 当前步 latent 状态（rollout 逐步迭代；t0 = 当前帧编码）。
        """
        ones = torch.ones((int(z_ego.shape[0]), 1), dtype=z_ego.dtype, device=z_ego.device)
        ld_slot = ld_pool
        if target_token is not None:
            proj = self.target_proj
            ld_slot = ld_pool + (proj(target_token) if proj is not None else target_token)
        token_parts = [
            z_od,
            z_ld,
            od_pool.unsqueeze(1),
            ld_slot.unsqueeze(1),
        ]
        mask_parts = [od_live, ld_live, ones, ones]
        token_parts.extend(
            [
                others_ctx.unsqueeze(1),
                z_ego.unsqueeze(1),
                nav_token.unsqueeze(1),
                signal_token.unsqueeze(1),
                latent.unsqueeze(1),
            ]
        )
        mask_parts.extend([ones, ones, nav_mask, signal_mask, ones])
        tokens = torch.cat(token_parts, dim=1)
        key_mask = torch.cat(mask_parts, dim=1)
        return tokens, key_mask

    def _step_context(
        self,
        mem: MemBank,
        pose_world: Tensor,
        world: Mapping[str, object],
        nav_token: Tensor,
        nav_mask: Tensor,
        signal_token: Tensor,
        signal_mask: Tensor,
        step_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """A4：rollout/教师强制单步重建 nav/signal（others 上下文保持 t0 静态）。

        ``pose_world`` = 世界系 ego 位姿（每步由 :func:`advance_pose_world` 推进，与
        教师强制同一步进语义）。无 ``route_world``/``ego_world`` 时原样返回 t0 上下文
        （旧输入逐位兼容）。v8（B3）：不再回写 others mem 的 nav 维（others 上下文是 t0
        静态 token，nav 由独立 token 承载；且 rollout 不再重编码 raw mem）。
        """
        route_world = world.get("route_world")
        route_world_mask = world.get("route_world_mask")
        if route_world is None or not torch.is_tensor(pose_world) or not torch.is_tensor(route_world):
            return nav_token, nav_mask, signal_token, signal_mask
        nav_feat, nav_mask = rebuild_nav_from_world(
            mem, pose_world, route_world, route_world_mask
        )
        nav_token = self.encoders.embed_nav(nav_feat.unsqueeze(1), nav_mask.reshape(-1, 1))[:, 0]
        signal_token, signal_mask = advance_signal(signal_token, signal_mask, int(step_index))
        return nav_token, nav_mask, signal_token, signal_mask

    def encode(self, obs: Mapping[str, Tensor], *, semantic_off: bool = False) -> dict[str, object]:
        """观测 → mem/编码 mem/**latent 状态**/plan-head latent/头令牌集合（不跑 rollout）。

        v8（B3）：状态 latent = **当前帧编码**（``embed_ego(ego_now)`` / ``embed_od(od_now,
        od_live)`` / ``embed_ld(ld_now, ld_live)``）；上下文 token = t0 静态
        （``od_pool/ld_pool = masked_mean(od_ctx/ld_ctx)``、``others_ctx``、nav/signal）。
        另携带 A4 的世界系输入（``route_world``/``ego_world``，缺失时 rollout 回退 t0 冻结 nav）
        与 rollout 图位姿模板（``frame``）。
        """
        mem = self._mem_from_obs(obs)
        nav_token, nav_mask, signal_token, signal_mask = self._context_tokens(obs, mem.batch)
        # R38-D1：④ Memory hook（默认 None/关 ⇒ 零开销；只读记录 mask/身份/注意力/历史顺序）
        probes = self.probes
        probes_on = bool(probes is not None and probes.enabled)
        encoded = self.mem_encoder.encode(
            self.encoders, mem, hooks=probes.memory if probes_on else None
        )
        # v8（B3）：状态 latent = 当前帧编码（单帧，非时序聚合）
        ones = torch.ones((mem.batch, ), dtype=encoded.ego_now.dtype, device=encoded.ego_now.device)
        z_ego = self.encoders.embed_ego(encoded.ego_now, ones)
        z_od = self.encoders.embed_od(encoded.od_now, encoded.od_live, ids=encoded.od_id_now)
        z_ld = self.encoders.embed_ld(encoded.ld_now, encoded.ld_live)
        if probes_on:
            # R38-D1 ①：t0 状态来源（real）+ ego/OD/LD/nav/Memory latent（只读 detach-clone）
            if probes.state.enabled:
                probes.state.capture(
                    step=0,
                    source="real",
                    z_ego=z_ego,
                    z_od=z_od,
                    z_ld=z_ld,
                    nav=nav_token,
                    memory={
                        "ego_ctx": encoded.ego_ctx,
                        "od_ctx": encoded.od_ctx,
                        "ld_ctx": encoded.ld_ctx,
                        "others_ctx": encoded.others_ctx,
                    },
                )
            # R38-D1 ⑤：linkage 锚点 z_t0（供 check() 断言 a0 读的是 t0 状态）
            if probes.chain.enabled:
                probes.chain.note_z_t0({"z_ego": z_ego, "z_od": z_od, "z_ld": z_ld})
        # 上下文 token（t0 静态）：池化来自时序聚合 ctx
        od_pool = _masked_mean(encoded.od_ctx, encoded.od_live)
        # R21-3A/25A：semantic 输入（current frame；缺键报错 = 旧数据 fail-fast）
        semantic = self._semantic_inputs_from_obs(obs) if self._is_semantic() else None
        # R21-25A 残差：head 槽基底 = legacy(ld_ctx)（与 baseline 同源）；plan 基底 = legacy(z_ld)；
        # 语义 delta 由 SemanticWiring 在**当前帧 z_ld** 上计算（t0 delta 供 rollout t0_only 复用）
        delta_t0 = None
        target_token = None
        if semantic is not None and not semantic_off:
            delta_t0, raw_target = self.semantic_wiring(z_ld, encoded.ld_live, **semantic)
            target_token = self.beta_token * raw_target
            head_context = (
                self.aggregate_ld(encoded.ld_ctx, encoded.ld_live, z_ego, nav_token, encoded.others_ctx)
                + self.beta_context * delta_t0
            )
            plan_context = (
                self.aggregate_ld(z_ld, encoded.ld_live, z_ego, nav_token, encoded.others_ctx)
                + self.beta_context * delta_t0
            )
        else:
            # 非 semantic 或 semantic_off（R21-26B：delta 监督的"基底"前向 = legacy 路径，
            # 语义贡献严格为 0；用于 delta_model = mu_on − mu_off.detach()）
            head_context = self.aggregate_ld(encoded.ld_ctx, encoded.ld_live, z_ego, nav_token, encoded.others_ctx)
            # plan 基底必须与 semantic 分支 β=0 逐位一致（z_ld 基底，而非 head 的 ld_ctx 基底）
            plan_context = (
                self.aggregate_ld(z_ld, encoded.ld_live, z_ego, nav_token, encoded.others_ctx)
                if self._is_semantic() else None
            )
        ld_context, target_token = head_context, target_token
        # R36 H1：训练期 target-lane token dropout（默认关；p=0 不消耗 RNG）
        target_lane_keep: Tensor | None = None
        if self.target_lane_dropout_p > 0.0 and self.training and target_token is not None:
            target_lane_keep = (
                torch.rand((mem.batch,), dtype=target_token.dtype, device=target_token.device)
                >= self.target_lane_dropout_p
            )
            target_token = target_token * target_lane_keep.unsqueeze(-1)
        ld_pool = ld_context
        if self.ld_mode == "set_attn" and self.ld_set_attn is not None:
            # Round-20 M2：LD set-attention 上下文（ego + nav + others 为 query）
            import torch as _torch

            batch = int(z_ld.shape[0])
            context = _torch.cat(
                [
                    z_ego.reshape(batch, -1),
                    nav_token.reshape(batch, -1),
                    encoded.others_ctx.reshape(batch, -1),
                ],
                dim=-1,
            )
            ld_pool = self.ld_set_attn(z_ld, encoded.ld_live, context)
        latent, ego_next, moe_aux = self.plan_step(
            z_ego, z_od, z_ld, encoded.od_live, encoded.ld_live, encoded.others_ctx,
            nav_token, signal_token,
            geometry_context=plan_context,
        )
        # R36 K1：结构化头（默认关；启用时要求 lane_anchor_xy）——这里只算 **t0 步**动作
        # （= a0 = action_mu），并把 t0 静态几何存进 encoded 供 rollout 逐步重算（step_geometry）。
        # 消融①（``policy``）不调用头（= 默认关路径逐位）；每次调用只产**当前步**动作（R38-P1
        # 后不存在一次性六步口径）。
        path_out: dict[str, Tensor] | None = None
        path_ctx: dict[str, Tensor] | None = None
        path_head = self.path_head
        if path_head is not None and self._path_head_enabled:
            anchor_xy, anchor_mask = anchor_from_obs(obs)
            long_exog, long_present = long_exog_from_obs(
                obs, batch=mem.batch, device=latent.device
            )
            target_lane_valid = target_lane_valid_from_obs(
                obs, batch=mem.batch, device=latent.device
            )
            path_out = path_head(
                latent,
                anchor_xy=anchor_xy,
                anchor_mask=anchor_mask,
                long_exog=long_exog,
                target_lane_valid=target_lane_valid,
            )
            if self.contract_asserts_enabled:
                # R36 L-A 契约①：t0 期头调用实例 id（a0 的来源；rollout 逐步调用同源断言用）
                self._path_head_call_ids.append(id(path_head))
            if probes_on and probes.policy.enabled:
                # R38-D1 ②：t0 步头调用（backbone latent / κ / ds,dθ / 辅助 / 输入 valid mask）
                probes.policy.capture(
                    step=0,
                    latent=latent,
                    outputs=path_out,
                    masks={
                        "anchor_mask": anchor_mask,
                        "target_lane_valid": path_out["target_lane_valid"],
                        "long_present": long_present,
                    },
                )
            if probes_on and probes.chain.enabled:
                # R38-D1 ⑤：调用链（a0 的来源；rollout 逐步调用同源/linkage 断言）
                probes.chain.on_policy_call(
                    step=0, instance=id(self.path_head), z={"z_ego": z_ego, "z_od": z_od, "z_ld": z_ld}
                )
            path_out["long_exog"] = long_exog
            path_out["long_present"] = long_present
            path_ctx = {
                "anchor_xy": anchor_xy,
                "anchor_mask": anchor_mask,
                "long_exog": long_exog,
                "long_present": long_present,
                "target_lane_valid": path_out["target_lane_valid"],
            }
            if self.path_head_trace is not None:
                self.path_head_trace.append(
                    {
                        "step": 0,
                        "latent": latent,
                        "action": path_out["action"],
                        "ds": path_out["ds_exec"],
                        "kappa_near": path_out["kappa_near"],
                        "wm_calls_before": 0,
                        "wm_advance_calls": 0,
                    }
                )
        tokens, key_mask = self._head_tokens(
            z_ego,
            z_od,
            z_ld,
            encoded.od_live,
            encoded.ld_live,
            od_pool,
            ld_pool,
            encoded.others_ctx,
            nav_token,
            nav_mask,
            signal_token,
            signal_mask,
            latent,
            target_token,
        )
        # A4 世界系输入（route_world/ego_world/route_world_mask；缺省 None ⇒ rollout 回退 t0 冻结）
        ego_world, route_world, route_world_mask = world_state_from_obs(obs)
        if torch.is_tensor(ego_world):
            ego_world = squeeze_batch_singletons(ego_world.float(), 2, "ego_world")
            if tuple(ego_world.shape) != (mem.batch, 3):
                raise ValueError(
                    f"obs['ego_world'] 形状应为 (B,3)（或 (B,1,3)），收到 {tuple(ego_world.shape)}"
                )
        if torch.is_tensor(route_world_mask):
            # 透传契约：mask 原样进 rollout/教师强制的 nav 重建（缺省 None ⇒ 全部顶点视为有效）。
            # 规范形状 (B,M)（容忍 (B,1,M) 单例槽位维，与 ego_world 同口径）。
            route_world_mask = route_world_mask.float()
            if route_world_mask.ndim == 3 and int(route_world_mask.shape[1]) == 1:
                route_world_mask = route_world_mask[:, 0]
            if route_world_mask.ndim != 2 or int(route_world_mask.shape[0]) != mem.batch:
                raise ValueError(
                    f"obs['route_world_mask'] 形状应为 (B,M)（或 (B,1,M)），收到 {tuple(route_world_mask.shape)}"
                )
        # R37 M1：warp 诊断可选数据（形状 (B,T,N,3)；缺省 None ⇒ hooks 标 n/a、读数 valid=0）
        warp_nodes = int(encoded.frame.pose.shape[1])
        warp_actual_next = _warp_diag_steps(
            obs, "warp_anchor_actual_next",
            batch=mem.batch, steps=self.rollout_steps, nodes=warp_nodes, ref=z_ego,
        )
        warp_residual_learned = _warp_diag_steps(
            obs, "warp_anchor_residual_learned",
            batch=mem.batch, steps=self.rollout_steps, nodes=warp_nodes, ref=z_ego,
        )
        return {
            "mem": mem,
            "encoded": encoded,
            "frame": encoded.frame,
            # v8（B3）：latent 状态 + t0 静态上下文
            "z_ego": z_ego,
            "z_od": z_od,
            "z_ld": z_ld,
            "od_pool": od_pool,
            "ld_pool": ld_pool,
            "others_ctx": encoded.others_ctx,
            "latent": latent,
            "ego_next": ego_next,
            "moe_aux": moe_aux,
            "nav_token": nav_token,
            "nav_mask": nav_mask,
            "signal_token": signal_token,
            "signal_mask": signal_mask,
            "tokens": tokens,
            "key_mask": key_mask,
            "route_world": route_world,
            "route_world_mask": route_world_mask,
            "ego_world": ego_world,
            # R36 K1：结构化头（默认关 ⇒ None）+ t0 静态几何上下文（rollout 逐步重算用）
            # 与 target-lane dropout 掩码
            "path": path_out,
            "path_ctx": path_ctx,
            "target_lane_dropout_keep": target_lane_keep,
            # R21-3A/25A：semantic 模式诊断/下游（baseline = None / 与 ld_pool 相同）
            "target_lane_token": target_token,
            "geometry_context": ld_context,          # head 槽上下文（ld_ctx 基底）
            "geometry_context_plan": plan_context,   # plan 融合上下文（z_ld 基底）
            "semantic_delta_t0": delta_t0,
            "semantic_inputs": semantic,
            "semantic_off": bool(semantic_off),
            # R37 M1：warp 诊断（5 项 hook 的 ③/④ 数据源；默认 None = n/a / 0）
            "warp_anchor_actual_next": warp_actual_next,
            "warp_anchor_residual_learned": warp_residual_learned,
        }

    # ---------------------------------------------------------------- rollout
    def _rollout(self, encoded: dict[str, object], action0: Tensor) -> dict[str, Tensor]:
        """6 步 **latent 自回归** rollout（状态链 detach + action/pose 链可微）。

        ``action0`` = 传入的 ``action_mu``（第 1 个计划动作；与 cheap path 同一计算）。
        返回：``traj_xy/traj_theta/plan``（位姿链）、``od_pred/ld_pred/
        od_presence_pred/od_entry_pred``（t0 帧物理解码，诊断/旧消费者）与
        ``z_ego_pred/z_od_pred/z_ld_pred``（latent 状态诊断）。

        v8（B3）：**不再拷贝/滑动 raw mem** —— 直接迭代 latent 状态；每步 ``st_gnn`` 得
        next latents，**作为下一步输入前 detach**（状态链切断；动作/位姿链保持可微，与旧
        语义一致）；每步头令牌用当前步 latents，plan head 融合输入 =
        ``[z_ego_k, mean(z_od_k), mean(z_ld_k), others, nav, signal]``，policy 自回归出后续
        动作（step0 = action_mu 契约不变）。
        A4：每步按世界系 ego 位姿重建 nav（``_step_context``；others 上下文保持 t0 静态）；
        无 ``route_world``/``ego_world`` 输入时回退 t0 冻结上下文（旧行为逐位兼容）。

        R36 K1：``path_head`` 启用且模式 = ``structured`` 时，每步在 latent 更新后用**同一**
        :class:`~net.path_head.StructuredPolicyHead` 实例、**当前** ``z_k`` 产出
        ``a_k=(ds_k, κ_near_k·ds_k)``（imagined 步静态几何 = t0 静态地图 + 想象位姿确定性重算，
        :func:`~net.path_head.step_geometry`）；``plan`` 由逐步动作组装（唯一动作生成器）。
        R38-P1：H1 一次性六步钉住（第二动作源）已删除；R38-P2a：K-anchor 锚计划出口已删除
        ⇒ 本函数的动作出口只有两个：structured 头 / 旧 policy（后者 = 消融口径）。

        R37 M1：``ego_warp_mode`` 决定 ``pose_k`` 口径（``off`` = 逐位不变 / ``analytic`` =
        解析 warp + OD t0 场景系速度）；``warp_hooks`` / ``warp_readout``（默认全关，零开销）
        逐步产出 5 项 hook 与分项读数（见 :meth:`set_warp_hooks` / :meth:`set_warp_readout`）。

        R38-D1：``probes``（默认 None/关，零开销）逐步只读采集 ①状态来源 / ②Policy /
        ③World Model / ⑤调用链（**ST-GNN 总调用单列**）；输出契约不变，探针经统一出口落盘。
        """
        mem: MemBank = encoded["mem"]  # 只读（nav 回退锚点）；不再拷贝/滑动
        enc: EncodedMem = encoded["encoded"]
        probes = self.probes
        probes_on = bool(probes is not None and probes.enabled)
        frame = encoded["frame"]
        z_ego: Tensor = encoded["z_ego"]
        z_od: Tensor = encoded["z_od"]
        z_ld: Tensor = encoded["z_ld"]
        od_live: Tensor = enc.od_live
        ld_live: Tensor = enc.ld_live
        od_pool: Tensor = encoded["od_pool"]
        ld_pool: Tensor = encoded["ld_pool"]
        others_ctx: Tensor = encoded["others_ctx"]
        latent: Tensor = encoded["latent"]
        nav_token: Tensor = encoded["nav_token"]
        nav_mask: Tensor = encoded["nav_mask"]
        signal_token: Tensor = encoded["signal_token"]
        signal_mask: Tensor = encoded["signal_mask"]
        world = {
            "route_world": encoded.get("route_world"),
            "route_world_mask": encoded.get("route_world_mask"),
            "ego_world": encoded.get("ego_world"),
        }
        pose_world = encoded.get("ego_world")  # A4 世界系位姿锚点（t0）；每步 advance_pose_world
        action = action0  # 第 1 个计划动作 = action_mu（与 cheap path 逐位一致）
        # R36 K1：结构化头（默认关 ⇒ None）。生产模式 ``structured`` = 每步调用同一头。
        # R38-P1：H1 一次性六步的尾段钉住分支已删除 ⇒ 不再有第二个动作源。
        # R38-P2b：结构化头 = ``self.policy.structured_action``（收编后单实体；兼容属性 ``path_head``）。
        path_mode = self.path_head_mode()
        path_head = self.path_head
        structured = path_mode == "structured" and path_head is not None
        path_ctx = encoded.get("path_ctx") if isinstance(encoded.get("path_ctx"), dict) else None
        path_out = encoded.get("path")
        if structured and path_ctx is None:
            raise ValueError(
                "structured 模式需要 encode() 产出的 path_ctx（t0 静态几何）；"
                "_rollout 不应脱离 forward/encode 直接调用"
            )
        progress_s = torch.zeros((mem.batch,), dtype=latent.dtype, device=latent.device)
        kappa_near_steps: list[Tensor] = []
        ds_steps: list[Tensor] = []
        kappa_mid_steps: list[Tensor] = []
        kappa_far_steps: list[Tensor] = []
        aux_progress_steps: list[Tensor] = []
        if structured and path_out is not None and isinstance(path_out, dict):
            kappa_near_steps.append(path_out["kappa_near"])
            ds_steps.append(path_out["ds_exec"])
            kappa_mid_steps.append(path_out["kappa_mid"])
            kappa_far_steps.append(path_out["kappa_far"])
            aux_progress_steps.append(path_out["aux_progress"])
        wm_advance_calls = 0

        # t0 锚定状态与图位姿模板（t0 帧；来自真实 obs，不 detach —— 检测任务）
        od_anchor = self.st_gnn.od_state_from_features(enc.od_now, od_live)
        ld_anchor = self.st_gnn.ld_state_from_features(enc.ld_now, ld_live)
        node_mask: Tensor = frame.node_mask
        od_pose_t0 = frame.pose[:, 1 : 1 + self.od_slots]
        ld_pose_t0 = frame.pose[:, 1 + self.od_slots :]
        od_velocity = enc.od_now[..., 2:4] * od_live.unsqueeze(-1)  # (B,S,2) t0 相对速度

        # R37 M1：warp 诊断（5 项 hook + 分项读数；默认全关 ⇒ 零张量操作、不消耗 RNG）
        warp_hooks = self.warp_hooks
        warp_hooks_on = bool(warp_hooks is not None and warp_hooks.enabled)
        warp_readout_on = bool(self.warp_readout)
        warp_diag = bool(warp_hooks_on or warp_readout_on)
        warp_actual_steps: Tensor | None = None
        warp_residual_steps: Tensor | None = None
        prior_err_steps: list[Tensor] = []
        residual_contrib_steps: list[Tensor] = []
        final_err_steps: list[Tensor] = []
        if warp_diag:
            actual = encoded.get("warp_anchor_actual_next")
            residual = encoded.get("warp_anchor_residual_learned")
            warp_actual_steps = (
                actual.to(dtype=latent.dtype, device=latent.device) if torch.is_tensor(actual) else None
            )
            warp_residual_steps = (
                residual.to(dtype=latent.dtype, device=latent.device)
                if torch.is_tensor(residual)
                else None
            )

        pose = torch.zeros((mem.batch, 3), dtype=latent.dtype, device=latent.device)
        xy: list[Tensor] = []
        theta: list[Tensor] = []
        actions: list[Tensor] = []
        od_steps: list[Tensor] = []
        ld_steps: list[Tensor] = []
        presence_steps: list[Tensor] = []
        entry_steps: list[Tensor] = []
        z_ego_steps: list[Tensor] = []
        z_od_steps: list[Tensor] = []
        z_ld_steps: list[Tensor] = []
        for k in range(1, self.rollout_steps + 1):
            # (a) 位姿链：action/pose 不 detach（6 点轨迹目标训练 plan head 的动作链）
            ds, dtheta = action[:, 0], action[:, 1]
            dx, dy = arc_step(ds, dtheta)
            pose = compose_pose(pose, dx, dy, dtheta)
            xy.append(pose[:, :2])
            theta.append(pose[:, 2])
            actions.append(action)
            if structured:
                # imagined 步几何进度 = 已执行 ds 之和（几何重算的确定性输入）
                progress_s = progress_s + ds.detach()

            # (a2) A4：推进世界系位姿（与教师强制同一步进语义）并逐步重建 nav/signal
            if torch.is_tensor(pose_world):
                pose_world = advance_pose_world(pose_world, ds, dtheta)
            nav_token, nav_mask, signal_token, signal_mask = self._step_context(
                mem, pose_world, world, nav_token, nav_mask, signal_token, signal_mask, k
            )

            # (b) 图位姿：ego = 累积位姿（detach：WM 输出不回传动作/策略链）；LD 静止。
            #     ego_warp_mode="off"（默认，逐位不变）= assemble_pose_k(off)：现行 t0 帧口径
            #     （OD 先验 = 通道相对速度，随体语义）；"analytic"（R37 M1）= assemble_pose_k(analytic)：
            #     t0 系当静态场景，OD 先验用 **t0 场景系速度** od_velocity_t0（v_rel+v_ego），
            #     整图再重表达进 ego_k 帧（同一刚体变换施加于 ego/OD/LD；ego 恒 (0,0,0)）。
            #     R38-P2b：legacy w1/w2 第二状态机已删（w1 只存于 fixture/legacy 快照）。
            ego_pose_det = pose.detach()
            if self.ego_warp_mode == "analytic":
                ego_speed = enc.ego_now[..., 0:1]  # t0 自车速度（沿自身 x 轴）
                ego_vel_t0 = torch.cat([ego_speed, torch.zeros_like(ego_speed)], dim=-1)
                # v_OD = v_rel + v_ego（掩码外槽位清零；禁止用相对速度推进后再扣自车运动）
                od_prior_vel = od_velocity_t0(od_velocity, ego_vel_t0) * od_live.unsqueeze(-1)
                warp_mode = "analytic"
            else:
                od_prior_vel = od_velocity
                warp_mode = "off"
            pose_k = assemble_pose_k(
                ego_pose_det, od_pose_t0, od_prior_vel, ld_pose_t0,
                mode=warp_mode, dt=self.dt, steps=k,
            )

            if warp_diag:
                # R37 M1 五项 hook / 分项读数（默认关 ⇒ 零张量操作）：
                # ① warp 前 anchor（t0 帧；与 pose_k 同一 OD 先验与 ego 累积位姿，仅缺 warp）。
                anchor_pre = assemble_pose_k(
                    ego_pose_det, od_pose_t0, od_prior_vel, ld_pose_t0,
                    mode="off", dt=self.dt, steps=k,
                )
                actual_k = None if warp_actual_steps is None else warp_actual_steps[:, k - 1]
                residual_k = None if warp_residual_steps is None else warp_residual_steps[:, k - 1]
                if warp_hooks_on:
                    warp_hooks.capture(
                        step=k,
                        anchor_pre=anchor_pre,
                        anchor_analytic=pose_k,
                        anchor_actual_next=actual_k,
                        residual_learned=residual_k,
                    )
                if warp_readout_on and actual_k is not None:
                    anchor_final = pose_k if residual_k is None else pose_k + residual_k
                    prior_err_steps.append(masked_pose_error(pose_k, actual_k, node_mask))
                    residual_contrib_steps.append(masked_pose_error(anchor_final, pose_k, node_mask))
                    final_err_steps.append(masked_pose_error(anchor_final, actual_k, node_mask))

            # (c) latent 自回归转移 + 物理解码（t0 帧预测，逐步堆叠）
            # R32 canary：action = 本步执行动作 a_{k-1}（与 pose 链同一动作；开关默认关 ⇒ 无变化）
            # R36 K1 梯度门：结构化模式喂给 WM 的动作 stop-gradient（前向逐位不变）⇒
            # WM 预测 loss 只回传到 WM，不经动作输入旁路到结构化头（WM 冻结时该旁路本无梯度）。
            wm_action = action
            if structured and self.path_head_wm_action_detach:
                wm_action = action.detach()
            if probes_on and probes.chain.enabled:
                # R38-D1 ⑤：ST-GNN **实际总调用**（单列；含最后一步不推进决策的调用）
                probes.chain.on_st_gnn_call(k)
            z_ego_next, z_od_next, z_ld_next, od_pred, ld_pred, presence_logit, entry_logit = (
                self.st_gnn(
                    z_ego=z_ego,
                    z_od=z_od,
                    z_ld=z_ld,
                    node_mask=node_mask,
                    pose=pose_k,
                    step_index=k,
                    od_anchor=od_anchor,
                    ld_anchor=ld_anchor,
                    action=wm_action,
                )
            )
            if probes_on and probes.wm.enabled:
                # R38-D1 ③：输入 action / z in（detach 前）/ z out / 物理解码 / presence/entry
                probes.wm.capture(
                    step=k,
                    action=wm_action,
                    z_in={"z_ego": z_ego, "z_od": z_od, "z_ld": z_ld},
                    z_out={"z_ego": z_ego_next, "z_od": z_od_next, "z_ld": z_ld_next},
                    od_pred=od_pred,
                    ld_pred=ld_pred,
                    presence_logit=presence_logit,
                    entry_logit=entry_logit,
                )
            if probes_on and probes.state.enabled:
                # R38-D1 ①：状态 z_k（= WM 第 k 步输出；决策 a_k 在循环末读取它）+ 本步 nav_k。
                # z_0 = t0 真实编码（encode 期已记，source=real）；k≥1 = state_source_label
                # （predicted；E1..E6 真值替换实验由脚本置 substituted）。Memory latent 为
                # t0 上下文（逐帧 mask/注意力在 ④ Memory hook）。
                probes.state.capture(
                    step=k,
                    source=probes.state_source_label,
                    z_ego=z_ego_next,
                    z_od=z_od_next,
                    z_ld=z_ld_next,
                    nav=nav_token,
                )
            od_steps.append(od_pred)
            ld_steps.append(ld_pred)
            presence_steps.append(presence_logit)
            entry_steps.append(entry_logit)
            z_ego_steps.append(z_ego_next)
            z_od_steps.append(z_od_next)
            z_ld_steps.append(z_ld_next)

            # (d) 状态链切断：预测 latent detach 后作为下一步输入（动作/位姿链不受影响）
            z_ego, z_od, z_ld = z_ego_next.detach(), z_od_next.detach(), z_ld_next.detach()

            # (e) 下一轮 plan head + 注意力头（最后一步之后不需要）
            if k < self.rollout_steps:
                semantic = encoded.get("semantic_inputs")
                step_target = encoded.get("target_lane_token")
                if self._is_semantic() and not encoded.get("semantic_off"):
                    # R21-25A：plan 基底逐 step legacy + 语义 delta（口径由 rollout_semantic_mode 门控）；
                    # head 槽保持 t0 上下文（与 baseline 同结构 ⇒ β=0 逐位）
                    plan_context_k, _ = self._rollout_step_geometry(
                        z_ld,
                        ld_live,
                        z_ego,
                        nav_token,
                        others_ctx,
                        semantic=semantic,
                        t0_delta=encoded.get("semantic_delta_t0"),
                        t0_target=encoded.get("target_lane_token"),
                    )
                    latent, _, _ = self.plan_step(
                        z_ego, z_od, z_ld, od_live, ld_live, others_ctx, nav_token, signal_token,
                        geometry_context=plan_context_k,
                    )
                elif self._is_semantic():
                    # R21-26B semantic_off：基底前向（legacy 路径，语义贡献严格 0）
                    plan_context_k = self.aggregate_ld(z_ld, ld_live, z_ego, nav_token, others_ctx)
                    latent, _, _ = self.plan_step(
                        z_ego, z_od, z_ld, od_live, ld_live, others_ctx, nav_token, signal_token,
                        geometry_context=plan_context_k,
                    )
                else:
                    latent, _, _ = self.plan_step(
                        z_ego, z_od, z_ld, od_live, ld_live, others_ctx, nav_token, signal_token
                    )
                if structured:
                    # R36 K1（架构契约 §六步调用图）：同一头 + **当前** z_k → 当前步动作
                    #   z_k 来自上一步 WM 推进（上面 (c)）；imagined 步静态几何由 t0 静态地图
                    #   + 想象位姿确定性重算（geometry detach：外源确定性输入）。
                    geometry = step_geometry(
                        path_ctx["anchor_xy"],
                        path_ctx["anchor_mask"],
                        path_ctx["long_exog"],
                        path_ctx["long_present"],
                        pose.detach(),
                        progress_s,
                    )
                    step_out = path_head(
                        latent,
                        anchor_xy=geometry["anchor_xy"],
                        anchor_mask=geometry["anchor_mask"],
                        long_exog=geometry["long_exog"],
                        target_lane_valid=path_ctx["target_lane_valid"],
                    )
                    if self.contract_asserts_enabled:
                        self._path_head_call_ids.append(id(path_head))
                    wm_advance_calls += 1
                    if probes_on and probes.policy.enabled:
                        # R38-D1 ②：逐步头调用（latent = plan 融合 backbone；κ/纵向/辅助/mask）
                        probes.policy.capture(
                            step=k,
                            latent=latent,
                            outputs=step_out,
                            masks={
                                "anchor_mask": geometry["anchor_mask"],
                                "target_lane_valid": path_ctx["target_lane_valid"],
                                "long_present": geometry["long_present"],
                            },
                        )
                    if probes_on and probes.chain.enabled:
                        # R38-D1 ⑤：a_k 读 z_k（linkage 断言）+ WM 决策推进（应为 5）
                        probes.chain.on_policy_call(
                            step=k,
                            instance=id(self.path_head),
                            z={"z_ego": z_ego, "z_od": z_od, "z_ld": z_ld},
                        )
                        probes.chain.on_wm_advance()
                    action = step_out["action"]
                    kappa_near_steps.append(step_out["kappa_near"])
                    ds_steps.append(step_out["ds_exec"])
                    kappa_mid_steps.append(step_out["kappa_mid"])
                    kappa_far_steps.append(step_out["kappa_far"])
                    aux_progress_steps.append(step_out["aux_progress"])
                    if self.path_head_trace is not None:
                        self.path_head_trace.append(
                            {
                                "step": k,
                                "latent": latent,
                                "action": action,
                                "ds": step_out["ds_exec"],
                                "kappa_near": step_out["kappa_near"],
                                "wm_calls_before": k,
                                "wm_advance_calls": wm_advance_calls,
                            }
                        )
                else:
                    # 默认关 / 消融①（``policy``）路径：旧 PolicyHead（唯一剩余非结构化出口）
                    tokens, key_mask = self._head_tokens(
                        z_ego,
                        z_od,
                        z_ld,
                        od_live,
                        ld_live,
                        od_pool,
                        ld_pool,
                        others_ctx,
                        nav_token,
                        nav_mask,
                        signal_token,
                        signal_mask,
                        latent,
                        encoded.get("target_lane_token"),
                    )
                    action, _ = self.policy(tokens, key_mask)

        result = {
            "traj_xy": torch.stack(xy, dim=1),
            "traj_theta": torch.stack(theta, dim=1),
            "plan": torch.stack(actions, dim=1),
            "od_pred": torch.stack(od_steps, dim=1),
            "ld_pred": torch.stack(ld_steps, dim=1),
            "od_presence_pred": torch.stack(presence_steps, dim=1),
            "od_entry_pred": torch.stack(entry_steps, dim=1),
            "z_ego_pred": torch.stack(z_ego_steps, dim=1),
            "z_od_pred": torch.stack(z_od_steps, dim=1),
            "z_ld_pred": torch.stack(z_ld_steps, dim=1),
        }
        if structured:
            # K1 诊断/门证据：逐步执行的 κ_near/ds + 辅助 κ_mid/κ_far/progress + 调用计数。
            # 口径：a0 来自 t0 编码期调用；a1..a5 = 5 次「WM 推进一步 → 同一头重算」。
            result["structured_kappa_near"] = torch.stack(kappa_near_steps, dim=1)
            result["structured_ds"] = torch.stack(ds_steps, dim=1)
            result["structured_aux_kappa_mid"] = torch.stack(kappa_mid_steps, dim=1)
            result["structured_aux_kappa_far"] = torch.stack(kappa_far_steps, dim=1)
            result["structured_aux_progress"] = torch.stack(aux_progress_steps, dim=1)
            result["structured_head_calls"] = torch.tensor(
                float(len(ds_steps)), dtype=latent.dtype, device=latent.device
            )
            result["structured_wm_advances"] = torch.tensor(
                float(wm_advance_calls), dtype=latent.dtype, device=latent.device
            )
            # R36 L-A 契约⑤（R38-P1 收紧为"路径不存在"）：一次性六步钉住路径已从 _rollout
            # 删除 ⇒ 结构上恒不可用（键保留 + **结构派生**：历史模式名回归或 ``one_shot_plan``
            # 复活 ⇒ 置 1 ⇒ 契约门失败；不是硬编码常量）。
            pinned_path_absent = (
                "oneshot" not in PATH_HEAD_MODES
                and not hasattr(self.path_head, "one_shot_plan")
            )
            result["structured_plan_override_used"] = torch.tensor(
                float(not pinned_path_absent), dtype=latent.dtype, device=latent.device
            )
        if warp_readout_on:
            # R37 M1 分项读数（诊断输出）：部署侧无真实下一帧 ⇒ ``warp_readout_valid=0``
            # （n/a；不产出误差键）。误差 = 逐 step 位置 L2 的 node_mask 掩码均值 [m]。
            result["warp_readout_valid"] = torch.tensor(
                float(warp_actual_steps is not None), dtype=latent.dtype, device=latent.device
            )
            if warp_actual_steps is not None:
                result["warp_prior_err"] = torch.stack(prior_err_steps, dim=1)
                result["warp_residual_contrib"] = torch.stack(residual_contrib_steps, dim=1)
                result["warp_final_err"] = torch.stack(final_err_steps, dim=1)
        return result

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        obs: Mapping[str, Tensor],
        *,
        forward_mode: str | None = None,
        semantic_off: bool = False,
        **legacy_signature: Any,
    ) -> dict[str, Tensor]:
        """完整前向：策略/价值/路由 + （可选）latent 自回归 rollout 与 t0 帧物理解码。

        **显式模式枚举**（GPT §七-H；映射表 = :data:`FORWARD_MODE_FLAGS`；R39-A1 全仓唯一开关）：
        ``forward_mode ∈ {policy_single_step, policy_wm_rollout, world_model_teacher_forcing,
        diagnostics}``；缺省 ``None`` ≡ ``world_model_teacher_forcing``（旧签名默认 (True, True)
        逐位保留）。旧 ``rollout=/world_model=`` 布尔兼容口**已删除**（R39-A1）：残留调用 ⇒
        ``ValueError`` fail-fast（文案指路 forward_mode；回归 = ``tests/test_r38_p3_forward_mode.py``）。

        - ``policy_single_step`` = 旧 ``(rollout=False, world_model=False)`` cheap path：只做
          编码（**不再跑任何 st_gnn 消息传递**）+ plan head + 策略/价值头，不跑 rollout，
          返回的 ``action_mu/action_logstd/value/router_logits/expert_weights/latent`` 与完整
          前向逐位一致（PPO 收集/BC update 用）；
        - ``policy_wm_rollout`` = 旧 ``(rollout=True, world_model=False)``：WM latent rollout
          明确在跑（轨迹/动作需要），但**不返回** ``od_pred/ld_pred/od_presence_pred/
          od_entry_pred`` 与 ``z_*_pred``；
        - ``world_model_teacher_forcing`` = 旧 ``(rollout=True, world_model=True)``：全量前向
          （WM 预测键返回；WM 训练/评估口径）；
        - ``diagnostics``：当前 = 全量前向旗标（与 WMTF 相同），语义保留给诊断/可视化调用方。

        ``wm_detach``：**形参已撤**（R38-P3-H；``**legacy_signature`` 兼容吞掉，恒 no-op ——
        v8 的 latent 状态链 detach 固定生效，从未有消融开关）；未知关键字 ⇒ ``TypeError``。

        MoE 输出口径（lane U1）：``primary + Σ_{i∈top2} g_i·expert_i``，推理与训练一致、
        全场景生效（无硬切/二值门/硬掩码）；phase 1 训练由 ``set_moe(enabled=False)`` 关闭。

        R36 L-A 旧头消歧（键契约）＋ R38-P2b 收编
        ----------------------------------------
        - ``path_head_mode() == "policy"``（默认关）：旧 ``PolicyHead`` 前向，
          输出 ``action_mu/action_logstd``（默认口径逐位不变）。
        - ``structured``：旧头**不前向**，输出 dict **无** ``policy_action_mu``
          与 ``action_logstd``（唯一动作源 = 结构化头；旧头诊断已离线化：
          ``set_policy_diag`` fail-fast，离线工具 ``tools/diagnostics/legacy_policy_diag.py``）。
          ``path_*`` 辅助键不受影响照常输出。正式 forward 只保留这一个动作出口切换。
        - ``contract_*``（``set_contract_asserts``，默认开；仅 structured + wm rollout 模式）：
          ``contract_policy_instance_ids_are_identical``/``contract_policy_call_count_is_six``/
          ``contract_world_model_call_count_is_five``/
          ``contract_auxiliary_future_outputs_not_consumed_by_lqr``/
          ``contract_deployment_does_not_use_plan_override``/``contract_runtime_ok``；
          eval/deploy 违反 ⇒ ``RuntimeError``（训练路径只落盘）。
        """
        rollout, world_model = resolve_forward_mode(forward_mode)
        _consume_legacy_forward_kwargs(legacy_signature)
        if self.path_head_trace is not None:
            # K1 门证据：每次 forward 先重置 trace（随后 encode 期的 t0 调用写入 step 0）。
            self.path_head_trace = []
        if self.contract_asserts_enabled:
            # R36 L-A 契约①：头调用实例 id 收集（本 forward 生命周期；默认关路径不触碰）
            self._path_head_call_ids = []
        encoded = self.encode(obs, semantic_off=bool(semantic_off))
        moe_aux: dict[str, Tensor] = encoded["moe_aux"]
        latent: Tensor = encoded["latent"]
        tokens: Tensor = encoded["tokens"]
        key_mask: Tensor = encoded["key_mask"]
        # A1：策略/价值 = 交叉注意力头（直吃令牌集合；含 plan_head 融合 latent token）
        # R38-P2b：正式 forward **只保留一个动作出口**——structured ⇒ 结构化头（旧 PolicyHead
        # 不跑；诊断已离线化，见 set_policy_diag / tools/diagnostics/legacy_policy_diag.py）；
        # 默认关 ⇒ 旧 PolicyHead（默认口径逐位不变）。
        path_out = encoded.get("path")
        path_out = path_out if isinstance(path_out, dict) else None
        action_logstd: Tensor | None = None
        if path_out is None:
            action_mu, action_logstd = self.policy(tokens, key_mask)
        else:
            action_mu = path_out["action_mu"]  # 旧头未前向 ⇒ 无 logstd 来源（文档化）
        out: dict[str, Tensor] = {
            "action_mu": action_mu,
            "value": self.value(tokens, key_mask),
            "router_logits": moe_aux["router_logits"],
            "expert_weights": moe_aux["expert_weights"],
            "latent": latent,
            "ego_next": encoded["ego_next"],
        }
        if action_logstd is not None:
            # 默认关路径：旧头提供 logstd（RL 未接线，方差来源与结构化头不同源）。
            out["action_logstd"] = action_logstd
        if path_out is not None:
            for path_key, path_value in path_out.items():
                if torch.is_tensor(path_value):
                    out[f"path_{path_key}"] = path_value
        # R36 H1：target-lane dropout 掩码（默认关 → None → 不输出）
        dropout_keep = encoded.get("target_lane_dropout_keep")
        if dropout_keep is not None:
            out["target_lane_dropout_keep"] = dropout_keep
        # R21-3A：semantic 诊断/下游（baseline 模式不输出 → 旧消费契约不变）
        if encoded.get("target_lane_token") is not None:
            out["target_lane_token"] = encoded["target_lane_token"]
            out["geometry_context"] = encoded["geometry_context"]
        if "load_balance_loss" in moe_aux:
            out["load_balance_loss"] = moe_aux["load_balance_loss"]
        if not rollout:
            if path_out is not None:
                # R36 K1：cheap path 无 WM ⇒ **不产生** WM 逐步 rollout 的 plan/位姿链。
                # 只给辅助预览（同头辅助输出；``plan_from_aux=1`` 显式标注），供诊断/监控；
                # 生产 LQR 参考必须走 wm rollout 模式（逐步组装 plan）。
                preview_plan = path_out["plan_aux"]
                replay = replay_plan(preview_plan)
                out["plan"] = preview_plan
                out["plan_from_aux"] = torch.ones((), dtype=action_mu.dtype, device=action_mu.device)
                out["traj_xy"] = path_out.get("traj_xy", replay["traj_xy"])
                out["traj_theta"] = path_out.get("traj_theta", replay["traj_theta"])
            if world_model:
                # 内部一致性不变量（FORWARD_MODE_FLAGS 不允许 world_model ∧ ¬rollout）
                raise ValueError("世界模型多步预测需要 rollout 模式（模式表不一致）")
            return out
        rolled = self._rollout(encoded, action_mu)
        if bool(self.probes is not None and self.probes.enabled) and self.probes.chain.enabled:
            # R38-D1 ⑤：统一出口 flags（plan 逐步组装 ⇒ plan_from_aux=0；plan0 == action_mu；
            # linkage 期望 z_pred = 逐步堆叠预测 latent）
            probes = self.probes
            probes.chain.note_flags(
                plan_from_aux=0.0,
                plan0_equals_action_mu=bool(torch.equal(rolled["plan"][:, 0, :], action_mu)),
            )
            probes.chain.note_z_pred(
                {
                    "z_ego": rolled.get("z_ego_pred"),
                    "z_od": rolled.get("z_od_pred"),
                    "z_ld": rolled.get("z_ld_pred"),
                }
            )
        out["traj_xy"] = rolled["traj_xy"]
        out["traj_theta"] = rolled["traj_theta"]
        out["plan"] = rolled["plan"]
        # R37 M1：warp 分项读数（默认关 ⇒ 不出现；见 set_warp_readout）
        for warp_key in (
            "warp_readout_valid",
            "warp_prior_err",
            "warp_residual_contrib",
            "warp_final_err",
        ):
            if warp_key in rolled:
                out[warp_key] = rolled[warp_key]
        if path_out is not None:
            # 逐步组装 plan（唯一动作生成器）+ 调用计数（契约门证据）；``plan_from_aux=0``。
            out["plan_from_aux"] = torch.zeros((), dtype=action_mu.dtype, device=action_mu.device)
            for key in (
                "structured_kappa_near",
                "structured_ds",
                "structured_aux_kappa_mid",
                "structured_aux_kappa_far",
                "structured_aux_progress",
                "structured_head_calls",
                "structured_wm_advances",
                "structured_plan_override_used",
            ):
                if key in rolled:
                    out[key] = rolled[key]
        if (
            self.contract_asserts_enabled
            and rollout
            and path_out is not None
            and self.path_head_mode() == "structured"
        ):
            # R36 L-A 运行时契约断言（GPT 7dcf6bd §四.1/§四.8；默认开，常数开销）：
            # eval/deploy（``training=False``）违反 ⇒ fail-fast；训练路径只计数落盘（不抛）。
            head_calls = float(rolled["structured_head_calls"])
            wm_advances = float(rolled["structured_wm_advances"])
            head_instances = float(len(set(self._path_head_call_ids)))
            override_used = float(rolled["structured_plan_override_used"])
            plan0_ok = bool(torch.equal(rolled["plan"][:, 0, :], action_mu))
            checks = {
                "policy_instance_ids_are_identical": head_instances == 1.0,
                "policy_call_count_is_six": head_calls == 6.0,
                "world_model_call_count_is_five": wm_advances == 5.0,
                "auxiliary_future_outputs_not_consumed_by_lqr": plan0_ok and override_used == 0.0,
                "deployment_does_not_use_plan_override": override_used == 0.0,
            }
            failed = sorted(name for name, ok in checks.items() if not ok)
            for name, ok in checks.items():
                out[f"contract_{name}"] = torch.tensor(
                    float(ok), dtype=action_mu.dtype, device=action_mu.device
                )
            out["contract_runtime_ok"] = torch.tensor(
                float(not failed), dtype=action_mu.dtype, device=action_mu.device
            )
            if failed and not self.training:
                raise RuntimeError(
                    f"[R36 L-A] structured 运行时契约断言失败（eval/deploy fail-fast）：{failed}；"
                    f"head_calls={head_calls} wm_advances={wm_advances} "
                    f"head_instances={head_instances} pinned_path_used={override_used} "
                    f"plan0_equals_action_mu={plan0_ok}"
                    f"（可用 model.set_contract_asserts(False) 关闭）"
                )
        if world_model:
            out["od_pred"] = rolled["od_pred"]
            out["ld_pred"] = rolled["ld_pred"]
            out["od_presence_pred"] = rolled["od_presence_pred"]
            out["od_entry_pred"] = rolled["od_entry_pred"]
            # v8（B3）：latent 状态诊断（rollout 逐步堆叠；不参与旧消费者契约）
            out["z_ego_pred"] = rolled["z_ego_pred"]
            out["z_od_pred"] = rolled["z_od_pred"]
            out["z_ld_pred"] = rolled["z_ld_pred"]
        # R21-27：只读计划曲率监控（纯派生，无训练逻辑；plan = (B,6,2) 的 (ds,dθ)）
        plan = out.get("plan")
        if torch.is_tensor(plan) and plan.ndim >= 2 and int(plan.shape[-1]) == 2:
            kappa = plan[..., 1] / plan[..., 0].clamp_min(1e-6)
            out["plan_kappa"] = kappa
            out["plan_delta_kappa"] = kappa[..., 1:] - kappa[..., :-1]
            out["plan_valid_kappa"] = (plan[..., 0].abs() >= 1e-3)
            # R21-29：Δκ 的 pairwise 有效掩码（相邻两步 ds 均 ≥ 0.5；无效低速点不得污染 Δκ 指标/投影）
            out["plan_valid_delta_kappa"] = (
                (plan[..., 0].abs() >= 0.5)[..., :-1] & (plan[..., 0].abs() >= 0.5)[..., 1:]
            )
        # R21-28：只读动作级物理量（final_*；legacy_* 见 semantic_action_decomposition）
        mu = out.get("action_mu")
        if torch.is_tensor(mu) and int(mu.shape[-1]) == 2:
            out["final_ds"] = mu[..., 0]
            out["final_dtheta"] = mu[..., 1]
            out["final_kappa"] = mu[..., 1] / mu[..., 0].clamp_min(1e-6)
            out["valid_kappa"] = (mu[..., 0].abs() >= 0.5)
        return out

    @torch.no_grad()
    def semantic_action_decomposition(
        self, obs: Mapping[str, Tensor], *, forward_mode: str = "policy_single_step", **legacy_signature: Any
    ) -> dict:
        """R21-27 只读监控：``final_action`` / ``legacy_action``（detach）/ ``semantic_delta_action`` + β。

        语义增量 = ``final − legacy``（legacy = semantic_off 基底前向，detach）。非 semantic 模式
        → delta 恒 0。**纯监控**：不参与训练目标（trainer 的 delta 监督自行计算，保持梯度语义）。

        R39-A1：``rollout=`` 布尔形参已删除（同 ``forward`` 口径）；``forward_mode`` 缺省 =
        ``policy_single_step``（= 旧缺省 ``rollout=False``，逐位不变）；残留 ``rollout=`` ⇒
        ``ValueError`` fail-fast。
        """
        if "rollout" in legacy_signature or "world_model" in legacy_signature:
            raise ValueError(
                "semantic_action_decomposition() 的布尔开关 rollout=/world_model= 已删除（R39-A1）；"
                f"请改用 forward_mode ∈ {FORWARD_MODES}（缺省 'policy_single_step'）"
            )
        if legacy_signature:
            raise TypeError(f"semantic_action_decomposition() 收到未知参数 {sorted(legacy_signature)}")
        final = self.forward(obs, forward_mode=str(forward_mode))["action_mu"]
        if not self._is_semantic():
            return {
                "final_action": final,
                "legacy_action": final.detach(),
                "semantic_delta_action": torch.zeros_like(final),
                "beta_context": 0.0,
                "beta_token": 0.0,
            }
        legacy = self.forward(obs, forward_mode=str(forward_mode), semantic_off=True)["action_mu"].detach()
        eps = 1e-6
        legacy_ds = legacy[..., 0]
        legacy_dtheta = legacy[..., 1]
        legacy_kappa = legacy_dtheta / legacy_ds.clamp_min(eps)
        final_kappa = final[..., 1] / final[..., 0].clamp_min(eps)
        return {
            "final_action": final,
            "legacy_action": legacy,
            "semantic_delta_action": final - legacy,
            "beta_context": float(self.beta_context.detach()) if self.beta_context is not None else 0.0,
            "beta_token": float(self.beta_token.detach()) if self.beta_token is not None else 0.0,
            # R21-28：横纵耦合审计（只读）
            "legacy_ds": legacy_ds,
            "legacy_dtheta": legacy_dtheta,
            "legacy_kappa": legacy_kappa,
            "final_ds": final[..., 0],
            "final_dtheta": final[..., 1],
            "final_kappa": final_kappa,
            "semantic_delta_kappa": final_kappa - legacy_kappa,
            "valid_kappa": (legacy_ds.abs() >= 0.5),
        }

    @torch.no_grad()
    def rollout(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """推理入口：与 :meth:`forward` 同输出，但全程 ``no_grad``。"""
        return self.forward(obs)
