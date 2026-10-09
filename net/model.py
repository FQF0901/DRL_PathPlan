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

v6 增补（docs/v6_net_design.md；A1/A2 去池化仍有效）
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
- K-anchor 保留（默认关）：``plan_anchors`` 的 ``lane_ctx`` 传 ``zeros(B,3)``（valid=0 →
  恒等变换）；``net.anchor.anchor_lane_context`` 保留但不再接线。

v7 结构迭代 B：K-anchor 计划头（``num_anchors>0``；默认 0 = 关闭 = 旧行为逐位不变）
---------------------------------------------------------------------------------------
- :class:`net.plan_head.PlanHead` 增加**选择头**（latent → K 锚 logits）、**连续速度头**
  （latent → 6 步 ds）与**逐锚 6×2 残差头**；方案 A 软混合
  ``plan = Σ_k p_k·(anchor_k + residual_k)``（``p = softmax(logits/τ)``；推理可 argmax），
  见 :mod:`net.anchor` 与 ``docs/v7_program_prereg.md`` §11。
- **rollout**：t0 一次算出的锚计划驱动尾段（step0 仍 = ``action_mu``，输出契约
  ``plan[:,0] == action_mu`` 不变；``traj_xy`` 与 ``plan`` 一致）；额外输出
  ``anchor_logits/anchor_probs/anchor_plan/anchor_speed/anchor_residual/anchor_ctx``。
- **PPO 兼容**：policy/PPO 路径（``action_mu/action_logstd/value/sample/logprob``）零改动；
  阶段 C 的 ``trainable_scope=design`` allowlist 不含锚头 → 计划头随阶段 B/phase3 训练后
  在 Stage C 默认冻结（WTA/CE 损失项在 ``freeze_mode=specific_only`` 下自动降级为 0）。

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

``forward(..., rollout=False, world_model=False)`` 是 PPO cheap path：省略
traj/rollout 多步键（不跑 st_gnn 消息传递），``action_mu/action_logstd/value`` 与完整前向
逐位一致。

向后兼容
--------
- 输入契约（:func:`net.mem.mem_from_obs`）= env schema v2 规范键
  （``ego_hist/od_hist/od_id_hist/od_presence_hist/ld_hist/others_hist`` + masks +
  ``hist_valid``；others 维默认 33 = schema v4：nav(11)+speed_limit(1)+signal(4)+
  static(5)+road_class(12)；旧 28 维数据自动重排 + 零填充 static 段 + 一次性告警）；
  缺失 ``ego_hist/others_hist`` 或 ``od_id_hist/od_presence_hist`` 时按模块 docstring
  的回退规则处理（旧数据集可直接复用）；
- ``nav/signal`` 键保留为可选上下文 token（v2 的规范上下文在 ``others`` 里）；
- ``wm_detach`` 形参保留但**恒为 no-op**：v8 的 latent 状态链 detach 语义是固定的，
  不提供消融开关。
"""

from __future__ import annotations

from typing import Mapping

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

from net.anchor import load_anchor_dictionary
from net.plan_head import PlanHead
from net.policy import ACTION_HIGH, ACTION_LOW, PolicyHead, ValueHead
from net.spatial import wrap_angle
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
        num_anchors: int = 0,
        anchor_path: str | None = None,
        anchor_temperature: float = 1.0,
        anchor_hard: bool = False,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.history_frames = int(history_frames)
        self.others_dim = int(others_dim)
        self.rollout_steps = int(wm_steps)
        self.dt = float(wm_dt)
        #: v7 结构迭代 B：K-anchor 计划头（0 = 关闭 = 旧行为逐位不变）
        self.num_anchors = int(num_anchors)

        self.encoders = ObsEncoders(hidden, od_slots, ld_slots, others_dim=others_dim)
        self.mem_encoder = MemEncoder(hidden)
        self.plan_head = PlanHead(
            hidden,
            num_experts,
            expert_hidden,
            router_hidden,
            primary_hidden=primary_hidden,
            top_k=moe_top_k,
            num_anchors=self.num_anchors,
            anchor_temperature=anchor_temperature,
            anchor_hard=anchor_hard,
        )
        if self.num_anchors > 0:
            # 锚字典：文件优先，缺失 → 内置默认（K=6 fix-3 原型）；非法文件直接报错。
            self.plan_head.set_anchors(
                load_anchor_dictionary(anchor_path, expected_k=self.num_anchors)
            )
        self.st_gnn = SpatioTemporalGNN(
            hidden, od_slots, ld_slots, steps=self.rollout_steps, dt=self.dt,
            spatial_layers=spatial_layers, od_knn=od_knn,
        )
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

    def set_moe(self, *, enabled: bool = True, load_balance_coef: float = 0.0) -> "DrivingModel":
        """MoE 运行时开关 + 负载均衡 α（lane U1；非参数，不进 state_dict）。

        phase 1（primary）关闭：专家不参与、输出严格 = primary；
        phase 2（specific）打开：``out = primary + Σ_{i∈top2} g_i·expert_i``（全场景生效）。
        推理默认开启（与训练 phase 2 口径一致）。
        """
        self.plan_head.set_moe(enabled=enabled, load_balance_coef=load_balance_coef)
        return self

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
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """**公开单步 plan head 入口**（rollout / Stage A/B 教师强制共用）。

        v8（B3）：融合输入 = ``[z_ego_k, mean(z_od_k), mean(z_ld_k), others, nav, signal]``
        （状态 latent 掩码均值池化 + t0 静态 others 上下文；nav 可为逐步重建 token）。
        返回 ``(latent, ego_next, moe_aux)``；``ego_next`` 监督"下一 ego 特征"，
        让 plan head/MoE 在教师强制路径中保持梯度（design-v1.2 §2.3 方案①）。
        """
        nav_token = squeeze_batch_singletons(nav_token, 2, "nav_token")
        signal_token = squeeze_batch_singletons(signal_token, 2, "signal_token")
        od_pool = _masked_mean(z_od, od_live)
        ld_pool = _masked_mean(z_ld, ld_live)
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
    ) -> tuple[Tensor, Tensor]:
        """交叉注意力头的令牌集合 + key mask（v8 B3，T=39）。

        顺序 = ``[z_od 16, z_ld 16, od_pool 1, ld_pool 1, others 1, z_ego 1, nav 1, signal 1,
        融合 latent 1]``；mask = ``[od_live, ld_live, 1, 1, 1, 1, nav_mask, signal_mask, 1]``
        （池化/others/z_ego/nav/signal/latent 中 nav/signal 按各自 mask，其余恒有效）。
        ``z_*`` = 当前步 latent 状态（rollout 逐步迭代；t0 = 当前帧编码）；
        ``od_pool/ld_pool`` = **t0 上下文**（``masked_mean(od_ctx/ld_ctx)``）。
        """
        ones = torch.ones((int(z_ego.shape[0]), 1), dtype=z_ego.dtype, device=z_ego.device)
        tokens = torch.cat(
            [
                z_od,
                z_ld,
                od_pool.unsqueeze(1),
                ld_pool.unsqueeze(1),
                others_ctx.unsqueeze(1),
                z_ego.unsqueeze(1),
                nav_token.unsqueeze(1),
                signal_token.unsqueeze(1),
                latent.unsqueeze(1),
            ],
            dim=1,
        )
        key_mask = torch.cat(
            [od_live, ld_live, ones, ones, ones, ones, nav_mask, signal_mask, ones],
            dim=1,
        )
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

    def encode(self, obs: Mapping[str, Tensor]) -> dict[str, object]:
        """观测 → mem/编码 mem/**latent 状态**/plan-head latent/头令牌集合（不跑 rollout）。

        v8（B3）：状态 latent = **当前帧编码**（``embed_ego(ego_now)`` / ``embed_od(od_now,
        od_live)`` / ``embed_ld(ld_now, ld_live)``）；上下文 token = t0 静态
        （``od_pool/ld_pool = masked_mean(od_ctx/ld_ctx)``、``others_ctx``、nav/signal）。
        另携带 A4 的世界系输入（``route_world``/``ego_world``，缺失时 rollout 回退 t0 冻结 nav）
        与 rollout 图位姿模板（``frame``）。
        """
        mem = self._mem_from_obs(obs)
        nav_token, nav_mask, signal_token, signal_mask = self._context_tokens(obs, mem.batch)
        encoded = self.mem_encoder.encode(self.encoders, mem)
        # v8（B3）：状态 latent = 当前帧编码（单帧，非时序聚合）
        ones = torch.ones((mem.batch, ), dtype=encoded.ego_now.dtype, device=encoded.ego_now.device)
        z_ego = self.encoders.embed_ego(encoded.ego_now, ones)
        z_od = self.encoders.embed_od(encoded.od_now, encoded.od_live, ids=encoded.od_id_now)
        z_ld = self.encoders.embed_ld(encoded.ld_now, encoded.ld_live)
        # 上下文 token（t0 静态）：池化来自时序聚合 ctx
        od_pool = _masked_mean(encoded.od_ctx, encoded.od_live)
        ld_pool = _masked_mean(encoded.ld_ctx, encoded.ld_live)
        latent, ego_next, moe_aux = self.plan_step(
            z_ego, z_od, z_ld, encoded.od_live, encoded.ld_live, encoded.others_ctx,
            nav_token, signal_token,
        )
        # v7 结构迭代 B：K-anchor 计划（v8：lane 通道已删除 → lane_ctx 恒为 zeros(B,3)，
        # valid=0 → 锚混合恒等变换；anchor_lane_context 保留但不再接线）
        anchor: dict[str, Tensor] | None = None
        if self.num_anchors > 0:
            lane_ctx = torch.zeros((mem.batch, 3), dtype=latent.dtype, device=latent.device)
            anchor = self.plan_head.plan_anchors(latent, lane_ctx)
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
            "anchor": anchor,
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
        """
        mem: MemBank = encoded["mem"]  # 只读（nav 回退锚点）；不再拷贝/滑动
        enc: EncodedMem = encoded["encoded"]
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
        # v7 结构迭代 B：K-anchor 计划（t0 一次性软混合；step0 仍由 action_mu 钉住 → 输出契约不变）
        anchor = encoded.get("anchor")
        anchor_plan: Tensor | None = anchor.get("plan") if isinstance(anchor, dict) else None

        # t0 锚定状态与图位姿模板（t0 帧；来自真实 obs，不 detach —— 检测任务）
        od_anchor = self.st_gnn.od_state_from_features(enc.od_now, od_live)
        ld_anchor = self.st_gnn.ld_state_from_features(enc.ld_now, ld_live)
        node_mask: Tensor = frame.node_mask
        od_pose_t0 = frame.pose[:, 1 : 1 + self.od_slots]
        ld_pose_t0 = frame.pose[:, 1 + self.od_slots :]
        od_velocity = enc.od_now[..., 2:4] * od_live.unsqueeze(-1)  # (B,S,2) t0 相对速度

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

            # (a2) A4：推进世界系位姿（与教师强制同一步进语义）并逐步重建 nav/signal
            if torch.is_tensor(pose_world):
                pose_world = advance_pose_world(pose_world, ds, dtheta)
            nav_token, nav_mask, signal_token, signal_mask = self._step_context(
                mem, pose_world, world, nav_token, nav_mask, signal_token, signal_mask, k
            )

            # (b) 图位姿（t0 帧）：ego = 累积位姿（detach：WM 输出不回传动作/策略链）；
            #     OD = t0 位姿 + k·dt·v_t0；LD 静止
            offset = float(k) * self.dt
            od_pose_k = torch.cat(
                [od_pose_t0[..., :2] + od_velocity * offset, od_pose_t0[..., 2:]], dim=-1
            )
            pose_k = torch.cat([pose.detach().unsqueeze(1), od_pose_k, ld_pose_t0], dim=1)

            # (c) latent 自回归转移 + 物理解码（t0 帧预测，逐步堆叠）
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
                )
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
                latent, _, _ = self.plan_step(
                    z_ego, z_od, z_ld, od_live, ld_live, others_ctx, nav_token, signal_token
                )
                if anchor_plan is None:
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
                    )
                    action, _ = self.policy(tokens, key_mask)
                else:
                    # K-anchor 方案 A：尾段计划由 t0 锚混合一次性给出（action/pose 链保持可微）
                    action = anchor_plan[:, k, :]

        return {
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

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        obs: Mapping[str, Tensor],
        *,
        rollout: bool = True,
        world_model: bool = True,
        wm_detach: bool | None = None,
    ) -> dict[str, Tensor]:
        """完整前向：策略/价值/路由 + （可选）latent 自回归 rollout 与 t0 帧物理解码。

        ``rollout=False, world_model=False`` 时走廉价路径：只做编码（**不再跑任何 st_gnn
        消息传递**）+ plan head + 策略/价值头，不跑 rollout，返回的
        ``action_mu/action_logstd/value/router_logits/expert_weights/latent`` 与完整前向
        逐位一致（PPO 收集/BC update 用）。

        ``world_model=False``（rollout=True）时仍执行 rollout（轨迹/动作需要），
        但**不返回** ``od_pred/ld_pred/od_presence_pred/od_entry_pred`` 与 ``z_*_pred``。

        ``wm_detach``：仅为兼容旧签名保留，**no-op**（v8 的 latent 状态链 detach 固定生效）。

        MoE 输出口径（lane U1）：``primary + Σ_{i∈top2} g_i·expert_i``，推理与训练一致、
        全场景生效（无硬切/二值门/硬掩码）；phase 1 训练由 ``set_moe(enabled=False)`` 关闭。
        """
        encoded = self.encode(obs)
        moe_aux: dict[str, Tensor] = encoded["moe_aux"]
        latent: Tensor = encoded["latent"]
        tokens: Tensor = encoded["tokens"]
        key_mask: Tensor = encoded["key_mask"]
        # A1：策略/价值 = 交叉注意力头（直吃令牌集合；含 plan_head 融合 latent token）
        action_mu, action_logstd = self.policy(tokens, key_mask)
        out: dict[str, Tensor] = {
            "action_mu": action_mu,
            "action_logstd": action_logstd,
            "value": self.value(tokens, key_mask),
            "router_logits": moe_aux["router_logits"],
            "expert_weights": moe_aux["expert_weights"],
            "latent": latent,
            "ego_next": encoded["ego_next"],
        }
        if "load_balance_loss" in moe_aux:
            out["load_balance_loss"] = moe_aux["load_balance_loss"]
        # v7 结构迭代 B：K-anchor 计划输出（num_anchors=0 时不存在 → 旧契约逐位不变）
        anchor = encoded.get("anchor")
        if isinstance(anchor, dict):
            out["anchor_logits"] = anchor["logits"]
            out["anchor_probs"] = anchor["probs"]
            out["anchor_plan"] = anchor["plan"]
            out["anchor_speed"] = anchor["speed_ds"]
            out["anchor_residual"] = anchor["residual"]
            out["anchor_ctx"] = anchor["ctx"]
        if not rollout:
            if world_model:
                raise ValueError("world_model=True 需要 rollout=True（多步预测是 rollout 的产物）")
            return out
        rolled = self._rollout(encoded, action_mu)
        out["traj_xy"] = rolled["traj_xy"]
        out["traj_theta"] = rolled["traj_theta"]
        out["plan"] = rolled["plan"]
        if world_model:
            out["od_pred"] = rolled["od_pred"]
            out["ld_pred"] = rolled["ld_pred"]
            out["od_presence_pred"] = rolled["od_presence_pred"]
            out["od_entry_pred"] = rolled["od_entry_pred"]
            # v8（B3）：latent 状态诊断（rollout 逐步堆叠；不参与旧消费者契约）
            out["z_ego_pred"] = rolled["z_ego_pred"]
            out["z_od_pred"] = rolled["z_od_pred"]
            out["z_ld_pred"] = rolled["z_ld_pred"]
        return out

    @torch.no_grad()
    def rollout(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """推理入口：与 :meth:`forward` 同输出，但全程 ``no_grad``。"""
        return self.forward(obs)
