"""``DrivingModel``：mem-bank 输入 → 注意力聚合 → plan head(MoE) → 递归 rollout + ST-GNN。

v2 数据流（mem-bank + 递归 rollout + plan-head MoE）
---------------------------------------------------
1. **输入**：4 个 per-modality 真 mem（``ego_hist/od_hist/ld_hist/others_hist`` + mask、
   ``od_id_hist/od_presence_hist``，见 :mod:`net.mem`）。**net 只读**；真 mem 的更新只由 env 完成。
2. **编码**：:class:`net.mem.MemEncoder` 把 6 帧 mem 编码成"编码 mem"——OD / Ego /
   Others 各做掩码注意力池化，**LD 不做时序**（直接用当前帧，理由见该模块 docstring）。
3. **Plan head**：4 个 mem 聚合 + nav/signal → MLP 融合 → **MoE**（primary 常开 + 8 个
   specific 专家、top-2 软混合）→ 产出**下一时刻 ego 特征**与策略/价值 latent；
   ``router_logits(8)``/``expert_weights(8)`` 原样输出。OD/LD 的**掩码均值池化只保留在
   这条融合路径**（v6 §1.2 去池化）。
4. **ST-GNN**：以"更新后的 ego mem + od/ld mem"为条件推演下一时刻 OD/LD（t0 帧预测
   空间），并给出 ``od_presence_pred``/``od_entry_pred``。
5. **递归 rollout（6 步）**：每次从真 mem **拷贝 4 份**，逐步：
   plan head(mem 副本) → 下一 ego 特征 → 挤入 ego 副本并弹出最老帧；
   ST-GNN → 下一 OD/LD 预测 → **逆变换到当前累积位姿系** → 重编码/携带静态属性 →
   挤入各自副本并弹出最老帧。重复 6 次（下次 rollout 重新拷贝）。
   真 mem 绝不被写入（``MemBank.clone`` 隔离；有断言测试）。
6. **detach 语义（规格第 5 条，固定不可配）**：来自真实 obs 的帧（step0）**不 detach**
   ——第一步预测/轨迹的梯度直达编码器（检测任务）；rollout 合成的后续 5 帧在**挤入
   mem 副本前 detach**（预测任务不回传状态链）。被切的是"帧（状态）"，
   **action/pose 链保持可微**：``traj_xy`` 的梯度沿 pose 链训练 plan head 的动作链。
7. **坐标**：ST-GNN 在 **t0 帧**预测；``od_pred_to_features``/``ld_pred_to_features``
   做 SE(2) 逆变换到当前累积位姿系，再重编码挤入 mem（既有正确实现，规格第 6 条）。

v6 增补（docs/v6_net_design.md，冻结）
-------------------------------------
- **A1/A2 去池化**：policy/value 不再吃池化 token，改为**交叉注意力头**
  （:class:`net.policy.CrossAttnHead`）直吃令牌集合（OD/LD 逐槽 + others + ego + nav +
  signal + plan_head 融合 latent；``_head_tokens``）；池化仅保留在 plan head 融合路径。
- **A3 t0 单次 st_gnn 消息传递**：``encode`` 对 t0 帧跑**一次** ``st_gnn.spatial``
  消息传递（对象级 OD/LD 节点特征供注意力头；见 :meth:`DrivingModel._t0_object_features`）。
  cheap path（``rollout=False``）与完整前向**都**经过 ``encode`` ⇒ collect/update 一致；
  6 步 rollout 内的逐步 st_gnn 语义原样不动。该 t0 pass 在 ``no_grad`` 下执行：
  st_gnn 的训练信号保持 WM 损失口径（traj/policy 损失不回传 st_gnn，既有测试锁定）。
- **A4 nav 修正**：``_rollout`` 内逐步按**世界系** ego 位姿重建 nav/signal 与 others mem
  的 nav 子向量（:func:`rebuild_nav_from_world` / :func:`sync_others_nav` /
  :func:`advance_signal`；无 ``route_world``/``ego_world`` 键时回退 t0 冻结，逐位兼容旧输入）。

v5 增补（结构迭代 A，仅令牌集合接线）
------------------------------------
- ``lane``（当前车道块，17 维）与 ``ttc``（OD 槽位 TTC 上下文，12 维）由
  :func:`net.mem.context_features_from_obs` 从当前帧 obs 读出并编码为 token；
  令牌集合变为 ``[OD 16, LD 16, lane 1, others 1, ego 1, nav 1, ttc 1, signal 1, latent 1]``
  （T=39；lane 与 LD 主块并列、ttc 与 nav 相邻）。旧 schema v4 数据缺键 → 全 0 + mask=0 +
  一次性告警（A4/v4 模式），不参与注意力、逐位兼容。
- lane/ttc token 为 **t0 上下文**：rollout 逐步复用（TTC 的逐步重算属后续 Lane B 的
  plan/rollout 路径；本迭代只做令牌集合接线）。
- ``plan_head`` 融合路径与 ST-GNN 空间节点（[ego, OD, LD]）**不动**（Lane B）。

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
``ld_pred (B,6,16,4)``（LD 预测作 rollout 输入/诊断 + **未来 LD 监督已恢复**：stage A
``direct_multi_step``（``ld_fut`` 前 4 维；lane P3-F），stage B phase 3 亦监督）、
``od_presence_pred (B,6,16)``、``od_entry_pred (B,6,16)``（logits）、``traj_theta (B,6)``。

``forward(..., rollout=False, world_model=False)`` 是 PPO cheap path：省略
traj/ST-GNN 多步键，``action_mu/action_logstd/value`` 与完整前向逐位一致。

向后兼容
--------
- 输入契约（:func:`net.mem.mem_from_obs`）= env schema v2 规范键
  （``ego_hist/od_hist/od_id_hist/od_presence_hist/ld_hist/others_hist`` + masks +
  ``hist_valid``；others 维默认 33 = schema v4：nav(11)+speed_limit(1)+signal(4)+
  static(5)+road_class(12)；旧 28 维数据自动重排 + 零填充 static 段 + 一次性告警）；
  缺失 ``ego_hist/others_hist`` 或 ``od_id_hist/od_presence_hist`` 时按模块 docstring
  的回退规则处理（旧数据集可直接复用）；
- ``nav/signal`` 键保留为可选上下文 token（v2 的规范上下文在 ``others`` 里）；
- ``wm_detach`` 形参保留但**恒为 no-op**：v2 的合成帧 detach 语义是固定的（规格第 5 条），
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
    LANE_DIM,
    LD_SLOTS,
    NAV_DIM,
    OD_SLOTS,
    SIGNAL_DIM,
    TTC_DIM,
    ObsEncoders,
)
from net.mem import (
    EncodedMem,
    MemBank,
    MemEncoder,
    context_features_from_obs,
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


# ------------------------------------------------------------- rollout 特征重建
def od_pred_to_features(od_pred: Tensor, pose: Tensor, carry: Tensor, mask: Tensor) -> Tensor:
    """t0 帧 OD 预测 ``(B,16,5)`` → ``pose`` 自车系下的原始 9 维特征。

    ``L/W/type_id`` 是静态属性，沿用 ``carry``（上一帧原始特征）；掩码外清零。
    """
    px, py, theta = pose[:, 0:1], pose[:, 1:2], pose[:, 2:3]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    dx = od_pred[..., 0] - px
    dy = od_pred[..., 1] - py
    heading = wrap_angle(od_pred[..., 4] - theta)
    features = torch.stack(
        [
            cos_t * dx + sin_t * dy,
            -sin_t * dx + cos_t * dy,
            cos_t * od_pred[..., 2] + sin_t * od_pred[..., 3],
            -sin_t * od_pred[..., 2] + cos_t * od_pred[..., 3],
            torch.cos(heading),
            torch.sin(heading),
            carry[..., 6],
            carry[..., 7],
            carry[..., 8],
        ],
        dim=-1,
    )
    return features * mask.unsqueeze(-1)


def ld_pred_to_features(ld_pred: Tensor, pose: Tensor, carry: Tensor, mask: Tensor) -> Tensor:
    """t0 帧 LD 预测 ``(B,16,4)`` → ``pose`` 自车系下的原始 7 维特征。

    ``speed_limit/线型`` 是静态属性，沿用 ``carry``；掩码外清零。
    """
    px, py, theta = pose[:, 0:1], pose[:, 1:2], pose[:, 2:3]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    dx = ld_pred[..., 0] - px
    dy = ld_pred[..., 1] - py
    heading = wrap_angle(ld_pred[..., 2] - theta)
    features = torch.stack(
        [
            cos_t * dx + sin_t * dy,
            -sin_t * dx + cos_t * dy,
            heading,
            ld_pred[..., 3],
            carry[..., 4],
            carry[..., 5],
            carry[..., 6],
        ],
        dim=-1,
    )
    return features * mask.unsqueeze(-1)


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

    **纯函数版**（不原地改写 ``mem.others``；调用方用 :func:`sync_others_nav` 赋值回写）。
    为什么不用 ``net.mem.rebuild_nav_from_world`` 的原地版本：rollout/教师强制的 ``mem``
    是 autograd 图中的 clone，原地写 ``mem.others`` 会在 backward 触发
    "modified by an inplace operation"（该张量已被 encoder 保存）；赋值新张量则安全。
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


def sync_others_nav(others: Tensor, nav_features: Tensor) -> Tensor:
    """A4：把 others mem **当前帧（末帧）** 的 nav 子向量（0..NAV_DIM）替换为重建值。

    历史帧保持各自时刻的 nav（rollout 不滑动 others mem，只更新"当前"任务条件）；
    返回新张量（拷贝隔离，不原地改写）。
    """
    if others.ndim != 3 or int(others.shape[-1]) < NAV_DIM:
        raise ValueError(f"others 应为 (B,T,F) 且 F>={NAV_DIM}，收到 {tuple(others.shape)}")
    if nav_features.ndim != 2 or int(nav_features.shape[-1]) != NAV_DIM:
        raise ValueError(f"nav_features 应为 (B,{NAV_DIM})，收到 {tuple(nav_features.shape)}")
    updated = others.clone()
    updated[:, -1, :NAV_DIM] = nav_features.to(updated.dtype)
    return updated


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
        expert_hidden: int = 192,
        router_hidden: int = 64,
        moe_top_k: int = 2,
        wm_steps: int = 6,
        wm_dt: float = 0.5,
        action_low: tuple[float, float] = ACTION_LOW,
        action_high: tuple[float, float] = ACTION_HIGH,
        log_std_init: float = -1.0,
        attn_heads: int = 4,
        attn_layers: int = 1,
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
        self.plan_head = PlanHead(hidden, num_experts, expert_hidden, router_hidden, top_k=moe_top_k)
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
        )
        self.value = ValueHead(hidden, num_heads=int(attn_heads), layers=int(attn_layers))

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

    def _struct_context_tokens(
        self, obs: Mapping[str, Tensor], batch: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """v5：当前车道块（``lane``）与 TTC token（``ttc``）→ token+mask 四元组。

        ``lane``/``ttc`` 是当前帧上下文通道（不进 6 帧历史）；缺键（旧 schema v4 数据）→
        全 0 + mask=0 + 一次性告警（见 :func:`net.mem.context_features_from_obs`）。
        """
        lane_feat, lane_mask = context_features_from_obs(obs, "lane", LANE_DIM, batch=batch)
        ttc_feat, ttc_mask = context_features_from_obs(obs, "ttc", TTC_DIM, batch=batch)
        lane_token = self.encoders.embed_lane(lane_feat.unsqueeze(1), lane_mask)[:, 0]
        ttc_token = self.encoders.embed_ttc(ttc_feat.unsqueeze(1), ttc_mask)[:, 0]
        return lane_token, lane_mask, ttc_token, ttc_mask

    # ---------------------------------------------------------------- 编码/规划
    def _plan(
        self, encoded: EncodedMem, nav_token: Tensor, signal_token: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """plan head 融合路径（v6：OD/LD 池化只在这里）。

        nav/signal token 兼容 ``(B,H)`` 与 ``(B,1,H)``（A4 调用点常直接传
        ``encoders.embed_nav(...)`` 的 (B,1,H) 输出）。
        """
        nav_token = squeeze_batch_singletons(nav_token, 2, "nav_token")
        signal_token = squeeze_batch_singletons(signal_token, 2, "signal_token")
        od_pool = _masked_mean(encoded.od_ctx, encoded.od_live)
        ld_pool = _masked_mean(encoded.ld_ctx, encoded.ld_live)
        return self.plan_head(
            encoded.ego_ctx, od_pool, ld_pool, encoded.others_ctx, nav_token, signal_token
        )

    def _t0_object_features(self, encoded: EncodedMem) -> tuple[Tensor, Tensor]:
        """A3：t0 帧**单次 st_gnn 消息传递** → 对象级特征 ``(od (B,S,H), ld (B,L,H))``。

        经 :meth:`SpatioTemporalGNN.node_features` 公共委托执行（step_index=1），不跑解码器
        ——注意力头只要对象级节点特征；模型不再直连 ``st_gnn.spatial``/``st_gnn.step_embed``。
        在 ``no_grad`` 下执行：st_gnn 的训练信号保持 WM 损失口径（traj/policy 损失不得
        回传 st_gnn；tests/test_stage_v11.py 锁定），本 pass 只提供"当前权重下的特征"。
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
        encoded: EncodedMem,
        nav_token: Tensor,
        nav_mask: Tensor,
        signal_token: Tensor,
        signal_mask: Tensor,
        latent: Tensor,
        lane_token: Tensor,
        lane_mask: Tensor,
        ttc_token: Tensor,
        ttc_mask: Tensor,
        od_obj: Tensor | None = None,
        ld_obj: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """A1：交叉注意力头的令牌集合 + key mask（v6 §1.2；v5 结构迭代 A 扩到 T ≤ 39）。

        顺序 = ``[OD 16, LD 16, lane 1, others 1, ego 1, nav 1, ttc 1, signal 1, 融合 latent 1]``；
        mask = ``[od_live, ld_live, lane_mask, 1, 1, nav_mask, ttc_mask, signal_mask, 1]``
        （ego/latent 恒有效）。``lane`` 与 LD 主块并列、``ttc`` 与 nav 相邻（v5）。
        ``od_obj/ld_obj`` = t0 单次 st_gnn 消息传递后的对象级特征（t0 头用）；
        缺省（rollout 步）直接用 ``encoded.od_ctx/ld_ctx``。
        """
        od_tokens = encoded.od_ctx if od_obj is None else od_obj
        ld_tokens = encoded.ld_ctx if ld_obj is None else ld_obj
        ones = torch.ones((int(encoded.ego_ctx.shape[0]), 1), dtype=encoded.ego_ctx.dtype, device=encoded.ego_ctx.device)
        tokens = torch.cat(
            [
                od_tokens,
                ld_tokens,
                lane_token.unsqueeze(1),
                encoded.others_ctx.unsqueeze(1),
                encoded.ego_ctx.unsqueeze(1),
                nav_token.unsqueeze(1),
                ttc_token.unsqueeze(1),
                signal_token.unsqueeze(1),
                latent.unsqueeze(1),
            ],
            dim=1,
        )
        key_mask = torch.cat(
            [encoded.od_live, encoded.ld_live, lane_mask, ones, ones, nav_mask, ttc_mask, signal_mask, ones],
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
        """A4：rollout 单步重建 nav/signal（+ others mem 的 nav 维同步）。

        ``pose_world`` = 世界系 ego 位姿（每步由 :func:`advance_pose_world` 推进，与
        教师强制同一步进语义）。无 ``route_world``/``ego_world`` 时原样返回 t0 上下文
        （旧输入逐位兼容）。
        """
        route_world = world.get("route_world")
        route_world_mask = world.get("route_world_mask")
        if route_world is None or not torch.is_tensor(pose_world) or not torch.is_tensor(route_world):
            return nav_token, nav_mask, signal_token, signal_mask
        nav_feat, nav_mask = rebuild_nav_from_world(
            mem, pose_world, route_world, route_world_mask
        )
        # others mem 的 nav 子向量（0..NAV_DIM）同步重建（不得只改独立 nav token）
        mem.others = sync_others_nav(mem.others, nav_feat)
        nav_token = self.encoders.embed_nav(nav_feat.unsqueeze(1), nav_mask.reshape(-1, 1))[:, 0]
        signal_token, signal_mask = advance_signal(signal_token, signal_mask, int(step_index))
        return nav_token, nav_mask, signal_token, signal_mask

    def encode(self, obs: Mapping[str, Tensor]) -> dict[str, object]:
        """观测 → mem/编码 mem/plan-head latent/头令牌集合（不跑递归 rollout）。

        v6：本方法含 **t0 帧单次 st_gnn 消息传递**（A3）——cheap path 与完整前向都经此，
        collect/update 一致执行；另携带 A4 的世界系输入（``route_world``/``ego_world``，
        缺失时 rollout 回退 t0 冻结 nav）。
        """
        mem = self._mem_from_obs(obs)
        nav_token, nav_mask, signal_token, signal_mask = self._context_tokens(obs, mem.batch)
        # v5（结构迭代 A）：当前车道块 + TTC 上下文 token（旧数据缺键 → 0 token + mask=0）
        lane_token, lane_mask, ttc_token, ttc_mask = self._struct_context_tokens(obs, mem.batch)
        encoded = self.mem_encoder.encode(self.encoders, mem)
        latent, ego_next, moe_aux = self._plan(encoded, nav_token, signal_token)
        od_obj, ld_obj = self._t0_object_features(encoded)  # A3：单次消息传递（no_grad）
        tokens, key_mask = self._head_tokens(
            encoded,
            nav_token,
            nav_mask,
            signal_token,
            signal_mask,
            latent,
            lane_token,
            lane_mask,
            ttc_token,
            ttc_mask,
            od_obj,
            ld_obj,
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
            "latent": latent,
            "ego_next": ego_next,
            "moe_aux": moe_aux,
            "nav_token": nav_token,
            "nav_mask": nav_mask,
            "signal_token": signal_token,
            "signal_mask": signal_mask,
            "lane_token": lane_token,
            "lane_mask": lane_mask,
            "ttc_token": ttc_token,
            "ttc_mask": ttc_mask,
            "tokens": tokens,
            "key_mask": key_mask,
            "route_world": route_world,
            "route_world_mask": route_world_mask,
            "ego_world": ego_world,
        }

    def plan_step(
        self,
        encoded_mem: EncodedMem,
        nav_token: Tensor,
        signal_token: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """**公开单步 plan head 入口**（rollout / Stage A 教师强制共用）。

        ``encoded_mem`` 为 :meth:`MemEncoder.encode` 的输出（mem 当前帧，可含教师强制帧）。
        返回 ``(latent, ego_next, moe_aux)``；Stage A 用 ``ego_next`` 监督"下一 ego 特征"，
        让 plan head/MoE 在教师强制路径中保持梯度（design-v1.2 §2.3 方案①）。
        """
        return self._plan(encoded_mem, nav_token, signal_token)

    # ---------------------------------------------------------------- rollout
    def _rollout(self, encoded: dict[str, object], action0: Tensor) -> dict[str, Tensor]:
        """6 步递归 rollout（拷贝隔离 + 合成帧 detach + action/pose 链可微）。

        ``action0`` = 传入的 ``action_mu``（第 1 个计划动作；与 cheap path 同一计算）。
        返回：``traj_xy/traj_theta/plan``（位姿链）与 ``od_pred/ld_pred/
        od_presence_pred/od_entry_pred``（t0 帧预测，逐步堆叠）。

        A4：每步按世界系 ego 位姿重建 nav/signal 与 others mem 的 nav 维（``_step_context``）；
        无 ``route_world``/``ego_world`` 输入时回退 t0 冻结上下文（旧行为逐位兼容）。
        A1：rollout 步的注意力头令牌用当步 ``enc`` 的逐槽特征（不额外跑消息传递——t0 单次
        pass 已在 ``encode`` 内；rollout 的 6 次 st_gnn 语义/成本原样不动）。
        """
        mem: MemBank = encoded["mem"].clone()  # 拷贝隔离：真 mem 绝不写回
        enc: EncodedMem = encoded["encoded"]
        latent: Tensor = encoded["latent"]
        ego_next: Tensor = encoded["ego_next"]
        nav_token: Tensor = encoded["nav_token"]
        nav_mask: Tensor = encoded["nav_mask"]
        signal_token: Tensor = encoded["signal_token"]
        signal_mask: Tensor = encoded["signal_mask"]
        # v5（结构迭代 A）：lane/ttc 是 t0 上下文 token（逐步复用；重算属 Lane B）
        lane_token: Tensor = encoded["lane_token"]
        lane_mask: Tensor = encoded["lane_mask"]
        ttc_token: Tensor = encoded["ttc_token"]
        ttc_mask: Tensor = encoded["ttc_mask"]
        world = {
            "route_world": encoded.get("route_world"),
            "route_world_mask": encoded.get("route_world_mask"),
            "ego_world": encoded.get("ego_world"),
        }
        pose_world = encoded.get("ego_world")  # A4 世界系位姿锚点（t0）；每步 advance_pose_world
        action = action0  # 第 1 个计划动作 = action_mu（与 cheap path 逐位一致）

        # t0 锚定状态（t0 帧；来自真实 obs，不 detach —— 检测任务）
        od_anchor = self.st_gnn.od_state_from_features(enc.od_now, enc.od_live)
        ld_anchor = self.st_gnn.ld_state_from_features(enc.ld_now, enc.ld_live)

        pose = torch.zeros((mem.batch, 3), dtype=latent.dtype, device=latent.device)
        xy: list[Tensor] = []
        theta: list[Tensor] = []
        actions: list[Tensor] = []
        od_steps: list[Tensor] = []
        ld_steps: list[Tensor] = []
        presence_steps: list[Tensor] = []
        entry_steps: list[Tensor] = []
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

            # (b) plan head 产出的下一 ego 特征 → 挤入 ego mem 副本（合成帧 detach）
            ego_frame = torch.cat([ego_next, action], dim=-1)  # reserved 维 = 执行动作
            mem.shift_ego(ego_frame.detach())

            # (c) ST-GNN：更新后的 ego mem + od/ld mem → t0 帧的下一 OD/LD 预测
            enc_k = self.mem_encoder.encode(self.encoders, mem)
            node_mask = enc_k.frame.node_mask
            od_pred, ld_pred, presence_logit, entry_logit = self.st_gnn(
                ego_ctx=enc_k.ego_ctx,
                od_ctx=enc_k.od_ctx,
                ld_ctx=enc_k.ld_ctx,
                node_mask=node_mask,
                pose=enc_k.frame.pose,
                step_index=k,
                od_anchor=od_anchor,
                ld_anchor=ld_anchor,
            )
            od_steps.append(od_pred)
            ld_steps.append(ld_pred)
            presence_steps.append(presence_logit)
            entry_steps.append(entry_logit)

            # (d) SE(2) 逆变换到当前累积位姿系 → detach → 挤入 od/ld mem 副本
            presence_prob = torch.sigmoid(presence_logit)
            entry_prob = torch.sigmoid(entry_logit)
            od_push_mask = (
                (presence_prob > 0.5) & ((enc_k.od_live > 0.5) | (entry_prob > 0.5))
            ).to(enc_k.od_now.dtype)
            od_body = od_pred_to_features(od_pred, pose, enc_k.od_now, od_push_mask).detach()
            ld_body = ld_pred_to_features(ld_pred, pose, enc_k.ld_now, enc_k.ld_live).detach()
            mem.shift_od_ld(
                od_body, od_push_mask, enc_k.od_id_now, presence_prob.detach(), ld_body, enc_k.ld_live
            )

            # (e) 下一轮 plan head + 注意力头（最后一步之后不需要）
            if k < self.rollout_steps:
                enc = self.mem_encoder.encode(self.encoders, mem)
                latent, ego_next, _ = self._plan(enc, nav_token, signal_token)
                tokens, key_mask = self._head_tokens(
                    enc,
                    nav_token,
                    nav_mask,
                    signal_token,
                    signal_mask,
                    latent,
                    lane_token,
                    lane_mask,
                    ttc_token,
                    ttc_mask,
                )
                action, _ = self.policy(tokens, key_mask)

        return {
            "traj_xy": torch.stack(xy, dim=1),
            "traj_theta": torch.stack(theta, dim=1),
            "plan": torch.stack(actions, dim=1),
            "od_pred": torch.stack(od_steps, dim=1),
            "ld_pred": torch.stack(ld_steps, dim=1),
            "od_presence_pred": torch.stack(presence_steps, dim=1),
            "od_entry_pred": torch.stack(entry_steps, dim=1),
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
        """完整前向：策略/价值/路由 + （可选）递归 rollout 与 t0 帧 OD/LD 预测。

        ``rollout=False, world_model=False`` 时走廉价路径：做编码（**含 A3 的 t0 单次
        st_gnn 消息传递**）+ plan head + 策略/价值头，不跑递归 rollout，返回的
        ``action_mu/action_logstd/value/router_logits/expert_weights/latent`` 与完整前向
        逐位一致（PPO 收集/BC update 用；collect/update 两路径一致执行该单次 pass）。

        ``world_model=False``（rollout=True）时仍执行 rollout（轨迹/动作需要），
        但**不返回** ``od_pred/ld_pred/od_presence_pred/od_entry_pred``。

        ``wm_detach``：仅为兼容旧签名保留，**no-op**（v2 的合成帧 detach 固定生效）。

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
        return out

    @torch.no_grad()
    def rollout(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """推理入口：与 :meth:`forward` 同输出，但全程 ``no_grad``。"""
        return self.forward(obs)
