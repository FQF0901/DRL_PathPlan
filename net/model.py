"""``DrivingModel``：观测 → 时序 GRU → 空间消息传递 → MoE → 策略/价值 + 世界模型 + B1 rollout。

数据流
------
1. **编码**：各通道共享投影到 H=96；节点顺序 ``[ego, 16 OD, 16 LD]``，
   nav/signal 为全局上下文 token；
2. **时序**：OD/LD 逐节点 GRU 过 6 帧历史，``hist_valid`` 门控预热补位帧；
   当前帧嵌入与历史隐状态相加融合；
3. **空间**：2 层消息传递（OD↔ego、LD↔ego、LD↔LD 相邻、OD↔OD 近邻，边=相对位姿）；
4. **池化 + MoE**：``[ego, mean(OD), mean(LD), nav, signal]`` 池化为场景 latent，
   MoE（primary 恒激活 + 8 个门控 expert）细化 latent；
5. **输出**：策略头 ``action_mu/action_logstd (B,2)``、价值 ``value (B,1)``、
   世界模型直接多步 ``od_pred (B,6,16,5) / ld_pred (B,6,16,4)``、
   路由 ``router_logits (B,8)``、``latent (B,H)``；
6. **B1 rollout**：策略自回归 ×6——每步用策略均值动作推进 ego 圆弧运动学，
   世界模型预测 t0 帧下的 OD/LD 下一步，变换回新自车系后重编码，
   得到 ``traj_xy (B,6,2)``（自车系，t=0.5..3.0 s 的 6 个端点）。

动作与运动学
------------
动作 ``(ds, dθ)`` = 下一个 0.5 s 的弧长 + 航向变化；:func:`arc_step` 是 net 内部
唯一圆弧实现（p2-contract §8.5 要求与 ``env/tracking`` 同一约定），
:func:`interpolate_actions` 提供 6→30 点插值一致性检查入口。

约束
----
- 输入 float32、batch 维在前；无效槽位不得影响输出（编码/池化/图均按 mask 屏蔽）；
- ``forward`` 完全确定性（世界模型噪声增强仅在显式 ``noise=True`` 时启用）；
- ``rollout`` 为 ``torch.no_grad()`` 包装，供推理/评估使用。
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from net.encoders import (
    EGO_DIM,
    H,
    HISTORY_FRAMES,
    LD_DIM,
    LD_SLOTS,
    NAV_DIM,
    OD_DIM,
    OD_SLOTS,
    SIGNAL_DIM,
    ObsEncoders,
)
from net.moe import MoEBlock
from net.policy import ACTION_HIGH, ACTION_LOW, PolicyHead, ValueHead
from net.spatial import SpatialEncoder, wrap_angle
from net.temporal import TemporalEncoder
from net.world_model import WorldModel

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
    """由刚执行的动作构造下一帧 ego 特征（8 维）。

    ``reserved0/1`` 写入刚执行的 ``(ds,dθ)``——与 p2-contract §8.4 的"上一动作占用保留维"一致。
    注意：``steer/curvature`` 无法从动作唯一恢复，沿用上一帧（rollout 内部自洽即可）。
    """
    speed = ds / dt
    a_long = (speed - prev_speed) / dt
    yaw_rate = dtheta / dt
    a_lat = speed * yaw_rate
    return torch.stack(
        [speed, a_long, a_lat, yaw_rate, ego_prev[:, 4], ego_prev[:, 5], ds, dtheta], dim=-1
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """``(B,S,H)`` 按 ``(B,S)`` 掩码求均值；全无效时返回 0。"""
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


# ---------------------------------------------------------------------- 主模型
class DrivingModel(nn.Module):
    """P2 端到端驾驶模型（编码器/时序/空间/MoE/世界模型/策略头可插拔装配）。"""

    def __init__(
        self,
        hidden: int = H,
        od_slots: int = OD_SLOTS,
        ld_slots: int = LD_SLOTS,
        history_frames: int = HISTORY_FRAMES,
        spatial_layers: int = 2,
        od_knn: int = 4,
        num_experts: int = 8,
        expert_hidden: int = 192,
        router_hidden: int = 64,
        wm_steps: int = 6,
        wm_dt: float = 0.5,
        action_low: tuple[float, float] = ACTION_LOW,
        action_high: tuple[float, float] = ACTION_HIGH,
        log_std_init: float = -1.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.history_frames = int(history_frames)
        self.wm_steps = int(wm_steps)

        self.encoders = ObsEncoders(hidden, od_slots, ld_slots)
        self.temporal = TemporalEncoder(hidden)
        self.spatial = SpatialEncoder(
            hidden,
            layers=spatial_layers,
            od_knn=od_knn,
            num_od=od_slots,
            num_ld=ld_slots,
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(5 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.latent_norm = nn.LayerNorm(hidden)
        self.moe = MoEBlock(hidden, num_experts, expert_hidden, router_hidden)
        self.world_model = WorldModel(hidden, od_slots, ld_slots, steps=wm_steps, dt=wm_dt)
        self.policy = PolicyHead(hidden, action_low, action_high, log_std_init=log_std_init)
        self.value = ValueHead(hidden)

    # ---------------------------------------------------------------- 输入校验
    def _validate_obs(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """校验并归一化观测字典（缺失/形状/类型错误立即报错，避免静默错位）。"""
        if not isinstance(obs, Mapping):
            raise TypeError(f"obs 必须是映射，收到 {type(obs).__name__}")
        expected: dict[str, tuple[int, ...]] = {
            "ego": (EGO_DIM, ),
            "od": (self.od_slots, OD_DIM),
            "od_mask": (self.od_slots, ),
            "ld": (self.ld_slots, LD_DIM),
            "ld_mask": (self.ld_slots, ),
            "nav": (NAV_DIM, ),
            "nav_mask": (1, ),
            "signal": (SIGNAL_DIM, ),
            "signal_mask": (1, ),
            "od_hist": (self.history_frames, self.od_slots, OD_DIM),
            "od_hist_mask": (self.history_frames, self.od_slots),
            "ld_hist": (self.history_frames, self.ld_slots, LD_DIM),
            "ld_hist_mask": (self.history_frames, self.ld_slots),
            "hist_valid": (self.history_frames, ),
        }
        normalized = dict(obs)
        batch = None
        for key, tail in expected.items():
            if key not in normalized:
                raise ValueError(f"obs 缺少必需通道 {key!r}")
            tensor = normalized[key]
            if not torch.is_tensor(tensor):
                raise TypeError(f"obs[{key!r}] 必须是 Tensor，收到 {type(tensor).__name__}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"obs[{key!r}] 必须是 float32，收到 {tensor.dtype}")
            if key in ("nav_mask", "signal_mask") and tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)  # 容忍 (B,) 单槽掩码
                normalized[key] = tensor
            if batch is None:
                batch = int(tensor.shape[0])
            if tuple(tensor.shape[1:]) != tail or int(tensor.shape[0]) != batch:
                raise ValueError(
                    f"obs[{key!r}] 形状应为 (B,{','.join(map(str, tail))})，收到 {tuple(tensor.shape)}"
                )
        if "ego_mask" in normalized:
            mask = normalized["ego_mask"]
            if tuple(mask.shape) != (batch, 1):
                raise ValueError(f"obs['ego_mask'] 形状应为 (B,1)，收到 {tuple(mask.shape)}")
        return normalized

    # ---------------------------------------------------------------- 编码
    def _scene_latent(
        self,
        frame,
        temporal: Tensor,
        nav_token: Tensor,
        signal_token: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """融合时序/上下文 → 空间消息传递 → 池化 → MoE，返回 ``(latent, moe_aux)``。"""
        batch = frame.nodes.shape[0]
        padding = torch.zeros((batch, 1, self.hidden), dtype=frame.nodes.dtype, device=frame.nodes.device)
        temporal_full = torch.cat([padding, temporal], dim=1)
        nodes = frame.nodes + temporal_full * frame.node_mask.unsqueeze(-1)
        nodes = self.spatial(nodes, frame.node_mask, frame.pose)

        od_mean = _masked_mean(nodes[:, 1 : 1 + self.od_slots], frame.od_mask)
        ld_mean = _masked_mean(nodes[:, 1 + self.od_slots :], frame.ld_mask)
        pooled = torch.cat([nodes[:, 0], od_mean, ld_mean, nav_token[:, 0], signal_token[:, 0]], dim=-1)
        latent_pre = self.latent_mlp(pooled)
        moe_out, moe_aux = self.moe(latent_pre)
        latent = self.latent_norm(latent_pre + moe_out)
        return latent, moe_aux

    def encode(self, obs: Mapping[str, Tensor]) -> dict[str, object]:
        """观测 → 场景 latent（不跑 autoregressive rollout；Stage B 可直接复用）。"""
        obs = self._validate_obs(obs)
        frame, nav_token, signal_token = self.encoders.encode_current(obs)
        history_feats, history_mask = self.encoders.encode_history(obs)
        temporal = self.temporal(history_feats, frame_mask=history_mask, valid=obs["hist_valid"])
        latent, moe_aux = self._scene_latent(frame, temporal, nav_token, signal_token)
        return {
            "frame": frame,
            "temporal": temporal,
            "nav_token": nav_token,
            "signal_token": signal_token,
            "latent": latent,
            "moe_aux": moe_aux,
        }

    # ---------------------------------------------------------------- rollout
    def _rollout_traj(
        self,
        frame0,
        temporal0: Tensor,
        nav_token: Tensor,
        signal_token: Tensor,
        latent0: Tensor,
        od_state0: Tensor,
        ld_state0: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """B1 自回归 rollout：策略 ×6 + 世界模型 OD/LD 推演。

        返回 ``(traj_xy (B,6,2), traj_theta (B,6), plan (B,6,2))``：
        轨迹为 **t0 自车系**下 t=0.5..3.0 s 的 6 个动作端点；
        ``plan`` 是 rollout 实际执行的 6 个策略动作（供直接多步世界模型使用）。
        """
        od_mask, ld_mask = frame0.od_mask, frame0.ld_mask
        pose = torch.zeros((latent0.shape[0], 3), dtype=latent0.dtype, device=latent0.device)
        ego_prev = frame0.ego_feat
        temporal = temporal0
        latent = latent0
        xy: list[Tensor] = []
        theta: list[Tensor] = []
        actions: list[Tensor] = []
        for k in range(self.wm_steps):
            action, _ = self.policy(latent)
            actions.append(action)
            ds, dtheta = action[:, 0], action[:, 1]
            dx, dy = arc_step(ds, dtheta)
            pose = compose_pose(pose, dx, dy, dtheta)
            xy.append(pose[:, :2])
            theta.append(pose[:, 2])
            if k >= self.wm_steps - 1:
                break
            od_pred, ld_pred, _ = self.world_model.step(
                latent, action, od_state0, ld_state0, k + 1
            )
            od_body = od_pred_to_features(od_pred, pose, frame0.od_feat, od_mask)
            ld_body = ld_pred_to_features(ld_pred, pose, frame0.ld_feat, ld_mask)
            ego_body = ego_next_features(
                ego_prev, ds, dtheta, dt=self.world_model.dt, prev_speed=ego_prev[:, 0]
            )
            new_frame = self.encoders.encode_frame(ego_body, od_body, od_mask, ld_body, ld_mask)
            history_step = self.encoders.embed_history_frame(od_body, od_mask, ld_body, ld_mask)
            temporal = self.temporal.step(history_step, temporal)
            latent, _ = self._scene_latent(new_frame, temporal, nav_token, signal_token)
            ego_prev = ego_body
        return torch.stack(xy, dim=1), torch.stack(theta, dim=1), torch.stack(actions, dim=1)

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        obs: Mapping[str, Tensor],
        *,
        rollout: bool = True,
        world_model: bool = True,
    ) -> dict[str, Tensor]:
        """完整前向：策略/价值/直接多步世界模型预测/B1 rollout。

        ``rollout=False, world_model=False`` 时走廉价路径：只做编码 + 策略/价值头，
        **不跑** B1 自回归 rollout 与 WM 直接多步；返回的 ``action_mu/action_logstd/
        value/router_logits/expert_weights/latent`` 与完整前向逐位一致，但省略
        ``traj_xy/traj_theta/od_pred/ld_pred``。收集（PPO rollout）不需要多步预测输出。
        """
        encoded = self.encode(obs)
        frame = encoded["frame"]
        latent = encoded["latent"]
        moe_aux = encoded["moe_aux"]

        action_mu, action_logstd = self.policy(latent)
        value = self.value(latent)
        out = {
            "action_mu": action_mu,
            "action_logstd": action_logstd,
            "value": value,
            "router_logits": moe_aux["router_logits"],
            "expert_weights": moe_aux["expert_weights"],
            "latent": latent,
        }
        if not rollout and not world_model:
            return out
        if not rollout:
            raise ValueError("world_model=True 需要 rollout=True（WM 多步以 B1 rollout 的计划为 ego 条件）")
        od_state = WorldModel.od_state_from_features(frame.od_feat, frame.od_mask)
        ld_state = WorldModel.ld_state_from_features(frame.ld_feat, frame.ld_mask)
        traj_xy, traj_theta, plan = self._rollout_traj(
            frame,
            encoded["temporal"],
            encoded["nav_token"],
            encoded["signal_token"],
            latent,
            od_state,
            ld_state,
        )
        out["traj_xy"] = traj_xy
        out["traj_theta"] = traj_theta
        if world_model:
            # 世界模型直接多步预测：以 rollout 的策略计划为 ego 条件（detach 后不回流到策略）
            od_pred, ld_pred, _ = self.world_model(latent, plan.detach(), od_state, ld_state)
            out["od_pred"] = od_pred
            out["ld_pred"] = ld_pred
        return out

    @torch.no_grad()
    def rollout(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """推理入口：与 :meth:`forward` 同输出，但全程 ``no_grad``。"""
        return self.forward(obs)
