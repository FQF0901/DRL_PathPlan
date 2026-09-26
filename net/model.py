"""``DrivingModel``：mem-bank 输入 → 注意力聚合 → plan head(MoE) → 递归 rollout + ST-GNN。

v2 数据流（mem-bank + 递归 rollout + plan-head MoE）
---------------------------------------------------
1. **输入**：4 个 per-modality 真 mem（``ego_hist/od_hist/ld_hist/others_hist`` + mask、
   ``od_id_hist/od_presence_hist``，见 :mod:`net.mem`）。**net 只读**；真 mem 的更新只由 env 完成。
2. **编码**：:class:`net.mem.MemEncoder` 把 6 帧 mem 编码成"编码 mem"——OD / Ego /
   Others 各做掩码注意力池化，**LD 不做时序**（直接用当前帧，理由见该模块 docstring）。
3. **Plan head**：4 个 mem 聚合 + nav/signal → MLP 融合 → **MoE**（primary 常开 + 8 个
   specific 专家、top-2 软混合）→ 产出**下一时刻 ego 特征**与策略/价值 latent；
   ``router_logits(8)``/``expert_weights(8)`` 原样输出。
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
``ld_pred (B,6,16,4)``（LD 预测只作 rollout 输入/诊断，**未来 LD 监督已移除**）、
``od_presence_pred (B,6,16)``、``od_entry_pred (B,6,16)``（logits）、``traj_theta (B,6)``。

``forward(..., rollout=False, world_model=False)`` 是 PPO cheap path：省略
traj/ST-GNN 多步键，``action_mu/action_logstd/value`` 与完整前向逐位一致。

向后兼容
--------
- 输入契约（:func:`net.mem.mem_from_obs`）= env schema v2 规范键
  （``ego_hist/od_hist/od_id_hist/od_presence_hist/ld_hist/others_hist`` + masks +
  ``hist_valid``；others 维默认 28）；缺失 ``ego_hist/others_hist`` 或
  ``od_id_hist/od_presence_hist`` 时按模块 docstring 的回退规则处理（旧数据集可直接复用）；
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
    LD_SLOTS,
    NAV_DIM,
    OD_SLOTS,
    SIGNAL_DIM,
    ObsEncoders,
)
from net.mem import EncodedMem, MemBank, MemEncoder, mem_from_obs, squeeze_batch_singletons
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
    """``(B,S,H)`` 按 ``(B,S)`` 掩码求均值；全无效时返回 0。"""
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


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
        self.policy = PolicyHead(hidden, action_low, action_high, log_std_init=log_std_init)
        self.value = ValueHead(hidden)

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

    def _context_tokens(self, obs: Mapping[str, Tensor], batch: int) -> tuple[Tensor, Tensor]:
        """nav / signal（可选；v2 的规范上下文在 ``others`` mem 里）→ ``(B,H)`` token。

        缺失时用 0 向量 + 无效掩码（编码器输出 0），不报错——避免与 env 的通道裁剪策略
        耦合；``norm/attention`` 的掩码路径已保证无效 token 不污染输出。
        """
        device = obs["hist_valid"].device
        nav = obs.get("nav")
        if torch.is_tensor(nav):
            nav = squeeze_batch_singletons(nav.float(), 2, "nav")
            if tuple(nav.shape) != (batch, NAV_DIM):
                raise ValueError(f"obs['nav'] 形状应为 (B,{NAV_DIM})（或 (B,1,{NAV_DIM})），收到 {tuple(nav.shape)}")
            nav_mask = obs.get("nav_mask")
            if nav_mask is None:
                nav_mask = torch.ones((batch, 1), dtype=nav.dtype, device=nav.device)
            else:
                nav_mask = nav_mask.float()
                if nav_mask.ndim == 1:
                    nav_mask = nav_mask.unsqueeze(-1)
                nav_mask = squeeze_batch_singletons(nav_mask, 2, "nav_mask")
                if tuple(nav_mask.shape) != (batch, 1):
                    raise ValueError(f"obs['nav_mask'] 形状应为 (B,1)，收到 {tuple(nav_mask.shape)}")
            nav_token = self.encoders.embed_nav(nav.unsqueeze(1), nav_mask.reshape(-1, 1))
        else:
            nav_token = torch.zeros((batch, 1, self.hidden), dtype=torch.float32, device=device)
        signal = obs.get("signal")
        if torch.is_tensor(signal):
            signal = squeeze_batch_singletons(signal.float(), 2, "signal")
            if tuple(signal.shape) != (batch, SIGNAL_DIM):
                raise ValueError(
                    f"obs['signal'] 形状应为 (B,{SIGNAL_DIM})（或 (B,1,{SIGNAL_DIM})），收到 {tuple(signal.shape)}"
                )
            signal_mask = obs.get("signal_mask")
            if signal_mask is None:
                signal_mask = torch.ones((batch, 1), dtype=signal.dtype, device=signal.device)
            else:
                signal_mask = signal_mask.float()
                if signal_mask.ndim == 1:
                    signal_mask = signal_mask.unsqueeze(-1)
                signal_mask = squeeze_batch_singletons(signal_mask, 2, "signal_mask")
                if tuple(signal_mask.shape) != (batch, 1):
                    raise ValueError(f"obs['signal_mask'] 形状应为 (B,1)，收到 {tuple(signal_mask.shape)}")
            signal_token = self.encoders.embed_signal(signal.unsqueeze(1), signal_mask.reshape(-1, 1))
        else:
            signal_token = torch.zeros((batch, 1, self.hidden), dtype=torch.float32, device=device)
        return nav_token[:, 0], signal_token[:, 0]

    # ---------------------------------------------------------------- 编码/规划
    def _plan(
        self, encoded: EncodedMem, nav_token: Tensor, signal_token: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        od_pool = _masked_mean(encoded.od_ctx, encoded.od_live)
        ld_pool = _masked_mean(encoded.ld_ctx, encoded.ld_live)
        return self.plan_head(
            encoded.ego_ctx, od_pool, ld_pool, encoded.others_ctx, nav_token, signal_token
        )

    def encode(self, obs: Mapping[str, Tensor]) -> dict[str, object]:
        """观测 → mem/编码 mem/plan-head latent（不跑递归 rollout）。"""
        mem = self._mem_from_obs(obs)
        nav_token, signal_token = self._context_tokens(obs, mem.batch)
        encoded = self.mem_encoder.encode(self.encoders, mem)
        latent, ego_next, moe_aux = self._plan(encoded, nav_token, signal_token)
        return {
            "mem": mem,
            "encoded": encoded,
            "frame": encoded.frame,
            "latent": latent,
            "ego_next": ego_next,
            "moe_aux": moe_aux,
            "nav_token": nav_token,
            "signal_token": signal_token,
        }

    def plan_step(
        self, encoded_mem: EncodedMem, nav_token: Tensor, signal_token: Tensor
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
        """
        mem: MemBank = encoded["mem"].clone()  # 拷贝隔离：真 mem 绝不写回
        enc: EncodedMem = encoded["encoded"]
        latent: Tensor = encoded["latent"]
        ego_next: Tensor = encoded["ego_next"]
        nav_token: Tensor = encoded["nav_token"]
        signal_token: Tensor = encoded["signal_token"]
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

            # (e) 下一轮 plan head（最后一步之后不需要）
            if k < self.rollout_steps:
                enc = self.mem_encoder.encode(self.encoders, mem)
                latent, ego_next, _ = self._plan(enc, nav_token, signal_token)
                action, _ = self.policy(latent)

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

        ``rollout=False, world_model=False`` 时走廉价路径：只做编码 + plan head +
        策略/价值头，不跑递归 rollout，返回的 ``action_mu/action_logstd/value/
        router_logits/expert_weights/latent`` 与完整前向逐位一致（PPO 收集用）。

        ``world_model=False``（rollout=True）时仍执行 rollout（轨迹/动作需要），
        但**不返回** ``od_pred/ld_pred/od_presence_pred/od_entry_pred``。

        ``wm_detach``：仅为兼容旧签名保留，**no-op**（v2 的合成帧 detach 固定生效）。
        """
        encoded = self.encode(obs)
        latent: Tensor = encoded["latent"]
        moe_aux: dict[str, Tensor] = encoded["moe_aux"]
        action_mu, action_logstd = self.policy(latent)
        out: dict[str, Tensor] = {
            "action_mu": action_mu,
            "action_logstd": action_logstd,
            "value": self.value(latent),
            "router_logits": moe_aux["router_logits"],
            "expert_weights": moe_aux["expert_weights"],
            "latent": latent,
            "ego_next": encoded["ego_next"],
        }
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
