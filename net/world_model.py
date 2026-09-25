"""世界模型：ego 条件化的 OD/LD 动力学（GRUCell 潜动态 + 解码器 + 直接多步损失）。

接口约定
--------
- **预测目标**（都是 t0 帧/自车系下的量，与 ``traj_xy`` 同一参考系）：

  - ``OD_PRED_DIM = 5``：``[dx, dy, vx, vy, heading_rel]``（位置、相对速度、朝向）；
  - ``LD_PRED_DIM = 4``：``[dx, dy, heading_rel, curvature]``。

  OD 的 ``L/W/type_id``、LD 的 ``speed_limit/线型`` 是静态属性，沿用上一帧，
  不进入预测目标（由调用方在重建特征时携带）。
- **ego 条件化**：每步输入计划动作 ``(ds, dθ)`` 的嵌入 + 步索引嵌入，
  经 GRUCell 更新潜状态 ``z_k``；解码器同时看到"上一状态 → 本步"的匀速/静止先验，
  输出**残差**。解码器输出层零初始化 ⇒ 未训练时世界模型严格退化为
  匀速（OD）/ 静止（LD）基线，可直接作为 Stage B 的对照下界。
- **直接多步损失**：:func:`direct_multi_step_loss` 对 K 步预测同时计算掩码损失
  （不是逐步 teacher-forcing 的滚动损失），配合 ``od_mask/ld_mask/valid``；
- **噪声增强**：训练时调用 ``noise=True``（或在外部用 :meth:`WorldModel.perturb`），
  对输入状态加高斯噪声；默认关闭，保证 ``forward`` 确定性。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from net.encoders import H, LD_SLOTS, OD_SLOTS

OD_PRED_DIM = 5
LD_PRED_DIM = 4
OD_PRED_NAMES: tuple[str, ...] = ("dx", "dy", "vx", "vy", "heading_rel")
LD_PRED_NAMES: tuple[str, ...] = ("dx", "dy", "heading_rel", "curvature")


def _zero_init_last(layer: nn.Linear) -> None:
    """输出层零初始化：让残差分支初始严格为 0。"""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class WorldModel(nn.Module):
    """ego 条件化的 OD/LD 多步动力学模型。"""

    def __init__(
        self,
        hidden: int = H,
        od_slots: int = OD_SLOTS,
        ld_slots: int = LD_SLOTS,
        steps: int = 6,
        dt: float = 0.5,
        *,
        noise_std: float = 0.05,
        noise_p: float = 0.3,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.steps = int(steps)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.noise_p = float(noise_p)

        self.action_embed = nn.Linear(2, hidden)
        self.step_embed = nn.Embedding(self.steps, hidden)
        self.dyn = nn.GRUCell(hidden, hidden)
        self.od_head = nn.Sequential(
            nn.Linear(hidden + OD_PRED_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, OD_PRED_DIM),
        )
        self.ld_head = nn.Sequential(
            nn.Linear(hidden + LD_PRED_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, LD_PRED_DIM),
        )
        _zero_init_last(self.od_head[-1])
        _zero_init_last(self.ld_head[-1])

    # ---------------------------------------------------------------- 状态/目标
    @staticmethod
    def od_state_from_features(od_feat: Tensor, od_mask: Tensor | None = None) -> Tensor:
        """OD 原始 9 维 -> 预测空间 5 维 ``[dx,dy,vx,vy,heading]``；掩码外清零。"""
        heading = torch.atan2(od_feat[..., 5], od_feat[..., 4]).unsqueeze(-1)
        state = torch.cat([od_feat[..., 0:4], heading], dim=-1)
        if od_mask is not None:
            state = state * od_mask.unsqueeze(-1)
        return state

    @staticmethod
    def ld_state_from_features(ld_feat: Tensor, ld_mask: Tensor | None = None) -> Tensor:
        """LD 原始 7 维 -> 预测空间 4 维 ``[dx,dy,heading,curvature]``；掩码外清零。"""
        state = ld_feat[..., 0:4]
        if ld_mask is not None:
            state = state * ld_mask.unsqueeze(-1)
        return state

    @staticmethod
    def targets_from_features(od_fut: Tensor, ld_fut: Tensor) -> tuple[Tensor, Tensor]:
        """未来帧原始特征 ``(B,K,16,9)/(B,K,16,7)`` -> 监督目标（t0 帧，不掩码）。"""
        return (
            WorldModel.od_state_from_features(od_fut),
            WorldModel.ld_state_from_features(ld_fut),
        )

    # ---------------------------------------------------------------- 噪声增强
    def perturb(self, x: Tensor) -> Tensor:
        """逐元素噪声增强（p=``noise_p`` 命中，幅度 ``noise_std``）。"""
        if self.noise_p <= 0.0 or self.noise_std <= 0.0:
            return x
        hit = (torch.rand_like(x) < self.noise_p).to(x.dtype)
        return x + torch.randn_like(x) * (self.noise_std * hit)

    # ---------------------------------------------------------------- 单步推演
    def step(
        self,
        latent: Tensor,
        action: Tensor,
        od_state: Tensor,
        ld_state: Tensor,
        step_index: int,
        *,
        noise: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """从 t0 锚定状态预测第 ``step_index``（1-based）步。

        Args:
            latent: ``(B,H)`` 当前潜状态 ``z_{k-1}``。
            action: ``(B,2)`` 该步执行的动作 ``(ds, dθ)``。
            od_state/ld_state: t0 帧下的初始状态（不随 step 递增，直接多步锚定）。
            step_index: 1..steps。
        Returns:
            ``(od_pred (B,16,5), ld_pred (B,16,4), latent_next (B,H))``。
        """
        if not 1 <= int(step_index) <= self.steps:
            raise ValueError(f"step_index 必须在 1..{self.steps}，收到 {step_index}")
        batch = latent.shape[0]
        index = torch.full((batch,), int(step_index) - 1, dtype=torch.long, device=latent.device)
        condition = self.action_embed(action) + self.step_embed(index)
        latent_next = self.dyn(condition, latent)

        if noise:
            od_state = self.perturb(od_state)
            ld_state = self.perturb(ld_state)

        # 先验：OD 匀速外推（位置 + k·dt·v），LD 在 t0 帧静止
        offset = step_index * self.dt
        prior_od = torch.stack(
            [
                od_state[..., 0] + od_state[..., 2] * offset,
                od_state[..., 1] + od_state[..., 3] * offset,
                od_state[..., 2],
                od_state[..., 3],
                od_state[..., 4],
            ],
            dim=-1,
        )
        prior_ld = ld_state

        # 残差解码（head 输入 = [z_k 广播到每个槽位, 先验]）
        z_od = latent_next.unsqueeze(1).expand(-1, self.od_slots, -1)
        z_ld = latent_next.unsqueeze(1).expand(-1, self.ld_slots, -1)
        od_pred = prior_od + self.od_head(torch.cat([z_od, prior_od], dim=-1))
        ld_pred = prior_ld + self.ld_head(torch.cat([z_ld, prior_ld], dim=-1))
        return od_pred, ld_pred, latent_next

    def forward(
        self,
        latent: Tensor,
        plan: Tensor,
        od_state: Tensor,
        ld_state: Tensor,
        *,
        noise: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """直接多步预测。

        Args:
            latent: ``(B,H)`` 当前潜状态。
            plan: ``(B,K,2)`` 计划动作序列（ego 条件）。
            od_state/ld_state: ``(B,16,5)/(B,16,4)`` t0 锚定状态。
        Returns:
            ``(od_pred (B,K,16,5), ld_pred (B,K,16,4), latent_seq (B,K,H))``。
        """
        if plan.ndim != 3 or plan.shape[1] != self.steps:
            raise ValueError(f"plan 形状应为 (B,{self.steps},2)，收到 {tuple(plan.shape)}")
        od_steps: list[Tensor] = []
        ld_steps: list[Tensor] = []
        latent_steps: list[Tensor] = []
        current = latent
        for k in range(1, self.steps + 1):
            od_pred, ld_pred, current = self.step(
                current, plan[:, k - 1], od_state, ld_state, k, noise=noise
            )
            od_steps.append(od_pred)
            ld_steps.append(ld_pred)
            latent_steps.append(current)
        return (
            torch.stack(od_steps, dim=1),
            torch.stack(ld_steps, dim=1),
            torch.stack(latent_steps, dim=1),
        )


def direct_multi_step_loss(
    od_pred: Tensor,
    ld_pred: Tensor,
    od_target: Tensor,
    ld_target: Tensor,
    od_mask: Tensor,
    ld_mask: Tensor,
    valid: Tensor | None = None,
    *,
    beta: float = 1.0,
) -> Tensor:
    """直接多步掩码损失（Huber + 角度 ``1-cosΔ``）。

    Args:
        od_pred/od_target: ``(B,K,16,5)``；``ld_pred/ld_target``: ``(B,K,16,4)``。
        od_mask/ld_mask: ``(B,K,16)`` 未来帧槽位掩码（1=计入损失）。
        valid: ``(B,K)`` 该未来步整体是否有效（warmup/截断），默认全 1。
        beta: SmoothL1 的 beta。

    Returns:
       标量损失（对所有有效槽位求均值；无有效项时返回 0，仍与计算图相连）。
    """

    def _smooth(err: Tensor) -> Tensor:
        return F.smooth_l1_loss(err, torch.zeros_like(err), beta=beta, reduction="none")

    od_linear = _smooth(od_pred[..., :4] - od_target[..., :4]).sum(dim=-1) / 4.0
    od_angle = 1.0 - torch.cos(od_pred[..., 4] - od_target[..., 4])
    od_err = od_linear + od_angle

    ld_linear = _smooth(ld_pred[..., :2] - ld_target[..., :2]).sum(dim=-1) / 2.0
    ld_linear = ld_linear + _smooth(ld_pred[..., 3:4] - ld_target[..., 3:4]).squeeze(-1)
    ld_angle = 1.0 - torch.cos(ld_pred[..., 2] - ld_target[..., 2])
    ld_err = ld_linear + ld_angle

    if valid is None:
        valid = torch.ones(od_mask.shape[:2], dtype=od_mask.dtype, device=od_mask.device)
    weight_od = od_mask * valid.unsqueeze(-1)
    weight_ld = ld_mask * valid.unsqueeze(-1)
    total = (od_err * weight_od).sum() + (ld_err * weight_ld).sum()
    count = weight_od.sum() + weight_ld.sum()
    return total / count.clamp(min=1.0)
