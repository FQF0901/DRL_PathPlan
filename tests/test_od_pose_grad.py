"""回归测试：空 OD 槽位（cos=sin=0）不得让 ``atan2(0,0)`` 的 NaN 反向污染 ``traj_xy`` 梯度。

背景：trainer 线发现 `net/encoders.py::od_pose` 的 `atan2(0,0)` 在反向为 NaN，
且 `mask` 乘法是 ``0 × NaN = NaN``，于是任何走 `traj_xy` 的损失都会变成 NaN。
修复：退化槽位改用 (cos=1, sin=0)，前向与旧实现一致（仍被 mask 乘为 0），反向有限。
"""

from __future__ import annotations

import torch

from net.encoders import OD_DIM, ObsEncoders
from net.model import DrivingModel
from tests.test_net_shapes import make_obs


def test_od_pose_forward_parity_for_valid_slots() -> None:
    """有效槽位（单位朝向向量）的位姿与旧实现一致（角度回卷后比较，确定性种子）。"""
    generator = torch.Generator().manual_seed(0)
    feat = torch.randn(2, 16, OD_DIM, generator=generator)
    theta = torch.randn(2, 16, generator=generator)
    feat[..., 4] = torch.cos(theta)
    feat[..., 5] = torch.sin(theta)
    mask = torch.ones(2, 16)
    pose = ObsEncoders.od_pose(feat, mask)
    wrapped = torch.atan2(torch.sin(theta), torch.cos(theta))  # atan2 的像域：[-π, π]
    reference = torch.cat([feat[..., 0:2], wrapped.unsqueeze(-1)], dim=-1) * mask.unsqueeze(-1)
    assert torch.allclose(pose, reference, atol=1e-6)


def test_empty_od_slots_gradients_finite() -> None:
    """一半 OD 槽位为空（mask=0 且特征全 0）时，traj_xy 损失的反向必须全有限。"""
    obs = make_obs(batch=2, seed=1, valid_frames=6)
    obs["od_mask"][:, 8:] = 0.0
    obs["od"][:, 8:, :] = 0.0
    obs["od_hist_mask"][:, :, 8:] = 0.0
    obs["od_hist"][:, :, 8:, :] = 0.0

    torch.manual_seed(0)
    model = DrivingModel()
    out = model(obs)
    assert torch.isfinite(out["traj_xy"]).all()
    loss = out["traj_xy"].pow(2).mean() + out["od_pred"].pow(2).mean()
    loss.backward()

    bad = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    assert not bad, f"non-finite gradients in: {bad[:5]}"
