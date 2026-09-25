"""收集廉价路径等价性：``forward(..., rollout=False, world_model=False)``。

PPO ``collect_rollout`` 只需策略/价值/路由头，不跑 B1 rollout 与 WM 直接多步。
本测试证明廉价路径与完整前向在 eval/固定种子下逐位一致（``action_mu``/
``action_logstd``/``value``），且不产出多步预测键。
"""

from __future__ import annotations

import torch

from net.model import DrivingModel
from tests.test_net_shapes import make_obs


def test_cheap_path_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = DrivingModel().eval()
    obs = make_obs(batch=3, seed=11)

    torch.manual_seed(0)
    full = model(obs)
    torch.manual_seed(0)
    cheap = model(obs, rollout=False, world_model=False)

    for key in ("action_mu", "action_logstd", "value"):
        assert torch.allclose(cheap[key], full[key], atol=1e-6), f"{key} 与完整前向不一致"
    # 多步预测键（BC 轨迹目标 / WM 辅助损失用）只在完整前向产出
    for key in ("traj_xy", "traj_theta", "od_pred", "ld_pred"):
        assert key in full
        assert key not in cheap
