import numpy as np
import torch
from net.model import StructuredPlanner
from pipeline.rollout import batch_from_obs, plan_actions, compose_plan_poses

def test_plan_actions_and_compose():
    B, T = 1, 6
    hist = {
        "od": np.zeros((T, 16, 11), np.float32), "od_mask": np.zeros((T, 16), np.float32),
        "ld": np.zeros((T, 16, 5, 9), np.float32), "ld_mask": np.zeros((T, 16, 5), np.float32),
        "nav": np.zeros((T, 11), np.float32), "nav_mask": np.ones((T, 1), np.float32),
        "ego": np.zeros((T, 8), np.float32), "ego_mask": np.ones((T, 1), np.float32),
        "hist_valid": np.ones(T, np.float32),
    }
    m = StructuredPlanner().eval()
    acts = plan_actions(m, batch_from_obs(hist), steps=6)
    assert acts.shape == (6, 4)
    p = compose_plan_poses(acts, np.zeros(3))
    assert p.shape == (30, 3)
    assert np.isfinite(p).all()
