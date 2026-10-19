import torch
from net.model import StructuredPlanner, count_parameters

def test_model_forward_and_budget():
    m = StructuredPlanner()
    n = count_parameters(m)
    assert 800_000 <= n <= 1_200_000, n
    B, T = 2, 6
    b = dict(
        od=torch.randn(B, T, 16, 11), od_mask=(torch.rand(B, T, 16) > .5).float(),
        ld=torch.randn(B, T, 16, 5, 9), ld_mask=(torch.rand(B, T, 16, 5) > .3).float(),
        nav=torch.randn(B, T, 11), nav_mask=torch.ones(B, T, 1),
        ego=torch.randn(B, T, 8), ego_mask=torch.ones(B, T, 1), hist_valid=torch.ones(B, T))
    out = m(b)
    assert out["action_mu"].shape == (B, 4)
    assert out["action_logstd"].shape == (B, 4)
    assert out["value"].shape == (B, 1)
    assert out["risk"].shape == (B, 1)
    assert torch.isfinite(out["action_mu"]).all()
