import math
import torch
from pipeline.distributions import TruncatedNormal, truncated_normal_nll


def test_samples_inside_support_and_logprob_finite():
    torch.manual_seed(0)
    loc = torch.tensor([[0.2, 0.5, 0.8, 0.1]])
    scale = torch.tensor([[0.2, 0.1, 0.3, 0.05]])
    d = TruncatedNormal(loc, scale)
    x = d.sample((2000,))
    assert x.shape == (2000, 1, 4)
    assert (x >= 0).all() and (x <= 1).all()
    lp = d.log_prob(x)
    assert torch.isfinite(lp).all()


def test_logprob_integrates_to_one():
    # 1-D quadrature; the support is bounded and the density is smooth.
    loc = torch.tensor([0.35])
    scale = torch.tensor([0.2])
    d = TruncatedNormal(loc, scale)
    grid = torch.linspace(1e-5, 1 - 1e-5, 20001)
    p = d.log_prob(grid).exp()
    integral = torch.trapz(p, grid)
    assert abs(float(integral) - 1.0) < 2e-3, float(integral)


def test_nll_flags_outside_without_rewriting():
    loc = torch.tensor([[0.5, 0.5]])
    log_scale = torch.tensor([[-1.0, -1.0]])
    target = torch.tensor([[0.5, 1.2]])
    nll, outside = truncated_normal_nll(target, loc, log_scale)
    assert nll.shape == (1,)
    assert outside.tolist() == [[False, True]]
    assert torch.isfinite(nll).all()
