#!/usr/bin/env python3
"""参数探针：实例化 ``DrivingModel``，打印各子模块参数量并断言总数 ≤ 1.5M。

用法::

    tools/venv-python net/param_probe.py

输出各模块参数量（含占比）与总计；预算见 p2-contract §2（≤1.5M）。
末尾用固定种子跑一次小 batch 前向 + ``rollout`` 形状冒烟，确保模型可实例化、可运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from net.model import DrivingModel

#: 参数预算（p2-contract §2：实测探针须打印每模块参数，≤1.5M）
PARAM_BUDGET = 1_500_000
#: 冒烟 batch（纯 CPU，固定种子）
SMOKE_BATCH = 2


def count_parameters(module: torch.nn.Module) -> int:
    """统计一个模块（含子模块）的可训练参数个数。"""
    return sum(param.numel() for param in module.parameters())


def make_dummy_obs(batch: int = SMOKE_BATCH, seed: int = 0) -> dict[str, torch.Tensor]:
    """构造与 ``env.obs.builder`` 同形状的假观测（仅用于形状冒烟）。"""
    generator = torch.Generator().manual_seed(seed)
    frames, od_slots, ld_slots = 6, 16, 16

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    od_mask = (rand(batch, od_slots) > -0.3).float()
    ld_mask = (rand(batch, ld_slots) > -0.3).float()
    od_hist_mask = od_mask.unsqueeze(1).expand(batch, frames, od_slots).clone()
    ld_hist_mask = ld_mask.unsqueeze(1).expand(batch, frames, ld_slots).clone()
    hist_valid = torch.zeros(batch, frames)
    hist_valid[:, frames - 3 :] = 1.0  # 预热 3 帧（前 3 帧为补位）
    od_hist = rand(batch, frames, od_slots, 9)
    ld_hist = rand(batch, frames, ld_slots, 7)
    od_hist[:, : frames - 3] = od_hist[:, frames - 3 : frames - 2]  # 复制最旧真实帧
    ld_hist[:, : frames - 3] = ld_hist[:, frames - 3 : frames - 2]
    return {
        "ego": rand(batch, 8),
        "od": rand(batch, od_slots, 9) * od_mask.unsqueeze(-1),
        "od_mask": od_mask,
        "ld": rand(batch, ld_slots, 7) * ld_mask.unsqueeze(-1),
        "ld_mask": ld_mask,
        "nav": rand(batch, 11),
        "nav_mask": torch.ones(batch, 1),
        "signal": rand(batch, 4),
        "signal_mask": torch.ones(batch, 1),
        "od_hist": od_hist,
        "od_hist_mask": od_hist_mask,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_hist_mask,
        "hist_valid": hist_valid,
    }


def main() -> int:
    torch.manual_seed(0)
    model = DrivingModel()
    total = count_parameters(model)

    rows: list[tuple[str, int]] = [
        ("encoders", count_parameters(model.encoders)),
        ("temporal", count_parameters(model.temporal)),
        ("spatial", count_parameters(model.spatial)),
        ("latent_mlp", count_parameters(model.latent_mlp)),
        ("moe", count_parameters(model.moe)),
        ("world_model", count_parameters(model.world_model)),
        ("policy", count_parameters(model.policy)),
        ("value", count_parameters(model.value)),
    ]
    head_params = sum(v for _, v in rows)
    print("== DrivingModel 参数探针（H=96, 8 experts 96->192->96）==")
    for name, value in rows:
        print(f"  {name:<12} {value:>9,}  ({value / total:6.1%})")
    print(f"  {'modules 小计':<12} {head_params:>9,}")
    print(f"  {'buffers 等':<12} {total - head_params:>9,}")
    print(f"  {'总计':<12} {total:>9,}  / 预算 {PARAM_BUDGET:,}  ({total / PARAM_BUDGET:.1%})")

    assert total <= PARAM_BUDGET, f"参数超预算：{total:,} > {PARAM_BUDGET:,}"

    model.eval()
    obs = make_dummy_obs()
    with torch.no_grad():
        out = model(obs)
        rolled = model.rollout(obs)
    print("== 形状冒烟 ==")
    for key in ("action_mu", "action_logstd", "value", "traj_xy", "od_pred", "ld_pred", "router_logits", "latent"):
        print(f"  {key:<15} forward={tuple(out[key].shape)}  rollout={tuple(rolled[key].shape)}")
    assert tuple(out["traj_xy"].shape) == (SMOKE_BATCH, 6, 2)
    print("OK: 参数与形状检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
