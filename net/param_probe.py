#!/usr/bin/env python3
"""参数探针：实例化 ``DrivingModel``，打印各子模块参数量并断言总数 ≤ 1.5M。

用法::

    tools/venv-python net/param_probe.py

输出两种配置：
- 默认构造（H=96 / experts 192）——与旧探针口径可比；
- 训练配置（``config/model.yaml``：H=128 / experts 256 / 8 experts）——ckpt 实际规模。
并打印相对 v1 架构（mem-bank 之前）的净变化。末尾用固定种子跑一次小 batch
forward + ``rollout`` 形状冒烟，确保模型可实例化、可运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from net.encoders import DEFAULT_OTHERS_DIM
from net.model import DrivingModel

#: 参数预算（p2-contract §2：实测探针须打印每模块参数，≤1.5M）
PARAM_BUDGET = 1_500_000
#: 冒烟 batch（纯 CPU，固定种子）
SMOKE_BATCH = 2
#: v1（mem-bank 之前）同配置的历史实测总量，用于报告净变化
V1_TOTALS = {96: 666_263, 128: 1_174_875}


def count_parameters(module: torch.nn.Module) -> int:
    """统计一个模块（含子模块）的可训练参数个数。"""
    return sum(param.numel() for param in module.parameters())


def make_dummy_obs(
    batch: int = SMOKE_BATCH, seed: int = 0, others_dim: int = DEFAULT_OTHERS_DIM
) -> dict[str, torch.Tensor]:
    """构造与 env schema v4 / trainer 组装后同形状的假观测（仅形状冒烟）。"""
    generator = torch.Generator().manual_seed(seed)
    frames, od_slots, ld_slots = 6, 16, 16

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    od_mask = (rand(batch, frames, od_slots) > -0.3).float()
    ld_mask = (rand(batch, frames, ld_slots) > -0.3).float()
    presence = od_mask * (rand(batch, frames, od_slots) > -0.3).float()
    hist_valid = torch.zeros(batch, frames)
    hist_valid[:, frames - 3 :] = 1.0  # 预热 3 帧（前 3 帧为补位）
    od_hist = rand(batch, frames, od_slots, 9) * od_mask.unsqueeze(-1)
    ld_hist = rand(batch, frames, ld_slots, 7) * ld_mask.unsqueeze(-1)
    ego_hist = rand(batch, frames, 1, 8)
    others_hist = rand(batch, frames, 1, others_dim)
    od_hist[:, : frames - 3] = od_hist[:, frames - 3 : frames - 2]  # 复制最旧真实帧
    ld_hist[:, : frames - 3] = ld_hist[:, frames - 3 : frames - 2]
    return {
        "ego_hist": ego_hist,
        "ego_hist_mask": torch.ones(batch, frames, 1),
        "od_hist": od_hist,
        "od_hist_mask": od_mask,
        "od_id_hist": (rand(batch, frames, od_slots).abs() * 10).long() + 1,
        "od_presence_hist": presence,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_mask,
        "others_hist": others_hist,
        "others_hist_mask": torch.ones(batch, frames, 1),
        "hist_valid": hist_valid,
        "ego": rand(batch, 1, 8),
        "od": od_hist[:, -1],
        "od_mask": od_mask[:, -1],
        "od_id": (rand(batch, od_slots).abs() * 10).long() + 1,
        "od_presence": presence[:, -1],
        "ld": ld_hist[:, -1],
        "ld_mask": ld_mask[:, -1],
        "others": rand(batch, 1, others_dim),
        "others_mask": torch.ones(batch, 1, 1),
        "nav": rand(batch, 1, 11),
        "nav_mask": torch.ones(batch, 1, 1),
        "signal": rand(batch, 1, 4),
        "signal_mask": torch.ones(batch, 1, 1),
    }


def probe(hidden: int, expert_hidden: int, label: str) -> int:
    """打印一种配置的各模块参数与总量；返回总量（超预算直接 assert 失败）。"""
    torch.manual_seed(0)
    model = DrivingModel(hidden=hidden, expert_hidden=expert_hidden)
    total = count_parameters(model)
    rows: list[tuple[str, int]] = [
        ("encoders", count_parameters(model.encoders)),
        ("mem_encoder", count_parameters(model.mem_encoder)),
        ("plan_head", count_parameters(model.plan_head) - count_parameters(model.plan_head.moe)),
        ("plan_head.moe", count_parameters(model.plan_head.moe)),
        ("st_gnn", count_parameters(model.st_gnn)),
        ("policy", count_parameters(model.policy)),
        ("value", count_parameters(model.value)),
    ]
    accounted = sum(value for _, value in rows)
    print(f"== DrivingModel 参数探针 [{label}]（H={hidden}）==")
    for name, value in rows:
        print(f"  {name:<14} {value:>9,}  ({value / total:6.1%})")
    print(f"  {'buffers 等':<14} {total - accounted:>9,}")
    delta = total - V1_TOTALS.get(hidden, total)
    change = f"{delta:+,}" if hidden in V1_TOTALS else "n/a"
    print(f"  {'总计':<14} {total:>9,}  / 预算 {PARAM_BUDGET:,}  ({total / PARAM_BUDGET:.1%})"
          f"  v1 净变化 {change}")
    assert total <= PARAM_BUDGET, f"参数超预算：{total:,} > {PARAM_BUDGET:,}"
    return total


def main() -> int:
    default_total = probe(96, 192, "默认/旧探针口径（H=96，expert=192 显式）")
    configured_total = probe(128, 76, "训练配置 config/model.yaml（H=128，expert=76）")

    model = DrivingModel().eval()
    obs = make_dummy_obs()
    with torch.no_grad():
        out = model(obs)
        rolled = model.rollout(obs)
    print("== 形状冒烟（默认构造） ==")
    for key in (
        "action_mu",
        "action_logstd",
        "value",
        "traj_xy",
        "plan",
        "od_pred",
        "ld_pred",
        "od_presence_pred",
        "od_entry_pred",
        "router_logits",
        "expert_weights",
        "latent",
    ):
        print(f"  {key:<18} forward={tuple(out[key].shape)}  rollout={tuple(rolled[key].shape)}")
    assert tuple(out["traj_xy"].shape) == (SMOKE_BATCH, 6, 2)
    assert tuple(out["od_presence_pred"].shape) == (SMOKE_BATCH, 6, 16)
    assert tuple(out["od_entry_pred"].shape) == (SMOKE_BATCH, 6, 16)
    print(f"OK: 参数与形状检查通过（默认 {default_total:,} / 训练配置 {configured_total:,}，预算 {PARAM_BUDGET:,}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
