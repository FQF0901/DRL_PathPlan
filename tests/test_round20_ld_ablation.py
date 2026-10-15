"""Round-20 消融开关单测：M1（关 LD-LD 链）/ M2（LD set-attention）——默认关、既有路径逐位不变。

覆盖：
- 默认 ``ld_mode=="baseline"``、spatial flags 默认（include_ld_chain=True/exclude_ld_nodes=False）、
  baseline 不新增参数（``ld_set_attn is None``）；
- ``build_adjacency``：默认有 LD↔LD 链边；``include_ld_chain=False`` 后**零** LD-LD 边，
  且 ego↔LD / ego↔OD 边不变；
- ``exclude_ld_nodes=True``：LD 节点经 spatial 后**输出 == 输入**（独立编码直通）；
- ``set_ld_mode("set_attn")``：惰性构造注意力模块；前向有限；**LD 槽序置换近乎不变**；
- 全无效 LD mask → 注意力输出零向量（无 NaN）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from net.spatial import LdSetAttention, SpatialEncoder
from tests.test_recovery_wiring import _tiny_model

NUM_OD, NUM_LD, H = 16, 16, 16


def _node_mask(batch: int = 2) -> torch.Tensor:
    mask = torch.ones(batch, 1 + NUM_OD + NUM_LD)
    mask[1, 1 + NUM_OD + 5:] = 0.0  # 第二批：部分 LD 无效
    return mask


def test_defaults_and_no_new_params() -> None:
    model = _tiny_model()
    assert model.ld_mode == "baseline"
    assert model.st_gnn.spatial.include_ld_chain is True
    assert model.st_gnn.spatial.exclude_ld_nodes is False
    assert model.ld_set_attn is None
    keys_before = set(model.state_dict().keys())
    model.set_ld_mode("no_chain")
    assert set(model.state_dict().keys()) == keys_before  # M1 不新增参数
    with pytest.raises(ValueError):
        model.set_ld_mode("bogus")


def test_m1_removes_ld_ld_edges_only() -> None:
    spatial = SpatialEncoder(H, layers=1, num_od=NUM_OD, num_ld=NUM_LD)
    mask = _node_mask()
    pose = torch.zeros(2, 1 + NUM_OD + NUM_LD, 3)
    base_adj = spatial.build_adjacency(mask, pose)
    ld_slice = slice(1 + NUM_OD, 1 + NUM_OD + NUM_LD)
    base_ld_edges = int(base_adj[:, ld_slice, ld_slice].sum())
    assert base_ld_edges > 0
    ego_ld_edges = int(base_adj[:, 0, ld_slice].sum())
    ego_od_edges = int(base_adj[:, 0, 1 : 1 + NUM_OD].sum())

    spatial.include_ld_chain = False
    adj = spatial.build_adjacency(mask, pose)
    assert int(adj[:, ld_slice, ld_slice].sum()) == 0  # LD-LD 链完全移除
    assert int(adj[:, 0, ld_slice].sum()) == ego_ld_edges  # ego↔LD 保留
    assert int(adj[:, 0, 1 : 1 + NUM_OD].sum()) == ego_od_edges  # ego↔OD 保留


def test_m2_ld_nodes_bypass_message_passing() -> None:
    torch.manual_seed(0)
    spatial = SpatialEncoder(H, layers=2, num_od=NUM_OD, num_ld=NUM_LD)
    mask = _node_mask()
    pose = torch.zeros(2, 1 + NUM_OD + NUM_LD, 3)
    nodes = torch.randn(2, 1 + NUM_OD + NUM_LD, H)
    base_out = spatial(nodes, mask, pose)
    ld_slice = slice(1 + NUM_OD, 1 + NUM_OD + NUM_LD)
    spatial.exclude_ld_nodes = True
    spatial.include_ld_chain = False
    out = spatial(nodes, mask, pose)
    valid_ld = mask[:, ld_slice] > 0.5  # 无效槽按契约输出 0，只比较有效槽
    assert torch.equal(
        out[:, ld_slice][valid_ld], nodes[:, ld_slice][valid_ld]
    )  # LD 直通（独立编码）
    assert not torch.equal(out[:, : 1 + NUM_OD], base_out[:, : 1 + NUM_OD])  # 其余节点受影响


def test_m2_set_attn_forward_and_permutation_invariance() -> None:
    model = _tiny_model()
    model.set_ld_mode("set_attn")
    assert model.ld_set_attn is not None
    torch.manual_seed(1)
    obs = {
        "ego": torch.randn(1, 1, 8),
        "ego_mask": torch.ones(1, 1),
        "ego_world": torch.randn(1, 1, 3),
        "ego_world_mask": torch.ones(1, 1),
        "od": torch.randn(1, NUM_OD, 9),
        "od_mask": torch.ones(1, NUM_OD),
        "od_presence": torch.ones(1, NUM_OD),
        "ld": torch.randn(1, NUM_LD, 7),
        "ld_mask": torch.ones(1, NUM_LD),
        "nav": torch.randn(1, 1, 11),
        "nav_mask": torch.ones(1, 1),
        "signal": torch.randn(1, 1, 4),
        "signal_mask": torch.ones(1, 1),
        "others": torch.randn(1, 1, 33),
        "others_mask": torch.ones(1, 1),
        "route_world": torch.randn(1, 64, 2),
        "route_world_mask": torch.ones(1, 64),
        "hist_valid": torch.ones(1, 6),
        "ego_hist": torch.randn(1, 6, 1, 8),
        "ego_hist_mask": torch.ones(1, 6, 1),
        "ld_hist": torch.randn(1, 6, NUM_LD, 7),
        "ld_hist_mask": torch.ones(1, 6, NUM_LD),
        "od_hist": torch.randn(1, 6, NUM_OD, 9),
        "od_hist_mask": torch.ones(1, 6, NUM_OD),
        "od_id_hist": torch.zeros(1, 6, NUM_OD, dtype=torch.long),
        "od_presence_hist": torch.ones(1, 6, NUM_OD),
        "others_hist": torch.randn(1, 6, 1, 33),
        "others_hist_mask": torch.ones(1, 6, 1),
    }
    out = model(obs, forward_mode="policy_wm_rollout")
    assert torch.isfinite(out["action_mu"]).all()
    perm = torch.randperm(NUM_LD)
    obs_perm = dict(obs)
    obs_perm["ld"] = obs["ld"][:, perm]
    obs_perm["ld_mask"] = obs["ld_mask"][:, perm]
    obs_perm["ld_hist"] = obs["ld_hist"][:, :, perm]
    obs_perm["ld_hist_mask"] = obs["ld_hist_mask"][:, :, perm]
    out_perm = model(obs_perm, forward_mode="policy_wm_rollout")
    diff = float((out["action_mu"] - out_perm["action_mu"]).abs().max())
    assert diff < 1e-4, f"M2 应对 LD 槽序近似不变（实测 {diff}）"


def test_m2_all_masked_ld_returns_zero() -> None:
    attn = LdSetAttention(H, heads=4, ctx_dim=3 * H)
    tokens = torch.randn(2, NUM_LD, H)
    mask = torch.zeros(2, NUM_LD)
    context = torch.randn(2, 3 * H)
    out = attn(tokens, mask, context)
    assert torch.equal(out, torch.zeros_like(out))
