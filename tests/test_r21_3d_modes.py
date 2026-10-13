"""R21-3D 契约测试：``semantic_no_chain`` 单变量模式 + rollout semantic 三种口径。

覆盖：
1. ``semantic_no_chain``：接受、仅关 LD-LD 链（MP 不动）、semantic 系判定、惰性构造共用；
2. ``set_rollout_semantic``：合法值/默认 ``full``/非法值报错；
3. 三种口径的消费行为（spy 计数 + 注入特征）：
   - ``full``：t0 + rollout 5 步各重建一次（semantic_wiring 调用 6 次）；
   - ``t0_only``：仅 t0 一次（后续复用 t0 context，不重复注入动态语义）；
   - ``static_role_only``：6 次重建，但 rollout 步的 lane_semantic **动态特征被置零**
     （lat/heading/margins/distance），静态特征（curv/valid/conf/change）保留；
4. ``full`` 与旧行为逐位一致（默认不变）；``semantic_no_chain`` 在**已训练**转移下与
   ``semantic`` 输出不同（链确实生效）；tiny 随机模型下两者恒等（零初始化转移）。
"""

from __future__ import annotations

import pytest
import torch

from net.model import DrivingModel
from tests.test_r21_3a_semantic_wiring import _semantic_model, _semantic_obs


def test_semantic_no_chain_flags_and_acceptance() -> None:
    model = _semantic_model()
    assert model.stgnn_flags() == (True, False)
    model.set_ld_mode("semantic_no_chain")
    assert model.stgnn_flags() == (False, False)
    assert model._is_semantic() is True
    assert model.semantic_wiring is not None
    # 回到 semantic 恢复链
    model.set_ld_mode("semantic")
    assert model.stgnn_flags() == (True, False)
    with pytest.raises(ValueError, match="未知 ld_mode"):
        model.set_ld_mode("semantic_chain_typo")


def test_set_rollout_semantic_values() -> None:
    model = _semantic_model()
    assert model.rollout_semantic_mode == "full"
    for mode in ("full", "t0_only", "static_role_only"):
        assert model.set_rollout_semantic(mode).rollout_semantic_mode == mode
    with pytest.raises(ValueError, match="未知 rollout_semantic"):
        model.set_rollout_semantic("bad_mode")


def _spy_wiring_calls(model: DrivingModel) -> list[dict]:
    """记录 semantic_wiring 前向收到的 lane_semantic（返回 list，元素为 (B,12) 张量）。"""
    calls: list[dict] = []
    original = model.semantic_wiring.forward

    def spy(z_ld, ld_live, **kwargs):
        calls.append({"lane_semantic": kwargs["lane_semantic"].detach().clone()})
        return original(z_ld, ld_live, **kwargs)

    model.semantic_wiring.forward = spy  # type: ignore[method-assign]
    return calls


def test_rollout_modes_consumption_counts_and_static_zeroing() -> None:
    obs = _semantic_obs(batch=2, seed=11)
    obs["lane_semantic"][..., 0] = 0.7  # lat 动态非零
    obs["lane_semantic"][..., 1] = 0.1  # heading 动态非零
    obs["lane_semantic"][..., 2] = 0.05  # curv 静态非零
    dynamic = DrivingModel.SEMANTIC_DYNAMIC_FEATURES
    static = tuple(i for i in range(12) if i not in dynamic)

    # full：t0 + rollout 5 步 = 6 次；动态特征保持
    model = _semantic_model()
    calls = _spy_wiring_calls(model)
    model(obs, rollout=True, world_model=False)
    assert len(calls) == 6, f"full 口径 semantic_wiring 调用 {len(calls)} != 6"
    assert all(float(c["lane_semantic"][..., 0].abs().max()) > 0 for c in calls)

    # t0_only：仅 1 次（rollout 复用 t0 context）
    model = _semantic_model()
    model.set_rollout_semantic("t0_only")
    calls = _spy_wiring_calls(model)
    model(obs, rollout=True, world_model=False)
    assert len(calls) == 1, f"t0_only 口径调用 {len(calls)} != 1"

    # static_role_only：6 次，但第 2..6 次动态特征置零、静态保留
    model = _semantic_model()
    model.set_rollout_semantic("static_role_only")
    calls = _spy_wiring_calls(model)
    model(obs, rollout=True, world_model=False)
    assert len(calls) == 6
    assert float(calls[0]["lane_semantic"][..., 0].abs().max()) > 0, "t0 应为全量语义"
    for call in calls[1:]:
        feats = call["lane_semantic"]
        assert float(feats[..., list(dynamic)].abs().max()) == 0.0, "rollout 动态特征必须置零"
        assert float(feats[..., list(static)].abs().max()) > 0, "rollout 静态特征必须保留"


def test_rollout_full_default_equals_explicit_full() -> None:
    model = _semantic_model()
    obs = _semantic_obs(batch=1, seed=12)
    base = model(obs, rollout=True, world_model=False)["plan"]
    model.set_rollout_semantic("full")
    again = model(obs, rollout=True, world_model=False)["plan"]
    assert torch.equal(base, again)


def test_semantic_chain_effect_with_scattered_transitions() -> None:
    """tiny 模型转移零初始化 ⇒ 链无效果；打散 ld_transition 后 no_chain 必须改变输出。"""
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=13)
    base = model(obs, rollout=True, world_model=False)["plan"]
    model.set_ld_mode("semantic_no_chain")
    same = model(obs, rollout=True, world_model=False)["plan"]
    assert torch.equal(base, same), "零初始化转移下链应恒等（结构性事实）"
    # 打散转移权重（诊断性局部扰动，不改生产代码）→ 链效应可见
    torch.nn.init.normal_(model.st_gnn.ld_transition[-1].weight, std=0.1)
    torch.nn.init.normal_(model.st_gnn.ld_transition[-1].bias, std=0.05)
    model.set_ld_mode("semantic")
    with_chain = model(obs, rollout=True, world_model=False)["plan"]
    model.set_ld_mode("semantic_no_chain")
    without_chain = model(obs, rollout=True, world_model=False)["plan"]
    assert not torch.allclose(with_chain, without_chain, atol=1e-7), "链开关必须改变 rollout 输出"


def test_plan_step_semantic_requires_context_for_no_chain_too() -> None:
    model = _semantic_model()
    model.set_ld_mode("semantic_no_chain")
    with pytest.raises(RuntimeError, match="禁止静默回退"):
        model.plan_step(
            torch.randn(1, 16), torch.randn(1, 16, 16), torch.randn(1, 16, 16),
            torch.ones(1, 16), torch.ones(1, 16), torch.randn(1, 16),
            torch.randn(1, 16), torch.randn(1, 16),
        )
