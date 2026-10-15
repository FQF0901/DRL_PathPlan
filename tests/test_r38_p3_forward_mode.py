"""R38-P3-H：``forward_mode`` 显式枚举 ↔ 旧 ``rollout=/world_model=`` 布尔组合**逐位对照**。

出处：GPT §七-H + ``docs/repo_handbook.md`` 附录 D（可达性审计 H 组 + R38 弃用清单）
H 组（"旧布尔组合 → 枚举的映射表 + 每组合一次前向逐位对照"）。断言面（CPU、无 metadrive）：

1. **映射表**：``FORWARD_MODE_FLAGS`` = 声明口径（``policy_single_step`` (F,F) /
   ``policy_wm_rollout`` (T,F) / ``world_model_teacher_forcing`` (T,T) / ``diagnostics`` (T,T)）；
2. **逐位对照**（plain + structured 两种装配）：旧布尔组合前向 ≡ 对应枚举前向
   （键集一致 + 逐键 ``max|Δ|=0``）；默认 ``forward(obs)`` ≡ ``forward_mode=
   "world_model_teacher_forcing"``（旧默认 (True, True) 逐位保留）；
3. **``diagnostics`` = 全量前向旗标**（与 WMTF 逐位相同；独立枚举名隔离诊断调用方）；
4. **混用禁止**：``forward_mode`` 与 ``rollout=/world_model=`` 同时给 ⇒ ``ValueError``；
   非法枚举值 ⇒ ``ValueError``（列出允许值）；
5. **旧签名透传 = 同结果**：``wm_detach=True/False`` 经 ``**legacy_signature`` 吞掉
   （恒 no-op，逐位同结果）；其他未知关键字 ⇒ ``TypeError``。
"""

from __future__ import annotations

import torch
import pytest

from net.model import FORWARD_MODE_FLAGS, FORWARD_MODES, DrivingModel, resolve_forward_mode
from tests.test_net_shapes import make_obs

BATCH = 2

#: 旧布尔组合 → 期望枚举（映射表口径；``diagnostics`` 单独断言）
_LEGACY_TO_MODE = {
    (False, False): "policy_single_step",
    (True, False): "policy_wm_rollout",
    (True, True): "world_model_teacher_forcing",
}


def _obs(structured: bool = False) -> dict[str, torch.Tensor]:
    obs = make_obs(batch=BATCH, seed=3)
    if structured:
        s = torch.linspace(0.0, 40.0, 41)
        obs["lane_anchor_xy"] = (
            torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
        )
        obs["path_dynamic_distance"] = torch.full((BATCH, ), 30.0)
        obs["path_dynamic_ttc"] = torch.full((BATCH, ), 8.0)
    return obs


def _model(structured: bool = False) -> DrivingModel:
    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    if structured:
        model.set_path_head(True)
    return model.eval()


def _assert_bitwise_equal(a: dict, b: dict) -> None:
    assert sorted(a.keys()) == sorted(b.keys()), (sorted(a.keys()), sorted(b.keys()))
    for key in a:
        ta, tb = a[key], b[key]
        if not torch.is_tensor(ta):
            continue
        assert tuple(ta.shape) == tuple(tb.shape), key
        assert ta.dtype == tb.dtype, key
        delta = float((ta.double() - tb.double()).abs().max()) if ta.numel() else 0.0
        assert delta == 0.0, f"{key}: max|Δ|={delta}"


def test_mapping_table_declared() -> None:
    assert FORWARD_MODES == (
        "policy_single_step",
        "policy_wm_rollout",
        "world_model_teacher_forcing",
        "diagnostics",
    )
    assert FORWARD_MODE_FLAGS == {
        "policy_single_step": (False, False),
        "policy_wm_rollout": (True, False),
        "world_model_teacher_forcing": (True, True),
        "diagnostics": (True, True),
    }
    # resolve 口径
    assert resolve_forward_mode("policy_single_step", None, None) == (False, False)
    assert resolve_forward_mode("policy_wm_rollout", None, None) == (True, False)
    assert resolve_forward_mode("world_model_teacher_forcing", None, None) == (True, True)
    assert resolve_forward_mode("diagnostics", None, None) == (True, True)
    # 旧布尔 shim：单给一方 ⇒ 另一方按旧签名默认 True（与旧行为逐位一致）
    assert resolve_forward_mode(None, None, None) == (True, True)
    assert resolve_forward_mode(None, False, False) == (False, False)
    assert resolve_forward_mode(None, True, None) == (True, True)
    assert resolve_forward_mode(None, None, False) == (True, False)


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
@pytest.mark.parametrize("flags", [(False, False), (True, False), (True, True)])
def test_legacy_booleans_bitwise_equal_forward_mode(structured: bool, flags: tuple) -> None:
    """每组合一次前向逐位对照（旧布尔 ≡ 显式枚举）。"""
    rollout, world_model = flags
    model = _model(structured)
    obs = _obs(structured)
    with torch.no_grad():
        legacy_out = model(obs, rollout=rollout, world_model=world_model)
        mode_out = model(obs, forward_mode=_LEGACY_TO_MODE[flags])
    _assert_bitwise_equal(legacy_out, mode_out)


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_default_forward_equals_teacher_forcing_mode(structured: bool) -> None:
    """默认 ``forward(obs)``（旧 (True, True)）= ``world_model_teacher_forcing`` 逐位。"""
    model = _model(structured)
    obs = _obs(structured)
    with torch.no_grad():
        default_out = model(obs)
        mode_out = model(obs, forward_mode="world_model_teacher_forcing")
        diag_out = model(obs, forward_mode="diagnostics")
    _assert_bitwise_equal(default_out, mode_out)
    # diagnostics = 全量前向旗标（独立枚举名隔离诊断调用方；当前行为等价）
    _assert_bitwise_equal(mode_out, diag_out)


def test_diagnostics_mode_declares_full_forward_flags() -> None:
    model = _model()
    with pytest.raises(ValueError, match="forward_mode"):
        model(_obs(), forward_mode="diagnostics", rollout=True)
    with pytest.raises(ValueError, match="forward_mode"):
        model(_obs(), forward_mode="policy_single_step", world_model=False)
    with pytest.raises(ValueError, match="允许值"):
        model(_obs(), forward_mode="oneshot")


def test_legacy_signature_wm_detach_swallowed_noop_and_unknown_kwarg_fails() -> None:
    """旧签名透传 = 同结果（wm_detach 恒 no-op）；未知键 ⇒ TypeError（fail-fast）。"""
    model = _model()
    obs = _obs()
    with torch.no_grad():
        base = model(obs)
        with_true = model(obs, rollout=True, world_model=True, wm_detach=True)
        with_false = model(obs, rollout=True, world_model=True, wm_detach=False)
    _assert_bitwise_equal(base, with_true)
    _assert_bitwise_equal(base, with_false)

    with pytest.raises(TypeError, match="未知参数"):
        model(obs, wm_rollout=True)
