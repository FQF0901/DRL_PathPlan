"""R39-A1：``forward_mode`` 显式枚举 = 唯一模式开关（布尔 shim 已删除 ⇒ fail-fast）。

出处：GPT R39 §二-A 任务 4/5（生产调用统一迁 ``forward_mode``；删除易误用布尔 shim、保留
fail-fast）+ ``docs/repo_handbook.md`` 附录 D（可达性审计 H 组）。断言面（CPU、无 metadrive）：

1. **映射表**：``FORWARD_MODE_FLAGS`` = 声明口径（``policy_single_step`` (F,F) /
   ``policy_wm_rollout`` (T,F) / ``world_model_teacher_forcing`` (T,T) / ``diagnostics`` (T,T)）；
2. **缺省**：``forward(obs)`` ≡ ``forward_mode="world_model_teacher_forcing"``（旧默认
   (True, True) 逐位保留）；``diagnostics`` = 全量前向旗标（独立枚举名隔离诊断调用方）；
3. **布尔 shim 已删除**：``rollout=/world_model=`` 残留调用 ⇒ ``ValueError``（文案指路
   ``forward_mode``）；`semantic_action_decomposition` 同口径；
4. **旧签名透传 = 同结果**：``wm_detach=True/False`` 经 ``**legacy_signature`` 吞掉
   （恒 no-op，逐位同结果）；其他未知关键字 ⇒ ``TypeError``。
"""

from __future__ import annotations

import torch
import pytest

from net.model import FORWARD_MODE_FLAGS, FORWARD_MODES, DrivingModel, resolve_forward_mode
from tests.test_net_shapes import make_obs

BATCH = 2


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


#: 旧布尔组合 → 期望枚举（历史映射口径；只作声明性对照，不再有布尔入口）
_LEGACY_TO_MODE = {
    (False, False): "policy_single_step",
    (True, False): "policy_wm_rollout",
    (True, True): "world_model_teacher_forcing",
}


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
    # resolve 口径（R39-A1：唯一形参 forward_mode；缺省 = WMTF 旧默认）
    assert resolve_forward_mode(None) == (True, True)
    assert resolve_forward_mode("policy_single_step") == (False, False)
    assert resolve_forward_mode("policy_wm_rollout") == (True, False)
    assert resolve_forward_mode("world_model_teacher_forcing") == (True, True)
    assert resolve_forward_mode("diagnostics") == (True, True)
    with pytest.raises(ValueError, match="允许值"):
        resolve_forward_mode("oneshot")


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


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_enum_modes_produce_declared_flag_semantics(structured: bool) -> None:
    """枚举语义未被本次迁移改变：cheap path 键集 ⊂ 全量；WM rollout 无物理预测键。"""
    model = _model(structured)
    obs = _obs(structured)
    with torch.no_grad():
        cheap = model(obs, forward_mode="policy_single_step")
        wm_rollout = model(obs, forward_mode="policy_wm_rollout")
        full = model(obs, forward_mode="world_model_teacher_forcing")
    # cheap path：不返回 WM 物理预测键（str/结构化同口径）
    for key in ("od_pred", "ld_pred"):
        assert key not in cheap
        # wm rollout：逐步组装（plan_from_aux=0）且不返回 WM 物理解码键
        assert key not in wm_rollout
        assert key in full
    if structured:
        assert cheap["plan_from_aux"].item() == 1.0
        assert wm_rollout["plan_from_aux"].item() == 0.0
    # 旅行键集关系：cheap ⊂ wm_rollout ⊂ full
    assert set(cheap).issubset(set(wm_rollout))
    assert set(wm_rollout).issubset(set(full))
    # cheap path 的策略/价值输出与全量前向逐位一致（枚举化的旧不变式）
    for key in ("action_mu", "value", "latent", "router_logits", "expert_weights"):
        _assert_bitwise_equal({key: cheap[key]}, {key: full[key]})


def test_boolean_shim_removed_fail_fast_points_to_forward_mode() -> None:
    """R39-A1：旧布尔口残留调用 ⇒ ValueError（文案含 forward_mode 指路）。"""
    model = _model()
    obs = _obs()
    with pytest.raises(ValueError, match="forward_mode"):
        model(obs, rollout=True)
    with pytest.raises(ValueError, match="forward_mode"):
        model(obs, rollout=False, world_model=False)
    with pytest.raises(ValueError, match="forward_mode"):
        model(obs, world_model=False)
    with pytest.raises(ValueError, match="forward_mode"):
        model(obs, forward_mode="policy_single_step", rollout=False)
    with pytest.raises(ValueError, match="允许值"):
        model(obs, forward_mode="oneshot")
    with pytest.raises(ValueError, match="forward_mode"):
        model.semantic_action_decomposition(obs, rollout=False)


def test_legacy_signature_wm_detach_swallowed_noop_and_unknown_kwarg_fails() -> None:
    """旧签名透传 = 同结果（wm_detach 恒 no-op）；未知键 ⇒ TypeError（fail-fast）。"""
    model = _model()
    obs = _obs()
    with torch.no_grad():
        base = model(obs)
        with_true = model(obs, forward_mode="world_model_teacher_forcing", wm_detach=True)
        with_false = model(obs, forward_mode="world_model_teacher_forcing", wm_detach=False)
    _assert_bitwise_equal(base, with_true)
    _assert_bitwise_equal(base, with_false)

    with pytest.raises(TypeError, match="未知参数"):
        model(obs, wm_rollout=True)


def test_semantic_action_decomposition_default_matches_old_rollout_false() -> None:
    """``semantic_action_decomposition`` 缺省 ≡ 旧 ``rollout=False``（cheap path）逐位。"""
    model = _model()
    obs = _obs()
    with torch.no_grad():
        dec_default = model.semantic_action_decomposition(obs)
        dec_explicit = model.semantic_action_decomposition(obs, forward_mode="policy_single_step")
    _assert_bitwise_equal(dec_default, dec_explicit)
