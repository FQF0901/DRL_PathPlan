"""R39-B2 学习残差（readout 列残差头 ΔA_θ + ``analytic_learned`` 档）：契约/逐位/接口测试。

出处：``.slim/deepwork/p1_v2/round39/b1_report.md`` §5（B2 训练建议）+ GPT R39 ``647f909``
§二（B2 = analytic + learned residual；B2 必须优于 B1）。

断言面：

1. **模式契约**：``WORLD_STATE_API_MODES`` 含 ``analytic_learned``；未挂头 ⇒ ``_rollout``
   fail-fast；``world_residual_scale`` 参数校验；挂头本身不改变默认路径。
2. **readout learned 载荷**：``None`` ⇒ 逐位不变；给定时仅作用于
   ``LEARNED_ALLOWED`` 四列 ∧ ``present=1`` 行（missing 行不加、值保持 0）；source 变
   ``learned``；形状/非有限 fail-fast。
3. **头契约**：末层零初始化 ⇒ Δ ≡ 0（严格）；输出有界（caps·tanh）；参数 ≤ ``HEAD_PARAM_CAP``；
   预算断言（``check_param_budget``；总量 ≤ ``TOTAL_PARAM_CAP`` = 2,000,000 绝对上限 =
   ``net/param_probe.TOTAL_CAP``）；输入形状 fail-fast。
4. **同一实现**：``readout_residual_features``/``apply_readout_residual`` 掩码 = base present；
   ``column_residual_loss`` = one-step 真值监督（掩码/scale/形状）。
5. **B2 ≡ B1（零头）**：同一模型同输入，``analytic`` 与（挂零初始化头的）``analytic_learned``
   输出逐位一致；扰动头 ⇒ 分离（残差真实入 readout）。
6. **ckpt 往返**：``save_residual_head``/``load_residual_head`` 逐位；格式不符 fail-fast；
   挂载后 state_dict 增 ``world_residual.*`` 键（默认 None ⇒ 键面不变）。
"""

from __future__ import annotations

import pytest
import torch

from net.model import DrivingModel
from net.world_residual import (
    DELTA_CAPS,
    HEAD_PARAM_CAP,
    READOUT_RESIDUAL_FIELDS,
    TOTAL_PARAM_CAP,
    ReadoutResidualHead,
    apply_readout_residual,
    check_param_budget,
    column_residual_loss,
    load_residual_head,
    readout_residual_features,
    residual_head_param_count,
    save_residual_head,
)
from net.world_state import (
    LEARNED_ALLOWED,
    POLICY_STATE_FIELDS,
    SOURCE_CODES,
    WORLD_STATE_API_MODES,
    WorldState,
    analytic_update,
    readout,
)

BATCH = 2


# --------------------------------------------------------------------------- #
# 小工具（口径同 ``tests/test_r39_b1_world_state.py``）
# --------------------------------------------------------------------------- #
def _tiny_structured(seed: int = 0) -> DrivingModel:
    torch.manual_seed(seed)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    model.set_path_head(True)
    model.set_path_head_ablation("structured")
    return model.eval()


def _obs_structured(batch: int = BATCH, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    frames, od_slots, ld_slots = 6, 16, 16

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    od_mask = (rand(batch, frames, od_slots) > -0.3).float()
    ld_mask = (rand(batch, frames, ld_slots) > -0.3).float()
    presence = od_mask * (rand(batch, frames, od_slots) > -0.3).float()
    od_hist = rand(batch, frames, od_slots, 9) * od_mask.unsqueeze(-1)
    ld_hist = rand(batch, frames, ld_slots, 7) * ld_mask.unsqueeze(-1)
    hist_valid = torch.zeros(batch, frames)
    hist_valid[:, frames - 3:] = 1.0
    obs: dict[str, torch.Tensor] = {
        "ego_hist": rand(batch, frames, 1, 8),
        "ego_hist_mask": torch.ones(batch, frames, 1),
        "od_hist": od_hist,
        "od_hist_mask": od_mask,
        "od_id_hist": (rand(batch, frames, od_slots).abs() * 10).long() + 1,
        "od_presence_hist": presence,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_mask,
        "others_hist": rand(batch, frames, 1, 33),
        "others_hist_mask": torch.ones(batch, frames, 1),
        "hist_valid": hist_valid,
        "ego": rand(batch, 1, 8),
        "od": od_hist[:, -1],
        "od_mask": od_mask[:, -1],
        "od_id": (rand(batch, od_slots).abs() * 10).long() + 1,
        "od_presence": presence[:, -1],
        "ld": ld_hist[:, -1],
        "ld_mask": ld_mask[:, -1],
        "nav": rand(batch, 1, 11),
        "nav_mask": torch.ones(batch, 1, 1),
        "signal": rand(batch, 1, 4),
        "signal_mask": torch.ones(batch, 1, 1),
    }
    s = torch.linspace(0.0, 40.0, 41)
    obs["lane_anchor_xy"] = (
        torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()
    )
    obs["lane_anchor_mask"] = torch.ones(batch, 41)
    for key, value in (
        ("path_left_margin", 3.0),
        ("path_right_margin", 3.0),
        ("path_static_distance", 40.0),
        ("path_dynamic_distance", 60.0),
        ("path_dynamic_ttc", 8.0),
        ("path_nav_progress", 0.2),
        ("path_distance_to_target_lane_end", 100.0),
    ):
        obs[key] = torch.full((batch, 1), float(value))
    return obs


def _synthetic_state(*, batch: int = 1, speed: float = 5.0, od_slots: int = 2,
                     ld_slots: int = 2, static_present: float = 1.0) -> WorldState:
    anchor = torch.stack(
        [torch.linspace(0.0, 40.0, 41), torch.zeros(41)], dim=-1
    ).unsqueeze(0).expand(batch, -1, -1).contiguous()
    od_anchor = torch.zeros(batch, od_slots, 5)
    od_feat = torch.zeros(batch, od_slots, 9)
    od_valid = torch.zeros(batch, od_slots)
    if od_slots >= 1:
        od_anchor[:, 0] = torch.tensor([20.0, 0.0, -2.0, 0.0, 0.0])
        od_feat[:, 0] = torch.tensor([20.0, 0.0, -2.0, 0.0, 1.0, 0.0, 4.0, 2.0, 1.0])
        od_valid[:, 0] = 1.0
    ld_anchor = torch.zeros(batch, ld_slots, 4)
    ld_anchor[:, 0, 0] = 10.0
    ld_valid = torch.zeros(batch, ld_slots)
    ld_valid[:, 0] = 1.0
    exog = torch.zeros(batch, 8)
    exog[:, 0] = speed
    exog[:, 1] = 3.0
    exog[:, 2] = 3.0
    exog[:, 3] = 40.0
    exog[:, 4] = 30.0
    exog[:, 5] = 8.0
    exog[:, 7] = 100.0
    present = torch.ones(batch, 8)
    present[:, 3] = float(static_present)  # static 列可关（验证 missing 行不加残差）
    return WorldState.t0(
        ego_speed=torch.full((batch, ), float(speed)),
        od_anchor=od_anchor,
        od_feat=od_feat,
        od_latent=torch.zeros(batch, od_slots, 8),
        od_valid=od_valid,
        ld_anchor=ld_anchor,
        ld_latent=torch.zeros(batch, ld_slots, 8),
        ld_valid=ld_valid,
        lane_anchor=anchor,
        lane_mask=torch.ones(batch, 41),
        lane_target_valid=torch.ones(batch),
        t0_exog=exog,
        t0_present=present,
    )


# =========================================================================== #
# 1. 模式契约 / fail-fast / 冻结面
# =========================================================================== #
def test_modes_contract_and_analytic_learned_requires_head() -> None:
    assert WORLD_STATE_API_MODES == ("off", "analytic", "analytic_residual", "analytic_learned")
    model = _tiny_structured()
    obs = _obs_structured()
    model.set_world_state_api("analytic_learned")
    with pytest.raises(ValueError, match="set_world_residual_head"):
        with torch.no_grad():
            model(obs, forward_mode="world_model_teacher_forcing")
    # 挂头后同前向可跑；缩放进 [0,4]
    model.set_world_residual_head(ReadoutResidualHead())
    with pytest.raises(ValueError):
        model.set_world_residual_scale(9.0)
    with torch.no_grad():
        out = model(obs, forward_mode="world_model_teacher_forcing")
    assert out["structured_ds"].shape[1] == 6
    # 非 Module fail-fast
    with pytest.raises(TypeError):
        model.set_world_residual_head(object())


def test_head_attach_changes_keyface_only_when_enabled() -> None:
    model = _tiny_structured()
    fresh = sorted(model.state_dict().keys())
    assert model.world_residual is None
    assert not any(key.startswith("world_residual.") for key in fresh)
    model.set_world_residual_head(ReadoutResidualHead())
    attached = sorted(model.state_dict().keys())
    assert any(key.startswith("world_residual.") for key in attached)
    assert len(attached) == len(fresh) + len([k for k in attached if k.startswith("world_residual.")])
    model.set_world_residual_head(None)
    assert sorted(model.state_dict().keys()) == fresh


# =========================================================================== #
# 2. readout learned 载荷
# =========================================================================== #
def test_readout_learned_none_is_bitwise_and_payload_applies_masked() -> None:
    state = _synthetic_state(batch=2, speed=5.0)
    base = readout(state)
    same = readout(state, learned=None)
    assert torch.equal(base["long_exog"], same["long_exog"])
    delta = torch.zeros(2, 4)
    delta[:, 0] = 0.5   # speed
    delta[:, 1] = 7.0   # static（present=1 行生效）
    delta[:, 2] = -3.0  # dynamic
    delta[:, 3] = 2.0   # ttc
    payload = {"delta": delta, "mask": torch.tensor([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])}
    view = readout(state, learned=payload)
    index = [POLICY_STATE_FIELDS.index(name) for name in READOUT_RESIDUAL_FIELDS]
    # 行 0 的 static 掩码=0 ⇒ 不加；行 1 加
    assert torch.allclose(view["long_exog"][0, index[1]], base["long_exog"][0, index[1]])
    assert torch.allclose(view["long_exog"][1, index[1]], base["long_exog"][1, index[1]] + 7.0)
    # speed/dyn/ttc 全行加
    assert torch.allclose(view["long_exog"][:, index[0]], base["long_exog"][:, index[0]] + 0.5)
    # source：命中行 = learned；其余列不变
    assert view["sources"]["speed"]["learned"] == 2
    assert base["sources"]["speed"].get("learned") is None
    assert view["sources"]["nav_progress"] == base["sources"]["nav_progress"]
    assert view["diagnostics"]["learned_applied"]["speed"] == 2
    assert view["diagnostics"]["learned_enabled"] is True


def test_readout_learned_missing_rows_stay_absent_and_shapes_fail_fast() -> None:
    state = _synthetic_state(batch=1, static_present=0.0)
    base = readout(state)
    delta = torch.full((1, 4), 3.0)
    payload = {"delta": delta, "mask": torch.ones(1, 4)}
    view = readout(state, learned=payload)
    # static present=0（missing）⇒ 值仍 0 ∧ source 仍 missing（学习残差不得制造存在性）
    assert float(view["long_exog"][0, POLICY_STATE_FIELDS.index("static_distance")]) == 0.0
    assert view["sources"]["static_distance"].get("missing") == 1
    assert view["sources"]["static_distance"].get("learned") is None
    with pytest.raises(ValueError):
        readout(state, learned={"delta": torch.zeros(1, 3), "mask": torch.ones(1, 4)})
    with pytest.raises(ValueError):
        readout(state, learned={"delta": torch.full((1, 4), float("nan")), "mask": torch.ones(1, 4)})
    with pytest.raises(TypeError):
        readout(state, learned=[1, 2, 3])


# =========================================================================== #
# 3. 残差头契约
# =========================================================================== #
def _head_inputs(batch: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "prev_exog": torch.randn(batch, 8, generator=generator),
        "prev_present": (torch.rand(batch, 8, generator=generator) > 0.2).float(),
        "action": torch.randn(batch, 2, generator=generator),
        "base_exog": torch.randn(batch, 4, generator=generator) * 5.0,
        "base_present": (torch.rand(batch, 4, generator=generator) > 0.2).float(),
    }


def test_residual_head_zero_init_bounded_and_param_cap() -> None:
    head = ReadoutResidualHead()
    inputs = _head_inputs()
    delta = head(**inputs)
    assert delta.shape == (3, len(READOUT_RESIDUAL_FIELDS))
    assert torch.equal(delta, torch.zeros_like(delta))  # 零初始化 ⇒ 严格零残差
    caps = torch.tensor(DELTA_CAPS)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 2.0)
    delta = head(**inputs)
    assert bool((delta.abs() <= caps + 1e-5).all())  # 有界（caps·tanh）
    params = residual_head_param_count(head)
    assert params == head.param_count() and params <= HEAD_PARAM_CAP
    report = check_param_budget(head)
    assert report["head_params"] == params and report["head_cap"] == HEAD_PARAM_CAP
    assert report["total_cap"] == TOTAL_PARAM_CAP == 2_000_000
    with pytest.raises(AssertionError):
        check_param_budget(head, head_cap=params - 1)
    # R40 §五.5 绝对上限（= net/param_probe.TOTAL_CAP；R39「基线×1.02」乘性规则废止）：
    # 阈值处通过、超一即拒
    assert check_param_budget(head, model_total=TOTAL_PARAM_CAP)["model_total_cap"] == TOTAL_PARAM_CAP
    with pytest.raises(AssertionError):
        check_param_budget(head, model_total=TOTAL_PARAM_CAP + 1)
    with pytest.raises(AssertionError):
        check_param_budget(head, model_total=10_000_000)
    with pytest.raises(ValueError):
        head(**{**inputs, "base_exog": torch.zeros(3, 3)})
    with pytest.raises(ValueError):
        head(**{**inputs, "prev_exog": torch.zeros(2, 8)})


def test_features_and_apply_use_base_present_as_mask() -> None:
    state = _synthetic_state(batch=1, static_present=0.0)
    prev = readout(state)
    analytic_update(state, ds=torch.tensor([2.0]), dtheta=torch.tensor([0.0]))
    base = readout(state)
    features = readout_residual_features(
        prev_view=prev, action=torch.tensor([[2.0, 0.0]]), base_view=base
    )
    assert features["prev_exog"].shape == (1, 8)
    assert features["base_exog"].shape == (1, 4)
    delta = torch.ones(1, 4)
    payload = apply_readout_residual(base_view=base, delta=delta)
    assert torch.equal(payload["mask"][0, 1], torch.tensor(0.0))  # static present=0
    assert torch.equal(payload["mask"][0, 0], torch.tensor(1.0))
    with pytest.raises(ValueError):
        apply_readout_residual(base_view=base, delta=torch.full((1, 4), float("inf")))


# =========================================================================== #
# 4. one-step 监督损失（= residual_loss 同语义）
# =========================================================================== #
def test_column_residual_loss_semantics_and_scale() -> None:
    base = torch.zeros(4, 4)
    real = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [1.0, 2.0, 3.0, 4.0],
         [1.0, 2.0, 3.0, 4.0],
         [1.0, 2.0, 3.0, 4.0]]
    )
    mask = torch.tensor(
        [[1.0, 0.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0],
         [0.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0]]
    )
    perfect = column_residual_loss(
        real, base_exog=base, base_present=mask, real_exog=real, real_present=mask
    )
    assert float(perfect["loss"]) < 1e-12
    assert torch.equal(perfect["mask"], mask)
    # 掩码行不计入（第 2 行掩码 = [0,1,0,0] ⇒ 只可扰动列 0/2/3）
    perturbed = real.clone()
    perturbed[2, [0, 2, 3]] = 100.0
    out = column_residual_loss(
        real, base_exog=base, base_present=mask, real_exog=perturbed, real_present=mask
    )
    assert float(out["loss"]) < 1e-12
    diff = column_residual_loss(
        real + 1.0, base_exog=base, base_present=mask, real_exog=real, real_present=mask
    )
    assert abs(float(diff["loss"]) - 0.5) < 1e-6  # smooth_l1(beta=1) 的 |diff|=1 ⇒ 0.5
    scaled = column_residual_loss(
        real + 1.0, base_exog=base, base_present=mask, real_exog=real, real_present=mask,
        scale=(0.5, 0.5, 0.5, 0.5),
    )
    assert float(scaled["loss"]) > float(diff["loss"])  # 归一化尺度 <1 ⇒ 损失放大
    with pytest.raises(ValueError):
        column_residual_loss(
            real, base_exog=base, base_present=mask, real_exog=real, real_present=mask, scale=(1.0,)
        )
    with pytest.raises(ValueError):
        column_residual_loss(
            real, base_exog=base, base_present=mask, real_exog=real, real_present=mask, mode="huber"
        )


# =========================================================================== #
# 5. B2 ≡ B1（零头）与扰动分离（端到端）
# =========================================================================== #
def test_analytic_learned_zero_head_equals_analytic_and_perturb_separates() -> None:
    model = _tiny_structured()
    obs = _obs_structured()
    obs["ego"] = obs["ego"].clone()
    obs["ego"][:, 0] = 6.0  # 速度列离开 ds_init 下界（否则 ds 被 clamp 钉住，无法分离）
    model.set_world_state_api("analytic")
    with torch.no_grad():
        reference = model(obs, forward_mode="world_model_teacher_forcing")
    model.set_world_state_api("analytic_learned")
    model.set_world_residual_head(ReadoutResidualHead())
    with torch.no_grad():
        zero_head = model(obs, forward_mode="world_model_teacher_forcing")
    for key in ("structured_ds", "structured_kappa_near", "plan", "traj_xy"):
        assert torch.equal(reference[key], zero_head[key]), f"零初始化头未逐位：{key}"
    # scale=0 同口径（诊断旋钮）
    model.set_world_residual_scale(0.0)
    with torch.no_grad():
        scaled_zero = model(obs, forward_mode="world_model_teacher_forcing")
    assert torch.equal(reference["structured_ds"], scaled_zero["structured_ds"])
    # 扰动头 ⇒ 分离（残差真实入 readout）
    head = ReadoutResidualHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.5)
    model.set_world_residual_head(head)
    model.set_world_residual_scale(1.0)
    with torch.no_grad():
        perturbed = model(obs, forward_mode="world_model_teacher_forcing")
    assert not torch.equal(reference["structured_ds"], perturbed["structured_ds"])
    model.set_world_state_api("off")


def test_hooks_record_learned_sources_for_b2() -> None:
    from net.probes import ProbeRecorder

    model = _tiny_structured()
    obs = _obs_structured()
    head = ReadoutResidualHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.3)
    probes = ProbeRecorder(enabled=False)
    probes.world_state.enabled = True
    model.set_probes(probes)
    model.set_world_residual_head(head)
    model.set_world_state_api("analytic_learned")
    with torch.no_grad():
        model(obs, forward_mode="world_model_teacher_forcing")
    summary = probes.world_state.summary()
    assert summary["checks"]["sources_legal"] is True
    assert summary["checks"]["held_whitelist"] is True
    learned_steps = [
        record["step"]
        for record in probes.world_state.records
        if record["sources"]["speed"].get("learned", 0) > 0
    ]
    assert learned_steps and 0 not in learned_steps  # step0 由 t0 真实列给出（无残差）
    model.set_world_state_api("off")


# =========================================================================== #
# 6. ckpt 往返
# =========================================================================== #
def test_save_load_residual_head_roundtrip(tmp_path) -> None:
    head = ReadoutResidualHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.7)
    inputs = _head_inputs(batch=2)
    path = save_residual_head(
        tmp_path / "head.pt", head, meta={"hidden": head.hidden, "note": "unit-test"}
    )
    loaded, meta = load_residual_head(path)
    assert meta["note"] == "unit-test" and meta["hidden"] == head.hidden
    assert torch.equal(loaded(**inputs), head(**inputs))
    assert residual_head_param_count(loaded) == residual_head_param_count(head)
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": head.state_dict(), "meta": {}}, str(bad))
    with pytest.raises(ValueError):
        load_residual_head(bad)
