"""R39-B1 统一 imagined WorldState → PolicyStateView：契约/逐位/接口测试。

出处：GPT R39 主方案 ``647f909`` §二 R39-B（九步循环）+ 补充裁决 ``fe5de55`` +
``docs/deepseekv4p1_argue.md`` 当期快照 §1/§2（A3 阻尼教训；残差复用 ``st_gnn`` 解码头
= 零新增参数）。

断言面：

1. **默认 off 逐位**：``set_world_state_api`` 开→关往返后输出全键 ``torch.equal``；
   开关不新增参数/buffer（state_dict 键面与 fresh 模型一致）；``analytic`` 不改变输出键集。
2. **参数校验**：未知 mode / α ∉ [0,1] / speed_max ≤ 0 ⇒ ``ValueError``；非 structured
   路径 + 开关非 off ⇒ ``_rollout`` fail-fast。
3. **analytic_update**：位姿 = ``arc_step/compose_pose`` 同实现；speed 阻尼公式
   ``v' = clamp(v + α(ds/dt − v), 0, v_max)``（α=0/0.5/1 + clamp）；OD 对象先验
   ``p += dt·(v_ref+vx, vy)``；无效槽位严格保持 0。
4. **readout 七量契约**：字段集/列序 = ``POLICY_STATE_FIELDS`` = ``LONG_EXOG_FEATURE_NAMES``；
   producer = ``FIELD_PRODUCERS`` 唯一写点；identity 位姿下 static/d2end/margin/nav/speed
   重算自洽；动态距离/TTC 与 ``env.obs.long_exog`` 公式一致；route progress
   analytic（可推断）/ held（不可推断，显式标注）。
5. **残差接口**：``residual_targets = 真实 − 解析``（one-step）；loss=0 ⟺ pred=真实；
   梯度沿 residual 可达；形状/非有限真值 fail-fast。
6. **zero-init 等价**：未扰动解码头 ⇒ ``analytic`` 与 ``analytic_residual`` 逐位一致；
   扰动 od_head ⇒ 二者分离（残差真实入 readout）。
7. **hooks**：``WorldStateHooks`` 自动断言（producer 唯一/held 白名单/missing⇒present=0/
   Imagined writeback=0）；篡改记录 ⇒ violations 非空。
8. **Imagined Memory 隔离**：帧为克隆副本（改源张量不影响记录）；``writeback=0``；
   rollout 结束自动 ``destroy``；销毁后写入 fail-fast。
"""

from __future__ import annotations

import pytest
import torch

from net.model import DrivingModel
from net.path_head import LONG_EXOG_FEATURE_NAMES
from net.probes import ProbeRecorder, WorldStateHooks
from net.world_state import (
    FIELD_PRODUCERS,
    HELD_ALLOWED,
    POLICY_STATE_FIELDS,
    SOURCE_CODES,
    SOURCES,
    WORLD_STATE_API_MODES,
    ImaginedMemory,
    WorldState,
    analytic_update,
    od_decode_prior,
    readout,
    residual_loss,
    residual_targets,
)

BATCH = 2


# --------------------------------------------------------------------------- #
# 小工具：tiny 模型 + 结构化观测
# --------------------------------------------------------------------------- #
def _tiny_structured(seed: int = 0) -> DrivingModel:
    torch.manual_seed(seed)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    model.set_path_head(True)
    model.set_path_head_ablation("structured")
    return model.eval()


def _obs_structured(batch: int = BATCH, seed: int = 0, *,
                    with_end: bool = True) -> dict[str, torch.Tensor]:
    """结构化 obs（8 列 path_* 全给；直线锚过原点 ⇒ identity 位姿 margin 自洽可判）。"""
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
    ):
        obs[key] = torch.full((batch, 1), float(value))
    if with_end:
        obs["path_distance_to_target_lane_end"] = torch.full((batch, 1), 100.0)
    return obs


def _synthetic_state(*, batch: int = 1, speed: float = 5.0, alpha: float = 1.0,
                     speed_max: float = 20.0, progress: float = 0.0,
                     end: float = 100.0, end_present: float = 1.0,
                     od_slots: int = 2, ld_slots: int = 2) -> WorldState:
    """手工 WorldState（t0 帧；直线锚过原点；OD 槽可判）。"""
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
    if od_slots >= 2:
        od_anchor[:, 1] = torch.tensor([-40.0, 3.0, 1.0, 0.0, 0.0])
        od_feat[:, 1] = torch.tensor([-40.0, 3.0, 1.0, 0.0, 1.0, 0.0, 4.0, 2.0, 1.0])
        od_valid[:, 1] = 1.0
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
    exog[:, 6] = progress
    exog[:, 7] = end
    present = torch.ones(batch, 8)
    present[:, 7] = float(end_present)
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
        speed_alpha=alpha,
        speed_max=speed_max,
    )


# =========================================================================== #
# 1. 默认 off 逐位 + 键面
# =========================================================================== #
def test_world_state_api_default_off_is_bitwise_and_keyface_stable() -> None:
    model = _tiny_structured()
    obs = _obs_structured()
    with torch.no_grad():
        baseline = model(obs)
    model.set_world_state_api("analytic")
    model.set_world_state_api("off")
    with torch.no_grad():
        again = model(obs)
    assert sorted(baseline) == sorted(again)
    for key in baseline:
        if torch.is_tensor(baseline[key]) and torch.is_tensor(again[key]):
            assert torch.equal(baseline[key], again[key]), f"默认路径逐位破坏：{key}"
    # 开关不新增参数/buffer：键面与 fresh 模型一致
    fresh_keys = sorted(model.state_dict().keys())
    model.set_world_state_api("analytic")
    assert sorted(model.state_dict().keys()) == fresh_keys
    model.set_world_state_api("off")
    # analytic 不改变输出键集（只有数值变化）
    model.set_world_state_api("analytic")
    with torch.no_grad():
        changed = model(obs)
    assert sorted(changed) == sorted(baseline)
    model.set_world_state_api("off")


def test_world_state_api_validation_and_structured_requirement() -> None:
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6).eval()
    with pytest.raises(ValueError):
        model.set_world_state_api("residual_only")
    with pytest.raises(ValueError):
        model.set_world_state_api("analytic", speed_alpha=1.5)
    with pytest.raises(ValueError):
        model.set_world_state_api("analytic", speed_max=0.0)
    assert model.world_state_api == "off"
    # 非 structured 路径 + 开关非 off：_rollout fail-fast
    model.set_world_state_api("analytic")
    with pytest.raises(ValueError, match="structured"):
        with torch.no_grad():
            model(_obs_structured())


# =========================================================================== #
# 2. analytic_update：运动学 + 阻尼 speed + 对象先验
# =========================================================================== #
def test_analytic_update_kinematics_and_damped_speed() -> None:
    from net.model import arc_step, compose_pose

    state = _synthetic_state(batch=2, speed=5.0)
    ds = torch.tensor([2.0, 1.0])
    dtheta = torch.tensor([0.2, -0.1])
    analytic_update(state, ds=ds, dtheta=dtheta)
    dx, dy = arc_step(ds, dtheta)
    expected_pose = compose_pose(torch.zeros(2, 3), dx, dy, dtheta)
    assert torch.equal(state.ego_pose, expected_pose)
    # α=1 ⇒ v' = ds/dt（clamp [0, v_max]）
    assert torch.allclose(state.ego_speed, torch.tensor([4.0, 2.0]), atol=1e-6)
    # progress 累积
    assert torch.allclose(state.progress_s, ds, atol=1e-6)
    # OD 对象先验：p += dt·(v_ref + vx, vy)（槽 0：v_ref=5 + vx=−2 ⇒ +1.5 m）
    assert torch.allclose(state.od_anchor[0, 0, :2], torch.tensor([21.5, 0.0]), atol=1e-6)
    # 槽 1：v_ref=5 + vx=1 ⇒ 世界系 6 m/s；p_x += 0.5·6 = +3
    assert torch.allclose(state.od_anchor[0, 1, :2], torch.tensor([-37.0, 3.0]), atol=1e-6)
    # 无效槽位严格 0（编码器空间不变量）
    if state.od_anchor.shape[1] > 0:
        invalid = state.od_valid < 0.5
        if invalid.any():
            assert torch.equal(state.od_anchor[invalid], torch.zeros_like(state.od_anchor[invalid]))

    # α=0 ⇒ speed held；α=0.5 ⇒ 半步；clamp 上限
    # ds/dt = 8 m/s：α=0 ⇒ hold；α=0.5 ⇒ 5+0.5·(8−5)=6.5；α=1 ⇒ 全自洽 8
    for alpha, expected in ((0.0, 5.0), (0.5, 6.5), (1.0, 8.0)):
        fresh = _synthetic_state(batch=1, speed=5.0, alpha=alpha)
        analytic_update(fresh, ds=torch.tensor([4.0]), dtheta=torch.tensor([0.0]))
        assert torch.allclose(fresh.ego_speed, torch.tensor([expected]), atol=1e-6)
    capped = _synthetic_state(batch=1, speed=19.0, speed_max=20.0)
    analytic_update(capped, ds=torch.tensor([10.0]), dtheta=torch.tensor([0.0]))
    assert torch.allclose(capped.ego_speed, torch.tensor([20.0]), atol=1e-6)


def test_analytic_update_residual_and_latent_memory_bookkeeping() -> None:
    state = _synthetic_state(batch=1)
    od_residual = torch.zeros(1, 2, 5)
    od_residual[0, 0, 0] = 0.5
    ld_residual = torch.zeros(1, 2, 4)
    ld_residual[0, 0, 3] = 0.25
    z_od = torch.ones(1, 2, 3)
    z_ld = torch.full((1, 2, 3), 2.0)
    analytic_update(
        state, ds=torch.tensor([1.0]), dtheta=torch.tensor([0.0]),
        od_residual=od_residual, ld_residual=ld_residual, z_od=z_od, z_ld=z_ld, step=1,
    )
    # 先验：20 + dt·(v_ref+vx) = 20 + 0.5·(5−2) = 21.5；ΔA：+0.5 ⇒ 22.0
    assert torch.allclose(state.od_anchor[0, 0, 0], torch.tensor(22.0), atol=1e-6)
    assert torch.allclose(state.ld_anchor[0, 0, 3], torch.tensor(0.25), atol=1e-6)
    assert torch.equal(state.od_latent, z_od)
    assert torch.equal(state.ld_latent, z_ld)
    assert len(state.memory.frames) == 2  # t0 real + step1 predicted
    assert state.memory.frames[-1].source == "predicted"
    # 残差形状错误 fail-fast
    with pytest.raises(ValueError):
        analytic_update(state, ds=torch.tensor([1.0]), dtheta=torch.tensor([0.0]),
                        od_residual=torch.zeros(1, 3, 5))
    # detach 语义（想象态非可微边界）：入状态张量不带梯度
    ds_grad = torch.tensor([1.0], requires_grad=True)
    pose_grad = torch.tensor([[1.0, 0.0, 0.1]], requires_grad=True)
    fresh = _synthetic_state(batch=1)
    analytic_update(fresh, ds=ds_grad, dtheta=torch.tensor([0.0]), pose=pose_grad)
    assert not fresh.ego_pose.requires_grad
    assert not fresh.ego_speed.requires_grad
    assert not fresh.od_anchor.requires_grad


# =========================================================================== #
# 3. readout：七量契约 + identity 自洽 + 与 env 公式一致
# =========================================================================== #
def test_readout_field_contract_and_identity_selfconsistency() -> None:
    state = _synthetic_state(batch=1, speed=5.0, progress=0.0)
    view = readout(state)
    assert tuple(view["fields"].keys()) == tuple(POLICY_STATE_FIELDS)
    assert tuple(LONG_EXOG_FEATURE_NAMES) == tuple(POLICY_STATE_FIELDS)
    assert view["producers"] == FIELD_PRODUCERS
    assert view["long_exog"].shape == (1, 8)
    # identity 位姿：static/d2end/margin/nav/speed 与 t0 列自洽
    assert float(view["fields"]["speed"][0]) == pytest.approx(5.0, abs=1e-6)
    assert float(view["fields"]["left_margin"][0]) == pytest.approx(3.0, abs=1e-4)
    assert float(view["fields"]["right_margin"][0]) == pytest.approx(3.0, abs=1e-4)
    assert float(view["fields"]["static_distance"][0]) == pytest.approx(40.0, abs=1e-4)
    assert float(view["fields"]["distance_to_target_lane_end"][0]) == pytest.approx(100.0, abs=1e-4)
    assert float(view["fields"]["nav_progress"][0]) == pytest.approx(0.0, abs=1e-6)
    # 动态距离 = ego→矩形最近距（槽 0：dx=20 前车 L=4 ⇒ 18；槽 1 更远 ⇒ min=18）
    assert float(view["fields"]["dynamic_distance"][0]) == pytest.approx(18.0, abs=1e-4)
    # TTC = min(dx/(−vx))（槽 0：20/2 = 10 s；cap 60）
    assert float(view["fields"]["dynamic_ttc"][0]) == pytest.approx(10.0, abs=1e-4)
    codes = view["source_codes"][0].tolist()
    assert all(SOURCES[code] in SOURCES for code in codes)
    assert codes[SOURCE_CODES["analytic"]] == SOURCE_CODES["analytic"]
    for name, counts in view["sources"].items():
        assert set(counts).issubset(set(SOURCES)), name


def test_readout_progress_shrink_margins_and_route_progress() -> None:
    state = _synthetic_state(batch=1, speed=5.0, progress=0.2, end=100.0)
    # route_length = 100 / (1 − 0.2) = 125 ⇒ ds=5 ⇒ progress += 0.04
    analytic_update(state, ds=torch.tensor([5.0]), dtheta=torch.tensor([0.0]))
    view = readout(state)
    assert float(view["fields"]["nav_progress"][0]) == pytest.approx(0.24, abs=1e-6)
    assert view["sources"]["nav_progress"] == {"analytic": 1}
    assert float(view["fields"]["static_distance"][0]) == pytest.approx(35.0, abs=1e-4)
    assert float(view["fields"]["distance_to_target_lane_end"][0]) == pytest.approx(95.0, abs=1e-4)
    # margin：直锚过原点，位姿前移 ⇒ lat≈0 ⇒ margin 保持 W/2
    assert float(view["fields"]["left_margin"][0]) == pytest.approx(3.0, abs=1e-3)


def test_readout_route_progress_held_when_length_not_derivable() -> None:
    state = _synthetic_state(batch=1, progress=0.2, end=100.0, end_present=0.0)
    analytic_update(state, ds=torch.tensor([5.0]), dtheta=torch.tensor([0.0]))
    view = readout(state)
    # route 长度不可推断 ⇒ held（显式标注；值保持）
    assert float(view["fields"]["nav_progress"][0]) == pytest.approx(0.2, abs=1e-6)
    assert view["sources"]["nav_progress"] == {"held": 1}
    assert view["source_codes"][0, POLICY_STATE_FIELDS.index("nav_progress")].item() == \
        SOURCE_CODES["held"]
    # d2end 缺 present ⇒ missing（值 0 + present=0，显式标注）
    assert float(view["fields"]["distance_to_target_lane_end"][0]) == 0.0
    assert float(view["present"]["distance_to_target_lane_end"][0]) == 0.0
    assert view["sources"]["distance_to_target_lane_end"] == {"missing": 1}


def test_readout_dynamic_matches_env_formula() -> None:
    from env.obs.long_exog import od_truth_slots

    state = _synthetic_state(batch=1, speed=5.0, od_slots=3)
    # 槽 2：接近对象（dx>0, vx<0）但更近 ⇒ 应主导 TTC；槽 1 后退对象不进 TTC
    state.od_anchor[0, 2] = torch.tensor([8.0, 1.0, -4.0, 0.0, 0.0])
    state.od_feat[0, 2] = torch.tensor([8.0, 1.0, -4.0, 0.0, 1.0, 0.0, 4.0, 2.0, 1.0])
    state.od_valid[0, 2] = 1.0
    view = readout(state)
    slots = state.od_feat[0].numpy()
    presence = state.od_valid[0].numpy()
    truth = od_truth_slots(slots, presence)
    assert float(view["fields"]["dynamic_distance"][0]) == pytest.approx(
        float(truth["dyn_distance"]), abs=1e-4
    )
    assert float(view["fields"]["dynamic_ttc"][0]) == pytest.approx(float(truth["ttc"]), abs=1e-4)
    assert float(view["present"]["dynamic_distance"][0]) == 1.0
    # TTC 口径：cap 60 + present 恒 1（env 公式同源）
    no_slots = _synthetic_state(batch=1, od_slots=1)
    no_slots.od_valid[:] = 0.0
    no_slots.od_anchor[:] = 0.0
    view_empty = readout(no_slots)
    assert float(view_empty["fields"]["dynamic_ttc"][0]) == pytest.approx(60.0, abs=1e-6)
    assert float(view_empty["fields"]["dynamic_distance"][0]) == 0.0
    assert float(view_empty["present"]["dynamic_distance"][0]) == 0.0
    assert view_empty["sources"]["dynamic_distance"] == {"missing": 1}


def test_readout_ld_ego_motion_compensation() -> None:
    state = _synthetic_state(batch=1)
    state.ld_anchor[0, 0] = torch.tensor([10.0, 2.0, 0.1, 0.01])
    state.ego_pose = torch.tensor([[3.0, 1.0, 0.5]])
    view = readout(state)
    ld_ego = view["diagnostics"]["ld_ego_k"][0, 0]
    theta = 0.5
    px = 10.0 - 3.0
    py = 2.0 - 1.0
    cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    assert float(ld_ego[0]) == pytest.approx(float(cos_t * px + sin_t * py), abs=1e-5)
    assert float(ld_ego[1]) == pytest.approx(float(-sin_t * px + cos_t * py), abs=1e-5)
    assert float(ld_ego[2]) == pytest.approx(0.1 - theta, abs=1e-5)
    assert float(ld_ego[3]) == pytest.approx(0.01, abs=1e-6)


# =========================================================================== #
# 4. 残差训练接口（one-step；本 lane 不训练）
# =========================================================================== #
def test_residual_interface_one_step_supervision() -> None:
    analytic_next = torch.zeros(2, 3, 5)
    real_next = torch.ones(2, 3, 5) * 2.0
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    target = residual_targets(analytic_next, real_next, valid)
    assert torch.allclose(target[0, 0], torch.full((5, ), 2.0))
    assert torch.equal(target[0, 2], torch.zeros(5))  # 无效槽不计入
    residual = target.clone().requires_grad_(True)
    out = residual_loss(residual, analytic_next, real_next, valid)
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(out["target"], target)

    shifted = (target + 0.5).requires_grad_(True)
    out2 = residual_loss(shifted, analytic_next, real_next, valid, mode="l1")
    assert float(out2["loss"]) > 0.0
    out2["loss"].backward()
    assert shifted.grad is not None and float(shifted.grad.abs().sum()) > 0.0
    with pytest.raises(ValueError):
        residual_loss(torch.zeros(2, 3, 4), analytic_next, real_next, valid)
    with pytest.raises(ValueError):
        residual_targets(analytic_next, torch.full((2, 3, 5), float("nan")), valid)


def test_od_decode_prior_mirrors_stgnn_prior() -> None:
    torch.manual_seed(0)
    from net.st_gnn import SpatioTemporalGNN

    gnn = SpatioTemporalGNN(hidden=32, od_slots=16, ld_slots=16, steps=6)
    gnn.eval()
    anchor = torch.randn(2, 16, 5)
    anchor[..., 0] *= 5.0
    with torch.no_grad():
        out = gnn(
            z_ego=torch.randn(2, 32), z_od=torch.randn(2, 16, 32), z_ld=torch.randn(2, 16, 32),
            node_mask=torch.cat([torch.ones(2, 1), torch.ones(2, 32)], dim=1),
            pose=torch.zeros(2, 33, 3), step_index=3, od_anchor=anchor, ld_anchor=None,
        )
    prior = od_decode_prior(anchor, step=3, dt=0.5)
    # 未训练 od_head 输出层零初始化 ⇒ od_pred == prior（公式镜像自证）
    assert torch.allclose(out[3], prior, atol=1e-6)


# =========================================================================== #
# 5. analytic_residual：zero-init 等价 / 扰动分离
# =========================================================================== #
def test_analytic_residual_zero_init_equals_analytic_and_perturb_separates() -> None:
    model = _tiny_structured()
    obs = _obs_structured()
    model.set_world_state_api("analytic")
    with torch.no_grad():
        out_an = model(obs)
    model.set_world_state_api("analytic_residual")
    with torch.no_grad():
        out_ar = model(obs)
    for key in out_an:
        if torch.is_tensor(out_an[key]) and out_an[key].dtype.is_floating_point:
            assert torch.equal(out_an[key], out_ar[key]), f"zero-init 残差应严格 0：{key}"
    # 扰动 st_gnn 解码头（模拟已训练 WM）⇒ 残差真实进入 readout；
    # 同时解冻纵向头末层（fresh 模型零初始化 ⇒ 几何特征本不影响动作）
    with torch.no_grad():
        generator = torch.Generator().manual_seed(1)
        odd = model.st_gnn.od_head[-1]
        odd.weight.copy_(torch.randn(odd.weight.shape, generator=generator, dtype=odd.weight.dtype) * 0.05)
        long_out = model.path_head.longitudinal[-1]
        long_out.weight.copy_(
            torch.randn(long_out.weight.shape, generator=generator, dtype=long_out.weight.dtype) * 0.05
        )
    model.set_world_state_api("analytic")
    with torch.no_grad():
        out_an2 = model(obs)
    model.set_world_state_api("analytic_residual")
    with torch.no_grad():
        out_ar2 = model(obs)
    plan_diff = (out_an2["plan"] - out_ar2["plan"]).abs().max().item()
    assert plan_diff > 0.0, "analytic_residual 残差未进入 readout"
    # 残差在 t0 h1 步不生效（a0 来自 encode；残差进 k≥1 步）
    assert torch.equal(out_an2["plan"][:, 0], out_ar2["plan"][:, 0])
    assert not torch.equal(out_an2["plan"][:, 1], out_ar2["plan"][:, 1])
    model.set_world_state_api("off")


# =========================================================================== #
# 6. hooks：producer 唯一 / held 白名单 / writeback=0
# =========================================================================== #
def test_world_state_hooks_checks_and_violation_detection() -> None:
    model = _tiny_structured()
    probes = ProbeRecorder(enabled=False)
    probes.world_state.enabled = True
    model.set_probes(probes)
    model.set_world_state_api("analytic")
    with torch.no_grad():
        model(_obs_structured())
    summary = probes.world_state.summary()
    assert summary["records"] == 7  # t0 + 5 想象步 + 末步（k=6 无头调用但有 readout）
    assert summary["steps"] == [0, 1, 2, 3, 4, 5, 6]
    assert all(summary["checks"].values()), summary
    assert summary["violations"] == []
    assert summary["field_producers"] == FIELD_PRODUCERS
    assert summary["checks"]["imagined_writeback_zero"] is True
    # 篡改 producer ⇒ violations 非空
    probes.world_state.records[1]["producers"]["speed"] = "rogue"
    bad = probes.world_state.check()
    assert bad["checks"]["producer_unique"] is False and bad["violations"]
    probes.world_state.records[1]["producers"]["speed"] = FIELD_PRODUCERS["speed"]
    # 违规 held（白名单外字段）⇒ violations 非空
    probes.world_state.records[1]["sources"]["speed"] = {"held": 1}
    bad2 = probes.world_state.check()
    assert bad2["checks"]["held_whitelist"] is False and bad2["violations"]
    # writeback 非 0 ⇒ violations
    probes.world_state.records[1]["sources"]["speed"] = {"analytic": 1}
    probes.world_state.records[-1]["memory"]["writeback"] = 1
    bad3 = probes.world_state.check()
    assert bad3["checks"]["imagined_writeback_zero"] is False
    model.set_world_state_api("off")


def test_world_state_hooks_disabled_by_default_zero_overhead() -> None:
    hooks = WorldStateHooks(enabled=False)
    state = _synthetic_state(batch=1)
    assert hooks.capture(step=1, state=state, view=readout(state)) is None
    assert hooks.records == [] and hooks.capture_calls == 0


# =========================================================================== #
# 7. Imagined Memory 隔离 / 销毁
# =========================================================================== #
def test_imagined_memory_isolation_and_destroy() -> None:
    memory = ImaginedMemory(batch=1)
    source = torch.ones(1, 3)
    frame = memory.clone_frame(
        step=0, source="real", ego_pose=source, ego_speed=source[:, 0],
        od_anchor=torch.zeros(1, 2, 5), od_valid=torch.ones(1, 2),
        ld_anchor=torch.zeros(1, 2, 4), ld_valid=torch.ones(1, 2),
        lane_anchor=torch.zeros(1, 4, 2),
    )
    source.fill_(5.0)  # 改源张量不影响记录（克隆副本）
    assert torch.equal(frame.ego_pose, torch.ones(1, 3))
    assert memory.writeback == 0
    assert memory.destroy().destroyed is True and memory.frames == []
    with pytest.raises(RuntimeError):
        memory.push(frame)
    assert memory.summary()["writeback"] == 0


def test_rollout_destroys_imagined_memory_and_keeps_real_mem_untouched() -> None:
    model = _tiny_structured()
    model.set_world_state_api("analytic")
    obs = _obs_structured()
    od_before = obs["od_hist"].clone()
    with torch.no_grad():
        model(obs)
    memory = model._world_state.memory
    assert memory.destroyed is True and memory.writeback == 0
    assert torch.equal(obs["od_hist"], od_before)  # Real Memory（obs mem）零写回
    model.set_world_state_api("off")


def test_world_state_api_modes_contract() -> None:
    assert WORLD_STATE_API_MODES == ("off", "analytic", "analytic_residual", "analytic_learned")
    assert tuple(POLICY_STATE_FIELDS) == tuple(LONG_EXOG_FEATURE_NAMES)
    assert set(FIELD_PRODUCERS) == set(POLICY_STATE_FIELDS)
    assert HELD_ALLOWED == ("nav_progress",)


def test_world_state_hooks_missing_rows_are_per_row() -> None:
    """逐行 missing 断言（同字段混源合法：missing 行的 present 必须 0）。"""
    hooks = WorldStateHooks(enabled=True)
    fields = {name: torch.zeros(2) for name in POLICY_STATE_FIELDS}
    present = {name: torch.ones(2) for name in POLICY_STATE_FIELDS}
    missing_row = POLICY_STATE_FIELDS.index("dynamic_distance")
    present["dynamic_distance"] = torch.tensor([0.0, 1.0])
    codes = torch.zeros(2, len(POLICY_STATE_FIELDS), dtype=torch.long)
    codes[:, missing_row] = torch.tensor([SOURCE_CODES["missing"], SOURCE_CODES["analytic"]])
    record = {
        "step": 1,
        "fields": fields,
        "present": present,
        "sources": {"dynamic_distance": {"missing": 1, "analytic": 1}},
        "producers": dict(FIELD_PRODUCERS),
        "source_codes": codes,
        "producer_assert_ok": True,
    }
    hooks._append(record)
    assert hooks.check()["checks"]["missing_implies_absent"] is True
    record["present"]["dynamic_distance"] = torch.ones(2)  # 反例：missing 行 present>0
    assert hooks.check()["checks"]["missing_implies_absent"] is False
