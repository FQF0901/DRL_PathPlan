"""R38-A4：想象步几何口径开关（``net.path_head.set_step_geometry``）单测。

覆盖两条确定性几何缺陷（出处 = ``.slim/deepwork/p1_v2/round38/a3_report.md`` §2.3/§2.4）：

- **① 零填充锚 foot 伪影**：``production`` 下重锚 foot 搜索不看 ``anchor_mask`` ⇒ 零填充锚
  （padding = ``(0,0)``）的"桥接段"穿过自车 ⇒ foot 落 padding、锚几何整段零化（
  ``anchor_reference.valid = 0``）；``fixed`` ⇒ foot 落回真实车道段、锚几何有效。
- **② margin 左右反号**：数据集口径（I1 ``_long_exog_row``）``left = W/2 + lat``、
  ``right = W/2 − lat``；``production`` 互换，``fixed`` 对齐（恒等位姿下 == t0 row 值）。

另含：默认 ``production`` 与旧公式逐位相等（回归护栏）、无缺陷输入下 ``fixed == production``、
开关状态机（非法取值 fail-fast / ``geometry_mode=None`` 读进程级全局 / 键集不变）。
"""

from __future__ import annotations

import pytest
import torch

from net.path_head import (
    STEP_GEOMETRY_DEFAULT_MODE,
    STEP_GEOMETRY_MODES,
    anchor_reference,
    get_step_geometry,
    set_step_geometry,
    step_geometry,
)

P = 8          # 锚折线点数（含零填充尾巴）
N_VALID = 5    # 有效点数
LANE_Y = 0.285  # 车道中心线相对自车横向偏移（ego t0 系 y；lat = −y 左正）
LANE_WIDTH = 10.5  # W = left0 + right0（t0 静态地图）


@pytest.fixture(autouse=True)
def _restore_mode():
    """模块级全局：每个测试后恢复默认（防跨测试污染）。"""
    set_step_geometry(STEP_GEOMETRY_DEFAULT_MODE)
    yield
    set_step_geometry(STEP_GEOMETRY_DEFAULT_MODE)


def _padded_anchor(*, y: float = LANE_Y, pad: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """车道锚：``x = 0,3,...,12``、``y = const``；尾部 ``pad`` 个零填充点（mask = 0）。"""
    xy = torch.zeros(1, P, 2)
    n_valid = N_VALID
    xs = torch.arange(n_valid, dtype=torch.float32) * 3.0
    xy[0, :n_valid, 0] = xs
    xy[0, :n_valid, 1] = float(y)
    mask = torch.zeros(1, P)
    mask[0, :n_valid] = 1.0
    if not pad:
        # 无零填充：把尾巴点接在末有效点上（mask 仍 0；零长段 ⇒ 无桥接段）
        xy[0, n_valid:, 0] = float(xs[-1])
        xy[0, n_valid:, 1] = float(y)
    return xy, mask


def _exog_dataset_convention(*, lat0: float = -LANE_Y) -> tuple[torch.Tensor, torch.Tensor]:
    """t0 外源几何（列序 = LONG_EXOG_FEATURE_NAMES）；margin 按**数据集口径**：
    ``left = W/2 + lat``、``right = W/2 − lat``（lat = 自车相对中心线横向偏移，左正）。"""
    exog = torch.zeros(1, 8)
    exog[0, 1] = 0.5 * LANE_WIDTH + float(lat0)
    exog[0, 2] = 0.5 * LANE_WIDTH - float(lat0)
    exog[0, 3] = 40.0
    exog[0, 7] = 60.0
    present = torch.zeros(1, 8)
    present[0, [0, 1, 2, 3, 7]] = 1.0
    return exog, present


def _run(pose_xy=2.0, pose_y=0.1, pose_theta=0.05, progress=2.0, *, y=LANE_Y, pad=True, mode=None):
    anchor, mask = _padded_anchor(y=y, pad=pad)
    exog, present = _exog_dataset_convention()
    pose = torch.tensor([[float(pose_xy), float(pose_y), float(pose_theta)]])
    progress_s = torch.tensor([float(progress)])
    kwargs = {} if mode is None else {"geometry_mode": mode}
    return step_geometry(anchor, mask, exog, present, pose, progress_s, **kwargs), (anchor, mask, exog, present)


# --------------------------------------------------------------------------- #
# ① 零填充锚 foot 伪影
# --------------------------------------------------------------------------- #
def test_zero_padding_foot_artifact_production_vs_fixed() -> None:
    prod, _ = _run(mode="production")
    fixed, _ = _run(mode="fixed")

    # production：foot 被"末有效点 → (0,0) 桥接段"吸到 padding 端点（foot_index ≥ 有效点数−1）
    assert int(prod["foot_index"][0]) >= N_VALID - 1
    # ⇒ 锚几何整段零化：total_s ≈ 0 ⇒ anchor_reference.valid = 0
    ref_prod = anchor_reference(prod["anchor_xy"], prod["anchor_mask"])
    assert float(ref_prod["valid"][0]) == 0.0
    assert float(ref_prod["total_s"][0]) == 0.0
    assert float(ref_prod["lateral_offset"][0]) == 0.0

    # fixed：foot 排除 mask=0 段 ⇒ 落回真实车道段（严格位于有效段内）
    assert int(fixed["foot_index"][0]) < N_VALID - 1
    ref_fixed = anchor_reference(fixed["anchor_xy"], fixed["anchor_mask"])
    assert float(ref_fixed["valid"][0]) == 1.0
    assert float(ref_fixed["total_s"][0]) > 1.0
    # 横向偏移 = 自车相对车道中心线（左正）：想象位姿左偏 ⇒ −(0.285 − 0.1)·cos(θ)
    # （重锚后 lat 以想象航向为基准 ⇒ 恒等点积 = 点到线垂距 × cos(θ)）
    expected_lat = -(LANE_Y - 0.1) * float(torch.cos(torch.tensor(0.05)))
    assert float(fixed["lateral_offset"][0]) == pytest.approx(expected_lat, abs=1e-6)


def test_foot_never_on_padding_segment_when_mask_given() -> None:
    """fixed 口径下 foot 段必须两个端点都有效（随机位姿扫掠）。"""
    generator = torch.Generator().manual_seed(0)
    for _ in range(32):
        pose_x = float(torch.empty(1).uniform_(0.5, 14.0, generator=generator))
        pose_y = float(torch.empty(1).uniform_(-1.5, 1.5, generator=generator))
        pose_th = float(torch.empty(1).uniform_(-0.4, 0.4, generator=generator))
        out, (anchor, mask, _, _) = _run(
            pose_xy=pose_x, pose_y=pose_y, pose_theta=pose_th, progress=1.0, mode="fixed"
        )
        foot = int(out["foot_index"][0])
        assert bool(mask[0, foot] > 0.5) and bool(mask[0, foot + 1] > 0.5)


# --------------------------------------------------------------------------- #
# ② margin 左右反号
# --------------------------------------------------------------------------- #
def test_margin_sign_dataset_convention() -> None:
    """fixed = 数据集口径 ``left = W/2 + lat_k``、``right = W/2 − lat_k``；production 互换。"""
    prod, (_, _, exog, _) = _run(mode="production")
    fixed, _ = _run(mode="fixed")
    lat = float(fixed["lateral_offset"][0])
    width = float(exog[0, 1] + exog[0, 2])
    half = 0.5 * width
    assert lat != pytest.approx(0.0, abs=1e-6)  # 非对称行才有左右之分

    assert float(fixed["long_exog"][0, 1]) == pytest.approx(half + lat, abs=1e-5)
    assert float(fixed["long_exog"][0, 2]) == pytest.approx(half - lat, abs=1e-5)
    # production = 互换（同一 lat_k 下 left = W/2 − lat_k、right = W/2 + lat_k）
    assert float(prod["long_exog"][0, 1]) == pytest.approx(half - float(prod["lateral_offset"][0]), abs=1e-5)
    assert float(prod["long_exog"][0, 2]) == pytest.approx(half + float(prod["lateral_offset"][0]), abs=1e-5)
    # 注：零填充行上 production/fixed 的 lat_k 本身不同（foot 位置不同）⇒ 同一 lat 的左右对调
    # 由 test_margin_identity_pose_matches_t0_row_values（无 padding + 恒等位姿）覆盖。


def test_margin_identity_pose_matches_t0_row_values() -> None:
    """恒等位姿 + 零进度（无 padding 情形）⇒ fixed margin == t0 row 值（数据集口径）；
    production 把 row 的 left/right 互换（A3 §3.2 行级证据的口径）。"""
    anchor, mask = _padded_anchor(pad=False)
    anchor = anchor.clone()
    anchor[0, :N_VALID, 0] = torch.arange(N_VALID, dtype=torch.float32) * 3.0 + 0.5  # 首点不在原点
    exog, present = _exog_dataset_convention()  # t0 row 值（数据集口径）
    pose = torch.zeros(1, 3)
    progress_s = torch.zeros(1)
    prod = step_geometry(anchor, mask, exog, present, pose, progress_s, geometry_mode="production")
    fixed = step_geometry(anchor, mask, exog, present, pose, progress_s, geometry_mode="fixed")

    torch.testing.assert_close(fixed["long_exog"][:, 1:3], exog[:, 1:3], rtol=0, atol=1e-5)
    # production：互换（lat ≈ −0.285 ≠ 0 ⇒ 可见）
    assert float(prod["long_exog"][0, 1]) == pytest.approx(float(exog[0, 2]), abs=1e-5)
    assert float(prod["long_exog"][0, 2]) == pytest.approx(float(exog[0, 1]), abs=1e-5)


# --------------------------------------------------------------------------- #
# 默认口径 / 逐位护栏 / 状态机
# --------------------------------------------------------------------------- #
def test_default_mode_is_production_and_matches_legacy_formulas() -> None:
    """默认（不传 geometry_mode）= production：foot 无掩码 argmin + 旧 margin 公式（逐位）。"""
    assert get_step_geometry() == "production"
    out, (anchor, mask, exog, present) = _run(mode=None)
    # 旧公式复算（= 改动前 net/path_head.step_geometry 的逐行口径）
    pose = torch.tensor([[2.0, 0.1, 0.05]])
    theta = pose[:, 2]
    cos_t, sin_t = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
    dx = anchor[..., 0] - pose[:, 0:1]
    dy = anchor[..., 1] - pose[:, 1:2]
    xy_k = torch.stack([cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy], dim=-1)
    seg = xy_k[:, 1:] - xy_k[:, :-1]
    denom = (seg * seg).sum(dim=-1).clamp_min(1e-6)
    param = (-(xy_k[:, :-1] * seg).sum(dim=-1) / denom).clamp(0.0, 1.0)
    foot = xy_k[:, :-1] + seg * param.unsqueeze(-1)
    foot_index = (foot * foot).sum(dim=-1).argmin(dim=-1)
    keep = torch.arange(P).unsqueeze(0) > foot_index.unsqueeze(1)
    xy_expect = torch.where(
        keep.unsqueeze(-1), xy_k,
        foot.gather(1, foot_index.reshape(1, 1, 1).expand(-1, 1, 2)).expand_as(xy_k),
    )
    torch.testing.assert_close(out["anchor_xy"], xy_expect, rtol=0, atol=0)
    assert torch.equal(out["foot_index"], foot_index)
    lat_k = -xy_expect[..., 0, 1]
    half = 0.5 * (exog[:, 1] + exog[:, 2])
    torch.testing.assert_close(out["long_exog"][:, 1], (half - lat_k).clamp_min(0.0), rtol=0, atol=0)
    torch.testing.assert_close(out["long_exog"][:, 2], (half + lat_k).clamp_min(0.0), rtol=0, atol=0)


def test_fixed_equals_production_when_no_defect_input() -> None:
    """无零填充 + 对称车道（lat=0）⇒ 两处修复都不改变任何输出（键集/数值逐位）。"""
    prod, _ = _run(pose_y=0.0, y=0.0, pad=False, mode="production")
    fixed, _ = _run(pose_y=0.0, y=0.0, pad=False, mode="fixed")
    assert set(prod) == set(fixed)
    for key, value in prod.items():
        if torch.is_tensor(value):
            assert torch.equal(value, fixed[key]), key


def test_setter_state_machine_and_global_consumption() -> None:
    assert STEP_GEOMETRY_MODES == ("production", "fixed", "fixed_t0margins")
    assert set_step_geometry("fixed") == "production"
    assert get_step_geometry() == "fixed"
    out_global, _ = _run(mode=None)          # 读进程级全局
    out_explicit, _ = _run(mode="fixed")     # 显式覆盖
    assert torch.equal(out_global["anchor_xy"], out_explicit["anchor_xy"])
    assert torch.equal(out_global["long_exog"], out_explicit["long_exog"])
    with pytest.raises(ValueError):
        set_step_geometry("nope")
    with pytest.raises(ValueError):
        _run(mode="nope")
    assert set_step_geometry("production") == "fixed"
    assert get_step_geometry() == "production"
