"""R38-P2a：``tools.legacy.k_anchor``（K-anchor 快照）单测。

覆盖：快照可加载（不被生产导入）/ 锚字典加载与校验 / WTA 分配 / 软混合 / 车道系往返 /
legacy ``PlanHead(num_anchors=6)`` 前向键集与零残差不变量 / 包自检与拟合 CLI 快照可跑。

口径来源：原 ``tests/test_plan_anchor.py`` §1-4（R38-P2a 迁移；模型集成与损失接线用例
随主链删除，改为 ``tests/test_r38_p2a_k_anchor_removed.py`` 的移除守卫）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.legacy.k_anchor.anchor import (
    ANCHOR_FILE_DEFAULT,
    ANCHOR_SEMANTICS,
    K_DEFAULT,
    anchor_lane_context,
    anchor_mixture,
    anchors_from_payload,
    assign_anchors,
    assigned_plan,
    default_anchors,
    from_lane_dtheta,
    lane_follow_dtheta,
    load_anchor_dictionary,
    mixture_probabilities,
    to_lane_dtheta,
)
from tools.legacy.k_anchor.plan_head import PlanHead

ROOT = Path(__file__).resolve().parents[1]
K_ANCHOR_DIR = ROOT / "tools" / "legacy" / "k_anchor"
ANCHOR_FILE = K_ANCHOR_DIR / "plan_anchors_k6.json"
BATCH = 4


# ---------------------------------------------------------------------- 快照加载
def test_snapshot_module_paths_and_default_file() -> None:
    """快照文件齐备；``ANCHOR_FILE_DEFAULT`` 仍指向退役前的 ``config/`` 路径（逐行拷贝口径）。"""
    for name in ("anchor.py", "plan_head.py", "fit_plan_anchors.py", "plan_anchors_k6.json", "README.md"):
        assert (K_ANCHOR_DIR / name).exists(), name
    assert ANCHOR_FILE_DEFAULT == "config/plan_anchors_k6.json"  # 快照常量不改写（仅文档化退役）
    assert len(ANCHOR_SEMANTICS) == K_DEFAULT


def test_selfcheck_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tools.legacy.k_anchor.selfcheck"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


def test_fit_cli_snapshot_defaults_to_legacy_dir() -> None:
    """拟合 CLI 快照可运行；``--out`` 默认已改为 legacy 目录（唯一改动，模块 docstring 标注）。"""
    proc = subprocess.run(
        [sys.executable, "-m", "tools.legacy.k_anchor.fit_plan_anchors", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    source = (K_ANCHOR_DIR / "fit_plan_anchors.py").read_text(encoding="utf-8")
    assert 'default="tools/legacy/k_anchor/plan_anchors_k6.json"' in source
    assert 'default="config/plan_anchors_k6.json"' not in source


# ---------------------------------------------------------------------- 1. 锚字典
def test_anchor_file_loads_and_matches_default_shape() -> None:
    tensor = load_anchor_dictionary(str(ANCHOR_FILE), expected_k=K_DEFAULT)
    assert tuple(tensor.shape) == (K_DEFAULT, 6, 2)
    assert bool(torch.isfinite(tensor).all())
    assert bool((tensor[..., 0] >= 0.0).all())
    payload = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    assert int(payload["k"]) == K_DEFAULT
    assert payload["feature"] == "cumdtheta_lane"
    assert len(payload["anchors"]) == K_DEFAULT
    # 内置默认与文件同源（截断精度内一致；文件缺失时回退内置）
    default = default_anchors(expected_k=K_DEFAULT)
    assert torch.allclose(tensor, default, atol=5e-4)


def test_anchor_missing_file_falls_back_to_default(tmp_path: Path) -> None:
    tensor = load_anchor_dictionary(str(tmp_path / "missing.json"), expected_k=K_DEFAULT)
    assert torch.allclose(tensor, default_anchors(), atol=0.0)


def test_anchor_invalid_payload_rejected() -> None:
    with pytest.raises(ValueError):
        anchors_from_payload({"anchors": [{"ds": [1.0] * 6, "dtheta_lane": [0.0] * 5}]})
    bad = {"anchors": [{"ds": [1.0] * 6, "dtheta_lane": [0.0] * 6} for _ in range(2)]}
    with pytest.raises(ValueError):
        anchors_from_payload(bad, expected_k=6)
    nonfinite = {
        "anchors": [{"ds": [float("nan")] * 6, "dtheta_lane": [0.0] * 6} for _ in range(6)]
    }
    with pytest.raises(ValueError):
        anchors_from_payload(nonfinite)


# ---------------------------------------------------------------------- 2. WTA
def test_assign_anchors_picks_nearest_shape() -> None:
    anchors = default_anchors()
    batch = 5
    index = torch.tensor([0, 1, 2, 3, 4])
    chain = torch.stack(
        [torch.stack([torch.full((6,), 3.5), anchors[k, :, 1]], dim=-1) for k in index], dim=0
    )
    valid = torch.ones(batch, 6)
    result = assign_anchors(
        chain, anchors[..., 1], valid, torch.zeros(batch), torch.zeros(batch), torch.ones(batch)
    )
    assert torch.equal(result["index"], index)
    assert bool(result["row_valid"].all())
    assert float(result["dist"].max()) < 1e-5
    invalid = assign_anchors(
        chain, anchors[..., 1], torch.zeros(batch, 6),
        torch.zeros(batch), torch.zeros(batch), torch.ones(batch),
    )
    assert not bool(invalid["row_valid"].any())
    assert bool((invalid["index"] == 0).all())


def test_assign_anchors_masks_steps() -> None:
    anchors = default_anchors()
    chain = torch.stack([torch.full((6,), 3.5), anchors[3, :, 1]], dim=-1).unsqueeze(0)
    valid = torch.zeros(1, 6)
    valid[:, 0] = 1.0
    result = assign_anchors(
        chain, anchors[..., 1], valid, torch.zeros(1), torch.zeros(1), torch.ones(1)
    )
    assert int(result["index"][0]) == 3
    assert bool(result["row_valid"][0])


# ---------------------------------------------------------------------- 3. 软混合
def test_mixture_sharp_temperature_and_hard_argmax() -> None:
    anchors = default_anchors()
    batch = 3
    logits = torch.zeros(batch, K_DEFAULT)
    logits[:, 4] = 10.0
    residual = torch.zeros(batch, K_DEFAULT, 6, 2)
    speed = torch.full((batch, 6), 3.5)
    zeros = torch.zeros(batch)
    probs = mixture_probabilities(logits, temperature=0.01)
    assert bool((probs.argmax(-1) == 4).all())
    assert float(probs.max()) > 0.999
    plan_soft, _ = anchor_mixture(
        logits, residual, anchors[..., 1], speed, zeros, zeros, torch.ones(batch),
        temperature=0.01,
    )
    hard = torch.zeros(batch, K_DEFAULT)
    hard[:, 4] = 1.0
    plan_hard, probs_hard = anchor_mixture(
        logits, residual, anchors[..., 1], speed, zeros, zeros, torch.ones(batch),
        temperature=0.01, hard=True,
    )
    assert torch.equal(probs_hard, hard)
    assert torch.allclose(plan_soft, plan_hard, atol=1e-3)
    expected = torch.stack([speed, anchors[4, :, 1].expand(batch, 6)], dim=-1)
    assert torch.allclose(plan_hard, expected, atol=1e-5)


def test_mixture_uniform_equals_anchor_mean() -> None:
    anchors = default_anchors()
    batch = 2
    logits = torch.zeros(batch, K_DEFAULT)
    residual = torch.zeros(batch, K_DEFAULT, 6, 2)
    speed = torch.full((batch, 6), 3.5)
    zeros = torch.zeros(batch)
    plan, probs = anchor_mixture(
        logits, residual, anchors[..., 1], speed, zeros, zeros, torch.ones(batch)
    )
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / K_DEFAULT))
    expected_dth = anchors[..., 1].mean(dim=0).expand(batch, 6)
    assert torch.allclose(plan[..., 1], expected_dth, atol=1e-5)
    assert torch.allclose(plan[..., 0], speed, atol=1e-5)


def test_assigned_plan_uses_only_indexed_anchor() -> None:
    anchors = default_anchors()
    residual = torch.randn(2, K_DEFAULT, 6, 2) * 0.1
    speed = torch.full((2, 6), 3.5)
    index = torch.tensor([1, 5])
    plan = assigned_plan(
        index, residual, anchors[..., 1], speed, torch.zeros(2), torch.zeros(2), torch.ones(2)
    )
    row = torch.arange(2)
    expected_dth = anchors[index][..., 1] + residual[row, index][..., 1]
    assert torch.allclose(plan[..., 1], expected_dth, atol=1e-5)
    assert torch.allclose(plan[..., 0], speed + residual[row, index][..., 0], atol=1e-5)


# ---------------------------------------------------------------------- 4. lane 帧
def test_lane_frame_roundtrip_and_identity() -> None:
    torch.manual_seed(0)
    batch = 6
    ds = torch.rand(batch, 6) * 3.0 + 1.0
    chain = torch.randn(batch, 6, 2)
    chain[..., 0] = ds
    heading_err = torch.randn(batch) * 0.1
    curvature = torch.randn(batch) * 0.01
    valid = torch.ones(batch)
    lane = to_lane_dtheta(chain, heading_err, curvature, valid)
    back = from_lane_dtheta(lane, ds, heading_err, curvature, valid)
    assert torch.allclose(back, chain, atol=1e-6)
    lane_invalid = to_lane_dtheta(chain, heading_err, curvature, torch.zeros(batch))
    assert torch.allclose(lane_invalid, chain, atol=0.0)
    follow = lane_follow_dtheta(ds, torch.full((batch,), 0.2), torch.zeros(batch))
    assert torch.allclose(follow[:, 0], torch.full((batch,), 0.2), atol=1e-6)
    assert torch.allclose(follow[:, 1:], torch.zeros(batch, 5), atol=1e-6)
    kappa = torch.full((batch,), 0.05)
    follow_k = lane_follow_dtheta(ds, torch.zeros(batch), kappa)
    assert torch.allclose(follow_k[:, 1:], kappa.reshape(-1, 1) * ds[:, 1:], atol=1e-6)


def test_anchor_lane_context_validity() -> None:
    lane = torch.zeros(3, 17)
    lane[:, 1] = torch.tensor([0.1, -0.2, 0.3])
    lane[:, 3] = torch.tensor([0.01, 0.02, -0.01])
    lane[:, 14] = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([[1.0], [1.0], [0.0]])
    ctx = anchor_lane_context(lane, mask)
    assert torch.allclose(ctx[:, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(ctx[:, 0], lane[:, 1])


# ---------------------------------------------------------------------- 5. legacy PlanHead
def _anchor_head() -> PlanHead:
    torch.manual_seed(0)
    head = PlanHead(hidden=32, num_experts=2, expert_hidden=16, num_anchors=K_DEFAULT)
    head.set_anchors(load_anchor_dictionary(str(ANCHOR_FILE), expected_k=K_DEFAULT))
    return head.eval()


def test_legacy_plan_head_returns_anchor_keys_and_zero_residual() -> None:
    head = _anchor_head()
    latent = torch.randn(BATCH, 32)
    lane_ctx = torch.zeros(BATCH, 3)  # v8 后 lane 通道删除 → 恒 zeros（valid=0 → 恒等）
    with torch.no_grad():
        out = head.plan_anchors(latent, lane_ctx)
    assert tuple(out["logits"].shape) == (BATCH, K_DEFAULT)
    assert tuple(out["probs"].shape) == (BATCH, K_DEFAULT)
    assert tuple(out["plan"].shape) == (BATCH, 6, 2)
    assert tuple(out["speed_ds"].shape) == (BATCH, 6)
    assert tuple(out["residual"].shape) == (BATCH, K_DEFAULT, 6, 2)
    assert tuple(out["ctx"].shape) == (BATCH, 3)
    assert bool(torch.isfinite(out["plan"]).all())
    assert float(out["residual"].abs().max()) < 1e-6  # 残差末层零初始化
    # 零残差 + lane_ctx 恒零 ⇒ plan = (speed_ds, Σ p_k·anchor_dθ_k)
    mix_dth = torch.einsum("bk,kd->bd", out["probs"], head.anchor_dth)
    expected = from_lane_dtheta(
        torch.stack([out["speed_ds"], mix_dth], dim=-1),
        out["speed_ds"], lane_ctx[:, 0], lane_ctx[:, 1], lane_ctx[:, 2],
    )
    assert torch.allclose(out["plan"], expected, atol=1e-5)


def test_legacy_plan_head_disabled_has_no_anchor_keys() -> None:
    """``num_anchors=0``（历史默认关）⇒ 无锚参数/无锚 buffer（旧 ckpt 键集逐位一致）。"""
    head = PlanHead(hidden=32, num_experts=2, expert_hidden=16)
    names = [name for name, _ in head.state_dict().items()]
    assert not any(name.startswith(("anchor", "speed_head", "residual_head")) for name in names)
    with pytest.raises(RuntimeError):
        head.plan_anchors(torch.randn(1, 32), torch.zeros(1, 3))
    with pytest.raises(RuntimeError):
        head.set_anchors(default_anchors())
