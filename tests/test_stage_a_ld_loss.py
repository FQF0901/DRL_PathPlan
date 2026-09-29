"""Stage A 未来 LD 损失（ld_coef=0 新语义，2026-09-30 拍板）回归。

规格：A 的 WM 损失组合保留 ``weighted_ld_multi_step_loss``（``ld_fut`` 前 4 维 + ``ld_mask``/``wm_valid``，
与 OD 同构），但权重 ``stages.A.world_model.ld_coef`` **默认 0.0**——LD 损失照算（监控口径），
A **不监督 LD 头**（2026-09-30 用户拍板：闭环证据 ld=0 链（L2→基座 0.328→E-β′ 0.436）优于
ld=0.02 链（phase2 0.222））；非零权重（config/CLI）时监督接通、回传 ``st_gnn.ld_head``。

覆盖：

① A 训练中 LD 损失被计算（监控）/默认不更新 LD 头；非零 coef 回传（合成 v2 小数据 CPU smoke）；
② ``ld_mask`` / ``wm_valid`` 掩码目标不贡献（monkeypatch ``build_future`` 注入损坏，损失不变）；
③ ``ld_coef`` 权重可配（config dict / CLI），且只影响总损失、监控口径仍记录 ``wm_loss_ld``；
④ loss 函数自身掩码语义（unit：掩码槽位无梯度/不贡献，有效项照常）。

全部 CPU 小数据 smoke（秒级），产物落 tmp_path。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline import stages as stages_mod
from pipeline.stages import _parse_args, load_config, run_stage_a
from pipeline.trainer import BCDataset, save_checkpoint, weighted_ld_multi_step_loss
from tests.v2_synthetic import TINY_MODEL_YAML, write_v2_dataset


def _tiny_ckpt(tmp_path: Path) -> Path:
    from net.model import DrivingModel

    torch.manual_seed(0)
    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16)
    path = tmp_path / "init_stageA.pt"
    save_checkpoint(path, model, meta={"stage": "A", "kind": "test-init"})
    return path


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _run_a(tmp_path: Path, *, out_name: str, ckpt: Path, extra: "list[str] | None" = None, config=None):
    dataset_dir = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    args = _parse_args([
        "--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(tmp_path / out_name),
        "--ckpt", str(ckpt), "--model-config", str(_model_cfg(tmp_path)),
        "--wm-epochs", "1", "--batch-size", "8", "--val-frac", "0.34",
        "--device", "cpu", "--seed", "0", "--no-plan-noise", "--no-monitor",
        *(extra or []),
    ])
    return run_stage_a(args, config if config is not None else {})


# ---------------------------------------------- ① 计算（监控）+ 默认不监督 LD 头
def test_stage_a_ld_loss_computed_and_default_not_supervised(tmp_path: Path) -> None:
    ckpt = _tiny_ckpt(tmp_path)
    before = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    metrics = _run_a(tmp_path, out_name="stage_a", ckpt=ckpt)

    # 规格/口径
    assert metrics["ld_loss"] == "direct_multi_step"
    assert metrics["ld_coef"] == 0.0, "2026-09-30 拍板：默认不监督 LD 头"
    assert np.isfinite(metrics["wm_loss_ld"]) and metrics["wm_loss_ld"] > 0.0, "LD 损失必须被计算（监控）"
    assert np.isfinite(metrics["val_loss_ld"]) and metrics["val_loss_ld"] > 0.0
    for k in range(1, 7):
        item = metrics["per_horizon"][f"h{k}"]
        assert "ld_loss" in item, f"per_horizon h{k} 缺 ld_loss"
        assert item["ld_loss"] == item["ld_loss"], "ld_loss 不应是 NaN（有有效 LD 目标）"

    # 默认 ld_coef=0：无梯度回传 → ld_head 逐元素不变（LD 头未训练属预期）
    after = torch.load(tmp_path / "stage_a" / "final.pt", map_location="cpu", weights_only=False)["model"]
    ld_names = [name for name in after if name.startswith("st_gnn.ld_head.")]
    assert ld_names, "checkpoint 缺少 st_gnn.ld_head"
    assert all(torch.equal(before[name], after[name]) for name in ld_names), (
        "ld_coef=0 时 st_gnn.ld_head 不应被更新（监督已关闭）"
    )


# ---------------------------------------------- ② ld_mask / wm_valid 掩码不贡献
def test_stage_a_ld_loss_respects_ld_mask_and_wm_valid(tmp_path: Path, monkeypatch) -> None:
    ckpt = _tiny_ckpt(tmp_path)
    real_build_future = stages_mod.build_future
    mode = {"value": "none"}

    def _patched(arrays, indices, **kwargs):
        future = real_build_future(arrays, indices, **kwargs)
        ld_fut = np.array(future["ld_fut"], dtype=np.float32, copy=True)
        ld_mask = np.array(future["ld_mask"], dtype=np.float32, copy=True)
        wm_valid = np.array(future["wm_valid"], dtype=np.float32, copy=True)
        if mode["value"] == "masked":
            # 只在 ld_mask=0（含 wm_valid=0/缺帧）的位置注入垃圾：损失不得变化
            ld_fut[ld_mask < 0.5] += 100.0
        elif mode["value"] == "valid_gate":
            # wm_valid=0 的帧：强行把 ld_mask 置 1 再注入垃圾 → 仍应由 valid 门控排除
            zero_valid = wm_valid < 0.5
            ld_mask[zero_valid] = 1.0
            ld_fut[zero_valid] += 100.0
        elif mode["value"] == "hit":
            # 有效目标注入垃圾：损失必须变化
            hit_pos = (ld_mask > 0.5) & (wm_valid[:, :, None] > 0.5)
            ld_fut[hit_pos] += 100.0
        future = dict(future)
        future["ld_fut"] = ld_fut
        future["ld_mask"] = ld_mask
        future["wm_valid"] = wm_valid
        return future

    monkeypatch.setattr(stages_mod, "build_future", _patched)

    baseline = _run_a(tmp_path, out_name="run_base", ckpt=ckpt)
    assert baseline["wm_loss_ld"] > 0.0
    # 合成数据里确实存在被掩码的位置（frame_usable/wm_valid=0 / 缺帧）
    probe = real_build_future(
        BCDataset.load(str(tmp_path / "bc_v2")).arrays,
        np.arange(6, dtype=np.int64),
        episode_key="episode_id", step_key="step", future=6, stride=5,
        keys=None, alignments=None, wm_valid=None,
    )
    assert np.any(np.asarray(probe["ld_mask"]) < 0.5), "合成数据应存在被掩码的 LD 目标位置"

    mode["value"] = "masked"
    masked = _run_a(tmp_path, out_name="run_masked", ckpt=ckpt)
    assert float(masked["wm_loss_ld"]) == pytest.approx(float(baseline["wm_loss_ld"]), rel=1e-9), (
        "ld_mask=0 的目标注入垃圾不得影响 LD 损失"
    )
    assert float(masked["wm_loss"]) == pytest.approx(float(baseline["wm_loss"]), rel=1e-9)

    mode["value"] = "valid_gate"
    gated = _run_a(tmp_path, out_name="run_gated", ckpt=ckpt)
    assert float(gated["wm_loss_ld"]) == pytest.approx(float(baseline["wm_loss_ld"]), rel=1e-9), (
        "wm_valid=0 的帧（即使 ld_mask=1）不得贡献 LD 损失"
    )

    mode["value"] = "hit"
    hit = _run_a(tmp_path, out_name="run_hit", ckpt=ckpt)
    assert float(hit["wm_loss_ld"]) != pytest.approx(float(baseline["wm_loss_ld"]), rel=1e-3), (
        "有效 LD 目标注入垃圾必须改变损失（否则监督没接上）"
    )


# ---------------------------------------------- ③ ld_coef 可配（config / CLI）
def test_stage_a_ld_coef_config_and_cli(tmp_path: Path) -> None:
    ckpt = _tiny_ckpt(tmp_path)
    before = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    repo_cfg = load_config("config/default.yaml")
    assert repo_cfg["stages"]["A"]["world_model"]["ld_coef"] == 0.0, (
        "config 默认 0.0（2026-09-30 拍板：ld=0 链闭环优于 ld=0.02 链）"
    )

    base = _run_a(tmp_path, out_name="coef_default", ckpt=ckpt)
    half = _run_a(tmp_path, out_name="coef_half", ckpt=ckpt, extra=["--wm-ld-coef", "0.25"])
    assert base["ld_coef"] == 0.0 and half["ld_coef"] == 0.25
    # LD 项本身照算（监控口径），权重只改总损失组合：
    # val_loss = od + ld_coef·ld + ego_next_coef·ego_next + presence_coef·presence + entry_coef·entry
    for metrics, coef in ((base, 0.0), (half, 0.25)):
        assert float(metrics["wm_loss_ld"]) > 0.0, "coef=0 时 LD 损失仍应计算（监控）"
        expected = (
            float(metrics["val_loss_od"])
            + coef * float(metrics["val_loss_ld"])
            + 0.1 * float(metrics["ego_next_loss"])
            + 0.1 * float(metrics["presence_loss"])
            + 0.1 * float(metrics["entry_loss"])
        )
        assert float(metrics["val_loss"]) == pytest.approx(expected, rel=1e-6), (
            f"ld_coef={coef} 时总损失组合不符"
        )
    assert float(half["val_loss"]) > float(base["val_loss"]), "非零 ld_coef 的 val 总损失应多一个 LD 项"

    # 监督接通/关闭：非零 coef → LD 回传 st_gnn.ld_head；默认 0 → 头不变
    after_base = torch.load(
        tmp_path / "coef_default" / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    after_half = torch.load(
        tmp_path / "coef_half" / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    ld_names = [name for name in after_half if name.startswith("st_gnn.ld_head.")]
    assert ld_names, "checkpoint 缺少 st_gnn.ld_head"
    assert any(not torch.equal(before[name], after_half[name]) for name in ld_names), (
        "非零 ld_coef 时 LD 损失必须回传 st_gnn.ld_head（监督未接通？）"
    )
    assert all(torch.equal(before[name], after_base[name]) for name in ld_names), (
        "默认 ld_coef=0 时 st_gnn.ld_head 不应被更新"
    )

    # config dict 覆盖（非 CLI 路径）
    cfg = _run_a(
        tmp_path, out_name="coef_config", ckpt=ckpt,
        config={"stages": {"A": {"world_model": {"ld_coef": 0.25}}}},
    )
    assert cfg["ld_coef"] == 0.25


# ---------------------------------------------- ④ loss 函数掩码语义（unit）
def test_weighted_ld_multi_step_loss_mask_and_valid_gates() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 6, 16, 4, requires_grad=True)
    target = torch.randn(2, 6, 16, 4)
    mask = torch.ones(2, 6, 16)
    mask[:, :, 8:] = 0.0            # 槽位掩码
    valid = torch.ones(2, 6)
    valid[1, 4:] = 0.0              # 帧有效性（wm_valid）
    frame_weight = torch.tensor([1.0, 2.0])
    loss, per_horizon = weighted_ld_multi_step_loss(
        pred, target, mask, frame_weight=frame_weight, valid=valid
    )
    assert loss.ndim == 0 and torch.isfinite(loss) and len(per_horizon) == 6
    loss.backward()
    grad = pred.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad[:, :, :8, :].abs().sum()) > 0.0
    assert float(grad[:, :, 8:, :].abs().sum()) == 0.0, "ld_mask=0 槽位不得有梯度"
    assert float(grad[1, 4:, :, :].abs().sum()) == 0.0, "valid=0 帧不得有梯度"

    corrupted = target.clone()
    corrupted[:, :, 8:, :] += 100.0
    corrupted[1, 4:, :, :] += 100.0
    masked_loss, _ = weighted_ld_multi_step_loss(
        pred.detach(), corrupted, mask, frame_weight=frame_weight, valid=valid
    )
    assert float(masked_loss) == pytest.approx(float(loss.detach()), rel=1e-9)
    changed = target.clone()
    changed[0, 0, 0, :] += 1.0
    changed_loss, _ = weighted_ld_multi_step_loss(
        pred.detach(), changed, mask, frame_weight=frame_weight, valid=valid
    )
    assert float(changed_loss) != pytest.approx(float(loss.detach()), rel=1e-6)
