"""Stage A WM 监督（v8 B3）回归：latent consistency 主损失 + 物理解码诊断 + 掩码语义。

v8 口径（spec 冻结）：

- **latent consistency（主）**：``z_*`` 逐步 vs 未来帧编码目标（detach），掩码加权
  smooth_l1；默认权重 **1.0**（``--wm-latent-coef`` 覆盖）；
- **物理解码（诊断）**：OD 0.1 / LD 0.02（先验+残差，t0 帧；``--wm-od-coef`` /
  ``--wm-ld-coef`` 覆盖）——LD 物理项默认非零 ⇒ 回传 ``st_gnn.ld_head``；置 0 时仅监控；
- ``ld_mask`` / ``wm_valid`` 掩码目标不贡献（monkeypatch ``build_future`` 注入损坏，损失不变）；
- 权重组合：``val_loss = latent_coef·L_latent + od_coef·L_od + ld_coef·L_ld +
  ego_next_coef·L_ego_next + presence_coef·L_presence + entry_coef·L_entry``；
- loss 函数自身掩码语义（unit）。

全部 CPU 小数据 smoke（秒级），产物落 tmp_path。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline import stages as stages_mod
from pipeline.stages import _parse_args, run_stage_a
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


# ---------------------------------------------- ① 计算 + 默认权重 + LD 头受监督
def test_stage_a_latent_and_physical_losses_computed(tmp_path: Path) -> None:
    ckpt = _tiny_ckpt(tmp_path)
    before = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    metrics = _run_a(tmp_path, out_name="stage_a", ckpt=ckpt)

    # 规格/口径（v8 B3）
    assert metrics["ld_loss"] == "direct_multi_step"
    assert metrics["latent_coef"] == 1.0
    assert metrics["od_coef"] == 0.1
    assert metrics["ld_coef"] == 0.02
    assert np.isfinite(metrics["wm_loss_latent"]) and metrics["wm_loss_latent"] > 0.0
    assert np.isfinite(metrics["val_loss_latent"]) and metrics["val_loss_latent"] > 0.0
    assert np.isfinite(metrics["wm_loss_ld"]) and metrics["wm_loss_ld"] > 0.0, "LD 物理损失必须被计算"
    assert np.isfinite(metrics["val_loss_ld"]) and metrics["val_loss_ld"] > 0.0
    for k in range(1, 7):
        item = metrics["per_horizon"][f"h{k}"]
        assert "latent_loss" in item, f"per_horizon h{k} 缺 latent_loss"
        assert item["latent_loss"] == item["latent_loss"], "latent_loss 不应是 NaN"
        assert "ld_loss" in item, f"per_horizon h{k} 缺 ld_loss"
        assert item["ld_loss"] == item["ld_loss"], "ld_loss 不应是 NaN（有有效 LD 目标）"

    # 默认 ld_coef=0.02：物理 LD 项受监督 → ld_head 被更新
    after = torch.load(tmp_path / "stage_a" / "final.pt", map_location="cpu", weights_only=False)["model"]
    ld_names = [name for name in after if name.startswith("st_gnn.ld_head.")]
    assert ld_names, "checkpoint 缺少 st_gnn.ld_head"
    assert any(not torch.equal(before[name], after[name]) for name in ld_names), (
        "默认 ld_coef=0.02 时物理 LD 损失必须回传 st_gnn.ld_head"
    )
    # latent 转移头受监督（latent consistency 主损失）
    transition_names = [name for name in after if name.startswith(("st_gnn.od_transition.", "st_gnn.ld_transition."))]
    assert transition_names, "checkpoint 缺少 st_gnn.*_transition"
    assert any(not torch.equal(before[name], after[name]) for name in transition_names), (
        "latent consistency 未更新 st_gnn 转移头"
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


# ---------------------------------------------- ③ 权重可配（CLI）+ 组合口径
def test_stage_a_loss_weights_cli_and_composition(tmp_path: Path) -> None:
    ckpt = _tiny_ckpt(tmp_path)
    before = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]

    base = _run_a(tmp_path, out_name="coef_default", ckpt=ckpt)
    half = _run_a(
        tmp_path, out_name="coef_half", ckpt=ckpt,
        extra=["--wm-latent-coef", "0.5", "--wm-od-coef", "0.25", "--wm-ld-coef", "0.0"],
    )
    assert base["latent_coef"] == 1.0 and base["od_coef"] == 0.1 and base["ld_coef"] == 0.02
    assert half["latent_coef"] == 0.5 and half["od_coef"] == 0.25 and half["ld_coef"] == 0.0
    # 各项本身照算（监控口径），权重只改总损失组合：
    for metrics in (base, half):
        assert float(metrics["wm_loss_latent"]) > 0.0
        assert float(metrics["wm_loss_ld"]) > 0.0, "coef=0 时 LD 物理损失仍应计算（监控）"
        expected = (
            float(metrics["latent_coef"]) * float(metrics["val_loss_latent"])
            + float(metrics["od_coef"]) * float(metrics["val_loss_od"])
            + float(metrics["ld_coef"]) * float(metrics["val_loss_ld"])
            + float(metrics["ego_next_coef"]) * float(metrics["ego_next_loss"])
            + float(metrics["presence_coef"]) * float(metrics["presence_loss"])
            + float(metrics["entry_coef"]) * float(metrics["entry_loss"])
        )
        assert float(metrics["val_loss"]) == pytest.approx(expected, rel=1e-6), (
            "v8 总损失组合不符（latent/od/ld + ego_next + presence/entry）"
        )

    # 监督接通/关闭：ld_coef=0 → ld_head 不变；默认 0.02 → 更新
    after_base = torch.load(
        tmp_path / "coef_default" / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    after_half = torch.load(
        tmp_path / "coef_half" / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    ld_names = [name for name in after_base if name.startswith("st_gnn.ld_head.")]
    assert ld_names, "checkpoint 缺少 st_gnn.ld_head"
    assert any(not torch.equal(before[name], after_base[name]) for name in ld_names), (
        "默认 ld_coef=0.02 时物理 LD 损失必须回传 st_gnn.ld_head"
    )
    assert all(torch.equal(before[name], after_half[name]) for name in ld_names), (
        "--wm-ld-coef 0.0 时 st_gnn.ld_head 不应被更新（监督已关闭）"
    )


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


# ---------------------------------------------- ⑤ latent consistency loss（unit）
def test_weighted_latent_consistency_loss_masks_and_combined_denominator() -> None:
    """latent consistency：三路联合加权均值；掩码/valid 门控；目标改动影响损失。"""
    from pipeline.trainer import weighted_latent_consistency_loss

    torch.manual_seed(0)
    batch, horizon, slots, hidden = 2, 6, 4, 8
    od_pred = torch.randn(batch, horizon, slots, hidden, requires_grad=True)
    ld_pred = torch.randn(batch, horizon, slots, hidden, requires_grad=True)
    ego_pred = torch.randn(batch, horizon, hidden, requires_grad=True)
    od_target = torch.randn(batch, horizon, slots, hidden)
    ld_target = torch.randn(batch, horizon, slots, hidden)
    ego_target = torch.randn(batch, horizon, hidden)
    od_mask = torch.ones(batch, horizon, slots)
    od_mask[:, :, 2:] = 0.0
    ld_mask = torch.ones(batch, horizon, slots)
    valid = torch.ones(batch, horizon)
    valid[1, 4:] = 0.0
    frame_weight = torch.tensor([1.0, 2.0])
    loss, per_horizon = weighted_latent_consistency_loss(
        od_pred, od_target, od_mask, ld_pred, ld_target, ld_mask, ego_pred, ego_target,
        frame_weight=frame_weight, valid=valid,
    )
    assert loss.ndim == 0 and torch.isfinite(loss) and len(per_horizon) == 6
    loss.backward()
    assert float(od_pred.grad[:, :, :2].abs().sum()) > 0.0
    assert float(od_pred.grad[:, :, 2:].abs().sum()) == 0.0, "od_mask=0 槽位不得有梯度"
    assert float(ego_pred.grad[1, 4:].abs().sum()) == 0.0, "valid=0 帧不得有梯度"
    assert np.isfinite(per_horizon[0]["od_loss"]) and np.isfinite(per_horizon[0]["ld_loss"])
    assert np.isfinite(per_horizon[0]["ego_loss"])
    # 掩码外目标注入垃圾不影响；有效目标改动影响
    corrupted = od_target.clone()
    corrupted[:, :, 2:] += 100.0
    masked_loss, _ = weighted_latent_consistency_loss(
        od_pred.detach(), corrupted, od_mask, ld_pred.detach(), ld_target, ld_mask,
        ego_pred.detach(), ego_target, frame_weight=frame_weight, valid=valid,
    )
    assert float(masked_loss) == pytest.approx(float(loss.detach()), rel=1e-9)
    changed = ego_target.clone()
    changed[0, 0] += 1.0
    changed_loss, _ = weighted_latent_consistency_loss(
        od_pred.detach(), od_target, od_mask, ld_pred.detach(), ld_target, ld_mask,
        ego_pred.detach(), changed, frame_weight=frame_weight, valid=valid,
    )
    assert float(changed_loss) != pytest.approx(float(loss.detach()), rel=1e-6)
    # ego 目标缺省（None）→ 只算 od/ld，仍有限
    no_ego, _ = weighted_latent_consistency_loss(
        od_pred.detach(), od_target, od_mask, ld_pred.detach(), ld_target, ld_mask,
        frame_weight=frame_weight, valid=valid,
    )
    assert torch.isfinite(no_ego)
