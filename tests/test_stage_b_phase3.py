"""stage B phase 3（迭代恢复训练）回归（lane P3-B）。

覆盖（用户定稿契约）：

① 数据集 = 当轮 dagger 目录**单独使用**（无 5k 行；无 worst/mild 权重 / 无 mining）；
② 全参数可训（含 WM；``frozen_params=0``）且主干/WM/专家/policy 真实更新；
③ LR 分组：base 主干 ×0.25 / experts+gate+residual_scale ×0.5；
④ 动作链多步目标（``plan`` 第 2..6 步 ↔ 未来 t+1..t+5 专家首步标签）与尾部逐帧掩码；
⑤ LD 恢复损失：形状 / 可回传 / ``ld_mask``×``valid`` 生效；
⑥ ``traj_aux=0``：traj 只做监控 —— 破坏 ``traj6`` 不影响任何损失项（监控曲线仍变）；
⑦ phase-3 入口（``tools/train.py --phase3``）合成小数据 smoke + 监控 Tier-1 tag；
⑧ 未来 OD/LD 目标落在 **t0 自车系**（平移+旋转合成帧断言；lane 路径与本地等价路径一致）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline.stages import (
    FrameWindows,
    _BC_STEP_STRIDE,
    _bc_channel_keys,
    _parse_args,
    _phase3_future_fn,
    phase3_action_chain_targets,
    run_stage_b_phase3,
)
from pipeline.trainer import (
    BCDataset,
    Phase3Config,
    build_phase3_optimizer,
    pretrain_bc_phase3,
    save_checkpoint,
    weighted_action_chain_loss,
    weighted_ld_multi_step_loss,
)
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _tiny_model():
    from net.model import DrivingModel

    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=8, expert_hidden=16)


def _write_arrays(directory: Path, arrays) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "expert_bc.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    (directory / "expert_bc.meta.json").write_text(
        json.dumps({"schema_version": 2, "history_stride": 5}, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _init_ckpt(tmp_path: Path) -> Path:
    path = tmp_path / "init.pt"
    save_checkpoint(path, _tiny_model(), meta={"stage": "B", "phase": "phase2"})
    return path


def _future_fn_for(dataset: BCDataset):
    arrays = dataset.arrays
    windows = FrameWindows(
        arrays,
        dataset.alignments,
        episode_key="episode_id",
        step_key="step",
        keys=_bc_channel_keys(),
        step_stride=_BC_STEP_STRIDE,
    )
    wm_valid = np.asarray(arrays["wm_valid"], dtype=np.float32) if "wm_valid" in arrays else None
    return _phase3_future_fn(dataset, windows, wm_valid)


def _csv_tags(directory: Path) -> dict:
    path = directory / "monitor" / "metrics.csv"
    assert path.exists(), f"监控 CSV 缺失：{path}"
    tags: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tags[row["tag"]] = float(row["value"])
    return tags


# ------------------------------------------------- ② 全参数可训（含 WM 真更新）
def test_phase3_all_parameters_trainable_and_updated() -> None:
    arrays, _ = make_v2_arrays(episodes=4, steps_per_episode=6)
    dataset = BCDataset(arrays, {"schema_version": 2, "label_names": None})
    model = _tiny_model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    cfg = Phase3Config(epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0)
    metrics = pretrain_bc_phase3(
        model, dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )
    assert metrics["frozen_params"] == 0, "phase 3 必须全参数可训（无 frozen）"
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert metrics["trainable_params"] == sum(1 for _ in model.parameters())
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    for prefix in ("encoders.", "mem_encoder.", "plan_head.moe.experts.", "st_gnn.", "policy."):
        assert any(name.startswith(prefix) for name in changed), f"{prefix} 未被更新（全参数解冻失效？）"


# ------------------------------------------------------------- ③ LR 分组
def test_phase3_lr_groups_scale_base_and_specific() -> None:
    model = _tiny_model()
    optimizer = build_phase3_optimizer(model, 1e-3, base_scale=0.25, specific_scale=0.5)
    assert [group.get("name") for group in optimizer.param_groups] == ["base", "specific"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-4)

    base_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    specific_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert not (base_ids & specific_ids), "参数组必须互斥"
    specific_names = {
        name for name, parameter in model.named_parameters() if id(parameter) in specific_ids
    }
    expected_specific = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(("plan_head.moe.experts.", "plan_head.moe.router.", "plan_head.moe.residual_scale"))
    }
    assert specific_names == expected_specific and expected_specific
    base_names = {name for name, parameter in model.named_parameters() if id(parameter) in base_ids}
    for prefix in ("encoders.", "plan_head.moe.primary.", "st_gnn.", "policy.", "value."):
        assert any(name.startswith(prefix) for name in base_names), f"{prefix} 应在 base 主干组"
    assert len(base_names) + len(specific_names) == len(list(model.named_parameters()))

    # 未解冻（全冻结）→ 拒绝静默退化
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="LR 分组为空"):
        build_phase3_optimizer(model, 1e-3)


# ----------------------------------------------- ④ 动作链目标 + 尾部掩码
def test_phase3_action_chain_targets_and_tail_mask() -> None:
    arrays = {
        "episode_id": np.asarray([0, 0, 0], dtype=np.int64),
        "step": np.asarray([0, 5, 15], dtype=np.int64),  # step 10 缺失（洞）
        "action": np.zeros((3, 6, 2), dtype=np.float32),
    }
    arrays["action"][:, 0, 0] = [1.0, 2.0, 3.0]  # 各帧首步 ds
    chain, valid = phase3_action_chain_targets(arrays, np.asarray([0]), future=6, stride=5)
    assert chain.shape == (1, 6, 2) and valid.shape == (1, 6)
    assert valid[0, 0] == 1.0 and chain[0, 0, 0] == pytest.approx(1.0)
    assert valid[0, 1] == 1.0 and chain[0, 1, 0] == pytest.approx(2.0)  # t+1 = step 5
    assert valid[0, 2] == 0.0, "缺失帧（step 10）必须 mask 0"
    assert valid[0, 3] == 1.0 and chain[0, 3, 0] == pytest.approx(3.0)  # t+3 = step 15
    assert float(valid[0, 4:].sum()) == 0.0, "窗口末端不足 → 尾部逐帧 mask"
    chain2, valid2 = phase3_action_chain_targets(arrays, np.asarray([1]), future=6, stride=5)
    assert chain2[0, 0, 0] == pytest.approx(2.0) and valid2[0, 0] == 1.0
    assert valid2[0, 1] == 0.0 and valid2[0, 2] == 1.0 and chain2[0, 2, 0] == pytest.approx(3.0)
    assert float(valid2[0, 3:].sum()) == 0.0
    chain3, valid3 = phase3_action_chain_targets(arrays, np.asarray([2]), future=6, stride=5)
    assert valid3[0, 0] == 1.0 and float(valid3[0, 1:].sum()) == 0.0, "episode 末帧 → 未来全 mask"

    plan = torch.zeros((1, 6, 2), requires_grad=True)
    target = torch.as_tensor(chain2)
    loss, per_step = weighted_action_chain_loss(plan, target, torch.as_tensor(valid2))
    assert len(per_step) == 5, "只监督第 2..6 步 → 5 个逐 step 口径"
    loss.backward()
    assert plan.grad is not None
    assert float(plan.grad[:, 0, :].abs().sum()) == 0.0, "第 1 步由 action_mu 首步损失监督，本项不吃"
    assert float(plan.grad[:, 1:, :].abs().sum()) > 0.0
    # 被 mask 的目标（缺帧/尾部）改动不影响损失
    corrupted = target.clone()
    corrupted[0, 1] += 100.0
    corrupted[0, 3:] += 100.0
    plan2 = torch.zeros((1, 6, 2), requires_grad=True)
    loss_corrupted, _ = weighted_action_chain_loss(plan2, corrupted, torch.as_tensor(valid2))
    assert float(loss_corrupted) == pytest.approx(float(loss.detach()), rel=1e-9)
    # 有效目标改动 → 损失变化
    changed = target.clone()
    changed[0, 2] += 1.0
    loss_changed, _ = weighted_action_chain_loss(torch.zeros((1, 6, 2)), changed, torch.as_tensor(valid2))
    assert float(loss_changed) != pytest.approx(float(loss.detach()), rel=1e-6)


# ------------------------------------------------ ⑤ LD 损失形状/回传/mask
def test_phase3_ld_loss_shapes_backprop_and_mask() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 6, 16, 4, requires_grad=True)
    target = torch.randn(2, 6, 16, 4)
    mask = torch.ones(2, 6, 16)
    mask[:, :, 8:] = 0.0
    frame_weight = torch.tensor([1.0, 2.0])
    valid = torch.ones(2, 6)
    valid[1, 4:] = 0.0
    loss, per_horizon = weighted_ld_multi_step_loss(
        pred, target, mask, frame_weight=frame_weight, valid=valid
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert len(per_horizon) == 6
    loss.backward()
    grad = pred.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad[:, :, :8, :].abs().sum()) > 0.0, "有效槽位必须可回传"
    assert float(grad[:, :, 8:, :].abs().sum()) == 0.0, "ld_mask=0 的槽位不得有梯度"
    assert float(grad[1, 4:, :, :].abs().sum()) == 0.0, "valid=0 的步不得有梯度"

    corrupted = target.clone()
    corrupted[:, :, 8:, :] += 100.0
    masked_loss, _ = weighted_ld_multi_step_loss(pred.detach(), corrupted, mask, frame_weight=frame_weight, valid=valid)
    assert float(masked_loss) == pytest.approx(float(loss.detach()), rel=1e-9)
    corrupted2 = target.clone()
    corrupted2[0, 0, 0, :] += 1.0
    changed_loss, _ = weighted_ld_multi_step_loss(pred.detach(), corrupted2, mask, frame_weight=frame_weight, valid=valid)
    assert float(changed_loss) != pytest.approx(float(loss.detach()), rel=1e-6)


# ------------------------------------------ ⑥ traj_aux=0：traj 只做监控
def test_phase3_traj_aux_zero_isolated_from_losses(tmp_path: Path) -> None:
    base, _ = make_v2_arrays(episodes=4, steps_per_episode=6)
    clean_dir = _write_arrays(tmp_path / "clean", base)
    corrupted = {k: np.array(v) for k, v in base.items()}
    corrupted["traj6"] = np.asarray(corrupted["traj6"], dtype=np.float32) + 100.0  # 100 m 级垃圾
    corrupt_dir = _write_arrays(tmp_path / "corrupt", corrupted)

    def _run(directory: Path) -> dict:
        dataset = BCDataset.load(str(directory))
        cfg = Phase3Config(
            epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0,
            traj_aux_weight=0.0,
        )
        return pretrain_bc_phase3(
            _tiny_model(), dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset)
        )

    clean = _run(clean_dir)
    broken = _run(corrupt_dir)
    for key in (
        "bc_loss",
        "bc_action_loss",
        "bc_action_chain_loss",
        "bc_ego_next_loss",
        "bc_od_loss",
        "bc_ld_loss",
        "bc_presence_loss",
        "bc_entry_loss",
        "bc_load_balance_loss",
    ):
        assert float(broken[key]) == pytest.approx(float(clean[key]), rel=1e-9), f"traj 垃圾影响 {key}"
    assert float(broken["bc_traj_mse"]) != pytest.approx(float(clean["bc_traj_mse"]), rel=1e-3), (
        "traj 监控曲线仍应看到垃圾目标"
    )


# ---------------------------------- ①⑦ phase-3 入口 smoke（dagger-only + 监控）
def test_phase3_entry_smoke_dagger_only_and_monitor(tmp_path: Path) -> None:
    five_k = write_v2_dataset(tmp_path / "BTC_fake_expert5k", episodes=10, steps_per_episode=6)
    dagger = write_v2_dataset(tmp_path / "BTC_test_dagger_r1", episodes=3, steps_per_episode=6)
    out_dir = tmp_path / "stage_b"
    cfg = {
        "stages": {
            "B": {
                "phase3": {
                    "rounds": 7,
                    "spec_pool": "env/specs/scenarios_train_5k.json",
                    "fail_target": 1000,
                    "shuffle_seed_base": 100,
                    "eval": {"spec": "env/specs/scenarios_eval500.json", "workers": 16},
                    "guard": {"stop_on_overall_drop": 0.05, "stop_on_easy_drop": 0.10},
                }
            }
        }
    }
    args = _parse_args([
        "--phase3", str(dagger), "--phase3-round", "3",
        "--ckpt", str(_init_ckpt(tmp_path)), "--out", str(out_dir),
        "--model-config", str(_model_cfg(tmp_path)), "--device", "cpu", "--seed", "0",
        "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--monitor",
    ])
    metrics = run_stage_b_phase3(args, cfg)

    dagger_rows = BCDataset.load(str(dagger)).count
    assert metrics["phase3"] is True and metrics["kind"] == "planner_bc_phase3"
    assert metrics["phase3_dir"] == str(dagger) and metrics["phase3_round"] == 3
    assert metrics["samples"] == dagger_rows and metrics["dataset/rows"] == float(dagger_rows)
    assert metrics["train_frames"] == dagger_rows, "训练行必须恰为当轮 dagger 行（无 5k）"
    assert dagger_rows < BCDataset.load(str(five_k)).count
    for key in ("bc_action_loss", "bc_action_chain_loss", "bc_ego_next_loss", "bc_od_loss",
                "bc_ld_loss", "bc_presence_loss", "bc_entry_loss"):
        assert np.isfinite(metrics[key]), f"缺少损失项 {key}"
    assert metrics["traj_monitor_only"] is True and metrics["losses"]["traj_aux"] == 0.0
    assert metrics["frozen_params"] == 0 and metrics["trainable_params"] > 0
    assert metrics["phase3_rounds"] == 7 and metrics["fail_target"] == 1000
    assert metrics["spec_pool"] == "env/specs/scenarios_train_5k.json"
    assert metrics["guard"] == {"stop_on_overall_drop": 0.05, "stop_on_easy_drop": 0.10}
    assert metrics["shuffle_seed"] == 103  # shuffle_seed_base(100) + round(3)
    assert (out_dir / "final.pt").exists() and (out_dir / "metrics.json").exists()
    reloaded = torch.load(out_dir / "final.pt", map_location="cpu", weights_only=False)
    assert reloaded["epoch"] == 1 and reloaded["meta"]["phase3_round"] == 3

    tags = _csv_tags(out_dir)
    for tag in (
        "loss/total", "loss/action", "loss/action_chain", "loss/ego_next", "loss/od",
        "loss/ld", "loss/presence", "loss/entry", "loss/load_balance",
        "ego/traj/err", "ego/traj/mse_m2", "ego/traj/mae_m", "ego/traj/fde_m", "ego/traj/mae_m/h1",
        "val/loss/action_chain", "val/loss/ld", "val/ego/traj/mae_m",
    ):
        assert tag in tags, f"phase 3 监控序列缺失：{tag}"


# ------------------------------------------------ tools/train.py CLI 入口
def test_phase3_train_cli_entry_smoke(tmp_path: Path) -> None:
    from tools.train import main as train_main

    dagger = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    out_dir = tmp_path / "stage_b"
    rc = train_main([
        "--phase3", str(dagger), "--phase3-round", "1",
        "--ckpt", str(_init_ckpt(tmp_path)), "--out", str(out_dir),
        "--model-config", str(_model_cfg(tmp_path)), "--device", "cpu",
        "--phase3-epochs", "1", "--batch-size", "8", "--limit-dataset", "18", "--no-monitor",
    ])
    assert rc == 0
    assert (out_dir / "final.pt").exists() and (out_dir / "metrics.json").exists()
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["phase3_dir"] == str(dagger) and payload["phase3_round"] == 1


# --------------------------- ⑧ 未来 OD/LD 目标对齐 t0 自车系（平移+旋转）
def test_phase3_future_targets_aligned_to_t0_frame() -> None:
    from env.obs.base import FrameAlignment
    from pipeline.stages import build_future
    from pipeline.stages import _local_build_future

    # t0 帧：原点、航向 0；未来帧：世界 (10,0)、航向 π/2。
    # 世界点 (5,0)（t0 系 dx=5,dy=0）在未来帧自车系 = (0,5)。
    count = 2
    arrays = {
        "episode_id": np.asarray([0, 0], dtype=np.int64),
        "step": np.asarray([0, 5], dtype=np.int64),
        "pose": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, np.pi / 2]], dtype=np.float32),
        "od": np.zeros((count, 16, 9), dtype=np.float32),
        "od_mask": np.ones((count, 16), dtype=np.float32),
        "od_id": np.full((count, 16), 7, dtype=np.int64),
        "od_presence": np.ones((count, 16), dtype=np.float32),
        "ld": np.zeros((count, 16, 7), dtype=np.float32),
        "ld_mask": np.ones((count, 16), dtype=np.float32),
        "ego": np.zeros((count, 8), dtype=np.float32),
    }
    arrays["od"][1, :, 0] = 0.0  # 未来帧自车系下的点 (0,5)
    arrays["od"][1, :, 1] = 5.0
    arrays["od"][1, :, 3] = 0.0
    arrays["od"][1, :, 4] = 1.0  # cosθ（未来帧下 heading=0）
    arrays["ld"][1, :, 0] = 0.0
    arrays["ld"][1, :, 1] = 5.0
    arrays["ld"][1, :, 2] = 0.3  # 未来帧下 lane heading_rel
    alignments = {
        "od": FrameAlignment(point_pairs=((0, 1),), vector_pairs=((2, 3), (4, 5))),
        "ld": FrameAlignment(point_pairs=((0, 1),), angle_dims=(2,)),
    }
    indices = np.asarray([0], dtype=np.int64)
    future = build_future(
        arrays, indices, episode_key="episode_id", step_key="step",
        future=1, stride=5, keys=_bc_channel_keys(), alignments=alignments,
    )
    # t0 系期望：点 (5,0)；ld heading = 0.3 - (θ0-θ1) = 0.3 + π/2
    assert future["od_fut"][0, 0, 0, 0] == pytest.approx(5.0, abs=1e-4)
    assert future["od_fut"][0, 0, 0, 1] == pytest.approx(0.0, abs=1e-4)
    assert future["ld_fut"][0, 0, 0, 0] == pytest.approx(5.0, abs=1e-4)
    assert future["ld_fut"][0, 0, 0, 1] == pytest.approx(0.0, abs=1e-4)
    assert future["ld_fut"][0, 0, 0, 2] == pytest.approx(0.3 + np.pi / 2, abs=1e-4)
    assert future["od_mask"][0, 0, 0] == 1.0 and future["ld_mask"][0, 0, 0] == 1.0

    # 对照：空对齐（identity）→ 仍是未来帧原始通道 (0,5)，证明上面是 SE(2) 对齐的结果
    identity = {"od": FrameAlignment(), "ld": FrameAlignment()}
    raw = build_future(
        arrays, indices, episode_key="episode_id", step_key="step",
        future=1, stride=5, keys=_bc_channel_keys(), alignments=identity,
    )
    assert raw["od_fut"][0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert raw["od_fut"][0, 0, 0, 1] == pytest.approx(5.0, abs=1e-6)

    # 本地等价路径（stack_history current_pose=pose[t0]）必须给出同一 t0 系结果
    local = _local_build_future(
        arrays, indices, episode_key="episode_id", step_key="step",
        future=1, stride=5, keys=_bc_channel_keys(), alignments=alignments, wm_valid=None,
    )
    assert np.allclose(local["od_fut"], future["od_fut"], atol=1e-5)
    assert np.allclose(local["ld_fut"], future["ld_fut"], atol=1e-5)
    assert np.allclose(local["ld_mask"], future["ld_mask"], atol=1e-6)
