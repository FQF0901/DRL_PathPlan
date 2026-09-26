"""GPU 快路径等价性测试（2026-09-26）：物化（切片 + pin H2D）vs 旧逐 batch 重建。

动机：H=128 / batch=256 时 Stage A 单 batch 墙钟 ~0.44 s，其中 >95% 是 CPU 侧逐样本
``build_obs_batch``（6 帧历史精确查表 + SE(2) 对齐）与未来目标查表。快路径把这两件事
一次性物化成连续数组（``MaterializedBCDataset`` / ``_materialize_future_targets``），
训练循环只做 ``arr[idx]`` 切片。

本文件锁定：快路径与旧路径**逐值等价**（obs/targets bitwise；Stage B loss/梯度；
Stage A loss_curve/grad_norms，容差 ≤1e-4）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from net.model import DrivingModel
from pipeline.stages import _parse_args, run_stage_a
from pipeline.trainer import (
    BCConfig,
    BCDataset,
    MaterializedBCDataset,
    SUPERVISED_LABELS,
    pretrain_bc,
)
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: 等价性容差（验收要求 ≤1e-4）
ATOL = 1e-6
RTOL = 1e-5


def _dataset(*, episodes: int = 6, steps_per_episode: int = 6) -> BCDataset:
    """v2 合成数据集 + router 软目标（覆盖 targets 的全部可选键）。"""
    arrays, _ = make_v2_arrays(episodes=episodes, steps_per_episode=steps_per_episode)
    rng = np.random.default_rng(7)
    arrays["router_soft_targets"] = rng.dirichlet(
        np.ones(8), size=arrays["episode_id"].shape[0]
    ).astype(np.float32)
    return BCDataset(
        arrays,
        {
            "schema_version": 2,
            "label_names": list(SUPERVISED_LABELS),
            "history_stride": 5,
            "history_storage": "per_frame",
        },
    )


def test_materialized_obs_and_targets_match_dataset_paths() -> None:
    """物化数组与 ``build_obs_batch`` / ``targets`` 逐值相同（含 chunk 边界）。"""
    dataset = _dataset()
    source = MaterializedBCDataset(dataset, chunk_size=5, logger=lambda _: None)
    indices = np.array([0, 3, 7, 11, 20, 35], dtype=np.int64)

    obs_old = dataset.build_obs_batch(indices)
    obs_new = source.obs_batch(indices)
    assert set(obs_old) == set(obs_new), "物化 obs 键集必须与旧路径一致"
    for key, value in obs_old.items():
        np.testing.assert_array_equal(value, obs_new[key], err_msg=f"obs[{key}] 不一致")

    targets_old = dataset.targets(indices)
    targets_new = source.targets_batch(indices)
    for key, value in targets_new.items():
        assert key in targets_old, f"物化目标多出键 {key}"
        np.testing.assert_array_equal(targets_old[key], value, err_msg=f"targets[{key}] 不一致")


def test_materialized_obs_slicing_is_read_only() -> None:
    """切片不写回物化数组（训练循环可安全并发读；DataLoader worker 亦共享只读）。"""
    dataset = _dataset()
    source = MaterializedBCDataset(dataset, chunk_size=4, logger=lambda _: None)
    key = sorted(source.arrays)[0]
    before = source.arrays[key].copy()
    source.obs_batch(np.arange(6, dtype=np.int64))
    np.testing.assert_array_equal(source.arrays[key], before)


def test_pretrain_bc_materialized_equivalent_to_old_path() -> None:
    """Stage B：同一初始化、同种子下，新旧路径的 loss / 梯度 / 训练后参数一致。"""
    torch.manual_seed(0)
    dataset = _dataset(episodes=5, steps_per_episode=6)
    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16, wm_steps=6)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source = MaterializedBCDataset(dataset, logger=lambda _: None)

    def _config() -> BCConfig:
        return BCConfig(
            epochs=2, batch_size=8, lr=1e-3, device="cpu", shuffle=True, seed=3, router_coef=0.1
        )

    metrics_old = pretrain_bc(model, dataset, _config(), logger=lambda _: None)
    grads_old = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    params_old = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert grads_old, "旧路径没有任何梯度（测试前提不成立）"

    model.load_state_dict(initial_state)
    metrics_new = pretrain_bc(
        model, dataset, _config(), logger=lambda _: None, batch_source=source
    )

    for key in ("bc_loss", "bc_traj_loss", "bc_action_loss", "bc_router_loss", "bc_action_mu_ds_mean"):
        assert metrics_new[key] == pytest.approx(metrics_old[key], rel=RTOL, abs=ATOL), key
    for name, grad in grads_old.items():
        new_grad = dict(model.named_parameters())[name].grad
        assert new_grad is not None, f"{name} 新路径无梯度"
        assert torch.allclose(grad, new_grad.detach(), atol=ATOL, rtol=RTOL), f"梯度不一致：{name}"
    for key, value in params_old.items():
        assert torch.allclose(value, model.state_dict()[key], atol=ATOL, rtol=RTOL), f"参数不一致：{key}"


def test_pretrain_bc_gradient_accumulation_matches_single_batch() -> None:
    """Stage B 梯度累积：micro=4/宏=8 与单 batch=8 的 loss/梯度一致（宏口径精确缩放）。"""
    torch.manual_seed(0)
    dataset = _dataset(episodes=5, steps_per_episode=6)
    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16, wm_steps=6)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source = MaterializedBCDataset(dataset, logger=lambda _: None)

    def _config(micro: int | None) -> BCConfig:
        return BCConfig(
            epochs=2, batch_size=8, micro_batch_size=micro, lr=1e-3, device="cpu",
            shuffle=True, seed=3, router_coef=0.1,
        )

    metrics_ref = pretrain_bc(model, dataset, _config(None), logger=lambda _: None, batch_source=source)
    grads_ref = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.load_state_dict(initial_state)
    metrics_accum = pretrain_bc(model, dataset, _config(4), logger=lambda _: None, batch_source=source)
    grads_accum = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    for key in ("bc_loss", "bc_traj_loss", "bc_action_loss", "bc_router_loss", "bc_action_mu_ds_mean"):
        assert metrics_accum[key] == pytest.approx(metrics_ref[key], rel=RTOL, abs=ATOL), key
    assert set(grads_ref) == set(grads_accum)
    for name, grad in grads_ref.items():
        # 逐元素梯度：fp32 CPU GEMM 在 batch=8 vs 4+4 下归约顺序不同 → 残差 ~1e-5（验收容差 1e-4）
        assert torch.allclose(grad, grads_accum[name], atol=1e-4, rtol=1e-4), f"梯度不一致：{name}"


def test_stage_a_gradient_accumulation_matches_single_batch(tmp_path: Path) -> None:
    """Stage A 梯度累积：micro=4/宏=16 与单 batch=16 一致（关闭 plan 噪声保证可比）。"""
    dataset_dir = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(TINY_MODEL_YAML, encoding="utf-8")
    common = [
        "--stage", "A", "--bc-dir", str(dataset_dir), "--model-config", str(model_cfg),
        "--wm-epochs", "2", "--val-frac", "0.34", "--batch-size", "16", "--eval-frames", "12",
        "--device", "cpu", "--seed", "0", "--no-plan-noise", "--materialize",
    ]

    torch.manual_seed(0)
    metrics_ref = run_stage_a(
        _parse_args(common + ["--out", str(tmp_path / "a_ref")]), {}
    )
    torch.manual_seed(0)
    metrics_accum = run_stage_a(
        _parse_args(common + ["--micro-batch-size", "4", "--out", str(tmp_path / "a_accum")]), {}
    )

    assert metrics_accum["loss_curve"] == pytest.approx(metrics_ref["loss_curve"], rel=RTOL, abs=ATOL)
    assert metrics_accum["grad_norms_first_batch"] == pytest.approx(
        metrics_ref["grad_norms_first_batch"], rel=RTOL, abs=ATOL
    )
    assert metrics_accum["grad_norms_last_batch"] == pytest.approx(
        metrics_ref["grad_norms_last_batch"], rel=RTOL, abs=ATOL
    )
    for key in ("wm_loss", "wm_loss_od", "val_loss", "model_ade", "presence_loss", "entry_loss"):
        value_accum, value_ref = float(metrics_accum[key]), float(metrics_ref[key])
        if np.isnan(value_accum) and np.isnan(value_ref):
            continue
        assert value_accum == pytest.approx(value_ref, rel=RTOL, abs=ATOL), key
    assert metrics_ref["grad_accum"] is False and metrics_accum["grad_accum"] is True
    assert metrics_accum["micro_batch_size"] == 4 and metrics_accum["batch_size"] == 16


def test_stage_a_materialized_equivalent_to_old_path(tmp_path: Path) -> None:
    """Stage A：同一初始化/种子下，新旧路径的 loss_curve 与梯度探针一致。"""
    dataset_dir = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(TINY_MODEL_YAML, encoding="utf-8")
    common = [
        "--stage", "A", "--bc-dir", str(dataset_dir), "--model-config", str(model_cfg),
        "--wm-epochs", "2", "--val-frac", "0.34", "--batch-size", "8", "--eval-frames", "12",
        "--device", "cpu", "--seed", "0", "--plan-noise-p", "0.3",
    ]

    torch.manual_seed(0)
    metrics_old = run_stage_a(
        _parse_args(common + ["--no-materialize", "--out", str(tmp_path / "a_old")]), {}
    )
    torch.manual_seed(0)
    metrics_new = run_stage_a(
        _parse_args(common + ["--materialize", "--out", str(tmp_path / "a_new")]), {}
    )

    assert metrics_new["loss_curve"] == pytest.approx(metrics_old["loss_curve"], rel=RTOL, abs=ATOL)
    assert metrics_new["grad_norms_first_batch"] == pytest.approx(
        metrics_old["grad_norms_first_batch"], rel=RTOL, abs=ATOL
    )
    assert metrics_new["grad_norms_last_batch"] == pytest.approx(
        metrics_old["grad_norms_last_batch"], rel=RTOL, abs=ATOL
    )
    for key in ("wm_loss", "val_loss", "model_ade", "model_fde", "ego_next_loss", "presence_loss", "entry_loss"):
        value_new, value_old = float(metrics_new[key]), float(metrics_old[key])
        if np.isnan(value_new) and np.isnan(value_old):
            continue  # 小样本下 ADE/FDE 可能无有效槽位（nan 与 nan 视为一致）
        assert value_new == pytest.approx(value_old, rel=RTOL, abs=ATOL), key
    # 快路径确实生效（物化 + 未来目标物化）
    assert metrics_new["materialize"] is True and metrics_new["fast_data"] is True
    assert metrics_old["materialize"] is False
