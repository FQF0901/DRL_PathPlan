"""P1-2 数据接线单测：recovery_v1 → 训练集合并/混采/历史覆盖（默认关闭、逐位不变）。

覆盖：
- 合并：只用 usable ∧ split train 行、val 仅计数、episode 偏移、train_idx/recovery_indices；
- 窗口行 obs 逐位不变（历史覆盖只作用于 recovery 行）；recovery 行用真实 od/ld 历史；
- traj_aux_valid：窗口行 0 / recovery 行 1（真实 traj6）；
- fail-closed：缺 obs 通道 / 缺 split 文件 / 无可用行 → SystemExit（不静默置零）；
- 混采：ratio=0 时 recovery 行绝不被抽中；ratio>0 时按 round(ratio×batch) 抽取（batch 级记录）；
- stages 级接线：--recovery-dir/--recovery-mix-ratio 生效；无 recovery-dir 时 metrics 不变。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.stages import (
    FrameWindows,
    _BC_STEP_STRIDE,
    _bc_channel_keys,
    _merge_recovery_rows,
    _parse_args,
    _phase3_future_fn,
    run_stage_b_phase3,
)
from pipeline.trainer import (
    BCDataset,
    PHASE3_WM_LOSS_KEYS,
    Phase3Config,
    pretrain_bc_phase3,
    save_checkpoint,
)
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays

_DIRECT_OBS = (
    "ego", "ego_mask", "ego_world", "ego_world_mask", "hist_valid",
    "ld", "ld_mask", "nav", "nav_mask", "od", "od_mask", "od_presence",
    "others", "others_mask", "route_world", "route_world_mask", "signal", "signal_mask",
)


def _write_window_dir(tmp_path: Path, *, episodes: int = 2, steps_per_episode: int = 3) -> tuple[Path, dict]:
    """窗口数据集（schema v2 + v8 扩展通道；expert_bc.npz）。"""
    arrays, _ = make_v2_arrays(episodes=episodes, steps_per_episode=steps_per_episode)
    n = int(arrays["episode_id"].size)
    arrays["ego_mask"] = np.ones((n, 1), dtype=np.float32)
    arrays["ego_world"] = arrays["pose"].astype(np.float32)[:, None, :]
    arrays["ego_world_mask"] = np.ones((n, 1), dtype=np.float32)
    arrays["route_world"] = np.zeros((n, 64, 2), dtype=np.float32)
    arrays["route_world_mask"] = np.ones((n, 64), dtype=np.float32)
    arrays["hist_valid"] = np.ones((n, 6), dtype=np.float32)
    od_id = np.asarray(arrays["od_id"], dtype=np.int64)
    arrays["od_id_hist"] = np.repeat(od_id[:, None], 6, axis=1)
    arrays["od_presence_hist"] = np.repeat(np.asarray(arrays["od_presence"])[:, None], 6, axis=1)
    directory = tmp_path / "window"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "expert_bc.npz", **arrays)
    (directory / "expert_bc.meta.json").write_text(
        json.dumps({"schema_version": 2, "history_stride": 5}), encoding="utf-8"
    )
    return directory, arrays


def _recovery_arrays(
    window_arrays: dict,
    *,
    n_train: int = 4,
    n_val: int = 2,
    unusable_train: int = 1,
    misaligned_train: int = 0,
) -> dict:
    n = n_train + n_val
    arrays: dict = {}
    for name in _DIRECT_OBS:
        if name in window_arrays:
            arrays[f"obs__{name}"] = np.asarray(window_arrays[name])[:n].copy()
    arrays["obs__od_id"] = np.asarray(window_arrays["od_id"])[:n].astype(np.float32)
    arrays["obs__od_id_hist"] = np.asarray(window_arrays["od_id_hist"])[:n].astype(np.float32)
    arrays["obs__od_presence_hist"] = np.asarray(window_arrays["od_presence_hist"])[:n].astype(np.float32)
    od = np.asarray(window_arrays["od"])[:n].astype(np.float32)
    ld = np.asarray(window_arrays["ld"])[:n].astype(np.float32)
    arrays["obs__od_hist"] = np.repeat(od[:, None], 6, axis=1)
    arrays["obs__od_hist_mask"] = np.ones((n, 6, od.shape[1]), dtype=np.float32)
    arrays["obs__ld_hist"] = np.repeat(ld[:, None], 6, axis=1)
    arrays["obs__ld_hist_mask"] = np.ones((n, 6, ld.shape[1]), dtype=np.float32)
    arrays["obs__hist_valid"] = np.ones((n, 6), dtype=np.float32)
    arrays["pool"] = np.array(["eval500"] * n)
    arrays["episode_id"] = np.array(list(range(n_train)) + [90 + i for i in range(n_val)], dtype=np.int64)
    arrays["step"] = (np.arange(n, dtype=np.int64) * 5 + 100)
    # 默认全部为"对齐行"（step == takeover_step）；misaligned_train 标记前 N 个 train 行为
    # 反事实错位行（step > takeover_step）
    takeover = arrays["step"].copy()
    if misaligned_train:
        takeover[:misaligned_train] -= 5
    arrays["takeover_step"] = takeover
    arrays["usable"] = np.array(
        [True] * (n_train - unusable_train) + [False] * unusable_train + [True] * n_val
    )
    arrays["class_label"] = np.array(["prevention"] * n_train + ["late"] * n_val)
    arrays["class_id"] = np.zeros(n, dtype=np.int8)
    arrays["window_index"] = np.zeros(n, dtype=np.int8)
    arrays["geometry"] = np.array(["curve"] * n)
    arrays["lane_lat"] = np.linspace(0.1, 0.5, n).astype(np.float32)
    arrays["expert_action_chain"] = np.asarray(window_arrays["action"])[:n].astype(np.float32)
    arrays["expert_recovery_traj"] = np.asarray(window_arrays["traj6"])[:n].astype(np.float32)
    actual = np.asarray(window_arrays["traj6"])[:n].astype(np.float32).copy()
    actual[min(2, n - 1), -1] = np.nan  # 第 3 行（可用 train 行）截断示例（另列容忍 NaN）
    arrays["expert_actual_recovery_traj"] = actual
    return arrays


def _write_recovery_dir(
    tmp_path: Path,
    window_arrays: dict,
    *,
    n_train: int = 4,
    n_val: int = 2,
    unusable_train: int = 1,
    misaligned_train: int = 0,
    name: str = "recovery",
    drop_keys: tuple = (),
) -> tuple[Path, dict]:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    arrays = _recovery_arrays(
        window_arrays, n_train=n_train, n_val=n_val, unusable_train=unusable_train,
        misaligned_train=misaligned_train,
    )
    for key in drop_keys:
        arrays.pop(key, None)
    np.savez_compressed(directory / "samples.npz", **arrays)
    (directory / "split_manifest.json").write_text(
        json.dumps(
            {
                "train_episodes": [f"eval500:{i}" for i in range(n_train)],
                "val_episodes": [f"eval500:{90 + i}" for i in range(n_val)],
            }
        ),
        encoding="utf-8",
    )
    return directory, arrays


def _load_window(directory: Path) -> BCDataset:
    return BCDataset.load(str(directory))


# --------------------------------------------------------------------------- #
# 合并：split/usable/偏移/掩码
# --------------------------------------------------------------------------- #

def test_merge_recovery_rows_uses_train_split_and_usable_only(tmp_path: Path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, rec_arrays = _write_recovery_dir(tmp_path, window_arrays)
    dataset = _load_window(window_dir)
    train_idx = np.arange(dataset.count, dtype=np.int64)

    merged, merged_idx, rec_index, traj_valid, info = _merge_recovery_rows(
        dataset, train_idx, recovery_dir, logger=lambda _: None
    )
    assert info["enabled"] is True
    assert info["train_rows"] == 3  # 4 train - 1 unusable
    assert info["val_usable_rows"] == 2
    assert info["student_obs_missing_rows"] == 0
    assert merged.count == dataset.count + 3
    assert rec_index.tolist() == [dataset.count, dataset.count + 1, dataset.count + 2]
    assert merged_idx.tolist() == list(range(dataset.count)) + rec_index.tolist()
    # episode_id：偏移 + 每行独立（原 episode 另存；保证 (episode_id, step) 唯一）
    offset = info["episode_id_offset"]
    assert offset > int(np.max(dataset.arrays["episode_id"]))
    assert merged.arrays["episode_id"][rec_index].tolist() == (offset + np.arange(3)).tolist()
    assert (
        merged.arrays["recovery_source_episode_id"][rec_index].tolist()
        == rec_arrays["episode_id"][:3].tolist()
    )
    assert (merged.arrays["recovery_source_episode_id"][: dataset.count] == -1).all()
    pairs = list(zip(merged.arrays["episode_id"].tolist(), merged.arrays["step"].tolist()))
    assert len(pairs) == len(set(pairs)), "(episode_id, step) 必须唯一（查表契约）"
    # 监督映射：action=专家链；traj6=专家 recovery 轨迹（真实）；执行轨迹另列
    np.testing.assert_array_equal(
        merged.arrays["action"][rec_index], rec_arrays["expert_action_chain"][:3]
    )
    np.testing.assert_array_equal(
        merged.arrays["traj6"][rec_index], rec_arrays["expert_recovery_traj"][:3]
    )
    assert np.isnan(merged.arrays["recovery_actual_traj"][: dataset.count]).all()
    # 截断行 NaN 原样保留（第 3 个 recovery 行）
    assert np.isnan(merged.arrays["recovery_actual_traj"][rec_index[2], -1]).all()
    # traj_aux_valid：窗口 0 / recovery 1
    assert traj_valid is not None
    assert traj_valid[: dataset.count].sum() == 0.0
    assert traj_valid[rec_index].sum() == 3.0
    # 未采集字段显式 NaN（不静默置零）
    assert np.isnan(merged.arrays["labels"][rec_index]).all()
    assert np.isnan(merged.arrays["traj30"][rec_index]).all()
    assert (merged.arrays["wm_valid"][rec_index] == 0.0).all()
    assert (merged.arrays["train_weight"][rec_index] == 1.0).all()


def test_merged_window_rows_obs_bitwise_unchanged(tmp_path: Path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, _ = _write_recovery_dir(tmp_path, window_arrays)
    dataset = _load_window(window_dir)
    before = dataset.build_obs_batch(np.arange(dataset.count, dtype=np.int64))
    merged, _, _, _, _ = _merge_recovery_rows(
        dataset, np.arange(dataset.count, dtype=np.int64), recovery_dir, logger=lambda _: None
    )
    after = merged.build_obs_batch(np.arange(dataset.count, dtype=np.int64))
    assert set(before) == set(after)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key], err_msg=f"窗口行 obs 通道 {key} 被改动")


def test_merged_recovery_rows_use_stored_history(tmp_path: Path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, rec_arrays = _write_recovery_dir(tmp_path, window_arrays)
    dataset = _load_window(window_dir)
    merged, _, rec_index, _, _ = _merge_recovery_rows(
        dataset, np.arange(dataset.count, dtype=np.int64), recovery_dir, logger=lambda _: None
    )
    obs = merged.build_obs_batch(rec_index)
    np.testing.assert_array_equal(obs["od_hist"], rec_arrays["obs__od_hist"][:3])
    np.testing.assert_array_equal(obs["ld_hist"], rec_arrays["obs__ld_hist"][:3])
    np.testing.assert_array_equal(obs["hist_valid"], rec_arrays["obs__hist_valid"][:3])


# --------------------------------------------------------------------------- #
# fail-closed
# --------------------------------------------------------------------------- #

def test_merge_recovery_fail_closed_missing_channel(tmp_path: Path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, _ = _write_recovery_dir(tmp_path, window_arrays, drop_keys=("obs__od",))
    dataset = _load_window(window_dir)
    with pytest.raises(SystemExit, match="缺少必需观测通道"):
        _merge_recovery_rows(
            dataset, np.arange(dataset.count, dtype=np.int64), recovery_dir, logger=lambda _: None
        )


def test_recovery_row_filter_takeover_only_default_and_all(tmp_path: Path) -> None:
    """P1-2 崩塌根因修复：默认只并入 step==takeover_step 的对齐行；all 才保留反事实行。"""
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, _ = _write_recovery_dir(
        tmp_path, window_arrays, n_train=4, n_val=2, unusable_train=0, misaligned_train=2
    )
    dataset = _load_window(window_dir)
    train_idx = np.arange(dataset.count, dtype=np.int64)
    merged, _, rec_index, _, info = _merge_recovery_rows(
        dataset, train_idx, recovery_dir, logger=lambda _: None
    )
    assert info["row_filter"] == "takeover_only"
    assert info["train_rows"] == 2
    assert info["counterfactual_rows_excluded"] == 2
    assert rec_index.size == 2 and merged.count == dataset.count + 2
    # all：保留全部 4 行（含 2 反事实错位行）
    merged_all, _, rec_all, _, info_all = _merge_recovery_rows(
        dataset, train_idx, recovery_dir, row_filter="all", logger=lambda _: None
    )
    assert info_all["train_rows"] == 4 and info_all["counterfactual_rows_excluded"] == 0
    assert rec_all.size == 4 and merged_all.count == dataset.count + 4


def test_recovery_row_filter_missing_takeover_fail_closed(tmp_path: Path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, _ = _write_recovery_dir(tmp_path, window_arrays, drop_keys=("takeover_step",))
    dataset = _load_window(window_dir)
    with pytest.raises(SystemExit, match="takeover_step"):
        _merge_recovery_rows(
            dataset, np.arange(dataset.count, dtype=np.int64), recovery_dir, logger=lambda _: None
        )


def test_merge_recovery_excludes_rows_without_student_obs(tmp_path: Path) -> None:
    """student obs 全 NaN 行（Pass-1 反事实行）→ 显式排除并计数（不静默置零）。"""
    window_dir, window_arrays = _write_window_dir(tmp_path)
    recovery_dir, rec_arrays = _write_recovery_dir(tmp_path, window_arrays)
    with np.load(recovery_dir / "samples.npz", allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["obs__ego"][1] = np.nan  # 第 2 个 train 行（usable）obs 缺失
    arrays["obs__ego_mask"][1] = np.nan
    np.savez_compressed(recovery_dir / "samples.npz", **arrays)
    dataset = _load_window(window_dir)
    merged, merged_idx, rec_index, _, info = _merge_recovery_rows(
        dataset, np.arange(dataset.count, dtype=np.int64), recovery_dir, logger=lambda _: None
    )
    assert info["train_rows"] == 2
    assert info["student_obs_missing_rows"] == 1
    assert rec_index.size == 2
    # 被排除行（原行号 1，episode 0 / step 105）不在 merged 中
    excluded_step = int(rec_arrays["step"][1])
    merged_steps = merged.arrays["step"][dataset.count:].tolist()
    assert excluded_step not in merged_steps


def test_merge_recovery_fail_closed_missing_files_and_no_rows(tmp_path: Path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path)
    dataset = _load_window(window_dir)
    train_idx = np.arange(dataset.count, dtype=np.int64)
    with pytest.raises(SystemExit, match="samples.npz 不存在"):
        _merge_recovery_rows(dataset, train_idx, tmp_path / "missing", logger=lambda _: None)
    recovery_dir, _ = _write_recovery_dir(tmp_path, window_arrays)
    (recovery_dir / "split_manifest.json").unlink()
    with pytest.raises(SystemExit, match="split_manifest.json 不存在"):
        _merge_recovery_rows(dataset, train_idx, recovery_dir, logger=lambda _: None)
    recovery_dir2, _ = _write_recovery_dir(
        tmp_path, window_arrays, n_train=2, n_val=0, unusable_train=2, name="recovery_all_unusable"
    )
    with pytest.raises(SystemExit, match="无可用训练行"):
        _merge_recovery_rows(dataset, train_idx, recovery_dir2, logger=lambda _: None)


# --------------------------------------------------------------------------- #
# 混采：ratio=0 不抽 recovery；ratio>0 按 round(ratio×batch)
# --------------------------------------------------------------------------- #

def _tiny_dataset(tmp_path: Path) -> tuple[BCDataset, np.ndarray]:
    window_dir, window_arrays = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    dataset = _load_window(window_dir)
    rec_index = np.arange(16, 24, dtype=np.int64)  # 后 8 行标记为 recovery（仅测试用）
    return dataset, rec_index


def _record_batches(dataset: BCDataset, cfg: Phase3Config):
    arrays = dataset.arrays
    windows = FrameWindows(
        arrays, dataset.alignments, episode_key="episode_id", step_key="step",
        keys=_bc_channel_keys(), step_stride=_BC_STEP_STRIDE,
    )
    base_fn = _phase3_future_fn(dataset, windows, None)
    seen: list = []

    def recording_fn(batch_indices, obs_np):
        seen.append(np.asarray(batch_indices, dtype=np.int64).copy())
        return base_fn(batch_indices, obs_np)

    model = _tiny_model()
    pretrain_bc_phase3(model, dataset, cfg, logger=lambda _: None, future_fn=recording_fn)
    return seen


def _tiny_model():
    from net.model import DrivingModel

    import torch

    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=8, expert_hidden=16)


def _cfg(**overrides) -> Phase3Config:
    # 混采测试与冻结配方正交 → 用 all（不触发 recovery 组契约）
    base = dict(
        epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0,
        val_indices=None, freeze_mode="all", freeze_prefixes=(),
    )
    base.update(overrides)
    return Phase3Config(**base)


def test_mix_ratio_zero_never_samples_recovery_rows(tmp_path: Path) -> None:
    dataset, rec_index = _tiny_dataset(tmp_path)
    batches = _record_batches(
        dataset, _cfg(recovery_mix_ratio=0.0, recovery_indices=rec_index)
    )
    assert batches, "必须记录到 batch"
    for batch in batches:
        assert not set(batch.tolist()) & set(rec_index.tolist())


def test_mix_ratio_positive_samples_recovery_rows_per_batch(tmp_path: Path) -> None:
    dataset, rec_index = _tiny_dataset(tmp_path)
    batches = _record_batches(
        dataset, _cfg(recovery_mix_ratio=0.5, recovery_indices=rec_index)
    )
    assert batches
    for batch in batches:
        assert sum(1 for index in batch if index in set(rec_index.tolist())) == 4  # round(0.5×8)


# --------------------------------------------------------------------------- #
# stages 级接线 + 默认关闭
# --------------------------------------------------------------------------- #

def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _run_stage3(tmp_path: Path, window_dir: Path, *, extra: list, out_name: str) -> dict:
    ckpt = tmp_path / f"init_{out_name}.pt"
    save_checkpoint(ckpt, _tiny_model(), meta={"stage": "B", "phase": "phase2"})
    args = _parse_args([
        "--phase3", str(window_dir), "--ckpt", str(ckpt), "--out", str(tmp_path / out_name),
        "--model-config", str(_model_cfg(tmp_path)), "--device", "cpu", "--seed", "0",
        "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--no-monitor",
        "--no-materialize",  # 走 future_fn 路径以便测试记录 batch
        "--phase3-freeze", "recovery", "--phase3-action-chain-source", "labels",
        *extra,
    ])
    return run_stage_b_phase3(args, {"stages": {"B": {"phase3": {"freeze": "recovery"}}}})


def test_stage3_recovery_dir_wiring_and_mix(tmp_path: Path, monkeypatch) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    recovery_dir, _ = _write_recovery_dir(tmp_path, window_arrays, n_train=4, n_val=2, unusable_train=0)

    # 记录 stages 层实际喂给 future_fn 的 batch（验证混采真的抽到 recovery 行）
    import pipeline.stages as stages_mod

    original = stages_mod._phase3_future_fn
    seen: list = []

    def recording_future_fn(dataset, windows, wm_valid, **kwargs):
        base_fn = original(dataset, windows, wm_valid, **kwargs)

        def wrapper(batch_indices, obs_np):
            seen.append(np.asarray(batch_indices, dtype=np.int64).copy())
            return base_fn(batch_indices, obs_np)

        return wrapper

    monkeypatch.setattr(stages_mod, "_phase3_future_fn", recording_future_fn)
    metrics = _run_stage3(
        tmp_path, window_dir,
        extra=["--recovery-dir", str(recovery_dir), "--recovery-mix-ratio", "0.5"],
        out_name="out_mix",
    )
    recovery = metrics["recovery"]
    assert recovery["enabled"] is True
    assert recovery["train_rows"] == 4 and recovery["val_usable_rows"] == 2
    assert recovery["mix_ratio"] == pytest.approx(0.5)
    assert recovery["traj_aux_valid_sum"] == pytest.approx(4.0)  # 仅 recovery 行 =1
    assert metrics["trainable_param_groups"] == ["trunk", "primary"]
    assert tuple(metrics["frozen_loss_keys"]) == PHASE3_WM_LOSS_KEYS
    base = recovery["recovery_row_base"]
    rec_index = set(range(base, base + 4))
    assert seen, "必须记录到 batch"
    train_batches = seen[: -(-28 // 8)]  # 训练 batch 数 = ceil(28/8)（其余为 epoch 末 val 自评）
    assert len(train_batches) == 4
    for batch in train_batches:
        assert sum(1 for index in batch if index in rec_index) == 4  # round(0.5×8)


def test_stage3_without_recovery_dir_metrics_unchanged(tmp_path: Path) -> None:
    window_dir, _ = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    metrics = _run_stage3(tmp_path, window_dir, extra=[], out_name="out_off")
    assert "recovery" not in metrics, "默认关闭不得落 recovery metrics 键"
    assert metrics["trainable_param_groups"] == ["trunk", "primary"]


def test_cli_defaults() -> None:
    args = _parse_args(["--phase3", "dummy", "--out", "dummy"])
    assert args.recovery_dir is None
    assert args.recovery_mix_ratio is None
