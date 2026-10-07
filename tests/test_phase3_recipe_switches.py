"""stage B phase 3 配方开关回归（lane P3-I）。

覆盖（用户定稿契约）：

① ``freeze=specific_only``：只训 experts/router/residual_scale（主干/WM/policy 冻结；
   冻结参数无梯度、不更新），损失自动降级（WM od/ld/presence/entry + ego_next = 0，
   保留 action/action_chain/load_balance）；
② ``freeze=all``（旧行为）不回归：全参数可训 + 上游监督项照常（与默认配置逐值一致）；
③ ``anchor`` 关闭 = 现行为（训练集恰为窗口 dagger 行，不碰 5k）；
④ ``anchor`` 开启 = 窗口行 + 锚行合并：行数/episode 偏移/权重（×mild_weight）/traj-aux 掩码正确，
   目标构建复用 ``_materialize_future_targets``（快慢路径同值）；
⑤ 配置优先级：CLI > config（freeze 与 anchor 两个方向）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline.stages import (
    FrameWindows,
    _BC_STEP_STRIDE,
    _SPECIFIC_PHASE_FREEZE,
    _bc_channel_keys,
    _merge_anchor_rows,
    _parse_args,
    _phase3_future_fn,
    run_stage_b_phase3,
)
from pipeline.trainer import (
    BCDataset,
    PHASE3_WM_LOSS_KEYS,
    Phase3Config,
    _PHASE3_SPECIFIC_PREFIXES,
    apply_freeze_prefixes,
    build_phase3_optimizer,
    phase3_effective_config,
    pretrain_bc_phase3,
    row_action_weights,
    save_checkpoint,
)
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_BACKBONE_PREFIXES = (
    "encoders.",
    "mem_encoder.",
    "plan_head.moe.primary.",
    "st_gnn.",
    "policy.",
    "value.",
)


def _tiny_model():
    from net.model import DrivingModel

    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=8, expert_hidden=16)


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _init_ckpt(tmp_path: Path) -> Path:
    path = tmp_path / "init.pt"
    save_checkpoint(path, _tiny_model(), meta={"stage": "B", "phase": "phase2"})
    return path


def _dataset(episodes: int = 4, steps_per_episode: int = 6) -> BCDataset:
    arrays, _ = make_v2_arrays(episodes=episodes, steps_per_episode=steps_per_episode)
    return BCDataset(arrays, {"schema_version": 2, "label_names": None})


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


# ------------------------------------ ① specific_only：冻结 + 损失降级 + 仅 specific 更新
def test_phase3_specific_only_freezes_backbone_and_degrades_wm_losses() -> None:
    dataset = _dataset()
    model = _tiny_model()
    # 真实起点 = stage B phase 2 final：phase 1 已训练 policy.mu（非零）。PolicyHead.mu 零初始化
    # 会让 raw=0 → d(action)/d(latent)=0，若起点是全新初始化则 action 梯度到不了 specific 组
    # （只剩 router 的负载均衡 aux）——那是初始化的性质，不是冻结开关的行为；此处模拟真实起点。
    with torch.no_grad():
        torch.manual_seed(0)
        model.policy.mu.weight.normal_(0.0, 0.1)
        model.policy.mu.bias.normal_(0.0, 0.1)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    cfg = Phase3Config(
        epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0,
        freeze_mode="specific_only", freeze_prefixes=_SPECIFIC_PHASE_FREEZE,
        val_indices=np.arange(dataset.count, dtype=np.int64),
    )
    metrics = pretrain_bc_phase3(
        model, dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )

    assert metrics["freeze_mode"] == "specific_only"
    assert tuple(metrics["frozen_loss_keys"]) == PHASE3_WM_LOSS_KEYS
    assert tuple(PHASE3_WM_LOSS_KEYS) == ("ego_next", "od", "ld", "presence", "entry", "latent")
    assert metrics["frozen_params"] > 0 and metrics["trainable_params"] > 0
    assert metrics["trainable_param_groups"] == ["specific"], "specific_only 只应有 specific 优化器组"

    # 损失自动降级：WM/ego_next 全 0；action/action_chain/load_balance 保留
    for key in (
        "bc_ego_next_loss",
        "bc_od_loss",
        "bc_ld_loss",
        "bc_latent_loss",
        "bc_presence_loss",
        "bc_entry_loss",
    ):
        assert float(metrics[key]) == 0.0, f"{key} 应降级为 0"
        assert float(metrics["val"][key]) == 0.0, f"val/{key} 应降级为 0"
    for key in ("bc_action_loss", "bc_action_chain_loss", "bc_load_balance_loss"):
        assert np.isfinite(metrics[key]) and float(metrics[key]) > 0.0, f"{key} 应保留梯度"

    # 冻结参数无梯度/不更新；specific 组真实更新
    changed: set = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert any(name.startswith(prefix) for prefix in _PHASE3_SPECIFIC_PREFIXES), (
                f"{name} 不应在 specific_only 下可训"
            )
        else:
            assert parameter.grad is None, f"冻结参数 {name} 不应有梯度"
            assert torch.equal(before[name], parameter.detach()), f"冻结参数 {name} 被更新"
        if not torch.equal(before[name], parameter.detach()):
            changed.add(name)
    for prefix in _PHASE3_SPECIFIC_PREFIXES:
        assert any(name.startswith(prefix) for name in changed), f"{prefix} 未更新"
    for prefix in _BACKBONE_PREFIXES:
        assert not any(name.startswith(prefix) for name in changed), f"{prefix} 不应更新"


# --------------------------------------- ② freeze=all（旧行为）与默认语义逐值一致
def test_phase3_freeze_all_matches_legacy_default_behavior() -> None:
    dataset = _dataset()
    future_fn = _future_fn_for(dataset)
    base = dict(epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0)
    legacy = pretrain_bc_phase3(
        _tiny_model(), dataset, Phase3Config(**base),
        logger=lambda _: None, future_fn=future_fn,
    )
    explicit = pretrain_bc_phase3(
        _tiny_model(), dataset, Phase3Config(**base, freeze_mode="all", freeze_prefixes=()),
        logger=lambda _: None, future_fn=future_fn,
    )
    assert Phase3Config().freeze_mode == "all", "dataclass 默认 = 旧行为（config 文件才切新配方）"
    assert legacy["freeze_mode"] == "all" and explicit["frozen_loss_keys"] == []
    assert legacy["frozen_params"] == 0 and legacy["trainable_param_groups"] == ["base", "specific"]
    assert explicit["frozen_params"] == 0
    for key in (
        "bc_loss", "bc_action_loss", "bc_action_chain_loss", "bc_ego_next_loss",
        "bc_od_loss", "bc_ld_loss", "bc_presence_loss", "bc_entry_loss", "bc_load_balance_loss",
    ):
        assert float(explicit[key]) == pytest.approx(float(legacy[key]), rel=1e-12), key
    for key in ("bc_od_loss", "bc_ld_loss", "bc_ego_next_loss", "bc_presence_loss", "bc_entry_loss"):
        assert np.isfinite(legacy[key]) and float(legacy[key]) > 0.0, f"all 模式 {key} 不得为 0"


# ------------------------------------- ②b 有效配置 helper（不改变调用方 config）
def test_phase3_effective_config_zeroes_wm_keys_without_mutating() -> None:
    cfg = Phase3Config(freeze_mode="specific_only", od_weight=0.005, ld_weight=0.002, ego_next_weight=0.1)
    effective, zeroed = phase3_effective_config(cfg)
    assert tuple(zeroed) == PHASE3_WM_LOSS_KEYS
    for key in PHASE3_WM_LOSS_KEYS:
        assert float(getattr(effective, f"{key}_weight")) == 0.0
    assert cfg.od_weight == pytest.approx(0.005) and cfg.ld_weight == pytest.approx(0.002)
    assert effective.action_weight == cfg.action_weight
    unchanged, none_zeroed = phase3_effective_config(Phase3Config())
    assert none_zeroed == () and unchanged.freeze_mode == "all"


# ----------------------------------------- ②c specific_only 优化器分组（base 空 = 合法）
def test_phase3_specific_only_optimizer_group_and_all_mode_guard() -> None:
    model = _tiny_model()
    frozen = apply_freeze_prefixes(model, _SPECIFIC_PHASE_FREEZE)
    assert frozen, "phase-2 冻结清单必须真的冻结参数"
    optimizer = build_phase3_optimizer(model, 1e-3, specific_scale=0.5, freeze_mode="specific_only")
    assert [group.get("name") for group in optimizer.param_groups] == ["specific"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    specific_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert specific_names and all(name.startswith(_PHASE3_SPECIFIC_PREFIXES) for name in specific_names)
    # all 模式（默认）在 base 空时仍拒绝静默退化
    with pytest.raises(ValueError, match="LR 分组为空"):
        build_phase3_optimizer(model, 1e-3)
    # specific 空（全冻结）→ 两种模式都拒绝
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="LR 分组为空"):
        build_phase3_optimizer(model, 1e-3, freeze_mode="specific_only")


# ---------------------------- ④a 合并：行数 / episode 偏移 / 权重 / traj-aux 掩码
def test_merge_anchor_rows_counts_weights_mask_and_lookup() -> None:
    window = _dataset(episodes=3, steps_per_episode=6)
    anchor = _dataset(episodes=5, steps_per_episode=6)
    window_train_idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)  # 模拟 episode 切分
    merged, merged_idx, traj_valid, info = _merge_anchor_rows(
        window, window_train_idx, anchor, "fake_anchor", 0.1, logger=lambda _: None
    )
    w_rows, a_rows = int(window.count), int(anchor.count)
    assert merged.count == w_rows + a_rows
    assert merged_idx.size == window_train_idx.size + a_rows
    assert np.array_equal(merged_idx[: window_train_idx.size], window_train_idx)
    assert np.array_equal(merged_idx[window_train_idx.size :], np.arange(w_rows, w_rows + a_rows))
    # episode_id 偏移：无碰撞，锚区间从窗口 max+1 起
    win_eps = np.asarray(window.arrays["episode_id"], dtype=np.int64)
    anchor_eps = np.asarray(merged.arrays["episode_id"], dtype=np.int64)[w_rows:]
    assert int(anchor_eps.min()) == int(win_eps.max()) + 1 == info["episode_id_offset"]
    assert not (set(anchor_eps.tolist()) & set(win_eps.tolist()))
    # 权重：窗口行不动；锚行 = 原 train_weight×配平×mild_weight
    mild = 0.1
    np.testing.assert_allclose(
        row_action_weights(merged, np.arange(w_rows, dtype=np.int64)),
        row_action_weights(window, np.arange(w_rows, dtype=np.int64)),
    )
    np.testing.assert_allclose(
        row_action_weights(merged, np.arange(w_rows, w_rows + a_rows, dtype=np.int64)),
        row_action_weights(anchor, np.arange(a_rows, dtype=np.int64)) * mild,
    )
    assert row_action_weights(anchor, np.arange(a_rows, dtype=np.int64)).sum() > 0.0
    # traj-aux 掩码：窗口 0 / 锚 1（窗口 traj6 是合成值）
    assert np.array_equal(traj_valid[:w_rows], np.zeros(w_rows, dtype=np.float32))
    assert np.array_equal(traj_valid[w_rows:], np.ones(a_rows, dtype=np.float32))
    assert info["window_traj_aux_masked"] == w_rows and info["anchor_traj_aux_valid"] == a_rows
    # 帧查表/未来目标在合并集上可用（锚行 idx 点查；episode 偏移后仍能定位未来帧）
    arrays = merged.arrays
    windows = FrameWindows(
        arrays, merged.alignments, episode_key="episode_id", step_key="step",
        keys=_bc_channel_keys(), step_stride=_BC_STEP_STRIDE,
    )
    wm_valid = np.asarray(arrays["wm_valid"], dtype=np.float32)
    future = _phase3_future_fn(merged, windows, wm_valid)(
        np.asarray([w_rows], dtype=np.int64), merged.build_obs_batch(np.asarray([w_rows], dtype=np.int64))
    )
    for key in ("od_fut", "ld_fut", "od_mask", "ld_mask", "action_chain", "action_chain_valid"):
        assert key in future and np.asarray(future[key]).shape[0] == 1
    assert float(np.asarray(future["action_chain_valid"]).sum()) > 0.0, "锚行未来目标必须可查表"


# --------------------- ④b 锚开启端到端：合并训练 + 快/慢路径（物化/逐 batch）一致
def test_phase3_anchor_enabled_entry_merges_rows_and_keeps_targets_equal(tmp_path: Path) -> None:
    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    anchor = write_v2_dataset(tmp_path / "BTC_fake_expert5k", episodes=10, steps_per_episode=6)
    window_rows = BCDataset.load(str(window)).count
    anchor_rows = BCDataset.load(str(anchor)).count
    ckpt = _init_ckpt(tmp_path)
    cfg = {"stages": {"B": {"phase3": {"freeze": "specific_only"}}}}

    def _run(name: str, extra: list[str]) -> dict:
        args = _parse_args([
            "--phase3", str(window), "--phase3-round", "2",
            "--ckpt", str(ckpt), "--out", str(tmp_path / name),
            "--model-config", str(_model_cfg(tmp_path)), "--device", "cpu", "--seed", "0",
            "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--no-monitor",
            *extra,
        ])
        return run_stage_b_phase3(args, cfg)

    materialized = _run("out_fast", [
        "--phase3-anchor", "--phase3-anchor-bc-dir", str(anchor), "--phase3-anchor-mild-weight", "0.1",
    ])
    per_batch = _run("out_slow", [
        "--phase3-anchor", "--phase3-anchor-bc-dir", str(anchor), "--phase3-anchor-mild-weight", "0.1",
        "--no-materialize",
    ])
    assert materialized["samples"] == window_rows + anchor_rows
    assert materialized["train_frames"] == window_rows + anchor_rows
    assert materialized["dataset/rows"] == float(window_rows + anchor_rows)
    assert materialized["anchor"]["enabled"] is True
    assert materialized["anchor"]["window_rows"] == window_rows
    assert materialized["anchor"]["anchor_rows"] == anchor_rows
    assert materialized["anchor"]["mild_weight"] == pytest.approx(0.1)
    assert materialized["anchor"]["traj_aux_valid_rows"] == window_rows + anchor_rows
    assert materialized["anchor"]["traj_aux_valid_sum"] == float(anchor_rows)
    assert materialized["freeze"] == "specific_only"
    for key in ("bc_ego_next_loss", "bc_od_loss", "bc_ld_loss", "bc_presence_loss", "bc_entry_loss"):
        assert float(materialized[key]) == 0.0
    assert (tmp_path / "out_fast" / "final.pt").exists()
    # 物化快路径与逐 batch 路径逐值一致（含 5k 锚行的未来目标/动作链）
    for key in (
        "bc_loss", "bc_action_loss", "bc_action_chain_loss", "bc_ego_next_loss",
        "bc_od_loss", "bc_ld_loss", "bc_presence_loss", "bc_entry_loss", "bc_load_balance_loss",
    ):
        assert float(per_batch[key]) == pytest.approx(float(materialized[key]), rel=1e-9), key


# ------------------------------------------- ③ anchor 关闭（含 config 显式 false）= 现行为
def test_phase3_anchor_disabled_stays_dagger_only(tmp_path: Path) -> None:
    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    anchor = write_v2_dataset(tmp_path / "BTC_fake_expert5k", episodes=10, steps_per_episode=6)
    window_rows = BCDataset.load(str(window)).count
    cfg = {
        "stages": {
            "B": {
                "phase3": {
                    "freeze": "all",
                    "anchor": {"enabled": False, "bc_dir": str(anchor), "mild_weight": 0.1},
                }
            }
        }
    }
    args = _parse_args([
        "--phase3", str(window), "--ckpt", str(_init_ckpt(tmp_path)),
        "--out", str(tmp_path / "out_off"),
        "--model-config", str(_model_cfg(tmp_path)), "--device", "cpu", "--seed", "0",
        "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--no-monitor",
    ])
    metrics = run_stage_b_phase3(args, cfg)
    assert metrics["samples"] == window_rows and metrics["train_frames"] == window_rows
    assert metrics["anchor"]["enabled"] is False
    assert metrics["freeze"] == "all" and metrics["frozen_params"] == 0
    assert metrics["losses"]["od"] > 0.0, "anchor 关闭 + freeze=all：现行为（上游监督保留）"


# ------------------------------------------- ⑤a 仓库配置新默认（freeze=specific_only，无锚）
def test_repo_config_phase3_recipe_defaults() -> None:
    from pipeline.stages import load_config

    p3 = load_config("config/default.yaml")["stages"]["B"]["phase3"]
    assert p3["freeze"] == "specific_only", "config/train.yaml 新默认必须是 specific_only"
    assert p3["anchor"]["enabled"] is False, "5k 锚默认关（保持无 5k）"
    assert float(p3["anchor"]["mild_weight"]) == pytest.approx(0.1)


# ------------------------------------------- ⑤b anchor.bc_dir=auto 解析（显式优先 / 缺失报错）
def test_resolve_anchor_bc_dir_auto_and_explicit(monkeypatch, tmp_path: Path) -> None:
    import pipeline.run_paths as run_paths

    from pipeline.stages import _resolve_anchor_bc_dir

    assert _resolve_anchor_bc_dir({"run": {"bc_dir": "runs/custom_bc"}}) == "runs/custom_bc"
    monkeypatch.setattr(run_paths, "latest_dataset", lambda root=None: tmp_path / "BTC1_expert5k")
    assert _resolve_anchor_bc_dir({"run": {"bc_dir": "auto"}}) == str(tmp_path / "BTC1_expert5k")
    monkeypatch.setattr(run_paths, "latest_dataset", lambda root=None: None)
    with pytest.raises(SystemExit, match="anchor.bc_dir=auto"):
        _resolve_anchor_bc_dir({})


# ------------------------------------------------- ⑤c 配置优先级：CLI > config（双向）
def test_phase3_cli_overrides_config_for_freeze_and_anchor(tmp_path: Path) -> None:
    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    anchor = write_v2_dataset(tmp_path / "BTC_fake_expert5k", episodes=10, steps_per_episode=6)
    window_rows = BCDataset.load(str(window)).count
    ckpt = _init_ckpt(tmp_path)
    cfg = {
        "stages": {
            "B": {
                "phase3": {
                    "freeze": "all",
                    "anchor": {"enabled": True, "bc_dir": str(anchor), "mild_weight": 0.5},
                }
            }
        }
    }

    def _run(name: str, extra: list[str]) -> dict:
        args = _parse_args([
            "--phase3", str(window), "--ckpt", str(ckpt), "--out", str(tmp_path / name),
            "--model-config", str(_model_cfg(tmp_path)), "--device", "cpu", "--seed", "0",
            "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--no-monitor",
            *extra,
        ])
        return run_stage_b_phase3(args, cfg)

    # config 说 all+锚开；CLI 覆盖为 specific_only + 锚关（两个方向同测）
    overridden = _run("out_cli", ["--phase3-freeze", "specific_only", "--no-phase3-anchor"])
    assert overridden["freeze"] == "specific_only" and overridden["frozen_params"] > 0
    assert overridden["anchor"]["enabled"] is False
    assert overridden["samples"] == window_rows and overridden["train_frames"] == window_rows
    assert overridden["losses"]["od"] == 0.0

    # CLI 只覆盖 mild_weight（config 的锚开 + freeze 保留 all）
    mild = _run("out_mild", [
        "--phase3-anchor", "--phase3-anchor-bc-dir", str(anchor), "--phase3-anchor-mild-weight", "0.05",
    ])
    assert mild["freeze"] == "all" and mild["anchor"]["enabled"] is True
    assert mild["anchor"]["mild_weight"] == pytest.approx(0.05), "CLI mild_weight 必须覆盖 config 0.5"
    assert mild["samples"] == window_rows + BCDataset.load(str(anchor)).count
