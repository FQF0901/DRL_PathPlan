#!/usr/bin/env python3
"""R41-H 守卫测试：**v7 同源 obs 存储 + 训练装配 = 评测装配（aligned）装配口**（自包含 CPU）。

覆盖：
1. store roundtrip（write_shard → merge_shards → V7ObsStore）逐位 + 锚 pad/mask + 缺行/无锚 fail-fast；
2. ``--assembly aligned`` 装配口：语义 K1e2 装配序 + 缺失语义键回填 + 断言（含 fill 后 missing=0）；
3. CLI 绑定：aligned ⇔ v7obs、legacy ⇒ base（非法组合 fail-fast；默认 legacy = 旧路径）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import lateral_v7obs  # noqa: E402
from tools import train_lateral  # noqa: E402


def _fake_episode(episode_id: int, frames: int, *, anchor_len: int = 4, valid: bool = True) -> dict:
    rng = np.random.RandomState(episode_id)
    obs = {
        "ego": rng.randn(frames, 1, 8).astype(np.float32),
        "ego_mask": np.ones((frames, 1), dtype=np.float32),
        "od_id": rng.randint(0, 5, size=(frames, 16)).astype(np.int64),
        "lane_anchor_valid": np.full((frames, 1), 1.0 if valid else 0.0, dtype=np.float32),
    }
    flat = (
        np.arange(frames * anchor_len * 2, dtype=np.float32).reshape(frames * anchor_len, 2) / 7.0
    )
    off = np.arange(frames + 1, dtype=np.int64) * anchor_len
    return {
        "steps": np.arange(frames, dtype=np.int64) * 5,
        "obs": obs,
        "anchor_flat": flat,
        "anchor_off": off,
        "anchor_meta": [{"n_points": anchor_len}] * frames,
        "termination": "max_step",
        "env_steps": int(frames * 5),
        "spec_id": int(episode_id),
        "episode_id": int(episode_id),
        "seed": 1000 + int(episode_id),
        "geometry_name": "straight",
    }


def _build_store(tmp_path: Path, episodes) -> lateral_v7obs.V7ObsStore:
    shard = tmp_path / "shard_0.npz"
    lateral_v7obs.write_shard(shard, episodes)
    lateral_v7obs.merge_shards([shard], tmp_path / "store", meta={"test": True})
    return lateral_v7obs.load_store(tmp_path / "store")


def test_store_roundtrip_bitwise(tmp_path):
    episodes = [_fake_episode(0, 3), _fake_episode(1, 2)]
    store = _build_store(tmp_path, episodes)
    assert store.n == 5
    rows = store.rows_of([(0, 0), (0, 10), (1, 5)])
    batch = store.obs_batch(rows, require_anchor=True)
    # 逐位 roundtrip
    assert np.array_equal(batch["ego"][0], episodes[0]["obs"]["ego"][0].reshape(-1))
    assert np.array_equal(batch["ego"][1], episodes[0]["obs"]["ego"][2].reshape(-1))
    assert np.array_equal(batch["ego"][2], episodes[1]["obs"]["ego"][1].reshape(-1))
    assert np.array_equal(batch["od_id"][0], np.asarray(episodes[0]["obs"]["od_id"][0], dtype=np.int64))
    # 单槽通道去槽位维
    assert batch["ego"].shape == (3, 8)
    assert batch["lane_anchor_xy"].shape == (3, 4, 2)
    assert batch["lane_anchor_mask"].sum() == 3 * 4
    assert np.array_equal(batch["lane_anchor_xy"][1], episodes[0]["anchor_flat"][8:12])
    assert np.array_equal(batch["lane_anchor_xy"][0], episodes[0]["anchor_flat"][0:4])
    # 合并排序（episode 主序）
    assert np.array_equal(np.asarray(store.episode_id[:3]), np.zeros(3, dtype=np.int64))
    assert np.array_equal(np.asarray(store.step[:3]), np.asarray([0, 5, 10], dtype=np.int64))


def test_store_requires_anchor_and_rows(tmp_path):
    episodes = [_fake_episode(0, 2, valid=True), _fake_episode(1, 2, valid=False)]
    store = _build_store(tmp_path, episodes)
    ok_rows = store.rows_of([(0, 0)])
    store.obs_batch(ok_rows, require_anchor=True)  # 不抛
    bad_rows = store.rows_of([(1, 0)])
    with pytest.raises(SystemExit, match="无有效注入锚"):
        store.obs_batch(bad_rows, require_anchor=True)
    with pytest.raises(SystemExit, match="不在 store 中"):
        store.rows_of([(99, 0)])
    # 乱序/跨帧批（锚 pad 到批内最大）
    mixed = store.rows_of([(0, 5), (0, 0)])
    batch = store.obs_batch(mixed, require_anchor=True)
    assert batch["lane_anchor_xy"].shape[0] == 2
    assert np.array_equal(batch["lane_anchor_xy"][0], episodes[0]["anchor_flat"][4:8])


def test_merged_store_rejects_duplicate(tmp_path):
    episode = _fake_episode(0, 2)
    shard = tmp_path / "shard_0.npz"
    lateral_v7obs.write_shard(shard, [episode, episode])  # 同 (ep, step) 重复
    with pytest.raises(SystemExit, match="重复"):
        lateral_v7obs.merge_shards([shard], tmp_path / "store", meta={})


def test_window_actions_matches_collect_expert_formula():
    poses = []
    for i in range(12):
        poses.append(np.asarray([0.5 * i, 0.1 * i, 0.01 * i], dtype=np.float64))
    actions = lateral_v7obs.window_actions(poses, 5, 1)
    # 手算：a=5,b=10；ds = Σ_{j=5..9} ‖p_{j+1}-p_j‖；dθ = wrap(θ10-θ5)
    length = sum(float(np.linalg.norm(poses[j + 1][:2] - poses[j][:2])) for j in range(5, 10))
    dtheta = float((poses[10][2] - poses[5][2] + math_pi()) % (2.0 * math_pi()) - math_pi())
    assert np.isclose(actions[0, 0], length)
    assert np.isclose(actions[0, 1], dtheta)


def math_pi() -> float:
    import math

    return math.pi


def test_parser_binding_defaults_and_failfast():
    args = train_lateral.parse_args(["--config", str(ROOT / "config/lateral_train.yaml")])
    assert args.assembly == "legacy"
    assert args.obs_source == "base"
    aligned = train_lateral.parse_args(
        ["--config", str(ROOT / "config/lateral_train.yaml"), "--assembly", "aligned"]
    )
    assert aligned.obs_source == "v7obs"
    with pytest.raises(SystemExit, match="必须搭配 --obs-source v7obs"):
        train_lateral.parse_args(
            ["--config", str(ROOT / "config/lateral_train.yaml"), "--assembly", "aligned",
             "--obs-source", "base"]
        )
    with pytest.raises(SystemExit, match="仅用于 --assembly aligned"):
        train_lateral.parse_args(
            ["--config", str(ROOT / "config/lateral_train.yaml"), "--obs-source", "v7obs"]
        )


def _tiny_model_config() -> dict:
    return {
        "hidden_dim": 32,
        "moe": {"experts": {"count": 2, "hidden_dim": 16}},
        "world_model": {"rollout_steps": 6},
        "policy": {},
        "value": {},
        "spatial": {},
    }


def test_build_aligned_model_semantic_fill(tmp_path, monkeypatch):
    """aligned 装配：语义构造 + 缺失语义键回填 + 断言（tiny 模型；不触达 runs/ 真实 ckpt）。"""
    import torch
    import yaml

    from pipeline.stages import build_model as stage_build_model

    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_model_config()), encoding="utf-8")
    # 语义构建（= 评测装配序）→ 全键 state 作为"canary"+ "fill"
    full = stage_build_model(_tiny_model_config())
    full.set_ld_mode("semantic")
    full.set_rollout_semantic("static_role_only")
    full.set_moe(enabled=False)
    full.set_path_head(True)
    state = {key: value.clone() for key, value in full.state_dict().items()}
    semantic_like = [
        key for key in state
        if key.startswith(("semantic_wiring.", "target_proj.")) or key in ("beta_context", "beta_token")
    ]
    assert len(semantic_like) >= 5
    canary = {key: value for key, value in state.items() if key not in set(semantic_like)}
    fill = {key: state[key] for key in semantic_like}
    canary_path = tmp_path / "canary.pt"
    fill_path = tmp_path / "fill.pt"
    torch.save({"model": canary}, canary_path)
    torch.save({"model": fill}, fill_path)

    args = train_lateral.parse_args(
        [
            "--config", str(ROOT / "config/lateral_train.yaml"),
            "--assembly", "aligned",
            "--model-config", str(config_path),
            "--init", str(canary_path),
            "--ckpt", str(canary_path),
            "--fill-from", str(fill_path),
            "--device", "cpu",
        ]
    )
    model, info = train_lateral.build_aligned_model(args)
    assert info["missing_keys"] == [] and info["unexpected_keys"] == []
    assert sorted(info["filled_keys"]) == sorted(semantic_like)
    assert model._is_semantic() and model.rollout_semantic_mode == "static_role_only"
    # 回填逐位
    sd = model.state_dict()
    for key in semantic_like:
        assert torch.equal(sd[key].cpu(), state[key].cpu()), key
    for key, value in canary.items():
        assert torch.equal(sd[key].cpu(), value.cpu()), key


def test_build_aligned_model_fails_without_fill(tmp_path):
    import torch
    import yaml

    from pipeline.stages import build_model as stage_build_model

    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_model_config()), encoding="utf-8")
    full = stage_build_model(_tiny_model_config())
    full.set_ld_mode("semantic")
    full.set_moe(enabled=False)
    full.set_path_head(True)
    state = dict(full.state_dict())
    semantic_like = {
        key for key in state
        if key.startswith(("semantic_wiring.", "target_proj.")) or key in ("beta_context", "beta_token")
    }
    canary = {key: value for key, value in state.items() if key not in semantic_like}
    canary_path = tmp_path / "canary.pt"
    torch.save({"model": canary}, canary_path)
    args = train_lateral.parse_args(
        [
            "--config", str(ROOT / "config/lateral_train.yaml"),
            "--assembly", "aligned",
            "--model-config", str(config_path),
            "--init", str(canary_path),
            "--ckpt", str(canary_path),
            "--fill-from", str(tmp_path / "missing.pt"),
            "--device", "cpu",
        ]
    )
    with pytest.raises(SystemExit, match="aligned 装载缺键"):
        train_lateral.build_aligned_model(args)
