"""v8 训练管线接线回归（obs v6 / T=39 令牌集合）：BCDataset / SINGLE_SLOT_CHANNELS / RolloutBuffer。

背景：obs v5（commit 0291f3a）曾在 pipeline 三处硬编码通道表未含 ``lane``/``ttc`` 导致新键被
``BCDataset`` 过滤；v8（obs v6）反方向清理：env 侧删除 ``lane``/``ttc`` 通道（Lane A），net 侧
删除对应编码器/令牌（Lane B，本文件）。本文件锁定 v8 接线：

1. ``BCDataset._obs_keys`` 不再含 lane/ttc；含 v6 通道（ego/od/ld/nav/signal/others）且
   ``build_obs_batch`` / ``MaterializedBCDataset`` 往返形状/数值一致；
2. ``DEFAULT_CHANNELS``（``pipeline.buffer``）不含 lane/ttc；缺键帧 0 填充 + mask=0 不报错；
3. ``SINGLE_SLOT_CHANNELS`` 单槽挤压仍作用于 ego/nav/signal/others（lane/ttc 已退出通道表）；
4. 端到端：v6 数据 → ``BCDataset`` batch → ``DrivingModel.encode`` 的 **T=39 令牌集合**
   ``[OD16, LD16, od_pool, ld_pool, others, ego, nav, signal, latent]``，后 7 个令牌恒有效。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.buffer import DEFAULT_CHANNELS, RolloutBuffer
from pipeline.trainer import (
    BCDataset,
    MaterializedBCDataset,
    SINGLE_SLOT_CHANNELS,
    SUPERVISED_LABELS,
    squeeze_single_slot,
)
from tests.v2_synthetic import make_v2_arrays

TOKEN_LENGTH = 39


def _v6_arrays(*, seed: int = 0) -> dict:
    """v2 合成数组（obs v6：无 lane/ttc 键，通道 = v2 基线集合）。"""
    arrays, _ = make_v2_arrays(episodes=4, steps_per_episode=6, seed=seed)
    return arrays


def _write_dataset(directory: Path, arrays: dict, *, schema: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "expert_bc.npz", **arrays)
    meta = {
        "schema_version": int(schema),
        "kind": "bc_expert",
        "label_names": list(SUPERVISED_LABELS),
        "history_stride": 5,
    }
    (directory / "expert_bc.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


# --------------------------------------------------------------------------- #
# 1) BCDataset：v6 键读取往返 / 无 lane/ttc
# --------------------------------------------------------------------------- #

def test_v6_dataset_roundtrip_without_struct_keys(tmp_path: Path) -> None:
    arrays = _v6_arrays()
    directory = _write_dataset(tmp_path / "bc_v6", arrays, schema=6)
    dataset = BCDataset.load(str(directory))

    assert "lane" not in dataset._obs_keys and "ttc" not in dataset._obs_keys
    for key in ("ego", "od", "ld", "nav", "signal", "others"):
        assert key in dataset._obs_keys, f"v6 通道 {key} 丢失"

    idx = np.arange(dataset.count, dtype=np.int64)
    batch = dataset.build_obs_batch(idx)
    assert "lane" not in batch and "ttc" not in batch
    assert batch["od"].shape == (dataset.count, 16, 9)
    assert batch["ld"].shape == (dataset.count, 16, 7)
    assert batch["others"].shape == (dataset.count, 33)

    # 物化快路径（训练实际消费）不得丢键/变形
    materialized = MaterializedBCDataset(dataset, chunk_size=5, logger=lambda _: None)
    mat_batch = materialized.obs_batch(idx)
    for key in ("ego", "od", "ld", "nav", "signal", "others"):
        assert key in mat_batch
        np.testing.assert_allclose(mat_batch[key], batch[key])


# --------------------------------------------------------------------------- #
# 2) SINGLE_SLOT_CHANNELS / RolloutBuffer 默认通道（v8：无 lane/ttc）
# --------------------------------------------------------------------------- #

def test_single_slot_channels_squeeze_without_struct_keys() -> None:
    assert "lane" not in SINGLE_SLOT_CHANNELS and "ttc" not in SINGLE_SLOT_CHANNELS
    batch = {
        "ego": np.zeros((2, 1, 8), dtype=np.float32),
        "nav": np.zeros((2, 1, 11), dtype=np.float32),
        "signal": np.zeros((2, 1, 4), dtype=np.float32),
        "od": np.zeros((2, 16, 9), dtype=np.float32),
    }
    out = squeeze_single_slot(batch)
    assert out["ego"].shape == (2, 8)
    assert out["nav"].shape == (2, 11)
    assert out["signal"].shape == (2, 4)
    assert out["od"].shape == (2, 16, 9)  # 多槽通道不动


def test_rollout_buffer_default_channels_without_struct_keys() -> None:
    assert "lane" not in DEFAULT_CHANNELS and "ttc" not in DEFAULT_CHANNELS
    assert DEFAULT_CHANNELS["others"] == (1, 33)

    buffer = RolloutBuffer(2)
    buffer.add_step(
        {
            "ego": np.full((1, 8), 0.5, dtype=np.float32),
            "ego_mask": np.ones((1,), dtype=np.float32),
        }
    )
    buffer.add_step({})  # 缺键（旧 replay）→ 0 填充 + mask=0，不报错
    assert buffer.obs["ego"].shape == (2, 1, 8)
    np.testing.assert_allclose(buffer.obs["ego"][0], 0.5)
    np.testing.assert_allclose(buffer.obs["ego"][1], 0.0)
    assert buffer.obs_mask["ego"][0, 0] == 1.0 and buffer.obs_mask["ego"][1, 0] == 0.0


# --------------------------------------------------------------------------- #
# 3) 端到端：v6 batch → net T=39 令牌集合（含池化/ego/latent 恒有效）
# --------------------------------------------------------------------------- #

def test_v6_batch_feeds_net_t39_tokens(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from net.model import DrivingModel

    arrays = _v6_arrays()
    dataset = BCDataset(arrays, {"schema_version": 6, "label_names": list(SUPERVISED_LABELS)})
    batch = dataset.build_obs_batch(np.arange(4, dtype=np.int64))
    obs = {key: torch.as_tensor(value) for key, value in batch.items()}

    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=2).eval()
    with torch.no_grad():
        encoded = model.encode(obs)

    # 令牌序 [OD16, LD16, od_pool, ld_pool, others, ego, nav, signal, latent]（T=39）
    assert tuple(encoded["tokens"].shape) == (4, TOKEN_LENGTH, 32)
    assert tuple(encoded["key_mask"].shape) == (4, TOKEN_LENGTH)
    assert bool(encoded["key_mask"][:, 32:].all()), "池化/others/ego/nav/signal/latent 必须恒有效"
    assert torch.equal(encoded["key_mask"][:, :16], encoded["encoded"].od_live)
    assert torch.equal(encoded["key_mask"][:, 16:32], encoded["encoded"].ld_live)
    assert not hasattr(model.encoders, "lane") and not hasattr(model.encoders, "ttc")
