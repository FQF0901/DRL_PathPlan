"""v5 训练管线接线（lane/ttc）回归：BCDataset / SINGLE_SLOT_CHANNELS / RolloutBuffer。

背景（obs v5 未决，fix-4）：obs v5（commit 0291f3a）在 ``env/obs``、``net``、``tools/collect_expert``
落地，但 pipeline 三处硬编码通道表未含 ``lane``/``ttc`` → v5 npz 的新键被 ``BCDataset`` 过滤、
PPO 默认缓冲无对应形状，训练时 net 走缺键回退（token mask=0）**新 token 不生效**。本文件锁定：

1. ``BCDataset._obs_keys`` 含 lane/ttc（顺序与 ``collect_expert.CURRENT_CHANNELS`` / builder 一致），
   ``build_obs_batch`` 与 ``MaterializedBCDataset`` 往返形状/数值；
2. 旧 v4 数据（无 lane/ttc 键）仍可加载，batch 无新键（缺键回退路径不变）；
3. ``SINGLE_SLOT_CHANNELS`` 含 lane/ttc（``(B,1,F) → (B,F)``）；
4. ``RolloutBuffer`` 默认通道含 lane/ttc（缺键填 0、有键透传 + mask）；
5. 端到端：v5 数据 → ``BCDataset`` batch → ``DrivingModel.encode`` 的 lane/ttc key_mask=1
   （证明接线生效而非缺键回退）。
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

LANE_DIM = 17
TTC_DIM = 12


def _v5_arrays(*, with_struct: bool = True, seed: int = 0) -> dict:
    """v2 合成数组 + 可选 v5 的 lane/ttc 通道（值可区分，验证往返）。"""
    arrays, _ = make_v2_arrays(episodes=4, steps_per_episode=6, seed=seed)
    if with_struct:
        count = len(arrays["episode_id"])
        rng = np.random.default_rng(seed + 1)
        arrays["lane"] = rng.normal(size=(count, 1, LANE_DIM)).astype(np.float32)
        arrays["lane_mask"] = np.ones((count, 1), dtype=np.float32)
        arrays["ttc"] = rng.normal(size=(count, 1, TTC_DIM)).astype(np.float32)
        arrays["ttc_mask"] = np.ones((count, 1), dtype=np.float32)
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
# 1) BCDataset：v5 键读取往返 / 旧 v4 兼容
# --------------------------------------------------------------------------- #

def test_v5_dataset_roundtrip_reads_lane_ttc(tmp_path: Path) -> None:
    arrays = _v5_arrays()
    directory = _write_dataset(tmp_path / "bc_v5", arrays, schema=5)
    dataset = BCDataset.load(str(directory))

    assert "lane" in dataset._obs_keys and "ttc" in dataset._obs_keys
    # 顺序与 collect_expert.CURRENT_CHANNELS / builder 一致：lane 紧跟 ld、ttc 在 others 之后
    assert dataset._obs_keys.index("lane") == dataset._obs_keys.index("ld") + 1
    assert dataset._obs_keys.index("ttc") == dataset._obs_keys.index("others") + 1

    idx = np.arange(dataset.count, dtype=np.int64)
    batch = dataset.build_obs_batch(idx)
    assert batch["lane"].shape == (dataset.count, LANE_DIM)
    assert batch["ttc"].shape == (dataset.count, TTC_DIM)
    assert batch["lane_mask"].shape == (dataset.count, 1)
    assert batch["ttc_mask"].shape == (dataset.count, 1)
    np.testing.assert_allclose(batch["lane"], arrays["lane"][:, 0])
    np.testing.assert_allclose(batch["ttc"], arrays["ttc"][:, 0])

    # 物化快路径（训练实际消费）不得丢新键/变形
    materialized = MaterializedBCDataset(dataset, chunk_size=5, logger=lambda _: None)
    mat_batch = materialized.obs_batch(idx)
    assert mat_batch["lane"].shape == (dataset.count, LANE_DIM)
    assert mat_batch["ttc"].shape == (dataset.count, TTC_DIM)
    np.testing.assert_allclose(mat_batch["lane"], batch["lane"])
    np.testing.assert_allclose(mat_batch["ttc"], batch["ttc"])


def test_v4_dataset_without_struct_keys_still_loads(tmp_path: Path) -> None:
    """旧 v4（无 lane/ttc）→ 自动跳过新键，batch 键集合与既有通道形状不变。"""
    arrays = _v5_arrays(with_struct=False)
    directory = _write_dataset(tmp_path / "bc_v4", arrays, schema=2)
    dataset = BCDataset.load(str(directory))

    assert "lane" not in dataset._obs_keys and "ttc" not in dataset._obs_keys
    batch = dataset.build_obs_batch(np.arange(dataset.count, dtype=np.int64))
    assert "lane" not in batch and "ttc" not in batch
    assert batch["od"].shape == (dataset.count, 16, 9)
    assert batch["ld"].shape == (dataset.count, 16, 7)


# --------------------------------------------------------------------------- #
# 2) SINGLE_SLOT_CHANNELS / RolloutBuffer 默认通道
# --------------------------------------------------------------------------- #

def test_single_slot_channels_include_lane_ttc() -> None:
    assert "lane" in SINGLE_SLOT_CHANNELS and "ttc" in SINGLE_SLOT_CHANNELS
    batch = {
        "lane": np.zeros((2, 1, LANE_DIM), dtype=np.float32),
        "ttc": np.zeros((2, 1, TTC_DIM), dtype=np.float32),
        "od": np.zeros((2, 16, 9), dtype=np.float32),
    }
    out = squeeze_single_slot(batch)
    assert out["lane"].shape == (2, LANE_DIM)
    assert out["ttc"].shape == (2, TTC_DIM)
    assert out["od"].shape == (2, 16, 9)  # 多槽通道不动


def test_rollout_buffer_default_channels_include_lane_ttc() -> None:
    from env.obs.lane import LANE_DIM as ENV_LANE_DIM
    from env.obs.ttc import TTC_DIM as ENV_TTC_DIM

    assert DEFAULT_CHANNELS["lane"] == (1, ENV_LANE_DIM)
    assert DEFAULT_CHANNELS["ttc"] == (1, ENV_TTC_DIM)
    # 与 builder 默认输出同序（lane 在 ld 后、ttc 在 others 后）
    keys = list(DEFAULT_CHANNELS)
    assert keys.index("lane") == keys.index("ld") + 1
    assert keys.index("ttc") == keys.index("others") + 1

    buffer = RolloutBuffer(2)
    buffer.add_step(
        {
            "lane": np.full((1, LANE_DIM), 0.5, dtype=np.float32),
            "lane_mask": np.ones((1,), dtype=np.float32),
            "ttc": np.full((1, TTC_DIM), 2.0, dtype=np.float32),
            "ttc_mask": np.ones((1,), dtype=np.float32),
        }
    )
    buffer.add_step({})  # 缺键（旧 replay）→ 0 填充 + mask=0，不报错
    assert buffer.obs["lane"].shape == (2, 1, LANE_DIM)
    assert buffer.obs["ttc"].shape == (2, 1, TTC_DIM)
    np.testing.assert_allclose(buffer.obs["lane"][0], 0.5)
    np.testing.assert_allclose(buffer.obs["ttc"][0], 2.0)
    np.testing.assert_allclose(buffer.obs["lane"][1], 0.0)
    np.testing.assert_allclose(buffer.obs["ttc"][1], 0.0)
    assert buffer.obs_mask["lane"][0, 0] == 1.0 and buffer.obs_mask["lane"][1, 0] == 0.0
    assert buffer.obs_mask["ttc"][0, 0] == 1.0 and buffer.obs_mask["ttc"][1, 0] == 0.0


# --------------------------------------------------------------------------- #
# 3) 端到端：v5 batch → net 令牌集合的 lane/ttc key_mask=1（非缺键回退）
# --------------------------------------------------------------------------- #

def test_v5_batch_feeds_net_lane_ttc_tokens(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from net.model import DrivingModel

    arrays = _v5_arrays()
    dataset = BCDataset(arrays, {"schema_version": 5, "label_names": list(SUPERVISED_LABELS)})
    batch = dataset.build_obs_batch(np.arange(4, dtype=np.int64))
    obs = {key: torch.as_tensor(value) for key, value in batch.items()}

    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=2).eval()
    with torch.no_grad():
        encoded = model.encode(obs)

    # 令牌序 [OD16, LD16, lane, others, ego, nav, ttc, signal, latent]（T=39）
    assert tuple(encoded["tokens"].shape) == (4, 39, 32)
    assert encoded["key_mask"][:, 32].all(), "lane token 被缺键回退（mask=0）"
    assert encoded["key_mask"][:, 36].all(), "ttc token 被缺键回退（mask=0）"
    assert bool(encoded["lane_mask"].all()) and bool(encoded["ttc_mask"].all())
