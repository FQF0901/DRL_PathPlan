"""schema v2 合成数据（测试共享；不依赖 metadrive/env 运行时）。

生成的小数据集覆盖训练回路需要的关键字段：
``train_weight/balance_weight/wm_valid/od_id/od_presence/router_soft_targets`` +
历史需要的 ``pose/step/episode_id`` 与动作/轨迹目标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pipeline.trainer import SUPERVISED_LABELS

#: 小型模型配置（yaml 文本；用于 Stage A/B 冒烟）
TINY_MODEL_YAML = (
    "hidden_dim: 16\n"
    "moe:\n"
    "  experts:\n"
    "    count: 8\n"
    "    hidden_dim: 16\n"
    "world_model:\n"
    "  rollout_steps: 6\n"
)


def make_v2_arrays(
    *,
    episodes: int = 6,
    steps_per_episode: int = 6,
    stride: int = 5,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[int, int]]:
    """构造 v2 数组字典；返回 ``(arrays, episode_start_rows)``。

    每个 episode 的 step = 0, 5, ..., (steps-1)*5（grid 化，满足 lane A 的 stride 断言）。
    """
    count = int(episodes * steps_per_episode)
    rng = np.random.default_rng(seed)
    episode_id = np.repeat(np.arange(episodes, dtype=np.int64), steps_per_episode)
    step = np.tile(np.arange(steps_per_episode, dtype=np.int64) * stride, episodes)

    pose = np.zeros((count, 3), dtype=np.float32)
    pose[:, 0] = np.arange(count, dtype=np.float32) * 1.5  # 世界系匀速前进
    ego = rng.normal(size=(count, 8)).astype(np.float32)
    ego[:, 0] = 5.0  # speed
    ego[:, 6:8] = 0.0

    od = np.zeros((count, 16, 9), dtype=np.float32)
    od[:, :, 4] = 1.0  # cosθ
    od[:, :, 0] = 10.0 + np.arange(16, dtype=np.float32)[None, :]  # dx
    od[:, :, 2] = 0.5  # vx
    od_mask = np.ones((count, 16), dtype=np.float32)

    ld = rng.normal(size=(count, 16, 7)).astype(np.float32)
    ld_mask = np.ones((count, 16), dtype=np.float32)
    nav = rng.normal(size=(count, 1, 11)).astype(np.float32)
    signal = rng.normal(size=(count, 1, 4)).astype(np.float32)
    # v2 规范上下文输入 others (1,28) = nav(11)+speed_limit(1)+signal(4)+road_class one-hot(12)
    others = np.zeros((count, 1, 28), dtype=np.float32)
    others[:, 0, :11] = nav[:, 0, :]
    others[:, 0, 11] = 0.5
    others[:, 0, 12:16] = signal[:, 0, :]
    others[:, 0, 16:28] = rng.normal(size=(count, 12)).astype(np.float32)

    od_id = np.tile(np.arange(1, 17, dtype=np.int64), (count, 1))
    od_presence = np.ones((count, 16), dtype=np.float32)
    od_presence[:, 12:] = 0.0  # 4 个空槽 → presence/entry 有正负样本
    od_mask[:, 12:] = 0.0

    labels = np.zeros((count, 8), dtype=np.float32)
    labels[:, 4] = 1.0  # on_curve
    labels[::3, 0] = 1.0  # cutin_active（多标签）→ 逐标签序列有覆盖

    action = np.zeros((count, 6, 2), dtype=np.float32)
    action[:, :, 0] = 3.0
    action[::3, 0, 0] = 0.5  # 急刹切片
    action[1::3, 0, 1] = 0.2  # 急转切片
    traj30 = np.zeros((count, 30, 2), dtype=np.float32)
    traj30[:, :, 0] = np.arange(1, 31, dtype=np.float32)[None, :] * 0.3
    traj6 = traj30[:, [4, 9, 14, 19, 24, 29]]

    train_weight = np.ones(count, dtype=np.float32)
    train_weight[::7] = 0.0  # 权重 0 的帧（不得作为动作/轨迹目标）
    balance_weight = np.where(episode_id % 2 == 0, 1.0, 2.0).astype(np.float32)
    wm_valid = np.ones((count, 6), dtype=np.float32)
    wm_valid[::5, 0] = 0.0  # 部分帧 horizon-1 目标不可用
    frame_usable = np.ones(count, dtype=np.float32)
    frame_usable[::7] = 0.0
    # 注意：fixture 不写入 ``router_soft_targets``（router 监督唯一来源 = 聚类 lane API，
    # 见 pipeline.clusters.soft_targets_from_obs）。

    arrays: Dict[str, np.ndarray] = {
        "episode_id": episode_id,
        "step": step,
        "pose": pose,
        "ego": ego,
        "od": od,
        "od_mask": od_mask,
        "ld": ld,
        "ld_mask": ld_mask,
        "nav": nav,
        "nav_mask": np.ones((count, 1), dtype=np.float32),
        "signal": signal,
        "signal_mask": np.ones((count, 1), dtype=np.float32),
        "others": others,
        "others_mask": np.ones((count, 1), dtype=np.float32),
        "action": action,
        "traj6": traj6,
        "traj30": traj30,
        "labels": labels,
        "labels_raw": labels,
        "sample_weight": np.ones(count, dtype=np.float32),
        "train_weight": train_weight,
        "balance_weight": balance_weight,
        "wm_valid": wm_valid,
        "frame_usable": frame_usable,
        "od_id": od_id,
        "od_presence": od_presence,
        "prev_action": action[:, 0, :],
        "roundtrip_dense_err": np.zeros(count, dtype=np.float32),
        "roundtrip_key_err": np.zeros(count, dtype=np.float32),
        "spec_id": np.zeros(count, dtype=np.int64),
        "seed": np.zeros(count, dtype=np.int64),
        "lane_lat": np.zeros(count, dtype=np.float32),
        "difficulty": np.array(["easy", "medium"] * (count // 2 + 1))[:count].astype("U16"),
        "geometry": np.array(["straight", "curve"] * (count // 2 + 1))[:count].astype("U16"),
        "split": np.array(["train"] * count, dtype="U16"),
        "balance_group": np.asarray(
            [f"{'easy' if i % 2 == 0 else 'medium'}/straight" for i in range(count)], dtype="U64"
        ),
    }
    starts = {int(ep): int(np.argmax(episode_id == ep)) for ep in np.unique(episode_id)}
    return arrays, starts


def write_v2_dataset(
    directory: Path,
    *,
    meta_extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Path:
    """把 :func:`make_v2_arrays` 落盘为 ``expert_bc.npz`` + meta（返回目录）。"""
    arrays, _ = make_v2_arrays(**kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "expert_bc.npz", **arrays)
    meta = {
        "schema_version": 2,
        "kind": "bc_expert",
        "label_names": list(SUPERVISED_LABELS),
        "history_storage": "per_frame",
        "history_stride": int(kwargs.get("stride", 5)),
    }
    if meta_extra:
        meta.update(meta_extra)
    (directory / "expert_bc.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return directory
